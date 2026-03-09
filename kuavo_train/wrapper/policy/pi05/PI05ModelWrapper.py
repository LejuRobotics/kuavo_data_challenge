"""
PI05 Model Wrapper with Cross-Attention Depth Fusion

This module extends PI05Pytorch to support Cross-Attention depth fusion (方案二).

方案二：Cross-Attention (双向注意力融合)
- 使用 PaliGemma Vision Tower 处理深度图像（与 RGB 共享编码器）
- RGB 和 Depth 特征通过双向 Cross-Attention 交互融合
- 参考文档：RGB_DEPTH_FUSION_COMPARISON.md

Note: PI05 only supports cross_attention method, not sequence_concat.
"""

import math
import logging
from typing import Optional, Tuple, List

import torch
import torch.nn.functional as F
from torch import Tensor
import torch.nn as nn

from lerobot.policies.pi05.modeling_pi05 import PI05Pytorch, resize_with_pad_torch, make_att_2d_masks
from kuavo_train.wrapper.policy.pi05.PI05ConfigWrapper import CustomPI05ConfigWrapper

logger = logging.getLogger(__name__)


class CrossModalAttentionFusion(nn.Module):
    """
    Cross-Attention fusion module for RGB and Depth features.
    
    Implements bidirectional cross-attention:
    - RGB as query, Depth as key/value
    - Depth as query, RGB as key/value
    """
    
    def __init__(self, embed_dim: int, num_heads: int = 8):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        
        # RGB to Depth attention (RGB queries, Depth keys/values)
        self.rgb_depth_attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            batch_first=False  # PI05 uses (seq_len, batch, embed_dim) format
        )
        
        # Depth to RGB attention (Depth queries, RGB keys/values)
        self.depth_rgb_attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            batch_first=False
        )
        
        # Layer normalization for residual connections
        self.norm_rgb = nn.LayerNorm(embed_dim)
        self.norm_depth = nn.LayerNorm(embed_dim)
    
    def forward(
        self, 
        rgb_features: torch.Tensor, 
        depth_features: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply cross-attention fusion between RGB and Depth features.
        
        Args:
            rgb_features: RGB features [B, N_rgb, C]
            depth_features: Depth features [B, N_depth, C]
        
        Returns:
            fused_rgb: RGB features enhanced with depth information [B, N_rgb, C]
            fused_depth: Depth features enhanced with RGB information [B, N_depth, C]
        """
        # Convert to (seq_len, batch, embed_dim) format for MultiheadAttention
        rgb_seq = rgb_features.transpose(0, 1)  # [N_rgb, B, C]
        depth_seq = depth_features.transpose(0, 1)  # [N_depth, B, C]
        
        # RGB as query, Depth as key/value
        rgb_fused, _ = self.rgb_depth_attn(
            query=rgb_seq,
            key=depth_seq,
            value=depth_seq
        )
        # Add residual and normalize
        rgb_fused = self.norm_rgb(rgb_fused + rgb_seq)
        
        # Depth as query, RGB as key/value
        depth_fused, _ = self.depth_rgb_attn(
            query=depth_seq,
            key=rgb_seq,
            value=rgb_seq
        )
        # Add residual and normalize
        depth_fused = self.norm_depth(depth_fused + depth_seq)
        
        # Convert back to (batch, seq_len, embed_dim) format
        rgb_fused = rgb_fused.transpose(0, 1)  # [B, N_rgb, C]
        depth_fused = depth_fused.transpose(0, 1)  # [B, N_depth, C]
        
        return rgb_fused, depth_fused


class CustomPI05Pytorch(PI05Pytorch):
    """
    Extended PI05Pytorch model with support for dual depth fusion methods.
    
    Supports:
    1. sequence_concat: Sequence concatenation (OpenVLA-Depth style) - uses independent ViT backbone
    2. cross_attention: Cross-attention bidirectional fusion - uses PaliGemma Vision Tower
    """
    
    def __init__(self, config: CustomPI05ConfigWrapper, rtc_processor=None):
        # Initialize parent class
        super().__init__(config, rtc_processor)
        
        # Initialize depth fusion components if depth is enabled
        if config.use_depth and config.depth_features:
            self._init_depth_fusion(config)
        else:
            self.depth_backbone = None
            self.depth_proj = None
            self.cross_modal_fusion = None
    
    def _init_depth_fusion(self, config: CustomPI05ConfigWrapper):
        """
        Initialize depth fusion components based on fusion method.
        
        Args:
            config: Configuration with depth fusion settings
        """
        fusion_method = config.depth_fusion_method
        
        if fusion_method == "cross_attention":
            self._init_cross_attention_fusion(config)
        elif fusion_method == "sequence_concat":
            self._init_sequence_concat_fusion(config)
        else:
            raise ValueError(
                f"Unsupported depth fusion method: {fusion_method}. "
                f"Supported methods: 'sequence_concat', 'cross_attention'"
            )
    
    def _init_cross_attention_fusion(self, config: CustomPI05ConfigWrapper):
        """
        Initialize cross-attention fusion components (方案二).
        
        方案二：Cross-Attention 双向注意力融合
        - 使用 PaliGemma Vision Tower 处理深度图像（与 RGB 共享编码器）
        - RGB 和 Depth 特征通过双向 Cross-Attention 交互融合
        """
        # Get LLM dimension from PaliGemma config
        llm_dim = self.paligemma_with_expert.paligemma.language_model.config.hidden_size
        
        # Cross-attention method uses PaliGemma vision tower for depth images
        # No independent ViT backbone needed
        self.depth_backbone = None
        
        # Get depth feature dimension from PaliGemma vision tower
        with torch.no_grad():
            dummy_input = torch.zeros(1, 3, *config.image_resolution, device=self.paligemma_with_expert.paligemma.device)
            dummy_depth_feat = self.paligemma_with_expert.embed_image(dummy_input)
            depth_dim = dummy_depth_feat.shape[-1]  # Same as RGB feature dimension
        
        # Projection layer (usually not needed since RGB and Depth use same vision tower)
        if depth_dim != llm_dim:
            self.depth_proj = nn.Linear(depth_dim, llm_dim)
        else:
            self.depth_proj = nn.Identity()
        
        # Cross-attention fusion module
        fusion_dim = config.depth_fusion_dim if config.depth_fusion_dim is not None else llm_dim
        self.cross_modal_fusion = CrossModalAttentionFusion(
            embed_dim=fusion_dim,
            num_heads=8
        )
        
        logger.info(
                f"Initialized cross-attention fusion (using PaliGemma vision tower for depth): "
                f"depth_dim={depth_dim}, llm_dim={llm_dim}, fusion_dim={fusion_dim}"
            )
    
    def _init_sequence_concat_fusion(self, config: CustomPI05ConfigWrapper):
        """
        Initialize sequence concatenation fusion components (方案一).
        
        方案一：Sequence Concat（OpenVLA-Depth 风格）
        - 使用独立的 ViT backbone 提取深度特征
        - RGB 和 Depth 特征序列拼接后输入 LLM
        - 参考文档：RGB_DEPTH_FUSION_COMPARISON.md
        """
        # Get LLM dimension from PaliGemma config
        llm_dim = self.paligemma_with_expert.paligemma.language_model.config.hidden_size
        
        # Check if offline mode is enabled
        import os
        from pathlib import Path
        
        # Set HuggingFace cache directory EARLY, before importing timm
        hf_home = os.getenv("HF_HOME")
        hf_hub_cache = os.getenv("HF_HUB_CACHE")
        transformers_cache = os.getenv("TRANSFORMERS_CACHE")
        
        # If cache directories are not set, try to detect common cache locations
        if not hf_hub_cache and not transformers_cache:
            possible_cache_paths = [
                "/data/wanglishan/.cache/huggingface/hub",
                os.path.expanduser("~/.cache/huggingface/hub"),
                "/data/wanglishan/.cache/huggingface",
            ]
            for cache_path_str in possible_cache_paths:
                cache_path = Path(cache_path_str)
                if cache_path.exists():
                    if cache_path.name == "hub":
                        hub_cache_dir = str(cache_path)
                        hf_home_dir = str(cache_path.parent)
                    else:
                        hub_cache_dir = str(cache_path / "hub")
                        hf_home_dir = str(cache_path)
                    
                    if not hf_hub_cache:
                        os.environ["HF_HUB_CACHE"] = hub_cache_dir
                        logger.info(f"Auto-detected HF_HUB_CACHE: {hub_cache_dir}")
                    if not hf_home:
                        os.environ["HF_HOME"] = hf_home_dir
                        logger.info(f"Auto-detected HF_HOME: {hf_home_dir}")
                    break
        
        hf_offline_env = os.getenv("HF_HUB_OFFLINE", "0")
        offline_mode = hf_offline_env.lower() in ("1", "true", "yes")
        use_pretrained = getattr(config, 'depth_backbone_pretrained', True)
        depth_backbone_created = False
        
        # Try to create depth backbone with pretrained weights
        if use_pretrained:
            # First, try to find model file in known cache locations
            model_file_found = None
            model_config_file = None
            possible_cache_dirs = [
                "/data/wanglishan/.cache/huggingface/hub",
                os.path.expanduser("~/.cache/huggingface/hub"),
            ]
            
            model_cache_name = f"models--timm--{config.depth_backbone}.augreg2_in21k_ft_in1k"
            for cache_dir in possible_cache_dirs:
                model_cache_path = Path(cache_dir) / model_cache_name
                if model_cache_path.exists():
                    # Check for model file in snapshots
                    snapshots_dir = model_cache_path / "snapshots"
                    if snapshots_dir.exists():
                        for snapshot_dir in snapshots_dir.iterdir():
                            model_file = snapshot_dir / "model.safetensors"
                            config_file = snapshot_dir / "config.json"
                            if model_file.exists():
                                # Follow symlink to actual file if it's a symlink
                                if model_file.is_symlink():
                                    actual_file = model_file.resolve()
                                else:
                                    actual_file = model_file
                                
                                if actual_file.exists():
                                    model_file_found = actual_file
                                    if config_file.exists():
                                        model_config_file = config_file
                                    logger.info(
                                        f"Found pretrained weights at: {actual_file}"
                                    )
                                    if model_config_file:
                                        logger.info(f"Found config file at: {model_config_file}")
                                    break
                        if model_file_found:
                            break
            
            try:
                # Try with current environment first
                if offline_mode:
                    logger.info(
                        f"Offline mode detected. Attempting to load pretrained weights for depth backbone "
                        f"'{config.depth_backbone}' from local cache..."
                    )
                else:
                    logger.info(
                        f"Attempting to create depth backbone '{config.depth_backbone}' with pretrained weights..."
                    )
                
                # Set HF_HUB_CACHE if we found a cache directory
                if model_file_found and not os.getenv("HF_HUB_CACHE"):
                    cache_parent = model_file_found.parent.parent.parent.parent
                    if cache_parent.name == "hub":
                        os.environ["HF_HUB_CACHE"] = str(cache_parent)
                        os.environ["HF_HOME"] = str(cache_parent.parent)
                        logger.info(
                            f"Set HF_HUB_CACHE={cache_parent} to use local cache"
                        )
                
                # Import timm here, after environment variables are set
                import timm
                
                # Try to load model with pretrained weights
                # If we found the model file directly, try loading it manually as fallback
                try:
                    self.depth_backbone = timm.create_model(
                        config.depth_backbone,
                        pretrained=True,
                        num_classes=0,
                        global_pool="",
                    )
                    depth_backbone_created = True
                    logger.info(
                        f"✅ Successfully created depth backbone '{config.depth_backbone}' with pretrained weights. "
                        f"{'(Loaded from local cache)' if offline_mode or model_file_found else ''}"
                    )
                except Exception as load_error:
                    # If timm.create_model fails but we have the model file, try manual loading
                    if model_file_found:
                        logger.info(
                            f"timm.create_model failed, attempting to load weights directly from {model_file_found}..."
                        )
                        try:
                            # Create model without pretrained weights first
                            self.depth_backbone = timm.create_model(
                                config.depth_backbone,
                                pretrained=False,
                                num_classes=0,
                                global_pool="",
                            )
                            # Load weights from safetensors file
                            from safetensors.torch import load_file
                            state_dict = load_file(str(model_file_found))
                            # Remove 'model.' prefix if present (some timm models have this)
                            cleaned_state_dict = {}
                            for k, v in state_dict.items():
                                new_key = k.replace('model.', '') if k.startswith('model.') else k
                                cleaned_state_dict[new_key] = v
                            # Try loading with strict=False to handle minor mismatches
                            missing_keys, unexpected_keys = self.depth_backbone.load_state_dict(cleaned_state_dict, strict=False)
                            if missing_keys:
                                logger.warning(f"Some keys were missing when loading weights: {len(missing_keys)} keys")
                            if unexpected_keys:
                                logger.warning(f"Some keys were unexpected when loading weights: {len(unexpected_keys)} keys")
                            depth_backbone_created = True
                            logger.info(
                                f"✅ Successfully loaded pretrained weights directly from {model_file_found}"
                            )
                        except Exception as manual_load_error:
                            logger.warning(
                                f"Failed to load weights manually: {manual_load_error}. "
                                f"Raising original error."
                            )
                            raise load_error  # Raise original error
                    else:
                        raise load_error  # No model file found, raise original error
            except Exception as e:
                # If failed and not in offline mode, try with offline mode enabled
                # This forces timm to use local cache only
                error_msg = str(e).lower()
                if not offline_mode and ("local cache" in error_msg or "not find" in error_msg or "cannot find" in error_msg):
                    logger.info(
                        f"Network download failed, attempting to load from local cache with offline mode..."
                    )
                    try:
                        # Temporarily enable offline mode to force local cache usage
                        original_offline = os.getenv("HF_HUB_OFFLINE", "0")
                        os.environ["HF_HUB_OFFLINE"] = "1"
                        
                        # Ensure cache paths are set
                        if model_file_found and not os.getenv("HF_HUB_CACHE"):
                            cache_parent = model_file_found.parent.parent.parent.parent
                            if cache_parent.name == "hub":
                                os.environ["HF_HUB_CACHE"] = str(cache_parent)
                                os.environ["HF_HOME"] = str(cache_parent.parent)
                        
                        # Import timm here, after environment variables are set
                        import timm
                        
                        self.depth_backbone = timm.create_model(
                            config.depth_backbone,
                            pretrained=True,
                            num_classes=0,
                            global_pool="",
                        )
                        # Restore original setting
                        os.environ["HF_HUB_OFFLINE"] = original_offline
                        depth_backbone_created = True
                        logger.info(
                            f"✅ Successfully created depth backbone '{config.depth_backbone}' with pretrained weights "
                            f"(loaded from local cache)."
                        )
                    except Exception as e2:
                        # Restore original setting
                        os.environ["HF_HUB_OFFLINE"] = original_offline
                        logger.warning(
                            f"⚠️ Failed to load pretrained weights for depth backbone '{config.depth_backbone}': {e2}. "
                            f"Cache path checked: {model_file_found if model_file_found else 'Not found'}. "
                            f"Falling back to model without pretrained weights."
                        )
                else:
                    # Not offline mode and error is not cache-related, or offline mode but failed
                    if offline_mode:
                        logger.warning(
                            f"⚠️ Failed to load pretrained weights for depth backbone '{config.depth_backbone}' "
                            f"from local cache: {e}. "
                            f"Cache path checked: {model_file_found if model_file_found else 'Not found'}. "
                            f"Please ensure pretrained weights are downloaded to local cache, or set "
                            f"depth_backbone_pretrained=false to train from scratch."
                        )
                    else:
                        logger.warning(
                            f"⚠️ Failed to load pretrained weights for depth backbone '{config.depth_backbone}': {e}. "
                            f"Falling back to model without pretrained weights."
                        )
        
        # If pretrained failed or not requested, try without pretrained weights
        if not depth_backbone_created:
            try:
                logger.info(f"Creating depth backbone '{config.depth_backbone}' without pretrained weights...")
                import timm
                self.depth_backbone = timm.create_model(
                    config.depth_backbone,
                    pretrained=False,
                    num_classes=0,
                    global_pool="",
                )
                if use_pretrained:
                    logger.warning(
                        f"⚠️ Created depth backbone '{config.depth_backbone}' without pretrained weights "
                        f"(offline mode or network unavailable)."
                    )
                else:
                    logger.info(
                        f"Created depth backbone '{config.depth_backbone}' without pretrained weights."
                    )
            except Exception as e2:
                logger.error(f"❌ Failed to create depth backbone even without pretrained weights: {e2}")
                raise
        
        # Get depth feature dimension
        try:
            with torch.no_grad():
                try:
                    device = next(self.paligemma_with_expert.parameters()).device
                except:
                    device = torch.device("cpu")
                dummy_input = torch.zeros(1, 3, *config.image_resolution, device=device)
                depth_features = self.depth_backbone.forward_features(dummy_input)
                if isinstance(depth_features, (list, tuple)):
                    depth_dim = depth_features[-1].shape[-1]
                else:
                    depth_dim = depth_features.shape[-1]
        except Exception as e:
            logger.error(f"Failed to get depth feature dimension: {e}")
            raise
        
        # Projection layer to align depth features with LLM dimension
        self.depth_proj = nn.Linear(depth_dim, llm_dim)
        
        # No cross-attention for sequence_concat
        self.cross_modal_fusion = None
        
        logger.info(
            f"Initialized sequence_concat fusion: "
            f"depth_dim={depth_dim}, llm_dim={llm_dim}"
        )
    
    def _preprocess_depth_images(
        self, 
        depth_images: List[torch.Tensor]
    ) -> List[torch.Tensor]:
        """
        Preprocess depth images for fusion.
        
        Handles:
        - Single channel depth images [B, 1, H, W] or [B, H, W]
        - Normalization to [0, 1] then [-1, 1]
        - Conversion to 3 channels for PaliGemma vision tower
        - Resize and pad to target resolution
        
        Args:
            depth_images: List of depth image tensors
        
        Returns:
            List of preprocessed depth images [B, 3, H, W]
        """
        preprocessed = []
        target_height = self.config.image_resolution[0]
        target_width = self.config.image_resolution[1] if len(self.config.image_resolution) > 1 else self.config.image_resolution[0]
        
        for depth_img in depth_images:
            # Handle different input formats
            if depth_img.dim() == 3:  # [B, H, W]
                depth_img = depth_img.unsqueeze(1)  # [B, 1, H, W]
            elif depth_img.dim() == 4 and depth_img.shape[1] != 1:
                # If already multi-channel, take first channel
                if depth_img.shape[1] > 1:
                    depth_img = depth_img[:, 0:1, :, :]
            
            # Normalize depth values to [0, 1]
            depth_min = depth_img.min()
            depth_max = depth_img.max()
            if depth_max > depth_min:
                depth_img = (depth_img - depth_min) / (depth_max - depth_min + 1e-8)
            
            # Convert to 3 channels (repeat single channel)
            if depth_img.shape[1] == 1:
                depth_img = depth_img.repeat(1, 3, 1, 1)  # [B, 3, H, W]
            
            # Resize and pad to target resolution
            depth_img = resize_with_pad_torch(
                depth_img,
                height=target_height,
                width=target_width,
                mode="bilinear"
            )
            
            # Normalize to [-1, 1] (matching RGB preprocessing)
            depth_img = depth_img * 2.0 - 1.0
            
            preprocessed.append(depth_img)
        
        return preprocessed
    
    def _extract_depth_features(
        self, 
        depth_images: List[torch.Tensor],
        use_paligemma: bool = False
    ) -> List[torch.Tensor]:
        """
        Extract depth features.
        
        Args:
            depth_images: List of preprocessed depth images [B, 3, H, W]
            use_paligemma: If True, use PaliGemma vision tower; if False, use depth backbone
        
        Returns:
            List of depth feature tensors [B, N_depth, C_depth]
        """
        depth_features = []
        
        if use_paligemma:
            # Use PaliGemma vision tower (for cross-attention method)
            for depth_img in depth_images:
                def depth_embed_func(depth_img):
                    return self.paligemma_with_expert.embed_image(depth_img)
                
                feat = self._apply_checkpoint(depth_embed_func, depth_img)
                depth_features.append(feat)
        else:
            # Use independent ViT backbone (for sequence_concat method)
            if self.depth_backbone is None:
                raise ValueError("Depth backbone not initialized. Set use_depth=True in config.")
            
            for depth_img in depth_images:
                # Extract features using depth backbone
                feat = self.depth_backbone.forward_features(depth_img)
                
                # Handle different output formats from timm models
                if isinstance(feat, (list, tuple)):
                    feat = feat[-1]  # Take last layer features
                
                # Reshape to [B, N, C] format
                if feat.dim() == 4:  # [B, C, H, W]
                    B, C, H, W = feat.shape
                    feat = feat.view(B, C, H * W).transpose(1, 2)  # [B, H*W, C]
                elif feat.dim() == 3:  # [B, N, C]
                    pass  # Already in correct format
                else:
                    raise ValueError(f"Unexpected depth feature shape: {feat.shape}")
                
                depth_features.append(feat)
        
        return depth_features
    
    def embed_prefix(
        self,
        images: List[torch.Tensor],
        img_masks: List[torch.Tensor],
        lang_tokens: torch.Tensor,
        lang_masks: torch.Tensor,
        depth_images: Optional[List[torch.Tensor]] = None,
        depth_masks: Optional[List[torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Embed images and language tokens with optional depth fusion.
        
        Supports both sequence_concat and cross_attention fusion methods.
        
        Args:
            images: List of RGB images [B, 3, H, W]
            img_masks: List of image masks [B]
            lang_tokens: Language tokens [B, L]
            lang_masks: Language masks [B, L]
            depth_images: Optional list of depth images
            depth_masks: Optional list of depth masks
        
        Returns:
            Tuple of (embeddings, pad_masks, att_masks)
        """
        embs = []
        pad_masks = []
        att_masks = []
        
        # Process RGB images (always present)
        rgb_embs = []
        rgb_img_masks = []
        
        for img, img_mask in zip(images, img_masks, strict=True):
            def image_embed_func(img):
                return self.paligemma_with_expert.embed_image(img)
            
            img_emb = self._apply_checkpoint(image_embed_func, img)
            bsize, num_img_embs = img_emb.shape[:2]
            
            rgb_embs.append(img_emb)
            rgb_img_masks.append(img_mask)
        
        # Process depth images if available
        if (self.config.use_depth and 
            depth_images is not None and 
            len(depth_images) > 0):
            
            fusion_method = self.config.depth_fusion_method
            logger.debug(f"Processing {len(depth_images)} depth images with fusion method: {fusion_method}")
            
            # Preprocess depth images
            preprocessed_depth = self._preprocess_depth_images(depth_images)
            
            if fusion_method == "sequence_concat":
                # Sequence Concat: Extract depth features using independent ViT backbone
                depth_feat_list = self._extract_depth_features(preprocessed_depth, use_paligemma=False)
                # Project depth features to LLM dimension
                depth_feat_list = [self.depth_proj(feat) for feat in depth_feat_list]
                
                # Concatenate RGB and Depth features sequentially
                for rgb_emb, depth_feat, img_mask in zip(rgb_embs, depth_feat_list, rgb_img_masks, strict=True):
                    bsize = rgb_emb.shape[0]
                    num_rgb = rgb_emb.shape[1]
                    num_depth = depth_feat.shape[1]
                    
                    # Concatenate RGB and Depth features
                    embs.append(rgb_emb)
                    pad_masks.append(img_mask[:, None].expand(bsize, num_rgb))
                    att_masks += [0] * num_rgb
                    
                    embs.append(depth_feat)
                    depth_mask = depth_masks[0] if depth_masks else img_mask
                    pad_masks.append(depth_mask[:, None].expand(bsize, num_depth))
                    att_masks += [0] * num_depth
                    
            elif fusion_method == "cross_attention":
                # Cross-Attention: Extract depth features using PaliGemma vision tower
                depth_feat_list = self._extract_depth_features(preprocessed_depth, use_paligemma=True)
                # Project depth features to LLM dimension (if needed)
                depth_feat_list = [self.depth_proj(feat) for feat in depth_feat_list]
                
                if len(rgb_embs) != len(depth_feat_list):
                    raise ValueError(
                        f"RGB and Depth image counts do not match: "
                        f"RGB images: {len(rgb_embs)}, Depth images: {len(depth_feat_list)}"
                    )
                
                # Apply cross-attention fusion
                fused_rgb_list = []
                fused_depth_list = []
                
                for rgb_emb, depth_feat in zip(rgb_embs, depth_feat_list, strict=True):
                    rgb_fused, depth_fused = self.cross_modal_fusion(rgb_emb, depth_feat)
                    fused_rgb_list.append(rgb_fused)
                    fused_depth_list.append(depth_fused)
                
                # Add both fused features to embeddings
                for rgb_fused, depth_fused, img_mask in zip(
                    fused_rgb_list, fused_depth_list, rgb_img_masks, strict=True
                ):
                    bsize = rgb_fused.shape[0]
                    num_rgb = rgb_fused.shape[1]
                    num_depth = depth_fused.shape[1]
                    
                    embs.append(rgb_fused)
                    pad_masks.append(img_mask[:, None].expand(bsize, num_rgb))
                    att_masks += [0] * num_rgb
                    
                    embs.append(depth_fused)
                    depth_mask = depth_masks[0] if depth_masks else img_mask
                    pad_masks.append(depth_mask[:, None].expand(bsize, num_depth))
                    att_masks += [0] * num_depth
        else:
            # No depth images, use RGB only
            for rgb_emb, img_mask in zip(rgb_embs, rgb_img_masks):
                bsize, num_img_embs = rgb_emb.shape[:2]
                embs.append(rgb_emb)
                pad_masks.append(img_mask[:, None].expand(bsize, num_img_embs))
                att_masks += [0] * num_img_embs
        
        # Process language tokens
        def lang_embed_func(lang_tokens):
            lang_emb = self.paligemma_with_expert.embed_language_tokens(lang_tokens)
            lang_emb_dim = lang_emb.shape[-1]
            return lang_emb * math.sqrt(lang_emb_dim)
        
        lang_emb = self._apply_checkpoint(lang_embed_func, lang_tokens)
        embs.append(lang_emb)
        pad_masks.append(lang_masks)
        
        num_lang_embs = lang_emb.shape[1]
        att_masks += [0] * num_lang_embs
        
        # Concatenate all embeddings
        embs = torch.cat(embs, dim=1)
        pad_masks = torch.cat(pad_masks, dim=1)
        att_masks = torch.tensor(att_masks, dtype=torch.bool, device=pad_masks.device)
        
        bsize = pad_masks.shape[0]
        att_masks = att_masks[None, :].expand(bsize, len(att_masks))
        
        return embs, pad_masks, att_masks
    
    def forward(
        self,
        images: List[Tensor],
        img_masks: List[Tensor],
        tokens: Tensor,
        masks: Tensor,
        actions: Tensor,
        depth_images: Optional[List[Tensor]] = None,
        depth_masks: Optional[List[Tensor]] = None,
        noise: Optional[Tensor] = None,
        time: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Forward pass with optional depth support.
        
        Extends parent forward to support depth images.
        Note: PI05 uses 'tokens' and 'masks' instead of 'lang_tokens' and 'lang_masks'.
        PI05 does not use 'state' parameter.
        """
        if noise is None:
            noise = self.sample_noise(actions.shape, actions.device)
        
        if time is None:
            time = self.sample_time(actions.shape[0], actions.device)
        
        time_expanded = time[:, None, None]
        x_t = time_expanded * noise + (1 - time_expanded) * actions
        u_t = noise - actions
        
        # Use extended embed_prefix with depth support
        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(
            images=images,
            img_masks=img_masks,
            lang_tokens=tokens,
            lang_masks=masks,
            depth_images=depth_images,
            depth_masks=depth_masks,
        )
        
        # PI05 embed_suffix signature: embed_suffix(noisy_actions, timestep) - no state parameter
        suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = self.embed_suffix(x_t, time)
        
        # Continue with original forward logic
        if (
            self.paligemma_with_expert.paligemma.language_model.layers[0].self_attn.q_proj.weight.dtype
            == torch.bfloat16
        ):
            suffix_embs = suffix_embs.to(dtype=torch.bfloat16)
            prefix_embs = prefix_embs.to(dtype=torch.bfloat16)
        
        pad_masks = torch.cat([prefix_pad_masks, suffix_pad_masks], dim=1)
        att_masks = torch.cat([prefix_att_masks, suffix_att_masks], dim=1)
        
        att_2d_masks = make_att_2d_masks(pad_masks, att_masks)
        position_ids = torch.cumsum(pad_masks, dim=1) - 1
        
        att_2d_masks_4d = self._prepare_attention_masks_4d(att_2d_masks)
        
        def forward_func(prefix_embs, suffix_embs, att_2d_masks_4d, position_ids, adarms_cond):
            # PI05 expects adarms_cond as a list [None, adarms_cond]
            # None for paligemma, adarms_cond for gemma_expert
            (_, suffix_out), _ = self.paligemma_with_expert.forward(
                attention_mask=att_2d_masks_4d,
                position_ids=position_ids,
                inputs_embeds=[prefix_embs, suffix_embs],
                adarms_cond=[None, adarms_cond],  # PI05 format: [None for paligemma, adarms_cond for gemma_expert]
            )
            return suffix_out
        
        suffix_out = self._apply_checkpoint(
            forward_func, prefix_embs, suffix_embs, att_2d_masks_4d, position_ids, adarms_cond
        )
        
        # Training: predict u_t = noise - actions; compute MSE with v_t = action_out_proj(suffix_out)
        suffix_out = suffix_out[:, -self.config.chunk_size :]
        suffix_out = suffix_out.to(dtype=torch.float32)
        v_t = self._apply_checkpoint(lambda s: self.action_out_proj(s), suffix_out)
        return F.mse_loss(u_t, v_t, reduction="none")
    
    def sample_actions(
        self,
        images: List[Tensor],
        img_masks: List[Tensor],
        tokens: Tensor,
        masks: Tensor,
        depth_images: Optional[List[Tensor]] = None,
        depth_masks: Optional[List[Tensor]] = None,
        noise=None,
        num_steps=None,
        **kwargs
    ) -> Tensor:
        """
        Sample actions with optional depth support.
        
        Extends parent sample_actions to support depth images.
        Note: PI05 uses 'tokens' and 'masks' instead of 'lang_tokens' and 'lang_masks'.
        """
        if num_steps is None:
            num_steps = self.config.num_inference_steps

        bsize = tokens.shape[0]
        device = tokens.device

        if noise is None:
            actions_shape = (
                bsize,
                self.config.chunk_size,
                self.config.max_action_dim,
            )
            noise = self.sample_noise(actions_shape, device)

        # Use extended embed_prefix with depth support
        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(
            images=images,
            img_masks=img_masks,
            lang_tokens=tokens,
            lang_masks=masks,
            depth_images=depth_images,
            depth_masks=depth_masks,
        )
        
        # Align with base PI05Pytorch.sample_actions: prefix-only forward for KV cache, then denoise loop
        from lerobot.policies.pi05.modeling_pi05 import make_att_2d_masks
        prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1
        prefix_att_2d_masks_4d = self._prepare_attention_masks_4d(prefix_att_2d_masks)

        self.paligemma_with_expert.paligemma.language_model.config._attn_implementation = "eager"  # noqa: SLF001
        _, past_key_values = self.paligemma_with_expert.forward(
            attention_mask=prefix_att_2d_masks_4d,
            position_ids=prefix_position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=True,
        )

        dt = -1.0 / num_steps
        original_past_key_values = past_key_values
        x_t = noise
        for step in range(num_steps):
            time = 1.0 + step * dt
            time_tensor = torch.tensor(time, dtype=torch.float32, device=device).expand(bsize)

            if original_past_key_values is not None:
                if hasattr(original_past_key_values, "clone"):
                    past_kv_copy = original_past_key_values.clone()
                else:
                    from transformers.cache_utils import DynamicCache
                    past_kv_copy = DynamicCache()
                    for i in range(len(original_past_key_values.key_cache)):
                        past_kv_copy.key_cache.append(original_past_key_values.key_cache[i].clone())
                        past_kv_copy.value_cache.append(original_past_key_values.value_cache[i].clone())
            else:
                past_kv_copy = None

            v_t = self.denoise_step(prefix_pad_masks, past_kv_copy, x_t, time_tensor)
            x_t = x_t + dt * v_t

        return x_t

