"""
PI0 Policy Wrapper with Depth Support

This module extends PI0Policy to support depth image fusion.
It handles depth image extraction from batch and passes them to the model.
"""

import logging
from typing import Optional, List, Tuple

import torch
from torch import Tensor

from lerobot.policies.pi0.modeling_pi0 import PI0Policy
from lerobot.utils.constants import OBS_LANGUAGE_TOKENS, OBS_LANGUAGE_ATTENTION_MASK, OBS_STATE, ACTION
from kuavo_train.wrapper.policy.pi0.PI0ConfigWrapper import CustomPI0ConfigWrapper
from kuavo_train.wrapper.policy.pi0.PI0ModelWrapper import CustomPI0Pytorch

logger = logging.getLogger(__name__)


class CustomPI0PolicyWrapper(PI0Policy):
    """
    Extended PI0Policy with depth image support.
    
    Supports two depth fusion methods:
    1. sequence_concat: Sequence concatenation (OpenVLA-Depth style)
    2. cross_attention: Cross-attention bidirectional fusion
    """
    
    config_class = CustomPI0ConfigWrapper
    name = "custom_pi0"
    
    def __init__(
        self,
        config: CustomPI0ConfigWrapper,
        **kwargs,
    ):
        """
        Initialize CustomPI0Policy with depth support.
        
        Args:
            config: CustomPI0ConfigWrapper configuration instance
            **kwargs: Additional arguments passed to parent class
        """
        # Initialize parent class first (this will call validate_features)
        super().__init__(config, **kwargs)
        
        # Backward compatibility: older configs or pure RGB checkpoints may be
        # loaded as the base PI0Config without depth-related fields. We inject
        # safe defaults here so deployment never crashes when depth is not used.
        if not hasattr(config, "use_depth"):
            config.use_depth = False
        if not hasattr(config, "depth_features"):
            config.depth_features = []
        
        # Replace the model with CustomPI0Pytorch if depth is explicitly enabled
        # and at least one depth feature is configured.
        if config.use_depth and config.depth_features:
            logger.info(
                f"Initializing CustomPI0Pytorch with depth fusion method: "
                f"{config.depth_fusion_method}"
            )
            # Reinitialize the model with CustomPI0Pytorch
            self.model = CustomPI0Pytorch(config, rtc_processor=self.rtc_processor)
            
            # Enable gradient checkpointing if requested
            if config.gradient_checkpointing:
                self.model.gradient_checkpointing_enable()
            
            self.model.to(config.device)
        else:
            # Use original model if depth is not enabled
            logger.info("Using original PI0Pytorch (depth not enabled)")
    
    def _get_device(self) -> torch.device:
        """
        Get the device of the model, handling DataParallel wrapping.
        
        In DataParallel mode, returns the primary device (first GPU in device_ids).
        Input tensors should be placed on the primary device, and DataParallel will
        automatically distribute them to other GPUs.
        
        Returns:
            torch.device: The primary device where input tensors should be placed
        """
        # Check if model is wrapped with DataParallel
        if isinstance(self.model, torch.nn.DataParallel):
            # In DataParallel, the primary device is the first device in device_ids
            # Input tensors should be on the primary device
            primary_device_id = self.model.device_ids[0]
            device = torch.device(f"cuda:{primary_device_id}")
        else:
            # Not wrapped with DataParallel, get device from model parameters
            try:
                device = next(self.parameters()).device
            except StopIteration:
                # Fallback to config device if no parameters
                device = torch.device(self.config.device)
        return device
    
    def _extract_depth_images(
        self, 
        batch: dict[str, Tensor]
    ) -> Tuple[Optional[List[Tensor]], Optional[List[Tensor]]]:
        """
        Extract depth images from batch.
        
        In DataParallel mode:
        - Batch is already distributed to the correct GPU by DataParallel
        - All operations should use the device from batch tensors
        - Do NOT move tensors, as they're already on the correct device
        
        Args:
            batch: Training/inference batch dictionary
        
        Returns:
            Tuple of (depth_images, depth_masks) or (None, None) if no depth images
        """
        if not self.config.use_depth or not self.config.depth_features:
            return None, None
        
        depth_images = []
        depth_masks = []
        
        # In DataParallel, batch is already on the correct GPU
        # Always infer device from batch tensors to ensure consistency
        device = None
        for value in batch.values():
            if isinstance(value, torch.Tensor):
                device = value.device
                break
        
        # Fallback: only if no tensors found in batch (shouldn't happen)
        if device is None:
            if isinstance(self.model, torch.nn.DataParallel):
                device = next(self.model.module.parameters()).device
            else:
                device = self._get_device()
        
        # Extract depth images based on depth_features config
        present_depth_keys = [
            key for key in self.config.depth_features 
            if key in batch
        ]
        
        if len(present_depth_keys) == 0:
            # No depth images in batch, return None
            logger.debug("No depth images found in batch")
            return None, None
        
        for key in present_depth_keys:
            depth_img = batch[key]
            
            # In DataParallel mode, batch is already on the correct GPU - DO NOT MOVE
            # Only move in single GPU mode if needed
            if not isinstance(self.model, torch.nn.DataParallel):
                if depth_img.device != device:
                    depth_img = depth_img.to(device)
            
            # Ensure float32 dtype for consistency
            if depth_img.dtype != torch.float32:
                depth_img = depth_img.to(torch.float32)
            
            # Handle different input formats
            # Support [B, C, H, W], [B, H, W], [B, H, W, 1]
            if depth_img.dim() == 3:  # [B, H, W]
                depth_img = depth_img.unsqueeze(1)  # [B, 1, H, W]
            elif depth_img.dim() == 4:
                if depth_img.shape[1] != 1 and depth_img.shape[1] > 1:
                    # Multi-channel, take first channel
                    depth_img = depth_img[:, 0:1, :, :]
                # If shape[1] == 1, already single channel
            
            depth_images.append(depth_img)
            
            # Create mask using depth_img.device to ensure it's on the same device
            # This is critical in DataParallel mode
            bsize = depth_img.shape[0]
            mask = torch.ones(bsize, dtype=torch.bool, device=depth_img.device)
            depth_masks.append(mask)
        
        return depth_images, depth_masks
    
    def _preprocess_images_safe(self, batch: dict[str, Tensor]) -> tuple[list[Tensor], list[Tensor]]:
        """
        Preprocess images for the model, with DataParallel support.
        
        This is a wrapper around the parent's _preprocess_images that handles
        DataParallel-wrapped models correctly.
        
        In DataParallel mode:
        - Batch is already distributed to the correct GPU by DataParallel
        - All operations should use the device from batch tensors
        - Do NOT move tensors, as they're already on the correct device
        
        Args:
            batch: Training/inference batch dictionary
        
        Returns:
            Tuple of (images, img_masks)
        """
        # In DataParallel, batch is already on the correct GPU
        # Always infer device from batch tensors to ensure consistency
        device = None
        for value in batch.values():
            if isinstance(value, torch.Tensor):
                device = value.device
                break
        
        # Fallback: only if no tensors found in batch (shouldn't happen)
        if device is None:
            if isinstance(self.model, torch.nn.DataParallel):
                device = next(self.model.module.parameters()).device
            else:
                device = self._get_device()
        
        images = []
        img_masks = []

        present_img_keys = [key for key in self.config.image_features if key in batch]
        missing_img_keys = [key for key in self.config.image_features if key not in batch]

        if len(present_img_keys) == 0:
            raise ValueError(
                f"All image features are missing from the batch. At least one expected. "
                f"(batch: {batch.keys()}) (image_features: {self.config.image_features})"
            )

        for key in present_img_keys:
            img = batch[key]

            # In DataParallel mode, batch is already on the correct GPU - DO NOT MOVE
            # Only move in single GPU mode if needed
            if not isinstance(self.model, torch.nn.DataParallel):
                if img.device != device:
                    img = img.to(device)

            # Ensure float32 dtype for consistency
            if img.dtype != torch.float32:
                img = img.to(torch.float32)

            # from openpi preprocess_observation_pytorch: Handle both [B, C, H, W] and [B, H, W, C] formats
            is_channels_first = img.shape[1] == 3  # Check if channels are in dimension 1

            if is_channels_first:
                # Convert [B, C, H, W] to [B, H, W, C] for processing
                img = img.permute(0, 2, 3, 1)

            # from openpi preprocess_observation_pytorch: Resize with padding if needed
            # Import resize_with_pad_torch if not already imported
            from lerobot.policies.pi0.modeling_pi0 import resize_with_pad_torch
            if img.shape[1:3] != self.config.image_resolution:
                img = resize_with_pad_torch(img, *self.config.image_resolution)

            # Normalize from [0,1] to [-1,1] as expected by siglip
            img = img * 2.0 - 1.0

            # from openpi preprocess_observation_pytorch: Convert back to [B, C, H, W] format if it was originally channels-first
            if is_channels_first:
                img = img.permute(0, 3, 1, 2)  # [B, H, W, C] -> [B, C, H, W]

            images.append(img)

            # Create mask using img.device to ensure it's on the same device as img
            # This is critical in DataParallel mode
            bsize = img.shape[0]
            mask = torch.ones(bsize, dtype=torch.bool, device=img.device)
            img_masks.append(mask)

        # Handle missing images by adding empty placeholders
        # Use the same logic as parent class: padded with -1 for SigLIP
        for _num_empty_cameras in range(len(missing_img_keys)):
            if len(images) > 0:
                # Use the last processed image as reference (ensures same device)
                ref_img = images[-1]
                img = torch.ones_like(ref_img) * -1  # padded with -1 for SigLIP
                mask = torch.zeros_like(img_masks[-1])  # mask is zero for empty cameras
            else:
                # No reference image, create default size
                # Use device from batch (already inferred above)
                h, w = self.config.image_resolution
                bsize = batch[present_img_keys[0]].shape[0] if present_img_keys else 1
                # Use device from batch to ensure consistency
                img = torch.ones(bsize, 3, h, w, dtype=torch.float32, device=device) * -1
                mask = torch.zeros(bsize, dtype=torch.bool, device=device)
            images.append(img)
            img_masks.append(mask)

        return images, img_masks
    
    def forward(
        self, 
        batch: dict[str, Tensor], 
        reduction: str = "mean"
    ) -> tuple[Tensor, dict]:
        """
        Run the batch through the model and compute the loss for training.
        
        Extended to support depth images for fusion.
        
        In DataParallel mode:
        - DataParallel automatically distributes batch to each GPU
        - The batch is already on the correct GPU when forward is called
        - DO NOT move batch tensors - use them directly
        
        Args:
            batch: Training batch containing observations and actions
            reduction: How to reduce the loss. Options:
                - "mean": Return scalar mean loss (default)
                - "none": Return per-sample losses of shape (batch_size,)
        
        Returns:
            Tuple of (loss, loss_dict)
        """
        # In DataParallel mode, batch is already distributed to the correct GPU
        # DO NOT move batch - DataParallel handles device placement
        # In single GPU mode, ensure batch is on the correct device
        if isinstance(self.model, torch.nn.DataParallel):
            # DataParallel has already distributed batch to each GPU
            # Use batch directly without any device movement
            batch_to_use = batch
        else:
            # Single GPU mode: ensure batch is on the correct device
            device = self._get_device()
            batch_to_use = {}
            for key, value in batch.items():
                if isinstance(value, torch.Tensor):
                    if value.device != device:
                        batch_to_use[key] = value.to(device, non_blocking=True)
                    else:
                        batch_to_use[key] = value
                else:
                    batch_to_use[key] = value
        
        # Prepare inputs (RGB images, language tokens, state, actions)
        # _preprocess_images_safe and _extract_depth_images handle DataParallel correctly
        images, img_masks = self._preprocess_images_safe(batch_to_use)
        lang_tokens, lang_masks = batch_to_use[OBS_LANGUAGE_TOKENS], batch_to_use[OBS_LANGUAGE_ATTENTION_MASK]
        state = self.prepare_state(batch_to_use)
        actions = self.prepare_action(batch_to_use)
        
        # Extract depth images if available
        depth_images, depth_masks = self._extract_depth_images(batch_to_use)
        
        # Check if model supports depth (CustomPI0Pytorch)
        if isinstance(self.model, CustomPI0Pytorch) and depth_images is not None:
            # Use extended forward with depth support
            losses = self.model.forward(
                images=images,
                img_masks=img_masks,
                lang_tokens=lang_tokens,
                lang_masks=lang_masks,
                state=state,
                actions=actions,
                depth_images=depth_images,
                depth_masks=depth_masks,
            )
        else:
            # Original forward (no depth)
            losses = self.model.forward(images, img_masks, lang_tokens, lang_masks, state, actions)
        
        # Truncate losses to actual action dimensions
        original_action_dim = self.config.output_features[ACTION].shape[0]
        losses = losses[:, :, :original_action_dim]
        
        loss_dict = {
            "loss_per_dim": losses.mean(dim=[0, 1]).detach().cpu().numpy().tolist(),
        }
        
        if reduction == "none":
            # Return per-sample losses (B,) by averaging over time and action dims
            per_sample_loss = losses.mean(dim=(1, 2))
            loss_dict["loss"] = per_sample_loss.mean().item()
            return per_sample_loss, loss_dict
        else:
            # Default: return scalar mean loss
            loss = losses.mean()
            loss_dict["loss"] = loss.item()
            return loss, loss_dict
    
    
    @torch.no_grad()
    def predict_action_chunk(
        self, 
        batch: dict[str, Tensor], 
        **kwargs
    ) -> Tensor:
        """
        Predict a chunk of actions given environment observations.
        
        Extended to support depth images for fusion.
        
        Args:
            batch: Inference batch containing observations
            **kwargs: Additional arguments for action selection
        
        Returns:
            Predicted action chunk [B, chunk_size, action_dim]
        """
        self.eval()
        
        # Prepare inputs
        # Override _preprocess_images to handle DataParallel
        images, img_masks = self._preprocess_images_safe(batch)
        lang_tokens, lang_masks = batch[OBS_LANGUAGE_TOKENS], batch[OBS_LANGUAGE_ATTENTION_MASK]
        state = self.prepare_state(batch)
        
        # Extract depth images if available
        depth_images, depth_masks = self._extract_depth_images(batch)
        
        # Predict actions using model.sample_actions
        if isinstance(self.model, CustomPI0Pytorch) and depth_images is not None:
            # Use extended sample_actions with depth support
            actions = self.model.sample_actions(
                images=images,
                img_masks=img_masks,
                lang_tokens=lang_tokens,
                lang_masks=lang_masks,
                state=state,
                depth_images=depth_images,
                depth_masks=depth_masks,
                **kwargs
            )
        else:
            # Use original sample_actions (no depth)
            actions = self.model.sample_actions(
                images, img_masks, lang_tokens, lang_masks, state, **kwargs
            )
        
        # Truncate to original action dimensions
        original_action_dim = self.config.output_features[ACTION].shape[0]
        actions = actions[:, :, :original_action_dim]
        
        return actions
    

