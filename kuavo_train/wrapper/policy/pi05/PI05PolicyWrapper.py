"""
PI05 Policy Wrapper with Depth Support

This module extends PI05Policy to support depth image fusion using Cross-Attention method.
It handles depth image extraction from batch and passes them to the model.
"""

import logging
from typing import Optional, List, Tuple

import torch
from torch import Tensor

from lerobot.policies.pi05.modeling_pi05 import PI05Policy
from lerobot.utils.constants import OBS_LANGUAGE_TOKENS, OBS_LANGUAGE_ATTENTION_MASK, ACTION
from kuavo_train.wrapper.policy.pi05.PI05ConfigWrapper import CustomPI05ConfigWrapper
from kuavo_train.wrapper.policy.pi05.PI05ModelWrapper import CustomPI05Pytorch

logger = logging.getLogger(__name__)


class CustomPI05PolicyWrapper(PI05Policy):
    """
    Extended PI05Policy with depth image support.
    
    PI05 supports both sequence_concat (方案一) and cross_attention (方案二) depth fusion methods.
    """
    
    config_class = CustomPI05ConfigWrapper
    name = "custom_pi05"
    
    def __init__(
        self,
        config: CustomPI05ConfigWrapper,
        **kwargs,
    ):
        """
        Initialize CustomPI05Policy with depth support.
        
        Args:
            config: CustomPI05ConfigWrapper configuration instance
            **kwargs: Additional arguments passed to parent class
        """
        # Initialize parent class first (this will call validate_features)
        super().__init__(config, **kwargs)

        # Backward compatibility: older configs or pure RGB checkpoints may be
        # loaded as the base PI05Config without depth-related fields. We inject
        # safe defaults here so deployment never crashes when depth is not used.
        if not hasattr(config, "use_depth"):
            config.use_depth = False
        if not hasattr(config, "depth_features"):
            config.depth_features = []
        
        # Replace the model with CustomPI05Pytorch if depth is enabled
        if config.use_depth and config.depth_features:
            fusion_method = config.depth_fusion_method
            logger.info(
                f"Initializing CustomPI05Pytorch with {fusion_method} depth fusion"
            )
            # Reinitialize the model with CustomPI05Pytorch
            self.model = CustomPI05Pytorch(config, rtc_processor=self.rtc_processor)
            
            # Enable gradient checkpointing if requested
            if config.gradient_checkpointing:
                self.model.gradient_checkpointing_enable()
            
            self.model.to(config.device)
        else:
            # Use original model if depth is not enabled
            logger.info("Using original PI05Pytorch (depth not enabled)")
    
    def _get_device(self) -> torch.device:
        """Get the device of the model, handling DataParallel wrapping."""
        if isinstance(self.model, torch.nn.DataParallel):
            primary_device_id = self.model.device_ids[0]
            device = torch.device(f"cuda:{primary_device_id}")
        else:
            try:
                device = next(self.parameters()).device
            except StopIteration:
                device = torch.device(self.config.device)
        return device
    
    def _extract_depth_images(
        self, 
        batch: dict[str, Tensor]
    ) -> Tuple[Optional[List[Tensor]], Optional[List[Tensor]]]:
        """Extract depth images from batch."""
        if not self.config.use_depth or not self.config.depth_features:
            return None, None
        
        depth_images = []
        depth_masks = []
        
        device = None
        for value in batch.values():
            if isinstance(value, torch.Tensor):
                device = value.device
                break
        
        if device is None:
            device = self._get_device()
        
        present_depth_keys = [
            key for key in self.config.depth_features 
            if key in batch
        ]
        
        if len(present_depth_keys) == 0:
            return None, None
        
        for key in present_depth_keys:
            depth_img = batch[key]
            
            if not isinstance(self.model, torch.nn.DataParallel):
                if depth_img.device != device:
                    depth_img = depth_img.to(device)
            
            if depth_img.dtype != torch.float32:
                depth_img = depth_img.to(torch.float32)
            
            if depth_img.dim() == 3:
                depth_img = depth_img.unsqueeze(1)
            elif depth_img.dim() == 4:
                if depth_img.shape[1] != 1 and depth_img.shape[1] > 1:
                    depth_img = depth_img[:, 0:1, :, :]
            
            depth_images.append(depth_img)
            bsize = depth_img.shape[0]
            mask = torch.ones(bsize, dtype=torch.bool, device=depth_img.device)
            depth_masks.append(mask)
        
        return depth_images, depth_masks
    
    def _preprocess_images_safe(self, batch: dict[str, Tensor]) -> tuple[list[Tensor], list[Tensor]]:
        """Preprocess images for the model, with DataParallel support."""
        device = None
        for value in batch.values():
            if isinstance(value, torch.Tensor):
                device = value.device
                break
        
        if device is None:
            device = self._get_device()
        
        images = []
        img_masks = []

        present_img_keys = [key for key in self.config.image_features if key in batch]
        missing_img_keys = [key for key in self.config.image_features if key not in batch]

        if len(present_img_keys) == 0:
            raise ValueError(
                f"All image features are missing from the batch. "
                f"(batch: {batch.keys()}) (image_features: {self.config.image_features})"
            )

        for key in present_img_keys:
            img = batch[key]

            if not isinstance(self.model, torch.nn.DataParallel):
                if img.device != device:
                    img = img.to(device)

            if img.dtype != torch.float32:
                img = img.to(torch.float32)

            is_channels_first = img.shape[1] == 3

            if is_channels_first:
                img = img.permute(0, 2, 3, 1)

            from lerobot.policies.pi05.modeling_pi05 import resize_with_pad_torch
            if img.shape[1:3] != self.config.image_resolution:
                img = resize_with_pad_torch(img, *self.config.image_resolution)

            img = img * 2.0 - 1.0

            if is_channels_first:
                img = img.permute(0, 3, 1, 2)

            images.append(img)
            bsize = img.shape[0]
            mask = torch.ones(bsize, dtype=torch.bool, device=img.device)
            img_masks.append(mask)

        for _num_empty_cameras in range(len(missing_img_keys)):
            if len(images) > 0:
                ref_img = images[-1]
                img = torch.ones_like(ref_img) * -1
                mask = torch.zeros_like(img_masks[-1])
            else:
                h, w = self.config.image_resolution
                bsize = batch[present_img_keys[0]].shape[0] if present_img_keys else 1
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
        """
        if isinstance(self.model, torch.nn.DataParallel):
            batch_to_use = batch
        else:
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
        
        images, img_masks = self._preprocess_images_safe(batch_to_use)
        tokens, masks = batch_to_use[OBS_LANGUAGE_TOKENS], batch_to_use[OBS_LANGUAGE_ATTENTION_MASK]
        actions = self.prepare_action(batch_to_use)
        
        depth_images, depth_masks = self._extract_depth_images(batch_to_use)
        
        if isinstance(self.model, CustomPI05Pytorch) and depth_images is not None:
            losses = self.model.forward(
                images=images,
                img_masks=img_masks,
                tokens=tokens,
                masks=masks,
                actions=actions,
                depth_images=depth_images,
                depth_masks=depth_masks,
            )
        else:
            losses = self.model.forward(images, img_masks, tokens, masks, actions)
        
        original_action_dim = self.config.output_features[ACTION].shape[0]
        losses = losses[:, :, :original_action_dim]
        
        loss_dict = {
            "loss_per_dim": losses.mean(dim=[0, 1]).detach().cpu().numpy().tolist(),
        }
        
        if reduction == "none":
            per_sample_loss = losses.mean(dim=(1, 2))
            loss_dict["loss"] = per_sample_loss.mean().item()
            return per_sample_loss, loss_dict
        else:
            loss = losses.mean()
            loss_dict["loss"] = loss.item()
            return loss, loss_dict
    
    @torch.no_grad()
    def predict_action_chunk(
        self, 
        batch: dict[str, Tensor], 
        **kwargs
    ) -> Tensor:
        """Predict a chunk of actions given environment observations."""
        self.eval()
        
        images, img_masks = self._preprocess_images_safe(batch)
        tokens, masks = batch[OBS_LANGUAGE_TOKENS], batch[OBS_LANGUAGE_ATTENTION_MASK]
        
        depth_images, depth_masks = self._extract_depth_images(batch)
        
        if isinstance(self.model, CustomPI05Pytorch) and depth_images is not None:
            actions = self.model.sample_actions(
                images=images,
                img_masks=img_masks,
                tokens=tokens,
                masks=masks,
                depth_images=depth_images,
                depth_masks=depth_masks,
                **kwargs
            )
        else:
            actions = self.model.sample_actions(images, img_masks, tokens, masks, **kwargs)
        
        original_action_dim = self.config.output_features[ACTION].shape[0]
        actions = actions[:, :, :original_action_dim]
        
        return actions

