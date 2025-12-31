import torch.nn as nn
from kuavo_train.wrapper.policy.act.ACTConfigWrapper import CustomACTConfigWrapper
import math
from collections import deque
from typing import Callable

import einops
import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812
import torchvision
from torch import Tensor, nn

from lerobot.utils.constants import OBS_ENV_STATE, OBS_IMAGES, OBS_STATE, ACTION
from lerobot.policies.utils import (
    get_device_from_parameters,
    get_dtype_from_parameters,
    get_output_shape,
)

from torchvision.ops.misc import FrozenBatchNorm2d
from torchvision.models._utils import IntermediateLayerGetter
from lerobot.policies.act.modeling_act import (ACT,
                                               ACTSinusoidalPositionEmbedding2d
                                                           )

OBS_DEPTH = "observation.depth"

class CrossModalAttentionFusion(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.rgb_to_depth_attn = nn.MultiheadAttention(embed_dim, num_heads)
        self.depth_to_rgb_attn = nn.MultiheadAttention(embed_dim, num_heads)
        self.norm_rgb = nn.LayerNorm(embed_dim)
        self.norm_depth = nn.LayerNorm(embed_dim)

    def forward(self, rgb_tokens_list, depth_tokens_list):
        """
        rgb_tokens_list: list of tensors, each (H*W, B, C)
        depth_tokens_list: list of tensors, each (H*W, B, C)
        """
        fused_rgb_list, fused_depth_list = [], []

        for rgb_tokens, depth_tokens in zip(rgb_tokens_list, depth_tokens_list):
            # RGB as Query, Depth as Key/Value
            rgb_q = rgb_tokens
            depth_kv = depth_tokens
            rgb_out, _ = self.rgb_to_depth_attn(query=rgb_q, key=depth_kv, value=depth_kv)
            rgb_out = self.norm_rgb(rgb_out + rgb_tokens)   # Residual + Normalisation

            # RGB as Query, Depth as Key/Value
            depth_q = depth_tokens
            rgb_kv = rgb_tokens
            depth_out, _ = self.depth_to_rgb_attn(query=depth_q, key=rgb_kv, value=rgb_kv)
            depth_out = self.norm_depth(depth_out + depth_tokens)

            fused_rgb_list.append(rgb_out)
            fused_depth_list.append(depth_out)

        return fused_rgb_list, fused_depth_list

class CustomACTModelWrapper(ACT):
    def __init__(self, config: CustomACTConfigWrapper):
        # BERT style VAE encoder with input tokens [cls, robot_state, *action_sequence].
        # The cls token forms parameters of the latent's distribution (like this [*means, *log_variances]).
        super().__init__(config)
        

        if self.config.use_depth and self.config.depth_features:
            depth_backbone_model = getattr(torchvision.models, config.depth_backbone)(
                replace_stride_with_dilation=[False, False, config.replace_final_stride_with_dilation],
                weights=config.pretrained_depth_backbone_weights,
                norm_layer=FrozenBatchNorm2d,
            )
            if isinstance(depth_backbone_model.conv1, nn.Conv2d):
                old_conv = depth_backbone_model.conv1
                depth_backbone_model.conv1 = nn.Conv2d(
                    in_channels=1,
                    out_channels=old_conv.out_channels,
                    kernel_size=old_conv.kernel_size,
                    stride=old_conv.stride,
                    padding=old_conv.padding,
                    bias=old_conv.bias is not None
                )
                with torch.no_grad():
                    depth_backbone_model.conv1.weight = nn.Parameter(old_conv.weight.mean(dim=1, keepdim=True))

            # Note: The assumption here is that we are using a ResNet model (and hence layer4 is the final
            # feature map).
            # Note: The forward method of this returns a dict: {"feature_map": output}.
            self.depth_backbone = IntermediateLayerGetter(depth_backbone_model, return_layers={"layer4": "feature_map"})

        if self.config.use_depth and self.config.depth_features:
            self.encoder_depth_feat_input_proj = nn.Conv2d(
                depth_backbone_model.fc.in_features, config.dim_model, kernel_size=1
            )

        if self.config.use_depth and self.config.depth_features:
            self.encoder_depth_feat_pos_embed = ACTSinusoidalPositionEmbedding2d(config.dim_model // 2)
            self.cross_modal_fusion = CrossModalAttentionFusion(embed_dim=config.dim_model, num_heads=8)
            self.cross_modal_fusion_proj = nn.Linear(config.dim_model * 2, config.dim_model)


    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, tuple[Tensor, Tensor] | tuple[None, None]]:
        """A forward pass through the Action Chunking Transformer (with optional VAE encoder).

        `batch` should have the following structure:
        {
            [robot_state_feature] (optional): (B, state_dim) batch of robot states.

            [image_features]: (B, n_cameras, C, H, W) batch of images.
                AND/OR
            [depth_features]: (B, n_cameras, 1, H, W) batch of depth images.
                AND/OR
            [env_state_feature]: (B, env_dim) batch of environment states.

            [action_feature] (optional, only if training with VAE): (B, chunk_size, action dim) batch of actions.
        }

        Returns:
            (B, chunk_size, action_dim) batch of action sequences
            Tuple containing the latent PDF's parameters (mean, log(σ²)) both as (B, L) tensors where L is the
            latent dimension.
        """
        if self.config.use_vae and self.training:
            assert ACTION in batch, (
                "actions must be provided when using the variational objective in training mode."
            )

        batch_size = batch[OBS_IMAGES][0].shape[0] if OBS_IMAGES in batch else batch[OBS_ENV_STATE].shape[0]
        n_cam = len(batch[OBS_IMAGES]) if OBS_IMAGES in batch else 0

        # Prepare the latent for input to the transformer encoder.
        if self.config.use_vae and ACTION in batch and self.training:
            # Prepare the input to the VAE encoder: [cls, *joint_space_configuration, *action_sequence].
            cls_embed = einops.repeat(
                self.vae_encoder_cls_embed.weight, "1 d -> b 1 d", b=batch_size
            )  # (B, 1, D)
            if self.config.robot_state_feature:
                robot_state_embed = self.vae_encoder_robot_state_input_proj(batch[OBS_STATE])
                robot_state_embed = robot_state_embed.unsqueeze(1)  # (B, 1, D)
            action_embed = self.vae_encoder_action_input_proj(batch[ACTION])  # (B, S, D)

            if self.config.robot_state_feature:
                vae_encoder_input = [cls_embed, robot_state_embed, action_embed]  # (B, S+2, D)
            else:
                vae_encoder_input = [cls_embed, action_embed]
            vae_encoder_input = torch.cat(vae_encoder_input, axis=1)

            # Prepare fixed positional embedding.
            # Note: detach() shouldn't be necessary but leaving it the same as the original code just in case.
            pos_embed = self.vae_encoder_pos_enc.clone().detach()  # (1, S+2, D)

            # Prepare key padding mask for the transformer encoder. We have 1 or 2 extra tokens at the start of the
            # sequence depending whether we use the input states or not (cls and robot state)
            # False means not a padding token.
            cls_joint_is_pad = torch.full(
                (batch_size, 2 if self.config.robot_state_feature else 1),
                False,
                device=batch[OBS_STATE].device,
            )
            key_padding_mask = torch.cat(
                [cls_joint_is_pad, batch["action_is_pad"]], axis=1
            )  # (bs, seq+1 or 2)

            # Forward pass through VAE encoder to get the latent PDF parameters.
            cls_token_out = self.vae_encoder(
                vae_encoder_input.permute(1, 0, 2),
                pos_embed=pos_embed.permute(1, 0, 2),
                key_padding_mask=key_padding_mask,
            )[0]  # select the class token, with shape (B, D)
            latent_pdf_params = self.vae_encoder_latent_output_proj(cls_token_out)
            mu = latent_pdf_params[:, : self.config.latent_dim]
            # This is 2log(sigma). Done this way to match the original implementation.
            log_sigma_x2 = latent_pdf_params[:, self.config.latent_dim :]

            # Sample the latent with the reparameterization trick.
            latent_sample = mu + log_sigma_x2.div(2).exp() * torch.randn_like(mu)
        else:
            # When not using the VAE encoder, we set the latent to be all zeros.
            mu = log_sigma_x2 = None
            # TODO(rcadene, alexander-soare): remove call to `.to` to speedup forward ; precompute and use buffer
            latent_sample = torch.zeros([batch_size, self.config.latent_dim], dtype=torch.float32).to(
                batch[OBS_STATE].device
            )

        # Prepare transformer encoder inputs.
        encoder_in_tokens = [self.encoder_latent_input_proj(latent_sample)]
        encoder_in_pos_embed = list(self.encoder_1d_feature_pos_embed.weight.unsqueeze(1))
        # Robot state token.
        if self.config.robot_state_feature:
            encoder_in_tokens.append(self.encoder_robot_state_input_proj(batch[OBS_STATE]))
        # Environment state token.
        if self.config.env_state_feature:
            encoder_in_tokens.append(self.encoder_env_state_input_proj(batch[OBS_ENV_STATE]))
        
        # encoder_in_tokens_campre = []
        # encoder_in_pos_embed_campre = []
        if self.config.image_features:
            # For a list of images, the H and W may vary but H*W is constant.
            # NOTE: If modifying this section, verify on MPS devices that
            # gradients remain stable (no explosions or NaNs).
            # for img in batch[OBS_IMAGES]:
            #     cam_features = self.backbone(img)["feature_map"]
            #     cam_pos_embed = self.encoder_cam_feat_pos_embed(cam_features).to(dtype=cam_features.dtype)
            #     cam_features = self.encoder_img_feat_input_proj(cam_features)

            #     # Rearrange features to (sequence, batch, dim).
            #     cam_features = einops.rearrange(cam_features, "b c h w -> (h w) b c")
            #     cam_pos_embed = einops.rearrange(cam_pos_embed, "b c h w -> (h w) b c")
                
            #     # Extend immediately instead of accumulating and concatenating
            #     # Convert to list to extend properly
            #     encoder_in_tokens_campre.extend(list(cam_features))
            #     encoder_in_pos_embed_campre.extend(list(cam_pos_embed))
            imgs = torch.cat(batch[OBS_IMAGES], dim=0)

            cam_features = self.backbone(imgs)["feature_map"]              #  (n_cam*B, C', H', W')
            pos_cam_input = einops.rearrange(cam_features, " (v b) c h w -> v b c h w", v=n_cam, b=batch_size)
            cam_pos_embed = torch.cat([self.encoder_cam_feat_pos_embed(pos_cam_input[i]).to(dtype=cam_features.dtype) for i in range(n_cam)], dim=0)
            cam_features = self.encoder_img_feat_input_proj(cam_features)

            cam_features = cam_features.view(n_cam, batch_size, cam_features.size(1), cam_features.size(2), cam_features.size(3))
            cam_pos_embed = cam_pos_embed.view(n_cam, 1, cam_pos_embed.size(1), cam_pos_embed.size(2), cam_pos_embed.size(3))

            # rearrange 
            cam_features = einops.rearrange(cam_features, "v b c h w -> v (h w) b c")
            cam_pos_embed = einops.rearrange(cam_pos_embed, "v b c h w -> v (h w) b c")

            # to list
            encoder_in_tokens_campre = [cam_features[v] for v in range(n_cam)]
            encoder_in_pos_embed_campre = [cam_pos_embed[v] for v in range(n_cam)]
        else:
            encoder_in_tokens_campre, encoder_in_pos_embed_campre = [], []

        if self.config.use_depth and OBS_DEPTH in batch:
            depths = torch.cat(batch[OBS_DEPTH], dim=0)  # (n_cam*B, 1, H, W)
            depth_features = self.depth_backbone(depths)["feature_map"]
            pos_depth_input = einops.rearrange(depth_features, " (v b) c h w -> v b c h w", v=n_cam, b=batch_size)
            depth_pos_embed = torch.cat([self.encoder_depth_feat_pos_embed(pos_depth_input[i]).to(dtype=depth_features.dtype) for i in range(n_cam)], dim=0)
            depth_features = self.encoder_depth_feat_input_proj(depth_features)

            depth_features = depth_features.view(n_cam, batch_size, depth_features.size(1), depth_features.size(2), depth_features.size(3))
            depth_pos_embed = depth_pos_embed.view(n_cam, 1, depth_pos_embed.size(1), depth_pos_embed.size(2), depth_pos_embed.size(3))

            depth_features = einops.rearrange(depth_features, "v b c h w -> v (h w) b c")
            depth_pos_embed = einops.rearrange(depth_pos_embed, "v b c h w -> v (h w) b c")

            encoder_in_tokens_depthpre = [depth_features[v] for v in range(n_cam)]
            encoder_in_pos_embed_depthpre = [depth_pos_embed[v] for v in range(n_cam)]
        else:
            encoder_in_tokens_depthpre, encoder_in_pos_embed_depthpre = [], []

        if self.config.use_depth and self.config.depth_features:
            fused_rgb_list, fused_depth_list = self.cross_modal_fusion(
                encoder_in_tokens_campre, encoder_in_tokens_depthpre
            )
            for rgb_feat, depth_feat in zip(fused_rgb_list, fused_depth_list):
                # The fused features are then dimensionality reduced using a linear layer
                fused_feat = torch.cat([rgb_feat, depth_feat], dim=-1)  # (H*W, B, 2C)
                fused_feat = self.cross_modal_fusion_proj(fused_feat)  # (H*W, B, C)
                encoder_in_tokens.extend(list(fused_feat))

            # fused_rgb_pos_list, fused_depth_pos_list = self.cross_modal_fusion(
            #     encoder_in_pos_embed_campre, encoder_in_pos_embed_depthpre
            # )
            # encoder_in_pos_embed.extend(fused_rgb_pos_list)
            # encoder_in_pos_embed.extend(fused_depth_pos_list)
            # print("encoder_in_pos_embed_campre: ", [v.shape for v in encoder_in_pos_embed_campre])
            for v in range(n_cam):
                encoder_in_pos_embed.extend(list(encoder_in_pos_embed_campre[v]))
            # encoder_in_pos_embed.extend(encoder_in_pos_embed_depthpre)
        else:
            for v in range(n_cam):
                encoder_in_tokens.extend(list(encoder_in_tokens_campre[v]))
                encoder_in_pos_embed.extend(list(encoder_in_pos_embed_campre[v]))
        # Stack all tokens along the sequence dimension.
        # print("encoder token shape: ", [t.shape for t in encoder_in_tokens])
        encoder_in_tokens = torch.stack(encoder_in_tokens, axis=0)
        encoder_in_pos_embed = torch.stack(encoder_in_pos_embed, axis=0)

        # Forward pass through the transformer modules.
        encoder_out = self.encoder(encoder_in_tokens, pos_embed=encoder_in_pos_embed)
        # TODO(rcadene, alexander-soare): remove call to `device` ; precompute and use buffer
        decoder_in = torch.zeros(
            (self.config.chunk_size, batch_size, self.config.dim_model),
            dtype=encoder_in_pos_embed.dtype,
            device=encoder_in_pos_embed.device,
        )
        decoder_out = self.decoder(
            decoder_in,
            encoder_out,
            encoder_pos_embed=encoder_in_pos_embed,
            decoder_pos_embed=self.decoder_pos_embed.weight.unsqueeze(1),
        )

        # Move back to (B, S, C).
        decoder_out = decoder_out.transpose(0, 1)

        actions = self.action_head(decoder_out)

        return actions, (mu, log_sigma_x2)
    


        
