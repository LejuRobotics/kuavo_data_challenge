# multimodal_diffusion_wrapper.py
import math
from typing import Optional, Dict, Any
import einops
import torch
import torch.nn as nn
from torch import Tensor
import torchvision

from lerobot.utils.constants import OBS_ENV_STATE, OBS_IMAGES, OBS_STATE
from kuavo_train.wrapper.policy.diffusion.DiffusionConfigWrapper import CustomDiffusionConfigWrapper
from lerobot.policies.utils import get_device_from_parameters, get_dtype_from_parameters, get_output_shape
from lerobot.policies.diffusion.modeling_diffusion import (
    _make_noise_scheduler,
    _replace_submodules,
    DiffusionConditionalUnet1d,
    SpatialSoftmax,
    DiffusionModel,
)
from kuavo_train.wrapper.policy.diffusion.transformer_diffusion import TransformerForDiffusion

# diffusers scheduler classes (factory expects these names)
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers import StableDiffusion3Pipeline
OBS_DEPTH = "observation.depth"


# ---------------------------
# Helper: safe scheduler factory
# ---------------------------
def _make_noise_scheduler_factory(name: str, **kwargs: Dict[str, Any]):
    if name == "DDPM":
        return DDPMScheduler(**kwargs)
    elif name == "DDIM":
        return DDIMScheduler(**kwargs)
    else:
        raise ValueError(f"Unsupported noise scheduler type {name}")


# ---------------------------
# Feature encoders (state)
# ---------------------------
class FeatureEncoder(nn.Module):
    """Simple MLP encoder for state features. Accepts [B, D] or [B, T, D]."""
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.ReLU(inplace=False),
            nn.Linear(out_dim, out_dim),
        )

    def forward(self, x: Tensor) -> Optional[Tensor]:
        if x is None:
            return None
        if x.dim() == 2:
            return self.net(x)  # (B, out_dim)
        elif x.dim() == 3:
            B, T, D = x.shape
            x_flat = x.view(B * T, D)
            out = self.net(x_flat).view(B, T, -1)
            return out  # (B, T, out_dim)
        else:
            raise ValueError("FeatureEncoder expects 2D or 3D tensor.")

class ResnetRgbEncoder(nn.Module):
    def __init__(self, config: CustomDiffusionConfigWrapper):
        super().__init__()
        backbone_model = getattr(torchvision.models, config.vision_backbone)(weights=config.pretrained_backbone_weights)
        self.backbone = nn.Sequential(*(list(backbone_model.children())[:-2]))
        if config.use_group_norm:
            if config.pretrained_backbone_weights:
                raise ValueError("Can't replace BatchNorm in pretrained model.")
            self.backbone = _replace_submodules(
                root_module=self.backbone,
                predicate=lambda x: isinstance(x, nn.BatchNorm2d),
                func=lambda x: nn.GroupNorm(num_groups=max(1, x.num_features // 16), num_channels=x.num_features),
            )
        images_shape = next(iter(config.image_features.values())).shape
        if config.resize_shape is not None:
            dummy_shape_h_w = config.resize_shape
        elif config.crop_shape is not None:
            if isinstance(list(config.crop_shape)[0], (list, tuple)):
                (x_start, x_end), (y_start, y_end) = config.crop_shape
                dummy_shape_h_w = (x_end - x_start, y_end - y_start)
            else:
                dummy_shape_h_w = config.crop_shape
        else:
            dummy_shape_h_w = images_shape[1:]
        dummy_shape = (1, images_shape[0], *dummy_shape_h_w)
        feature_map_shape = get_output_shape(self.backbone, dummy_shape)[1:]
        self.pool = SpatialSoftmax(feature_map_shape, num_kp=config.spatial_softmax_num_keypoints)
        self.feature_dim = config.spatial_softmax_num_keypoints * 2
        self.out = nn.Linear(self.feature_dim, self.feature_dim)
        self.relu = nn.ReLU()

    def forward(self, x: Tensor) -> Tensor:
        x = self.backbone(x)
        x = torch.flatten(self.pool(x), start_dim=1)
        x = self.relu(self.out(x))
        return x  # (B, feature_dim)


class ResnetDepthEncoder(nn.Module):
    def __init__(self, config: CustomDiffusionConfigWrapper):
        super().__init__()
        backbone_model = getattr(torchvision.models, config.depth_backbone)(weights=config.pretrained_backbone_weights)
        modules = list(backbone_model.children())[:-2]
        if isinstance(modules[0], nn.Conv2d):
            old_conv = modules[0]
            modules[0] = nn.Conv2d(
                in_channels=1,
                out_channels=old_conv.out_channels,
                kernel_size=old_conv.kernel_size,
                stride=old_conv.stride,
                padding=old_conv.padding,
                bias=old_conv.bias is not None
            )
            with torch.no_grad():
                modules[0].weight = nn.Parameter(old_conv.weight.mean(dim=1, keepdim=True))
        self.backbone = nn.Sequential(*modules)
        if config.use_group_norm:
            if config.pretrained_backbone_weights:
                raise ValueError("Can't replace BatchNorm in pretrained model.")
            self.backbone = _replace_submodules(
                root_module=self.backbone,
                predicate=lambda x: isinstance(x, nn.BatchNorm2d),
                func=lambda x: nn.GroupNorm(num_groups=max(1, x.num_features // 16), num_channels=x.num_features),
            )
        images_shape = next(iter(config.depth_features.values())).shape
        
        if config.resize_shape is not None:
            dummy_shape_h_w = config.resize_shape
        elif config.crop_shape is not None:
            if isinstance(list(config.crop_shape)[0], (list, tuple)):
                (x_start, x_end), (y_start, y_end) = config.crop_shape
                dummy_shape_h_w = (x_end - x_start, y_end - y_start)
            else:
                dummy_shape_h_w = config.crop_shape
        else:
            dummy_shape_h_w = images_shape[1:]
        dummy_shape = (1, 1, *dummy_shape_h_w)

        feature_map_shape = get_output_shape(self.backbone, dummy_shape)[1:]
        self.pool = SpatialSoftmax(feature_map_shape, num_kp=config.spatial_softmax_num_keypoints)
        self.feature_dim = config.spatial_softmax_num_keypoints * 2
        self.out = nn.Linear(self.feature_dim, self.feature_dim)
        self.relu = nn.ReLU()

    def forward(self, x: Tensor) -> Tensor:
        x = self.backbone(x)
        x = torch.flatten(self.pool(x), start_dim=1)
        x = self.relu(self.out(x))
        return x




class DiffusionRgbEncoder(nn.Module):
    def __init__(self, config: CustomDiffusionConfigWrapper):
        super().__init__()
        self.config = config
        if "resnet" in config.vision_backbone:
            self.model = ResnetRgbEncoder(config)
        else:
            raise ValueError(f"Unknown vision backbone: {config.vision_backbone}")
        self.feature_dim = self.model.feature_dim
    def forward(self, x: Tensor) -> Tensor:
        return self.model(x)


class DiffusionDepthEncoder(nn.Module):
    def __init__(self, config: CustomDiffusionConfigWrapper):
        super().__init__()
        self.config = config
        if "resnet" in config.depth_backbone:
            self.model = ResnetDepthEncoder(config)
        else:
            raise ValueError(f"Unknown depth backbone: {config.depth_backbone}")
        self.feature_dim = self.model.feature_dim
    def forward(self, x: Tensor) -> Tensor:
        return self.model(x)


# ---------------------------
# State-guided fusion block (no discrete logic here)
# ---------------------------
class StateGuidedFusionBlock(nn.Module):
    """
    Projects modality features to a shared hidden dim and performs cross-attention.
    Inputs:
      - vis_feat: (B, N_v, vis_dim)
      - dep_feat: (B, N_d, dep_dim) or None
      - state_feat: (B, state_dim) or None  # ALREADY encoded / discretized in wrapper if required
    """
    def __init__(self, vis_dim: int, dep_dim: Optional[int], state_dim: Optional[int],
                 hidden_dim: int = 256, num_heads: int = 8):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.proj_vis = nn.Linear(vis_dim, hidden_dim)
        self.use_depth = dep_dim is not None
        if self.use_depth:
            self.proj_dep = nn.Linear(dep_dim, hidden_dim)

        self.use_state = state_dim is not None
        if self.use_state:
            # state_feat is expected already to be final size (wrapper ensures this)
            self.state_proj = nn.Linear(state_dim, hidden_dim)

        self.cross_attn = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads, batch_first=True)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=False),
            nn.Linear(hidden_dim, hidden_dim)
        )

    def forward(self, vis_feat: Tensor, dep_feat: Optional[Tensor], state_feat: Optional[Tensor]) -> Tensor:
        # vis_feat: (B*s, N_v, vis_dim)
        B = vis_feat.shape[0]
        vis_proj = self.proj_vis(vis_feat)  # (B*s, N_v, hidden)

        if self.use_depth and dep_feat is not None:
            dep_proj = self.proj_dep(dep_feat)  # (B*s, N_d, hidden)
            kv = torch.cat([vis_proj, dep_proj], dim=1)  # (B*s, N_v+N_d, hidden)
        else:
            kv = vis_proj  # (B*s, N_v, hidden)

        if self.use_state and state_feat is not None:
            # state_feat is (B*s, state_dim)
            state_emb = self.state_proj(state_feat)  # (B*s, hidden)
            query = state_emb.unsqueeze(1)  # (B*s, 1, hidden)
        else:
            query = vis_proj.mean(dim=1, keepdim=True)  # (B*s, 1, hidden)

        fused, _ = self.cross_attn(query=query, key=kv, value=kv)
        fused = self.mlp(fused).squeeze(1)  # (B*s, hidden)
        return fused


# ---------------------------
# Main wrapper: integrate encoders, fusion, and diffusion model
# ---------------------------
class CustomDiffusionModelWrapper(DiffusionModel):
    def __init__(self, config: CustomDiffusionConfigWrapper):
        # ensure parent init runs with safe backbone
        orig_vis = config.vision_backbone
        config.vision_backbone = "resnet18"
        orig_noise_scheduler = config.noise_scheduler_type
        config.noise_scheduler_type = "DDPM"
        super().__init__(config)
        config.vision_backbone = orig_vis
        config.noise_scheduler_type = orig_noise_scheduler

        self.config = config
        global_cond_dim = 0

        # ---- state encoder in WRAPPER (only place for discrete or mlp logic) ----
        self.state_encoder = None
        final_state_dim = None
        if getattr(self.config, "robot_state_feature", None) is not None:
            state_dim = self.config.robot_state_feature.shape[0]
            if getattr(self.config, "use_state_encoder", False):
                out_dim = getattr(self.config, "state_feature_dim", 128)
                self.state_encoder = FeatureEncoder(state_dim, out_dim)
                final_state_dim = out_dim
                global_cond_dim += final_state_dim
            else:
                final_state_dim = state_dim
                global_cond_dim += final_state_dim

        # ---- RGB encoders ----
        self.rgb_feat_dim = 0
        if getattr(self.config, "image_features", None):
            num_images = len(self.config.image_features)
            if getattr(self.config, "use_separate_rgb_encoder_per_camera", False):
                encs = [DiffusionRgbEncoder(config) for _ in range(num_images)]
                self.rgb_encoder = nn.ModuleList(encs)
                feat_dim = encs[0].feature_dim
            else:
                self.rgb_encoder = DiffusionRgbEncoder(config)
                feat_dim = self.rgb_encoder.feature_dim
            self.rgb_feat_dim = feat_dim * num_images
            global_cond_dim += self.rgb_feat_dim
            self.rgb_attn_layer = nn.MultiheadAttention(embed_dim=feat_dim, num_heads=getattr(self.config, "rgb_attn_heads", 8), batch_first=True)

        # ---- Depth encoders (optional) ----
        self.depth_feat_dim = 0
        if getattr(self.config, "use_depth", False) and getattr(self.config, "depth_features", None):
            num_depth = len(self.config.depth_features)
            if getattr(self.config, "use_separate_depth_encoder_per_camera", False):
                encs = [DiffusionDepthEncoder(config) for _ in range(num_depth)]
                self.depth_encoder = nn.ModuleList(encs)
                feat_dim = encs[0].feature_dim
            else:
                self.depth_encoder = DiffusionDepthEncoder(config)
                feat_dim = self.depth_encoder.feature_dim
            self.depth_feat_dim = feat_dim * num_depth
            global_cond_dim += self.depth_feat_dim
            self.depth_attn_layer = nn.MultiheadAttention(embed_dim=feat_dim, num_heads=getattr(self.config, "depth_attn_heads", 8), batch_first=True)

            # RGB <-> Depth cross-attn modules
            self.multimodalfuse = nn.ModuleDict({
                "rgb_q": nn.MultiheadAttention(embed_dim=feat_dim, num_heads=getattr(self.config, "multimodal_heads", 8), batch_first=True),
                "depth_q": nn.MultiheadAttention(embed_dim=feat_dim, num_heads=getattr(self.config, "multimodal_heads", 8), batch_first=True),
            })

        # ---- state-guided fusion block ----
        self.fusion_hidden = getattr(self.config, "fusion_hidden_dim", 256)
        self.state_guided = None
        if getattr(self.config, "state_fuse", False):
            vis_dim_for_fusion = (self.rgb_attn_layer.embed_dim if hasattr(self, "rgb_attn_layer") else self.rgb_feat_dim)
            dep_dim_for_fusion = (self.depth_attn_layer.embed_dim if hasattr(self, "depth_attn_layer") else None)
            state_dim_for_fusion = final_state_dim
            self.state_guided = StateGuidedFusionBlock(
                vis_dim=vis_dim_for_fusion,
                dep_dim=dep_dim_for_fusion,
                state_dim=state_dim_for_fusion,
                hidden_dim=self.fusion_hidden,
                num_heads=getattr(self.config, "fusion_heads", 8)
            )
            global_cond_dim += self.fusion_hidden

        # ---- env state ----
        if getattr(self.config, "env_state_feature", None) is not None:
            global_cond_dim += self.config.env_state_feature.shape[0]

        # ---- core diffusion model ----
        if config.use_unet:
            self.unet = DiffusionConditionalUnet1d(config, global_cond_dim=global_cond_dim * config.n_obs_steps)
        elif config.use_transformer:
            self.unet = TransformerForDiffusion(
                input_dim=config.output_features["action"].shape[0],
                output_dim=config.output_features["action"].shape[0],
                horizon=config.horizon,
                n_obs_steps=config.n_obs_steps,
                cond_dim=global_cond_dim,
                n_layer=self.config.transformer_n_layer,
                n_head=self.config.transformer_n_head,
                n_emb=self.config.transformer_n_emb,
                p_drop_emb=self.config.transformer_dropout,
                p_drop_attn=self.config.transformer_dropout,
                causal_attn=False,
                time_as_cond=True,
                obs_as_cond=True,
                n_cond_layers=0,
            )
        else:
            raise ValueError("Either `use_unet` or `use_transformer` must be True in config.")

        # ---- scheduler ----
        self.noise_scheduler = _make_noise_scheduler_factory(
            config.noise_scheduler_type,
            num_train_timesteps=config.num_train_timesteps,
            beta_start=config.beta_start,
            beta_end=config.beta_end,
            beta_schedule=config.beta_schedule,
            clip_sample=config.clip_sample,
            clip_sample_range=config.clip_sample_range,
            prediction_type=config.prediction_type,
        )
        # self.noise_scheduler = _make_noise_scheduler_factory(
        #     config.noise_scheduler_type,
        #     config.scheduler_params
        # )
        self.num_inference_steps = config.num_inference_steps or self.noise_scheduler.config.num_train_timesteps

    def _prepare_global_conditioning(self, batch: Dict[str, Tensor]) -> Tensor:
        """
        Encode & fuse modalities into (B, S, cond_dim).
        Behavior:
          - if both rgb & depth: compute rgb_q & dep_q tokens (cross-attn outputs), flatten for cond features,
            AND create concat tokens (rgb_q_cat = cat(rgb_q, dep_q)) as tokens for state-guided fusion.
          - if only rgb: use rgb tokens.
          - state encoding is performed here (wrapper); state_for_fusion will be encoded/flattened and passed to fusion block.
        """
        B = batch[OBS_STATE].shape[0]
        S = batch[OBS_STATE].shape[1]  # n_obs_steps
        feats = []

        # ---------- RGB ----------
        img_features = None  # tokens shape (B*S, N_cam_tokens, feat)
        if getattr(self.config, "image_features", None):
            if getattr(self.config, "use_separate_rgb_encoder_per_camera", False):
                imgs = einops.rearrange(batch[OBS_IMAGES], "b s n ... -> n (b s) ...")
                enc_outs = [enc(im) for enc, im in zip(self.rgb_encoder, imgs)]
                img_cat = torch.cat(enc_outs)  # (n * B*s, feat)
                img_features = einops.rearrange(img_cat, "(n b s) f -> (b s) n f", b=B, s=S)
            else:
                imgs = einops.rearrange(batch[OBS_IMAGES], "b s n ... -> (b s n) ...")
                out = self.rgb_encoder(imgs)  # (b*s*n, feat)
                img_features = einops.rearrange(out, "(b s n) f -> (b s) n f", b=B, s=S)
            # self-attn over tokens
            img_features = self.rgb_attn_layer(img_features, img_features, img_features)[0]  # (b*s, n, feat)

        # ---------- Depth (optional) ----------
        depth_features = None
        if getattr(self.config, "use_depth", False) and getattr(self.config, "depth_features", None) and (OBS_DEPTH in batch):
            if getattr(self.config, "use_separate_depth_encoder_per_camera", False):
                depths = einops.rearrange(batch[OBS_DEPTH], "b s n ... -> n (b s) ...")
                enc_outs = [enc(d) for enc, d in zip(self.depth_encoder, depths)]
                dep_cat = torch.cat(enc_outs)
                depth_features = einops.rearrange(dep_cat, "(n b s) f -> (b s) n f", b=B, s=S)
            else:
                depths = einops.rearrange(batch[OBS_DEPTH], "b s n ... -> (b s n) ...")
                out = self.depth_encoder(depths)
                depth_features = einops.rearrange(out, "(b s n) f -> (b s) n f", b=B, s=S)
            depth_features = self.depth_attn_layer(depth_features, depth_features, depth_features)[0]  # (b*s, n, feat)

        # ---------- RGB <-> Depth fusion (if both exist) ----------
        # Keep both token forms (rgb_q_tokens, dep_q_tokens) for state-guided fusion (we will concat them).
        rgb_q_tokens = None
        dep_q_tokens = None
        if (img_features is not None) and (depth_features is not None) and hasattr(self, "multimodalfuse"):
            rgb_q_tokens = self.multimodalfuse["rgb_q"](img_features, depth_features, depth_features)[0]  # (b*s, n, feat)
            dep_q_tokens = self.multimodalfuse["depth_q"](depth_features, img_features, img_features)[0]  # (b*s, n, feat)
            # For global_cond feats we flatten (B, S, n*feat)
            rgb_q_flat = einops.rearrange(rgb_q_tokens, "(b s) n f -> b s (n f)", b=B, s=S)
            dep_q_flat = einops.rearrange(dep_q_tokens, "(b s) n f -> b s (n f)", b=B, s=S)
            feats.extend([rgb_q_flat, dep_q_flat])
        elif img_features is not None:
            # only rgb available
            feats.append(einops.rearrange(img_features, "(b s) n f -> b s (n f)", b=B, s=S))
        elif depth_features is not None:
            feats.append(einops.rearrange(depth_features, "(b s) n f -> b s (n f)", b=B, s=S))

        # ---------- State encoding (WRAPPER does this) ----------
        state_tensor = None
        if getattr(self.config, "robot_state_feature", None) is not None:
            state_tensor = batch[OBS_STATE]  # (B, S, state_dim)
            if self.state_encoder is not None:
                # encoder may accept (B, S, D) and returns (B, S, out_dim)
                state_emb = self.state_encoder(state_tensor)  # (B, S, final_state_dim)
                feats.append(state_emb)
            else:
                feats.append(state_tensor)

        # ---------- Env state ----------
        if getattr(self.config, "env_state_feature", None) is not None:
            feats.append(batch[OBS_ENV_STATE])

        # ---------- State-guided fusion: run on per-(b*s) samples ----------
        if getattr(self, "state_guided", None) is not None:
            # prepare tokens for fusion block
            # choose tokens: if rgb_q_tokens & dep_q_tokens exist -> concat them (combined tokens),
            # else fallback to img_features (or depth_features if only depth exists)
            if (rgb_q_tokens is not None) and (dep_q_tokens is not None):
                # concat tokens along sequence dim to give more information to fusion
                # vis_tokens_for_fusion = torch.cat([rgb_q_tokens, dep_q_tokens], dim=1)  # (b*s, n_r + n_d, feat)
                # vis_tokens_for_fusion = rgb_q_tokens
                # dep_tokens_for_fusion = dep_q_tokens
                vis_tokens_for_fusion = img_features
                dep_tokens_for_fusion = depth_features
                # print("~~~~~~~~~~~~~~~~~~~~~~~~~~use rgb_q and dep_q~~~~~~~~~~~~~~~~~~~~~~~~~~")
            elif img_features is not None:
                vis_tokens_for_fusion = img_features  # (b*s, n, feat)
                dep_tokens_for_fusion = None
            elif depth_features is not None:
                vis_tokens_for_fusion = depth_features
                dep_tokens_for_fusion = None
            else:
                vis_tokens_for_fusion = None
                dep_tokens_for_fusion = None

            # prepare state for fusion: should be (B*s, final_state_dim) or None
            if state_tensor is not None:
                if self.state_encoder is not None:
                    state_for_fusion = state_emb.view(B * S, -1)  # (B*S, final_state_dim)
                else:
                    state_for_fusion = state_tensor.view(B * S, -1)  # raw
            else:
                state_for_fusion = None

            # call fusion block if we have visual tokens
            if vis_tokens_for_fusion is not None:
                fused_vec = self.state_guided(vis_tokens_for_fusion, dep_tokens_for_fusion, state_for_fusion)  # (B*s, fusion_hidden)
                fused_vec = einops.rearrange(fused_vec, "(b s) f -> b s f", b=B, s=S)  # (B, S, fusion_hidden)
                feats.append(fused_vec)

        # Final concat -> (B, S, cond_dim)
        if len(feats) == 0:
            return torch.zeros((B, S, 0), device=next(self.parameters()).device)
        if self.config.use_unet:
            return torch.cat(feats, dim=-1).flatten(start_dim=1)
        else:
            return torch.cat(feats, dim=-1)

    # ---------------------------
    # Inference sampling
    # ---------------------------
    def conditional_sample(self, batch_size: int, global_cond: Optional[Tensor] = None, generator=None, noise: Tensor | None = None) -> Tensor:
        device = get_device_from_parameters(self)
        dtype = get_dtype_from_parameters(self)

        sample = (
                    noise
                    if noise is not None
                    else torch.randn(
                        size=(batch_size, self.config.horizon, self.config.action_feature.shape[0]),
                        dtype=dtype,
                        device=device,
                        generator=generator,
                    )
                )
        self.noise_scheduler.set_timesteps(self.num_inference_steps)

        for t in self.noise_scheduler.timesteps:
            model_output = self.unet(
                sample,
                torch.full((batch_size,), t, dtype=torch.long, device=device),
                global_cond=global_cond,
            )
            # pass eta if scheduler supports it (DDIM uses eta)
            if "eta" in self.noise_scheduler.step.__code__.co_varnames:
                step_out = self.noise_scheduler.step(model_output, t, sample, eta=getattr(self.config, "ddim_eta", 0.0), generator=generator)
            else:
                step_out = self.noise_scheduler.step(model_output, t, sample, generator=generator)
            sample = getattr(step_out, "prev_sample", step_out)

        return sample
