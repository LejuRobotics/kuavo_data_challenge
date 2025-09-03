# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# GLIDE: https://github.com/openai/glide-text2im
# MAE: https://github.com/facebookresearch/mae/blob/main/models_mae.py
# --------------------------------------------------------

import torch
import torch.nn as nn
import numpy as np
import math
from timm.models.vision_transformer import Attention, Mlp

def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

#################################################################################
#               Embedding Layers for Timesteps and Condition                     #
#################################################################################

class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb

class ConditionEmbedder(nn.Module):
    """
    Embeds continuous condition vectors into vector representations.
    """
    def __init__(self, input_dim, hidden_size):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )

    def forward(self, y):
        return self.mlp(y)

#################################################################################
#                                 Core DiT Model                                #
#################################################################################

class DiTBlock(nn.Module):
    """
    A DiT block with adaptive layer norm zero (adaLN-Zero) conditioning.
    """
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, **block_kwargs):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True, **block_kwargs)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
        x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x

class FinalLayer(nn.Module):
    """
    The final layer of DiT.
    """
    def __init__(self, hidden_size, output_dim):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, output_dim, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x

class DiT(nn.Module):
    """
    Diffusion model with a Transformer backbone for sequence inputs.
    """
    def __init__(
        self,
        input_length: int,      # Sequence length T
        input_dim: int,         # Input feature dimension d1
        cond_dim: int,          # Condition dimension d2
        hidden_size: int = 1152,
        depth: int = 12,
        num_heads: int = 16,
        mlp_ratio: float = 4.0,
    ):
        super().__init__()
        self.input_length = input_length
        self.input_dim = input_dim
        self.cond_dim = cond_dim
        
        # Input projection (d1 to D)
        self.input_proj = nn.Linear(input_dim, hidden_size)
        
        # Positional embedding (learnable)
        self.pos_embed = nn.Parameter(torch.zeros(1, input_length, hidden_size))
        
        # Embedders for timestep and condition
        self.t_embedder = TimestepEmbedder(hidden_size)
        self.y_embedder = ConditionEmbedder(cond_dim, hidden_size)
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            DiTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio) for _ in range(depth)
        ])
        
        # Final output layer (D to d1)
        self.final_layer = FinalLayer(hidden_size, input_dim)
        self.initialize_weights()

    def initialize_weights(self):
        # Initialize input projection
        nn.init.xavier_uniform_(self.input_proj.weight)
        if self.input_proj.bias is not None:
            nn.init.constant_(self.input_proj.bias, 0)
        
        # Initialize positional embedding
        nn.init.normal_(self.pos_embed, std=0.02)
        
        # Initialize condition embedder
        for layer in self.y_embedder.mlp:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0)
        
        # Initialize timestep embedder
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)
        
        # Zero-out adaLN modulation layers
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)
        
        # Zero-out final layer
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def forward(self,x: torch.Tensor, timestep: torch.Tensor | int, global_cond=None):
        """
        Forward pass of DiT for sequences.
        x: (N, T, d1) tensor of input sequences
        t: (N,) tensor of diffusion timesteps
        y: (N, d2) tensor of condition vectors
        """
        t = timestep
        y = global_cond
        # Project input to hidden size D
        x = self.input_proj(x)  # (N, T, D)
        
        # Add positional embedding
        x = x + self.pos_embed  # (N, T, D)
        
        # Embed timestep and condition
        t_emb = self.t_embedder(t)    # (N, D)
        y_emb = self.y_embedder(y)    # (N, D)
        c = t_emb + y_emb             # (N, D)
        
        # Process through transformer blocks
        for block in self.blocks:
            x = block(x, c)  # (N, T, D)
        
        # Project back to input dimension d1
        x = self.final_layer(x, c)  # (N, T, d1)
        return x

    def forward_with_cfg(self, x, t, y, cfg_scale):
        """
        Forward pass with classifier-free guidance.
        """
        half = x[: len(x) // 2]
        combined = torch.cat([half, half], dim=0)
        model_out = self.forward(combined, t, y)
        cond_out, uncond_out = torch.split(model_out, len(model_out) // 2, dim=0)
        return uncond_out + cfg_scale * (cond_out - uncond_out)

#################################################################################
#                                   DiT Configs                                 #
#################################################################################

def DiT_XL(input_length, input_dim, cond_dim, **kwargs):
    return DiT(input_length, input_dim, cond_dim, 
               hidden_size=1152, depth=28, num_heads=16, **kwargs)

def DiT_L(input_length, input_dim, cond_dim, **kwargs):
    return DiT(input_length, input_dim, cond_dim, 
               hidden_size=1024, depth=24, num_heads=16, **kwargs)

def DiT_B(input_length, input_dim, cond_dim, **kwargs):
    return DiT(input_length, input_dim, cond_dim, 
               hidden_size=768, depth=12, num_heads=12, **kwargs)

def DiT_S(input_length, input_dim, cond_dim, **kwargs):
    return DiT(input_length, input_dim, cond_dim, 
               hidden_size=384, depth=12, num_heads=6, **kwargs)

DiT_models = {
    'DiT-XL': DiT_XL, # emb_dim 1152, layer 28, head 16
    'DiT-L': DiT_L, # emb_dim 1024, layer 24, head 16
    'DiT-B': DiT_B, # emb_dim 768, layer 12, head 12
    'DiT-S': DiT_S, # emb_dim 384, layer 12, head 6
}

if __name__ == "__main__":
    net = DiT_S(16,2,260)
    action = torch.rand(5,16,2)
    t = torch.randint(0,100,(5,))
    cond = torch.rand(5,260)
    out = net(action, t, cond)
    print(out.shape)
