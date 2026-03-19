"""
MAE Encoder
===========
ViT encoder cho Masked Autoencoder.
- Patch embedding + positional embedding (sin-cos, fixed)
- Random masking (75% default)
- Chỉ encode visible patches (hiệu quả, tránh distribution mismatch)
- Transformer blocks + LayerNorm
"""

import torch
import torch.nn as nn
import numpy as np

from .patch_embed import PatchEmbed
from .pos_embed import get_2d_sincos_pos_embed
from .masking import random_masking
from .transformer import Block


class MAEEncoder(nn.Module):
    """
    MAE Encoder: Patch embed → Mask → Transformer trên visible patches.
    
    Args:
        img_size: Kích thước ảnh đầu vào (224)
        patch_size: Kích thước patch (16 hoặc 14)
        in_chans: Số kênh ảnh (3)
        embed_dim: Chiều embedding (768/1024/1280)
        depth: Số transformer blocks (12/24/32)
        num_heads: Số attention heads (12/16/16)
        mlp_ratio: Tỉ lệ expansion MLP (4.0)
        norm_layer: Factory function cho LayerNorm
    """
    
    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_chans=3,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.,
        norm_layer=None,
    ):
        super().__init__()
        norm_layer = norm_layer or (lambda d: nn.LayerNorm(d, eps=1e-6))
        
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches
        
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + 1, embed_dim),
            requires_grad=False
        )
        
        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, norm_layer=norm_layer)
            for _ in range(depth)
        ])
        self.norm = norm_layer(embed_dim)
        
        self.embed_dim = embed_dim
        self.num_patches = num_patches
        self.initialize_weights()
    
    def initialize_weights(self):
        pos_embed = get_2d_sincos_pos_embed(
            self.pos_embed.shape[-1],
            int(self.num_patches ** 0.5),
            cls_token=True
        )
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))
        
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        
        nn.init.normal_(self.cls_token, std=0.02)
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
    
    def forward(self, x, mask_ratio=0.75):
        """
        Encode visible patches only.
        
        Args:
            x: (B, 3, H, W) ảnh đầu vào
            mask_ratio: Tỉ lệ mask (0.75)
        
        Returns:
            latent: (B, L_visible+1, embed_dim) — CLS + visible patches
            mask: (B, L) — binary mask (1=masked, 0=visible)
            ids_restore: (B, L) — indices để khôi phục thứ tự gốc
        """
        x = self.patch_embed(x)
        x = x + self.pos_embed[:, 1:, :]
        
        x, mask, ids_restore = random_masking(x, mask_ratio)
        
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        
        return x, mask, ids_restore
