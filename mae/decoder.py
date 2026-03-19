"""
MAE Decoder
===========
Lightweight decoder cho Masked Autoencoder.
- Nhận encoded visible patches + mask tokens
- Unshuffle về thứ tự gốc
- Transformer blocks để reconstruct pixel values
- Prediction head: project về patch_size^2 * 3
"""

import torch
import torch.nn as nn
import numpy as np

from .pos_embed import get_2d_sincos_pos_embed
from .transformer import Block


class MAEDecoder(nn.Module):
    """
    MAE Decoder: Project → Unshuffle → Transformer → Predict pixels.
    
    Args:
        num_patches: Số patches (196 hoặc 256)
        patch_size: Kích thước patch (16 hoặc 14)
        in_chans: Số kênh ảnh (3)
        encoder_embed_dim: Chiều embedding encoder (768/1024/1280)
        decoder_embed_dim: Chiều embedding decoder (512)
        decoder_depth: Số transformer blocks decoder (8)
        decoder_num_heads: Số attention heads decoder (16)
        mlp_ratio: Tỉ lệ expansion MLP (4.0)
        norm_layer: Factory function cho LayerNorm
    """
    
    def __init__(
        self,
        num_patches=196,
        patch_size=16,
        in_chans=3,
        encoder_embed_dim=768,
        decoder_embed_dim=512,
        decoder_depth=8,
        decoder_num_heads=16,
        mlp_ratio=4.,
        norm_layer=None,
    ):
        super().__init__()
        norm_layer = norm_layer or (lambda d: nn.LayerNorm(d, eps=1e-6))
        
        self.decoder_embed = nn.Linear(encoder_embed_dim, decoder_embed_dim, bias=True)
        
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        
        self.decoder_pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + 1, decoder_embed_dim),
            requires_grad=False
        )
        
        self.decoder_blocks = nn.ModuleList([
            Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, norm_layer=norm_layer)
            for _ in range(decoder_depth)
        ])
        self.decoder_norm = norm_layer(decoder_embed_dim)
        
        self.decoder_pred = nn.Linear(
            decoder_embed_dim, patch_size ** 2 * in_chans, bias=True
        )
        
        self.num_patches = num_patches
        self.decoder_embed_dim = decoder_embed_dim
        self.initialize_weights()
    
    def initialize_weights(self):
        dec_pos_embed = get_2d_sincos_pos_embed(
            self.decoder_pos_embed.shape[-1],
            int(self.num_patches ** 0.5),
            cls_token=True
        )
        self.decoder_pos_embed.data.copy_(
            torch.from_numpy(dec_pos_embed).float().unsqueeze(0)
        )
        
        nn.init.normal_(self.mask_token, std=0.02)
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
    
    def forward(self, x, ids_restore):
        """
        Decode full sequence with mask tokens.
        
        Args:
            x: (B, L_visible+1, encoder_embed_dim) — encoder output (CLS + visible)
            ids_restore: (B, L) — indices để khôi phục thứ tự gốc
        
        Returns:
            pred: (B, L, patch_size^2 * 3) — predicted pixel values (no CLS)
        """
        x = self.decoder_embed(x)
        
        mask_tokens = self.mask_token.repeat(
            x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1
        )
        
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)
        x_ = torch.gather(
            x_, dim=1,
            index=ids_restore.unsqueeze(-1).expand(-1, -1, x.shape[2])
        )
        x = torch.cat([x[:, :1, :], x_], dim=1)
        
        x = x + self.decoder_pos_embed
        
        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)
        
        x = self.decoder_pred(x)
        x = x[:, 1:, :]
        
        return x
