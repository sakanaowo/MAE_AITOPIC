"""
Transformer Block cho MAE ViT-Large
=====================================
Pre-norm architecture:
- LayerNorm → Attention → Residual
- LayerNorm → MLP → Residual

Key naming khớp checkpoint:
- norm1, attn (qkv, proj)
- norm2, mlp (fc1, fc2)
"""

import torch.nn as nn
from .attention import Attention


class Mlp(nn.Module):
    """MLP block: Linear → GELU → Linear."""
    
    def __init__(self, in_features, hidden_features=None, out_features=None, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features * 4
        
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Block(nn.Module):
    """
    Transformer encoder block.
    
    Tên attribute khớp checkpoint:
    - self.norm1 → blocks.{i}.norm1
    - self.attn  → blocks.{i}.attn (qkv, proj)
    - self.norm2 → blocks.{i}.norm2
    - self.mlp   → blocks.{i}.mlp (fc1, fc2)
    
    Note: LayerNorm eps=1e-6 để khớp official implementation.
    Official: partial(nn.LayerNorm, eps=1e-6)
    """
    
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0.,
                 norm_layer=None):
        super().__init__()
        norm_layer = norm_layer or (lambda d: nn.LayerNorm(d, eps=1e-6))
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads, qkv_bias, attn_drop, drop)
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(dim, int(dim * mlp_ratio), drop=drop)
    
    def forward(self, x):
        # Pre-norm + Residual
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x
