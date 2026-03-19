"""
Transformer Block
================
- Pre-norm architecture (LayerNorm trước Attention và MLP)
- Residual connections
- MLP với GELU activation
"""

import torch
import torch.nn as nn
from .attention import Attention

class Mlp(nn.Module):
    """MLP block với expansion ratio."""
    
    def __init__(self, in_features, hidden_features=None, out_features=None, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features * 4
        
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()  # Smoother than ReLU
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
    """Transformer encoder block."""
    
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
    
if __name__ == "__main__":
    block = Block(dim=768, num_heads=12)
    x = torch.randn(2, 197, 768)  # (B, N, C)
    out = block(x)
    assert out.shape == (2, 197, 768), f"Expected shape (2, 197, 768), but got {out.shape}"
    print("Transformer Block output shape:", out.shape)