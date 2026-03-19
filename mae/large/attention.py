"""
Multi-Head Self-Attention cho MAE ViT-Large
=============================================
- QKV projection: nn.Linear(dim, dim*3, bias=qkv_bias)
  → qkv_bias=True (khớp checkpoint blocks.*.attn.qkv.bias)
- Scaled Dot-Product: softmax(QK^T / √d_k) × V
- Output projection: nn.Linear(dim, dim)
"""

import torch
import torch.nn as nn


class Attention(nn.Module):
    """Multi-head self-attention module."""
    
    def __init__(self, dim, num_heads, qkv_bias=True, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5  # 1/√d_k
        
        # QKV projection — bias=True để khớp checkpoint
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        
        # Output projection
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
    
    def forward(self, x):
        B, N, C = x.shape
        
        # (B, N, 3*C) → (B, N, 3, num_heads, head_dim) → (3, B, num_heads, N, head_dim)
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)  # mỗi: (B, num_heads, N, head_dim)
        
        # Scaled dot-product attention
        attn = (q @ k.transpose(-2, -1)) * self.scale  # (B, num_heads, N, N)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        
        # Weighted sum
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        
        # Output projection
        x = self.proj(x)
        x = self.proj_drop(x)
        return x
