"""
Multi-head Attention module for Masked Autoencoders (MAE).
==============================
- Query, Key, Value projections
- Scaled Dot-Product Attention: softmax(QK^T/sqrt(d_k))V
- Multi-head: multiple attention heads concatenated
==============================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class Attention(nn.Module):
    def __init__(self,dim, num_heads, qkv_bias=True, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5 # Scaling factor for dot-product attention
        
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias) 
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        
    def forward(self, x):
        B,N,C = x.shape  # Batch size, sequence length, embedding dimension

        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv=qkv.permute(2, 0, 3, 1, 4)  # (3, B, num_heads, N, head_dim)
        q, k, v = qkv.unbind(0)  # each: (B, num_heads, N, head_dim)
        
        attn = (q @ k.transpose(-2, -1)) * self.scale  # (B, num_heads, N, N)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)  # (B, N, C)

        x = self.proj(x)
        x = self.proj_drop(x)
        return x