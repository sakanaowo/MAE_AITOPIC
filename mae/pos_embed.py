""" 
positional embedding utils
2d sinusoidal positional embedding
==============================
- each patch gets a (x, y) coordinate in the 14x14 grid (for 224x224 image with 16x16 patches)
- encode: convert (x, y) to a high-dim vector using sine and cosine functions but in different frequencies
- like in "Attention is all you need" paper
==============================

"""

import torch
import numpy as np

def get_1d_sincos_pos_embed(embed_dim, positions):
    """
    1D sinusoidal positional embedding.
    
    Args:
        embed_dim: Output dimension
        positions: Array of positions (N,)
    
    Returns:
        pos_embed: (N, embed_dim)
    """
    assert embed_dim % 2 == 0, "embed_dim must be even"
    
    # Frequency bands: 10000^(-2i/d)
    omega = np.arange(embed_dim // 2, dtype=np.float32)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)
    
    positions = positions.reshape(-1)  # (N,)
    out = np.outer(positions, omega)   # (N, D/2)
    
    # Concatenate sin và cos
    emb_sin = np.sin(out)
    emb_cos = np.cos(out)
    pos_embed = np.concatenate([emb_sin, emb_cos], axis=1)  # (N, D)
    
    return pos_embed

def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False):
    """_summary_

    Args:
        embed_dim: Dimension của embedding (768 cho ViT-Base)
        grid_size: Kích thước grid (14 cho 224/16)
        cls_token: Có thêm position cho CLS token không
    
    Returns:
        pos_embed: (grid_size*grid_size, embed_dim) hoặc (1+grid_size*grid_size, embed_dim)
    """
    grid_h=np.arange(grid_size, dtype=np.float32)
    grid_w=np.arange(grid_size, dtype=np.float32)
    grid=np.meshgrid(grid_w, grid_h)  # (2, grid_size, grid_size)
    grid=np.stack(grid, axis=0)  # (2, grid_size, grid_size)
    grid=grid.reshape([2, 1, grid_size, grid_size])  # (2,1,grid_size,grid_size)
    
    pos_embed_h=get_1d_sincos_pos_embed(embed_dim//2, grid[0].reshape(-1))  # (grid_size*grid_size, D/2)
    pos_embed_w=get_1d_sincos_pos_embed(embed_dim//2, grid[1].reshape(-1))  # (grid_size*grid_size, D/2)
    pos_embed=np.concatenate([pos_embed_h, pos_embed_w], axis=1)  # (grid_size*grid_size, D)
    
    if cls_token:
        pos_embed=np.concatenate(
            [np.zeros([1, embed_dim]), pos_embed], axis=0
        )
    return pos_embed
if __name__ == "__main__":
    pos_embed=get_2d_sincos_pos_embed(768, 14, cls_token=True)
    print("Positional Embedding shape:", pos_embed.shape)  # Expected: (197, 768)
    