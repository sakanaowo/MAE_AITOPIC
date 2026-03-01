"""
Positional Embedding cho MAE ViT-Large
=======================================
2D sinusoidal positional embedding:
- Mỗi patch có tọa độ (x, y) trong grid 14×14
- Encode tọa độ thành vector cao chiều bằng sin/cos ở nhiều tần số
- Tham khảo: "Attention is all you need" paper
"""

import numpy as np


def get_1d_sincos_pos_embed(embed_dim, positions):
    """
    1D sinusoidal positional embedding.
    
    Args:
        embed_dim: Chiều output embedding
        positions: Array vị trí (N,)
    
    Returns:
        pos_embed: (N, embed_dim)
    """
    assert embed_dim % 2 == 0, "embed_dim phải là số chẵn"
    
    # Frequency bands: 10000^(-2i/d)
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)
    
    positions = positions.reshape(-1)  # (N,)
    out = np.einsum('m,d->md', positions, omega)  # (N, D/2)
    
    emb_sin = np.sin(out)  # (N, D/2)
    emb_cos = np.cos(out)  # (N, D/2)
    pos_embed = np.concatenate([emb_sin, emb_cos], axis=1)  # (N, D)
    
    return pos_embed


def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False):
    """
    2D sinusoidal positional embedding.
    
    Args:
        embed_dim: Chiều embedding (1024 cho ViT-Large)
        grid_size: Kích thước grid (14 cho 224/16)
        cls_token: Có thêm position cho CLS token không
    
    Returns:
        pos_embed: (grid_size*grid_size, embed_dim) hoặc (1+grid_size*grid_size, embed_dim)
    """
    grid_h = np.arange(grid_size, dtype=np.float64)
    grid_w = np.arange(grid_size, dtype=np.float64)
    grid = np.meshgrid(grid_w, grid_h)  # 2 arrays (grid_size, grid_size)
    grid = np.stack(grid, axis=0)  # (2, grid_size, grid_size)
    grid = grid.reshape([2, 1, grid_size, grid_size])
    
    # Nửa đầu embed_dim cho h, nửa sau cho w
    pos_embed_h = get_1d_sincos_pos_embed(embed_dim // 2, grid[0].reshape(-1))
    pos_embed_w = get_1d_sincos_pos_embed(embed_dim // 2, grid[1].reshape(-1))
    pos_embed = np.concatenate([pos_embed_h, pos_embed_w], axis=1)  # (L, D)
    
    if cls_token:
        # CLS token có position = 0 vector
        pos_embed = np.concatenate(
            [np.zeros([1, embed_dim]), pos_embed], axis=0
        )
    
    return pos_embed
