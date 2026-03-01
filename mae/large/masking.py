"""
Random Masking cho MAE ViT-Large
=================================
- Shuffle patches ngẫu nhiên
- Giữ lại (1 - mask_ratio) patches
- mask: 1 = masked, 0 = visible
"""

import torch


def random_masking(x, mask_ratio=0.75):
    """
    Random masking trên batch of patch embeddings.
    
    Args:
        x: (B, L, D) - patch embeddings
        mask_ratio: Tỉ lệ patches bị mask (0.75 = 75%)
    
    Returns:
        x_masked: (B, L_visible, D) - chỉ visible patches
        mask: (B, L) - binary mask, 1 = masked, 0 = visible
        ids_restore: (B, L) - indices để khôi phục thứ tự gốc
    """
    B, L, D = x.shape
    len_keep = int(L * (1 - mask_ratio))
    
    # Random noise để shuffle
    noise = torch.rand(B, L, device=x.device)
    
    # Sort noise → thứ tự shuffle
    ids_shuffle = torch.argsort(noise, dim=1)  # ascend: nhỏ = giữ, lớn = bỏ
    ids_restore = torch.argsort(ids_shuffle, dim=1)  # inverse shuffle
    
    # Giữ lại len_keep patches đầu tiên (sau shuffle)
    ids_keep = ids_shuffle[:, :len_keep]
    x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).expand(-1, -1, D))
    
    # Tạo binary mask
    mask = torch.ones([B, L], device=x.device)
    mask[:, :len_keep] = 0
    mask = torch.gather(mask, dim=1, index=ids_restore)  # unshuffle về thứ tự gốc
    
    return x_masked, mask, ids_restore
