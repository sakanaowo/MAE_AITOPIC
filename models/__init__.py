"""
MAE Model Variants
==================
Factory functions cho 3 biến thể MAE chính thức:
    - mae_vit_base_patch16:  ViT-Base  (768-dim,  12 blocks, patch 16) ~111M params
    - mae_vit_large_patch16: ViT-Large (1024-dim, 24 blocks, patch 16) ~330M params
    - mae_vit_huge_patch14:  ViT-Huge  (1280-dim, 32 blocks, patch 14) ~657M params

Usage:
    from models import mae_vit_base_patch16, mae_vit_large_patch16, mae_vit_huge_patch14
    
    model = mae_vit_large_patch16()
    model.load_pretrained('checkpoints/mae_pretrain_vit_large.pth')
"""

from .mae_vit_base import mae_vit_base_patch16
from .mae_vit_large import mae_vit_large_patch16
from .mae_vit_huge import mae_vit_huge_patch14

__all__ = [
    'mae_vit_base_patch16',
    'mae_vit_large_patch16',
    'mae_vit_huge_patch14',
]
