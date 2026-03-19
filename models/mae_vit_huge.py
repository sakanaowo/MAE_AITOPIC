"""
MAE ViT-Huge (patch14)
======================
Config từ bài báo gốc:
    Encoder: embed_dim=1280, depth=32, num_heads=16
    Decoder: embed_dim=512, depth=8, num_heads=16
    Patches: 16×16 = 256 (patch_size=14, img=224)
    Params: ~657M total

LƯU Ý:
    - patch_size=14 (khác 16 của Base/Large)
    - num_patches=256 (khác 196)
    - pos_embed shape: (1, 257, 1280) encoder, (1, 257, 512) decoder
    - Inference: ~2.7 GB VRAM (bs=1, FP32) → Chạy được trên RTX 3060 12GB
    - Training: >12 GB VRAM → Cần GPU lớn hơn (A100/V100 32GB+)

Checkpoint:
    mae_pretrain_vit_huge.pth (encoder-only, 390 keys)
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from mae import MaskedAutoencoder


def mae_vit_huge_patch14(**kwargs):
    """Tạo MAE ViT-Huge/14 với config chuẩn."""
    model = MaskedAutoencoder(
        img_size=224,
        patch_size=14,
        in_chans=3,
        embed_dim=1280,
        depth=32,
        num_heads=16,
        decoder_embed_dim=512,
        decoder_depth=8,
        decoder_num_heads=16,
        mlp_ratio=4.,
        **kwargs
    )
    return model
