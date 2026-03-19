"""
MAE ViT-Base (patch16)
======================
Config từ bài báo gốc:
    Encoder: embed_dim=768, depth=12, num_heads=12
    Decoder: embed_dim=512, depth=8, num_heads=16
    Patches: 14×14 = 196 (patch_size=16, img=224)
    Params: ~111M total

Checkpoint:
    mae_pretrain_vit_base.pth (encoder-only)
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from mae import MaskedAutoencoder


def mae_vit_base_patch16(**kwargs):
    """Tạo MAE ViT-Base/16 với config chuẩn."""
    model = MaskedAutoencoder(
        img_size=224,
        patch_size=16,
        in_chans=3,
        embed_dim=768,
        depth=12,
        num_heads=12,
        decoder_embed_dim=512,
        decoder_depth=8,
        decoder_num_heads=16,
        mlp_ratio=4.,
        **kwargs
    )
    return model
