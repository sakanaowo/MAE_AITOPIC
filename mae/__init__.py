"""
MAE — Masked Autoencoders Are Scalable Vision Learners
=====================================================
Tái tạo kiến trúc từ bài báo: He et al., 2022

Modules:
    - MAEEncoder: ViT encoder (patch embed + mask + transformer)
    - MAEDecoder: Lightweight decoder (unshuffle + transformer + predict)
    - MaskedAutoencoder: Full MAE model (encoder + decoder + loss)

Building blocks:
    - PatchEmbed: Conv2d patch embedding
    - Block: Pre-norm transformer block (Attention + MLP)
    - Attention: Multi-head self-attention
    - random_masking: Random patch masking
    - get_2d_sincos_pos_embed: 2D sinusoidal positional embedding
"""

from .mae import MaskedAutoencoder, MAE
from .encoder import MAEEncoder
from .decoder import MAEDecoder
from .patch_embed import PatchEmbed
from .transformer import Block, Mlp
from .attention import Attention
from .masking import random_masking
from .pos_embed import get_2d_sincos_pos_embed

__all__ = [
    'MaskedAutoencoder',
    'MAE',
    'MAEEncoder',
    'MAEDecoder',
    'PatchEmbed',
    'Block',
    'Mlp',
    'Attention',
    'random_masking',
    'get_2d_sincos_pos_embed',
]
