---
phase: implementation
title: Implementation Guide
description: Hướng dẫn kỹ thuật implement MAE model variants
---

# Implementation Guide

## Development Setup

**How do we get started?**

### Prerequisites

- Python 3.8+
- PyTorch ≥ 1.8
- numpy, matplotlib (for visualization)
- Existing modules in `mae/` package

### Environment Setup

```bash
# Activate conda environment
conda activate mae  # or your env name

# Verify
python -c "import torch; print(torch.__version__)"
python -c "from mae.patch_embed import PatchEmbed; print('OK')"
```

### Configuration

- No config files needed — all hyperparameters passed as constructor args
- Default values match paper's recommended settings

## Code Structure

**How is the code organized?**

### Current Structure (Before)

```
mae/
├── __init__.py          # Empty
├── attention.py         # ✅ Multi-head attention
├── transformer.py       # ✅ Transformer Block + MLP
├── patch_embed.py       # ✅ Patch Embedding
├── pos_embed.py         # ✅ Sinusoidal positional embedding
├── masking.py           # ✅ Random masking
├── encoder.py           # ❌ Empty
├── decoder.py           # ❌ Empty
├── mae.py               # ✅ Full MAE (ViT-Base, monolithic)
└── mae_large.py         # ⚠️ Duplicate logic, to be deprecated
```

### Target Structure (After)

```
mae/                         # Core reusable modules
├── __init__.py              # Exports all public APIs
├── attention.py             # Multi-head self-attention
├── transformer.py           # Transformer Block (Pre-norm + MLP)
├── patch_embed.py           # Patch Embedding (Conv2d projection)
├── pos_embed.py             # 2D sinusoidal positional embedding
├── masking.py               # Random masking utility
├── encoder.py               # ✨ NEW: MAEEncoder class
├── decoder.py               # ✨ NEW: MAEDecoder class
└── mae.py                   # Refactored: composes Encoder + Decoder

models/                      # ✨ NEW: Model variants folder
├── __init__.py              # Factory functions & registry
├── README.md                # Documentation & comparison
├── mae_vit_base.py          # ViT-Base/16 config
├── mae_vit_large.py         # ViT-Large/16 config
└── mae_vit_huge.py          # ViT-Huge/14 config

tests/
├── test_mae.py              # Updated tests
├── test_encoder_decoder.py  # ✨ NEW: Encoder/Decoder unit tests
├── test_model_variants.py   # ✨ NEW: All variant tests
└── comparision.py           # Updated comparison vs official
```

### Naming Conventions

- Classes: `PascalCase` — `MAEEncoder`, `MAEDecoder`, `MaskedAutoencoder`
- Factory functions: `snake_case` — `mae_vit_base_patch16()`, matching official convention
- Files: `snake_case` — `mae_vit_base.py`
- Constants: `UPPER_CASE` — `DEFAULT_MASK_RATIO = 0.75`

## Implementation Notes

**Key technical details to remember:**

### Core Feature 1: MAEEncoder (`mae/encoder.py`)

**What it does**: Wraps N transformer blocks + final LayerNorm. Operates only on visible patches.

```python
class MAEEncoder(nn.Module):
    """
    MAE Encoder — applies transformer blocks to visible patch tokens.

    Paper Section 3.2: "Our encoder is a ViT but applied only on visible,
    unmasked patches. Just as in a standard ViT, our encoder embeds patches
    by a linear projection with added positional embeddings, and then
    processes the resulting set via a series of Transformer blocks."

    Note: Masking, patch embedding, and CLS token are handled by the
    parent MaskedAutoencoder class. This module is purely the transformer
    backbone.
    """
    def __init__(self, embed_dim, depth, num_heads, mlp_ratio=4.0, norm_layer=nn.LayerNorm):
        super().__init__()
        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio) for _ in range(depth)
        ])
        self.norm = norm_layer(embed_dim)

    def forward(self, x):
        for blk in self.blocks:
            x = blk(x)
        return self.norm(x)
```

**Key decisions**:

- Encoder does NOT own `patch_embed`, `pos_embed`, `cls_token`, or masking
- These remain in `MaskedAutoencoder` for flexibility
- Encoder is just "stack of blocks + norm" — clean, composable

### Core Feature 2: MAEDecoder (`mae/decoder.py`)

**What it does**: Projects encoder output, inserts mask tokens, reconstructs pixel values.

```python
class MAEDecoder(nn.Module):
    """
    MAE Decoder — lightweight module for pre-training reconstruction.

    Paper Section 3.3: "The MAE decoder is only used during pre-training
    to perform the image reconstruction task... the decoder can be flexibly
    designed in a manner that is independent of the encoder design."

    Pipeline:
    1. Linear projection: encoder_dim → decoder_dim
    2. Insert mask tokens at masked positions
    3. Unshuffle to original patch order
    4. Add decoder positional embedding
    5. Apply M transformer blocks
    6. Linear prediction: decoder_dim → patch_size² × in_chans
    """
    def __init__(self, encoder_embed_dim, decoder_embed_dim, num_patches,
                 patch_size, in_chans=3, depth=8, num_heads=16, mlp_ratio=4.0):
        super().__init__()
        self.decoder_embed = nn.Linear(encoder_embed_dim, decoder_embed_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        self.decoder_pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + 1, decoder_embed_dim),
            requires_grad=False
        )
        self.blocks = nn.ModuleList([
            Block(decoder_embed_dim, num_heads, mlp_ratio) for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(decoder_embed_dim)
        self.pred = nn.Linear(decoder_embed_dim, patch_size**2 * in_chans)

    def forward(self, x, ids_restore):
        # 1. Project
        x = self.decoder_embed(x)
        # 2. Insert mask tokens + unshuffle
        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).expand(-1, -1, x.shape[2]))
        x = torch.cat([x[:, :1, :], x_], dim=1)
        # 3. Add pos embed
        x = x + self.decoder_pos_embed
        # 4. Transform
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        # 5. Predict pixels, remove CLS
        x = self.pred(x)
        x = x[:, 1:, :]
        return x
```

### Core Feature 3: Model Variants (`models/`)

Each variant file follows the same pattern:

```python
# models/mae_vit_base.py
"""
MAE ViT-Base/16
===============
Paper config: embed_dim=768, depth=12, num_heads=12, patch_size=16
Decoder: embed_dim=512, depth=8, num_heads=16
Total params: ~111M
"""

from mae.mae import MaskedAutoencoder

def mae_vit_base_patch16(**kwargs):
    model = MaskedAutoencoder(
        patch_size=16, embed_dim=768, depth=12, num_heads=12,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4.0, **kwargs
    )
    return model
```

### Patterns & Best Practices

1. **Composition over inheritance**: `MaskedAutoencoder` composes `MAEEncoder` + `MAEDecoder`, not inherits
2. **Factory pattern**: Each variant is a function that returns configured `MaskedAutoencoder`
3. **Docstrings reference paper**: Every class/method cites the relevant paper section
4. **Weight initialization**: Follow official — Xavier uniform for Linear, normal(std=0.02) for tokens, constant for LayerNorm

### Critical Implementation Detail: ViT-Huge patch_size=14

```python
# ViT-Huge uses patch_size=14, NOT 16
# This changes num_patches: (224/14)² = 16² = 256
# vs Base/Large: (224/16)² = 14² = 196

# Impact:
# - pos_embed shape: (1, 257, 1280) instead of (1, 197, 768)
# - decoder_pos_embed shape: (1, 257, 512) instead of (1, 197, 512)
# - mask: (B, 256) instead of (B, 196)
# - All handled automatically by PatchEmbed.num_patches
```

## Integration Points

**How do pieces connect?**

```python
# Full data flow:
model = MaskedAutoencoder(
    img_size=224, patch_size=16, in_chans=3,
    embed_dim=768, depth=12, num_heads=12,  # Encoder config
    decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16  # Decoder config
)

# Inside forward():
#   1. patch_embed(imgs)                    → (B, 196, 768)
#   2. + pos_embed[:, 1:]                   → (B, 196, 768)
#   3. random_masking(x, 0.75)             → (B, 49, 768), mask, ids_restore
#   4. cat(cls_token, x)                    → (B, 50, 768)
#   5. encoder.forward(x)                   → (B, 50, 768)
#   6. decoder.forward(x, ids_restore)      → (B, 196, 768)
#   7. forward_loss(imgs, pred, mask)        → scalar
```

## Error Handling

- `assert img.shape[2] == img.shape[3]`: Ensure square images
- `assert img.shape[2] % patch_size == 0`: Ensure divisible by patch_size
- Shape assertions in patchify/unpatchify

## Performance Considerations

| Variant      | Encoder FLOPs | Memory (batch=1) | Params |
| ------------ | ------------- | ---------------- | ------ |
| ViT-Base/16  | ~16 GFLOPs    | ~2 GB            | ~111M  |
| ViT-Large/16 | ~61 GFLOPs    | ~6 GB            | ~330M  |
| ViT-Huge/14  | ~167 GFLOPs   | ~16 GB           | ~657M  |

- **75% masking saves ~4× encoder compute** (process 25% of patches)
- For testing, always use CPU + small batch to avoid OOM
