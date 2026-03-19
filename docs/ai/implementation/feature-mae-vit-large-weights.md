---
phase: implementation
title: Hướng dẫn triển khai MAE ViT-Large
description: Chi tiết kỹ thuật và hướng dẫn triển khai
---

# Hướng dẫn triển khai

## Cài đặt môi trường

- Python 3.8+, PyTorch 2.0+, numpy
- File trọng số: `data/mae_pretrain_vit_large.pth`

## Cấu trúc code

```
mae/large/
├── __init__.py             # Export MAEViTLarge, mae_vit_large_patch16
├── mae_vit_large.py        # Class chính MAEViTLarge
├── attention.py            # Multi-head Attention
├── transformer.py          # Transformer Block (Attention + MLP)
├── patch_embed.py          # Patch Embedding
├── pos_embed.py            # Positional Embedding
└── masking.py              # Random masking
```

## Chi tiết triển khai từng module

### 1. `pos_embed.py` — Sinusoidal Positional Embedding
- `get_1d_sincos_pos_embed(embed_dim, positions)` → `(N, embed_dim)`
- `get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token)` → `(L, embed_dim)` hoặc `(1+L, embed_dim)`
- Tần số: `1/10000^(2i/d)`, concatenate sin và cos

### 2. `masking.py` — Random Masking
- `random_masking(x, mask_ratio)` → `(x_masked, mask, ids_restore)`
- Shuffle bằng `torch.argsort(noise)`, giữ `len_keep = int(L * (1 - mask_ratio))` patches
- mask: `1 = masked, 0 = visible`

### 3. `patch_embed.py` — Patch Embedding
- `Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)`
- **Quan trọng:** Conv2d mặc định có bias → khớp với checkpoint `patch_embed.proj.bias`
- Output: `(B, num_patches, embed_dim)`

### 4. `attention.py` — Multi-head Attention
- `qkv = nn.Linear(dim, dim*3, bias=qkv_bias)` — **qkv_bias=True** (checkpoint có `attn.qkv.bias`)
- `proj = nn.Linear(dim, dim)` — projection đầu ra
- Scaled dot-product: `softmax(QK^T / √d_k) × V`

### 5. `transformer.py` — Transformer Block
- Pre-norm architecture: `LayerNorm → Attention → Residual → LayerNorm → MLP → Residual`
- MLP: `Linear → GELU → Linear` với expansion ratio = `mlp_ratio`
- Key naming: `norm1, attn, norm2, mlp` (khớp checkpoint)

### 6. `mae_vit_large.py` — Model chính
- Encoder: PatchEmbed + CLS token + 24 Blocks + LayerNorm
- Decoder: Linear projection + mask token + 8 Blocks + prediction head
- `load_pretrained_encoder(path)`: Load chỉ encoder weights từ checkpoint

## Chiến lược load trọng số

1. `torch.load(path, map_location='cpu')`
2. Lấy `ckpt['model']`
3. `self.load_state_dict(state_dict, strict=False)`
4. Kiểm tra: tất cả missing keys phải là decoder-related
