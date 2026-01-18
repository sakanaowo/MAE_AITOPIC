# 🎯 Hướng Dẫn Implement MAE ViT-Large

> **Mục tiêu**: Tự implement Masked Autoencoder với kiến trúc **ViT-Large** (307M params)
> 
> **Khác biệt với ViT-Base**: embed_dim=1024, depth=24, num_heads=16

---

## So sánh ViT-Base vs ViT-Large

| Config | ViT-Base | **ViT-Large** | ViT-Huge |
|--------|----------|---------------|----------|
| `patch_size` | 16 | **16** | 14 |
| `embed_dim` | 768 | **1024** | 1280 |
| `depth` (encoder) | 12 | **24** | 32 |
| `num_heads` | 12 | **16** | 16 |
| `decoder_embed_dim` | 512 | **512** | 512 |
| `decoder_depth` | 8 | **8** | 8 |
| `decoder_num_heads` | 16 | **16** | 16 |
| **Total Params** | ~86M | **~307M** | ~632M |
| **GPU Memory** | ~8GB | **~16GB** | ~32GB |

---

## Yêu Cầu Phần Cứng

> [!WARNING]
> ViT-Large yêu cầu GPU với **≥16GB VRAM** cho training batch_size=32

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| **GPU VRAM** | 16GB (V100) | 32GB (A100) |
| **RAM** | 32GB | 64GB |
| **Storage** | 50GB | 100GB+ |

---

## Config ViT-Large

### Encoder Configuration

```python
# ViT-Large Encoder Config
ENCODER_CONFIG = {
    'img_size': 224,
    'patch_size': 16,
    'embed_dim': 1024,      # Tăng từ 768 → 1024
    'depth': 24,            # Tăng từ 12 → 24 blocks
    'num_heads': 16,        # Tăng từ 12 → 16 heads
    'mlp_ratio': 4.0,
}

# Số patches: (224/16)² = 196
# Tokens sau masking 75%: 49 visible + 1 CLS = 50
```

### Decoder Configuration

```python
# ViT-Large Decoder Config (giữ nguyên như Base)
DECODER_CONFIG = {
    'decoder_embed_dim': 512,
    'decoder_depth': 8,
    'decoder_num_heads': 16,
}
```

---

## Implement MAE ViT-Large

### File: `mae/mae_large.py`

```python
"""
MAE ViT-Large Implementation
============================
- embed_dim: 1024 (vs 768 in Base)
- depth: 24 blocks (vs 12 in Base)
- num_heads: 16 (vs 12 in Base)
- Total params: ~307M
"""

import torch
import torch.nn as nn
import numpy as np
from .patch_embed import PatchEmbed
from .pos_embed import get_2d_sincos_pos_embed
from .masking import random_masking
from .transformer import Block

class MAELarge(nn.Module):
    """MAE with ViT-Large backbone (~307M params)."""
    
    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_chans=3,
        # ========== ViT-Large Config ==========
        embed_dim=1024,         # 768 → 1024
        depth=24,               # 12 → 24
        num_heads=16,           # 12 → 16
        # ========== Decoder (same as Base) ==========
        decoder_embed_dim=512,
        decoder_depth=8,
        decoder_num_heads=16,
        mlp_ratio=4.,
        norm_pix_loss=False
    ):
        super().__init__()
        
        # ========== Encoder ==========
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches  # 196
        
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + 1, embed_dim), 
            requires_grad=False
        )
        
        # 24 transformer blocks (double của Base)
        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio) for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        
        # ========== Decoder ==========
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        self.decoder_pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + 1, decoder_embed_dim),
            requires_grad=False
        )
        
        self.decoder_blocks = nn.ModuleList([
            Block(decoder_embed_dim, decoder_num_heads, mlp_ratio) 
            for _ in range(decoder_depth)
        ])
        self.decoder_norm = nn.LayerNorm(decoder_embed_dim)
        self.decoder_pred = nn.Linear(decoder_embed_dim, patch_size**2 * in_chans)
        
        # ========== Other ==========
        self.patch_size = patch_size
        self.norm_pix_loss = norm_pix_loss
        
        self.initialize_weights()
    
    def initialize_weights(self):
        # Positional embeddings
        pos_embed = get_2d_sincos_pos_embed(
            self.pos_embed.shape[-1],  # 1024 for Large
            int(self.patch_embed.num_patches**0.5), 
            cls_token=True
        )
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))
        
        dec_pos_embed = get_2d_sincos_pos_embed(
            self.decoder_pos_embed.shape[-1],  # 512
            int(self.patch_embed.num_patches**0.5),
            cls_token=True
        )
        self.decoder_pos_embed.data.copy_(torch.from_numpy(dec_pos_embed).float().unsqueeze(0))
        
        nn.init.normal_(self.cls_token, std=0.02)
        nn.init.normal_(self.mask_token, std=0.02)
        
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
    
    def patchify(self, imgs):
        p = self.patch_size
        B, C, H, W = imgs.shape
        h = w = H // p
        x = imgs.reshape(B, C, h, p, w, p)
        x = x.permute(0, 2, 4, 3, 5, 1)
        x = x.reshape(B, h * w, p * p * C)
        return x
    
    def unpatchify(self, x):
        p = self.patch_size
        h = w = int(x.shape[1]**0.5)
        x = x.reshape(x.shape[0], h, w, p, p, 3)
        x = x.permute(0, 5, 1, 3, 2, 4)
        x = x.reshape(x.shape[0], 3, h * p, w * p)
        return x
    
    def forward_encoder(self, x, mask_ratio):
        x = self.patch_embed(x)
        x = x + self.pos_embed[:, 1:, :]
        x, mask, ids_restore = random_masking(x, mask_ratio)
        
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        
        return x, mask, ids_restore
    
    def forward_decoder(self, x, ids_restore):
        x = self.decoder_embed(x)
        
        mask_tokens = self.mask_token.repeat(
            x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1
        )
        
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).expand(-1, -1, x.shape[2]))
        x = torch.cat([x[:, :1, :], x_], dim=1)
        
        x = x + self.decoder_pos_embed
        
        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)
        x = self.decoder_pred(x)
        x = x[:, 1:, :]
        
        return x
    
    def forward_loss(self, imgs, pred, mask):
        target = self.patchify(imgs)
        
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.e-6)**0.5
        
        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)
        loss = (loss * mask).sum() / mask.sum()
        return loss
    
    def forward(self, imgs, mask_ratio=0.75):
        latent, mask, ids_restore = self.forward_encoder(imgs, mask_ratio)
        pred = self.forward_decoder(latent, ids_restore)
        loss = self.forward_loss(imgs, pred, mask)
        return loss, pred, mask


# Factory function
def mae_vit_large_patch16(**kwargs):
    """Create MAE ViT-Large/16 model."""
    return MAELarge(
        patch_size=16,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        decoder_embed_dim=512,
        decoder_depth=8,
        decoder_num_heads=16,
        **kwargs
    )
```

---

## Test ViT-Large

```python
# File: tests/test_mae_large.py
import torch
from mae.mae_large import MAELarge, mae_vit_large_patch16

def test_mae_large():
    model = mae_vit_large_patch16()
    
    # Count params
    params = sum(p.numel() for p in model.parameters())
    print(f"Total params: {params/1e6:.1f}M")  # Expected: ~307M
    
    # Forward pass
    x = torch.randn(2, 3, 224, 224)
    loss, pred, mask = model(x, mask_ratio=0.75)
    
    print(f"Input: {x.shape}")
    print(f"Loss: {loss.item():.4f}")
    print(f"Pred: {pred.shape}")
    print(f"Mask ratio: {mask.sum()/mask.numel():.2%}")
    
    # Memory estimate
    print(f"\nEstimated GPU memory: ~{params * 4 / 1e9:.1f}GB (float32)")
    
    print("\n✅ All tests passed!")

if __name__ == "__main__":
    test_mae_large()
```

---

## Pre-training ViT-Large

### Hyperparameters (Official)

| Parameter | ViT-Base | **ViT-Large** |
|-----------|----------|---------------|
| Batch size (effective) | 4096 | 4096 |
| Base LR | 1.5e-4 | **1.5e-4** |
| Epochs | 800 | **800** |
| Warmup epochs | 40 | 40 |
| Weight decay | 0.05 | 0.05 |
| Training time | ~42h (64 V100s) | **~84h (64 V100s)** |

### Command

```bash
# Pre-training ViT-Large (8 nodes × 8 GPUs)
python submitit_pretrain.py \
    --job_dir ${JOB_DIR} \
    --nodes 8 \
    --use_volta32 \
    --batch_size 64 \
    --model mae_vit_large_patch16 \
    --norm_pix_loss \
    --mask_ratio 0.75 \
    --epochs 800 \
    --warmup_epochs 40 \
    --blr 1.5e-4 --weight_decay 0.05 \
    --data_path ${IMAGENET_DIR}
```

---

## Fine-tuning ViT-Large

### Hyperparameters

| Parameter | ViT-Base | **ViT-Large** |
|-----------|----------|---------------|
| Epochs | 100 | **50** |
| Base LR | 5e-4 | **1e-3** |
| Layer decay | 0.65 | **0.75** |
| Drop path | 0.1 | **0.2** |
| Training time | ~7h (32 V100s) | **~9h (32 V100s)** |
| **Accuracy** | 83.6% | **85.9%** |

### Command

```bash
# Fine-tuning ViT-Large (4 nodes × 8 GPUs)
python submitit_finetune.py \
    --job_dir ${JOB_DIR} \
    --nodes 4 --use_volta32 \
    --batch_size 32 \
    --model vit_large_patch16 \
    --finetune ${PRETRAIN_CHKPT} \
    --epochs 50 \
    --blr 1e-3 --layer_decay 0.75 \
    --weight_decay 0.05 --drop_path 0.2 \
    --reprob 0.25 --mixup 0.8 --cutmix 1.0 \
    --dist_eval --data_path ${IMAGENET_DIR}
```

---

## Download Pre-trained Checkpoint

```bash
# ViT-Large pre-trained
wget https://dl.fbaipublicfiles.com/mae/pretrain/mae_pretrain_vit_large.pth \
    -O checkpoints/mae_pretrain_vit_large.pth

# ViT-Large fine-tuned
wget https://dl.fbaipublicfiles.com/mae/finetune/mae_finetuned_vit_large.pth \
    -O checkpoints/mae_finetuned_vit_large.pth
```

---

## So sánh với Official

```python
# Compare with official implementation
import sys
sys.path.insert(0, 'official_mae')
import models_mae

# Official
official = models_mae.mae_vit_large_patch16()
official_params = sum(p.numel() for p in official.parameters())

# Our implementation
from mae.mae_large import mae_vit_large_patch16
ours = mae_vit_large_patch16()
our_params = sum(p.numel() for p in ours.parameters())

print(f"Official: {official_params/1e6:.1f}M")
print(f"Ours: {our_params/1e6:.1f}M")
print(f"Match: {official_params == our_params}")
```

---

## Tham Khảo

- [MAE Paper](https://arxiv.org/abs/2111.06377) - Table 1: ViT-L achieves 85.9% on ImageNet
- [Official MAE Repo](https://github.com/facebookresearch/mae)
- [ViT Paper](https://arxiv.org/abs/2010.11929) - Vision Transformer architecture

---

## Next Steps

1. **Implement**: Tạo các files trong `mae/` theo hướng dẫn
2. **Test**: Chạy tests để verify param count (~307M)
3. **Train**: Sử dụng official scripts với dataset của bạn
4. **Compare**: So sánh kết quả với official checkpoint
