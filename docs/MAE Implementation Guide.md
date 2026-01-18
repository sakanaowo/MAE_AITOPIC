# 🎯 Hướng Dẫn Tái Tạo Kiến Trúc MAE

> **Mục tiêu**: Tự tay implement Masked Autoencoder từ paper "Masked Autoencoders Are Scalable Vision Learners"

---

## Bước 1: Setup Environment

### 1.1 Yêu Cầu Hệ Thống

| Thành phần | Yêu cầu | Ghi chú |
|------------|---------|---------|
| **Python** | 3.8+ | Khuyến nghị 3.8 - 3.10 |
| **CUDA** | 11.x hoặc 12.x | Check với `nvcc --version` |
| **GPU** | NVIDIA với ≥8GB VRAM | V100/A100 khuyến nghị cho training |
| **RAM** | ≥16GB | ≥32GB cho training |

### 1.2 Tạo Project và Virtual Environment

```bash
# 1. Tạo project folder
mkdir -p ~/Code/mae-reproduction && cd ~/Code/mae-reproduction

# 2. Tạo virtual environment
python -m venv venv
source venv/bin/activate  # Linux/macOS
# hoặc: .\venv\Scripts\activate  # Windows
```

### 1.3 Cài Đặt Dependencies

> [!IMPORTANT]
> Official repo dựa trên **`timm==0.3.2`** và là modification của [DeiT repo](https://github.com/facebookresearch/deit). 
> Cần một [fix](https://github.com/rwightman/pytorch-image-models/issues/420#issuecomment-776459842) để hoạt động với **PyTorch 1.8.1+**.

```bash
# 1. Cài PyTorch (chọn version phù hợp với CUDA)
# CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
# CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
# CPU only
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# 2. Cài timm (quan trọng: dùng đúng version)
pip install timm==0.3.2

# 3. Cài các dependencies khác
pip install tensorboard matplotlib numpy pillow

# 4. Cho distributed training (multi-node)
pip install submitit
```

### 1.4 Fix cho timm==0.3.2 với PyTorch 1.8.1+

> [!WARNING]
> Nếu gặp lỗi với `timm==0.3.2` và PyTorch ≥1.8.1, cần apply fix sau:

```python
# File: venv/lib/python3.x/site-packages/timm/models/layers/helpers.py
# Thay đổi dòng:
# from torch._six import container_abcs
# Thành:
import collections.abc as container_abcs
```

Hoặc chạy lệnh:
```bash
# Tự động fix
sed -i 's/from torch._six import container_abcs/import collections.abc as container_abcs/g' \
    venv/lib/python3.*/site-packages/timm/models/layers/helpers.py
```

### 1.5 Cấu Trúc Project (Tự Implement)

> [!NOTE]
> Mục tiêu là **tự implement MAE từ đầu** để hiểu sâu kiến trúc.  
> `official_mae/` được clone để **tham khảo và verify** kết quả.

```bash
# Clone official repo để tham khảo
git clone https://github.com/facebookresearch/mae.git official_mae

# Tạo cấu trúc cho việc tự implement
mkdir -p mae checkpoints data tests
touch mae/__init__.py
```

**Cấu trúc project:**

```
MAE_AITOPIC/
├── mae/                        # 🎯 TỰ IMPLEMENT - Core MAE từ đầu
│   ├── __init__.py
│   ├── patch_embed.py         # Patch Embedding module
│   ├── pos_embed.py           # Positional Embedding functions
│   ├── masking.py             # Random Masking logic
│   ├── attention.py           # Multi-Head Self-Attention
│   ├── transformer.py         # Transformer Block
│   ├── encoder.py             # MAE Encoder
│   ├── decoder.py             # MAE Decoder
│   └── mae.py                 # Full MAE Model
├── official_mae/               # 📚 REFERENCE - Official Facebook Research
│   ├── models_mae.py          # So sánh implementation
│   ├── util/pos_embed.py      # Tham khảo pos embedding
│   └── ...
├── checkpoints/                # Model checkpoints
├── data/                       # Datasets
├── tests/                      # Unit tests
├── train.py                   # Training script
├── visualize.py               # Visualization
└── docs/                       # Documentation
```

### So sánh khi Tự Implement vs Official

| Component | Tự Implement (`mae/`) | Tham khảo (`official_mae/`) |
|-----------|----------------------|----------------------------|
| Patch Embed | `mae/patch_embed.py` | Dùng `timm.PatchEmbed` |
| Pos Embed | `mae/pos_embed.py` | `util/pos_embed.py` |
| Masking | `mae/masking.py` | `models_mae.py:random_masking()` |
| Transformer | `mae/transformer.py` | Dùng `timm.Block` |
| Full Model | `mae/mae.py` | `models_mae.py:MaskedAutoencoderViT` |

### 1.6 Verify Installation

```bash
# Check PyTorch + CUDA
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"

# Check timm
python -c "import timm; print(f'timm: {timm.__version__}')"

# Check all dependencies
python -c "
import torch
import torchvision
import timm
from timm.models.vision_transformer import PatchEmbed, Block
print('✓ All dependencies OK')
print(f'  PyTorch: {torch.__version__}')
print(f'  TorchVision: {torchvision.__version__}')
print(f'  timm: {timm.__version__}')
print(f'  CUDA available: {torch.cuda.is_available()}')
"
```

### 1.7 Dependencies Reference

| Package | Version | Mục đích |
|---------|---------|----------|
| `torch` | ≥1.8.1 | Deep learning framework |
| `torchvision` | ≥0.9.1 | Image transforms, datasets |
| `timm` | **0.3.2** | Vision Transformer components |
| `tensorboard` | latest | Training visualization |
| `matplotlib` | latest | Reconstruction visualization |
| `numpy` | latest | Numerical operations |
| `pillow` | latest | Image processing |
| `submitit` | latest | SLURM job submission (multi-node) |

---

## Bước 2: Download Checkpoints (Để Verify)

Download checkpoints official để **verify implementation** và **chạy visualization demo**:

```bash
mkdir -p checkpoints

# ViT-Base checkpoint (để so sánh)
wget https://dl.fbaipublicfiles.com/mae/pretrain/mae_pretrain_vit_base.pth -O checkpoints/mae_pretrain_vit_base.pth
```

---

## Bước 3: Implement Patch Embedding

> **Lý thuyết**: Chia ảnh 224×224 thành 196 patches (14×14 grid), mỗi patch 16×16 pixels.
> Mỗi patch được flatten và project thành vector D-dimensional.

**File**: `mae/patch_embed.py`

```python
"""
Patch Embedding Module
=====================
- Input: Image (B, 3, 224, 224)
- Output: Patch tokens (B, 196, 768)

Công thức:
- num_patches = (img_size / patch_size)² = (224/16)² = 196
- Mỗi patch: 16×16×3 = 768 pixels → project thành D-dim vector
"""

import torch
import torch.nn as nn

class PatchEmbed(nn.Module):
    """Chia ảnh thành patches và project thành embeddings."""
    
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2  # 196 patches
        
        # Conv2d với kernel_size=stride=patch_size tương đương linear projection per patch
        # Tại sao dùng Conv2d thay vì Linear?
        # → Hiệu quả hơn về memory và compute (không cần reshape)
        self.proj = nn.Conv2d(
            in_chans, embed_dim, 
            kernel_size=patch_size, 
            stride=patch_size
        )
    
    def forward(self, x):
        """
        x: (B, 3, 224, 224) - batch of images
        returns: (B, 196, 768) - patch embeddings
        """
        # (B, 3, 224, 224) → (B, 768, 14, 14) → (B, 768, 196) → (B, 196, 768)
        x = self.proj(x)           # (B, embed_dim, H/P, W/P)
        x = x.flatten(2)           # (B, embed_dim, num_patches)
        x = x.transpose(1, 2)      # (B, num_patches, embed_dim)
        return x
```

**Test**:
```python
# Test PatchEmbed
from mae.patch_embed import PatchEmbed

pe = PatchEmbed(img_size=224, patch_size=16, embed_dim=768)
x = torch.randn(2, 3, 224, 224)
out = pe(x)
assert out.shape == (2, 196, 768), f"Expected (2, 196, 768), got {out.shape}"
print(f"✓ PatchEmbed: {x.shape} → {out.shape}")
```

> **📚 So sánh với Official**: Official dùng `timm.models.vision_transformer.PatchEmbed` với cùng logic.

---

## Bước 4: Implement Positional Embedding

> **Lý thuyết**: Transformer không có khái niệm vị trí (permutation invariant).
> 2D sinusoidal positional embedding encode vị trí (x, y) của mỗi patch.
> MAE dùng **fixed** positional embedding (không học).

**File**: `mae/pos_embed.py`

```python
"""
2D Sinusoidal Positional Embedding
=================================
- Mỗi patch có vị trí (x, y) trong grid 14×14
- Encode position bằng sin/cos ở các frequencies khác nhau
- Giống Transformer gốc nhưng mở rộng cho 2D

Công thức cho 1D:
PE(pos, 2i) = sin(pos / 10000^(2i/d))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d))
"""

import numpy as np
import torch

def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False):
    """
    Tạo 2D sinusoidal positional embedding.
    
    Args:
        embed_dim: Dimension của embedding (768 cho ViT-Base)
        grid_size: Kích thước grid (14 cho 224/16)
        cls_token: Có thêm position cho CLS token không
    
    Returns:
        pos_embed: (grid_size*grid_size, embed_dim) hoặc (1+grid_size*grid_size, embed_dim)
    """
    # Tạo grid positions
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # (2, grid_size, grid_size)
    grid = np.stack(grid, axis=0)       # (2, grid_size, grid_size)
    grid = grid.reshape([2, 1, grid_size, grid_size])
    
    # Embed mỗi chiều (x, y) với embed_dim/2
    pos_embed_h = get_1d_sincos_pos_embed(embed_dim // 2, grid[0].reshape(-1))
    pos_embed_w = get_1d_sincos_pos_embed(embed_dim // 2, grid[1].reshape(-1))
    pos_embed = np.concatenate([pos_embed_h, pos_embed_w], axis=1)  # (H*W, D)
    
    if cls_token:
        # CLS token ở position 0 (pad với zeros)
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    
    return pos_embed

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
```

**Test**:
```python
# Test Positional Embedding
from mae.pos_embed import get_2d_sincos_pos_embed

pos = get_2d_sincos_pos_embed(embed_dim=768, grid_size=14, cls_token=True)
assert pos.shape == (197, 768), f"Expected (197, 768), got {pos.shape}"
print(f"✓ Positional Embedding: shape = {pos.shape}")
```

> **📚 So sánh với Official**: Xem `official_mae/util/pos_embed.py`

---

## Bước 5: Implement Random Masking

> **Lý thuyết - Key Innovation của MAE**:
> - Mask **75%** patches ngẫu nhiên (cao hơn BERT 15% vì ảnh có redundancy cao)
> - Chỉ encode **visible patches** (25%) → tiết kiệm 3-4× compute
> - Tránh "information leak" khi dùng mask token trong encoder

**File**: `mae/masking.py`

```python
"""
Random Masking Strategy
=======================
- Mask ratio 75%: Giữ lại 25% patches (49 patches từ 196)
- Random shuffle để đảm bảo random selection
- Trả về ids_restore để unshuffle khi decode
"""

import torch

def random_masking(x, mask_ratio=0.75):
    """
    Thực hiện random masking trên batch of patch embeddings.
    
    Args:
        x: (B, L, D) - patch embeddings
        mask_ratio: Tỉ lệ patches bị mask (0.75 = 75%)
    
    Returns:
        x_masked: (B, L_visible, D) - chỉ visible patches
        mask: (B, L) - binary mask, 1 = masked, 0 = visible
        ids_restore: (B, L) - indices để khôi phục thứ tự gốc
    """
    B, L, D = x.shape
    len_keep = int(L * (1 - mask_ratio))  # Số patches giữ lại (49)
    
    # Random noise để shuffle
    noise = torch.rand(B, L, device=x.device)  # (B, L)
    
    # Sort noise → ids_shuffle chứa thứ tự mới
    ids_shuffle = torch.argsort(noise, dim=1)
    # ids_restore: inverse của ids_shuffle (để unshuffle)
    ids_restore = torch.argsort(ids_shuffle, dim=1)
    
    # Giữ lại len_keep patches đầu tiên (sau khi shuffle)
    ids_keep = ids_shuffle[:, :len_keep]
    
    # Gather visible patches
    x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).expand(-1, -1, D))
    
    # Tạo binary mask (1 = masked, 0 = visible)
    mask = torch.ones([B, L], device=x.device)
    mask[:, :len_keep] = 0
    # Unshuffle mask về thứ tự gốc
    mask = torch.gather(mask, dim=1, index=ids_restore)
    
    return x_masked, mask, ids_restore
```

**Test**:
```python
# Test Random Masking
from mae.masking import random_masking

x = torch.randn(2, 196, 768)
x_masked, mask, ids_restore = random_masking(x, mask_ratio=0.75)

assert x_masked.shape == (2, 49, 768), f"Expected (2, 49, 768), got {x_masked.shape}"
assert mask.shape == (2, 196), f"Expected (2, 196), got {mask.shape}"
assert mask.sum(dim=1).mean() == 147, f"Expected 147 masked, got {mask.sum(dim=1).mean()}"
print(f"✓ Masking: {x.shape} → {x_masked.shape}, mask ratio: {mask.sum()/mask.numel():.2%}")
```

> **📚 So sánh với Official**: Xem `official_mae/models_mae.py:random_masking()`

---

## Bước 6: Implement Attention & Transformer Block

### 6.1 Multi-Head Self-Attention

**File**: `mae/attention.py`

```python
"""
Multi-Head Self-Attention
========================
- Query, Key, Value projections
- Scaled dot-product attention: softmax(QK^T / sqrt(d)) * V
- Multiple heads cho parallel attention patterns
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class Attention(nn.Module):
    """Multi-Head Self-Attention module."""
    
    def __init__(self, dim, num_heads=8, qkv_bias=True, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5  # 1/sqrt(d_k)
        
        # Combined QKV projection (hiệu quả hơn 3 projections riêng)
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
    
    def forward(self, x):
        """
        x: (B, N, C) - N tokens, C channels
        returns: (B, N, C)
        """
        B, N, C = x.shape
        
        # QKV projection: (B, N, 3*C) → (B, N, 3, num_heads, head_dim)
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, num_heads, N, head_dim)
        q, k, v = qkv.unbind(0)  # Mỗi cái (B, num_heads, N, head_dim)
        
        # Scaled dot-product attention
        attn = (q @ k.transpose(-2, -1)) * self.scale  # (B, num_heads, N, N)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        
        # Apply attention to values
        x = (attn @ v).transpose(1, 2)  # (B, N, num_heads, head_dim)
        x = x.reshape(B, N, C)  # Concatenate heads
        
        # Output projection
        x = self.proj(x)
        x = self.proj_drop(x)
        return x
```

### 6.2 Transformer Block

**File**: `mae/transformer.py`

```python
"""
Transformer Block
================
- Pre-norm architecture (LayerNorm trước Attention và MLP)
- Residual connections
- MLP với GELU activation
"""

import torch
import torch.nn as nn
from .attention import Attention

class Mlp(nn.Module):
    """MLP block với expansion ratio."""
    
    def __init__(self, in_features, hidden_features=None, out_features=None, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features * 4
        
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()  # Smoother than ReLU
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class Block(nn.Module):
    """Transformer encoder block."""
    
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0.):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, num_heads, qkv_bias, attn_drop, drop)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = Mlp(dim, int(dim * mlp_ratio), drop=drop)
    
    def forward(self, x):
        # Pre-norm + Residual
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x
```

**Test**:
```python
# Test Transformer Block
from mae.transformer import Block

block = Block(dim=768, num_heads=12)
x = torch.randn(2, 197, 768)  # 196 patches + 1 CLS
out = block(x)
assert out.shape == x.shape, f"Expected {x.shape}, got {out.shape}"
print(f"✓ Transformer Block: {x.shape} → {out.shape}")
```

---

## Bước 7: Implement Full MAE Model

**File**: `mae/mae.py`

```python
"""
Masked Autoencoder (MAE)
=======================
Key Architecture:
- Asymmetric encoder-decoder
- Encoder: Large (12 blocks), chỉ encode visible patches
- Decoder: Light (8 blocks, 512-dim), decode full sequence
- No mask token trong encoder (efficient + avoid distribution mismatch)
"""

import torch
import torch.nn as nn
import numpy as np
from .patch_embed import PatchEmbed
from .pos_embed import get_2d_sincos_pos_embed
from .masking import random_masking
from .transformer import Block

class MAE(nn.Module):
    """Masked Autoencoder with ViT backbone."""
    
    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_chans=3,
        embed_dim=768,      # Encoder dimension
        depth=12,           # Encoder depth (blocks)
        num_heads=12,       # Encoder heads
        decoder_embed_dim=512,   # Decoder dimension (lighter)
        decoder_depth=8,         # Decoder depth
        decoder_num_heads=16,
        mlp_ratio=4.,
        norm_pix_loss=False      # Normalize pixel values in loss
    ):
        super().__init__()
        
        # ========== Encoder ==========
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches  # 196
        
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        # Fixed positional embedding (không học)
        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + 1, embed_dim), 
            requires_grad=False
        )
        
        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio) for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        
        # ========== Decoder ==========
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim)
        
        # Mask token: learnable, thêm vào vị trí masked khi decode
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        
        self.decoder_pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + 1, decoder_embed_dim),
            requires_grad=False
        )
        
        self.decoder_blocks = nn.ModuleList([
            Block(decoder_embed_dim, decoder_num_heads, mlp_ratio) for _ in range(decoder_depth)
        ])
        self.decoder_norm = nn.LayerNorm(decoder_embed_dim)
        
        # Prediction head: project về pixel values
        self.decoder_pred = nn.Linear(decoder_embed_dim, patch_size**2 * in_chans)
        
        # ========== Other ==========
        self.patch_size = patch_size
        self.norm_pix_loss = norm_pix_loss
        
        self.initialize_weights()
    
    def initialize_weights(self):
        # Positional embeddings
        pos_embed = get_2d_sincos_pos_embed(
            self.pos_embed.shape[-1], 
            int(self.patch_embed.num_patches**0.5), 
            cls_token=True
        )
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))
        
        dec_pos_embed = get_2d_sincos_pos_embed(
            self.decoder_pos_embed.shape[-1],
            int(self.patch_embed.num_patches**0.5),
            cls_token=True
        )
        self.decoder_pos_embed.data.copy_(torch.from_numpy(dec_pos_embed).float().unsqueeze(0))
        
        # Tokens
        nn.init.normal_(self.cls_token, std=0.02)
        nn.init.normal_(self.mask_token, std=0.02)
        
        # Apply init to linear layers
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
        """
        imgs: (B, 3, H, W)
        returns: (B, L, patch_size**2 * 3)
        """
        p = self.patch_size
        B, C, H, W = imgs.shape
        assert H == W and H % p == 0
        h = w = H // p
        x = imgs.reshape(B, C, h, p, w, p)
        x = x.permute(0, 2, 4, 3, 5, 1)  # (B, h, w, p, p, C)
        x = x.reshape(B, h * w, p * p * C)
        return x
    
    def unpatchify(self, x):
        """
        x: (B, L, patch_size**2 * 3)
        returns: (B, 3, H, W)
        """
        p = self.patch_size
        h = w = int(x.shape[1]**0.5)
        assert h * w == x.shape[1]
        
        x = x.reshape(x.shape[0], h, w, p, p, 3)
        x = x.permute(0, 5, 1, 3, 2, 4)  # (B, 3, h, p, w, p)
        x = x.reshape(x.shape[0], 3, h * p, w * p)
        return x
    
    def forward_encoder(self, x, mask_ratio):
        """Encode visible patches only."""
        # Patch embed
        x = self.patch_embed(x)  # (B, 196, 768)
        
        # Add pos embed (không có CLS)
        x = x + self.pos_embed[:, 1:, :]
        
        # Masking
        x, mask, ids_restore = random_masking(x, mask_ratio)  # (B, 49, 768)
        
        # Append CLS token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)  # (B, 50, 768)
        
        # Transformer blocks
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        
        return x, mask, ids_restore
    
    def forward_decoder(self, x, ids_restore):
        """Decode full sequence với mask tokens."""
        # Embed tokens
        x = self.decoder_embed(x)  # (B, 50, 512)
        
        # Append mask tokens
        mask_tokens = self.mask_token.repeat(
            x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1
        )  # (B, 147, 512)
        
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # No CLS
        # Unshuffle về thứ tự gốc
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).expand(-1, -1, x.shape[2]))
        x = torch.cat([x[:, :1, :], x_], dim=1)  # Add CLS back
        
        # Add decoder pos embed
        x = x + self.decoder_pos_embed
        
        # Transformer blocks
        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)
        
        # Predict pixel values
        x = self.decoder_pred(x)
        
        # Remove CLS token
        x = x[:, 1:, :]
        
        return x
    
    def forward_loss(self, imgs, pred, mask):
        """
        MSE loss on masked patches only.
        """
        target = self.patchify(imgs)  # (B, 196, 768)
        
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.e-6)**0.5
        
        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)  # (B, 196) - per patch loss
        
        # Average loss chỉ trên masked patches
        loss = (loss * mask).sum() / mask.sum()
        return loss
    
    def forward(self, imgs, mask_ratio=0.75):
        latent, mask, ids_restore = self.forward_encoder(imgs, mask_ratio)
        pred = self.forward_decoder(latent, ids_restore)
        loss = self.forward_loss(imgs, pred, mask)
        return loss, pred, mask
```

---

## Bước 8: Test và Verify Implementation

### 8.1 Unit Tests

```python
# File: tests/test_mae.py
import torch
import sys
sys.path.insert(0, '.')

from mae.mae import MAE

def test_mae():
    model = MAE(
        img_size=224,
        patch_size=16,
        embed_dim=768,
        depth=12,
        num_heads=12,
        decoder_embed_dim=512,
        decoder_depth=8,
    )
    
    # Test forward
    x = torch.randn(2, 3, 224, 224)
    loss, pred, mask = model(x, mask_ratio=0.75)
    
    print(f"✓ Input shape: {x.shape}")
    print(f"✓ Loss: {loss.item():.4f}")
    print(f"✓ Pred shape: {pred.shape}")
    print(f"✓ Mask shape: {mask.shape}")
    print(f"✓ Mask ratio: {mask.sum()/mask.numel():.2%}")
    
    # Count params
    params = sum(p.numel() for p in model.parameters())
    print(f"✓ Total params: {params/1e6:.1f}M")
    
    # Test encoder only
    latent, mask, ids_restore = model.forward_encoder(x, mask_ratio=0.75)
    print(f"✓ Encoder output: {latent.shape}")
    
    print("\n✅ All tests passed!")

if __name__ == "__main__":
    test_mae()
```

### 8.2 So sánh với Official

```python
# File: tests/compare_with_official.py
import torch
import sys
sys.path.insert(0, '.')
sys.path.insert(0, 'official_mae')

# Load official model
import models_mae as official_mae

# Load our implementation
from mae.mae import MAE

def compare_models():
    # Create models
    our_model = MAE()
    official_model = official_mae.mae_vit_base_patch16()
    
    # Compare param counts
    our_params = sum(p.numel() for p in our_model.parameters())
    official_params = sum(p.numel() for p in official_model.parameters())
    
    print(f"Our params: {our_params/1e6:.1f}M")
    print(f"Official params: {official_params/1e6:.1f}M")
    
    # Forward pass comparison
    x = torch.randn(1, 3, 224, 224)
    
    our_loss, _, _ = our_model(x)
    official_loss, _, _ = official_model(x)
    
    print(f"Our loss: {our_loss.item():.4f}")
    print(f"Official loss: {official_loss.item():.4f}")
    
    print("\n✅ Comparison complete!")

if __name__ == "__main__":
    compare_models()
```

---

## Model Configurations

### Kiến Trúc Model

| Config | Patch Size | Embed Dim | Encoder Depth | Encoder Heads | Params |
|--------|------------|-----------|---------------|---------------|--------|
| ViT-Base/16 | 16 | 768 | 12 | 12 | ~86M |
| ViT-Large/16 | 16 | 1024 | 24 | 16 | ~307M |
| ViT-Huge/14 | 14 | 1280 | 32 | 16 | ~632M |

### Pre-trained Checkpoints

| Model | Pre-trained Checkpoint | MD5 | Fine-tuned Checkpoint |
|-------|------------------------|-----|------------------------|
| ViT-Base | [mae_pretrain_vit_base.pth](https://dl.fbaipublicfiles.com/mae/pretrain/mae_pretrain_vit_base.pth) | `8cad7c` | [mae_finetuned_vit_base.pth](https://dl.fbaipublicfiles.com/mae/finetune/mae_finetuned_vit_base.pth) |
| ViT-Large | [mae_pretrain_vit_large.pth](https://dl.fbaipublicfiles.com/mae/pretrain/mae_pretrain_vit_large.pth) | `b8b06e` | [mae_finetuned_vit_large.pth](https://dl.fbaipublicfiles.com/mae/finetune/mae_finetuned_vit_large.pth) |
| ViT-Huge | [mae_pretrain_vit_huge.pth](https://dl.fbaipublicfiles.com/mae/pretrain/mae_pretrain_vit_huge.pth) | `9bdbb0` | [mae_finetuned_vit_huge.pth](https://dl.fbaipublicfiles.com/mae/finetune/mae_finetuned_vit_huge.pth) |

### ImageNet Classification Results

| Dataset | ViT-B | ViT-L | ViT-H | ViT-H₄₄₈ |
|---------|-------|-------|-------|----------|
| **ImageNet-1K** (no external data) | 83.6 | 85.9 | 86.9 | **87.8** |
| ImageNet-Corruption (error ↓) | 51.7 | 41.8 | **33.8** | 36.8 |
| ImageNet-Adversarial | 35.9 | 57.1 | 68.2 | **76.7** |
| ImageNet-Rendition | 48.3 | 59.9 | 64.4 | **66.5** |
| ImageNet-Sketch | 34.5 | 45.3 | 49.6 | **50.9** |

### Transfer Learning Results  

| Dataset | ViT-B | ViT-L | ViT-H | ViT-H₄₄₈ |
|---------|-------|-------|-------|----------|
| iNaturalists 2017 | 70.5 | 75.7 | 79.3 | **83.4** |
| iNaturalists 2018 | 75.4 | 80.1 | 83.0 | **86.8** |
| iNaturalists 2019 | 80.5 | 83.4 | 85.7 | **88.3** |
| Places205 | 63.9 | 65.8 | 65.9 | **66.8** |
| Places365 | 57.9 | 59.4 | 59.8 | **60.3** |

---

## Chi Tiết Cấu Hình Pre-training

### Multi-node Training (Recommended)

```bash
# ViT-Large (8 nodes x 8 GPUs = 64 GPUs)
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

### Pre-training Hyperparameters

| Parameter | Value | Ghi chú |
|-----------|-------|---------|
| **Effective Batch Size** | 4096 | `batch_size × nodes × gpus_per_node` |
| **Base Learning Rate** | 1.5e-4 | Actual LR = BLR × batch_size / 256 |
| **Weight Decay** | 0.05 | |
| **Mask Ratio** | 75% | Cao hơn BERT (15%) vì ảnh có redundancy cao |
| **Epochs** | 800 | |
| **Warmup Epochs** | 40 | |
| **Norm Pixel Loss** | `--norm_pix_loss` | Better representation learning |
| **Training Time** | ~42h | Trên 64 V100 GPUs (800 epochs) |

### Model Variants

```bash
# ViT-Base
--model mae_vit_base_patch16

# ViT-Large (recommended default)
--model mae_vit_large_patch16

# ViT-Huge
--model mae_vit_huge_patch14
```

---

## Chi Tiết Cấu Hình Fine-tuning

### Multi-node Training

```bash
# ViT-Base (4 nodes x 8 GPUs = 32 GPUs)
python submitit_finetune.py \
    --job_dir ${JOB_DIR} \
    --nodes 4 \
    --batch_size 32 \
    --model vit_base_patch16 \
    --finetune ${PRETRAIN_CHKPT} \
    --epochs 100 \
    --blr 5e-4 --layer_decay 0.65 \
    --weight_decay 0.05 --drop_path 0.1 --reprob 0.25 --mixup 0.8 --cutmix 1.0 \
    --dist_eval --data_path ${IMAGENET_DIR}
```

### Fine-tuning Hyperparameters theo Model

| Parameter | ViT-Base | ViT-Large | ViT-Huge |
|-----------|----------|-----------|----------|
| **Nodes** | 4 | 4 | 8 |
| **Batch Size (per GPU)** | 32 | 32 | 16 |
| **Effective Batch Size** | 1024 | 1024 | 1024 |
| **Epochs** | 100 | 50 | 50 |
| **Base LR** | 5e-4 | 1e-3 | 1e-3 |
| **Layer Decay** | 0.65 | 0.75 | 0.75 |
| **Drop Path** | 0.1 | 0.2 | 0.3 |
| **Weight Decay** | 0.05 | 0.05 | 0.05 |
| **Mixup** | 0.8 | 0.8 | 0.8 |
| **Cutmix** | 1.0 | 1.0 | 1.0 |
| **RandErase (reprob)** | 0.25 | 0.25 | 0.25 |
| **Training Time** | ~7h11m | ~8h52m | ~13h9m |
| **Accuracy (mean)** | 83.57% | 85.87% | 86.93% |

### Single-node Training

```bash
# ViT-Base (1 node x 8 GPUs với gradient accumulation)
OMP_NUM_THREADS=1 python -m torch.distributed.launch --nproc_per_node=8 main_finetune.py \
    --accum_iter 4 \
    --batch_size 32 \
    --model vit_base_patch16 \
    --finetune ${PRETRAIN_CHKPT} \
    --epochs 100 \
    --blr 5e-4 --layer_decay 0.65 \
    --weight_decay 0.05 --drop_path 0.1 --mixup 0.8 --cutmix 1.0 --reprob 0.25 \
    --dist_eval --data_path ${IMAGENET_DIR}
```

> **Note**: `--accum_iter 4` simulates 4 nodes để duy trì effective batch size = 1024

### Evaluation

```bash
# Evaluate ViT-Base
python main_finetune.py --eval --resume mae_finetuned_vit_base.pth --model vit_base_patch16 --batch_size 16 --data_path ${IMAGENET_DIR}
# Output: Acc@1 83.664 Acc@5 96.530

# Evaluate ViT-Large
python main_finetune.py --eval --resume mae_finetuned_vit_large.pth --model vit_large_patch16 --batch_size 16 --data_path ${IMAGENET_DIR}
# Output: Acc@1 85.952 Acc@5 97.570

# Evaluate ViT-Huge
python main_finetune.py --eval --resume mae_finetuned_vit_huge.pth --model vit_huge_patch14 --batch_size 16 --data_path ${IMAGENET_DIR}
# Output: Acc@1 86.928 Acc@5 98.088
```

---

## Chi Tiết Cấu Hình Linear Probing

```bash
# ViT-Base
python submitit_linprobe.py \
    --job_dir ${JOB_DIR} \
    --nodes 4 \
    --batch_size 512 \
    --model vit_base_patch16 --cls_token \
    --finetune ${PRETRAIN_CHKPT} \
    --epochs 90 \
    --blr 0.1 \
    --weight_decay 0.0 \
    --dist_eval --data_path ${IMAGENET_DIR}
```

### Linear Probing Hyperparameters

| Parameter | Value | Ghi chú |
|-----------|-------|---------|
| **Effective Batch Size** | 16384 | 512 × 4 × 8 |
| **Base Learning Rate** | 0.1 | |
| **Weight Decay** | 0.0 | |
| **Epochs** | 90 (Base), 50 (Large/Huge) | |
| **Training Time** | ~2h20m | 90 epochs trên 32 V100 GPUs |

### Linear Probing Results

| Model | Paper (TF/TPU) | This Repo (PT/GPU) |
|-------|----------------|-------------------|
| ViT-Base | 68.0 | 67.8 |
| ViT-Large | 75.8 | 76.0 |
| ViT-Huge | 76.6 | 77.2 |

---

## Technical Notes

> [!IMPORTANT]
> - Pre-trained models với `--norm_pix_loss` (1600 epochs) có fine-tuning hyperparameters khác với baseline unnormalized pixels
> - Sử dụng `--global_pool` cho fine-tuning thay vì `--cls_token` để tránh NaN với ViT-Huge trên GPU
> - PyTorch+GPU implementation sử dụng `torch.cuda.amp` (automatic mixed precision)

---

## Key Design Decisions

1. **Masking ratio 75%** — Cao hơn BERT (15%) vì ảnh có redundancy cao
2. **Asymmetric encoder-decoder** — Encoder lớn, decoder nhẹ (512-d, 8 blocks)
3. **No mask token in encoder** — Tránh distribution mismatch, tăng tốc 3-4×
4. **Pixel reconstruction** — Không cần tokenizer phức tạp
5. **Minimal augmentation** — Random crop + flip là đủ

---

## Tham Khảo

- **Official repo**: https://github.com/facebookresearch/mae
- **Paper**: https://arxiv.org/abs/2111.06377
- **Colab demo**: https://colab.research.google.com/github/facebookresearch/mae/blob/main/demo/mae_visualize.ipynb
