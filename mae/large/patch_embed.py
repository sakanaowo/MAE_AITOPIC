"""
Patch Embedding cho MAE ViT-Large
===================================
Chia ảnh thành patches và embed:
- Input: (B, 3, 224, 224)
- Output: (B, 196, 1024) cho ViT-Large
- Dùng Conv2d với kernel_size=stride=patch_size
- CÓ bias (khớp checkpoint patch_embed.proj.bias)
"""

import torch.nn as nn


class PatchEmbed(nn.Module):
    """Chia ảnh thành patches và project vào embedding space."""
    
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=1024):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2  # 196 cho 224×224 với 16×16
        
        # Conv2d: kernel_size=patch_size, stride=patch_size
        # bias=True (mặc định) → khớp với checkpoint patch_embed.proj.bias
        self.proj = nn.Conv2d(
            in_chans, embed_dim,
            kernel_size=patch_size,
            stride=patch_size
        )
    
    def forward(self, x):
        # x: (B, C, H, W) → (B, embed_dim, H/p, W/p)
        x = self.proj(x)
        # flatten spatial dims: (B, embed_dim, num_patches)
        x = x.flatten(2)
        # transpose: (B, num_patches, embed_dim)
        x = x.transpose(1, 2)
        return x
