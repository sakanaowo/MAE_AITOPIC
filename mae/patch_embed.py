"""
preprocessing things for MAE:
Patch Embedding: split image into patches and embed them
IN: Image (B,3,224,224)
OUT: Patch Embedding (B, 196, 768)
==============================
num_patches = (img_size // patch_size) ** 2
each patch: (patch_size*patch_size*3, ) pixed to 768-dim vector
"""

import torch
import torch.nn as nn

class PatchEmbed(nn.Module):
    """Split image into patches and embed them."""
    
    def __init__(self, img_size=224,patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        
        # Conv2d witch kernel_size=patch_size and stride=patch_size
        self.proj = nn.Conv2d(
            in_chans, embed_dim, 
            kernel_size=patch_size,
            stride=patch_size
        )
        
    def forward(self, x):
        # x: (B, embed_dim, H/patch_size, W/patch_size)
        x = self.proj(x)
        
        # flatten: (B, embed_dim, num_patches)
        x = x.flatten(2)
        
        # transpose: (B, num_patches, embed_dim)
        x = x.transpose(1, 2)
        
        return x

# if __name__ == "__main__":
#     pe=PatchEmbed()
#     x=torch.randn(2,3,224,224)
#     out=pe(x)
#     assert out.shape==(2,196,768), f"Expected shape (2,196,768), but got {out.shape}"
#     print("PatchEmbed output shape:", out.shape)