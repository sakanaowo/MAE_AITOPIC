"""
Masked Autoencoder (MAE)
=======================
Key Architecture:
- Asymmetric encoder-decoder
- Encoder: Large ViT, chỉ encode visible patches
- Decoder: Light (8 blocks, 512-dim), decode full sequence
- No mask token trong encoder (efficient + avoid distribution mismatch)

Refactored: Compose từ MAEEncoder + MAEDecoder modules.
"""

import torch
import torch.nn as nn
import numpy as np
from .encoder import MAEEncoder
from .decoder import MAEDecoder


class MaskedAutoencoder(nn.Module):
    """
    Masked Autoencoder với ViT backbone.
    
    Compose từ MAEEncoder + MAEDecoder, hỗ trợ mọi variant
    (Base/Large/Huge) qua config parameters.
    
    Args:
        img_size: Kích thước ảnh (224)
        patch_size: Kích thước patch (16 hoặc 14)
        in_chans: Số kênh ảnh (3)
        embed_dim: Chiều embedding encoder
        depth: Số transformer blocks encoder
        num_heads: Số attention heads encoder
        decoder_embed_dim: Chiều embedding decoder
        decoder_depth: Số transformer blocks decoder
        decoder_num_heads: Số attention heads decoder
        mlp_ratio: Tỉ lệ expansion MLP
        norm_pix_loss: Normalize pixel values trong loss
    """
    
    def __init__(
        self,
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
        norm_pix_loss=False,
    ):
        super().__init__()
        
        norm_layer = lambda d: nn.LayerNorm(d, eps=1e-6)
        num_patches = (img_size // patch_size) ** 2
        
        self.encoder = MAEEncoder(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            norm_layer=norm_layer,
        )
        
        self.decoder = MAEDecoder(
            num_patches=num_patches,
            patch_size=patch_size,
            in_chans=in_chans,
            encoder_embed_dim=embed_dim,
            decoder_embed_dim=decoder_embed_dim,
            decoder_depth=decoder_depth,
            decoder_num_heads=decoder_num_heads,
            mlp_ratio=mlp_ratio,
            norm_layer=norm_layer,
        )
        
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.norm_pix_loss = norm_pix_loss
    
    # === Proxy attributes cho backward compatibility ===
    @property
    def patch_embed(self):
        return self.encoder.patch_embed
    
    @property
    def cls_token(self):
        return self.encoder.cls_token
    
    @property
    def pos_embed(self):
        return self.encoder.pos_embed
    
    @property
    def blocks(self):
        return self.encoder.blocks
    
    @property
    def norm(self):
        return self.encoder.norm
    
    @property
    def decoder_embed(self):
        return self.decoder.decoder_embed
    
    @property
    def mask_token(self):
        return self.decoder.mask_token
    
    @property
    def decoder_pos_embed(self):
        return self.decoder.decoder_pos_embed
    
    @property
    def decoder_blocks(self):
        return self.decoder.decoder_blocks
    
    @property
    def decoder_norm(self):
        return self.decoder.decoder_norm
    
    @property
    def decoder_pred(self):
        return self.decoder.decoder_pred
    
    def patchify(self, imgs):
        """imgs: (B, 3, H, W) → (B, L, patch_size**2 * 3)"""
        p = self.patch_size
        B, C, H, W = imgs.shape
        assert H == W and H % p == 0
        h = w = H // p
        x = imgs.reshape(B, C, h, p, w, p)
        x = x.permute(0, 2, 4, 3, 5, 1)
        x = x.reshape(B, h * w, p * p * C)
        return x
    
    def unpatchify(self, x):
        """x: (B, L, patch_size**2 * 3) → (B, 3, H, W)"""
        p = self.patch_size
        h = w = int(x.shape[1] ** 0.5)
        assert h * w == x.shape[1]
        x = x.reshape(x.shape[0], h, w, p, p, 3)
        x = x.permute(0, 5, 1, 3, 2, 4)
        x = x.reshape(x.shape[0], 3, h * p, w * p)
        return x
    
    def forward_encoder(self, x, mask_ratio):
        return self.encoder(x, mask_ratio)
    
    def forward_decoder(self, x, ids_restore):
        return self.decoder(x, ids_restore)
    
    def forward_loss(self, imgs, pred, mask):
        """MSE loss chỉ trên masked patches."""
        target = self.patchify(imgs)
        
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.e-6) ** 0.5
        
        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)
        loss = (loss * mask).sum() / mask.sum()
        return loss
    
    def forward(self, imgs, mask_ratio=0.75):
        latent, mask, ids_restore = self.forward_encoder(imgs, mask_ratio)
        pred = self.forward_decoder(latent, ids_restore)
        loss = self.forward_loss(imgs, pred, mask)
        return loss, pred, mask
    
    def load_pretrained(self, checkpoint_path, map_location='cpu'):
        """
        Load pretrained weights (auto-detect encoder-only hoặc full).
        
        Returns:
            (missing_keys, unexpected_keys)
        """
        checkpoint = torch.load(checkpoint_path, map_location=map_location, weights_only=True)
        
        if 'model' in checkpoint:
            state_dict = checkpoint['model']
        else:
            state_dict = checkpoint
        
        # Remap flat keys → nested encoder/decoder keys
        remapped = {}
        for k, v in state_dict.items():
            if k.startswith('decoder_') or k == 'mask_token':
                # decoder keys → decoder.xxx
                remapped[f'decoder.{k}'] = v
            elif k in ('patch_embed.proj.weight', 'patch_embed.proj.bias',
                       'cls_token', 'pos_embed', 'norm.weight', 'norm.bias') or \
                 k.startswith('blocks.'):
                # encoder keys → encoder.xxx
                remapped[f'encoder.{k}'] = v
            else:
                remapped[k] = v
        
        has_decoder = any(k.startswith('decoder.decoder_') or k == 'decoder.mask_token'
                         for k in remapped.keys())
        
        if has_decoder:
            result = self.load_state_dict(remapped, strict=True)
            print(f"✅ Loaded FULL checkpoint ({len(state_dict)} keys)")
        else:
            result = self.load_state_dict(remapped, strict=False)
            print(f"⚠️  Loaded ENCODER-ONLY checkpoint ({len(state_dict)} keys)")
        
        return result.missing_keys, result.unexpected_keys
    
    # Backward-compatible alias
    load_pretrained_encoder = load_pretrained


# Backward compatibility alias
MAE = MaskedAutoencoder