"""
MAE ViT-Large — Masked Autoencoder với ViT-Large backbone
==========================================================
Tái tạo từ bài báo: "Masked Autoencoders Are Scalable Vision Learners" (He et al., 2022)

Kiến trúc chính:
- Asymmetric encoder-decoder
- Encoder: ViT-Large (24 blocks, 1024-dim, 16 heads)
  → Chỉ encode visible patches (hiệu quả, tránh distribution mismatch)
- Decoder: Lighter (8 blocks, 512-dim, 16 heads)
  → Decode full sequence với mask tokens
- Random masking 75% patches
- LayerNorm eps=1e-6 (khớp official: partial(nn.LayerNorm, eps=1e-6))

Checkpoint:
- Full (encoder+decoder, 398 keys): mae_visualize_vit_large.pth
  URL: https://dl.fbaipublicfiles.com/mae/visualize/mae_visualize_vit_large.pth
  → Dùng cho reconstruction / visualization
- Encoder-only (294 keys): mae_pretrain_vit_large.pth
  URL: https://dl.fbaipublicfiles.com/mae/pretrain/mae_pretrain_vit_large.pth
  → Dùng cho fine-tuning (decoder random → loss cao nếu dùng để reconstruct)
"""

import torch
import torch.nn as nn
import numpy as np

from .patch_embed import PatchEmbed
from .transformer import Block
from .pos_embed import get_2d_sincos_pos_embed
from .masking import random_masking


class MAEViTLarge(nn.Module):
    """
    Masked Autoencoder với ViT-Large backbone.
    
    Encoder: embed_dim=1024, depth=24, num_heads=16
    Decoder: embed_dim=512, depth=8, num_heads=16
    """
    
    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_chans=3,
        embed_dim=1024,       # ViT-Large encoder dimension
        depth=24,             # 24 encoder blocks
        num_heads=16,         # 16 attention heads
        decoder_embed_dim=512,
        decoder_depth=8,
        decoder_num_heads=16,
        mlp_ratio=4.0,
        norm_pix_loss=False,
    ):
        super().__init__()
        
        # ========== Encoder ==========
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches  # 196 cho 224×224 / 16×16
        
        # CLS token: learnable
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        
        # Positional embedding: cố định (sin-cos), không học
        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + 1, embed_dim),
            requires_grad=False
        )
        
        # LayerNorm factory: eps=1e-6 để khớp official (partial(nn.LayerNorm, eps=1e-6))
        norm_layer = lambda dim: nn.LayerNorm(dim, eps=1e-6)
        
        # 24 Transformer blocks
        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, norm_layer=norm_layer)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim, eps=1e-6)
        
        # ========== Decoder ==========
        # Linear projection từ encoder dim → decoder dim
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)
        
        # Mask token: learnable, thêm vào vị trí masked khi decode
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        
        # Decoder positional embedding
        self.decoder_pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + 1, decoder_embed_dim),
            requires_grad=False
        )
        
        # 8 Decoder Transformer blocks
        self.decoder_blocks = nn.ModuleList([
            Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, norm_layer=norm_layer)
            for _ in range(decoder_depth)
        ])
        self.decoder_norm = nn.LayerNorm(decoder_embed_dim, eps=1e-6)
        
        # Prediction head: decoder dim → pixel values
        self.decoder_pred = nn.Linear(
            decoder_embed_dim, patch_size ** 2 * in_chans, bias=True
        )
        
        # ========== Khác ==========
        self.patch_size = patch_size
        self.norm_pix_loss = norm_pix_loss
        
        # Khởi tạo trọng số
        self.initialize_weights()
    
    def initialize_weights(self):
        """Khởi tạo positional embeddings (sin-cos) và trọng số."""
        # Encoder positional embedding
        pos_embed = get_2d_sincos_pos_embed(
            self.pos_embed.shape[-1],
            int(self.patch_embed.num_patches ** 0.5),
            cls_token=True
        )
        self.pos_embed.data.copy_(
            torch.from_numpy(pos_embed).float().unsqueeze(0)
        )
        
        # Decoder positional embedding
        dec_pos_embed = get_2d_sincos_pos_embed(
            self.decoder_pos_embed.shape[-1],
            int(self.patch_embed.num_patches ** 0.5),
            cls_token=True
        )
        self.decoder_pos_embed.data.copy_(
            torch.from_numpy(dec_pos_embed).float().unsqueeze(0)
        )
        
        # Khởi tạo patch_embed.proj như nn.Linear (thay vì Conv2d)
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        
        # Khởi tạo tokens
        nn.init.normal_(self.cls_token, std=0.02)
        nn.init.normal_(self.mask_token, std=0.02)
        
        # Khởi tạo Linear và LayerNorm
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
    
    def patchify(self, imgs):
        """
        Chia ảnh thành patches.
        
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
        Ghép patches lại thành ảnh.
        
        x: (B, L, patch_size**2 * 3)
        returns: (B, 3, H, W)
        """
        p = self.patch_size
        h = w = int(x.shape[1] ** 0.5)
        assert h * w == x.shape[1]
        x = x.reshape(x.shape[0], h, w, p, p, 3)
        x = x.permute(0, 5, 1, 3, 2, 4)  # (B, 3, h, p, w, p)
        x = x.reshape(x.shape[0], 3, h * p, w * p)
        return x
    
    def forward_encoder(self, x, mask_ratio):
        """
        Encode chỉ visible patches.
        
        x: (B, 3, H, W)
        returns: (latent, mask, ids_restore)
        """
        # Patch embed: (B, 196, 1024)
        x = self.patch_embed(x)
        
        # Thêm positional embedding (không có CLS)
        x = x + self.pos_embed[:, 1:, :]
        
        # Random masking: (B, 49, 1024) khi mask_ratio=0.75
        x, mask, ids_restore = random_masking(x, mask_ratio)
        
        # Thêm CLS token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)  # (B, 50, 1024)
        
        # 24 Transformer blocks
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        
        return x, mask, ids_restore
    
    def forward_decoder(self, x, ids_restore):
        """
        Decode full sequence với mask tokens.
        
        x: encoder output (B, L_visible+1, 1024)
        returns: (B, L, patch_size**2 * 3)
        """
        # Project encoder dim → decoder dim
        x = self.decoder_embed(x)  # (B, L_visible+1, 512)
        
        # Thêm mask tokens vào vị trí masked
        mask_tokens = self.mask_token.repeat(
            x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1
        )
        
        # Ghép visible tokens + mask tokens (bỏ CLS)
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)
        # Unshuffle về thứ tự gốc
        x_ = torch.gather(
            x_, dim=1,
            index=ids_restore.unsqueeze(-1).expand(-1, -1, x.shape[2])
        )
        # Thêm CLS lại
        x = torch.cat([x[:, :1, :], x_], dim=1)
        
        # Thêm decoder positional embedding
        x = x + self.decoder_pos_embed
        
        # 8 Decoder Transformer blocks
        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)
        
        # Predict pixel values
        x = self.decoder_pred(x)
        
        # Bỏ CLS token
        x = x[:, 1:, :]
        
        return x
    
    def forward_loss(self, imgs, pred, mask):
        """
        MSE loss chỉ trên masked patches.
        
        imgs: (B, 3, H, W)
        pred: (B, L, patch_size**2 * 3)
        mask: (B, L), 0 = keep, 1 = remove
        """
        target = self.patchify(imgs)
        
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.e-6) ** 0.5
        
        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)  # (B, L) - loss mỗi patch
        
        # Trung bình loss chỉ trên masked patches
        loss = (loss * mask).sum() / mask.sum()
        return loss
    
    def forward(self, imgs, mask_ratio=0.75):
        """Full forward pass: encode → decode → loss."""
        latent, mask, ids_restore = self.forward_encoder(imgs, mask_ratio)
        pred = self.forward_decoder(latent, ids_restore)
        loss = self.forward_loss(imgs, pred, mask)
        return loss, pred, mask
    
    def load_pretrained(self, checkpoint_path, map_location='cpu'):
        """
        Load trọng số từ checkpoint pretrained (tự detect encoder-only hoặc full).
        
        Facebook MAE cung cấp 2 loại checkpoint:
        - Encoder-only (294 keys): dl.fbaipublicfiles.com/mae/pretrain/mae_pretrain_vit_large.pth
          → Dùng cho fine-tuning, decoder giữ nguyên random init
        - Full encoder+decoder (398 keys): dl.fbaipublicfiles.com/mae/visualize/mae_visualize_vit_large.pth
          → Dùng cho reconstruction/visualization, load toàn bộ model
        
        Args:
            checkpoint_path: Đường dẫn tới file .pth
            map_location: Device để load (mặc định 'cpu')
        
        Returns:
            (missing_keys, unexpected_keys): Tuple các keys missing/unexpected
        """
        checkpoint = torch.load(checkpoint_path, map_location=map_location, weights_only=True)
        
        # Checkpoint có format {'model': state_dict}
        if 'model' in checkpoint:
            state_dict = checkpoint['model']
        else:
            state_dict = checkpoint
        
        # Detect loại checkpoint
        has_decoder = any(k.startswith('decoder') or k.startswith('mask_token') 
                         for k in state_dict.keys())
        
        if has_decoder:
            # Full checkpoint → load strict
            result = self.load_state_dict(state_dict, strict=True)
            print(f"✅ Loaded FULL checkpoint ({len(state_dict)} keys — encoder + decoder)")
        else:
            # Encoder-only → load non-strict
            result = self.load_state_dict(state_dict, strict=False)
            print(f"⚠️  Loaded ENCODER-ONLY checkpoint ({len(state_dict)} keys)")
            print(f"   Decoder weights giữ nguyên random init → loss sẽ cao!")
            print(f"   Để reconstruction tốt, dùng full checkpoint: mae_visualize_vit_large.pth")
        
        # Báo cáo
        missing = result.missing_keys
        unexpected = result.unexpected_keys
        
        if missing:
            decoder_missing = [k for k in missing if any(
                k.startswith(p) for p in ['decoder_', 'mask_token']
            )]
            other_missing = [k for k in missing if k not in decoder_missing]
            
            if decoder_missing and not has_decoder:
                print(f"   ({len(decoder_missing)} decoder keys missing — bình thường cho encoder-only checkpoint)")
            if other_missing:
                print(f"❌ {len(other_missing)} encoder keys MISSING (lỗi!):")
                for k in other_missing:
                    print(f"   - {k}")
        
        if unexpected:
            print(f"❌ {len(unexpected)} unexpected keys:")
            for k in unexpected:
                print(f"   - {k}")
        
        return missing, unexpected
    
    # Backward-compatible alias
    def load_pretrained_encoder(self, checkpoint_path, map_location='cpu'):
        """Alias cho load_pretrained() — backward compatibility."""
        return self.load_pretrained(checkpoint_path, map_location)


def mae_vit_large_patch16(**kwargs):
    """
    Factory function tạo MAE ViT-Large với config chuẩn từ bài báo.
    
    Config: embed_dim=1024, depth=24, num_heads=16
            decoder: embed_dim=512, depth=8, num_heads=16
    """
    model = MAEViTLarge(
        img_size=224,
        patch_size=16,
        in_chans=3,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        decoder_embed_dim=512,
        decoder_depth=8,
        decoder_num_heads=16,
        **kwargs
    )
    return model
