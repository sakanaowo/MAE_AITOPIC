import torch
import torch.nn as nn
import numpy as np

from .patch_embed import PatchEmbed
from .transformer import Block
from .pos_embed import get_2d_sincos_pos_embed
from .masking import random_masking

class MAELarge(nn.Module):
    def __init__(
        self, 
        img_size=224,patch_size=16,in_chans=3,
        embed_dim=1024,depth=24,num_heads=16,
        decoder_embed_dim=512,decoder_depth=8,decoder_num_heads=16, mlp_ratio=4.0, norm_pix_loss=False
    ):
        super().__init__()
        
        # == Encoder ==
        self.patch_embed = PatchEmbed(img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches # (img_size // patch_size) ** 2 -> 196 for 224x224 with 16x16 patches
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(
            torch.zeros(1,num_patches + 1, decoder_embed_dim),
            requires_grad=False
        )
        
        # Transformer encoder blocks
        self.blocks = nn.ModuleList([
            Block(embed_dim,num_heads,mlp_ratio) for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        
        # == Decoder ==
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        self.decoder_pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + 1, decoder_embed_dim),
            requires_grad=False
        )
        
        self.decoder_blocks = nn.ModuleList([
            Block(decoder_embed_dim, decoder_num_heads, mlp_ratio) for _ in range(decoder_depth)
        ])
        self.decoder_norm = nn.LayerNorm(decoder_embed_dim)
        self.decoder_pred = nn.Linear(decoder_embed_dim, patch_size**2 * in_chans)
        
        self.patch_size = patch_size
        self.norm_pix_loss = norm_pix_loss
        
        self.initialize_weights()
        
    def initialize_weights(self):
        pos_embed = get_2d_sincos_pos_embed(
            self.pos_embed.shape[-1],
            int((self.patch_embed.num_patches)**0.5),
            cls_token=True
        )
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        dec_pos_embed = get_2d_sincos_pos_embed(
            self.decoder_pos_embed.shape[-1],
            int((self.patch_embed.num_patches)**0.5),
            cls_token=True
        )
        
        self.decoder_pos_embed.data.copy_(torch.from_numpy(dec_pos_embed).float().unsqueeze(0))
        
        nn.init.normal_(self.cls_token, std=0.02)
        nn.init.normal_(self.mask_token, std=0.02)
        
        self.apply(self._init_weights)
        
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
            
    def patchify(self,imgs):
        p=self.patch_size
        B,C,H,W=imgs.shape
        h=w=H//p
        x=imgs.reshape(B,C,h,p,w,p)
        x = x.permute(0,5,1,3,2,4).reshape(x.shape[0],3, h*p, p*p*w)
        return x 
    
    def unpatchify(self, x):
        p = self.patch_size
        h=w=int(x.shape[1]**0.5)
        x=x.reshape(x.shape[0], h, w, p, p, 3)
        x = x.permute(0,5,1,3,2,4).reshape(x.shape[0], 3, h*p, w*p)
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
    
    
def mae_large_vit_large_patch16(**kwargs):
    model = MAELarge(
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