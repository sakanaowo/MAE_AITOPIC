import sys
import torch
sys.path.append(".")

from mae.mae import MAE

def test_mae():
    model = MAE(
        img_size=224,
        patch_size=16,
        in_chans=3,
        embed_dim=768,
        depth=12,
        num_heads=12,
        decoder_embed_dim=512,
        decoder_depth=8,
    )
    
    x = torch.randn(2, 3, 224, 224)  # (B, C, H, W)
    loss, pred, mask = model(x, mask_ratio=0.75)
    
    print(f"Input shape: {x.shape}")
    print("Loss:", loss.item())
    print("Pred shape:", pred.shape)
    print("Mask shape:", mask.shape)
    print(f"Mask ratio: {mask.sum()/mask.numel():.2%}")
    
    params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {params/1e6:.2f}M")
    
    latent, mask, ids_restore = model.forward_encoder(x, mask_ratio=0.75)
    print("Encoder output:", latent.shape)    
    
if __name__ == "__main__":
    test_mae()