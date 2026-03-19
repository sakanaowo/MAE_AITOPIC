"""
Test all MAE variants — Base, Large, Huge
==========================================
1. Module tests (Encoder, Decoder independently)
2. Forward pass tests for all 3 variants
3. Weight loading tests (Large + Huge checkpoints)
4. Parameter count verification
5. Backward compatibility test
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import torch.nn as nn


def test_encoder_module():
    """Test MAEEncoder independently."""
    print("=" * 60)
    print("TEST 1: MAEEncoder module")
    print("=" * 60)
    
    from mae.encoder import MAEEncoder
    
    # ViT-Base config
    encoder = MAEEncoder(img_size=224, patch_size=16, embed_dim=768, depth=12, num_heads=12)
    x = torch.randn(2, 3, 224, 224)
    latent, mask, ids_restore = encoder(x, mask_ratio=0.75)
    
    assert latent.shape == (2, 50, 768), f"Encoder output shape sai: {latent.shape}"
    assert mask.shape == (2, 196), f"Mask shape sai: {mask.shape}"
    assert ids_restore.shape == (2, 196), f"ids_restore shape sai: {ids_restore.shape}"
    print(f"  Base encoder: {latent.shape} ✓")
    
    # ViT-Huge config (patch_size=14 → 256 patches)
    encoder_huge = MAEEncoder(img_size=224, patch_size=14, embed_dim=1280, depth=32, num_heads=16)
    latent_h, mask_h, ids_h = encoder_huge(x, mask_ratio=0.75)
    
    assert latent_h.shape == (2, 65, 1280), f"Huge encoder shape sai: {latent_h.shape}"
    assert mask_h.shape == (2, 256), f"Huge mask shape sai: {mask_h.shape}"
    print(f"  Huge encoder: {latent_h.shape} ✓")
    
    print("✅ TEST 1 PASSED\n")


def test_decoder_module():
    """Test MAEDecoder independently."""
    print("=" * 60)
    print("TEST 2: MAEDecoder module")
    print("=" * 60)
    
    from mae.decoder import MAEDecoder
    
    # ViT-Base decoder
    decoder = MAEDecoder(
        num_patches=196, patch_size=16, encoder_embed_dim=768,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16
    )
    
    # Simulate encoder output: CLS + 49 visible patches
    x = torch.randn(2, 50, 768)
    ids_restore = torch.randperm(196).unsqueeze(0).expand(2, -1)
    pred = decoder(x, ids_restore)
    
    assert pred.shape == (2, 196, 768), f"Decoder output shape sai: {pred.shape}"
    print(f"  Base decoder: {pred.shape} ✓")
    
    # ViT-Huge decoder (256 patches, patch_size=14)
    decoder_huge = MAEDecoder(
        num_patches=256, patch_size=14, encoder_embed_dim=1280,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16
    )
    
    x_h = torch.randn(2, 65, 1280)
    ids_h = torch.randperm(256).unsqueeze(0).expand(2, -1)
    pred_h = decoder_huge(x_h, ids_h)
    
    assert pred_h.shape == (2, 256, 588), f"Huge decoder shape sai: {pred_h.shape}"
    print(f"  Huge decoder: {pred_h.shape} ✓")
    
    print("✅ TEST 2 PASSED\n")


def test_mae_base():
    """Test MAE ViT-Base forward pass."""
    print("=" * 60)
    print("TEST 3: MAE ViT-Base forward pass")
    print("=" * 60)
    
    from models import mae_vit_base_patch16
    
    model = mae_vit_base_patch16()
    model.eval()
    
    x = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        loss, pred, mask = model(x, mask_ratio=0.75)
    
    assert pred.shape == (2, 196, 768), f"Pred shape sai: {pred.shape}"
    assert mask.shape == (2, 196), f"Mask shape sai: {mask.shape}"
    assert not torch.isnan(loss), "Loss is NaN!"
    
    params = sum(p.numel() for p in model.parameters())
    print(f"  Params: {params/1e6:.1f}M")
    print(f"  Loss: {loss.item():.4f}")
    print(f"  Pred: {pred.shape}, Mask: {mask.shape}")
    
    # Check modular structure
    assert hasattr(model, 'encoder'), "Missing encoder attribute"
    assert hasattr(model, 'decoder'), "Missing decoder attribute"
    
    print("✅ TEST 3 PASSED\n")


def test_mae_large():
    """Test MAE ViT-Large forward pass."""
    print("=" * 60)
    print("TEST 4: MAE ViT-Large forward pass")
    print("=" * 60)
    
    from models import mae_vit_large_patch16
    
    model = mae_vit_large_patch16()
    model.eval()
    
    x = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        loss, pred, mask = model(x, mask_ratio=0.75)
    
    assert pred.shape == (2, 196, 768), f"Pred shape sai: {pred.shape}"
    assert mask.shape == (2, 196), f"Mask shape sai: {mask.shape}"
    assert not torch.isnan(loss), "Loss is NaN!"
    
    # Encoder output check
    with torch.no_grad():
        latent, _, _ = model.forward_encoder(x, mask_ratio=0.75)
    assert latent.shape[2] == 1024, f"Encoder dim sai: {latent.shape[2]}"
    
    params = sum(p.numel() for p in model.parameters())
    enc_params = sum(p.numel() for p in model.encoder.parameters())
    print(f"  Total params: {params/1e6:.1f}M (encoder: {enc_params/1e6:.1f}M)")
    print(f"  Loss: {loss.item():.4f}")
    assert enc_params > 300e6, f"Encoder params quá ít: {enc_params/1e6:.1f}M"
    
    print("✅ TEST 4 PASSED\n")


def test_mae_huge():
    """Test MAE ViT-Huge forward pass."""
    print("=" * 60)
    print("TEST 5: MAE ViT-Huge forward pass")
    print("=" * 60)
    
    from models import mae_vit_huge_patch14
    
    model = mae_vit_huge_patch14()
    model.eval()
    
    x = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        loss, pred, mask = model(x, mask_ratio=0.75)
    
    # patch_size=14: 16*16=256 patches, each = 14*14*3 = 588 pixels
    assert pred.shape == (2, 256, 588), f"Pred shape sai: {pred.shape}"
    assert mask.shape == (2, 256), f"Mask shape sai: {mask.shape}"
    assert not torch.isnan(loss), "Loss is NaN!"
    
    # Encoder output: 25% of 256 = 64 visible + 1 CLS = 65 tokens
    with torch.no_grad():
        latent, _, _ = model.forward_encoder(x, mask_ratio=0.75)
    assert latent.shape == (2, 65, 1280), f"Encoder output shape sai: {latent.shape}"
    
    params = sum(p.numel() for p in model.parameters())
    enc_params = sum(p.numel() for p in model.encoder.parameters())
    print(f"  Total params: {params/1e6:.1f}M (encoder: {enc_params/1e6:.1f}M)")
    print(f"  Loss: {loss.item():.4f}")
    print(f"  Patches: 256, Pred: {pred.shape}")
    assert params > 600e6, f"Total params quá ít: {params/1e6:.1f}M"
    
    print("✅ TEST 5 PASSED\n")


def test_load_large_weights():
    """Test load pretrained weights cho ViT-Large."""
    print("=" * 60)
    print("TEST 6: Load ViT-Large pretrained weights")
    print("=" * 60)
    
    ckpt_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'mae_pretrain_vit_large.pth')
    if not os.path.exists(ckpt_path):
        print("  ⏭️  SKIPPED (checkpoint not found)")
        return
    
    from models import mae_vit_large_patch16
    
    model = mae_vit_large_patch16()
    missing, unexpected = model.load_pretrained(ckpt_path)
    
    assert len(unexpected) == 0, f"Unexpected keys: {unexpected}"
    for k in missing:
        assert 'decoder' in k or 'mask_token' in k, f"Encoder key missing: {k}"
    
    # Forward pass with loaded weights
    model.eval()
    x = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        loss, pred, mask = model(x, mask_ratio=0.75)
    
    assert not torch.isnan(loss), "Loss is NaN after loading weights!"
    print(f"  Loss with pretrained encoder: {loss.item():.4f}")
    print(f"  Missing keys: {len(missing)} (all decoder — expected)")
    
    print("✅ TEST 6 PASSED\n")


def test_load_huge_weights():
    """Test load pretrained weights cho ViT-Huge."""
    print("=" * 60)
    print("TEST 7: Load ViT-Huge pretrained weights")
    print("=" * 60)
    
    ckpt_path = os.path.join(os.path.dirname(__file__), '..', 
                             'checkpoints', 'official_checkpoints', 'mae_pretrain_vit_huge.pth')
    if not os.path.exists(ckpt_path):
        print("  ⏭️  SKIPPED (checkpoint not found)")
        return
    
    from models import mae_vit_huge_patch14
    
    model = mae_vit_huge_patch14()
    
    # Load and check keys
    checkpoint = torch.load(ckpt_path, map_location='cpu', weights_only=True)
    ckpt_keys = set(checkpoint['model'].keys())
    print(f"  Checkpoint keys: {len(ckpt_keys)}")
    
    missing, unexpected = model.load_pretrained(ckpt_path)
    
    assert len(unexpected) == 0, f"Unexpected keys: {unexpected}"
    for k in missing:
        assert 'decoder' in k or 'mask_token' in k, f"Encoder key missing: {k}"
    
    # Forward pass
    model.eval()
    x = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        loss, pred, mask = model(x, mask_ratio=0.75)
    
    assert pred.shape == (2, 256, 588), f"Pred shape sai: {pred.shape}"
    assert not torch.isnan(loss), "Loss is NaN after loading weights!"
    print(f"  Loss with pretrained encoder: {loss.item():.4f}")
    print(f"  Pred shape: {pred.shape}")
    print(f"  Missing keys: {len(missing)} (all decoder — expected)")
    
    print("✅ TEST 7 PASSED\n")


def test_huge_gpu_inference():
    """Test ViT-Huge inference trên GPU (nếu có)."""
    print("=" * 60)
    print("TEST 8: ViT-Huge GPU inference")
    print("=" * 60)
    
    if not torch.cuda.is_available():
        print("  ⏭️  SKIPPED (no CUDA)")
        return
    
    from models import mae_vit_huge_patch14
    
    model = mae_vit_huge_patch14()
    
    # Load weights nếu có
    ckpt_path = os.path.join(os.path.dirname(__file__), '..',
                             'checkpoints', 'official_checkpoints', 'mae_pretrain_vit_huge.pth')
    if os.path.exists(ckpt_path):
        model.load_pretrained(ckpt_path)
        print("  (loaded pretrained weights)")
    
    model = model.cuda().eval()
    
    torch.cuda.reset_peak_memory_stats()
    
    # Inference
    x = torch.randn(1, 3, 224, 224, device='cuda')
    with torch.no_grad():
        loss, pred, mask = model(x, mask_ratio=0.75)
    
    peak_mem = torch.cuda.max_memory_allocated() / 1e9
    
    assert pred.shape == (1, 256, 588), f"Pred shape sai: {pred.shape}"
    assert not torch.isnan(loss), "Loss is NaN!"
    
    print(f"  Loss: {loss.item():.4f}")
    print(f"  Peak VRAM: {peak_mem:.2f} GB")
    print(f"  Pred shape: {pred.shape}")
    
    # Test larger batch
    for bs in [2, 4, 8]:
        torch.cuda.reset_peak_memory_stats()
        with torch.no_grad():
            x = torch.randn(bs, 3, 224, 224, device='cuda')
            loss, pred, mask = model(x, mask_ratio=0.75)
        peak = torch.cuda.max_memory_allocated() / 1e9
        print(f"  bs={bs}: peak {peak:.2f} GB, loss={loss.item():.4f}")
    
    print("✅ TEST 8 PASSED\n")


def test_backward_compat():
    """Test backward compatibility: MAE alias still works."""
    print("=" * 60)
    print("TEST 9: Backward compatibility")
    print("=" * 60)
    
    from mae.mae import MAE
    
    model = MAE(img_size=224, patch_size=16, embed_dim=768, depth=12, num_heads=12)
    x = torch.randn(2, 3, 224, 224)
    loss, pred, mask = model(x, mask_ratio=0.75)
    
    assert pred.shape == (2, 196, 768)
    
    # Proxy attributes
    assert model.patch_embed is model.encoder.patch_embed
    assert model.blocks is model.encoder.blocks
    assert model.mask_token is model.decoder.mask_token
    
    print(f"  MAE alias works, pred: {pred.shape}")
    print("✅ TEST 9 PASSED\n")


def test_param_counts():
    """Verify parameter counts match expected values."""
    print("=" * 60)
    print("TEST 10: Parameter count verification")
    print("=" * 60)
    
    from models import mae_vit_base_patch16, mae_vit_large_patch16, mae_vit_huge_patch14
    
    configs = [
        ("ViT-Base",  mae_vit_base_patch16,  100e6, 120e6),
        ("ViT-Large", mae_vit_large_patch16, 320e6, 340e6),
        ("ViT-Huge",  mae_vit_huge_patch14,  650e6, 670e6),
    ]
    
    for name, factory, min_p, max_p in configs:
        model = factory()
        total = sum(p.numel() for p in model.parameters())
        enc = sum(p.numel() for p in model.encoder.parameters())
        dec = sum(p.numel() for p in model.decoder.parameters())
        print(f"  {name}: {total/1e6:.1f}M total (enc={enc/1e6:.1f}M, dec={dec/1e6:.1f}M)")
        assert min_p < total < max_p, f"{name} params {total/1e6:.1f}M out of range [{min_p/1e6:.0f}M, {max_p/1e6:.0f}M]"
    
    print("✅ TEST 10 PASSED\n")


if __name__ == "__main__":
    print("🚀 MAE Full Test Suite\n")
    
    test_encoder_module()
    test_decoder_module()
    test_mae_base()
    test_mae_large()
    test_mae_huge()
    test_load_large_weights()
    test_load_huge_weights()
    test_huge_gpu_inference()
    test_backward_compat()
    test_param_counts()
    
    print("=" * 60)
    print("🎉 ALL TESTS PASSED!")
    print("=" * 60)
