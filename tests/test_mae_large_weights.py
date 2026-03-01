"""
Test MAE ViT-Large — Kiểm thử load trọng số pretrained
========================================================
1. Tạo model MAEViTLarge
2. So sánh state_dict keys với checkpoint
3. Load trọng số và chạy forward pass
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
from mae.large import MAEViTLarge, mae_vit_large_patch16


def test_key_matching():
    """So sánh state_dict keys giữa model và checkpoint."""
    print("=" * 60)
    print("TEST 1: So sánh state_dict keys")
    print("=" * 60)
    
    # Tạo model
    model = mae_vit_large_patch16()
    model_keys = set(model.state_dict().keys())
    
    # Load checkpoint
    ckpt_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'mae_pretrain_vit_large.pth')
    checkpoint = torch.load(ckpt_path, map_location='cpu')
    ckpt_keys = set(checkpoint['model'].keys())
    
    # Keys trong checkpoint nhưng KHÔNG trong model
    extra_in_ckpt = ckpt_keys - model_keys
    if extra_in_ckpt:
        print(f"❌ {len(extra_in_ckpt)} keys trong checkpoint KHÔNG có trong model:")
        for k in sorted(extra_in_ckpt):
            print(f"   - {k}")
    else:
        print(f"✅ Tất cả {len(ckpt_keys)} checkpoint keys đều có trong model")
    
    # Keys trong model nhưng KHÔNG trong checkpoint (phải toàn decoder)
    extra_in_model = model_keys - ckpt_keys
    decoder_keys = [k for k in extra_in_model if any(
        k.startswith(p) for p in ['decoder_', 'mask_token']
    )]
    encoder_missing = [k for k in extra_in_model if k not in decoder_keys]
    
    print(f"📋 {len(decoder_keys)} decoder keys chỉ có trong model (bình thường)")
    if encoder_missing:
        print(f"❌ {len(encoder_missing)} encoder keys chỉ có trong model (LỖI!):")
        for k in sorted(encoder_missing):
            print(f"   - {k}")
    
    # Kiểm tra shape
    ckpt_sd = checkpoint['model']
    model_sd = model.state_dict()
    shape_mismatch = []
    for k in ckpt_keys & model_keys:
        if ckpt_sd[k].shape != model_sd[k].shape:
            shape_mismatch.append((k, ckpt_sd[k].shape, model_sd[k].shape))
    
    if shape_mismatch:
        print(f"❌ {len(shape_mismatch)} shape mismatches:")
        for k, cs, ms in shape_mismatch:
            print(f"   - {k}: ckpt={cs} vs model={ms}")
    else:
        print(f"✅ Tất cả shapes khớp")
    
    assert len(extra_in_ckpt) == 0, "Có keys trong checkpoint không có trong model!"
    assert len(encoder_missing) == 0, "Có encoder keys thiếu!"
    assert len(shape_mismatch) == 0, "Có shape mismatch!"
    print("✅ TEST 1 PASSED\n")


def test_load_weights():
    """Load trọng số pretrained vào model."""
    print("=" * 60)
    print("TEST 2: Load trọng số pretrained")
    print("=" * 60)
    
    model = mae_vit_large_patch16()
    ckpt_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'mae_pretrain_vit_large.pth')
    
    missing, unexpected = model.load_pretrained_encoder(ckpt_path)
    
    # Kiểm tra không có unexpected keys
    assert len(unexpected) == 0, f"Unexpected keys: {unexpected}"
    
    # Kiểm tra missing keys toàn decoder
    for k in missing:
        assert any(k.startswith(p) for p in ['decoder_', 'mask_token']), \
            f"Encoder key missing: {k}"
    
    print("✅ TEST 2 PASSED\n")


def test_forward_pass():
    """Chạy forward pass với trọng số đã load."""
    print("=" * 60)
    print("TEST 3: Forward pass với trọng số pretrained")
    print("=" * 60)
    
    model = mae_vit_large_patch16()
    ckpt_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'mae_pretrain_vit_large.pth')
    model.load_pretrained_encoder(ckpt_path)
    model.eval()
    
    # Forward pass
    x = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        loss, pred, mask = model(x, mask_ratio=0.75)
    
    print(f"Input shape:  {x.shape}")
    print(f"Pred shape:   {pred.shape}")
    print(f"Mask shape:   {mask.shape}")
    print(f"Loss:         {loss.item():.4f}")
    print(f"Mask ratio:   {mask.sum() / mask.numel():.2%}")
    
    # Kiểm tra shapes
    assert pred.shape == (2, 196, 768), f"Pred shape sai: {pred.shape}"
    assert mask.shape == (2, 196), f"Mask shape sai: {mask.shape}"
    assert not torch.isnan(loss), "Loss là NaN!"
    assert not torch.isinf(loss), "Loss là Inf!"
    
    # Encoder only test
    with torch.no_grad():
        latent, mask2, ids_restore = model.forward_encoder(x, mask_ratio=0.75)
    print(f"Encoder output: {latent.shape}")
    assert latent.shape[2] == 1024, f"Encoder dim sai: {latent.shape[2]}"
    
    print("✅ TEST 3 PASSED\n")


def test_param_count():
    """Kiểm tra số parameters."""
    print("=" * 60)
    print("TEST 4: Parameter count")
    print("=" * 60)
    
    model = mae_vit_large_patch16()
    
    total_params = sum(p.numel() for p in model.parameters())
    encoder_params = sum(
        p.numel() for n, p in model.named_parameters()
        if not any(n.startswith(prefix) for prefix in ['decoder_', 'mask_token'])
    )
    decoder_params = total_params - encoder_params
    
    print(f"Total params:   {total_params / 1e6:.1f}M")
    print(f"Encoder params: {encoder_params / 1e6:.1f}M")
    print(f"Decoder params: {decoder_params / 1e6:.1f}M")
    
    # ViT-Large encoder ~ 304M params
    assert encoder_params > 300e6, f"Encoder params quá ít: {encoder_params / 1e6:.1f}M"
    
    print("✅ TEST 4 PASSED\n")


if __name__ == "__main__":
    print("🚀 Bắt đầu kiểm thử MAE ViT-Large\n")
    
    test_key_matching()
    test_load_weights()
    test_forward_pass()
    test_param_count()
    
    print("=" * 60)
    print("🎉 TẤT CẢ TESTS PASSED!")
    print("=" * 60)
