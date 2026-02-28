---
phase: testing
title: Testing Strategy
description: Chiến lược test cho việc tái tạo MAE model variants
---

# Testing Strategy

## Test Coverage Goals

**What level of testing do we aim for?**

- Unit test coverage: 100% cho `encoder.py`, `decoder.py`, và tất cả model variants
- Integration test: Forward pass end-to-end cho mỗi variant
- Verification test: So sánh output với official implementation
- Shape tests: Mọi intermediate tensor shape phải đúng

## Unit Tests

**What individual components need testing?**

### MAEEncoder (`mae/encoder.py`)

- [ ] Test output shape với ViT-Base config (embed_dim=768, depth=12, heads=12)
- [ ] Test output shape với ViT-Large config (embed_dim=1024, depth=24, heads=16)
- [ ] Test output shape với ViT-Huge config (embed_dim=1280, depth=32, heads=16)
- [ ] Test encoder preserves batch dimension
- [ ] Test encoder with different sequence lengths (196 vs 256 patches)

### MAEDecoder (`mae/decoder.py`)

- [ ] Test output shape: pred = (B, num_patches, patch_size²×3)
- [ ] Test decoder handles different num_patches (196 for p=16, 256 for p=14)
- [ ] Test mask token insertion produces correct sequence length
- [ ] Test decoder positional embedding shape matches num_patches

### MaskedAutoencoder (`mae/mae.py` — refactored)

- [ ] Test forward returns (loss, pred, mask) tuple
- [ ] Test loss is scalar
- [ ] Test pred shape = (B, num_patches, patch_size²×3)
- [ ] Test mask shape = (B, num_patches) with correct ratio
- [ ] Test forward_encoder output shape
- [ ] Test forward_decoder output shape
- [ ] Test patchify/unpatchify roundtrip
- [ ] Test norm_pix_loss=True vs False affects loss value

### Model Variants (`models/`)

- [ ] `mae_vit_base_patch16()` — instantiate, forward, check shapes
- [ ] `mae_vit_large_patch16()` — instantiate, forward, check shapes
- [ ] `mae_vit_huge_patch14()` — instantiate, forward, check shapes (patch_size=14!)
- [ ] All factory functions accept \*\*kwargs overrides
- [ ] Parameter count within expected range

## Integration Tests

**How do we test component interactions?**

- [ ] **End-to-end forward pass**: `imgs → loss, pred, mask` without errors
- [ ] **Encoder ↔ Decoder**: Encoder output feeds correctly into Decoder
- [ ] **Masking consistency**: `mask.sum() / mask.numel() ≈ mask_ratio`
- [ ] **Gradient flow**: `loss.backward()` completes without error
- [ ] **Different mask ratios**: Test with 0.5, 0.75, 0.9

## Verification Tests (vs Official)

**Compare our implementation against official MAE**

- [ ] **Parameter count comparison** (all 3 variants):

| Variant      | Expected Params | Tolerance |
| ------------ | --------------- | --------- |
| ViT-Base/16  | ~111M           | ±0.5M     |
| ViT-Large/16 | ~330M           | ±1M       |
| ViT-Huge/14  | ~657M           | ±2M       |

- [ ] **Output shape comparison**: Same shapes for same input
- [ ] **(Optional) Weight transfer**: Load official weights → compare forward output (diff < 1e-5)

## Test Data

**What data do we use for testing?**

```python
# Standard test fixtures
def sample_images(batch_size=2, img_size=224):
    return torch.randn(batch_size, 3, img_size, img_size)

# For ViT-Huge (patch_size=14), img_size=224 still works
# 224 / 14 = 16 → 256 patches
```

- **Test inputs**: Random tensors `torch.randn(B, 3, 224, 224)`
- **Batch sizes**: 1 (minimum), 2 (default), 4 (stress)
- **No real images needed** for shape/correctness tests

## Test Reporting & Coverage

### Expected Test File Structure

```
tests/
├── test_encoder_decoder.py    # Unit tests for Encoder, Decoder
├── test_model_variants.py     # All 3 variants factory + forward
├── test_mae.py                # Updated: refactored MAE tests
└── comparision.py             # Updated: compare all variants vs official
```

### Running Tests

```bash
# Run all tests
python -m pytest tests/ -v

# Run specific test
python -m pytest tests/test_model_variants.py -v

# Run with output
python tests/test_mae.py
```

## Manual Testing

**What requires human validation?**

- [ ] Visual inspection: Instantiate each model, print summary
- [ ] Memory usage: Check GPU memory for each variant (if GPU available)
- [ ] Import check: `from models import mae_vit_base_patch16` works from project root

## Performance Testing

| Test                    | Variant   | Metric    | Target            |
| ----------------------- | --------- | --------- | ----------------- |
| Forward time (CPU, B=1) | ViT-Base  | Wall time | < 5s              |
| Forward time (CPU, B=1) | ViT-Large | Wall time | < 15s             |
| Forward time (CPU, B=1) | ViT-Huge  | Wall time | < 30s             |
| Backward pass           | ViT-Base  | Completes | No OOM on 8GB GPU |

---

## 🤖 Model Evaluation (AI/ML Projects)

**How do we measure model correctness?**

| Metric               | Target                         | Method            |
| -------------------- | ------------------------------ | ----------------- |
| Param count match    | ±0.5% vs official              | `sum(p.numel())`  |
| Output shape correct | Exact match                    | Assert shapes     |
| Loss is finite       | `not torch.isnan(loss)`        | Forward pass      |
| Gradient exists      | All params have `.grad`        | `loss.backward()` |
| Roundtrip patchify   | `unpatchify(patchify(x)) == x` | Tensor comparison |

### Test Cases Summary

```python
# Example test structure
class TestMAEEncoder:
    def test_base_config(self):
        enc = MAEEncoder(embed_dim=768, depth=12, num_heads=12)
        x = torch.randn(2, 50, 768)  # 49 visible + 1 CLS
        out = enc(x)
        assert out.shape == (2, 50, 768)

    def test_huge_config(self):
        enc = MAEEncoder(embed_dim=1280, depth=32, num_heads=16)
        x = torch.randn(1, 65, 1280)  # 64 visible + 1 CLS
        out = enc(x)
        assert out.shape == (1, 65, 1280)

class TestModelVariants:
    def test_mae_vit_base(self):
        model = mae_vit_base_patch16()
        imgs = torch.randn(2, 3, 224, 224)
        loss, pred, mask = model(imgs)
        assert pred.shape == (2, 196, 768)  # 196 patches, 16²×3
        assert mask.shape == (2, 196)

    def test_mae_vit_huge(self):
        model = mae_vit_huge_patch14()
        imgs = torch.randn(1, 3, 224, 224)
        loss, pred, mask = model(imgs)
        assert pred.shape == (1, 256, 588)  # 256 patches, 14²×3
        assert mask.shape == (1, 256)

    def test_param_counts(self):
        base = mae_vit_base_patch16()
        large = mae_vit_large_patch16()
        huge = mae_vit_huge_patch14()
        assert 100e6 < sum(p.numel() for p in base.parameters()) < 120e6
        assert 320e6 < sum(p.numel() for p in large.parameters()) < 340e6
        assert 640e6 < sum(p.numel() for p in huge.parameters()) < 670e6
```
