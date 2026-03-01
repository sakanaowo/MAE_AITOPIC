---
phase: testing
title: Chiến lược kiểm thử MAE ViT-Large
description: Test plan cho việc tái tạo model và load trọng số
---

# Chiến lược kiểm thử

## Mục tiêu test

- 100% flow load trọng số được test
- Xác nhận key matching giữa model và checkpoint
- Forward pass hoạt động đúng sau khi load

## Unit Tests

### Load trọng số (`tests/test_mae_large_weights.py`)
- [x] Test: Tất cả 294 checkpoint keys được load vào model ✅
- [x] Test: Không có unexpected keys từ `load_state_dict` ✅
- [x] Test: Chỉ decoder keys được báo "missing" (104 keys) ✅
- [x] Test: Forward pass cho output hợp lệ sau khi load ✅
- [x] Test: Số parameters đúng — Encoder: 303.3M ✅

### Kiến trúc model (`tests/test_mae_large_weights.py`)
- [x] Test: Model tạo được với đúng config ViT-Large ✅
- [x] Test: Output shape đúng cho input `(2, 3, 224, 224)` → Pred `(2, 196, 768)` ✅
- [x] Test: Encoder output shape `(2, 50, 1024)` (49 visible + 1 CLS khi mask_ratio=0.75) ✅
- [x] Test: Decoder output shape `(2, 196, 768)` (196 patches × 16×16×3) ✅

## Kiểm thử thủ công

- [x] Chạy test script và xác nhận tất cả 4 tests passed ✅
- [x] So sánh parameter count: Encoder 303.3M + Decoder 26.2M = Total 329.5M ✅

## Lệnh chạy test

```bash
cd /home/sakana/Code/PTIT/AITOPIC/MAE_AITOPIC
/home/sakana/miniconda3/envs/MAE/bin/python3 tests/test_mae_large_weights.py
```

## Kết quả test (2026-03-01)

```
🚀 Bắt đầu kiểm thử MAE ViT-Large

============================================================
TEST 1: So sánh state_dict keys
============================================================
✅ Tất cả 294 checkpoint keys đều có trong model
📋 104 decoder keys chỉ có trong model (bình thường)
✅ Tất cả shapes khớp
✅ TEST 1 PASSED

============================================================
TEST 2: Load trọng số pretrained
============================================================
✅ Loaded 294 keys từ checkpoint
⚠️  104 decoder keys missing (bình thường - checkpoint chỉ có encoder)
✅ TEST 2 PASSED

============================================================
TEST 3: Forward pass với trọng số pretrained
============================================================
✅ Loaded 294 keys từ checkpoint
⚠️  104 decoder keys missing (bình thường - checkpoint chỉ có encoder)
Input shape:  torch.Size([2, 3, 224, 224])
Pred shape:   torch.Size([2, 196, 768])
Mask shape:   torch.Size([2, 196])
Loss:         1.9183
Mask ratio:   75.00%
Encoder output: torch.Size([2, 50, 1024])
✅ TEST 3 PASSED

============================================================
TEST 4: Parameter count
============================================================
Total params:   329.5M
Encoder params: 303.3M
Decoder params: 26.2M
✅ TEST 4 PASSED

============================================================
🎉 TẤT CẢ TESTS PASSED!
============================================================
```

---

## 🤖 Đánh giá Model

| Metric | Mục tiêu | Kết quả | Trạng thái |
|--------|----------|---------|------------|
| Keys loaded | 294/294 | 294/294 | ✅ PASSED |
| Unexpected keys | 0 | 0 | ✅ PASSED |
| Shape mismatches | 0 | 0 | ✅ PASSED |
| Forward pass | Không lỗi | Loss = 1.9183 | ✅ PASSED |
| Encoder params | ~304M | 303.3M | ✅ PASSED |
| Decoder params | ~26M | 26.2M | ✅ PASSED |
| Pred shape | (B, 196, 768) | (2, 196, 768) | ✅ PASSED |
| Mask ratio | 75% | 75.00% | ✅ PASSED |

## Đối chiếu với Tiêu chí thành công (Requirements)

| # | Tiêu chí | Kết quả | Trạng thái |
|---|----------|---------|------------|
| 1 | `load_pretrained_encoder()` load được 294 encoder keys | 294/294 keys loaded | ✅ ĐẠT |
| 2 | Chỉ decoder keys được báo "missing" | 104 decoder keys missing | ✅ ĐẠT |
| 3 | Forward pass chạy đúng sau khi load | Loss=1.9183, output shapes đúng | ✅ ĐẠT |
| 4 | Không có shape mismatch hay unexpected keys | 0 mismatches, 0 unexpected | ✅ ĐẠT |
