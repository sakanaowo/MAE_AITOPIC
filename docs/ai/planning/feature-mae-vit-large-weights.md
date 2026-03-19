---
phase: planning
title: Kế hoạch triển khai MAE ViT-Large
description: Phân chia công việc và timeline
---

# Kế hoạch & Phân chia công việc

## Milestones

- [ ] M1: Tái tạo tất cả module trong `mae/large/`
- [ ] M2: Verify key matching với checkpoint
- [ ] M3: Load trọng số và test forward pass thành công

## Phân chia công việc

### Giai đoạn 1: Tái tạo các module cơ bản
- [ ] Task 1.1: Tạo `mae/large/pos_embed.py` — Sinusoidal positional embedding
- [ ] Task 1.2: Tạo `mae/large/masking.py` — Random masking utility
- [ ] Task 1.3: Tạo `mae/large/patch_embed.py` — Patch embedding (Conv2d)
- [ ] Task 1.4: Tạo `mae/large/attention.py` — Multi-head attention (qkv_bias=True)
- [ ] Task 1.5: Tạo `mae/large/transformer.py` — Transformer Block

### Giai đoạn 2: Tái tạo model MAE chính
- [ ] Task 2.1: Tạo `mae/large/mae_vit_large.py` — Class MAEViTLarge đầy đủ
- [ ] Task 2.2: Thêm method `load_pretrained_encoder()`
- [ ] Task 2.3: Tạo `mae/large/__init__.py` — Exports

### Giai đoạn 3: Kiểm thử
- [ ] Task 3.1: Tạo `tests/test_mae_large_weights.py`
- [ ] Task 3.2: So sánh state_dict keys model vs checkpoint
- [ ] Task 3.3: Load trọng số và chạy forward pass
- [ ] Task 3.4: Báo cáo kết quả

## Phụ thuộc

- File trọng số `data/mae_pretrain_vit_large.pth` phải tồn tại
- PyTorch, numpy phải được cài đặt
- Các module phải được tạo theo thứ tự: pos_embed → masking → patch_embed → attention → transformer → mae_vit_large

## Rủi ro & Biện pháp

| Rủi ro | Ảnh hưởng | Biện pháp |
|--------|-----------|-----------|
| Key mismatch | Load trọng số thất bại | So sánh key chi tiết trước khi load |
| Shape mismatch | Runtime error | Verify shape từng tensor |
