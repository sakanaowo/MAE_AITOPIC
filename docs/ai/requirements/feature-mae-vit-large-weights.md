---
phase: requirements
title: Tái tạo MAE ViT-Large từ bài báo gốc
description: Tái tạo model MAE với kiến trúc ViT-Large để load trọng số pretrained
---

# Yêu cầu & Phân tích bài toán

## Mô tả vấn đề

- Team đã có file trọng số pretrained từ bài báo gốc MAE (He et al., 2022)
- Cần **tái tạo lại** model MAE ViT-Large trong folder mới `mae/large/` (tách biệt khỏi folder `mae/` hiện tại)
- Model phải khớp chính xác với kiến trúc checkpoint để load trọng số thành công

## Mục tiêu

**Mục tiêu chính:**

- Tái tạo đầy đủ kiến trúc MAE ViT-Large trong `mae/large/`
- Đảm bảo `state_dict()` keys khớp hoàn toàn với checkpoint
- Cung cấp method `load_pretrained()` để load trọng số (tự detect encoder-only hoặc full)
- LayerNorm eps=1e-6 khớp official implementation

**Ngoài phạm vi:**

- Không train MAE từ đầu (đã có trọng số pretrained)
- Không chỉnh sửa code trong folder `mae/` hiện tại

## User Stories

- Là researcher, tôi muốn load trọng số MAE ViT-Large pretrained để **reconstruct/visualize** (cần full checkpoint)
- Là researcher, tôi muốn load encoder-only weights để **fine-tune** trên downstream tasks
- Là developer, tôi muốn code MAE ViT-Large được tổ chức rõ ràng trong folder riêng `mae/large/`
- Là developer, tôi muốn nhận thông báo rõ ràng khi load trọng số — biết loại checkpoint nào đang dùng

## Tiêu chí thành công

1. `model.load_pretrained("data/mae_visualize_vit_large.pth")` load full 398 keys → loss thấp (~1.0 random, thấp hơn trên ảnh thật)
2. `model.load_pretrained("data/mae_pretrain_vit_large.pth")` load 294 encoder keys → cảnh báo rõ decoder random
3. Forward pass chạy đúng sau khi load trọng số
4. Không có shape mismatch hay unexpected keys
5. LayerNorm eps=1e-6 ở tất cả layers (encoder + decoder)

## Ràng buộc & Giả định

### Checkpoint files

Facebook MAE cung cấp **2 loại checkpoint** (đều ViT-Large):

| File                          | Keys                  | URL                                        | Mục đích                        |
| ----------------------------- | --------------------- | ------------------------------------------ | ------------------------------- |
| `mae_visualize_vit_large.pth` | 398 (encoder+decoder) | `dl.fbaipublicfiles.com/mae/visualize/...` | Reconstruction / visualization  |
| `mae_pretrain_vit_large.pth`  | 294 (encoder-only)    | `dl.fbaipublicfiles.com/mae/pretrain/...`  | Fine-tuning (không cần decoder) |

**QUAN TRỌNG:** README gốc liệt kê pretrain checkpoint dưới mục "Fine-tuning with pre-trained checkpoints" — đây là checkpoint **để fine-tune**, KHÔNG phải full model. Nếu cần reconstruction, phải dùng visualize checkpoint.

### Kiến trúc ViT-Large theo bài báo

- `embed_dim=1024`, `depth=24`, `num_heads=16`
- `patch_size=16`, `mlp_ratio=4.0`
- Decoder: `embed_dim=512`, `depth=8`, `num_heads=16`
- **LayerNorm eps=1e-6** (official: `partial(nn.LayerNorm, eps=1e-6)`)
- `norm_pix_loss=False` mặc định trong code (official pretrain dùng `--norm_pix_loss` True)

### Cấu trúc keys trong checkpoint

**Encoder (294 keys):**

| Component                | Số keys | Shape ví dụ                               |
| ------------------------ | ------- | ----------------------------------------- |
| `cls_token`              | 1       | `[1, 1, 1024]`                            |
| `pos_embed`              | 1       | `[1, 197, 1024]`                          |
| `patch_embed.proj`       | 2       | weight `[1024, 3, 16, 16]`, bias `[1024]` |
| `blocks.{0-23}.*`        | 264     | 11 keys/block × 24 blocks                 |
| `blocks.*.attn.qkv.bias` | 24      | `[3072]` mỗi block                        |
| `norm`                   | 2       | weight + bias `[1024]`                    |

**Decoder (104 keys — chỉ có trong visualize checkpoint):**

| Component                | Số keys | Shape ví dụ                        |
| ------------------------ | ------- | ---------------------------------- |
| `mask_token`             | 1       | `[1, 1, 512]`                      |
| `decoder_embed`          | 2       | weight `[512, 1024]`, bias `[512]` |
| `decoder_pos_embed`      | 1       | `[1, 197, 512]`                    |
| `decoder_blocks.{0-7}.*` | 96      | 12 keys/block × 8 blocks           |
| `decoder_norm`           | 2       | weight + bias `[512]`              |
| `decoder_pred`           | 2       | weight `[768, 512]`, bias `[768]`  |

### Loss benchmarks (random noise input, norm_pix_loss=False)

| Checkpoint              | Loss  | Ghi chú                   |
| ----------------------- | ----- | ------------------------- |
| Full (visualize)        | ~1.0  | Encoder + decoder trained |
| Encoder-only (pretrain) | ~1.87 | Decoder random → loss cao |
| Random init             | ~1.74 | Toàn bộ random            |

Code mới đặt trong `mae/large/`, không dùng lại folder `mae/` hiện tại.
