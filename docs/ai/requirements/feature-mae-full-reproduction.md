---
phase: requirements
title: Requirements & Problem Understanding
description: Tái tạo toàn bộ model variants từ bài báo MAE (Masked Autoencoders Are Scalable Vision Learners)
---

# Requirements & Problem Understanding

## Problem Statement

**What problem are we solving?**

- Cần tái tạo (reproduce) toàn bộ kiến trúc model từ bài báo **"Masked Autoencoders Are Scalable Vision Learners"** (He et al., 2021, arXiv:2111.06377) — bao gồm cả 3 biến thể: **ViT-Base**, **ViT-Large**, **ViT-Huge**.
- Hiện tại code đã implement rời rạc trong `mae/mae.py` (ViT-Base) và `mae/mae_large.py` (ViT-Large), nhưng:
  - Chưa có ViT-Huge
  - `encoder.py` và `decoder.py` còn **rỗng** — chưa tách module encoder/decoder riêng biệt
  - Thiếu factory functions thống nhất để tạo model
  - Chưa có folder trình bày chuyên biệt cho từng model variant
  - Thiếu documentation rõ ràng mapping giữa paper → code
- Đối tượng: sinh viên, nhà nghiên cứu muốn hiểu sâu kiến trúc MAE thông qua việc tự implement từng thành phần.

## Goals & Objectives

**What do we want to achieve?**

### Primary Goals

1. **Tách module rõ ràng**: Encoder, Decoder là các module độc lập, có thể tái sử dụng
2. **Implement đầy đủ 3 model variants** từ paper:
   - `mae_vit_base_patch16` — ViT-B/16 (embed_dim=768, depth=12, heads=12, ~111M params)
   - `mae_vit_large_patch16` — ViT-L/16 (embed_dim=1024, depth=24, heads=16, ~330M params)
   - `mae_vit_huge_patch14` — ViT-H/14 (embed_dim=1280, depth=32, heads=16, **patch_size=14**, ~657M params)
3. **Tạo folder `models/` riêng biệt** trình bày tất cả model variants kèm documentation
4. **Verify output** nhất quán với official implementation (`official_mae/models_mae.py`)

### Secondary Goals

- Mỗi module có docstring giải thích quan hệ với paper (section, equation)
- Factory functions và model registry thống nhất
- Notebook demo cho từng variant

### Non-goals (explicitly out of scope)

- Training pipeline (pre-training, fine-tuning) — sẽ là feature riêng
- Data loading / augmentation pipeline
- Distributed training utilities
- Fine-tuning head (linear probing, end-to-end fine-tune)

## User Stories & Use Cases

**How will users interact with the solution?**

1. **As a researcher**, I want to instantiate any MAE variant with a single function call so that I can quickly experiment:

   ```python
   from models import mae_vit_base_patch16, mae_vit_large_patch16, mae_vit_huge_patch14
   model = mae_vit_large_patch16()
   loss, pred, mask = model(images)
   ```

2. **As a student**, I want to understand từng module riêng biệt (Encoder, Decoder, Attention, PatchEmbed...) so that I can trace data flow through the architecture:

   ```python
   from mae.encoder import MAEEncoder
   from mae.decoder import MAEDecoder
   encoder = MAEEncoder(embed_dim=768, depth=12, num_heads=12)
   ```

3. **As a developer**, I want a `models/` folder with dedicated files cho từng variant so that I can compare hyperparameters side-by-side:

   ```
   models/
   ├── README.md              # Overview & comparison table
   ├── mae_vit_base.py        # ViT-Base variant
   ├── mae_vit_large.py       # ViT-Large variant
   ├── mae_vit_huge.py        # ViT-Huge variant
   └── __init__.py            # Factory & registry
   ```

4. **As a reviewer**, I want to verify our implementation matches the official one by comparing parameter counts and output tensors.

## Success Criteria

**How will we know when we're done?**

| Criterion                                           | Measurement                                                                  |
| --------------------------------------------------- | ---------------------------------------------------------------------------- |
| All 3 variants instantiate without error            | Unit test pass                                                               |
| Parameter counts match official ±0.1%               | `mae_vit_base`: ~111M, `mae_vit_large`: ~330M, `mae_vit_huge`: ~657M         |
| Output shape correct                                | loss scalar, pred `(B, num_patches, patch_size²×3)`, mask `(B, num_patches)` |
| `encoder.py` and `decoder.py` fully implemented     | Non-empty, tested independently                                              |
| `models/` folder with all variants + README         | Files exist and documented                                                   |
| Forward pass output matches official (same weights) | Tensor diff < 1e-5                                                           |

## Constraints & Assumptions

**What limitations do we need to work within?**

### Technical Constraints

- ViT-Huge (`patch_size=14`) tạo ra **256 patches** (16×16 grid) thay vì 196 — cần đảm bảo code xử lý đúng
- ViT-Huge yêu cầu ≥32GB VRAM — testing trên CPU hoặc cần mixed precision
- Tất cả module phải không phụ thuộc `timm` (tự implement from scratch)

### Assumptions

- Decoder luôn dùng cấu hình `embed_dim=512, depth=8, heads=16` cho mọi variant (theo paper)
- Sử dụng sinusoidal positional embedding (fixed, not learned)
- Input image size cố định 224×224

## Questions & Open Items

**What do we still need to clarify?**

- [x] ViT-Huge dùng `patch_size=14` → cần verify `num_patches = (224/14)² = 256`
- [ ] Có cần hỗ trợ variable image size không? (Tạm thời: No, fix 224×224)
- [ ] Naming convention: dùng `MAEViTBase` hay `mae_vit_base_patch16` function?
  - **Decision**: Cả hai — Class name PascalCase, factory function snake_case (theo official)

---

## 🤖 Data Requirements (AI/ML Projects)

**What data do we need?**

- **Pre-training**: ImageNet-1K (1.28M images, ~150GB) — chưa cần ở bước này
- **Testing/Demo**: Random tensors hoặc single image cho visualization
- **Verification**: Official pretrained weights để compare output

| Data              | Purpose             | Required Now?       |
| ----------------- | ------------------- | ------------------- |
| Random tensors    | Unit tests          | Yes                 |
| Single test image | Visualization demo  | Nice-to-have        |
| Official weights  | Output verification | Nice-to-have        |
| ImageNet-1K       | Pre-training        | No (future feature) |

## Model Variants (from Paper)

| Config                   | ViT-Base/16 | ViT-Large/16 | ViT-Huge/14 |
| ------------------------ | ----------- | ------------ | ----------- |
| `patch_size`             | 16          | 16           | **14**      |
| `embed_dim`              | 768         | 1024         | 1280        |
| `depth` (encoder blocks) | 12          | 24           | 32          |
| `num_heads` (encoder)    | 12          | 16           | 16          |
| `decoder_embed_dim`      | 512         | 512          | 512         |
| `decoder_depth`          | 8           | 8            | 8           |
| `decoder_num_heads`      | 16          | 16           | 16          |
| `mlp_ratio`              | 4.0         | 4.0          | 4.0         |
| `num_patches`            | 196         | 196          | **256**     |
| Approx. params           | ~111M       | ~330M        | ~657M       |
