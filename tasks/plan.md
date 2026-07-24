# Implementation Plan: StyleMaster Image-to-Image

Nguồn: `SPEC.md` (thư mục gốc). Mục tiêu: một lệnh CLI (content image + style image + prompt) →
1 file PNG giữ bố cục content, đổi phong cách theo style — chạy dễ trên **host GPU thuê**.

## Overview

Code i2i đã được viết trên máy dev (macOS, không GPU/weights). Kế hoạch này đưa từ trạng thái
"code đã viết, chưa verify" → "chạy được và verify trên host". Rủi ro lớn nhất là **ControlNet với
`num_frames=1`** (nhánh V2V vốn cho 81 frame) — được đẩy lên sớm để *fail fast*.

## Architecture Decisions

- **Tái sử dụng nhánh V2V** (ControlNet gray-tile + StyleModel), đặt `num_frames=1` thay vì viết
  pipeline mới → không đụng code lõi (`diffsynth/`, `styleproj.py`), giảm rủi ro lệch checkpoint keys.
- **CLI flags thay vì CSV** → chạy 1 ảnh/lần gọn; wrapper `run_i2i.sh` override bằng env var.
- **Weights không commit** (đã gitignore) → host tự `download_ckpt.py`. Repo chỉ mang code.
- **Fallback `--num_frames`** giữ trong script để xử lý rủi ro temporal=1 mà không phải sửa code.

## Dependency Graph

```
[Slice A] commit + push code (máy dev)
      │
      ▼
[Slice B] host: pip install -e . + download_ckpt.py   ← foundation, chặn mọi thứ sau
      │
      ▼
[Slice C] host: chạy end-to-end 1 ảnh → ra PNG         ← RỦI RO CAO (num_frames=1)
      │                                                   fallback: --num_frames 5
      ▼
[Slice D] host: tinh chỉnh style/structure + ghi lại tham số tốt
```

Bottom-up: A → B → C → D. Không parallelize được (chuỗi phụ thuộc tuyến tính, 1 host).

## Task List

### Phase 1 — Foundation (máy dev)

#### Task 1: Commit + push code i2i
**Description:** Đưa toàn bộ setup lên repo để host pull về.
**Acceptance criteria:**
- [ ] `SPEC.md`, `.gitignore`, `inference_stylemaster_i2i.py`, `run_i2i.sh` được commit.
- [ ] `git status` sạch (trừ `CLAUDE.md` nếu chủ ý không commit).
- [ ] Không có file weights (`*.ckpt/*.safetensors/*.pth`) trong diff.
**Verification:**
- [ ] `git show --stat HEAD` chỉ liệt kê 4 file trên (+ CLAUDE.md nếu chọn).
- [ ] `git ls-files stylemaster-wan/checkpoints stylemaster-wan/models` rỗng.
- [ ] `git push` thành công.
**Dependencies:** None
**Files touched:** (git only) `SPEC.md`, `.gitignore`, `stylemaster-wan/inference_stylemaster_i2i.py`, `stylemaster-wan/run_i2i.sh`
**Scope:** XS

### Checkpoint: Foundation
- [ ] Code đã ở trên remote, host pull được, không kèm weights.

### Phase 2 — Môi trường host (host GPU)

#### Task 2: Dựng env + tải weights trên host
**Description:** Chuẩn bị runtime để inference chạy được.
**Acceptance criteria:**
- [ ] `git pull` lấy được code i2i.
- [ ] `conda activate stylemaster` + `pip install -e .` xong không lỗi.
- [ ] `python download_ckpt.py` tải đủ: `checkpoints/stylemaster.ckpt`, `checkpoints/controlnet.ckpt`,
      và 3 file trong `models/Wan-AI/Wan2.1-T2V-1.3B/`.
**Verification:**
- [ ] `ls checkpoints/*.ckpt` → 2 file.
- [ ] `ls models/Wan-AI/Wan2.1-T2V-1.3B/{diffusion_pytorch_model.safetensors,models_t5_umt5-xxl-enc-bf16.pth,Wan2.1_VAE.pth}` tồn tại.
- [ ] `nvidia-smi` thấy GPU; `python -c "import torch; print(torch.cuda.is_available())"` → True.
**Dependencies:** Task 1
**Files touched:** none (chỉ tải dữ liệu)
**Scope:** S

### Checkpoint: Môi trường
- [ ] Weights + GPU sẵn sàng. Nếu tải fail → dừng, xử lý mạng/HF token trước.

### Phase 3 — End-to-end (host GPU) — RỦI RO CAO

#### Task 3: Chạy i2i lần đầu ra PNG
**Description:** Chạy toàn bộ đường đi thật với 1 ảnh content + 1 ảnh style có sẵn trong repo.
**Acceptance criteria:**
- [ ] `bash run_i2i.sh <content.jpg> example_test_data/style_images/vangough.png "<prompt>"` chạy hết.
- [ ] Tạo được `./results/i2i/out.png` (ảnh hợp lệ, mở được).
- [ ] Không lỗi shape/dtype/checkpoint.
**Verification:**
- [ ] `python -c "from PIL import Image; Image.open('results/i2i/out.png').verify()"` không lỗi.
- [ ] Log in ra `Control video shape: (1, 3, 1, 480, 832)` và `Saved image -> ...`.
**Fallback (nếu lỗi ở temporal=1):**
- [ ] Chạy lại `NUM_FRAMES=5 bash run_i2i.sh ...`; nếu OK → ghi chú vào SPEC là num_frames=1 không hỗ trợ.
**Dependencies:** Task 2
**Files touched:** none (hoặc SPEC.md nếu phải ghi fallback)
**Scope:** S

#### Task 4: Xác nhận đúng nghĩa image-to-image
**Description:** Kiểm chứng output giữ bố cục content + mang style của style image (mắt thường).
**Acceptance criteria:**
- [ ] Output nhận ra bố cục/cấu trúc của content image.
- [ ] Output mang phong cách của style image (màu/nét/chất liệu).
**Verification:**
- [ ] So sánh trực quan content vs output (bố cục) và style vs output (phong cách).
- [ ] Chạy thêm 1 style khác (vd `ukiyoe.jpg`) → phong cách output đổi theo.
**Dependencies:** Task 3
**Files touched:** none
**Scope:** XS

### Checkpoint: Core (sau Task 3–4)
- [ ] i2i chạy end-to-end, PNG đúng nghĩa restyle. Nếu chỉ chạy được ở fallback → cập nhật SPEC/todo.

### Phase 4 — Tinh chỉnh (host GPU)

#### Task 5: Dò tham số style/structure tốt
**Description:** Tìm bộ tham số cân bằng giữa bám cấu trúc và độ đậm style.
**Acceptance criteria:**
- [ ] Thử ≥3 cấu hình qua env var: ví dụ `STYLE_CFG` ∈ {2,3,5}, `CN_SCALE`/`CN_END` ∈ {0.7,1.0}.
- [ ] Chọn được 1 bộ mặc định "đẹp" cho use case của bạn.
**Verification:**
- [ ] Tăng `STYLE_CFG` → style rõ đậm hơn; tăng `CN_SCALE`/`CN_END` → bám cấu trúc chặt hơn.
- [ ] Ghi bộ tham số tốt vào SPEC.md (mục Testing/Commands).
**Dependencies:** Task 4
**Files touched:** `SPEC.md`
**Scope:** S

### Checkpoint: Complete
- [ ] Tất cả acceptance criteria SPEC (T1–T4) đạt; tham số mặc định đã chốt; sẵn sàng dùng thường xuyên.

## Risks and Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| ControlNet không nhận `num_frames=1` (V2V vốn 81 frame) | High | Fallback `--num_frames 5` đã cài sẵn; verify sớm ở Task 3 |
| `download_ckpt.py` fail (HF/ModelScope, mạng, token) | High | Kiểm tra ở Task 2 trước khi chạy; login HF nếu cần |
| Ảnh khác 480×832 bị CenterCrop lệch tỉ lệ | Med | Giữ mặc định 480×832; cắt/chuẩn bị ảnh content trước |
| Thiếu ảnh content mẫu trong repo | Med | Bạn cung cấp 1 ảnh; hoặc trích 1 frame từ `example_test_data/girl.mp4` |
| VRAM không đủ cho Wan2.1-1.3B + ControlNet | Med | `tiled=True` đã bật; giảm `--steps`; dùng GPU ≥ ~12GB |

## Open Questions

- Ảnh content dùng để test là gì? (chưa có sẵn trong repo)
- Bộ tham số mặc định mong muốn thiên về "giữ cấu trúc" hay "style mạnh"?
- Có commit `CLAUDE.md` cùng lần push này không?
