# Spec: StyleMaster Image-to-Image (restyle 1 ảnh)

## Objective

Cho phép chạy **thử nghiệm image-to-image** dễ dàng trên StyleMaster: đưa vào **1 ảnh nội dung**
(content image) + **1 ảnh style** + 1 prompt text → tạo ra **1 ảnh mới** giữ nguyên bố cục/cấu trúc
của ảnh nội dung nhưng mang phong cách của ảnh style, và lưu ra file **PNG**.

- **User:** chính bạn (nghiên cứu/thử nghiệm), chạy trên máy có **GPU CUDA**.
- **Cơ chế:** dùng lại **nhánh V2V** của repo (ControlNet "gray-tile" giữ cấu trúc + StyleModel bơm
  style), nhưng đặt `num_frames = 1` để pipeline chỉ sinh **một khung hình** thay vì video.
- **Cách chạy:** một script mới nhận tham số qua **CLI flags** (không cần sửa CSV), chạy 1 ảnh/lần.
- **Thành công khi:** một lệnh `python inference_stylemaster_i2i.py --content_image ... --style_image ...
  --prompt ...` sinh ra 1 file `.png` đúng style, không lỗi, không cần chỉnh code.

### Deployment flow (quan trọng)
Code được **setup sẵn trên máy này (macOS, KHÔNG có GPU/weights)** → **commit + push** lên repo →
**pull về host GPU đã thuê** → trên host: `pip install -e .` (nếu chưa) → `python download_ckpt.py`
→ chạy script. Vì vậy **không chạy/verify được ở máy dev**; kiểm chứng thực tế diễn ra trên host.

### Acceptance criteria (cụ thể, kiểm chứng được)
1. Chạy được script chỉ với 3 flag bắt buộc: `--content_image`, `--style_image`, `--prompt`.
2. Output là **1 file PNG** tại đường dẫn `--output` (mặc định `./results/i2i/output.png`).
3. Ảnh output **giữ bố cục** của content image (nhờ ControlNet) và **đổi phong cách** theo style image.
4. Các tham số style/structure điều chỉnh được qua CLI: `--style_cfg_scale`, `--cfg_scale`,
   `--controlnet_conditioning_scale`, `--controlnet_guidance_end`, `--seed`, `--steps`, `--height`, `--width`.
5. Không sửa hành vi model lõi (styleproj.py, diffsynth/) — chỉ thêm 1 script mới + 1 run script.

## Tech Stack

- Python 3.10, PyTorch (bf16), CUDA GPU. Conda env `stylemaster` (theo CLAUDE.md).
- Model: **Wan2.1-T2V-1.3B** + checkpoints StyleMaster (`stylemaster.ckpt`) + ControlNet (`controlnet.ckpt`).
- Package `diffsynth` vendored, cài editable (`pip install -e .`).
- Tái sử dụng: `styleproj.py` (Processor/StyleModel), `WanVideoStyleMasterPipeline`, `WanControlNet`.

## Commands

```bash
# --- Trên máy dev (macOS): chỉ commit & push, KHÔNG chạy ---
git add SPEC.md .gitignore \
        stylemaster-wan/inference_stylemaster_i2i.py stylemaster-wan/run_i2i.sh
git commit -m "Add image-to-image inference script + spec"
git push

# ===================== Các bước dưới đây chạy TRÊN HOST GPU =====================
# 0) Pull code + môi trường (chạy 1 lần, trong stylemaster-wan/)
git pull
cd stylemaster-wan
conda create --name stylemaster python=3.10 && conda activate stylemaster
pip install -e .

# 1) Tải weights (BẮT BUỘC — checkpoints/ và models/ được gitignore, không có sẵn sau khi pull)
python download_ckpt.py
# -> checkpoints/stylemaster.ckpt, checkpoints/controlnet.ckpt
# -> models/Wan-AI/Wan2.1-T2V-1.3B/{diffusion_pytorch_model.safetensors, models_t5_umt5-xxl-enc-bf16.pth, Wan2.1_VAE.pth}

# 2) Chạy image-to-image (1 ảnh) — cách gọn nhất:
python inference_stylemaster_i2i.py \
  --content_image path/to/content.jpg \
  --style_image   example_test_data/style_images/vangough.png \
  --prompt        "a young child walking through a field of tall grass and poppies" \
  --output        ./results/i2i/out.png

# 3) Hoặc dùng wrapper đã gói sẵn tham số:
bash run_i2i.sh path/to/content.jpg example_test_data/style_images/vangough.png "your prompt"
```

Lint/format/test: **không có** trong repo (không ruff/black/pytest/CI). "Build" = `pip install -e .`.

## Project Structure

```
StyleMaster/
├── SPEC.md                              # tài liệu này (đã commit)
└── stylemaster-wan/
    ├── inference_stylemaster.py         # T2V gốc (không đụng)
    ├── inference_stylemaster_v2v.py     # V2V gốc (tham chiếu, không đụng)
    ├── inference_stylemaster_i2i.py     # MỚI: script image-to-image (CLI flags)
    ├── run_i2i.sh                       # MỚI: wrapper bash gói tham số mặc định
    ├── styleproj.py                     # StyleModel/Processor (không đụng)
    ├── diffsynth/                       # pipeline + model (không đụng)
    ├── checkpoints/                     # weights StyleMaster + ControlNet (tải về)
    ├── models/Wan-AI/Wan2.1-T2V-1.3B/   # weights Wan2.1 (tải về)
    └── results/i2i/                     # MỚI: nơi lưu PNG output
```

## Code Style

Bám sát style hiện có của `inference_stylemaster_v2v.py`: argparse, PIL, torch bf16, comment ngắn.
Điểm khác cốt lõi vs V2V (minh hoạ):

```python
NUM_FRAMES = 1  # image-to-image: chỉ sinh 1 khung hình

# content -> tín hiệu control "gray-tile" 1 frame (giữ cấu trúc, bỏ màu gốc)
control_process = v2.Compose([
    v2.CenterCrop(size=(H, W)),
    v2.Resize(size=(H // 8, W // 8), antialias=True),   # down 8x
    v2.Resize(size=(H, W), antialias=True),             # up lại
    v2.Grayscale(num_output_channels=3),
    v2.ToTensor(),
    v2.Normalize(mean=[0.5]*3, std=[0.5]*3),
])
content = crop_and_resize(Image.open(args.content_image).convert("RGB"), H, W)
control = control_process(content)                       # (C,H,W)
control_video = rearrange(control, "C H W -> 1 C 1 H W") # (B,C,T=1,H,W) bf16 cuda

# style embedding qua Processor (giống các script gốc)
image_embeds = processor.process_images([Image.open(args.style_image)])

video = pipe(
    prompt=args.prompt, negative_prompt=NEG_PROMPT,
    target_style=image_embeds,
    control_video=control_video, controlnet=controlnet,
    controlnet_conditioning_scale=args.controlnet_conditioning_scale,
    controlnet_guidance_start=0.0, controlnet_guidance_end=args.controlnet_guidance_end,
    cfg_scale=args.cfg_scale, style_cfg_scale=args.style_cfg_scale,
    num_frames=NUM_FRAMES, height=H, width=W,
    num_inference_steps=args.steps, seed=args.seed, tiled=True,
)
os.makedirs(os.path.dirname(args.output), exist_ok=True)
video[0].save(args.output)  # pipe trả về list PIL Image -> lấy frame đầu lưu PNG
```

Negative prompt: giữ nguyên chuỗi tiếng Trung mặc định của repo.

## Testing Strategy

Repo **không có unit test**. Kiểm chứng bằng smoke test thủ công:
- **T1 (setup):** `ls checkpoints/*.ckpt` và các file trong `models/Wan-AI/Wan2.1-T2V-1.3B/` tồn tại.
- **T2 (chạy):** lệnh ở mục Commands #2 chạy hết không lỗi, tạo `./results/i2i/out.png`.
- **T3 (đúng nghĩa i2i):** so sánh mắt thường — output giữ bố cục content, đổi style theo style image.
- **T4 (điều chỉnh):** tăng `--style_cfg_scale` → style đậm hơn; tăng `--controlnet_conditioning_scale`
  hoặc `--controlnet_guidance_end` → bám cấu trúc content chặt hơn.

## Boundaries

- **Always:** giữ `num_frames=1`; `content_image` xử lý qua đúng `control_process` gray-tile như V2V;
  lưu `video[0]` ra PNG; load `stylemaster.ckpt` với `strict=True` sau khi graft `style_model`/`_img`.
- **Ask first:** thay đổi mặc định `height/width` khác 480×832; thêm dependency mới; sửa `styleproj.py`
  hay bất kỳ file trong `diffsynth/`; đổi cách graft/checkpoint keys.
- **Never:** commit weights/checkpoints; sửa file trong `models/` hay `diffsynth/` để "ép" chạy;
  xoá script gốc `inference_stylemaster*.py`.

## Success Criteria

Một lệnh CLI 3 flag sinh ra 1 PNG giữ cấu trúc + đúng style, trên GPU, không cần sửa code — đạt cả
4 smoke test T1–T4.

## Open Questions / Rủi ro (kiểm chứng TRÊN HOST GPU)

1. **ControlNet với `num_frames=1`:** nhánh V2V vốn thiết kế cho 81 frame; không verify được ở máy dev
   (không có GPU/weights). Script đã hỗ trợ **fallback**: nếu lỗi shape ở temporal=1, chạy lại với
   `--num_frames 5` (hoặc `NUM_FRAMES=5 bash run_i2i.sh ...`) — vẫn chỉ lưu `video[0]` ra PNG.
2. Kích thước ảnh khác 480×832 phải chia hết 16 (pipeline tự resize) — có thể lệch tỉ lệ do CenterCrop.
3. Chưa có ảnh content mẫu trong repo — cần cung cấp 1 ảnh để test T2/T3 trên host.

## Trạng thái

- [x] `inference_stylemaster_i2i.py` — đã tạo (CLI flags, num_frames=1, lưu PNG).
- [x] `run_i2i.sh` — đã tạo (wrapper + override bằng env var), đã `chmod +x`.
- [x] `.gitignore` — thêm `stylemaster-wan/results/` để không commit output.
- [ ] Verify T1–T4 trên host GPU sau khi `download_ckpt.py`.
