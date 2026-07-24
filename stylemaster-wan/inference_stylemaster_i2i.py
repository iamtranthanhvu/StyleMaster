"""
StyleMaster Image-to-Image inference.

Restyle a single content image: giu bo cuc/cau truc cua content image (qua ControlNet
"gray-tile") nhung doi phong cach theo style image, roi luu ra 1 file PNG.

Chay tren may co GPU CUDA, sau khi da `python download_ckpt.py`.

Vi du:
    python inference_stylemaster_i2i.py \
        --content_image path/to/content.jpg \
        --style_image   example_test_data/style_images/vangough.png \
        --prompt        "a young child walking through a field of tall grass and poppies" \
        --output        ./results/i2i/out.png
"""

import os
import argparse

import torch
import torch.nn as nn
import torchvision
from torchvision.transforms import v2
from einops import rearrange
from PIL import Image

from diffsynth import ModelManager, WanVideoStyleMasterPipeline
from diffsynth.models.kolors_text_encoder import RMSNorm
from diffsynth.models.wan_video_dit import WanControlNet
from styleproj import Processor, StyleModel


# Negative prompt mac dinh cua repo (tieng Trung) - giu nguyen.
NEG_PROMPT = "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"


def crop_and_resize(image, height, width):
    """Scale + center-crop mot PIL image ve dung (height, width)."""
    w, h = image.size
    scale = max(width / w, height / h)
    image = torchvision.transforms.functional.resize(
        image,
        (round(h * scale), round(w * scale)),
        interpolation=torchvision.transforms.InterpolationMode.BILINEAR,
    )
    return image


def parse_args():
    parser = argparse.ArgumentParser(description="StyleMaster Image-to-Image inference")
    # --- Bat buoc ---
    parser.add_argument("--content_image", type=str, required=True,
                        help="Anh noi dung (giu bo cuc/cau truc).")
    parser.add_argument("--style_image", type=str, required=True,
                        help="Anh style (phong cach dich).")
    parser.add_argument("--prompt", type=str, required=True,
                        help="Prompt mo ta noi dung anh.")
    # --- Output ---
    parser.add_argument("--output", type=str, default="./results/i2i/output.png",
                        help="Duong dan file PNG output.")
    # --- Checkpoints ---
    parser.add_argument("--ckpt_path", type=str, default="checkpoints/stylemaster.ckpt")
    parser.add_argument("--controlnet_ckpt_path", type=str, default="checkpoints/controlnet.ckpt")
    # --- Kich thuoc / sampling ---
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=832)
    parser.add_argument("--num_frames", type=int, default=1,
                        help="Image-to-image dung 1. Tang len (vd 5) neu ControlNet ken temporal=1; "
                             "output van la frame dau tien.")
    parser.add_argument("--steps", type=int, default=50, help="So buoc denoise.")
    parser.add_argument("--seed", type=int, default=0)
    # --- CFG / do manh style & structure ---
    parser.add_argument("--cfg_scale", type=float, default=8.0, help="CFG text.")
    parser.add_argument("--style_cfg_scale", type=float, default=3.0,
                        help="CFG style: cao hon = style dam hon.")
    parser.add_argument("--controlnet_conditioning_scale", type=float, default=1.0,
                        help="Cao hon = bam cau truc content chat hon.")
    parser.add_argument("--controlnet_num_layer_stride", type=int, default=2)
    parser.add_argument("--controlnet_guidance_start", type=float, default=0.0)
    parser.add_argument("--controlnet_guidance_end", type=float, default=0.7,
                        help="Ti le buoc ap ControlNet (0-1). Cao hon = bam cau truc lau hon.")
    return parser.parse_args()


def build_control_video(content_path, height, width, num_frames, dtype, device):
    """content image -> tin hieu control 'gray-tile' shape (B, C, T, H, W)."""
    control_process = v2.Compose([
        v2.CenterCrop(size=(height, width)),
        v2.Resize(size=(height // 8, width // 8), antialias=True),  # down 8x
        v2.Resize(size=(height, width), antialias=True),            # up lai
        v2.Grayscale(num_output_channels=3),                        # xam nhung giu 3 kenh
        v2.ToTensor(),
        v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    image = Image.open(content_path).convert("RGB")
    image = crop_and_resize(image, height, width)
    control = control_process(image)                       # (C, H, W)
    control = rearrange(control, "C H W -> C 1 H W")        # (C, T=1, H, W)
    if num_frames > 1:
        control = control.repeat(1, num_frames, 1, 1)      # lap thanh T frame
    control = control.unsqueeze(0)                         # (B=1, C, T, H, W)
    return control.to(dtype=dtype, device=device)


def main():
    args = parse_args()

    # 1. Load Wan2.1 pretrained models
    model_manager = ModelManager(torch_dtype=torch.bfloat16, device="cpu")
    model_manager.load_models([
        "models/Wan-AI/Wan2.1-T2V-1.3B/diffusion_pytorch_model.safetensors",
        "models/Wan-AI/Wan2.1-T2V-1.3B/models_t5_umt5-xxl-enc-bf16.pth",
        "models/Wan-AI/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth",
    ])
    pipe = WanVideoStyleMasterPipeline.from_model_manager(model_manager, device="cuda")

    # 2. Graft cac module StyleMaster vao DiT (phai lam truoc khi load ckpt)
    processor = Processor().eval()
    dim = pipe.dit.blocks[0].cross_attn.q.weight.shape[0]
    pipe.dit.style_model = StyleModel()
    dim_s = pipe.dit.style_model.cross_attention_dim
    for block in pipe.dit.blocks:
        block.cross_attn.k_img = nn.Linear(dim_s, dim)
        block.cross_attn.v_img = nn.Linear(dim_s, dim)
        block.cross_attn.norm_k_img = RMSNorm(dim)

    # 3. Load style checkpoint
    state_dict = torch.load(args.ckpt_path, map_location="cpu")
    pipe.dit.load_state_dict(state_dict, strict=True)

    # 4. Load ControlNet (giu cau truc content image)
    config = {
        "dim": pipe.dit.dim,
        "in_dim": pipe.dit.patch_embedding.in_channels,
        "ffn_dim": pipe.dit.blocks[0].ffn_dim,
        "text_dim": pipe.dit.text_embedding[0].in_features,
        "freq_dim": pipe.dit.freq_dim,
        "eps": 1e-6,
        "patch_size": pipe.dit.patch_size,
        "num_heads": pipe.dit.blocks[0].num_heads,
        "num_layers": len(pipe.dit.blocks),
        "has_image_input": pipe.dit.has_image_input,
        "num_layer_stride": args.controlnet_num_layer_stride,
    }
    controlnet = WanControlNet(**config)
    controlnet.load_state_dict(torch.load(args.controlnet_ckpt_path, map_location="cpu"), strict=True)
    controlnet.to("cuda").to(dtype=torch.bfloat16).eval()
    print("ControlNet loaded.")

    pipe.to("cuda")
    pipe.to(dtype=torch.bfloat16)

    # 5. Chuan bi input
    control_video = build_control_video(
        args.content_image, args.height, args.width, args.num_frames,
        dtype=torch.bfloat16, device="cuda",
    )
    print(f"Control video shape: {tuple(control_video.shape)}")
    image_embeds = processor.process_images([Image.open(args.style_image)])

    # 6. Inference (image-to-image: num_frames=1)
    video = pipe(
        prompt=args.prompt,
        negative_prompt=NEG_PROMPT,
        target_style=image_embeds,
        control_video=control_video,
        controlnet=controlnet,
        controlnet_conditioning_scale=args.controlnet_conditioning_scale,
        controlnet_guidance_start=args.controlnet_guidance_start,
        controlnet_guidance_end=args.controlnet_guidance_end,
        cfg_scale=args.cfg_scale,
        style_cfg_scale=args.style_cfg_scale,
        num_frames=args.num_frames,
        height=args.height,
        width=args.width,
        num_inference_steps=args.steps,
        seed=args.seed,
        tiled=True,
    )

    # 7. Luu frame dau tien ra PNG
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    video[0].save(args.output)
    print(f"Saved image -> {args.output}")


if __name__ == "__main__":
    main()
