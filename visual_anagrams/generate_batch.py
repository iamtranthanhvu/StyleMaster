import argparse
from pathlib import Path
from PIL import Image
import random
import time
import torch
from diffusers import DiffusionPipeline
import torchvision.transforms.functional as TF

from visual_anagrams.views import get_views
from visual_anagrams.samplers import sample_stage_1, sample_stage_2
from visual_anagrams.utils import add_args, save_illusion, save_metadata
import os

def read_random_line(filepath, n):
    with open(filepath, 'r') as f:
        lines = f.readlines()
        return random.sample(lines, n)

# Parse args
parser = argparse.ArgumentParser()
parser = add_args(parser)
parser.add_argument('--object_list', type=Path, help='Path to the object list file')
parser.add_argument('--style_list', type=Path, help='Path to the style list file')
parser.add_argument('--mod_list', type=Path, help='Path to the style list file')
parser.add_argument('--select_times', type=int, default=1, help='Number of times to generate random elements')
args = parser.parse_args()

# Do admin stuff
save_dir = Path(args.save_dir)
print('save', save_dir)

# Load reference image (for inverse problems)
if args.ref_im_path is not None:
    ref_im = Image.open(args.ref_im_path)
    ref_im = TF.to_tensor(ref_im) * 2 - 1
    print(ref_im.shape)
else:
    ref_im = None

# Make DeepFloyd IF stage I
stage_1 = DiffusionPipeline.from_pretrained(
                "DeepFloyd/IF-I-M-v1.0",
                variant="fp16",
                torch_dtype=torch.float16)
stage_1.enable_model_cpu_offload()
stage_1 = stage_1.to(args.device)

# Make DeepFloyd IF stage II
stage_2 = DiffusionPipeline.from_pretrained(
                "DeepFloyd/IF-II-M-v1.0",
                text_encoder=None,
                variant="fp16",
                torch_dtype=torch.float16,
            )
stage_2.enable_model_cpu_offload()
stage_2 = stage_2.to(args.device)

# Make DeepFloyd IF stage III (which is just Stable Diffusion 4x Upsampler)
if args.generate_1024:
    stage_3 = DiffusionPipeline.from_pretrained(
                    "stabilityai/stable-diffusion-x4-upscaler", 
                    torch_dtype=torch.float16
                )
    stage_3.enable_model_cpu_offload()
    stage_3 = stage_3.to(args.device)

for _ in range(args.select_times):
    # Select random prompts and style
    prompts = read_random_line(args.object_list, 2)
    style = read_random_line(args.style_list, 1)[0]
    mod = read_random_line(args.mod_list, 1)[0]

    # Get prompt embeddings
    prompts = [f'a {style.strip()} of {p.strip()}'.strip() for p in prompts]
    print(prompts)
    prompt_embeds = [stage_1.encode_prompt(p) for p in prompts]
    prompt_embeds, negative_prompt_embeds = zip(*prompt_embeds)
    prompt_embeds = torch.cat(prompt_embeds)
    negative_prompt_embeds = torch.cat(negative_prompt_embeds)  # These are just null embeds

    # Get views
    views = get_views(args.views, view_args=args.view_args)

    # Save metadata
    save_metadata(views, args, save_dir)

    # Sample illusions
    for i in range(args.num_samples):
        # Admin stuff
        generator = torch.manual_seed(args.seed + i)
        current_time = time.time()
        sample_dir = save_dir / str(current_time)
        sample_dir.mkdir(exist_ok=True, parents=True)

        # Save prompts to a file
        prompt_file = sample_dir / "prompts.txt"
        with open(prompt_file, 'w') as f:
            for prompt in prompts:
                f.write(f"{prompt}\n")

        # Sample 64x64 image
        image = sample_stage_1(stage_1, 
                               prompt_embeds,
                               negative_prompt_embeds,
                               views,
                               ref_im=ref_im,
                               num_inference_steps=args.num_inference_steps,
                               guidance_scale=args.guidance_scale,
                               reduction=args.reduction,
                               generator=generator)
        # save_illusion(image, views, sample_dir)

        # Sample 256x256 image, by upsampling 64x64 image
        image = sample_stage_2(stage_2,
                               image,
                               prompt_embeds,
                               negative_prompt_embeds, 
                               views,
                               ref_im=ref_im,
                               num_inference_steps=args.num_inference_steps,
                               guidance_scale=args.guidance_scale,
                               reduction=args.reduction,
                               noise_level=args.noise_level,
                               generator=generator)
        save_illusion(image, views, sample_dir)

        if args.generate_1024:
            # Naively upsample to 1024x1024 using first prompt
            #   n.b. This is just the SD upsampler, and does not 
            #   take into account the other views. Results may be
            #   poor for the other view. See readme for more details
            image_1024 = stage_3(
                            prompt=prompts[0], 
                            image=image, 
                            noise_level=0,
                            output_type='pt',
                            generator=generator).images
            save_illusion(image_1024 * 2 - 1, views, sample_dir)