# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

StyleMaster is the official research code for the CVPR 2025 paper *"StyleMaster: Stylize Your Video with Artistic Generation and Translation"* (HKUST + Kuaishou). Given a reference **style image** + a text prompt, it either generates a new video in that style (T2V) or restyles an existing video (V2V). It is a Python/PyTorch, GPU/CUDA research repo built on the **Wan2.1-T2V-1.3B** diffusion model. There is no web/service layer.

## Repository layout (three loosely-coupled subprojects)

- `stylemaster-wan/` — the main implementation (~95% of real code): training + inference on Wan2.1. Contains a **vendored, modified fork of DiffSynth-Studio** in `diffsynth/`.
- `visual_anagrams/` — independent sub-project (fork of "Visual Anagrams") that generates the **illusion dataset** used to supervise global style extraction. Has its own `setup.py`, `environment.yml`, and entry scripts; not imported by the main model.
- `style_extraction/` — standalone minimal reference implementation of the style-extraction algorithm.
- `models/` — placeholder tree for downloaded Wan2.1 weights (`models/Wan-AI/Wan2.1-T2V-1.3B/`).
- `evaluation/` — eval assets (`stylized_videos_for_test.csv`, `style_set/`).

Note: `basic_modules.py` and `styleproj.py`/`style_extraction_module.py` are near-duplicated at the top level, in `style_extraction/`, and in `stylemaster-wan/`. The pipeline imports the **`stylemaster-wan/styleproj.py`** copy — edit that one for model behavior.

## Environments & setup

Two separate conda environments are required; they are not compatible.

**Main model** (`stylemaster-wan/`, Python 3.10):
```bash
cd stylemaster-wan
conda create --name stylemaster python=3.10 && conda activate stylemaster
pip install -e .            # installs the vendored `diffsynth` package (editable)
python download_ckpt.py     # pulls KwaiVGI/StyleMaster -> checkpoints/, Wan2.1-T2V-1.3B -> models/
```

**Dataset generation** (`visual_anagrams/`, Python 3.9, Linux only):
```bash
cd visual_anagrams
conda env create -f environment.yml && conda activate visual_anagrams
```

## Common commands

All entry points are plain `python` scripts configured via **argparse / CLI flags** (there are no config files). Multi-GPU runs set `CUDA_VISIBLE_DEVICES`.

**Inference** (run inside `stylemaster-wan/`):
```bash
python inference_stylemaster.py       # T2V: style image + prompt -> stylized video
python inference_stylemaster_v2v.py   # V2V: restyle an input video (adds gray-tile ControlNet)
```

**Training** — two-phase (see `stylemaster-wan/script.sh` for the full documented workflow):
```bash
# Phase A: pre-encode VAE latents/embeddings to disk (--task data_process)
CUDA_VISIBLE_DEVICES="0,..,7" python train_stylemaster.py --task data_process \
  --dataset_path ./data --metadata_file_name image_example.csv --output_path ./models \
  --text_encoder_path "models/Wan-AI/Wan2.1-T2V-1.3B/models_t5_umt5-xxl-enc-bf16.pth" \
  --vae_path "models/Wan-AI/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth" --tiled --num_frames 1 --height 480 --width 832

# Phase B stage 1 (images), then stage 2 (video) resumes from stage-1 ckpt:
CUDA_VISIBLE_DEVICES="0,..,7" python train_stylemaster.py --task train \
  --dataset_path ./data --metadata_file_name image_example.csv --output_path ./models/train \
  --dit_path "models/Wan-AI/Wan2.1-T2V-1.3B/diffusion_pytorch_model.safetensors" \
  --steps_per_epoch 8000 --max_epochs 100 --learning_rate 1e-4
# stage 2: swap to video_example.csv and add --resume_ckpt_path "first_stage_ckpt"
```
`train_stylemaster_v2v.py` is the V2V (ControlNet + style) training variant.

**Illusion dataset** (run inside `visual_anagrams/`):
```bash
bash generate_batch.sh   # batch-generate visual-anagram pairs
python cal.py            # filter low-quality pairs by CLIP score
```

## Tooling status (important)

- **No build system, lint, format, or typecheck** is configured anywhere (no ruff/black/isort/mypy, no Makefile/pyproject/tox). "Build" = `pip install -e .`.
- **No unit tests.** `visual_anagrams/tests/*.sh` are example generation runs (e.g. `bash tests/flip.sh`), not assertions — running them produces sample images, not pass/fail.
- **No CI, pre-commit hooks, or editor rule files.**
- Many comments and the default negative prompt are in **Chinese**.

## Architecture (the parts that span multiple files)

**The central pattern: runtime monkey-patching of the Wan DiT.** StyleMaster is not a standalone model — it is extra layers grafted onto the base Wan2.1 DiT at load time. Both inference scripts and the training `LightningModel` do this identically (see `inference_stylemaster.py` ~lines 159–172):
1. Load Wan2.1 DiT + T5 + VAE via `ModelManager` → `WanVideoStyleMasterPipeline.from_model_manager`.
2. Attach `pipe.dit.style_model = StyleModel()`, and add `k_img` / `v_img` / `norm_k_img` layers to **every** `pipe.dit.blocks[i].cross_attn`.
3. Only then does `pipe.dit.load_state_dict(merged_ckpt, strict=True)` succeed. **Any change to the graft must be mirrored in the checkpoint's keys.**

**Style pipeline data flow (T2V):**
- `styleproj.py::Processor.process_images` produces two CLIP (ViT-H/14) embeddings per reference image: normal (patch tokens via `last_hidden_state`) and **content-shuffled** (`ContentShuffleDetector` scrambles spatial content so CLIP encodes style, not subject).
- `styleproj.py::StyleModel.project_image_embeddings` builds `style_feat` = **local tokens** (patches, after `drop_tokens_by_similarity` drops text-correlated patches, then a Q-former self-attention) + **global tokens** (MLP over the shuffled embedding). Defaults: 2 local + 14 global tokens, dim `cross_attention_dim=4096`.
- Denoising threads `style_feat` through each DiT block; `CrossAttention.forward` injects style via a **second cross-attention branch** (`k_img`/`v_img`) added to the text cross-attention output.
- **Triple-CFG**: the pipeline computes three predictions (full / no-style-no-text / text-only) combined with separate `cfg_scale` (text) and `style_cfg_scale` (style). See `diffsynth/pipelines/wan_video_stylemaster.py`.

**V2V additions:** the input video is downsampled 8×, upsampled, and grayscaled into a "gray-tile" control signal. `WanControlNet` (a strided subset of DiT blocks) produces residuals injected additively every `control_stride=2` layers, gated by `controlnet_guidance_start/end` step ratios (`should_apply_controlnet`). See `inference_stylemaster_v2v.py` and `WanControlNet` in `diffsynth/models/wan_video_dit.py`.

**Vendored `diffsynth/` structure** (`import diffsynth`, installed editable — NOT the upstream pip package):
- `models/model_manager.py` — `ModelManager`, loads Wan weights by state-dict pattern-matching. Wan model defs: `models/wan_video_{dit,text_encoder,vae,image_encoder}.py`. (flux/sd/sdxl/hunyuan/cog/step files are inherited DiffSynth cruft, unused here.)
- `pipelines/wan_video_stylemaster.py` — the StyleMaster pipeline (denoising loop, triple-CFG, ControlNet); `base.py` is `BasePipeline`.
- `schedulers/flow_match.py` — `FlowMatchScheduler` (flow-matching).
- `prompters/wan_prompter.py` — UMT5-xxl text encoding + tokenizer configs.
- `vram_management/` — `enable_vram_management` / `AutoWrappedModule` transparent CPU-offload wrappers used throughout the pipeline for GPU memory.

**Training conventions:**
- PyTorch **Lightning**; two tasks: `data_process` (`LightningModelForDataProcess`, pre-encodes to disk) then `train` (`LightningModel`).
- Selective trainable params by **name-matching**: everything is frozen, then only modules whose name contains `"style_model"` or `"_img"` are unfrozen. Three `training_mode`s: `style`, `controlnet`, `style_w_controlnet`.
- Datasets are CSV-driven: `stylemaster-wan/data/{image,video}_example.csv`, `example_test_data/metadata.csv` (columns include `text`/`caption`, `style`, optional control/video paths).

## Files to read first

`stylemaster-wan/inference_stylemaster.py` (the graft + end-to-end flow) → `stylemaster-wan/styleproj.py` (style algorithm) → `stylemaster-wan/diffsynth/pipelines/wan_video_stylemaster.py` (denoising, triple-CFG, ControlNet) → `stylemaster-wan/diffsynth/models/wan_video_dit.py` (`CrossAttention`, `DiTBlock`, `WanModel`, `WanControlNet`).
