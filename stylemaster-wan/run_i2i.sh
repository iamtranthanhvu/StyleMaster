#!/usr/bin/env bash
# Wrapper chay StyleMaster image-to-image (restyle 1 anh).
#
# Cach dung:
#   bash run_i2i.sh <content_image> <style_image> "<prompt>" [output.png]
#
# Vi du:
#   bash run_i2i.sh content.jpg example_test_data/style_images/vangough.png \
#       "a young child walking through a field of tall grass and poppies"
#
# Chinh nhanh do manh style/structure bang bien moi truong:
#   STYLE_CFG=5 CN_SCALE=1.2 CN_END=0.8 SEED=42 bash run_i2i.sh ...

set -euo pipefail

if [ "$#" -lt 3 ]; then
  echo "Usage: bash run_i2i.sh <content_image> <style_image> \"<prompt>\" [output.png]" >&2
  exit 1
fi

CONTENT="$1"
STYLE="$2"
PROMPT="$3"
OUTPUT="${4:-./results/i2i/out.png}"

# Tham so co the override qua env var (deu co default trong script python)
STYLE_CFG="${STYLE_CFG:-3.0}"      # style dam hon = cao hon
CFG="${CFG:-8.0}"                  # CFG text
CN_SCALE="${CN_SCALE:-1.0}"        # bam cau truc content = cao hon
CN_END="${CN_END:-0.7}"           # ti le buoc ap ControlNet
STEPS="${STEPS:-50}"
SEED="${SEED:-0}"
HEIGHT="${HEIGHT:-480}"
WIDTH="${WIDTH:-832}"
NUM_FRAMES="${NUM_FRAMES:-1}"      # tang len (vd 5) neu ControlNet ken temporal=1

cd "$(dirname "$0")"

python inference_stylemaster_i2i.py \
  --content_image "$CONTENT" \
  --style_image "$STYLE" \
  --prompt "$PROMPT" \
  --output "$OUTPUT" \
  --height "$HEIGHT" \
  --width "$WIDTH" \
  --num_frames "$NUM_FRAMES" \
  --steps "$STEPS" \
  --seed "$SEED" \
  --cfg_scale "$CFG" \
  --style_cfg_scale "$STYLE_CFG" \
  --controlnet_conditioning_scale "$CN_SCALE" \
  --controlnet_guidance_end "$CN_END"
