python generate_batch.py \
    --name test \
    --views identity jigsaw \
    --num_samples 1 \
    --num_inference_steps 30 \
    --guidance_scale 10.0 \
    --object_list object.list \
    --style_list style.list \
    --mod_list modifier.list \
    --select_times 10000