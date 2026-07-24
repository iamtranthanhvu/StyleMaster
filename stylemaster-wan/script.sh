## inference
python inference_stylemaster.py 

## data process

###image
CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7" python train_stylemaster.py   --task data_process   --dataset_path ./data  --metadata_file_name image_example.csv --output_path ./models   --text_encoder_path "models/Wan-AI/Wan2.1-T2V-1.3B/models_t5_umt5-xxl-enc-bf16.pth"   --vae_path "models/Wan-AI/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth"   --tiled   --num_frames 1   --height 480   --width 832 --dataloader_num_workers 2   


###video
CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7" python train_stylemaster.py   --task data_process   --dataset_path ./data  --metadata_file_name video_example.csv --output_path ./models   --text_encoder_path "models/Wan-AI/Wan2.1-T2V-1.3B/models_t5_umt5-xxl-enc-bf16.pth"   --vae_path "models/Wan-AI/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth"   --tiled   --num_frames 77   --height 480   --width 832 --dataloader_num_workers 2   

##train
###train-stage1
CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7" python train_stylemaster.py   --task train  --dataset_path ./data --metadata_file_name image_example.csv  --output_path ./models/train   --dit_path "models/Wan-AI/Wan2.1-T2V-1.3B/diffusion_pytorch_model.safetensors"  --steps_per_epoch 8000   --max_epochs 100   --learning_rate 1e-4   --accumulate_grad_batches 1     --dataloader_num_workers 4

###train-stage2
CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7" python train_stylemaster.py   --task train  --dataset_path ./data --metadata_file_name video_example.csv  --output_path ./models/train   --dit_path "models/Wan-AI/Wan2.1-T2V-1.3B/diffusion_pytorch_model.safetensors" --resume_ckpt_path "first_stage_ckpt"  --steps_per_epoch 8000   --max_epochs 100   --learning_rate 1e-4   --accumulate_grad_batches 1     --dataloader_num_workers 4
