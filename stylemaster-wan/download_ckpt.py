
from huggingface_hub import snapshot_download

# 下载模型到指定目录
snapshot_download("KwaiVGI/StyleMaster", local_dir="checkpoints")

from modelscope import snapshot_download

snapshot_download("Wan-AI/Wan2.1-T2V-1.3B", local_dir="models/Wan-AI/Wan2.1-T2V-1.3B")