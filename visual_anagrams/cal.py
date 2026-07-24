import os
import clip
import torch
from PIL import Image
import csv

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

def compute_similarity(text, image_path):
    image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)
    text = clip.tokenize([text]).to(device)

    with torch.no_grad():
        image_features = model.encode_image(image)
        text_features = model.encode_text(text)

    similarity = torch.cosine_similarity(image_features, text_features)
    return similarity.item()

def save_to_csv(file_path, folder_paths):
    with open(file_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Folder Path"])
        for path in folder_paths:
            writer.writerow([path])

# 遍历子文件夹
base_dir = "./results"
folders_0_25 = []
folders_0_2 = []
folders_0_15 = []
folders_0_3 = []
folders_0_28 = []
all_folders = []

for subdir in os.listdir(base_dir):
    subdir_path = os.path.join(base_dir, subdir)
    if os.path.isdir(subdir_path):
        prompt_path = os.path.join(subdir_path, "prompts.txt")
        image_0_path = os.path.join(subdir_path, "sample_256_view_0.png")
        image_1_path = os.path.join(subdir_path, "sample_256_view_1.png")

        if os.path.exists(prompt_path) and os.path.exists(image_0_path) and os.path.exists(image_1_path):
            with open(prompt_path, "r") as f:
                prompts = f.readlines()

            if len(prompts) >= 2:
                # 只取每行的最后两个词
                prompt_0 = ' '.join(prompts[0].strip().split()[-2:])
                prompt_1 = ' '.join(prompts[1].strip().split()[-2:])

                similarity_0 = compute_similarity(prompt_0, image_0_path)
                similarity_1 = compute_similarity(prompt_1, image_1_path)

                print(f"Subdir: {subdir}")
                print(f"Similarity between last two words of first prompt and sample_256_view_0.png: {similarity_0}")
                print(f"Similarity between last two words of second prompt and sample_256_view_1.png: {similarity_1}")
                print()

                # 检查相似度并存储路径
                if similarity_0 > 0.25 and similarity_1 > 0.25:
                    folders_0_25.append(subdir_path)
                if similarity_0 > 0.2 and similarity_1 > 0.2:
                    folders_0_2.append(subdir_path)
                if similarity_0 > 0.15 and similarity_1 > 0.15:
                    folders_0_15.append(subdir_path)
                if similarity_0 > 0.3 and similarity_1 > 0.3:
                    folders_0_3.append(subdir_path)
                if similarity_0 > 0.28 and similarity_1 > 0.28:
                    folders_0_28.append(subdir_path)
                all_folders.append(subdir_path)

# 保存结果到CSV文件
save_to_csv("folders_0_25.csv", folders_0_25)
save_to_csv("folders_0_2.csv", folders_0_2)
save_to_csv("folders_0_15.csv", folders_0_15)
save_to_csv("folders_0_3.csv", folders_0_3)
save_to_csv("folders_0_28.csv", folders_0_28)
save_to_csv("all_folders.csv", all_folders)