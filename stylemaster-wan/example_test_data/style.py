import os
import shutil
import pandas as pd

# 读取CSV文件
csv_path = '/m2v_intern/yezixuan/T2V_Models/StyleMaster/example_test_data/metadata.csv'
df = pd.read_csv(csv_path)

# 目标目录
target_dir = '/m2v_intern/yezixuan/T2V_Models/StyleMaster/example_test_data/style_images'

# 确保目标目录存在
os.makedirs(target_dir, exist_ok=True)

# 复制图片并更新路径
for index, row in df.iterrows():
    original_path = row['style']
    if os.path.exists(original_path):
        # 获取图片文件名
        filename = os.path.basename(original_path)
        # 目标路径
        new_path = os.path.join(target_dir, filename)
        print(original_path,new_path)
        # 复制图片
        shutil.copy2(original_path, new_path)
        # 更新CSV中的路径
        df.at[index, 'style'] = new_path
    else:
        print(f"Warning: {original_path} does not exist.")

# 保存更新后的CSV文件
df.to_csv(csv_path, index=False)
print("CSV文件已更新并保存。")