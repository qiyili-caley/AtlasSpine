import os
from PIL import Image


def flip_images_in_folder(src_folder):
    # 目标文件夹
    dst_folder = src_folder + "_flipped"
    os.makedirs(dst_folder, exist_ok=True)

    for filename in os.listdir(src_folder):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp',)):
            src_path = os.path.join(src_folder, filename)
            with Image.open(src_path) as img:
                img.save(os.path.join(dst_folder, filename))
                flipped = img.transpose(Image.FLIP_LEFT_RIGHT)
                name, ext = os.path.splitext(filename)
                new_filename = f"{name}_flipped{ext}"
                dst_path = os.path.join(dst_folder, new_filename)
                flipped.save(dst_path)
                print(f"已生成：{dst_path}")


# 使用方法
flip_images_in_folder("sample")

import pandas as pd

# 读取标签文件（csv格式，excel请用pd.read_excel）
df = pd.read_excel('label_screw.xlsx')

# 遍历每一列，复制并重命名
for col in df.columns:
    df[str(col) + '_flipped'] = df[col]

# 保存回原文件或另存为新文件
df.to_excel('label_screw_flipped.xlsx', index=False)