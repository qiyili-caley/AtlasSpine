import cv2
import numpy as np
import os
import re

def clean_path(path):
    # 只用于路径！不要用于数据内容
    return re.sub(r'[\x00-\x1F\x7F]', '', path.strip())

# 文件夹设置
img_dir = clean_path("sample")
label_dir = clean_path("results_obb\exp")
out_dir = clean_path("results_obb\masks")
os.makedirs(out_dir, exist_ok=True)

img_extensions = ['.jpg', '.jpeg', '.png', '.JPG']

for label_file in os.listdir(label_dir):
    label_file = clean_path(label_file)
    if not label_file.endswith('_info.txt'):
        continue
    img_name = label_file.replace('_info.txt', '')

    img_path = None
    for ext in img_extensions:
        candidate = os.path.join(img_dir, img_name + ext)
        if os.path.exists(candidate):
            img_path = candidate
            break

    if img_path is None:
        print(f"[WARN] 未找到原图: {img_name}")
        continue

    print(f"[INFO] 正在读取图片: {repr(img_path)}")
    img = cv2.imread(img_path)
    if img is None:
        print(f"[WARN] 无法读取: {img_path}")
        continue

    H, W = img.shape[:2]
    mask = np.zeros((H, W), dtype=np.uint8)  # 全黑底板

    label_path = os.path.join(label_dir, label_file)
    print(f"[INFO] 正在读取标注: {repr(label_path)}")
    with open(label_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()[1:]  # 跳过表头

    for line in lines:
        line = line.strip()
        if not line:
            continue
        parts = line.split('\t')
        if len(parts) == 1:
            parts = line.split()
        if len(parts) == 1:
            parts = line.split(',')
        if len(parts) != 6:
            print(f"[WARN] 标注行格式异常: {repr(line)}")
            continue

        id, cx, cy, w, h, angle = parts
        cx, cy, w, h, angle = map(float, [cx, cy, w, h, angle])
        # 弧度转角度，OpenCV要求angle为度
        angle_degree = angle * 180 / np.pi
        # 构造椎骨旋转矩形
        rect = ((cx, cy), (w, h), angle_degree)
        box = cv2.boxPoints(rect)
        box = np.int32(box)
        # 在mask上填充白色椎骨区域
        cv2.fillPoly(mask, [box], 255)

    # 保存mask
    out_path = os.path.join(out_dir, f"{img_name}_mask.png")
    cv2.imwrite(out_path, mask)
    print(f"[OK] 已保存mask：{repr(out_path)}")