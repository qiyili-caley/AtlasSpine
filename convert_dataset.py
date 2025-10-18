import os
import cv2
import numpy as np
import json
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import shutil

def create_yolo_directory_structure(output_dir):
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'images', 'train'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'images', 'val'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'labels', 'train'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'labels', 'val'), exist_ok=True)
    print(f"已创建YOLO OBB数据集目录结构: {output_dir}")
    return output_dir

def extract_vertebrae_box_from_json(json_path):
    """
    从json文件中提取每个椎骨的四点多边形坐标
    返回: list of np.array, 每个shape (4,2)
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        info = json.load(f)
    boxes = []
    for shape in info.get('shapes', []):
        if shape['label'] != '1':
            continue
        points = shape['points']
        if len(points) != 4:
            # 如果是多边形（>4点），可以加最小外接矩形处理
            points = np.array(points, dtype=np.float32)
            if points.shape[0] < 4:
                continue
            rect = cv2.minAreaRect(points)
            box = cv2.boxPoints(rect)
            box = np.array(box, dtype=np.float32)
        else:
            box = np.array(points, dtype=np.float32)
        boxes.append(box)
    # 返回boxes, image width, height
    width = info.get('imageWidth')
    height = info.get('imageHeight')
    return boxes, width, height

def json_to_yolo_obb_corners_format(json_path):
    """
    将json文件转换为YOLO OBB四角点标注格式
    每行: class x1 y1 x2 y2 x3 y3 x4 y4
    """
    boxes, width, height = extract_vertebrae_box_from_json(json_path)
    yolo_lines = []
    for box in boxes:
        norm_coords = []
        for (x, y) in box:
            norm_coords.append(x / width)
            norm_coords.append(y / height)
        line = f"0 " + " ".join([f"{coord:.6f}" for coord in norm_coords])
        yolo_lines.append(line)
    return yolo_lines

def process_spine_dataset_json(image_dir, label_dir, output_dir, test_size=0.2):
    create_yolo_directory_structure(output_dir)
    all_images = [f for f in os.listdir(image_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    all_labels = [f for f in os.listdir(label_dir) if f.lower().endswith('.json')]
    common_files = []
    for img_file in all_images:
        base_name = os.path.splitext(img_file)[0]
        label_file = f"{base_name}.json"
        if label_file in all_labels:
            common_files.append(base_name)
    print(f"找到{len(common_files)}对匹配的图像和json标签")
    train_files, val_files = train_test_split(common_files, test_size=test_size, random_state=42)
    print(f"训练集: {len(train_files)}个文件, 验证集: {len(val_files)}个文件")
    process_file_set_json(image_dir, label_dir, output_dir, train_files, "train")
    process_file_set_json(image_dir, label_dir, output_dir, val_files, "val")
    create_dataset_yaml(output_dir)
    print("数据集转换完成!")

def process_file_set_json(image_dir, label_dir, output_dir, file_list, mode):
    print(f"处理{mode}集...")
    for base_name in tqdm(file_list):
        image_files = [f for f in os.listdir(image_dir) if f.startswith(base_name) and
                       f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        if not image_files:
            continue
        image_file = image_files[0]
        image_path = os.path.join(image_dir, image_file)
        label_file = f"{base_name}.json"
        label_path = os.path.join(label_dir, label_file)
        if not os.path.exists(label_path):
            continue
        # 检查图片是否能读取
        image = cv2.imread(image_path)
        if image is None:
            print(f"无法读取图像: {image_path}")
            continue
        yolo_lines = json_to_yolo_obb_corners_format(label_path)
        if not yolo_lines:
            print(f"警告: {label_path} 没有识别到椎骨多边形")
            continue
        dest_image_path = os.path.join(output_dir, 'images', mode, image_file)
        shutil.copy(image_path, dest_image_path)
        dest_label_path = os.path.join(output_dir, 'labels', mode,
                                       f"{os.path.splitext(image_file)[0]}.txt")
        with open(dest_label_path, 'w') as f:
            for line in yolo_lines:
                f.write(line + '\n')

def create_dataset_yaml(output_dir):
    yaml_path = os.path.join(output_dir, 'dataset.yaml')
    with open(yaml_path, 'w') as f:
        f.write("# YOLO OBB脊椎椎骨检测数据集配置\n")
        f.write(f"path: {output_dir}\n")
        f.write("train: images/train\n")
        f.write("val: images/val\n\n")
        f.write("names:\n")
        f.write("  0: \"vertebra\" \n")
    print(f"数据集配置文件已保存至: {yaml_path}")

def main():
    image_dir = 'sample'  # 原始图像目录
    label_dir = 'sample_label'  # json标签目录
    output_dir = 'yolo_spine_dataset'  # YOLO数据集输出目录
    process_spine_dataset_json(image_dir, label_dir, output_dir)

if __name__ == "__main__":
    main()