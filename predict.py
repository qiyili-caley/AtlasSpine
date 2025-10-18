from ultralytics import YOLO
import cv2
import numpy as np
import os
from tqdm import tqdm

MODEL_PATH = 'spine_detection/yolo11n-obb/weights/best.pt'
SOURCE_DIR = 'sample_flipped'
SAVE_DIR = 'results_obb'
EXP_DIR = os.path.join(SAVE_DIR, "exp")

os.makedirs(EXP_DIR, exist_ok=True)

model = YOLO(MODEL_PATH)

results = model.predict(
    SOURCE_DIR,
    imgsz=1024,
    conf=0.35,
    iou=0.30,
    save=True,
    project=SAVE_DIR,
    name="exp"
)

for result in tqdm(results, desc="处理OBB结果"):
    img_path = result.path
    img_name = os.path.basename(img_path)
    img = cv2.imread(img_path)
    if img is None:
        print(f"无法读取图像 {img_path}")
        continue

    vertebrae_info = []
    # 正确读取OBB结果
    if hasattr(result.obb, "xywhr") and result.obb.xywhr is not None:
        boxes = result.obb.xywhr.cpu().numpy()
        confs = result.obb.conf.cpu().numpy()
        for i, (box, conf) in enumerate(zip(boxes, confs)):
            cx, cy, w, h, angle = box
            vertebrae_info.append({
                'id': i + 1,
                'cx': cx,
                'cy': cy,
                'width': w,
                'height': h,
                'angle': angle,
            })
            angle_degree = float(angle) * 180 / np.pi
            rect = ((cx, cy), (w, h), angle_degree)
            box_points = cv2.boxPoints(rect)
            cv2.drawContours(img, [np.int32(box_points)], 0, (0, 0, 255), 2)
            # 蓝色ID
            cv2.putText(img, f"ID:{i+1}", (int(cx), int(cy)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
            # 绿色角度
            cv2.putText(img, f"{angle:.1f}", (int(cx), int(cy)+20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    else:
        print(f"{img_name} 未检测到 OBB 结果")
        continue

    # 保存效果图
    result_path = os.path.join(EXP_DIR, img_name)
    cv2.imwrite(result_path, img)

    # 保存检测信息到txt
    info_path = os.path.join(EXP_DIR, f"{os.path.splitext(img_name)[0]}_info.txt")
    with open(info_path, "w", encoding="utf-8") as f:
        f.write("id\tcx\tcy\twidth\theight\tangle\n")
        for v in vertebrae_info:
            f.write(f"{v['id']}\t{v['cx']:.2f}\t{v['cy']:.2f}\t{v['width']:.2f}\t{v['height']:.2f}\t{v['angle']:.2f}\n")

    print(f"已保存效果图：{result_path}")
    print(f"已保存检测信息：{info_path}")