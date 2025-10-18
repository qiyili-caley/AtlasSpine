from ultralytics import YOLO
import argparse
import os

def parse_args():
    parser = argparse.ArgumentParser(description='YOLO旋转框(OBB)脊椎骨检测训练')
    parser.add_argument('--data', type=str, default='yolo_spine_dataset/dataset.yaml',
                        help='数据集配置文件路径，需指向YOLO OBB格式数据集')
    parser.add_argument('--epochs', type=int, default=120, help='训练轮数')
    parser.add_argument('--batch-size', type=int, default=4, help='批次大小')
    parser.add_argument('--img-size', type=int, default=1024, help='图像尺寸')
    parser.add_argument('--model', type=str, default='yolo11n-obb.pt',
                        help='预训练OBB模型权重路径')
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='训练设备，如"cpu"、"cuda:0"等')
    parser.add_argument('--workers', type=int, default=4, help='数据加载线程数')
    parser.add_argument('--patience', type=int, default=12, help='早停耐心值')
    parser.add_argument('--project', type=str, default='spine_detection',
                        help='项目输出目录名')
    parser.add_argument('--name', type=str, default='yolo11n-obb',
                        help='训练任务名/权重存储文件夹名')
    return parser.parse_args()

def main():
    args = parse_args()
    print("=== 脊椎骨YOLO旋转框(OBB)检测训练 ===")
    print(f"训练轮数: {args.epochs}, 批次大小: {args.batch_size}, 图像尺寸: {args.img_size}")
    print(f"使用模型权重: {args.model}")
    print(f"数据集配置文件: {args.data}")
    print(f"输出目录: {args.project}/{args.name}")

    # 加载OBB模型，确保task为'obb'
    model = YOLO(args.model)

    # 开始训练
    results = model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.img_size,
        batch=args.batch_size,
        patience=args.patience,
        project=args.project,
        name=args.name,
        device=args.device,
        workers=args.workers,
        lr0=0.0007,
        lrf=0.007,
        save=True,
        plots=True,
        pretrained=True,
        verbose=True,
    )

    # 训练完成后验证模型
    print("\n=== 模型验证 ===")
    metrics = model.val(
        data=args.data,
        batch=args.batch_size,
        imgsz=args.img_size,
        device=args.device,
        project=args.project,
        name=args.name + "_val"
    )
    print(f"验证指标: {metrics}")
    print(f"模型权重已保存至: {os.path.join(args.project, args.name, 'weights')}")

if __name__ == "__main__":
    main()