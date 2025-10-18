# Spine Detection and Surgical Prediction  

**脊椎检测与手术预测**  

---

## Overview | 项目简介  

This repository provides a complete pipeline for **spine detection, post-processing, feature extraction, and surgical prediction** based on deep learning and traditional machine learning methods.
本项目实现了一个基于深度学习与传统机器学习相结合的**脊椎检测、后处理、特征提取与手术预测**全流程系统。  

---

## Dataset Access | 数据集获取

The dataset used in this project is publicly shared on Google Drive and includes both pre- and post-surgery X-ray images as well as corresponding labels.

本项目所使用的数据集已在 Google Drive 上公开共享，包含 手术前后脊柱X光片 以及配套的标注文件。

Dataset Contents | 数据集内容说明
datasets  
├── before_surgery            # Pre-surgery X-ray images | 手术前X光片  
├── after_surgery             # Post-surgery X-ray images | 手术后X光片  
├── label_seg                 # Vertebra annotations in JSON and jpg format | 骨骼识别JSON和jpg标注  
├── label_screw.xlsx          # Screw placement labels | 加钉预测Excel标注文件    

🔗 Download Link | 下载链接

[👉 Click here to access the dataset on Google Drive](https://drive.google.com/drive/u/2/folders/1utVv9962s883051bhD2wOeqe8skQfJIH)

[👉 点击此处访问 Google Drive 数据集](https://drive.google.com/drive/u/2/folders/1utVv9962s883051bhD2wOeqe8skQfJIH)

## Project Structure | 项目结构  

```
├── sample/                     # Original sample data | 原始样本数据  
├── sample_label/               # Original annotation files | 原始标注文件  
├── yolo_spine_dataset/         # Converted dataset for YOLO training | 转换后用于训练的数据集  
├── spine_detection/            # Model training results | 模型训练结果  
├── results_obb/                # Model prediction results | 模型预测结果  
├── screw_results.xlsx          # Screw placement prediction results | 加钉预测结果表格  
│
├── convert_dataset.py          # Convert JSON annotations to TXT and split dataset | JSON标注转TXT并划分数据集  
├── train.py                    # Training script for the detection model | 模型训练脚本  
├── predict.py                  # Prediction script for detection results | 模型预测脚本  
├── post_processing.py          # Post-processing of prediction results | 预测结果后处理  
├── mask_with_direction.py      # Generate direction arrows and verify mask accuracy | 输出方向箭头与掩码验证  
├── txt_result_turn_mask.py     # Auxiliary script for mask verification | 掩码验证辅助脚本  
├── classification.py           # XGBoost classifier: compute Cobb angle & curvature features | XGBoost分类器，计算Cobb角度与曲率特征  
├── screw_predict.py            # Predict screw placement based on classification results | 是否加钉预测脚本  
└── test.py                     # Evaluate classifier accuracy | 分类器精度验证脚本  

```

## Key Features | 主要功能  
  
* Automatic spine detection and labeling  
* Post-processing  
* Feature extraction of Cobb angle & curvature  
* Screw placement prediction  
* XGBoost-based classification and evaluation  



