import pandas as pd
import numpy as np
import os
import glob
import joblib

# ========== 参数 ==========
MODEL_PATH = 'spine_nail_xgb.pkl'
SCALER_PATH = 'spine_nail_scaler.pkl'
SELECTOR_PATH = 'spine_nail_feature_selector.pkl'
TXT_DIR = r'E:\项目\python\strp_yolo\results_obb\exp'
OUTPUT_EXCEL = 'screw_results.xlsx'

def add_sliding_window_features(df, col, window=3):
    df[f'{col}_mean_w{window}'] = df[col].rolling(window, center=True, min_periods=1).mean()
    df[f'{col}_std_w{window}'] = df[col].rolling(window, center=True, min_periods=1).std().fillna(0)
    df[f'{col}_range_w{window}'] = df[col].rolling(window, center=True, min_periods=1).apply(lambda x: np.max(x)-np.min(x), raw=True)
    return df

# ========== 特征工程函数 ==========
def extract_features(df):
    df = df.sort_values('id').reset_index(drop=True)
    df['angle_deg'] = df['angle'] * 180 / np.pi

    # 全局特征
    k, b = np.polyfit(df['cx'], df['cy'], 1)
    df['coronal_offset'] = np.abs(df['cy'] - (k * df['cx'] + b))
    first = df.iloc[0]
    last = df.iloc[-1]
    L = np.linalg.norm([last['cx'] - first['cx'], last['cy'] - first['cy']])
    cobb_angle = df['angle_deg'].max() - df['angle_deg'].min()
    coronal_offset_rate = df['coronal_offset'].max() / L if L != 0 else 0
    sagittal_balance = (first['cy'] - last['cy']) / L if L != 0 else 0
    apex_row = df.loc[df['coronal_offset'].idxmax()]
    x1, y1 = df.iloc[0]['cx'], df.iloc[0]['cy']
    x2, y2 = df.iloc[-1]['cx'], df.iloc[-1]['cy']
    xa, ya = apex_row['cx'], apex_row['cy']
    num = abs((y2 - y1) * xa - (x2 - x1) * ya + x2 * y1 - y2 * x1)
    den = np.sqrt((y2 - y1) ** 2 + (x2 - x1) ** 2)
    apex_dist = num / den if den != 0 else 0
    apex_dist_rate = apex_dist / L if L != 0 else 0
    global_features = [cobb_angle, coronal_offset_rate, sagittal_balance, apex_dist_rate]

    # 局部特征
    df['norm_cx'] = (df['cx'] - first['cx']) / L if L != 0 else 0
    df['norm_cy'] = (df['cy'] - first['cy']) / L if L != 0 else 0
    df['norm_id'] = df.index / (len(df) - 1) if len(df) > 1 else 0
    df['height_ratio'] = df['height'] / df['height'].max()
    df['hw_ratio'] = df['height'] / df['width']

    # wedge
    h = df['height'].values
    wedge = np.zeros(len(df))
    for i in range(1, len(df) - 1):
        wedge[i] = 1 - h[i] / ((h[i - 1] + h[i + 1]) / 2)
    df['wedge'] = wedge

    # rot_grad
    theta = df['angle_deg'].values
    rot_grad = np.zeros(len(df))
    for i in range(1, len(df) - 1):
        rot_grad[i] = theta[i] - (theta[i - 1] + theta[i + 1]) / 2
    df['rot_grad'] = rot_grad

    # curvature
    x = df['cx'].values
    y = df['cy'].values
    curvature = np.zeros(len(df))
    for i in range(1, len(df) - 1):
        x1, x2, x3 = x[i - 1], x[i], x[i + 1]
        y1, y2, y3 = y[i - 1], y[i], y[i + 1]
        num = abs((x3 - x2) * (y2 - y1) - (y3 - y2) * (x2 - x1))
        denom = np.sqrt((x3 - x2) ** 2 + (y3 - y2) ** 2) * np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        curvature[i] = num / denom if denom != 0 else 0
    df['curvature'] = curvature

    # 拟合中线的距离
    k, b = np.polyfit(df['cx'], df['cy'], 1)
    df['coronal_offset'] = np.abs(df['cy'] - (k * df['cx'] + b))

    # 滑动窗口特征（必须在curvature生成后再做）
    for slide_col in ['height', 'angle_deg', 'coronal_offset', 'curvature']:
        df = add_sliding_window_features(df, slide_col, window=3)

    # 高级特征
    df['height_zscore'] = (df['height'] - df['height'].mean()) / (df['height'].std() + 1e-8)
    df['height_outlier'] = (np.abs(df['height'] - df['height'].mean()) > 2*df['height'].std()).astype(int)
    df['is_local_max'] = ((df['height'] > df['height'].shift(1)) & (df['height'] > df['height'].shift(-1))).astype(int)
    df['hw_curvature'] = df['hw_ratio'] * df['curvature']
    df['height_rotgrad'] = df['height_ratio'] * df['rot_grad']

    # 特征名顺序必须和训练时完全一致
    local_feat = [
        'norm_cx', 'norm_cy', 'norm_id', 'height_ratio', 'hw_ratio', 'angle_deg', 'wedge', 'rot_grad', 'curvature', 'coronal_offset',
        'height_mean_w3', 'height_std_w3', 'height_range_w3',
        'angle_deg_mean_w3', 'angle_deg_std_w3', 'angle_deg_range_w3',
        'coronal_offset_mean_w3', 'coronal_offset_std_w3', 'coronal_offset_range_w3',
        'curvature_mean_w3', 'curvature_std_w3', 'curvature_range_w3',
        'height_zscore', 'height_outlier', 'is_local_max', 'hw_curvature', 'height_rotgrad'
    ]

    features = []
    for _, row in df.iterrows():
        feat = global_features + [row[c] for c in local_feat]
        features.append(feat)
    return np.array(features)

# ========== 加载模型与工具 ==========
clf = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
selector = joblib.load(SELECTOR_PATH)

# ========== 读取txt批量预测 ==========
all_cases = {}  # {case_id: [结果字符串, ...]}

txt_files = glob.glob(os.path.join(TXT_DIR, '*.txt'))
for txt_file in txt_files:
    filename = os.path.basename(txt_file)
    case_id = filename.replace('_info.txt', '')

    df = pd.read_csv(txt_file, sep='\t')
    if len(df) < 17:
        print(f"[WARN] {txt_file} less than 17 rows, skip")
        continue

    features = extract_features(df)
    # 1. 标准化
    features_scaled = scaler.transform(features)
    # 2. 特征选择
    features_sel = selector.transform(features_scaled)
    # 3. 预测
    preds = clf.predict(features_sel)
    probs = clf.predict_proba(features_sel)[:, 1] if hasattr(clf, "predict_proba") else [None] * len(preds)

    # 每个病例一列，字符串格式为 0(0.12) 或 1(0.87)
    col = [f"{int(pred)}({prob:.2f})" if prob is not None else str(int(pred)) for pred, prob in zip(preds, probs)]
    # 保证只有17行
    if len(col) > 17:
        col = col[:17]
    elif len(col) < 17:
        col += [''] * (17 - len(col))
    all_cases[case_id] = col


# 构建DataFrame并输出Excel
case_ids = sorted(all_cases.keys())
matrix = pd.DataFrame({cid: all_cases[cid] for cid in case_ids})
matrix.index = [f"{i+1}" for i in range(17)]  # 行标签可选
matrix.to_excel(OUTPUT_EXCEL, index=True)
print(f"预测完成，结果已保存至 {OUTPUT_EXCEL}")

# 输出每条的特征数
print(f"每个预测样本的特征数: {features.shape[1]}")