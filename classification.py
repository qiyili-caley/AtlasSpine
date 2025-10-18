import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectFromModel
import joblib
import xgboost as xgb
import matplotlib.pyplot as plt


def find_or_create_label_file():
    """查找或创建标签文件"""
    # 首先尝试直接读取
    if os.path.exists('label_screw_flipped.xlsx'):
        return pd.read_excel('label_screw_flipped.xlsx', header=0, index_col=None,
                             usecols=lambda x: x not in ['ordinal'], engine="openpyxl")

    # 如果不存在，尝试在项目目录中搜索
    for root, dirs, files in os.walk('.'):
        if 'label_screw_flipped.xlsx' in files:
            file_path = os.path.join(root, 'label_screw_flipped.xlsx')
            print(f"在 {file_path} 找到文件")
            return pd.read_excel(file_path, header=0, index_col=None,
                                 usecols=lambda x: x not in ['ordinal'], engine="openpyxl")

    # 如果还是找不到，尝试重新生成
    print("未找到 label_screw_flipped.xlsx，尝试重新生成...")
    if os.path.exists('label_screw.xlsx'):
        df = pd.read_excel('label_screw.xlsx')
        for col in df.columns:
            df[str(col) + '_flipped'] = df[col]
        df.to_excel('label_screw_flipped.xlsx', index=False)
        print("已重新生成 label_screw_flipped.xlsx")
        return df
    else:
        raise FileNotFoundError("既找不到 label_screw_flipped.xlsx，也找不到 label_screw.xlsx")


# 使用修复后的函数
label_df = find_or_create_label_file()

# 剩下的代码保持不变...
features = []
labels = []
ids = []
label_dir = 'results_obb/exp'



def add_sliding_window_features(df, col, window=3):
    df[f'{col}_mean_w{window}'] = df[col].rolling(window, center=True, min_periods=1).mean()
    df[f'{col}_std_w{window}'] = df[col].rolling(window, center=True, min_periods=1).std().fillna(0)
    df[f'{col}_range_w{window}'] = df[col].rolling(window, center=True, min_periods=1).apply(lambda x: np.max(x)-np.min(x), raw=True)
    return df

# 1. 读取加钉标签
# label_df = pd.read_excel('label_screw_flipped.xlsx', header=0, index_col=None, usecols=lambda x: x not in ['ordinal'], engine="openpyxl")

# features = []
# labels = []
# ids = []
# label_dir = 'results_obb/exp'

# 记录特征名（顺序必须与local_feat一致！）
local_feat = [
    'norm_cx', 'norm_cy', 'norm_id', 'height_ratio', 'hw_ratio', 'angle_deg', 'wedge', 'rot_grad', 'curvature', 'coronal_offset',
    # 滑动窗口特征
    'height_mean_w3', 'height_std_w3', 'height_range_w3',
    'angle_deg_mean_w3', 'angle_deg_std_w3', 'angle_deg_range_w3',
    'coronal_offset_mean_w3', 'coronal_offset_std_w3', 'coronal_offset_range_w3',
    'curvature_mean_w3', 'curvature_std_w3', 'curvature_range_w3',
    # 高级交互与异常
    'height_zscore', 'height_outlier', 'is_local_max', 'hw_curvature', 'height_rotgrad'
]
global_feat_names = ['cobb_angle', 'coronal_offset_rate', 'sagittal_balance', 'apex_dist_rate']
all_feature_names = global_feat_names + local_feat

for col in label_df.columns:
    y_list = label_df[col].values  # 0/1，每一行对应一个椎骨
    label_ids = np.arange(1, len(y_list) + 1)
    label_df_this = pd.DataFrame({'id': label_ids, 'y': y_list})

    fpath = os.path.join(label_dir, f"{col}_info.txt")
    if not os.path.exists(fpath):
        print(f'[WARN] {fpath} not found, skip')
        continue

    df = pd.read_csv(fpath, sep='\t')
    df = df.sort_values('id').reset_index(drop=True)
    df['angle_deg'] = df['angle'] * 180 / np.pi

    # ========== 全局特征 ==========
    k, b = np.polyfit(df['cx'], df['cy'], 1)
    df['coronal_offset'] = np.abs(df['cy'] - (k*df['cx'] + b))
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

    # ========== 局部特征 ==========
    # 标准化坐标
    df['norm_cx'] = (df['cx'] - first['cx']) / L if L != 0 else 0
    df['norm_cy'] = (df['cy'] - first['cy']) / L if L != 0 else 0
    df['norm_id'] = df.index / (len(df) - 1) if len(df) > 1 else 0
    # 高度比
    df['height_ratio'] = df['height'] / df['height'].max()
    # 高宽比
    df['hw_ratio'] = df['height'] / df['width']

    # 楔形变（边界为0）
    h = df['height'].values
    wedge = np.zeros(len(df))
    for i in range(1, len(df) - 1):
        wedge[i] = 1 - h[i] / ((h[i - 1] + h[i + 1]) / 2)
    df['wedge'] = wedge
    # 旋转梯度（边界为0）
    theta = df['angle_deg'].values
    rot_grad = np.zeros(len(df))
    for i in range(1, len(df) - 1):
        rot_grad[i] = theta[i] - (theta[i - 1] + theta[i + 1]) / 2
    df['rot_grad'] = rot_grad
    # 曲率特征（两端为0）
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
    # 滑动窗口特征
    for slide_col in ['height', 'angle_deg', 'coronal_offset', 'curvature']:
        df = add_sliding_window_features(df, slide_col, window=3)
    # 到拟合中线的距离（coronal_offset）
    k, b = np.polyfit(df['cx'], df['cy'], 1)
    df['coronal_offset'] = np.abs(df['cy'] - (k * df['cx'] + b))

    # 高级特征工程补充
    df['height_zscore'] = (df['height'] - df['height'].mean()) / (df['height'].std() + 1e-8)
    df['height_outlier'] = (np.abs(df['height'] - df['height'].mean()) > 2*df['height'].std()).astype(int)
    df['is_local_max'] = ((df['height'] > df['height'].shift(1)) & (df['height'] > df['height'].shift(-1))).astype(int)
    df['hw_curvature'] = df['hw_ratio'] * df['curvature']
    df['height_rotgrad'] = df['height_ratio'] * df['rot_grad']

    merged = pd.merge(df, label_df_this, on='id', how='inner')
    if len(merged) == 0:
        print(f'[WARN] {col} 没有可用的id对齐，跳过')
        continue

    for _, row in merged.iterrows():
        feat = global_features + [row[c] for c in local_feat]
        features.append(feat)
        labels.append(row['y'])
        ids.append((col, int(row['id'])))

X = np.array(features)
y = np.array(labels)

# ========== 1. 标准化 ==========
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
joblib.dump(scaler, 'spine_nail_scaler.pkl')

print(f"样本数: {X.shape[0]}, 特征维度: {X.shape[1]}")

# ========== 2. 特征自动筛选 ==========
# 用一个弱XGBoost做特征筛选（不影响主模型效果）
selector_clf = xgb.XGBClassifier(
    n_estimators=80, max_depth=6, learning_rate=0.03,
    reg_alpha=2.0, reg_lambda=1.0,
    scale_pos_weight=np.sum(y == 0) / np.sum(y == 1),
     eval_metric='logloss', random_state=42
)
selector = SelectFromModel(selector_clf, threshold="median")  # 选一半重要的特征
selector.fit(X_scaled, y)
X_sel = selector.transform(X_scaled)
joblib.dump(selector, 'spine_nail_feature_selector.pkl')

# 输出被选中的特征名
selected_names = np.array(all_feature_names)[selector.get_support()]
print("被选中的特征：")
for name in selected_names:
    print(name)
print(f"筛选后特征数: {len(selected_names)}")

# ========== 3. 主模型训练 ==========
X_train, X_test, y_train, y_test = train_test_split(X_sel, y, test_size=0.1, random_state=42)
n_pos = np.sum(y_train == 1)
n_neg = np.sum(y_train == 0)
scale_pos_weight = n_neg / n_pos

clf = xgb.XGBClassifier(
    n_estimators=10000,
    max_depth=10,
    learning_rate=0.01,
    subsample=0.8,
    colsample_bytree=0.8,
    gamma=1.0,
    min_child_weight=6,
    scale_pos_weight=scale_pos_weight,
    reg_alpha=0.1,
    reg_lambda=1.0,
    eval_metric='logloss',
    random_state=42
)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

print(classification_report(y_test, y_pred))

# ========== 4. 特征重要性输出 ==========
importances = clf.feature_importances_
indices = np.argsort(importances)[::-1]

try:
    # 尝试使用seaborn绘图
    import seaborn as sns
    import matplotlib.pyplot as plt

    plt.figure(figsize=(12, 8))
    sns.barplot(x=importances[indices], y=[selected_names[i] for i in indices])
    plt.title('Top Feature Importances (XGBoost)')
    plt.xlabel('Importance')
    plt.tight_layout()
    plt.savefig("feature_importance.png", dpi=300, bbox_inches='tight')
    plt.show(block=False)  # 非阻塞模式
    plt.pause(2)  # 显示2秒后继续
    plt.close()   # 关闭图形
    print("特征重要性图已保存")

except ImportError:
    print("seaborn未安装，使用纯文本输出")
except Exception as e:
    print(f"绘图失败: {e}")

print("特征重要性TOP10:")
for i in range(min(10, len(importances))):
    print(f"{selected_names[indices[i]]}: {importances[indices[i]]:.4f}")

# ========== 5. 保存模型 ==========
joblib.dump(clf, 'spine_nail_xgb.pkl')
# 在您现有的XGBoost训练代码之后添加：

# ========== 6. 逻辑回归分析 ==========
print("\n" + "=" * 60)
print("开始逻辑回归分析")
print("=" * 60)

try:
    print("正在导入statsmodels...")
    import statsmodels.api as sm

    # 使用与XGBoost相同的特征数据
    print(f"逻辑回归分析 - 样本数: {X_sel.shape[0]}, 特征数: {X_sel.shape[1]}")

    # 添加截距项
    print("添加截距项...")
    X_with_intercept = sm.add_constant(X_sel)

    # 使用更稳定的拟合方法
    print("开始拟合逻辑回归模型...")
    logit_model = sm.Logit(y, X_with_intercept)

    # 使用更保守的参数
    logit_result = logit_model.fit(
        disp=1,  # 显示迭代信息
        maxiter=50,  # 减少最大迭代次数
        method='bfgs',  # 使用不同的优化算法
        tol=1e-4  # 降低收敛精度
    )

    print("模型拟合完成，生成结果...")

    # 创建详细结果表
    coefficients = logit_result.params
    p_values = logit_result.pvalues
    odds_ratios = np.exp(coefficients)

    results_table = pd.DataFrame({
        'Feature': ['Intercept'] + list(selected_names),
        'Coefficient': coefficients,
        'P_Value': p_values,
        'Odds_Ratio': odds_ratios
    })

    # 按P值排序
    results_table = results_table.sort_values('P_Value')

    print("\n逻辑回归分析结果 (按P值排序):")
    print("=" * 70)

    significant_count = 0
    for _, row in results_table.iterrows():
        if row['Feature'] != 'Intercept':
            star = "***" if row['P_Value'] < 0.001 else "**" if row['P_Value'] < 0.01 else "*" if row[
                                                                                                      'P_Value'] < 0.05 else ""
            direction = "增加" if row['Coefficient'] > 0 else "减少"
            if row['P_Value'] < 0.05:
                significant_count += 1
            print(
                f"{star} {row['Feature']:25} {direction:4}螺钉需求 (OR={row['Odds_Ratio']:.3f}, p={row['P_Value']:.4f})")

    print(f"\n显著变量 (P < 0.05): {significant_count}个")

    # 保存结果
    results_table.to_csv('spine_screw_logistic_results.csv', index=False)
    print(f"详细结果已保存至: 'spine_screw_logistic_results.csv'")

except Exception as e:
    print(f"逻辑回归分析错误: {e}")
    print("尝试替代方案...")

    # 替代方案：使用sklearn的逻辑回归
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import log_loss

        print("使用sklearn进行逻辑回归分析...")

        # 使用sklearn的逻辑回归
        lr_model = LogisticRegression(
            penalty='none',  # 无正则化
            max_iter=1000,
            random_state=42
        )
        lr_model.fit(X_sel, y)

        # 计算P值的近似值（基于系数和标准误）
        from scipy import stats

        # 预测概率
        pred_proba = lr_model.predict_proba(X_sel)

        # 计算标准误（近似）
        n = X_sel.shape[0]
        p = X_sel.shape[1]
        standard_errors = np.sqrt(np.diag(np.linalg.pinv(np.dot(X_sel.T, X_sel)))) * np.sqrt(
            log_loss(y, pred_proba) / (n - p - 1))

        # 计算t统计量和P值
        t_stats = lr_model.coef_[0] / standard_errors
        p_values = 2 * (1 - stats.t.cdf(np.abs(t_stats), n - p - 1))

        # 创建结果表
        results_table = pd.DataFrame({
            'Feature': list(selected_names),
            'Coefficient': lr_model.coef_[0],
            'P_Value': p_values,
            'Odds_Ratio': np.exp(lr_model.coef_[0])
        })

        # 按P值排序
        results_table = results_table.sort_values('P_Value')

        print("\nSklearn逻辑回归分析结果 (按P值排序):")
        print("=" * 70)

        significant_count = 0
        for _, row in results_table.iterrows():
            star = "***" if row['P_Value'] < 0.001 else "**" if row['P_Value'] < 0.01 else "*" if row[
                                                                                                      'P_Value'] < 0.05 else ""
            direction = "增加" if row['Coefficient'] > 0 else "减少"
            if row['P_Value'] < 0.05:
                significant_count += 1
            print(
                f"{star} {row['Feature']:25} {direction:4}螺钉需求 (OR={row['Odds_Ratio']:.3f}, p={row['P_Value']:.4f})")

        print(f"\n显著变量 (P < 0.05): {significant_count}个")

        # 保存结果
        results_table.to_csv('spine_screw_logistic_results_sklearn.csv', index=False)
        print(f"详细结果已保存至: 'spine_screw_logistic_results_sklearn.csv'")

    except Exception as e2:
        print(f"Sklearn替代方案也失败: {e2}")

# ========== 7. 综合模型评估 ==========
print("\n" + "=" * 60)
print("启动独立评估模块")
print("=" * 60)

try:
    # 导入评估类
    from model_evaluation import SpineModelEvaluator

    # 创建评估器实例 - 使用当前作用域的变量
    evaluator = SpineModelEvaluator(
        model_path='spine_nail_xgb.pkl',
        scaler_path='spine_nail_scaler.pkl',
        selector_path='spine_nail_feature_selector.pkl',
        feature_names=selected_names  # 这个变量应该存在
    )

    # 执行综合评估 - 传递所有需要的变量
    evaluation_results = evaluator.evaluate_model(
        X_test=X_test,  # 测试集特征
        y_test=y_test,  # 测试集标签
        X_sel=X_sel,  # 筛选后的特征（用于后续分析）
        y=y,  # 完整标签
        selected_names=selected_names  # 筛选后的特征名
    )

    print("\n" + "=" * 60)
    print("所有评估完成！")
    print("=" * 60)

except NameError as e:
    print(f"变量未定义错误: {e}")
    print("请检查以下变量是否已定义:")
    print(f"  selected_names: {'selected_names' in locals()}")
    print(f"  X_test: {'X_test' in locals()}")
    print(f"  y_test: {'y_test' in locals()}")
    print(f"  X_sel: {'X_sel' in locals()}")
    print(f"  y: {'y' in locals()}")
except Exception as e:
    print(f"评估模块执行失败: {e}")
    import traceback


    traceback.print_exc()
