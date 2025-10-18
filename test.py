import pandas as pd
import re

# 读取预测和标注表格
pred_path = 'screw_results.xlsx'  # 你的预测结果Excel文件
gt_path = 'label_screw_flipped.xlsx'      # 你的标注Excel文件

pred_raw = pd.read_excel(pred_path, header=None)
gt_raw = pd.read_excel(gt_path, header=None)

# 全部编号转str并strip
pred_ids = [str(x).strip() for x in pred_raw.iloc[0, 1:] if pd.notna(x)]
gt_ids = [str(x).strip() for x in gt_raw.iloc[0, 1:] if pd.notna(x)]

pred_ids_set = set(pred_ids)
gt_ids_set = set(gt_ids)

common_ids = sorted(list(pred_ids_set & gt_ids_set))

pred_header = [str(x).strip() for x in pred_raw.iloc[0, :]]
gt_header = [str(x).strip() for x in gt_raw.iloc[0, :]]

total = 0
correct = 0

# 新增混淆矩阵四个变量
tp = fp = tn = fn = 0

def parse_pred(cell):
    m = re.match(r'([01])', str(cell))
    return int(m.group(1)) if m else None

for case_id in common_ids:
    if case_id not in pred_header or case_id not in gt_header:
        print(f"Warning: 编号 {case_id} 在表头找不到，已跳过")
        continue
    pred_col = pred_header.index(case_id)
    gt_col = gt_header.index(case_id)
    pred_cells = pred_raw.iloc[1:18, pred_col].tolist()
    gt_cells = gt_raw.iloc[1:18, gt_col].tolist()
    pred_bin = [parse_pred(x) for x in pred_cells]
    gt_bin = [int(x) if str(x).strip() in ['0', '1'] else None for x in gt_cells]
    for p, g in zip(pred_bin, gt_bin):
        if p is not None and g is not None:
            total += 1
            if p == g:
                correct += 1
            # 混淆矩阵统计
            if g == 1 and p == 1:
                tp += 1
            elif g == 0 and p == 1:
                fp += 1
            elif g == 0 and p == 0:
                tn += 1
            elif g == 1 and p == 0:
                fn += 1

accuracy = correct / total if total > 0 else 0
recall = tp / (tp + fn) if (tp+fn) > 0 else 0
precision = tp / (tp + fp) if (tp+fp) > 0 else 0
f1 = 2 * precision * recall / (precision + recall) if (precision+recall) > 0 else 0

print(f"\n交集病例总对比数: {total}")
print(f"预测正确数: {correct}")
print(f"准确率: {accuracy:.4f}")
print(f"召回率: {recall:.4f}")
print(f"精确率: {precision:.4f}")
print(f"F1分数: {f1:.4f}")