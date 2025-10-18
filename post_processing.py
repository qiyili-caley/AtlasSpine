import os
import numpy as np
import re
from scipy.interpolate import splprep, splev

def clean_path(path):
    return re.sub(r'[\x00-\x1F\x7F]', '', path.strip())

def fit_spine_curve(cx_list, cy_list, s=5):
    tck, u = splprep([cx_list, cy_list], s=s)
    return tck, u

def get_curve_tangent_angle(tck, u_query):
    dx, dy = splev(u_query, tck, der=1)
    angle = np.arctan2(dy, dx)
    return angle

def angle_diff(a, b):
    diff = np.arctan2(np.sin(a-b), np.cos(a-b))
    return abs(diff)

def remove_curve_outliers(vertebrae, max_dist=80, min_n=12):
    if len(vertebrae) < 3:
        return vertebrae
    cx_list = [v['cx'] for v in vertebrae]
    cy_list = [v['cy'] for v in vertebrae]
    tck, u = fit_spine_curve(cx_list, cy_list, s=5)
    fit_cx, fit_cy = splev(u, tck)
    filtered = []
    for v in vertebrae:
        dists = [(v['cx'] - x) ** 2 + (v['cy'] - y) ** 2 for x, y in zip(fit_cx, fit_cy)]
        min_dist = np.sqrt(min(dists))
        filtered.append((v, min_dist))
    # 距离排序，保留距离最近的min_n个
    filtered = sorted(filtered, key=lambda x: x[1])
    keep = [v for v,d in filtered if d < max_dist]
    if len(keep) < min_n:
        keep = [v for v,d in filtered[:min_n]]
    keep = sorted(keep, key=lambda v: v['cy'])
    return keep

def adjust_angles_for_vertebrae(vertebrae):
    if len(vertebrae) < 2:
        return vertebrae
    vertebrae = sorted(vertebrae, key=lambda v: v['cy'])
    cx_list = [v['cx'] for v in vertebrae]
    cy_list = [v['cy'] for v in vertebrae]
    tck, u = fit_spine_curve(cx_list, cy_list, s=5)
    for idx, v in enumerate(vertebrae):
        cx, cy, angle_raw = v['cx'], v['cy'], v['angle']
        w, h = v['w'], v['h']
        dists = [(cx - x)**2 + (cy - y)**2 for x, y in zip(cx_list, cy_list)]
        i_min = np.argmin(dists)
        u_query = u[i_min]
        curve_angle = get_curve_tangent_angle(tck, u_query)
        candidates = [angle_raw + k*np.pi/2 for k in range(4)]
        best_angle = angle_raw
        min_delta = 1e9
        best_k = 0
        for k, cand in enumerate(candidates):
            delta = angle_diff(cand, curve_angle)
            if delta < min_delta:
                min_delta = delta
                best_angle = cand
                best_k = k
        best_angle = (best_angle + np.pi) % (2 * np.pi) - np.pi
        if best_k % 2 == 1:
            w, h = h, w
        v['angle'] = best_angle
        v['w'], v['h'] = w, h
    return vertebrae

def read_label(label_path):
    vertebrae = []
    with open(label_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        if not lines:
            return vertebrae, ""
        header = lines[0]
        for line in lines[1:]:
            parts = line.strip().split('\t')
            if len(parts) == 1:
                parts = line.strip().split()
            if len(parts) == 1:
                parts = line.strip().split(',')
            if len(parts) != 6:
                continue
            id, cx, cy, w, h, angle = parts
            vertebrae.append({
                'id': int(id),
                'cx': float(cx),
                'cy': float(cy),
                'w': float(w),
                'h': float(h),
                'angle': float(angle)
            })
    return vertebrae, header

def keep_17_vertebrae(bone_info):
    bone_info = sorted(bone_info, key=lambda x: x['cy'])
    n = len(bone_info)
    if n <= 17:
        kept = bone_info
    elif 18 <= n <= 20:
        kept = bone_info[n-17:n]
    else:
        kept = bone_info[3:20]
    for i, b in enumerate(kept, 1):
        b['id'] = i
    return kept

def write_label(label_path, vertebrae, header):
    with open(label_path, 'w', encoding='utf-8') as f:
        f.write(header.strip() + '\n')
        for v in vertebrae:
            line = f"{v['id']}\t{v['cx']:.3f}\t{v['cy']:.3f}\t{v['w']:.3f}\t{v['h']:.3f}\t{v['angle']:.6f}\n"
            f.write(line)

if __name__ == "__main__":
    label_dir = clean_path(r"E:\项目\python\strp_yolo\results_obb\exp")
    for fname in os.listdir(label_dir):
        if not fname.endswith('_info.txt'):
            continue
        path = os.path.join(label_dir, fname)
        bone_info, header = read_label(path)
        bone_info = remove_curve_outliers(bone_info, max_dist=80, min_n=12)       # 1. 曲线距离法剔除异常点
        bone_info = adjust_angles_for_vertebrae(bone_info)                        # 2. 角度宽高修正
        kept_bone_info = keep_17_vertebrae(bone_info)                             # 3. 规范为17块
        write_label(path, kept_bone_info, header)
        print(f"{fname}: outlier removed, angle adjusted, kept {len(kept_bone_info)} vertebrae and overwrote file.")