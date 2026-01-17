import numpy as np
import torch
from typing import List, Dict, Tuple
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

import torch.nn.functional as F

# sklearn 是可选依赖：很多训练环境没有装 sklearn，这里提供 fallback 实现避免直接崩溃
try:
    from sklearn.metrics import f1_score as _sk_f1_score
    from sklearn.metrics import average_precision_score as _sk_average_precision_score
    from sklearn.metrics import roc_auc_score as _sk_roc_auc_score
except Exception:
    _sk_f1_score = None
    _sk_average_precision_score = None
    _sk_roc_auc_score = None


def _f1_score_binary(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Binary F1, y_true/y_pred are 0/1 arrays."""
    y_true = y_true.astype(np.int64)
    y_pred = y_pred.astype(np.int64)
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    denom = (2 * tp + fp + fn)
    return float(0.0 if denom == 0 else (2 * tp) / denom)


def _average_precision_binary(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """
    Average precision (AP) for binary labels.
    Implements the step-wise area under precision-recall curve (like sklearn).
    """
    y_true = y_true.astype(np.int64)
    y_score = y_score.astype(np.float64)
    # sort by score desc
    order = np.argsort(-y_score, kind="mergesort")
    y_true = y_true[order]
    y_score = y_score[order]

    # if no positive, AP = 0 by convention (sklearn returns 0.0)
    n_pos = int(np.sum(y_true == 1))
    if n_pos == 0:
        return 0.0

    tp = np.cumsum(y_true == 1)
    fp = np.cumsum(y_true == 0)
    precision = tp / np.maximum(tp + fp, 1)
    recall = tp / n_pos

    # AP: sum over recall increments * precision at that point
    # Find indices where score changes (optional), but simplest step integration:
    # integrate precision as a function of recall using trapezoid on step is slightly off;
    # use standard "precision at each positive" formulation:
    pos_idx = np.where(y_true == 1)[0]
    return float(np.mean(precision[pos_idx]))


def _roc_auc_binary(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """
    ROC-AUC for binary labels using rank statistic (Mann–Whitney U).
    Handles ties by average ranks.
    """
    y_true = y_true.astype(np.int64)
    y_score = y_score.astype(np.float64)
    n_pos = int(np.sum(y_true == 1))
    n_neg = int(np.sum(y_true == 0))
    if n_pos == 0 or n_neg == 0:
        # undefined; sklearn would raise. For our eval, return None-like sentinel via exception.
        raise ValueError("ROC AUC is undefined with no positive or no negative samples.")

    # rank data with average ranks for ties
    order = np.argsort(y_score, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(y_score) + 1, dtype=np.float64)

    # average ranks for ties
    sorted_scores = y_score[order]
    i = 0
    while i < len(sorted_scores):
        j = i
        while j + 1 < len(sorted_scores) and sorted_scores[j + 1] == sorted_scores[i]:
            j += 1
        if j > i:
            avg_rank = (i + 1 + j + 1) / 2.0
            ranks[order[i:j + 1]] = avg_rank
        i = j + 1

    sum_ranks_pos = float(np.sum(ranks[y_true == 1]))
    auc = (sum_ranks_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auc)

def geometric_projection(pred_3d_vec, gt_2d_vec):
    """
    CVPR 2025: Geometric Projection (GP)
    Args:
        pred_3d_vec: 预训练模型预测的 3D 向量 [B, 3] (x, y, z)
        gt_2d_vec:   GazeFollow 的 2D 真实标签 [B, 2] (x, y)
    Returns:
        pseudo_3d_vec: 修正后的 3D 伪标签 [B, 3]
    """
    # 1. 获取预测向量在 XY 平面上的模长 r = sqrt(x^2 + y^2)
    # 这代表模型认为视线偏离 Z 轴（深度）的程度
    pred_xy = pred_3d_vec[:, :2]
    r_pred = torch.norm(pred_xy, dim=1, keepdim=True) + 1e-8

    # 2. 归一化 2D GT (只取方向)
    # 这一步对应论文中让 projected gaze aligned with 2D ground truth
    norm_gt = torch.norm(gt_2d_vec, dim=1, keepdim=True) + 1e-8
    unit_gt_2d = gt_2d_vec / norm_gt

    # 3. 组合: 新的 XY = (2D GT的方向) * (预测的XY模长)
    pseudo_xy = unit_gt_2d * r_pred

    # 4. 保留预测的 Z 分量
    pseudo_z = pred_3d_vec[:, 2:]

    # 5. 拼接并归一化 (确保是单位向量)
    pseudo_label = torch.cat([pseudo_xy, pseudo_z], dim=1)
    pseudo_label = F.normalize(pseudo_label, p=2, dim=1)

    return pseudo_label

def get_2d_gaze_vector(eye_norm, gaze_norm):
    """
    计算2D平面上的单位方向向量 (x, y)
    """
    dx = gaze_norm['x'] - eye_norm['x']
    dy = gaze_norm['y'] - eye_norm['y']
    
    # 计算模长
    norm = np.sqrt(dx**2 + dy**2)
    
    # 防止除以0 (极少数情况眼睛和注视点重合)
    if norm < 1e-6:
        return [0.0, 0.0]
    
    # 归一化，只保留方向信息
    return [dx / norm, dy / norm]

# ==========================================
# 1. 几何计算 (Geometry)
# ==========================================

def angles_to_2d_vector_torch(angles):
    """ Tensor [B, 2] (yaw, pitch) -> Tensor [B, 2] (dx, dy) """
    yaw = angles[:, 0]
    pitch = angles[:, 1]
    
    dx = -torch.cos(pitch) * torch.sin(yaw)
    dy = -torch.sin(pitch)
    
    vec = torch.stack([dx, dy], dim=1)
    norm = torch.norm(vec, dim=1, keepdim=True) + 1e-6
    return vec / norm

def gazeto3d(gaze):
    """ 
    将 (yaw, pitch) 转换为 3D 向量 (x, y, z)
    兼容 List, Tuple, Numpy, Tensor
    """
    # 1. 提取 yaw, pitch 数值
    if isinstance(gaze, (list, tuple, np.ndarray)):
        # 确保是 flat 的
        gaze = np.array(gaze).flatten()
        if gaze.size != 2:
            raise ValueError(f"gazeto3d expected input size 2 (yaw, pitch), got {gaze.size}")
        yaw = float(gaze[0])
        pitch = float(gaze[1])
    elif isinstance(gaze, torch.Tensor):
        gaze = gaze.flatten()
        yaw = gaze[0].item()
        pitch = gaze[1].item()
    else:
        raise TypeError(f"Unsupported type: {type(gaze)}")

    gaze_gt = np.zeros([3])
    # 坐标系定义: x = -cos(p)sin(y), y = -sin(p), z = -cos(p)cos(y)
    gaze_gt[0] = -np.cos(pitch) * np.sin(yaw)
    gaze_gt[1] = -np.sin(pitch)
    gaze_gt[2] = -np.cos(pitch) * np.cos(yaw)
    
    return gaze_gt

def calculate_3d_angular_error(gaze_pred: np.ndarray, gaze_gt: np.ndarray) -> float:
    """ 计算 3D 角度误差 (Degree) """
    v_pred = gazeto3d(gaze_pred)
    v_gt = gazeto3d(gaze_gt)
    
    total = np.sum(v_pred * v_gt)
    norm = np.linalg.norm(v_pred) * np.linalg.norm(v_gt)
    cos_val = total / (norm + 1e-7)
    cos_val = np.clip(cos_val, -1.0, 1.0)
    return np.arccos(cos_val) * 180 / np.pi

# def calculate_2d_angular_error(pred_vec: np.ndarray, gt_vec: np.ndarray) -> float:
#     """ 计算 2D 平面向量的角度误差 (Degree) """
#     # pred_vec, gt_vec: [dx, dy]
#     # 归一化
#     pred_n = pred_vec / (np.linalg.norm(pred_vec) + 1e-7)
#     gt_n = gt_vec / (np.linalg.norm(gt_vec) + 1e-7)
    
#     cos_val = np.sum(pred_n * gt_n)
#     cos_val = np.clip(cos_val, -1.0, 1.0)
#     return np.arccos(cos_val) * 180 / np.pi

# ==========================================
# 2. 热图解码 (Heatmap Decoding)
# ==========================================

def get_heatmap_preds(heatmaps: torch.Tensor) -> np.ndarray:
    """
    从热图中提取最大值坐标 (x, y)，归一化到 [0, 1]
    heatmaps: [B, H, W] or [B, 1, H, W]
    Returns: [B, 2] (x, y)
    """
    if heatmaps.dim() == 4:
        heatmaps = heatmaps.squeeze(1) # [B, H, W]
    
    B, H, W = heatmaps.shape
    device = heatmaps.device
    
    # 展平寻找最大值索引
    heatmaps_flat = heatmaps.view(B, -1) # [B, H*W]
    max_vals, max_indices = torch.max(heatmaps_flat, dim=1)
    
    # 计算坐标
    preds_y = (max_indices // W).float() / H
    preds_x = (max_indices % W).float() / W
    
    # Stack [x, y]
    coords = torch.stack([preds_x, preds_y], dim=1) # [B, 2]
    return coords.cpu().numpy()

# ==========================================
# 3. 评估指标 (Metrics)
# ==========================================

def l2_dist(p1, p2):
    return np.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)

def gazefollow_auc(heatmap, gt_gazex, gt_gazey, height, width):
    """
    计算 GazeFollow AUC 指标
    heatmap: [H_map, W_map] (Tensor)
    gt_gazex, gt_gazey: List of normalized coordinates
    height, width: Evaluation grid size (e.g. 224 or 448)
    """
    target_map = np.zeros((height, width))
    for x_norm, y_norm in zip(gt_gazex, gt_gazey):
        if x_norm >= 0 and y_norm >= 0:
            x, y = int(x_norm * float(width)), int(y_norm * float(height))
            x = min(x, width - 1)
            y = min(y, height - 1)
            target_map[y, x] = 1
            
    # 如果没有有效的 GT 点，返回 None
    if np.sum(target_map) == 0:
        return None

    # Resize heatmap to evaluation grid
    # Input heatmap is [H, W], need [1, 1, H, W] for interpolate
    resized_heatmap = torch.nn.functional.interpolate(
        heatmap.unsqueeze(0).unsqueeze(0), 
        (height, width), 
        mode='bilinear', 
        align_corners=False
    ).squeeze() # -> [H, W]
    
    auc = roc_auc_score(target_map.flatten(), resized_heatmap.cpu().numpy().flatten())
    return auc

def compute_l2_metrics(pred_pt: np.ndarray, gt_points_list: List[Dict]) -> Tuple[float, float]:
    """
    pred_pt: [x, y]
    gt_points_list: List of dict {'x':..., 'y':...} or List of list [x, y]
    """
    dists = []
    for gt in gt_points_list:
        # 兼容 dataset 可能返回的两种格式
        if isinstance(gt, dict):
            gt_pt = [gt['x'], gt['y']]
        else:
            gt_pt = gt # list or array
        
        dists.append(l2_dist(pred_pt, gt_pt))
    
    return np.mean(dists), np.min(dists)

def compute_io_metrics(pred_probs: List[float], gt_labels: List[int]):
    """ In/Out F1 & AP """
    preds_cls = [1 if p > 0.5 else 0 for p in pred_probs]
    f1 = f1_score(gt_labels, preds_cls, zero_division=0)
    ap = average_precision_score(gt_labels, pred_probs) if len(gt_labels) > 0 else 0.0
    return f1, ap

def compute_bleu_metrics(pred_str: str, gt_str_list: List[str]):
    """ BLEU-4 """
    # NLTK expect list of tokens
    refs = [s.split() for s in gt_str_list]
    hyp = pred_str.split()
    
    smooth = SmoothingFunction().method1
    # weights=(0.25, 0.25, 0.25, 0.25) -> BLEU-4
    score = sentence_bleu(refs, hyp, smoothing_function=smooth)
    
    # Max BLEU logic (Standard is just average against references, but here we return same for compatibility)
    return score, score