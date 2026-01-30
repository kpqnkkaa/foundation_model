import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import argparse
from datetime import datetime
import numpy as np
import os
import random
import torch
import torch.nn as nn
import wandb
import logging
from tqdm import tqdm
import cv2
import math
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
# 新增导入 Tokenizer
from transformers import GPT2Tokenizer

from gazelle.backbone import SAMBackboneWrapper 
from gazelle.model import GazeLLE 
import torchvision.transforms.functional as F_vis
import torch.nn.functional as F

cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)

os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"

# 确保这里的导入指向你提供的 dataloader.py
from gazelle.dataloader import GazeDataset, GazeEstimationDataset, collate_fn
from gazelle.model import get_gazelle_model
from gazelle.utils import gazefollow_auc, gazefollow_l2

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default="dinov2_vitb_multi_input")
parser.add_argument('--data_path', type=str, default='/mnt/nvme1n1/lululemon/xjj/datasets/resized/gazefollow_extended')
parser.add_argument('--is_mix_gaze_estimation', default=False, action='store_true')
parser.add_argument('--estimation_batch_size', type=int, default=64)
parser.add_argument('--eth_label', type=str, default='/mnt/nvme1n1/lululemon/xjj/datasets/ETH-Gaze/Label/train_temp.label')
parser.add_argument('--eth_img', type=str, default='/mnt/nvme1n1/lululemon/xjj/datasets/ETH-Gaze/Image')
parser.add_argument('--gaze360_label', type=str, default='/mnt/nvme1n1/lululemon/xjj/datasets/Gaze360/Label/train.label')
parser.add_argument('--gaze360_img', type=str, default='/mnt/nvme1n1/lululemon/xjj/datasets/Gaze360/Image')
parser.add_argument('--ckpt_save_dir', type=str, default='./experiments')
parser.add_argument('--wandb_project', type=str, default=None)
parser.add_argument('--exp_name', type=str, default=None)
parser.add_argument('--is_partial_input', default=False, action='store_true')
parser.add_argument('--log_iter', type=int, default=10)
parser.add_argument('--max_epochs', type=int, default=15)
parser.add_argument('--batch_size', type=int, default=60)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--n_workers', type=int, default=8)
args = parser.parse_args()

# ==========================================
# 可视化核心函数
# ==========================================
def visualize_comparison(image_path, bbox, 
                         gt_point, gt_mask, gt_text, gt_dir_idx,
                         pred_point, pred_mask, pred_text, pred_dir_idx,
                         save_path):
    """
    左右对比可视化：
    Left: Ground Truth (Red Theme) - 红色系
    Right: Prediction (Blue Theme) - 蓝色系
    """
    # 1. 加载基础图像
    img = Image.open(image_path).convert('RGB')
    w, h = img.size
    
    # 2. 创建左右子图 (1行2列)
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    
    # --- 辅助函数：绘制方向箭头 ---
    def draw_arrow(ax, cx, cy, dir_idx, color):
        # 0:Right, 1:TR, 2:Up, 3:TL, 4:Left, 5:BL, 6:Down, 7:BR
        idx_to_angle = {
            0: 0, 1: 45, 2: 90, 3: 135, 
            4: 180, 5: 225, 6: 270, 7: 315
        }
        if dir_idx is not None and dir_idx != -100:
            # 兼容 Tensor 或 int
            d_i = dir_idx.item() if torch.is_tensor(dir_idx) else int(dir_idx)
            
            if d_i in idx_to_angle:
                angle_deg = idx_to_angle[d_i]
                angle_rad = math.radians(angle_deg)
                arrow_len = w * 0.15
                # 图像坐标系 Y 向下，往上(90度)需要负 dy
                dx = math.cos(angle_rad) * arrow_len
                dy = -math.sin(angle_rad) * arrow_len 
                ax.arrow(cx, cy, dx, dy, color=color, width=w*0.005, head_width=w*0.02, 
                         length_includes_head=True)

    # 获取 Observer BBox 中心点 (用于画箭头起点)
    bx1, by1, bx2, by2 = bbox # Normalized
    cx, cy = (bx1 + bx2) / 2 * w, (by1 + by2) / 2 * h

    # ==========================================
    # LEFT Plot: Ground Truth (Red)
    # ==========================================
    ax = axes[0]
    ax.imshow(img)
    ax.set_title(f"Ground Truth\n{gt_text}", fontsize=12, color='darkred', wrap=True)
    ax.axis('off')
    
    # 1. GT Observer Box (Red)
    rect = patches.Rectangle((bx1*w, by1*h), (bx2-bx1)*w, (by2-by1)*h, 
                             linewidth=2, edgecolor='red', facecolor='none')
    ax.add_patch(rect)
    
    # 2. GT Point (Red Circle)
    if gt_point is not None:
        ax.scatter(gt_point[0]*w, gt_point[1]*h, c='red', s=100, marker='o', edgecolors='white', linewidth=2)
        
    # 3. GT Seg Mask (Red Overlay)
    if gt_mask is not None:
        if torch.is_tensor(gt_mask): g_m = gt_mask.detach().cpu().numpy()
        else: g_m = gt_mask
        if g_m.ndim == 3: g_m = g_m[0]
        
        # Resize 到原图
        g_m_resized = cv2.resize(g_m, (w, h), interpolation=cv2.INTER_NEAREST)
        overlay = np.zeros((h, w, 4))
        overlay[..., 0] = 1.0 # R 通道
        # Alpha: 只要 > 0.5 就算前景，透明度 0.4
        overlay[..., 3] = (g_m_resized > 0.5).astype(np.float32) * 0.4
        ax.imshow(overlay, interpolation='nearest')
        
    # 4. GT Direction (Red Arrow)
    draw_arrow(ax, cx, cy, gt_dir_idx, 'red')

    # ==========================================
    # RIGHT Plot: Prediction (Blue)
    # ==========================================
    ax = axes[1]
    ax.imshow(img)
    ax.set_title(f"Prediction\n{pred_text}", fontsize=12, color='darkblue', wrap=True)
    ax.axis('off')
    
    # 1. Observer Box (Blue - 用于 Context)
    rect = patches.Rectangle((bx1*w, by1*h), (bx2-bx1)*w, (by2-by1)*h, 
                             linewidth=2, edgecolor='blue', facecolor='none')
    ax.add_patch(rect)
    
    if pred_point is not None:
        ax.scatter(pred_point[0]*w, pred_point[1]*h, c='blue', s=100, marker='o', edgecolors='white', linewidth=2)

    # 3. Pred Seg Mask (Blue Overlay)
    if pred_mask is not None:
        if torch.is_tensor(pred_mask): p_m = pred_mask.detach().cpu().numpy()
        else: p_m = pred_mask
        if p_m.ndim == 3: p_m = p_m[0]
        
        p_m_resized = cv2.resize(p_m, (w, h))
        overlay = np.zeros((h, w, 4))
        overlay[..., 2] = 1.0 # B 通道
        # Alpha: 强力显示，只要概率 > 0.1 就显示，透明度 0.6
        overlay[..., 3] = (p_m_resized > 0.1).astype(np.float32) * 0.6
        ax.imshow(overlay, interpolation='nearest')
        
    # 4. Pred Direction (Blue Arrow)
    draw_arrow(ax, cx, cy, pred_dir_idx, 'blue')

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', dpi=150)
    plt.close()

# ==========================================
# 辅助函数
# ==========================================
def merge_batches(batches):
    batches = [b for b in batches if b is not None]
    if len(batches) == 0: return None
    if len(batches) == 1: return batches[0]
    merged = []
    num_items = len(batches[0]) 
    for i in range(num_items):
        item_example = batches[0][i]
        if isinstance(item_example, torch.Tensor):
            merged.append(torch.cat([b[i] for b in batches], dim=0))
        elif isinstance(item_example, list):
            new_list = []
            for b in batches: new_list.extend(b[i])
            merged.append(new_list)
        elif isinstance(item_example, tuple):
            new_list = []
            for b in batches: new_list.extend(list(b[i]))
            merged.append(new_list)
        else:
            raise TypeError(f"Unsupported type: {type(item_example)}")
    return tuple(merged)

def get_infinite_iter(loader):
    iterator = iter(loader)
    while True:
        try:
            yield next(iterator)
        except StopIteration:
            iterator = iter(loader)
            yield next(iterator)

class DataParallelWrapper(nn.DataParallel):
    def forward(self, input_dict):
        if 'images' in input_dict: batch_size = input_dict['images'].shape[0]
        else: first_key = next(iter(input_dict)); batch_size = len(input_dict[first_key])
        num_replicas = len(self.device_ids)
        split_size = (batch_size + num_replicas - 1) // num_replicas
        replicas_inputs = []
        for i in range(num_replicas):
            start_idx = i * split_size; end_idx = min((i + 1) * split_size, batch_size)
            if start_idx >= end_idx: break
            target_device = self.device_ids[i]; replica_dict = {}
            for k, v in input_dict.items():
                if isinstance(v, torch.Tensor): replica_dict[k] = v[start_idx:end_idx].to(target_device)
                elif isinstance(v, list): replica_dict[k] = v[start_idx:end_idx]
                elif v is None: replica_dict[k] = None
            replicas_inputs.append((replica_dict,))
        replicas = self.replicate(self.module, self.device_ids[:len(replicas_inputs)])
        outputs = self.parallel_apply(replicas, replicas_inputs, None)
        gathered_output = {}
        if len(outputs) > 0:
            for key in outputs[0].keys():
                first_val = outputs[0][key]
                if first_val is None: gathered_output[key] = None; continue
                if isinstance(first_val, torch.Tensor) and first_val.ndim == 0:
                      stacked_loss = torch.stack([out[key].to(self.output_device) for out in outputs])
                      gathered_output[key] = stacked_loss.mean()
                elif isinstance(first_val, torch.Tensor) and first_val.ndim > 0:
                    gathered_output[key] = torch.cat([out[key].to(self.output_device) for out in outputs], dim=0)
                elif isinstance(first_val, list):
                    merged_list = []
                    for out in outputs:
                        val_list = out[key]; 
                        if len(val_list) > 0 and isinstance(val_list[0], torch.Tensor): merged_list.extend([t.to(self.output_device) for t in val_list])
                        else: merged_list.extend(val_list)
                    gathered_output[key] = merged_list
        return gathered_output

def setup_logger(log_file):
    logger = logging.getLogger('gazelle_logger'); logger.setLevel(logging.INFO)
    if not logger.handlers:
        fh = logging.FileHandler(log_file, mode='w'); fh.setFormatter(logging.Formatter('%(asctime)s - %(message)s')); logger.addHandler(fh)
        sh = logging.StreamHandler(); sh.setFormatter(logging.Formatter('%(asctime)s - %(message)s')); logger.addHandler(sh)
    return logger

def main():
    if args.wandb_project is None: args.wandb_project = args.model
    if args.exp_name is None: args.exp_name = args.model
    wandb.init(project=args.wandb_project, name=args.exp_name, config=vars(args))
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    exp_dir = os.path.join(args.ckpt_save_dir, args.exp_name, timestamp)
    os.makedirs(exp_dir, exist_ok=True)
    logger = setup_logger(os.path.join(exp_dir, 'log.txt'))
    
    # 初始化 Tokenizer 
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    # 初始化模型
    model, transform = get_gazelle_model(args.model)
    
    model.cuda()
    if torch.cuda.device_count() > 1: model = DataParallelWrapper(model)

    # 数据加载
    gazefollow_train = GazeDataset('gazefollow', args.data_path, 'train', transform, is_mix_gaze_estimation=args.is_mix_gaze_estimation, is_partial_input=args.is_partial_input)
    train_dl_gf = torch.utils.data.DataLoader(gazefollow_train, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn, num_workers=args.n_workers)
    
    if args.is_mix_gaze_estimation:
        est_bs_per_set = args.estimation_batch_size // 2
        eth_train = GazeEstimationDataset(args.eth_label, args.eth_img, transform, is_train=True)
        gaze360_train = GazeEstimationDataset(args.gaze360_label, args.gaze360_img, transform, is_train=True)
        iter_eth = get_infinite_iter(torch.utils.data.DataLoader(eth_train, batch_size=est_bs_per_set, shuffle=True, collate_fn=collate_fn, num_workers=4))
        iter_g360 = get_infinite_iter(torch.utils.data.DataLoader(gaze360_train, batch_size=est_bs_per_set, shuffle=True, collate_fn=collate_fn, num_workers=4))

    eval_dataset = GazeDataset('gazefollow', args.data_path, 'test', transform, is_partial_input=args.is_partial_input)
    eval_dl = torch.utils.data.DataLoader(eval_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn, num_workers=args.n_workers)

    criterion_logits = nn.BCEWithLogitsLoss(reduction='none') 
    criterion_seg = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([10.0]).cuda(), reduction='none')
    criterion_ce = nn.CrossEntropyLoss(ignore_index=-100) 
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.max_epochs, eta_min=1e-7)

    best_min_l2 = 1.0

    for epoch in range(args.max_epochs):
        model.train()
        pbar = tqdm(enumerate(train_dl_gf), total=len(train_dl_gf), desc=f"Epoch {epoch}", unit="batch", dynamic_ncols=True)
        
        for cur_iter, batch_gf in pbar:
            batches_to_merge = [batch_gf]
            if args.is_mix_gaze_estimation:
                if iter_eth is not None: batches_to_merge.append(next(iter_eth))
                if iter_g360 is not None: batches_to_merge.append(next(iter_g360))
            
            final_batch = merge_batches(batches_to_merge)
            imgs, bboxes, eyes, gazex, gazey, inout, heights, widths, heatmaps, observer_expressions, gaze_directions, gaze_point_expressions, seg_mask, is_face_crop_mode, gaze3d, has_3d = final_batch

            # =========================================================
            # [Indices Separation]
            # =========================================================
            is_est = has_3d.bool().cuda()
            is_gf = ~is_est
            
            num_gf = is_gf.sum()
            num_est = is_est.sum()

            optimizer.zero_grad()
            preds = model({
                "images": imgs.cuda(), "bboxes": [[bbox] for bbox in bboxes], "eyes": eyes, 
                "observer_expression_ids": observer_expressions.cuda(), "gaze_point_expression_ids": gaze_point_expressions.cuda()
            })
            loss = 0.0
            
            # 用于记录日志的 Loss (只包含 GF)
            log_hm_loss = 0.0
            log_text_loss = 0.0
            log_seg_loss = 0.0
            log_dir_loss = 0.0
            log_g3d_loss = 0.0

            # --- A. Heatmap Loss (Separate Calculation) ---
            if isinstance(preds['heatmap'], list): heatmap_preds = torch.stack(preds['heatmap']).squeeze(dim=1)
            else: heatmap_preds = preds['heatmap'].squeeze(dim=1)
            
            # 计算 Per-Sample Loss
            pixel_loss = criterion_logits(heatmap_preds, heatmaps.cuda())
            loss_per_sample_hm = pixel_loss.mean(dim=(1, 2)) # [B]
            
            # 1. GF Heatmap (主任务)
            if num_gf > 0:
                loss_hm_gf = loss_per_sample_hm[is_gf].mean()
                loss += loss_hm_gf # 权重 1.0
                log_hm_loss = loss_hm_gf.item()
            else:
                loss_hm_gf = torch.tensor(0.0)

            # 2. Est Heatmap (辅助任务: 学习全0)
            if num_est > 0:
                loss_hm_est = loss_per_sample_hm[is_est].mean()
                loss += loss_hm_est * 0.1 # 权重 0.01
            
            # --- B. Seg Loss (Separate) ---
            if preds['seg'] is not None:
                if isinstance(preds['seg'], list): seg_preds = torch.stack(preds['seg']).squeeze(dim=1)
                else: seg_preds = preds['seg'].squeeze(dim=1)
                
                seg_loss_per_sample = criterion_seg(seg_preds, seg_mask.cuda()).mean(dim=(1,2))
                
                if num_gf > 0:
                    loss_seg_gf = seg_loss_per_sample[is_gf].mean()
                    loss += loss_seg_gf * 0.5 # 原有权重
                    log_seg_loss = loss_seg_gf.item()
                
                if num_est > 0:
                    loss_seg_est = seg_loss_per_sample[is_est].mean()
                    loss += loss_seg_est * 0.5 * 0.1 # 更小的辅助权重

            # --- C. Text Loss ---
            if preds['text_loss'] is not None:
                # preds['text_loss'] 现在是 [B] 维度的 Tensor
                
                # 1. GazeFollow (主任务，记录日志)
                if num_gf > 0:
                    loss_text_gf = preds['text_loss'][is_gf].mean()
                    loss += loss_text_gf * 0.01 # 原始权重
                    log_text_loss = loss_text_gf.item()
                
                # 2. Estimation (辅助任务，outside of image，给小权重)
                if num_est > 0:
                    loss_text_est = preds['text_loss'][is_est].mean()
                    loss += loss_text_est * 0.01 * 0.1 # 额外乘 0.1 的缩小系数

            # --- D. Direction Loss ---
            # Est 数据标签是 -100，Loss 自动为 0。所以计算出的 Loss 本身就是纯 GF 的。
            if preds['direction'] is not None and num_gf > 0:
                if isinstance(preds['direction'], list): dir_preds = torch.stack(preds['direction']).squeeze(dim=1)
                else: dir_preds = preds['direction'].squeeze(dim=1)
                
                clean_target = gaze_directions.clone()
                clean_target[(clean_target < 0) | (clean_target >= 8)] = -100
                
                # 这里 Est 的样本 Loss 为 0，均值会被 Est 样本数量稀释
                # 严格来说应该只对 GF 求 mean
                dir_loss_raw = criterion_ce(dir_preds, clean_target.cuda()) # Scalar Mean
                
                # 如果 criterion_ce 是 mean reduction，我们无法剔除 Est 的分母影响。
                # 但这通常影响不大。如果非常严格，需要把 criterion 改为 reduction='none'。
                loss += dir_loss_raw * 0.02
                log_dir_loss = dir_loss_raw.item()

            # --- E. Gaze 3D Loss (Est Only) ---
            if preds['gaze3d'] is not None and args.is_mix_gaze_estimation:
                if isinstance(preds['gaze3d'], list): gaze3d_preds = torch.stack(preds['gaze3d']).squeeze(dim=1)
                else: gaze3d_preds = preds['gaze3d'].squeeze(dim=1)
                
                loss_gaze3d_raw = F.l1_loss(gaze3d_preds, gaze3d.cuda(), reduction='none').mean(dim=1)
                
                # 只有 Est 有效
                if num_est > 0:
                    loss_g3d_est = loss_gaze3d_raw[is_est].mean()
                    loss += loss_g3d_est * 0.1
                    log_g3d_loss = loss_g3d_est.item()
            
            loss.backward()
            optimizer.step()
            
            # --- Logging (Only GF metrics where possible) ---
            print_dict = {}
            print_dict['heatmap_loss'] = log_hm_loss # Pure GF
            if preds['text_loss'] is not None: print_dict['text_loss'] = log_text_loss
            if preds['direction'] is not None: print_dict['direction_loss'] = log_dir_loss # Mostly GF
            if preds['seg'] is not None: print_dict['seg_loss'] = log_seg_loss # Pure GF
            if preds['gaze3d'] is not None: print_dict['gaze3d_loss'] = log_g3d_loss # Pure Est
            print_dict['total_loss'] = loss.item()
            
            pbar.set_postfix(print_dict)

            if cur_iter % args.log_iter == 0: 
                wandb.log(print_dict)
                logger.info(f"Iter {cur_iter}/{len(train_dl_gf)} "+", ".join([f"{k}: {v:.4f}" for k, v in print_dict.items()]))

        scheduler.step()
        ckpt_path_last = os.path.join(exp_dir, 'last.pt')
        model_to_save = model.module if hasattr(model, 'module') else model
        torch.save(model_to_save.get_gazelle_state_dict(), ckpt_path_last)

        # ==========================================
        # 验证与可视化
        # ==========================================
        logger.info(f"[Epoch {epoch} Evaluation]")
        model.eval()
        avg_l2s, min_l2s, aucs = [], [], []
        
        for cur_iter, batch in tqdm(enumerate(eval_dl), total=len(eval_dl)):
            imgs, bboxes, eyes, gazex, gazey, inout, heights, widths, observer_expressions, gaze_directions, gaze_point_expressions, seg_mask, is_face_crop_mode, gaze3d, has_3d = batch
            
            input_dict = {
                "images": imgs.cuda(), 
                "bboxes": [[bbox] for bbox in bboxes], 
                "eyes": eyes, 
                "observer_expression_ids": observer_expressions.cuda(),
                # 传入 tokenizer 以便 decoder 获取特殊 token id
                "tokenizer": tokenizer 
            }

            # [关键] 只有在需要可视化时 (cur_iter == 0)，才开启文本生成
            # 这样可以节省其他 iter 的推理时间
            if cur_iter == 0:
                input_dict["generate_text"] = True
                
                # 运行模型 (生成模式)
                torch.no_grad()
                with torch.no_grad():
                    preds = model(input_dict)
                
                # 此时 preds['text_generated'] 含有 token IDs
                # preds['text_loss'] 为 None
            else:
                # 运行模型 (Loss 模式，用于计算指标)
                with torch.no_grad():
                    preds = model(input_dict)
             
            if isinstance(preds['heatmap'], list): heatmap_preds = torch.stack(preds['heatmap']).squeeze(dim=1)
            else: heatmap_preds = preds['heatmap'].squeeze(dim=1)
            heatmap_preds = heatmap_preds.cpu()
            
            # --- 每一批次的第一个样本进行可视化保存 ---
            # ==========================================
            # 可视化部分 (仅第一批次)
            # ==========================================
            if cur_iter == 0:
                epoch_vis_dir = os.path.join(exp_dir, f"vis_epoch_{epoch}")
                os.makedirs(epoch_vis_dir, exist_ok=True)
                
                num_to_vis = min(10, len(imgs))
                
                for idx_in_batch in range(num_to_vis):
                    global_idx = cur_iter * args.batch_size + idx_in_batch
                    img_idx, head_idx = eval_dataset.data_idxs[global_idx]
                    rel_path = eval_dataset.data[img_idx]['path']
                    img_full_path = os.path.join(eval_dataset.path, rel_path)
                    
                    # --- 1. Prepare GT Data ---
                    cur_bbox = bboxes[idx_in_batch] # Normalized [x1, y1, x2, y2]
                    
                    # GT Point (Normalized [x, y])
                    cur_gt_point = [gazex[idx_in_batch][0], gazey[idx_in_batch][0]]
                    
                    # GT Mask (keep as Tensor or None, visualizer handles it)
                    cur_gt_mask = None
                    if seg_mask is not None:
                        cur_gt_mask = seg_mask[idx_in_batch]
                    
                    # GT Text
                    gt_ids = gaze_point_expressions[idx_in_batch]
                    gt_text = tokenizer.decode(gt_ids, skip_special_tokens=True)
                    
                    # GT Direction
                    gt_dir_idx = -100
                    if gaze_directions is not None:
                        gt_dir_idx = gaze_directions[idx_in_batch] # Tensor or int

                    # --- 2. Prepare Pred Data ---
                    cur_hm = heatmap_preds[idx_in_batch].numpy()
                    if cur_hm.ndim > 2: cur_hm = cur_hm[0]
                    hm_idx = np.unravel_index(np.argmax(cur_hm), cur_hm.shape)
                    pred_y_norm = (hm_idx[0] + 0.5) / cur_hm.shape[0]
                    pred_x_norm = (hm_idx[1] + 0.5) / cur_hm.shape[1]
                    cur_pred_point = [pred_x_norm, pred_y_norm]

                    # [B] Pred Mask (修正 List 索引逻辑)
                    cur_pred_mask = None
                    if preds.get('seg') is not None:
                        seg_out = preds['seg'] # 这是一个 List: [Tensor(1, H, W), Tensor(1, H, W), ...]
                        
                        # 1. 提取当前图片的 Tensor
                        if isinstance(seg_out, list):
                            # 直接取对应索引，而不是取最后一个 [-1]
                            curr_img_seg = seg_out[idx_in_batch] 
                        else:
                            curr_img_seg = seg_out[idx_in_batch]
                            
                        # 2. 处理 Tensor 内部 (通常 GazeFollow 每张图只有 1 个目标)
                        # curr_img_seg shape: [Num_People, H, W] -> 通常是 [1, 64, 64]
                        if curr_img_seg.ndim >= 3:
                            cur_pred_mask = torch.sigmoid(curr_img_seg[0]) # 取第一个人
                        else:
                            cur_pred_mask = torch.sigmoid(curr_img_seg)

                    # [C] Pred Text (修正 List 索引逻辑)
                    pred_text = "N/A"
                    if 'text_generated' in preds and preds['text_generated'] is not None:
                        gen_out = preds['text_generated'] # List: [Tensor(1, Seq), ...]
                        
                        if isinstance(gen_out, list):
                            curr_token_ids = gen_out[idx_in_batch]
                        else:
                            curr_token_ids = gen_out[idx_in_batch]
                        
                        # 如果 tensor 维度是 [Num_People, Seq]，取第一个人
                        if curr_token_ids.ndim > 1: curr_token_ids = curr_token_ids[0]
                        
                        pred_text = tokenizer.decode(curr_token_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True).strip()
                        logger.info(f"Sample {idx_in_batch} Generated: {pred_text}")
                        
                    elif 'text_loss' in preds:
                         res_text = preds['text_loss'] # List: [Tensor(1, Vocab), ...]
                         if isinstance(res_text, list): 
                             raw = res_text[idx_in_batch]
                         else: 
                             raw = res_text[idx_in_batch]
                             
                         if torch.is_tensor(raw) and raw.ndim > 0:
                              # 取第一个人的 logits
                              if raw.ndim > 1: raw = raw[0]
                              pred_text = f"[Argmax] {tokenizer.decode(torch.argmax(raw, -1), skip_special_tokens=True)}"

                    # [D] Pred Direction (修正 List 索引逻辑 - 解决 IndexError)
                    cur_pred_dir_idx = -100
                    if preds.get('direction') is not None:
                        d_out = preds['direction'] # List: [Tensor(1, 8), Tensor(1, 8)...]
                        
                        # 1. 提取当前图片的 Tensor
                        if isinstance(d_out, list):
                            d_val = d_out[idx_in_batch] # Tensor shape [Num_People, 8]
                        else:
                            d_val = d_out[idx_in_batch]
                        
                        # 2. 取 argmax
                        if d_val.ndim > 1: 
                            # d_val[0] 取第一个人
                            cur_pred_dir_idx = torch.argmax(d_val[0]).item()
                        else:
                            cur_pred_dir_idx = torch.argmax(d_val).item()

                    # --- 3. Call New Visualizer ---
                    vis_save_path = os.path.join(epoch_vis_dir, f"sample_{idx_in_batch}.png")
                    
                    visualize_comparison(
                        image_path=img_full_path,
                        bbox=cur_bbox,
                        gt_point=cur_gt_point,
                        gt_mask=cur_gt_mask,
                        gt_text=gt_text,
                        gt_dir_idx=gt_dir_idx,
                        pred_point=cur_pred_point,
                        pred_mask=cur_pred_mask,
                        pred_text=pred_text,
                        pred_dir_idx=cur_pred_dir_idx,
                        save_path=vis_save_path
                    )
                
                logger.info(f"Saved {num_to_vis} comparison visualizations to: {epoch_vis_dir}")

            # 指标计算
            for i in range(heatmap_preds.shape[0]):
                auc = gazefollow_auc(heatmap_preds[i], gazex[i], gazey[i], heights[i], widths[i])
                avg_l2, min_l2 = gazefollow_l2(heatmap_preds[i], gazex[i], gazey[i])
                aucs.append(auc); avg_l2s.append(avg_l2); min_l2s.append(min_l2)
                
        epoch_min_l2 = np.mean(min_l2s)
        epoch_avg_l2 = np.mean(avg_l2s)
        epoch_auc = np.mean(aucs)
        logger.info(f"Eval Epoch {epoch}: Min L2={epoch_min_l2:.4f}, Avg L2={epoch_avg_l2:.4f}, AUC={epoch_auc:.4f}")
        if epoch_min_l2 < best_min_l2:
            best_min_l2 = epoch_min_l2
            torch.save(model_to_save.get_gazelle_state_dict(), os.path.join(exp_dir, 'best.pt'))

if __name__ == '__main__':
    main()