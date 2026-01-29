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
parser.add_argument('--model', type=str, default="dinov2_vitb_lora_multi_output")
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
def visualize_step(image_path, preds, bbox, gt_point, save_path):
    img = Image.open(image_path).convert('RGB')
    w, h = img.size
    fig, ax = plt.subplots(1, figsize=(10, 10))
    ax.imshow(img)

    # 1. Observer BBox (蓝色) - bbox: [x1, y1, x2, y2] (Normalized)
    bx1, by1, bx2, by2 = bbox
    rect = patches.Rectangle((bx1*w, by1*h), (bx2-bx1)*w, (by2-by1)*h, 
                             linewidth=3, edgecolor='blue', facecolor='none', label='Observer')
    ax.add_patch(rect)

    # 2. GT Point (蓝色圆点)
    ax.scatter(gt_point[0]*w, gt_point[1]*h, c='blue', s=120, marker='o', edgecolors='white', label='GT')

    # 3. Pred Point (绿色叉号)
    if 'heatmap' in preds and preds['heatmap'] is not None:
        hm = preds['heatmap'].detach().cpu().squeeze().numpy()
        if hm.ndim > 2: hm = hm[0]
        idx = np.unravel_index(np.argmax(hm), hm.shape)
        # heatmap coordinate -> image coordinate
        pred_y, pred_x = idx[0] * (h / hm.shape[0]), idx[1] * (w / hm.shape[1])
        ax.scatter(pred_x, pred_y, c='lime', s=150, marker='x', linewidths=3, label='Pred Point')

    # 4. Seg Mask (绿色透明层)
    if 'seg' in preds and preds['seg'] is not None:
        seg = torch.sigmoid(preds['seg']).detach().cpu().squeeze().numpy()
        if seg.ndim == 3: seg = seg[0]
        seg_resized = cv2.resize(seg, (w, h))
        mask_overlay = np.zeros((h, w, 4))
        mask_overlay[..., 1] = 1.0  # Green
        mask_overlay[..., 3] = seg_resized * 0.6
        ax.imshow(mask_overlay)

    # 5. [修改] Direction Arrow (黄色箭头)
    if 'direction' in preds and preds['direction'] is not None:
        dir_data = preds['direction'].detach().cpu()
        # 获取分类索引 (0-7)
        if dir_data.ndim > 0 and dir_data.shape[-1] == 8:
            dir_idx = torch.argmax(dir_data).item()
        else:
            dir_idx = dir_data.item()

        # 定义每个 Label 对应的数学角度 (标准逆时针: 0度为右, 90度为上)
        # Label定义: 0:R, 1:TR, 2:Top, 3:TL, 4:L, 5:BL, 6:Bottom, 7:BR
        idx_to_angle = {
            0: 0,    # Right
            1: 45,   # Top-Right
            2: 90,   # Above (Top)
            3: 135,  # Top-Left
            4: 180,  # Left
            5: 225,  # Bottom-Left
            6: 270,  # Below (Bottom)
            7: 315   # Bottom-Right
        }
        
        if dir_idx in idx_to_angle:
            angle_deg = idx_to_angle[dir_idx]
            angle_rad = math.radians(angle_deg)

            # 计算箭头长度 (图像宽度的 15%)
            arrow_len = w * 0.15
            
            # 计算偏移量 (dx, dy)
            # 注意: 图像坐标系中 y 是向下的，而 math.sin 假设 y 向上
            # 所以 "Top/Above" (90度) 时 sin(90)=1, 我们需要 dy 为负值才能往上画
            dx = math.cos(angle_rad) * arrow_len
            dy = -math.sin(angle_rad) * arrow_len 

            # 起点：BBox 中心
            cx = (bx1 + bx2) / 2 * w
            cy = (by1 + by2) / 2 * h
            
            # 绘制箭头
            ax.arrow(cx, cy, dx, dy, color='lime', width=w*0.005, head_width=w*0.02, 
                    length_includes_head=True, label='Pred Dir')

    title_str = preds.get('text', "Gaze Detection Output")
    plt.title(f"{title_str}", fontsize=14, color='darkgreen', bbox=dict(facecolor='white', alpha=0.8))
    
    # 整理图例
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(), loc='upper right')
    
    plt.axis('off')
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0.1)
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
                
                seg_loss_per_sample = criterion_logits(seg_preds, seg_mask.cuda()).mean(dim=(1,2))
                
                if num_gf > 0:
                    loss_seg_gf = seg_loss_per_sample[is_gf].mean()
                    loss += loss_seg_gf * 0.1 # 原有权重
                    log_seg_loss = loss_seg_gf.item()
                
                if num_est > 0:
                    loss_seg_est = seg_loss_per_sample[is_est].mean()
                    loss += loss_seg_est * 0.1 * 0.1 # 更小的辅助权重

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
            
            with torch.no_grad():
                preds = model({
                    "images": imgs.cuda(), "bboxes": [[bbox] for bbox in bboxes], "eyes": eyes, 
                    "observer_expression_ids": observer_expressions.cuda()
                })
             
            if isinstance(preds['heatmap'], list): heatmap_preds = torch.stack(preds['heatmap']).squeeze(dim=1)
            else: heatmap_preds = preds['heatmap'].squeeze(dim=1)
            heatmap_preds = heatmap_preds.cpu()
            
            # --- 每一批次的第一个样本进行可视化保存 ---
            # --- 验证集可视化保存逻辑 ---
            if cur_iter == 0:
                epoch_vis_dir = os.path.join(exp_dir, f"vis_epoch_{epoch}")
                os.makedirs(epoch_vis_dir, exist_ok=True)
                
                num_to_vis = min(10, len(imgs))
                
                for idx_in_batch in range(num_to_vis):
                    global_idx = cur_iter * args.batch_size + idx_in_batch
                    img_idx, head_idx = eval_dataset.data_idxs[global_idx]
                    rel_path = eval_dataset.data[img_idx]['path']
                    img_full_path = os.path.join(eval_dataset.path, rel_path)
                    
                    cur_hm = heatmap_preds[idx_in_batch]
                    
                    # --- 修正 Seg 获取逻辑 ---
                    cur_seg = None
                    if preds.get('seg') is not None:
                        seg_output = preds['seg']
                        # 如果是 list (通常包含多个尺度)，取最后一个尺度
                        if isinstance(seg_output, list):
                            seg_output = seg_output[-1]
                        
                        # 核心修正：检查 batch 维度
                        if seg_output.ndim >= 3: # [B, H, W] 或 [B, C, H, W]
                            if seg_output.shape[0] > idx_in_batch:
                                cur_seg = seg_output[idx_in_batch]
                            else:
                                # 如果 B 维度不够，说明 DataParallel 还没拼接或者被 split 了
                                # 取当前可用的第一个
                                cur_seg = seg_output[0] 
                        else:
                            cur_seg = seg_output

                    # --- 文本解码逻辑 ---
                    gt_ids = gaze_point_expressions[idx_in_batch]
                    gt_text = tokenizer.decode(gt_ids, skip_special_tokens=True)
                    
                    pred_text = "N/A"
                    if 'text_loss' in preds and preds['text_loss'] is not None:
                        res_text = preds['text_loss']
                        
                        # 同理处理文本的 batch 索引
                        if isinstance(res_text, list):
                            raw_logits = res_text[idx_in_batch] if len(res_text) > idx_in_batch else res_text[0]
                        else:
                            raw_logits = res_text[idx_in_batch] if res_text.shape[0] > idx_in_batch else res_text[0]
                        
                        if torch.is_tensor(raw_logits) and raw_logits.ndim >= 2:
                            token_ids = torch.argmax(raw_logits, dim=-1)
                            if idx_in_batch == 0:
                                logger.info(f"Epoch {epoch} Sample 0 TokenIDs: {token_ids[:10].tolist()}")
                            pred_text = tokenizer.decode(token_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True).strip()

                    cur_dir = None
                    if preds.get('direction') is not None:
                        dir_output = preds['direction']
                        # 处理 batch 和 list 情况
                        if isinstance(dir_output, list):
                            # 如果是多尺度列表，取最后一层，然后取 batch index
                            last_scale = dir_output[-1] if isinstance(dir_output[-1], torch.Tensor) else dir_output[0]
                            cur_dir = last_scale[idx_in_batch] if last_scale.shape[0] > idx_in_batch else last_scale[0]
                        else:
                            cur_dir = dir_output[idx_in_batch] if dir_output.shape[0] > idx_in_batch else dir_output[0]

                    vis_preds = {
                        'heatmap': cur_hm,
                        'seg': cur_seg,
                        'text': f"GT: {gt_text}\nPred: {pred_text}",
                        'direction': cur_dir
                    }
                    
                    vis_save_path = os.path.join(epoch_vis_dir, f"sample_{idx_in_batch}.png")
                    visualize_step(
                        img_full_path, 
                        vis_preds, 
                        bboxes[idx_in_batch],  
                        [gazex[idx_in_batch][0], gazey[idx_in_batch][0]], 
                        vis_save_path
                    )
                
                logger.info(f"Saved {num_to_vis} visualizations to: {epoch_vis_dir}")
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