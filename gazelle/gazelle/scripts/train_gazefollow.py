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

    # 1. Observer BBox (蓝色) - bbox 传入应为 [x1, y1, x2, y2]
    bx1, by1, bx2, by2 = bbox
    rect = patches.Rectangle((bx1*w, by1*h), (bx2-bx1)*w, (by2-by1)*h, 
                             linewidth=3, edgecolor='blue', facecolor='none', label='Observer')
    ax.add_patch(rect)

    # 2. GT Point (蓝色)
    ax.scatter(gt_point[0]*w, gt_point[1]*h, c='blue', s=120, marker='o', edgecolors='white', label='GT')

    # 3. Pred Point (绿色)
    if 'heatmap' in preds and preds['heatmap'] is not None:
        hm = preds['heatmap'].detach().cpu().squeeze().numpy() # 确保是 [64, 64]
        if hm.ndim > 2: hm = hm[0] # 防止有残留的 batch/channel 维度
        idx = np.unravel_index(np.argmax(hm), hm.shape)
        pred_y, pred_x = idx[0] * (h / hm.shape[0]), idx[1] * (w / hm.shape[1])
        ax.scatter(pred_x, pred_y, c='lime', s=150, marker='x', linewidths=3, label='Pred')

    # 4. Seg Mask (绿色透明层) - 修正 ValueError
    if 'seg' in preds and preds['seg'] is not None:
        seg = torch.sigmoid(preds['seg']).detach().cpu().squeeze().numpy()
        # 核心修正：如果 seg 是 [H, W, C] 或 [C, H, W]，只取第一个通道
        if seg.ndim == 3:
            seg = seg[0] # 取第一层 Mask
            
        seg_resized = cv2.resize(seg, (w, h)) # 此时 seg_resized 形状为 (h, w)
        
        mask_overlay = np.zeros((h, w, 4))
        mask_overlay[..., 1] = 1.0  # Green
        mask_overlay[..., 3] = seg_resized * 0.6  # Alpha 赋值
        ax.imshow(mask_overlay)

    title_str = preds.get('text', "Gaze Detection Output")
    plt.title(f"{title_str}", fontsize=14, color='darkgreen', bbox=dict(facecolor='white', alpha=0.8))
    plt.legend(loc='upper right')
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
    if args.is_mix_gaze_estimation:
        backbone = SAMBackboneWrapper(model_type="vit_b", in_size=(448, 448), backbone_type="dinov2", is_lora=False, is_multi_input=True)
        transform = backbone.get_transform((448, 448))
        model = GazeLLE(backbone, inout=False, is_sam_prompt=True, is_mix_gaze_estimation=True, is_multi_output=False)
    else:
        model, transform = get_gazelle_model(args.model)
    
    model.cuda()
    if torch.cuda.device_count() > 1: model = DataParallelWrapper(model)

    # 数据加载
    gazefollow_train = GazeDataset('gazefollow', args.data_path, 'train', transform, is_partial_input=args.is_partial_input)
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
        pbar = tqdm(enumerate(train_dl_gf), total=len(train_dl_gf), desc=f"Epoch {epoch}")
        
        for cur_iter, batch_gf in pbar:
            batches_to_merge = [batch_gf]
            if args.is_mix_gaze_estimation:
                batches_to_merge.append(next(iter_eth))
                batches_to_merge.append(next(iter_g360))
            
            final_batch = merge_batches(batches_to_merge)
            imgs, bboxes, eyes, gazex, gazey, inout, heights, widths, heatmaps, observer_expressions, gaze_directions, gaze_point_expressions, seg_mask, is_face_crop_mode, gaze3d, has_3d = final_batch

            is_est = has_3d.bool().cuda()
            is_gf = ~is_est
            num_gf = is_gf.sum()

            optimizer.zero_grad()
            preds = model({
                "images": imgs.cuda(), "bboxes": [[bbox] for bbox in bboxes], "eyes": eyes, 
                "observer_expression_ids": observer_expressions.cuda(), "gaze_point_expression_ids": gaze_point_expressions.cuda()
            })
            
            loss = 0.0
            # --- Heatmap Loss ---
            if isinstance(preds['heatmap'], list): heatmap_preds = torch.stack(preds['heatmap']).squeeze(dim=1)
            else: heatmap_preds = preds['heatmap'].squeeze(dim=1)
            pixel_loss = criterion_logits(heatmap_preds, heatmaps.cuda())
            loss_per_sample_hm = pixel_loss.mean(dim=(1, 2))
            
            if num_gf > 0:
                loss_hm_gf = loss_per_sample_hm[is_gf].mean()
                loss += loss_hm_gf
            
            # 文本损失计算（根据 model.py 的 loss 返回值）
            if 'text_loss' in preds and preds['text_loss'] is not None:
                loss += preds['text_loss'].mean()

            if is_est.any():
                loss += loss_per_sample_hm[is_est].mean() * 0.1

            loss.backward()
            optimizer.step()
            pbar.set_postfix({'loss': loss.item()})

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
            if cur_iter == 0:
                idx_in_batch = 0
                global_idx = cur_iter * args.batch_size + idx_in_batch
                img_idx, head_idx = eval_dataset.data_idxs[global_idx]
                rel_path = eval_dataset.data[img_idx]['path']
                img_full_path = os.path.join(eval_dataset.path, rel_path)
                
                cur_hm = heatmap_preds[idx_in_batch]
                
                cur_seg = None
                if preds.get('seg') is not None:
                    if isinstance(preds['seg'], list):
                        cur_seg = preds['seg'][-1][idx_in_batch]
                    else:
                        cur_seg = preds['seg'][idx_in_batch]

                # --- 文字映射逻辑 ---
                # 原始 GT 文本
                gt_ids = gaze_point_expressions[idx_in_batch]
                gt_text = tokenizer.decode(gt_ids, skip_special_tokens=True)
                
                # 预测文本（从推理模式的 logits 反解）
                pred_text = "N/A"
                if 'text_logits' in preds and preds['text_logits'] is not None:
                    # 简单贪婪解码：取最后一个时间步或序列最大概率
                    logits = preds['text_logits'][idx_in_batch] # [Seq, Vocab]
                    token_ids = torch.argmax(logits, dim=-1)
                    pred_text = tokenizer.decode(token_ids, skip_special_tokens=True)

                vis_preds = {
                    'heatmap': cur_hm,
                    'seg': cur_seg,
                    'text': f"GT: {gt_text}\nPred: {pred_text}"
                }
                
                vis_save_path = os.path.join(exp_dir, f"vis_epoch_{epoch}.png")
                
                visualize_step(
                    img_full_path, 
                    vis_preds, 
                    bboxes[idx_in_batch],  # 确保这里是 [x1, y1, x2, y2]
                    [gazex[idx_in_batch][0], gazey[idx_in_batch][0]], 
                    vis_save_path
                )
                logger.info(f"Saved visualization to: {vis_save_path}")

            # 指标计算
            for i in range(heatmap_preds.shape[0]):
                auc = gazefollow_auc(heatmap_preds[i], gazex[i], gazey[i], heights[i], widths[i])
                avg_l2, min_l2 = gazefollow_l2(heatmap_preds[i], gazex[i], gazey[i])
                aucs.append(auc); avg_l2s.append(avg_l2); min_l2s.append(min_l2)
                
        logger.info(f"Eval Epoch {epoch}: Min L2={np.mean(min_l2s):.4f}, AUC={np.mean(aucs):.4f}")
        scheduler.step()

if __name__ == '__main__':
    main()