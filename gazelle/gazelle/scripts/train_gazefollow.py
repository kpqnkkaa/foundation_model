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

from gazelle.backbone import SAMBackboneWrapper 
from gazelle.model import GazeLLE 
import torchvision.transforms.functional as F_vis
import torch.nn.functional as F

cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)

os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"

from gazelle.dataloader import GazeDataset, GazeEstimationDataset, collate_fn
from gazelle.model import get_gazelle_model
from gazelle.utils import gazefollow_auc, gazefollow_l2

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default="dinov2_vitb_lora_multi_output")
parser.add_argument('--data_path', type=str, default='/mnt/nvme1n1/lululemon/xjj/datasets/resized/gazefollow_extended')
parser.add_argument('--is_mix_gaze_estimation', default=False, action='store_true')
parser.add_argument('--estimation_batch_size', type=int, default=64, help="Total batch size for estimation data")
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
parser.add_argument('--batch_size', type=int, default=60) # GF Batch Size
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--n_workers', type=int, default=8)
args = parser.parse_args()

# --- 新增：可视化辅助函数 ---
def visualize_step(image_path, preds, bboxes, gt_point, save_path):
    try:
        img = Image.open(image_path).convert('RGB')
        w, h = img.size
        fig, ax = plt.subplots(1, figsize=(10, 10))
        ax.imshow(img)

        # 1. Observer BBox (蓝色)
        bx1, by1, bx2, by2 = bboxes[0] # 取第一个 bbox
        rect = patches.Rectangle((bx1*w, by1*h), (bx2-bx1)*w, (by2-by1)*h, 
                                 linewidth=3, edgecolor='blue', facecolor='none', label='Observer BBox')
        ax.add_patch(rect)

        # 2. GT Point (蓝色圆点)
        ax.scatter(gt_point[0]*w, gt_point[1]*h, c='blue', s=120, marker='o', edgecolors='white', label='GT Point')

        # 3. Pred Point from Heatmap (绿色叉号)
        if 'heatmap' in preds and preds['heatmap'] is not None:
            hm = preds['heatmap'].detach().cpu().numpy()
            idx = np.unravel_index(np.argmax(hm), hm.shape)
            pred_y, pred_x = idx[0] * (h / hm.shape[0]), idx[1] * (w / hm.shape[1])
            ax.scatter(pred_x, pred_y, c='lime', s=150, marker='x', linewidths=3, label='Pred Point')

        # 4. Seg Mask (绿色透明层)
        if 'seg' in preds and preds['seg'] is not None:
            seg = torch.sigmoid(preds['seg']).detach().cpu().numpy()
            seg_resized = cv2.resize(seg, (w, h))
            mask_overlay = np.zeros((h, w, 4))
            mask_overlay[..., 1] = 1.0 # Green
            mask_overlay[..., 3] = seg_resized * 0.6 # Alpha
            ax.imshow(mask_overlay)

        # 5. Text (标题)
        title_str = preds.get('text', "Gaze Estimation")
        plt.title(f"Visual Result: {title_str}", fontsize=14, color='darkgreen', bbox=dict(facecolor='white', alpha=0.8))
        
        plt.legend(loc='upper right')
        plt.axis('off')
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0.1)
        plt.close()
    except Exception as e:
        print(f"Visualization failed: {e}")

# --- 原有辅助函数 ---
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
    logger.info(f"Experiment Config: {vars(args)}")

    if args.is_mix_gaze_estimation:
        logger.info("Initializing Model with Mix Gaze Estimation Mode")
        backbone = SAMBackboneWrapper(model_type="vit_b", in_size=(448, 448), backbone_type="dinov2", is_lora=False, is_multi_input=True)
        transform = backbone.get_transform((448, 448))
        model = GazeLLE(backbone, inout=False, is_sam_prompt=True, is_mix_gaze_estimation=True, is_multi_output=False)
    else:
        model, transform = get_gazelle_model(args.model)
    
    model.cuda()
    if torch.cuda.device_count() > 1: model = DataParallelWrapper(model)

    gazefollow_train = GazeDataset('gazefollow', args.data_path, 'train', transform, is_partial_input=args.is_partial_input)
    train_dl_gf = torch.utils.data.DataLoader(gazefollow_train, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn, num_workers=args.n_workers)
    
    iter_eth = None
    iter_g360 = None
    if args.is_mix_gaze_estimation:
        logger.info("Loading ETH and Gaze360 Datasets as separate streams...")
        est_bs_per_set = args.estimation_batch_size // 2
        if est_bs_per_set > 0:
            eth_train = GazeEstimationDataset(args.eth_label, args.eth_img, transform, is_train=True)
            gaze360_train = GazeEstimationDataset(args.gaze360_label, args.gaze360_img, transform, is_train=True)
            eth_dl = torch.utils.data.DataLoader(eth_train, batch_size=est_bs_per_set, shuffle=True, collate_fn=collate_fn, num_workers=args.n_workers//2)
            gaze360_dl = torch.utils.data.DataLoader(gaze360_train, batch_size=est_bs_per_set, shuffle=True, collate_fn=collate_fn, num_workers=args.n_workers//2)
            iter_eth = get_infinite_iter(eth_dl)
            iter_g360 = get_infinite_iter(gaze360_dl)

    eval_dataset = GazeDataset('gazefollow', args.data_path, 'test', transform, is_partial_input=args.is_partial_input)
    eval_dl = torch.utils.data.DataLoader(eval_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn, num_workers=args.n_workers)

    criterion_logits = nn.BCEWithLogitsLoss(reduction='none') 
    criterion_ce = nn.CrossEntropyLoss(ignore_index=-100) 
    
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.max_epochs, eta_min=1e-7)

    best_min_l2 = 1.0; best_epoch = None

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
            log_hm_loss = 0.0; log_text_loss = 0.0; log_seg_loss = 0.0; log_dir_loss = 0.0; log_g3d_loss = 0.0

            if isinstance(preds['heatmap'], list): heatmap_preds = torch.stack(preds['heatmap']).squeeze(dim=1)
            else: heatmap_preds = preds['heatmap'].squeeze(dim=1)
            
            pixel_loss = criterion_logits(heatmap_preds, heatmaps.cuda())
            loss_per_sample_hm = pixel_loss.mean(dim=(1, 2))
            
            if num_gf > 0:
                loss_hm_gf = loss_per_sample_hm[is_gf].mean()
                loss += loss_hm_gf
                log_hm_loss = loss_hm_gf.item()

            if num_est > 0:
                loss_hm_est = loss_per_sample_hm[is_est].mean()
                loss += loss_hm_est * 0.1 
            
            if preds['seg'] is not None:
                if isinstance(preds['seg'], list): seg_preds = torch.stack(preds['seg']).squeeze(dim=1)
                else: seg_preds = preds['seg'].squeeze(dim=1)
                seg_loss_per_sample = criterion_logits(seg_preds, seg_mask.cuda()).mean(dim=(1,2))
                if num_gf > 0:
                    loss_seg_gf = seg_loss_per_sample[is_gf].mean()
                    loss += loss_seg_gf * 0.1
                    log_seg_loss = loss_seg_gf.item()
                if num_est > 0:
                    loss_seg_est = seg_loss_per_sample[is_est].mean()
                    loss += loss_seg_est * 0.1 * 0.1

            if preds['text_loss'] is not None:
                if num_gf > 0:
                    loss_text_gf = preds['text_loss'][is_gf].mean()
                    loss += loss_text_gf * 0.01
                    log_text_loss = loss_text_gf.item()
                if num_est > 0:
                    loss_text_est = preds['text_loss'][is_est].mean()
                    loss += loss_text_est * 0.01 * 0.1

            if preds['direction'] is not None and num_gf > 0:
                if isinstance(preds['direction'], list): dir_preds = torch.stack(preds['direction']).squeeze(dim=1)
                else: dir_preds = preds['direction'].squeeze(dim=1)
                clean_target = gaze_directions.clone()
                clean_target[(clean_target < 0) | (clean_target >= 8)] = -100
                dir_loss_raw = criterion_ce(dir_preds, clean_target.cuda())
                loss += dir_loss_raw * 0.02
                log_dir_loss = dir_loss_raw.item()

            if preds['gaze3d'] is not None and args.is_mix_gaze_estimation:
                if isinstance(preds['gaze3d'], list): gaze3d_preds = torch.stack(preds['gaze3d']).squeeze(dim=1)
                else: gaze3d_preds = preds['gaze3d'].squeeze(dim=1)
                loss_gaze3d_raw = F.l1_loss(gaze3d_preds, gaze3d.cuda(), reduction='none').mean(dim=1)
                if num_est > 0:
                    loss_g3d_est = loss_gaze3d_raw[is_est].mean()
                    loss += loss_g3d_est * 0.1
                    log_g3d_loss = loss_g3d_est.item()
            
            loss.backward()
            optimizer.step()
            
            print_dict = {'heatmap_loss': log_hm_loss, 'total_loss': loss.item()}
            if preds['text_loss'] is not None: print_dict['text_loss'] = log_text_loss
            if preds['direction'] is not None: print_dict['direction_loss'] = log_dir_loss
            if preds['seg'] is not None: print_dict['seg_loss'] = log_seg_loss
            if preds['gaze3d'] is not None: print_dict['gaze3d_loss'] = log_g3d_loss
            
            pbar.set_postfix(print_dict)
            if cur_iter % args.log_iter == 0: 
                wandb.log(print_dict)
                logger.info(f"Iter {cur_iter}/{len(train_dl_gf)} "+", ".join([f"{k}: {v:.4f}" for k, v in print_dict.items()]))

        scheduler.step()
        ckpt_path_last = os.path.join(exp_dir, 'last.pt')
        model_to_save = model.module if hasattr(model, 'module') else model
        torch.save(model_to_save.get_gazelle_state_dict(), ckpt_path_last)

        # --- Evaluation & Visualization ---
        logger.info(f"[Epoch {epoch} Evaluation]")
        model.eval()
        avg_l2s, min_l2s, aucs = [], [], []
        for cur_iter, batch in tqdm(enumerate(eval_dl), total=len(eval_dl)):
             imgs, bboxes, eyes, gazex, gazey, inout, heights, widths, observer_expressions, gaze_directions, gaze_point_expressions, seg_mask, is_face_crop_mode, gaze3d, has_3d = batch
             with torch.no_grad():
                preds = model({"images": imgs.cuda(), "bboxes": [[bbox] for bbox in bboxes], "eyes": eyes, "observer_expression_ids": observer_expressions.cuda()})
             
             if isinstance(preds['heatmap'], list): heatmap_preds = torch.stack(preds['heatmap']).squeeze(dim=1)
             else: heatmap_preds = preds['heatmap'].squeeze(dim=1)
             heatmap_preds = heatmap_preds.cpu()
             
             # 在 Eval 的第一个 batch 进行一次可视化
             if cur_iter == 0:
                idx = 0 # 选 batch 里的第一个样本
                vis_preds = {
                    'heatmap': heatmap_preds[idx],
                    'seg': preds['seg'][idx] if preds['seg'] is not None else None,
                    'text': "Epoch Evaluation - Gaze Detection" 
                }
                # 注意：这里需要原始图片路径，如果是 GazeDataset，可以通过 eval_dataset.img_names 获取
                # 但 eval_dl 返回的是经过变换的 Tensor。为了演示，我们假设用 batch 里的信息尝试恢复
                vis_save_path = os.path.join(exp_dir, f"vis_epoch_{epoch}.png")
                # 如果数据集能提供 path 信息则更佳，此处暂用 placeholder 逻辑
                try:
                    img_path = eval_dataset.images[idx] # 假设数据集有 images 属性
                    visualize_step(img_path, vis_preds, [bboxes[idx]], [gazex[idx][0], gazey[idx][0]], vis_save_path)
                    logger.info(f"Visualization saved to {vis_save_path}")
                except:
                    pass

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