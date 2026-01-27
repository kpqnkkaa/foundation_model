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
cv2.setNumThreads(0) # 禁止 OpenCV 多线程，防止死锁
cv2.ocl.setUseOpenCL(False)

# 1. 强制设置可见显卡为 2, 3
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

from gazelle.dataloader import GazeDataset, collate_fn
from gazelle.model import get_gazelle_model
from gazelle.utils import gazefollow_auc, gazefollow_l2

from pcgrad import PCGrad  # 确保安装: pip install pcgrad

class MultiTaskLoss(nn.Module):
    def __init__(self, num_tasks=4, anchor_task_idx=0):
        super().__init__()
        self.num_tasks = num_tasks
        self.anchor_task_idx = anchor_task_idx
        
        # 我们只需要学习 (num_tasks - 1) 个参数
        # 这里的 params 对应除了 anchor 以外的其他任务
        self.log_vars_aux = nn.Parameter(torch.zeros(num_tasks - 1))
        
        # 计算让权重等于 1.0 所需要的 log_var 值
        # Weight = 0.5 * exp(-log_var) = 1.0  =>  log_var = -ln(2)
        self.fixed_log_var = -math.log(2.0) 

    def forward(self, input_losses):
        """
        input_losses: list of (loss, task_idx)
        """
        weighted_losses = []
        
        # 用于追踪当前用到第几个可学习参数
        aux_idx = 0 
        
        # 我们需要根据 task_idx 重新排序 input_losses 或者在循环里判断
        # 为了安全，建议直接按顺序遍历 losses_to_optimize
        for loss, task_idx in input_losses:
            
            if task_idx == self.anchor_task_idx:
                # === 主任务 (Heatmap) ===
                # 强制使用固定权重 1.0
                # L_weighted = 1.0 * Loss + const
                # 注意：这里加上 0.5 * fixed_log_var 虽然是常数，但为了保持 Loss 量级一致建议保留
                precision = 1.0
                log_var = self.fixed_log_var
                
                # 这里的 Loss 直接保留原大小，梯度倍率为 1.0
                weighted_loss = precision * loss + 0.5 * log_var
                
            else:
                # === 辅助任务 (Seg, Dir, Text) ===
                # 使用可学习的参数
                current_log_var = self.log_vars_aux[aux_idx]
                aux_idx += 1
                
                precision = torch.exp(-current_log_var)
                weighted_loss = precision * loss + 0.5 * current_log_var
            
            weighted_losses.append(weighted_loss)
            
        return weighted_losses

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default="dinov2_vitb_multi_input")
parser.add_argument('--data_path', type=str, default='/mnt/nvme1n1/lululemon/xjj/datasets/resized/gazefollow_extended')
parser.add_argument('--ckpt_save_dir', type=str, default='./experiments')
parser.add_argument('--wandb_project', type=str, default=None)
parser.add_argument('--exp_name', type=str, default="dinov2_vitb_multi_input_is_partial_input")
parser.add_argument('--is_partial_input', default=False, action='store_true')
parser.add_argument('--log_iter', type=int, default=10, help='how often to log loss during training')
parser.add_argument('--max_epochs', type=int, default=15)
parser.add_argument('--batch_size', type=int, default=60)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--n_workers', type=int, default=8)
args = parser.parse_args()

# --- 自定义多卡包装器 (终极修复版) ---
class DataParallelWrapper(nn.DataParallel):
    def forward(self, input_dict):
        # 1. 自动获取 Batch Size (假设 input_dict 里至少有一个 key 是 Tensor 或 List)
        # 通常我们用 'images' 来确定 batch size
        if 'images' in input_dict:
            batch_size = input_dict['images'].shape[0]
        else:
            #以此类推，找一个存在的key
            first_key = next(iter(input_dict))
            batch_size = len(input_dict[first_key])

        num_replicas = len(self.device_ids)
        split_size = (batch_size + num_replicas - 1) // num_replicas

        replicas_inputs = []
        for i in range(num_replicas):
            start_idx = i * split_size
            end_idx = min((i + 1) * split_size, batch_size)
            
            if start_idx >= end_idx:
                break
            
            target_device = self.device_ids[i]
            replica_dict = {}

            # ==========================================
            # 【修改点】: images 也放进循环统一处理
            # ==========================================
            for k, v in input_dict.items():
                # 处理 Tensor (images, eyes, expr_ids, expression_ids 等)
                if isinstance(v, torch.Tensor):
                    replica_dict[k] = v[start_idx:end_idx].to(target_device)
                
                # 处理 List (bboxes 等)
                elif isinstance(v, list):
                    replica_dict[k] = v[start_idx:end_idx]
                
                # 处理 None (比如 eval 时 expression_ids 为 None)
                elif v is None:
                    replica_dict[k] = None
            
            # 包装进 tuple
            replicas_inputs.append((replica_dict,))

        # 2. 复制模型并并行前向传播
        replicas = self.replicate(self.module, self.device_ids[:len(replicas_inputs)])
        outputs = self.parallel_apply(replicas, replicas_inputs, None)
        
        # 3. 手动收集结果
        gathered_output = {}
        if len(outputs) > 0:
            for key in outputs[0].keys():
                first_val = outputs[0][key]
                
                if first_val is None:
                    gathered_output[key] = None
                    continue
                    
                # ==========================================
                # 【修改点】: 专门处理 expression_loss (Scalar)
                # ==========================================
                # 如果是 0 维 Tensor (Scalar)，说明是 Loss，取平均
                if isinstance(first_val, torch.Tensor) and first_val.ndim == 0:
                     # 将各卡上的 loss stack 起来取 mean，并转回主卡
                     stacked_loss = torch.stack([out[key].to(self.output_device) for out in outputs])
                     gathered_output[key] = stacked_loss.mean()

                # 处理普通 Tensor (logits, heatmaps 等，维度 > 0)
                elif isinstance(first_val, torch.Tensor) and first_val.ndim > 0:
                    gathered_output[key] = torch.cat(
                        [out[key].to(self.output_device) for out in outputs], dim=0
                    )

                # 处理 List
                elif isinstance(first_val, list):
                    merged_list = []
                    for out in outputs:
                        val_list = out[key]
                        # 只有 tensor list 需要 .to()，普通 list 不用
                        if len(val_list) > 0 and isinstance(val_list[0], torch.Tensor):
                            merged_list.extend([t.to(self.output_device) for t in val_list])
                        else:
                            merged_list.extend(val_list)
                    gathered_output[key] = merged_list
                
        return gathered_output

# --- 日志设置函数 ---
def setup_logger(log_file):
    logger = logging.getLogger('gazelle_logger')
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        # 1. 文件输出
        fh = logging.FileHandler(log_file, mode='w')
        formatter = logging.Formatter('%(asctime)s - %(message)s')
        fh.setFormatter(formatter)
        logger.addHandler(fh)
        
        # 2. 【新增】屏幕/终端输出
        sh = logging.StreamHandler()
        sh.setFormatter(formatter)
        logger.addHandler(sh)
    return logger

def main():
    if args.wandb_project is None:
        args.wandb_project = args.model
    if args.exp_name is None:
        args.exp_name = args.model
    wandb.init(
        project=args.wandb_project,
        name=args.exp_name,
        config=vars(args)
    )
    
    # 创建实验目录
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    exp_dir = os.path.join(args.ckpt_save_dir, args.exp_name, timestamp)
    os.makedirs(exp_dir, exist_ok=True)
    
    # 初始化日志
    log_file_path = os.path.join(exp_dir, 'log.txt')
    logger = setup_logger(log_file_path)
    print(f"训练开始。详细日志将输出到: {log_file_path}")
    logger.info(f"Experiment Config: {vars(args)}")

    model, transform = get_gazelle_model(args.model)
    model.cuda()
    # mt_loss = MultiTaskLoss(num_tasks=4, anchor_task_idx=0).cuda()

    # for param in model.backbone.parameters(): # freeze backbone
    #     param.requires_grad = False
    
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Learnable parameters: {n_params}")

    # --- 多卡并行设置 ---
    if torch.cuda.device_count() > 1:
        print(f"Detected {torch.cuda.device_count()} GPUs. Using DataParallelWrapper.")
        model = DataParallelWrapper(model)

    train_dataset = GazeDataset('gazefollow', args.data_path, 'train', transform, is_partial_input=args.is_partial_input)
    train_dl = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn, num_workers=args.n_workers)
    
    eval_dataset = GazeDataset('gazefollow', args.data_path, 'test', transform, is_partial_input=args.is_partial_input)
    eval_dl = torch.utils.data.DataLoader(eval_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn, num_workers=args.n_workers)

    # 在 main 函数中定义 optimizer
    # # 将 Fusion 层的参数学习率设低一点 (例如 1e-4)，Head 设高一点 (1e-3)
    # param_dicts = [
    #     {"params": [p for n, p in model.named_parameters() if "fusion" in n and p.requires_grad], "lr": args.lr * 0.1},
    #     {"params": [p for n, p in model.named_parameters() if "fusion" not in n and p.requires_grad], "lr": args.lr},
    # ]

    # # 1. 区分参数组
    # finetune_params = []   # 存放 Fusion, LoRA, Prompt Encoder
    # scratch_params = []    # 存放 GazeLLE Head, Linear, Projector

    # for name, param in model.named_parameters():
    #     if not param.requires_grad:
    #         continue
        
    #     # 判定逻辑：如果是 backbone 里的 fusion 或 prompt_encoder 或 lora，归为微调组
    #     if "backbone" in name and ("fusion" in name or "prompt" in name or "lora" in name):
    #         finetune_params.append(param)
    #         # print(f"Finetune (Low LR): {name}") # 调试用
    #     else:
    #         scratch_params.append(param)
    #         # print(f"Scratch (High LR): {name}") # 调试用

    # param_dicts = [
    #     {'params': scratch_params, 'lr': args.lr},
    #     {'params': finetune_params, 'lr': args.lr * 0.1}
    # ]
    criterion_bce = nn.BCEWithLogitsLoss()
    criterion_ce = nn.CrossEntropyLoss(ignore_index=-1) # 用于Direction
    
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    # optimizer = torch.optim.Adam(list(model.parameters()) + list(mt_loss.parameters()), lr=args.lr)
    # pcgrad = PCGrad(optimizer)
    # optimizer = torch.optim.AdamW(param_dicts, lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.max_epochs, eta_min=1e-7)

    best_min_l2 = 1.0
    best_epoch = None

    for epoch in range(args.max_epochs):
        # --- TRAIN EPOCH ---
        model.train()
        logger.info(f"\n[Epoch {epoch} Training]")
        
        pbar = tqdm(enumerate(train_dl), total=len(train_dl), 
                    desc=f"Epoch {epoch} [Train]", unit="batch", dynamic_ncols=True)
        
        epoch_losses = []
        
        for cur_iter, batch in pbar:
            imgs, bboxes, eyes, gazex, gazey, inout, heights, widths, heatmaps, observer_expressions, gaze_directions, gaze_point_expressions, seg_mask, is_face_crop_mode = batch

            optimizer.zero_grad()
            
            # gaze_point_expression_ids用于计算 Text Generation Loss (仅训练时需要)
            preds = model({"images": imgs.cuda(), "bboxes": [[bbox] for bbox in bboxes], "eyes": eyes, "observer_expression_ids": observer_expressions.cuda(), "gaze_point_expression_ids": gaze_point_expressions.cuda()})
            losses_to_optimize = []

            if isinstance(preds['heatmap'], list):
                 heatmap_preds = torch.stack(preds['heatmap']).squeeze(dim=1)
            else:
                 heatmap_preds = preds['heatmap'].squeeze(dim=1)

            loss = torch.tensor(0.0, device=heatmap_preds.device)
            heatmap_loss = criterion_bce(heatmap_preds, heatmaps.cuda())
            loss += heatmap_loss
            # heatmap_loss = 0
            # losses_to_optimize.append((heatmap_loss, 0))

            if preds['text_loss'] is not None:
                text_loss = preds['text_loss']
                loss += text_loss*0.01
                # losses_to_optimize.append((text_loss*0.01, 3))
            else:
                text_loss = None
            
            if preds['seg'] is not None:
                if isinstance(preds['seg'], list):
                    preds['seg'] = torch.stack(preds['seg']).squeeze(dim=1)
                else:
                    preds['seg'] = preds['seg'].squeeze(dim=1)
                seg_loss = criterion_bce(preds['seg'], seg_mask.cuda())
                loss += seg_loss*0.1
                # losses_to_optimize.append((seg_loss*0.1, 1))
            else:
                seg_loss = None

            if preds['direction'] is not None:
                if isinstance(preds['direction'], list):
                    preds['direction'] = torch.stack(preds['direction']).squeeze(dim=1)
                else:
                    preds['direction'] = preds['direction'].squeeze(dim=1)
                direction_loss = criterion_ce(preds['direction'], gaze_directions.cuda())
                loss += direction_loss*0.02
                # losses_to_optimize.append((direction_loss*0.02, 2))
            else:
                direction_loss = None

            # weighted_losses = mt_loss(losses_to_optimize)
            # weighted_losses = [loss for loss, _ in losses_to_optimize]
            # pcgrad.pc_backward(weighted_losses)
            # loss = sum(weighted_losses)
            loss.backward()
            optimizer.step()
            
            epoch_losses.append(heatmap_loss.item())

            if text_loss is not None:
                pbar.set_postfix({'heatmap_loss': f"{heatmap_loss.item():.4f}", 'text_loss': f"{text_loss.item():.4f}", 'seg_loss': f"{seg_loss.item():.4f}", 'direction_loss': f"{direction_loss.item():.4f}", 'loss': f"{loss.item():.4f}"})
            else:
                pbar.set_postfix({'loss': f"{loss.item():.4f}"})

            if cur_iter % args.log_iter == 0:
                if text_loss is not None:
                    wandb.log({"train/heatmap_loss": heatmap_loss.item(), "train/text_loss": text_loss.item(), "train/seg_loss": seg_loss.item(), "train/direction_loss": direction_loss.item(), "train/loss": loss.item()})
                    logger.info(f"Iter {cur_iter}/{len(train_dl)}, heatmap Loss={heatmap_loss.item():.4f}, Text Loss={text_loss.item():.4f}, Seg Loss={seg_loss.item():.4f}, Direction Loss={direction_loss.item():.4f}, Loss={loss.item():.4f}")
                else:
                    wandb.log({"train/heatmap_loss": heatmap_loss.item(), "train/loss": loss.item()})
                    logger.info(f"Iter {cur_iter}/{len(train_dl)}, heatmap Loss={heatmap_loss.item():.4f}, Loss={loss.item():.4f}")
        scheduler.step()
        avg_train_loss = np.mean(epoch_losses)
        logger.info(f"End of Epoch {epoch} Train - Avg Loss: {avg_train_loss:.4f}")

        # --- [MODIFIED] 保存 Last 模型 ---
        # 无论结果如何，每一轮都覆盖保存 'last.pt'
        ckpt_path_last = os.path.join(exp_dir, 'last.pt')
        # 兼容 DataParallel 获取原始 module
        model_to_save = model.module if hasattr(model, 'module') else model
        torch.save(model_to_save.get_gazelle_state_dict(), ckpt_path_last)
        logger.info(f"Saved last checkpoint to {ckpt_path_last}")

        # --- EVAL EPOCH ---
        logger.info(f"[Epoch {epoch} Evaluation]")
        model.eval()
        
        avg_l2s = []
        min_l2s = []
        aucs = []
        
        eval_pbar = tqdm(enumerate(eval_dl), total=len(eval_dl), 
                         desc=f"Epoch {epoch} [Eval]", unit="batch", dynamic_ncols=True)
        
        for cur_iter, batch in eval_pbar:
            imgs, bboxes, eyes, gazex, gazey, inout, heights, widths, observer_expressions, gaze_directions, gaze_point_expressions, seg_mask, is_face_crop_mode = batch

            with torch.no_grad():
                preds = model({"images": imgs.cuda(), "bboxes": [[bbox] for bbox in bboxes], "eyes": eyes, "observer_expression_ids": observer_expressions})

            if isinstance(preds['heatmap'], list):
                 heatmap_preds = torch.stack(preds['heatmap']).squeeze(dim=1)
            else:
                 heatmap_preds = preds['heatmap'].squeeze(dim=1)
            
            heatmap_preds = heatmap_preds.cpu()
            
            for i in range(heatmap_preds.shape[0]):
                auc = gazefollow_auc(heatmap_preds[i], gazex[i], gazey[i], heights[i], widths[i])
                avg_l2, min_l2 = gazefollow_l2(heatmap_preds[i], gazex[i], gazey[i])
                aucs.append(auc)
                avg_l2s.append(avg_l2)
                min_l2s.append(min_l2)

        epoch_avg_l2 = np.mean(avg_l2s)
        epoch_min_l2 = np.mean(min_l2s)
        epoch_auc = np.mean(aucs)

        wandb.log({"eval/auc": epoch_auc, "eval/min_l2": epoch_min_l2, "eval/avg_l2": epoch_avg_l2, "epoch": epoch})
        
        result_msg = "EVAL EPOCH {}: AUC={}, Min L2={}, Avg L2={}".format(
            epoch, round(epoch_auc, 4), round(epoch_min_l2, 4), round(epoch_avg_l2, 4)
        )
        logger.info(result_msg)
        
        # --- [MODIFIED] 保存 Best 模型 ---
        if epoch_min_l2 < best_min_l2:
            best_min_l2 = epoch_min_l2
            best_epoch = epoch
            
            # 如果是当前最好，保存为 'best.pt' (覆盖旧的 best)
            ckpt_path_best = os.path.join(exp_dir, 'best.pt')
            torch.save(model_to_save.get_gazelle_state_dict(), ckpt_path_best)
            logger.info(f"New best performance! Saved best checkpoint to {ckpt_path_best}")

    final_msg = "Completed training. Best Min L2 of {} obtained at epoch {}".format(round(best_min_l2, 4), best_epoch)
    print(f"\n{final_msg}")
    logger.info(final_msg)

if __name__ == '__main__':
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    main()