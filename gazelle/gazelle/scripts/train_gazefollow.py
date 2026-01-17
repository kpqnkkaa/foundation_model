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

# 1. 强制设置可见显卡为 2, 3
os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"

from gazelle.dataloader import GazeDataset, collate_fn
from gazelle.model import get_gazelle_model
from gazelle.utils import gazefollow_auc, gazefollow_l2

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default="sam_vitb")
parser.add_argument('--data_path', type=str, default='/mnt/nvme1n1/lululemon/xjj/datasets/resized/gazefollow_extended')
parser.add_argument('--ckpt_save_dir', type=str, default='./experiments')
parser.add_argument('--wandb_project', type=str, default='gazelle')
parser.add_argument('--exp_name', type=str, default='train_sam_vitb_gazefollow')
parser.add_argument('--log_iter', type=int, default=10, help='how often to log loss during training')
parser.add_argument('--max_epochs', type=int, default=15)
parser.add_argument('--batch_size', type=int, default=60)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--n_workers', type=int, default=8)
args = parser.parse_args()

# --- 自定义多卡包装器 (终极修复版) ---
class DataParallelWrapper(nn.DataParallel):
    def forward(self, input_dict):
        # input_dict 是 main 函数里传进来的那个字典
        images = input_dict['images'] # Tensor [Batch, ...]
        bboxes = input_dict['bboxes'] # List [Batch, ...]

        # 1. 手动切分数据
        batch_size = images.shape[0]
        num_replicas = len(self.device_ids)
        split_size = (batch_size + num_replicas - 1) // num_replicas

        replicas_inputs = []
        for i in range(num_replicas):
            start_idx = i * split_size
            end_idx = min((i + 1) * split_size, batch_size)
            
            if start_idx >= end_idx:
                break
            
            # 获取当前切片应该去的目标显卡
            target_device = self.device_ids[i]
            
            replica_dict = {
                # 关键修复：显式移动 Tensor 到目标显卡
                "images": images[start_idx:end_idx].to(target_device),
                "bboxes": bboxes[start_idx:end_idx]
            }
            # 关键修复：包装进 Tuple，防止解包错误
            replicas_inputs.append((replica_dict,))

        # 2. 复制模型并并行前向传播
        replicas = self.replicate(self.module, self.device_ids[:len(replicas_inputs)])
        
        # 关键修复：传入 None 作为 kwargs
        outputs = self.parallel_apply(replicas, replicas_inputs, None)
        
        # 3. 手动收集结果 (Custom Gather)
        gathered_output = {}
        if len(outputs) > 0:
            for key in outputs[0].keys():
                first_val = outputs[0][key]
                
                if first_val is None:
                    gathered_output[key] = None
                    continue
                    
                # 处理 List 类型 (heatmap, inout 等)
                if isinstance(first_val, list):
                    merged_list = []
                    for out in outputs:
                        val_list = out[key]
                        device_adjusted_list = [t.to(self.output_device) for t in val_list]
                        merged_list.extend(device_adjusted_list)
                    gathered_output[key] = merged_list
                
                # 处理 Tensor 类型 (如果有的话)
                elif isinstance(first_val, torch.Tensor):
                    gathered_output[key] = torch.cat(
                        [out[key].to(self.output_device) for out in outputs], dim=0
                    )
        
        return gathered_output

# --- 日志设置函数 ---
def setup_logger(log_file):
    logger = logging.getLogger('gazelle_logger')
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        fh = logging.FileHandler(log_file, mode='w')
        formatter = logging.Formatter('%(asctime)s - %(message)s')
        fh.setFormatter(formatter)
        logger.addHandler(fh)
    return logger

def main():
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

    for param in model.backbone.parameters(): # freeze backbone
        param.requires_grad = False
    
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Learnable parameters: {n_params}")

    # --- 多卡并行设置 ---
    if torch.cuda.device_count() > 1:
        print(f"Detected {torch.cuda.device_count()} GPUs. Using DataParallelWrapper.")
        model = DataParallelWrapper(model)

    train_dataset = GazeDataset('gazefollow', args.data_path, 'train', transform)
    train_dl = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn, num_workers=args.n_workers)
    
    eval_dataset = GazeDataset('gazefollow', args.data_path, 'test', transform)
    eval_dl = torch.utils.data.DataLoader(eval_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn, num_workers=args.n_workers)

    loss_fn = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
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
            imgs, bboxes, gazex, gazey, inout, heights, widths, heatmaps = batch

            optimizer.zero_grad()
            
            preds = model({"images": imgs.cuda(), "bboxes": [[bbox] for bbox in bboxes]})
            
            if isinstance(preds['heatmap'], list):
                 heatmap_preds = torch.stack(preds['heatmap']).squeeze(dim=1)
            else:
                 heatmap_preds = preds['heatmap'].squeeze(dim=1)

            loss = loss_fn(heatmap_preds, heatmaps.cuda())
            loss.backward()
            optimizer.step()
            
            epoch_losses.append(loss.item())

            pbar.set_postfix({'loss': f"{loss.item():.4f}"})

            if cur_iter % args.log_iter == 0:
                wandb.log({"train/loss": loss.item()})
                logger.info(f"Iter {cur_iter}/{len(train_dl)}, Loss={loss.item():.4f}")

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
            imgs, bboxes, gazex, gazey, inout, heights, widths = batch

            with torch.no_grad():
                preds = model({"images": imgs.cuda(), "bboxes": [[bbox] for bbox in bboxes]})

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