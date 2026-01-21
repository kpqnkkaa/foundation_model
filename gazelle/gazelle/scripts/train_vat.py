import argparse
from datetime import datetime
import numpy as np
import os
import random
from sklearn.metrics import average_precision_score
import torch
import torch.nn as nn
import wandb
import logging
from tqdm import tqdm

# 1. 强制设置可见显卡
os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"

from gazelle.dataloader import GazeDataset, collate_fn
from gazelle.model import get_gazelle_model
from gazelle.utils import vat_auc, vat_l2

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default="gazelle_dinov2_vitb14_inout")
parser.add_argument('--init_ckpt', type=str, default='./checkpoints/gazelle_dinov2_vitb14.pt', help='checkpoint for initialization (trained on GazeFollow)')
parser.add_argument('--data_path', type=str, default='/mnt/nvme1n1/lululemon/xjj/datasets/resized/videoattentiontarget')
parser.add_argument('--frame_sample_every', type=int, default=6)
parser.add_argument('--ckpt_save_dir', type=str, default='./experiments')
parser.add_argument('--wandb_project', type=str, default='gazelle')
parser.add_argument('--exp_name', type=str, default='train_gazelle_vitb_vat')
parser.add_argument('--log_iter', type=int, default=10, help='how often to log loss during training')
parser.add_argument('--max_epochs', type=int, default=8)
parser.add_argument('--batch_size', type=int, default=60)
parser.add_argument('--inout_loss_lambda', type=float, default=1.0)
parser.add_argument('--lr_non_inout', type=float, default=1e-5)
parser.add_argument('--lr_inout', type=float, default=1e-2)
parser.add_argument('--n_workers', type=int, default=8)
args = parser.parse_args()

# --- 核心修复：自定义多卡包装器 (终极版) ---
class DataParallelWrapper(nn.DataParallel):
    def forward(self, input_dict):
        # input_dict 是 main 函数里传进来的字典
        images = input_dict['images'] 
        bboxes = input_dict['bboxes'] 

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
            
            target_device = self.device_ids[i]

            replica_dict = {
                # 关键修复：显式移动 Tensor 到该 Replica 所在的显卡上
                "images": images[start_idx:end_idx].to(target_device),
                "bboxes": bboxes[start_idx:end_idx]
            }
            # 关键修复：将字典包装在元组中 (dict,)
            replicas_inputs.append((replica_dict,))

        # 2. 复制模型并并行运行
        replicas = self.replicate(self.module, self.device_ids[:len(replicas_inputs)])
        # 关键修复：显式传入 kwargs=None
        outputs = self.parallel_apply(replicas, replicas_inputs, None)
        
        # 3. 手动收集结果 (Custom Gather)
        # 解决 List 类型被错误 stack 导致维度变成 [30, 2, ...] 的问题
        gathered_output = {}
        if len(outputs) > 0:
            for key in outputs[0].keys():
                first_val = outputs[0][key]
                if first_val is None:
                    gathered_output[key] = None
                    continue
                
                # 如果是 List (heatmap, inout)，使用 extend 拼接
                if isinstance(first_val, list):
                    merged_list = []
                    for out in outputs:
                        # 必须移回主设备 (output_device)
                        val_list = out[key]
                        device_adjusted_list = [t.to(self.output_device) for t in val_list]
                        merged_list.extend(device_adjusted_list)
                    gathered_output[key] = merged_list
                
                # 如果是 Tensor，使用 cat 拼接
                elif isinstance(first_val, torch.Tensor):
                    gathered_output[key] = torch.cat(
                        [out[key].to(self.output_device) for out in outputs], dim=0
                    )

        return gathered_output

# --- 日志设置 ---
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
    
    # 实验目录与日志
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    exp_dir = os.path.join(args.ckpt_save_dir, args.exp_name, timestamp)
    os.makedirs(exp_dir, exist_ok=True)
    
    log_file_path = os.path.join(exp_dir, 'log.txt')
    logger = setup_logger(log_file_path)
    print(f"训练启动。详细日志请查看: {log_file_path}")
    logger.info(f"Experiment Config: {vars(args)}")

    # 模型初始化
    model, transform = get_gazelle_model(args.model)
    
    # 加载预训练权重
    print("Initializing from {}".format(args.init_ckpt))
    if os.path.exists(args.init_ckpt):
        # 增加 map_location='cpu'
        model.load_gazelle_state_dict(torch.load(args.init_ckpt, map_location='cpu')) 
    else:
        logger.warning(f"Checkpoint {args.init_ckpt} not found! Training from scratch.")

    model.cuda()

    # 冻结 backbone
    for param in model.backbone.parameters():
        param.requires_grad = False
    
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Learnable parameters: {n_params}")

    # --- 多卡并行设置 ---
    if torch.cuda.device_count() > 1:
        print(f"Detected {torch.cuda.device_count()} GPUs. Using Custom DataParallelWrapper.")
        model = DataParallelWrapper(model)

    # 数据集
    train_dataset = GazeDataset('videoattentiontarget', args.data_path, 'train', transform, in_frame_only=False, sample_rate=args.frame_sample_every)
    train_dl = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn, num_workers=args.n_workers)
    
    eval_dataset = GazeDataset('videoattentiontarget', args.data_path, 'test', transform, in_frame_only=False, sample_rate=args.frame_sample_every)
    eval_dl = torch.utils.data.DataLoader(eval_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn, num_workers=args.n_workers)

    heatmap_loss_fn = nn.BCELoss()
    inout_loss_fn = nn.BCELoss()
    
    # 优化器参数组
    param_groups = [
        {'params': [param for name, param in model.named_parameters() if "inout" in name], 'lr': args.lr_inout},
        {'params': [param for name, param in model.named_parameters() if "inout" not in name], 'lr': args.lr_non_inout}
    ]
    optimizer = torch.optim.Adam(param_groups)

    best_epoch = None
    best_metric = 0.0 

    for epoch in range(args.max_epochs):
        # --- TRAIN EPOCH ---
        model.train()
        logger.info(f"\n[Epoch {epoch} Training]")
        
        pbar = tqdm(enumerate(train_dl), total=len(train_dl), 
                    desc=f"Epoch {epoch} [Train]", unit="batch", dynamic_ncols=True)
        
        epoch_losses = []

        for cur_iter, batch in pbar:
            imgs, bboxes, eyes, gazex, gazey, inout, heights, widths, heatmaps = batch

            optimizer.zero_grad()
            
            # 使用包装器处理输入
            # wrapper 会负责分发到 gpu1 并移回结果
            preds = model({"images": imgs.cuda(), "bboxes": [[bbox] for bbox in bboxes], "eyes": None, "expr_ids": observer_expressions.cuda()})
            
            # 处理多卡返回结果
            # 经过 Custom Gather 后，preds['heatmap'] 是一个长度为 Batch Size 的扁平 List
            if isinstance(preds['heatmap'], list):
                # stack -> [Batch, 1, H, W] -> squeeze -> [Batch, H, W]
                heatmap_preds = torch.stack(preds['heatmap']).squeeze(dim=1)
                inout_preds = torch.stack(preds['inout']).squeeze(dim=1)
            else:
                heatmap_preds = preds['heatmap'].squeeze(dim=1)
                inout_preds = preds['inout'].squeeze(dim=1)

            # compute heatmap loss only for in-frame gaze targets
            target_device = heatmap_preds.device
            inout_bool = inout.bool().to(target_device)
            target_heatmaps = heatmaps.to(target_device)
            target_inout = inout.float().to(inout_preds.device)

            if inout_bool.sum() > 0:
                heatmap_loss = heatmap_loss_fn(heatmap_preds[inout_bool], target_heatmaps[inout_bool])
            else:
                heatmap_loss = torch.tensor(0.0).to(target_device)

            inout_loss = inout_loss_fn(inout_preds, target_inout)
            loss = heatmap_loss + args.inout_loss_lambda * inout_loss
            
            loss.backward()
            optimizer.step()
            
            epoch_losses.append(loss.item())
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})

            if cur_iter % args.log_iter == 0:
                wandb.log({
                    "train/loss": loss.item(),
                    "train/heatmap_loss": heatmap_loss.item(),
                    "train/inout_loss": inout_loss.item()
                })
                logger.info(f"Iter {cur_iter}/{len(train_dl)}, Loss={loss.item():.4f} (HM: {heatmap_loss.item():.4f}, IO: {inout_loss.item():.4f})")

        avg_loss = np.mean(epoch_losses)
        logger.info(f"End of Epoch {epoch} Train - Avg Loss: {avg_loss:.4f}")

        # 保存 Checkpoint
        ckpt_path = os.path.join(exp_dir, 'epoch_{}.pt'.format(epoch))
        model_to_save = model.module if hasattr(model, 'module') else model
        torch.save(model_to_save.get_gazelle_state_dict(), ckpt_path)
        logger.info(f"Saved checkpoint to {ckpt_path}")

        # --- EVAL EPOCH ---
        logger.info(f"[Epoch {epoch} Evaluation]")
        model.eval()
        
        l2s = []
        aucs = []
        all_inout_preds = []
        all_inout_gts = []
        
        eval_pbar = tqdm(enumerate(eval_dl), total=len(eval_dl), 
                         desc=f"Epoch {epoch} [Eval]", unit="batch", dynamic_ncols=True)

        for cur_iter, batch in eval_pbar:
            imgs, bboxes, eyes, gazex, gazey, inout, heights, widths = batch

            with torch.no_grad():
                preds = model({"images": imgs.cuda(), "bboxes": [[bbox] for bbox in bboxes], "eyes": None, "expr_ids": observer_expressions.cuda()})

            if isinstance(preds['heatmap'], list):
                heatmap_preds = torch.stack(preds['heatmap']).squeeze(dim=1)
                inout_preds = torch.stack(preds['inout']).squeeze(dim=1)
            else:
                heatmap_preds = preds['heatmap'].squeeze(dim=1)
                inout_preds = preds['inout'].squeeze(dim=1)
            
            # 转 CPU 计算指标
            heatmap_preds = heatmap_preds.cpu()
            inout_preds = inout_preds.cpu()
            
            for i in range(heatmap_preds.shape[0]):
                if inout[i] == 1: # in-frame
                    auc = vat_auc(heatmap_preds[i], gazex[i][0], gazey[i][0])
                    l2 = vat_l2(heatmap_preds[i], gazex[i][0], gazey[i][0])
                    aucs.append(auc)
                    l2s.append(l2)
                
                all_inout_preds.append(inout_preds[i].item())
                all_inout_gts.append(inout[i])

        epoch_l2 = np.mean(l2s) if l2s else 0.0
        epoch_auc = np.mean(aucs) if aucs else 0.0
        epoch_inout_ap = average_precision_score(all_inout_gts, all_inout_preds)

        wandb.log({"eval/auc": epoch_auc, "eval/l2": epoch_l2, "eval/inout_ap": epoch_inout_ap, "epoch": epoch})
        
        result_msg = "EVAL EPOCH {}: AUC={}, L2={}, Inout AP={}".format(
            epoch, round(epoch_auc, 4), round(epoch_l2, 4), round(epoch_inout_ap, 4)
        )
        logger.info(result_msg)

        if epoch_auc > best_metric:
            best_metric = epoch_auc
            best_epoch = epoch

    final_msg = f"Completed training. Best AUC of {round(best_metric, 4)} obtained at epoch {best_epoch}"
    print(f"\n{final_msg}")
    logger.info(final_msg)

if __name__ == '__main__':
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    main()