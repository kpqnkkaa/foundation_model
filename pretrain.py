import os
# 设置 HuggingFace 镜像
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import argparse
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
from datetime import datetime

from dataset import GazeEstimationDataset, BalancedEstimationDataset
from model import EstimationBranch

def get_args():
    parser = argparse.ArgumentParser(description="Gaze Estimation Branch Pretraining")
    
    parser.add_argument('--gpus', type=str, default='2,3')
    parser.add_argument('--save_dir', type=str, default="/mnt/nvme1n1/lululemon/fm_shijing/foudation_model")
    
    # === 数据集路径 (保持与 train.py 一致) ===
    # parser.add_argument('--eth_label', type=str, default='/mnt/nvme1n1/lululemon/xjj/datasets/ETH-Gaze/Label/train_temp.label')
    # parser.add_argument('--eth_img', type=str, default='/mnt/nvme1n1/lululemon/xjj/datasets/ETH-Gaze/Image')
    parser.add_argument('--eth_label', type=str, default='None')
    parser.add_argument('--eth_img', type=str, default='None')
    parser.add_argument('--pretrain_pth', type=str, default='/mnt/nvme1n1/lululemon/fm_shijing/foudation_model/Pretrain_ETH-G360_Gaze360_EstBranch_20260111_1813/pretrain_latest.pth')
    parser.add_argument('--gaze360_label', type=str, default='/mnt/nvme1n1/lululemon/xjj/datasets/Gaze360_new/Label/train.label')
    parser.add_argument('--gaze360_img', type=str, default='/mnt/nvme1n1/lululemon/xjj/datasets/Gaze360_new/Image')
    
    # 训练参数
    parser.add_argument('--batch_size', type=int, default=512, help="Pretrain can use larger batch size")
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--num_workers', type=int, default=16)
    
    return parser.parse_args()

def main():
    args = get_args()
    
    # 1. 环境设置
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    run_name = "Pretrain"
    # run_name加上训练数据的名称
    if args.eth_label and args.eth_label != 'None':
        run_name += "_ETH"
    if args.gaze360_label and args.gaze360_label != 'None':
        run_name += "_Gaze360"
        
    run_name += f"_EstBranch_{timestamp}"
    run_dir = os.path.join(args.save_dir, run_name)
    os.makedirs(run_dir, exist_ok=True)
    
    print(f"=== Start Pretraining (Estimation Branch Only) ===")
    print(f"Save Dir: {run_dir}")

    # 2. 数据集 (Balanced Sampling)
    transform_face = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    est_datasets = []
    
    # 加载 ETH-Gaze (如果配置了且不是 None)
    if args.eth_label and args.eth_label != 'None':
        if os.path.exists(args.eth_label):
            print("Loading ETH-Gaze...")
            est_datasets.append(GazeEstimationDataset(args.eth_label, args.eth_img, transform=transform_face))
        else:
            print(f"Warning: ETH path not found: {args.eth_label}")
    else:
        print("Skipping ETH-Gaze (Not configured)")
        
    # 加载 Gaze360
    if args.gaze360_label and args.gaze360_label != 'None':
        if os.path.exists(args.gaze360_label):
            print("Loading Gaze360...")
            est_datasets.append(GazeEstimationDataset(args.gaze360_label, args.gaze360_img, transform=transform_face))
        else:
            print(f"Warning: Gaze360 path not found: {args.gaze360_label}")
    
    if not est_datasets:
        raise ValueError("No estimation datasets found! Please check paths.")

    # 使用 BalancedEstimationDataset 进行均匀混合
    train_dataset = BalancedEstimationDataset(est_datasets)
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True, # Shuffle index pool
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True
    )
    print(f"Balanced Dataset Ready. Epoch Length: {len(train_dataset)}")

    # 3. 模型构建 (只实例化 EstimationBranch)
    print("Building Estimation Model...")
    model = EstimationBranch().to(device)
    
    # === 加载预训练权重 (如果存在) ===
    if args.pretrain_pth and args.pretrain_pth != 'None':
        if os.path.isfile(args.pretrain_pth):
            print(f"Loading pretrained weights from: {args.pretrain_pth}")
            checkpoint = torch.load(args.pretrain_pth, map_location=device)
            # 兼容保存时可能是 'model_state_dict' 或者是直接 state_dict 的情况
            state_dict = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint
            
            # 去除 DataParallel 可能引入的 'module.' 前缀
            new_state_dict = {}
            for k, v in state_dict.items():
                name = k.replace('module.', '')
                new_state_dict[name] = v
            
            # 加载权重，strict=False 允许微调时有部分层不匹配
            msg = model.load_state_dict(new_state_dict, strict=False)
            print(f"Pretrained weights loaded. {msg}")
        else:
            print(f"Warning: Pretrain path {args.pretrain_pth} does not exist. Starting from scratch.")
    else:
        print("No pretrain path provided. Starting from scratch.")

    # 多卡并行
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs")
        model = torch.nn.DataParallel(model)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.L1Loss() # 3D Angle Regression Loss

    # 4. 训练循环
    best_loss = float('inf')
    
    for epoch in range(args.epochs):
        model.train()
        train_dataset.shuffle() # [重要] 每个 epoch 重新打乱 BalancedDataset 的索引池
        
        running_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        
        for batch_data in pbar:
            # GazeEstimationDataset 返回的是 {'face_img': ..., '3D_ground_truth': ...}
            imgs = batch_data['face_img'].to(device)
            gts = batch_data['3D_ground_truth'].to(device) # [B, 2] (yaw, pitch)
            
            optimizer.zero_grad()
            
            # Forward
            # EstimationBranch forward 返回 (angles, features)
            outputs = model(imgs)
            
            # 兼容性处理
            if isinstance(outputs, (tuple, list)):
                preds = outputs[0]
            else:
                preds = outputs
            
            loss = criterion(preds, gts)
            
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            pbar.set_postfix({'l1_loss': loss.item()})
            
        avg_loss = running_loss / len(train_loader)
        print(f"Epoch {epoch+1} Finished. Avg Loss: {avg_loss:.4f}")
        
        # 保存 Checkpoint
        model_to_save = model.module if isinstance(model, torch.nn.DataParallel) else model
        
        state = {
            'epoch': epoch,
            'model_state_dict': model_to_save.state_dict(), # 保存的是纯净的 EstimationBranch 参数
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': avg_loss,
            'args': vars(args)
        }
        
        # 保存 Latest
        torch.save(state, os.path.join(run_dir, "pretrain_latest.pth"))
        
        # 保存 Best
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(state, os.path.join(run_dir, "pretrain_best.pth"))
            print(f"New Best Model Saved! (Loss: {best_loss:.4f})")

    print(f"Pretraining Finished. Best weights at: {os.path.join(run_dir, 'pretrain_best.pth')}")

if __name__ == "__main__":
    main()