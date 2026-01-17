import os
# 设置 HuggingFace 镜像
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import argparse
import glob
import time
import torch
import numpy as np
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
from collections import OrderedDict

# === 引入项目模块 ===
from dataset import GazeEstimationDataset
from model import EstimationBranch
import utils

# ==========================================
# 0. 辅助功能: 自动寻找最新 Pretrain 模型
# ==========================================
def find_latest_pretrain_checkpoint(root_dir):
    """
    遍历 root_dir 下所有以 'Pretrain' 开头的文件夹，
    找到修改时间最新的那个，并加载其中的 pretrain_best.pth
    """
    print(f"Searching for latest Pretrain experiment in: {root_dir} ...")
    
    if not os.path.exists(root_dir):
        raise FileNotFoundError(f"Model directory not found: {root_dir}")

    # 1. 查找所有 Pretrain 开头的文件夹
    search_pattern = os.path.join(root_dir, "Pretrain*")
    all_dirs = glob.glob(search_pattern)
    
    # 过滤出是文件夹的路径
    all_dirs = [d for d in all_dirs if os.path.isdir(d)]
    
    if not all_dirs:
        raise FileNotFoundError(f"No 'Pretrain*' directories found in {root_dir}")

    # 2. 按修改时间排序，取最新的文件夹
    latest_dir = max(all_dirs, key=os.path.getmtime)
    print(f"Found latest experiment dir: {latest_dir}")
    
    # 3. 寻找 best 权重
    best_ckpt = os.path.join(latest_dir, "pretrain_best.pth")
    if not os.path.exists(best_ckpt):
        print(f"Warning: 'pretrain_best.pth' not found, trying 'pretrain_latest.pth'...")
        best_ckpt = os.path.join(latest_dir, "pretrain_latest.pth")
        
    if not os.path.exists(best_ckpt):
        raise FileNotFoundError(f"No .pth checkpoint found in {latest_dir}")

    mod_time = time.ctime(os.path.getmtime(best_ckpt))
    print(f"Target Checkpoint: {best_ckpt}")
    print(f"Last modified: {mod_time}")
    
    return best_ckpt

# ==========================================
# 1. 评估逻辑
# ==========================================
def eval_estimation_branch(model, dataloader, device, dataset_name):
    model.eval()
    errors = []
    print(f"\n>>> Evaluating {dataset_name} (Estimation Branch)...")
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc=dataset_name):
            # GazeEstimationDataset 返回 dict: {'face_img':..., '3D_ground_truth':...}
            imgs = batch['face_img'].to(device)
            gts = batch['3D_ground_truth'].cpu().numpy() # [B, 2] (yaw, pitch)

            # Forward (EstimationBranch 只返回 angles, features)
            # 我们只需要 angles
            outputs = model(imgs)
            
            # 兼容处理：如果模型返回 tuple (angles, feats)，取第一个
            if isinstance(outputs, tuple):
                pred_angles = outputs[0]
            else:
                pred_angles = outputs
            
            pred_angles = pred_angles.cpu().numpy()
            
            for i in range(len(imgs)):
                # 使用 utils 计算 3D 角度误差
                err = utils.calculate_3d_angular_error(pred_angles[i], gts[i])
                errors.append(err)
            
    mean_err = np.mean(errors)
    print(f"[{dataset_name}] Mean 3D Angular Error: {mean_err:.4f}°")
    return mean_err

# ==========================================
# 2. Main
# ==========================================
def get_args():
    parser = argparse.ArgumentParser(description="Gaze Estimation Branch Testing")
    
    parser.add_argument('--gpus', type=str, default='0')
    
    # 模型路径相关
    parser.add_argument('--checkpoint', type=str, default=None, help="Specific .pth path. If None, auto search in model_dir.")
    parser.add_argument('--model_dir', type=str, default="/mnt/nvme1n1/lululemon/fm_shijing/foudation_model", help="Root dir containing Pretrain_* folders")
    
    parser.add_argument('--batch_size', type=int, default=128)
    
    # === 数据集路径 (默认值复用 test.py) ===
    parser.add_argument('--eth_label', type=str, default='/mnt/nvme1n1/lululemon/xjj/datasets/ETH-Gaze/Label/test_temp.label')
    parser.add_argument('--eth_img', type=str, default='/mnt/nvme1n1/lululemon/xjj/datasets/ETH-Gaze/Image')
    parser.add_argument('--gaze360_label', type=str, default='/mnt/nvme1n1/lululemon/xjj/datasets/Gaze360/Label/test.label')
    parser.add_argument('--gaze360_img', type=str, default='/mnt/nvme1n1/lululemon/xjj/datasets/Gaze360/Image')
    parser.add_argument('--mpii_label', type=str, default="/mnt/nvme1n1/lululemon/xjj/datasets/MPIIGaze/Label")
    parser.add_argument('--mpii_img', type=str, default="/mnt/nvme1n1/lululemon/xjj/datasets/MPIIGaze/Image")
    parser.add_argument('--diap_label', type=str, default="/mnt/nvme1n1/lululemon/xjj/datasets/EyeDiap/Label")
    parser.add_argument('--diap_img', type=str, default="/mnt/nvme1n1/lululemon/xjj/datasets/EyeDiap/Image")
    
    return parser.parse_args()

def main():
    args = get_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. 确定 Checkpoint
    if args.checkpoint is None:
        try:
            args.checkpoint = find_latest_pretrain_checkpoint(args.model_dir)
        except Exception as e:
            print(f"Error finding checkpoint: {e}")
            return
    else:
        print(f"Using specified checkpoint: {args.checkpoint}")

    print(f"Loading Model from: {args.checkpoint}")
    
    # 2. 加载模型 (EstimationBranch)
    # 注意：这里我们只加载 EstimationBranch，不是 UnifiedGazeModel
    model = EstimationBranch().to(device)
    
    checkpoint = torch.load(args.checkpoint, map_location=device)
    # pretrain.py 保存的是 model_to_save.state_dict()
    # 如果当时用了 DataParallel，key 可能没有 module. 前缀 (取决于保存逻辑)，
    # 但为了鲁棒性，还是处理一下前缀
    state_dict = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint
    
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k.replace('module.', '')
        new_state_dict[name] = v
        
    try:
        model.load_state_dict(new_state_dict, strict=True)
        print("Model loaded successfully (Strict=True).")
    except Exception as e:
        print(f"Strict loading failed, trying strict=False. Error: {e}")
        model.load_state_dict(new_state_dict, strict=False)

    # 3. 准备数据预处理
    transform_face = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 4. 评估配置列表
    est_configs = [
        ("ETH-Gaze", args.eth_label, args.eth_img),
        ("Gaze360", args.gaze360_label, args.gaze360_img),
        ("MPII", args.mpii_label, args.mpii_img),
        ("Diap", args.diap_label, args.diap_img)
    ]

    # 5. 循环评估
    for name, label_path, img_path in est_configs:
        # 只有路径非空且存在时才评测
        if label_path and img_path and os.path.exists(label_path):
            ds = GazeEstimationDataset(label_path, img_path, transform=transform_face, is_train=False)
            
            # 使用默认 colllate_fn 即可，因为 GazeEstimationDataset 返回的是标准字典结构
            dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=4)
            
            eval_estimation_branch(model, dl, device, name)
        else:
            if label_path: 
                # 仅当路径被设置但文件不存在时打印跳过提示
                if not os.path.exists(label_path):
                    print(f"Skipping {name} (Label file not found: {label_path})")

if __name__ == "__main__":
    main()