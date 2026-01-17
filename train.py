import os
# 设置 HuggingFace 镜像
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import argparse
import json
import time
from datetime import datetime
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from tqdm import tqdm
import numpy as np
from transformers import GPT2Tokenizer

# 引入你的模块
from dataset import GazeEstimationDataset, GazeFollowDataset, JointFusionDataset
from model import UnifiedGazeModel
from loss import GazeSystemLoss
import utils 

# ==========================================
# 0. 全局 Tokenizer (GPT-2)
# ==========================================
class UnifiedTokenizer:
    def __init__(self, max_len=25):
        self.max_len = max_len
        print("Loading GPT2 Tokenizer (Global)...")
        self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        self.tokenizer.pad_token = self.tokenizer.eos_token 

    def encode(self, text_list):
        clean_texts = [t if isinstance(t, str) else "" for t in text_list]
        encoded = self.tokenizer(
            clean_texts,
            padding='max_length',
            truncation=True,
            max_length=self.max_len,
            return_tensors='pt'
        )
        return encoded['input_ids']

tokenizer = UnifiedTokenizer(max_len=25)
pad_id = tokenizer.tokenizer.pad_token_id

# ==========================================
# 1. Collate Function (修复版)
# ==========================================
def gaze_collate_fn(batch):
    est_faces, est_3d_gts = [], []
    fol_faces, fol_scenes, fol_3d_gts = [], [], []
    fol_2d_gts, fol_bbox_gts, fol_2d_angles_gts = [], [], []
    fol_gaze_front_back, fol_gaze_dir_cls = [], []
    fol_bboxes, fol_eyes, fol_obs_exprs = [], [], []
    fol_gaze_exprs, fol_gaze_pts, fol_inouts = [], [], []
    task_types = []

    # 用于验证的原始数据列表，必须与 task_types 一一对应
    val_gt_points = []
    val_gt_inouts = []

    for item in batch:
        # --- 1. Est Data ---
        ge_list = item['gaze_estimation'] 
        for ge in ge_list:
            est_faces.append(ge['face_img'])
            est_3d_gts.append(ge['3D_ground_truth'])
            task_types.append('Estimation') 
            
            # Estimation 样本没有 gaze point GT，填 None 占位
            val_gt_points.append(None)
            val_gt_inouts.append(None)

        # --- 2. Follow Data ---
        gf = item['gaze_following']
        gf_est = gf['estimation_subtask']
        gf_fol = gf['following_subtask']

        fol_faces.append(gf_est['face_img'])
        fol_3d_gts.append(gf_est['3D_ground_truth'])
        fol_scenes.append(gf_fol['scene_img'])
        fol_bboxes.append(gf_fol['face_bbox_pixel'])
        fol_bbox_gts.append(gf_fol['face_bbox_gt_norm'])
        fol_eyes.append(gf_fol['eye_point_pixel'])
        fol_obs_exprs.append(gf_fol['observer_expression'])
        fol_2d_gts.append(gf_fol['2D_ground_truth'])
        fol_2d_angles_gts.append(gf_fol['2D_angles_ground_truth_rad'])

        fol_gaze_exprs.append(gf_fol['gaze_point_expressions'][0] if len(gf_fol['gaze_point_expressions']) > 0 else "")
        fol_gaze_pts.append(torch.tensor(gf_fol['gaze_points_norm'][0], dtype=torch.float32))
        fol_inouts.append(torch.tensor(gf_fol['in_outs'][0], dtype=torch.float32))
        
        task_types.append('Following')
        
        # Following 样本，添加真实的原始列表数据
        val_gt_points.append(gf_fol['gaze_points_norm']) # List[Tensor]
        val_gt_inouts.append(gf_fol['in_outs'])          # List[int]

    # Concatenate
    all_face_imgs = torch.stack(est_faces + fol_faces)
    all_scene_imgs = torch.stack(fol_scenes) if fol_scenes else None
    
    batch_est_len = len(est_faces)
    batch_fol_len = len(fol_faces)
    
    est_3d_gts_3d = torch.stack(est_3d_gts)
    est_3d_gts_padded = torch.cat([est_3d_gts_3d, torch.zeros(batch_fol_len, 2)])
    
    fol_3d_gts_3d = torch.stack(fol_3d_gts)
    fol_3d_gts_padded = torch.cat([torch.zeros(batch_est_len, 2), fol_3d_gts_3d])

    obs_tokens = tokenizer.encode(fol_obs_exprs)
    gaze_expr_tokens = tokenizer.encode(fol_gaze_exprs)

    batch_dict = {
        "face_img": all_face_imgs,
        "scene_img": all_scene_imgs,
        "observer_tokens": obs_tokens,
        "face_bbox": torch.stack(fol_bboxes) if fol_bboxes else None,
        "face_bbox_gt": torch.stack(fol_bbox_gts) if fol_bbox_gts else None,

        "eye_point_norm": torch.stack(fol_eyes) if fol_eyes else None,
        "3D_est_ground_truth": est_3d_gts_padded,
        "3D_fol_ground_truth": fol_3d_gts_padded,
        "gaze_points_norm": torch.stack(fol_gaze_pts) if fol_gaze_pts else None,
        "gaze_point_expressions_ids": gaze_expr_tokens,
        "in_outs": torch.stack(fol_inouts) if fol_inouts else None,
        "2D_fol_ground_truth": torch.stack(fol_2d_gts) if fol_2d_gts else None,
        "2D_angles_fol_ground_truth": torch.stack(fol_2d_angles_gts) if fol_2d_angles_gts else None,
        # 使用构建好的对齐列表
        "gt_points_list_val": val_gt_points,
        "gt_inouts_list_val": val_gt_inouts
    }
    return batch_dict, task_types

def get_args():
    parser = argparse.ArgumentParser(description="Unified Gaze Training (Gaze-LLE Config)")
    
    parser.add_argument('--gpus', type=str, default='2,3')
    parser.add_argument('--save_dir', type=str, default = "/mnt/nvme1n1/lululemon/fm_shijing/foudation_model")
    parser.add_argument('--resume', type=str, default=None, help="Path to a checkpoint file (e.g., .../best.pth)")
    parser.add_argument('--w_est_pretrain', action='store_true', help="Whether to use the estimation branch pretrained model")

    # 数据集路径
    parser.add_argument('--eth_label', type=str, default='/mnt/nvme1n1/lululemon/xjj/datasets/ETH-Gaze/Label/train_temp.label')
    parser.add_argument('--eth_img', type=str, default='/mnt/nvme1n1/lululemon/xjj/datasets/ETH-Gaze/Image')
    parser.add_argument('--gaze360_label', type=str, default='/mnt/nvme1n1/lululemon/xjj/datasets/Gaze360/Label/train.label')
    parser.add_argument('--gaze360_img', type=str, default='/mnt/nvme1n1/lululemon/xjj/datasets/Gaze360/Image')
    parser.add_argument('--gf_json', type=str, default='/mnt/nvme1n1/lululemon/xjj/result/information_fusion/merged_GazeFollow_train_EN_Qwen_Qwen3-VL-32B-Instruct.json')
    
    # [修改] 训练参数 (Gaze-LLE Paper Settings)
    parser.add_argument('--batch_size', type=int, default=60) # Paper uses 60
    parser.add_argument('--est_batch_multiplier', type=int, default=8, help="Batch size multiplier for Gaze Estimation")
    parser.add_argument('--epochs', type=int, default=15) # Paper uses 15 for GazeFollow
    
    # [修改] 学习率配置：支持差分学习率
    parser.add_argument('--lr', type=float, default=1e-3, help="Base learning rate (for Estimation branch if not specified otherwise)") 
    parser.add_argument('--lr_io', type=float, default=None, help="Learning rate for In/Out params (If None, uses --lr)")
    parser.add_argument('--lr_gaze_decoder', type=float, default=None, help="Learning rate for other Gaze Decoder params (If None, uses --lr)")
    
    parser.add_argument('--num_workers', type=int, default=16)
    parser.add_argument('--seed', type=int, default=42)
    
    parser.add_argument('--log_interval', type=int, default=50, help='Print log every N batches')

    # prompt种类
    parser.add_argument('--prompt_type', type=str, default=['bbox'], choices=['bbox','text', 'point'])
    
    # [修改] 损失权重 (Match Gaze-LLE: Only Heatmap & In/Out)
    parser.add_argument('--alpha_est_est', type=float, default=1.0) # 保留 Estimation 分支训练
    parser.add_argument('--alpha_est_fol', type=float, default=0.0)
    parser.add_argument('--alpha_text', type=float, default=0.0)
    parser.add_argument('--alpha_point', type=float, default=1.0) # Paper uses BCE, no large scalar mentioned
    parser.add_argument('--alpha_io', type=float, default=0.0)    # Paper uses lambda=1 for VAT
    parser.add_argument('--alpha_joint', type=float, default=0.0) # Single stream, no fusion alignment
    parser.add_argument('--alpha_head_mask', type=float, default=0.0) # Input prompt, not output
    parser.add_argument('--alpha_gaze_dir', type=float, default=0.0)  # No vector supervision
    parser.add_argument('--alpha_gaze_dir_cls', type=float, default=0.0)
    parser.add_argument('--alpha_gaze_front_back', type=float, default=0.0)

    args = parser.parse_args()
    return args

def generate_model_name(args):
    datasets = []
    if args.eth_label: datasets.append("ETH")
    if args.gaze360_label: datasets.append("G360")
    if args.gf_json: datasets.append("GF")
    
    ds_str = "-".join(datasets)
    loss_str = "GazeLLE_Config" 
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    name = f"UnifiedGaze_{ds_str}_{loss_str}_{timestamp}"
    return name

# ==========================================
# 3. 训练与验证流程
# ==========================================

def train_one_epoch(model, dataloader, criterion, optimizer, device, epoch, scaler, args, log_file_path):
    model.train()
    running_loss = 0.0
    loss_stats = {"l_est_est": 0, "l_est_fol": 0, "l_text": 0, "l_point": 0, "l_io": 0, "l_joint": 0, "l_head_mask": 0, "l_gaze_dir": 0, "l_gaze_dir_cls": 0, "l_gaze_front_back": 0}
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch+1} [Train]")
    
    for batch_idx, (batch_data, task_types) in enumerate(pbar):
        for k, v in batch_data.items():
            if isinstance(v, torch.Tensor):
                batch_data[k] = v.to(device)
        
        # 过滤输入
        if 'bbox' not in args.prompt_type:
            batch_data['observer_tokens'] = None
        if 'text' not in args.prompt_type:
            batch_data['face_bbox'] = None
        if 'point' not in args.prompt_type:
            batch_data['eye_point_norm'] = None

        optimizer.zero_grad()
        
        outputs = model(batch_data)
        loss, loss_dict_batch = criterion(outputs, batch_data, task_types)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        running_loss += loss.item()
        for k, v in loss_dict_batch.items():
            loss_stats[k] += v
            
        pbar.set_postfix({'loss': loss.item()})

        if (batch_idx + 1) % args.log_interval == 0:
            current_lr = optimizer.param_groups[0]['lr']
            
            log_step_str = (
                f"Epoch [{epoch+1}/{args.epochs}][{batch_idx+1}/{len(dataloader)}] "
                f"Loss: {loss.item():.4f} | LR: {current_lr:.2e} | "
            )
            if args.alpha_est_est != 0: log_step_str += f" | l_est_est: {loss_dict_batch['l_est_est']:.4f}"
            if args.alpha_est_fol != 0: log_step_str += f" | l_est_fol: {loss_dict_batch['l_est_fol']:.4f}"
            if args.alpha_text != 0: log_step_str += f" | l_text: {loss_dict_batch['l_text']:.4f}"
            if args.alpha_point != 0: log_step_str += f" | l_point: {loss_dict_batch['l_point']:.4f}"
            if args.alpha_io != 0: log_step_str += f" | l_io: {loss_dict_batch['l_io']:.4f}"
            if args.alpha_joint != 0: log_step_str += f" | l_joint: {loss_dict_batch['l_joint']:.4f}"
            if args.alpha_head_mask != 0: log_step_str += f" | l_head_mask: {loss_dict_batch['l_head_mask']:.4f}"
            if args.alpha_gaze_dir != 0: log_step_str += f" | l_gaze_dir: {loss_dict_batch['l_gaze_dir']:.4f}"
            if args.alpha_gaze_dir_cls != 0: log_step_str += f" | l_gaze_dir_cls: {loss_dict_batch['l_gaze_dir_cls']:.4f}"
            if args.alpha_gaze_front_back != 0: log_step_str += f" | l_gaze_front_back: {loss_dict_batch['l_gaze_front_back']:.4f}"
            with open(log_file_path, 'a') as f:
                f.write(log_step_str + '\n')

    avg_loss = running_loss / len(dataloader)
    for k in loss_stats:
        loss_stats[k] /= len(dataloader)
        
    return avg_loss, loss_stats

def evaluate_model(model, dataloader, device, args):
    model.eval()
    
    val_stats = {
        'est_ang_errs': [], # 3D Estimation Error
        'fol_l2_dist': [],  # Following Point Error
    }
    
    print("\n>>> Running Validation...")
    with torch.no_grad():
        for batch_data, task_types in tqdm(dataloader, desc="Validation"):
            for k, v in batch_data.items():
                if isinstance(v, torch.Tensor):
                    batch_data[k] = v.to(device)
            
            if 'bbox' not in args.prompt_type:
                batch_data['observer_tokens'] = None
            if 'text' not in args.prompt_type:
                batch_data['face_bbox'] = None
            if 'point' not in args.prompt_type:
                batch_data['eye_point_norm'] = None

            outputs = model(batch_data)
            
            # === 1. Eval Estimation Branch (3D Angle) ===
            if 'pred_angles' in outputs and '3D_est_ground_truth' in batch_data:
                pred_angs = outputs['pred_angles']
                gt_angs = batch_data['3D_est_ground_truth']
                
                is_est = torch.tensor([1 if t == 'Estimation' else 0 for t in task_types], device=device).bool()
                if is_est.any():
                    curr_pred = pred_angs[is_est].cpu().numpy()
                    curr_gt = gt_angs[is_est].cpu().numpy()
                    for i in range(len(curr_pred)):
                        err = utils.calculate_3d_angular_error(curr_pred[i], curr_gt[i])
                        val_stats['est_ang_errs'].append(err)

            # === 2. Eval Following Branch (L2 Distance) ===
            if 'pred_gaze_point' in outputs:
                pred_map = outputs['pred_gaze_point'] # [N_fol, 1, 56, 56]
                pred_xy = utils.get_heatmap_preds(pred_map) # [N_fol, 2]
                
                gt_points_list = batch_data['gt_points_list_val'] 
                gt_inouts_list = batch_data['gt_inouts_list_val']
                
                fol_pred_idx = 0
                
                # 遍历 task_types
                for idx, t_type in enumerate(task_types):
                    if t_type == 'Following':
                        gts = gt_points_list[idx] 
                        inouts = gt_inouts_list[idx]
                        
                        if fol_pred_idx < len(pred_xy):
                            cur_pred = pred_xy[fol_pred_idx]
                            fol_pred_idx += 1 
                            
                            if inouts is not None:
                                is_in_frame = (1 in inouts)
                                if is_in_frame:
                                    l2_avg, _ = utils.compute_l2_metrics(cur_pred, gts)
                                    val_stats['fol_l2_dist'].append(l2_avg)

    mean_est_err = np.mean(val_stats['est_ang_errs']) if val_stats['est_ang_errs'] else 999.0
    mean_fol_l2 = np.mean(val_stats['fol_l2_dist']) if val_stats['fol_l2_dist'] else 999.0
    
    print(f"Validation Results:")
    print(f"  > Estimation Mean Angle Error: {mean_est_err:.4f}")
    print(f"  > Following Mean L2 Dist:      {mean_fol_l2:.4f}")
    
    return mean_est_err, mean_fol_l2

def main():
    args = get_args()
    
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus

    model_base_name = generate_model_name(args)
    
    run_dir = os.path.join(args.save_dir, model_base_name)
    if not os.path.exists(run_dir):
        os.makedirs(run_dir)
        
    print(f"Start Training: {model_base_name}")
    print(f"Artifacts will be saved to: {run_dir}")

    if not args.resume:
        with open(os.path.join(run_dir, 'config.json'), 'w') as f:
            json.dump(vars(args), f, indent=4)

    log_file_path = os.path.join(run_dir, 'train.log')

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"[Config] GPUs: {args.gpus}")
    print(f"[Config] Batch Size: {args.batch_size}")
    
    if torch.cuda.is_available():
        print(f"Visible GPU Count: {torch.cuda.device_count()}")
    else:
        print("WARNING: Running on CPU!")

    # 2. Dataset
    print("Initializing Datasets...")
    transform_face = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    transform_scene = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((448, 448)), # Match Gaze-LLE Input Size
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    try:
        est_datasets = []
        if os.path.exists(args.eth_label):
            est_datasets.append(GazeEstimationDataset(args.eth_label, args.eth_img, transform=transform_face))
        if os.path.exists(args.gaze360_label):
            est_datasets.append(GazeEstimationDataset(args.gaze360_label, args.gaze360_img, transform=transform_face))
        
        follow_datasets = []
        if os.path.exists(args.gf_json):
            follow_datasets.append(GazeFollowDataset(args.gf_json, transform_face=transform_face, transform_scene=transform_scene, is_train=True))

        if not est_datasets or not follow_datasets:
            raise ValueError("Must have at least one Estimation dataset and one Following dataset.")

        joint_dataset = JointFusionDataset(est_datasets, follow_datasets, est_batch_multiplier=args.est_batch_multiplier)
        
        # 划分验证集
        val_size = int(len(joint_dataset) * 0.1)
        train_size = len(joint_dataset) - val_size
        train_dataset, val_dataset = random_split(joint_dataset, [train_size, val_size], generator=torch.Generator().manual_seed(args.seed))
        
        print(f"Dataset Split: Train={len(train_dataset)}, Val={len(val_dataset)}")

        train_loader = DataLoader(
            train_dataset, 
            batch_size=args.batch_size, 
            shuffle=True, 
            num_workers=args.num_workers,
            collate_fn=gaze_collate_fn,
            pin_memory=True,
            drop_last=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            collate_fn=gaze_collate_fn,
            pin_memory=True,
            drop_last=False
        )
        
    except Exception as e:
        print(f"Error loading datasets: {e}")
        import traceback
        traceback.print_exc()
        return

    # 3. Model
    print("Building Model...")
    model = UnifiedGazeModel(args.save_dir, args.w_est_pretrain).to(device)
    
    criterion = GazeSystemLoss(
        alpha_est_est=args.alpha_est_est,
        alpha_est_fol=args.alpha_est_fol,
        alpha_follow_text=args.alpha_text,
        alpha_follow_point=args.alpha_point,
        alpha_follow_io=args.alpha_io,
        alpha_joint=args.alpha_joint, 
        alpha_head_mask=args.alpha_head_mask,
        alpha_gaze_dir=args.alpha_gaze_dir,
        alpha_gaze_dir_cls=args.alpha_gaze_dir_cls,
        alpha_gaze_front_back=args.alpha_gaze_front_back,
        ignore_index=pad_id
    ).to(device)

    # [修改] 学习率分组 (Group LR)
    # 1. Estimation Branch (Use args.lr as default)
    est_params = list(model.estimation_branch.parameters())
    
    # 2. Following Branch - Split into IO and Other
    fol_io_params = []
    fol_other_params = []
    
    # 定义属于 In/Out 任务的模块名称
    io_module_names = ['inout_head', 'task_token'] 
    
    for name, param in model.following_branch.named_parameters():
        if not param.requires_grad:
            continue
            
        # 判断参数是否属于 IO 模块
        is_io = any(n in name for n in io_module_names)
        
        if is_io:
            fol_io_params.append(param)
        else:
            fol_other_params.append(param)

    # 确定学习率：如果命令行没指定，就回退到 args.lr
    lr_io = args.lr_io if args.lr_io is not None else args.lr
    lr_gaze_decoder = args.lr_gaze_decoder if args.lr_gaze_decoder is not None else args.lr
    
    print(f"LR Settings: Base={args.lr}, IO={lr_io}, GazeDecoder={lr_gaze_decoder}")

    param_groups = [
        {'params': est_params, 'lr': args.lr}, 
        {'params': fol_io_params, 'lr': lr_io},
        {'params': fol_other_params, 'lr': lr_gaze_decoder}
    ]
    
    optimizer = optim.Adam(param_groups, lr=args.lr)

    # [修改] 使用 Cosine Annealing (Gaze-LLE Paper)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    
    scaler = torch.cuda.amp.GradScaler()

    if torch.cuda.device_count() > 1:
        print(f"Using DataParallel on {torch.cuda.device_count()} GPUs")
        model = torch.nn.DataParallel(model)

    # 4. Resume
    start_epoch = 0
    best_fol_l2 = float('inf') 

    if args.resume:
        if os.path.isfile(args.resume):
            print(f"Loading checkpoint: {args.resume}")
            checkpoint = torch.load(args.resume, map_location=device)
            try:
                model.load_state_dict(checkpoint['model_state_dict'])
            except RuntimeError:
                from collections import OrderedDict
                new_state_dict = OrderedDict()
                state_dict = checkpoint['model_state_dict']
                for k, v in state_dict.items():
                    name = k.replace('module.', '') if 'module.' in k else k
                    new_state_dict[name] = v
                model.load_state_dict(new_state_dict, strict=False)

            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            if 'scheduler_state_dict' in checkpoint and checkpoint['scheduler_state_dict']:
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            
            start_epoch = checkpoint['epoch'] + 1
            best_fol_l2 = checkpoint.get('best_fol_l2', float('inf'))
            print(f"Resumed from Epoch {start_epoch}. Best Follow L2: {best_fol_l2:.4f}")
        else:
            print(f"No checkpoint found at '{args.resume}'.")

    # ==========================================
    # [新增] Sanity Check: 开始前先运行一次 Validation
    # ==========================================
    print("Running initial sanity check validation...")
    try:
        val_est_err, val_fol_l2 = evaluate_model(model, val_loader, device, args)
        with open(log_file_path, 'a') as f:
            f.write(f"Initial Check | Val Est Err: {val_est_err:.4f} | Val Fol L2: {val_fol_l2:.4f}\n")
        print("Sanity check passed.")
    except Exception as e:
        print(f"Sanity check failed: {e}")
        import traceback
        traceback.print_exc()
        return

    # 5. Loop
    for epoch in range(start_epoch, args.epochs):
        start_time = time.time()

        # Train
        train_loss, train_stats = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch, scaler, args, log_file_path
        )
        
        # Validation
        val_est_err, val_fol_l2 = evaluate_model(model, val_loader, device, args)
        
        # [修改] Scheduler Step (Cosine 不需要传入指标)
        scheduler.step()
        
        # 获取当前的学习率 (打印第一个组的，或者全部)
        lrs = [group['lr'] for group in optimizer.param_groups]
        current_lr_str = "/".join([f"{lr:.2e}" for lr in lrs])
        
        # Log
        log_str = f"=== Epoch [{epoch+1}/{args.epochs}] Finished | Train Loss: {train_loss:.4f} | Val Fol L2: {val_fol_l2:.4f} | LRs: {current_lr_str} ==="
        if args.alpha_est_est != 0: log_str += f" | l_est_est: {train_stats['l_est_est']:.4f}"
        if args.alpha_est_fol != 0: log_str += f" | l_est_fol: {train_stats['l_est_fol']:.4f}"
        if args.alpha_text != 0: log_str += f" | l_text: {train_stats['l_text']:.4f}"
        if args.alpha_point != 0: log_str += f" | l_point: {train_stats['l_point']:.4f}"
        if args.alpha_io != 0: log_str += f" | l_io: {train_stats['l_io']:.4f}"
        if args.alpha_joint != 0: log_str += f" | l_joint: {train_stats['l_joint']:.4f}"
        if args.alpha_head_mask != 0: log_str += f" | l_head_mask: {train_stats['l_head_mask']:.4f}"
        if args.alpha_gaze_dir != 0: log_str += f" | l_gaze_dir: {train_stats['l_gaze_dir']:.4f}"
        if args.alpha_gaze_dir_cls != 0: log_str += f" | l_gaze_dir_cls: {train_stats['l_gaze_dir_cls']:.4f}"
        if args.alpha_gaze_front_back != 0: log_str += f" | l_gaze_front_back: {train_stats['l_gaze_front_back']:.4f}"
        
        with open(log_file_path, 'a') as f:
            f.write(log_str + '\n')
            f.write(f"Val Est Err: {val_est_err:.4f} | Val Fol L2: {val_fol_l2:.4f}\n")

        # Save
        model_to_save = model.module if isinstance(model, torch.nn.DataParallel) else model
        checkpoint_dict = {
            'epoch': epoch,
            'model_state_dict': model_to_save.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'loss': train_loss,
            'best_fol_l2': best_fol_l2,
            'args': vars(args)
        }

        if val_fol_l2 < best_fol_l2:
            best_fol_l2 = val_fol_l2
            torch.save(checkpoint_dict, os.path.join(run_dir, "best.pth"))
            print(f"Saved Best Model (Follow L2: {best_fol_l2:.4f}) to {run_dir}/best.pth")

        torch.save(checkpoint_dict, os.path.join(run_dir, "latest.pth"))
        print(f"Time per epoch: {(time.time() - start_time):.2f}s\n")

    print("Training Finished.")

if __name__ == "__main__":
    main()