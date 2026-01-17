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
from transformers import GPT2Tokenizer
from collections import OrderedDict
import json

# === 引入项目模块 ===
from dataset import GazeEstimationDataset, GazeFollowDataset
from model import UnifiedGazeModel
import utils
import sys 

# ==========================================
# 0. 辅助功能: 自动寻找最新模型
# ==========================================
def find_latest_checkpoint(root_dir):
    """
    遍历 root_dir 下所有的 前缀为 UnifiedGaze_ 的 .pth 文件，返回修改时间最新的那个
    """
    print(f"Searching for latest checkpoint in: {root_dir} ...")
    
    if not os.path.exists(root_dir):
        raise FileNotFoundError(f"Model directory not found: {root_dir}")

    # 递归查找所有 .pth 文件
    search_pattern = os.path.join(root_dir, "UnifiedGaze_*", "*.pth")
    all_ckpts = glob.glob(search_pattern, recursive=True)
    
    if not all_ckpts:
        raise FileNotFoundError(f"No .pth files found in {root_dir}")

    # 按修改时间排序
    latest_ckpt = max(all_ckpts, key=os.path.getmtime)
    
    mod_time = time.ctime(os.path.getmtime(latest_ckpt))
    print(f"Found latest checkpoint: {latest_ckpt}")
    print(f"Last modified: {mod_time}")
    
    return latest_ckpt

# ==========================================
# 1. 文本生成与 Tokenizer
# ==========================================
class UnifiedTokenizer:
    def __init__(self, max_len=25):
        self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        self.tokenizer.pad_token = self.tokenizer.eos_token 
        self.max_len = max_len

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

    def decode(self, token_ids):
        text = self.tokenizer.decode(token_ids, skip_special_tokens=True)
        return text.strip()

# ==========================================
# 2. Collate Function
# ==========================================
def test_collate_fn(batch):
    # 判断任务类型
    if 'following_subtask' in batch[0]:
        task_type = 'following'
    else:
        task_type = 'estimation'

    if task_type == 'estimation':
        imgs = torch.stack([item['face_img'] for item in batch])
        gts = torch.stack([item['3D_ground_truth'] for item in batch])
        dict_data = {'task': 'estimation', 'face_img': imgs, '3D_est_ground_truth': gts}
        return dict_data

    elif task_type == 'following':
        fol_faces, fol_scenes, fol_bboxes, fol_eyes, fol_3d_gts = [], [], [], [], []
        fol_obs_texts, fol_gt_points, fol_gt_inouts, fol_gt_exprs = [], [], [], []
        fol_widths, fol_heights = [], []

        for item in batch:
            ge = item['estimation_subtask']
            gf = item['following_subtask']

            fol_faces.append(ge['face_img'])
            fol_3d_gts.append(ge['3D_ground_truth'])
            
            fol_scenes.append(gf['scene_img'])
            fol_bboxes.append(gf['face_bbox_pixel'])
            fol_eyes.append(gf['eye_point_pixel']) 
            fol_widths.append(gf['width'])
            fol_heights.append(gf['height'])
            # Metadata
            fol_obs_texts.append(gf.get('observer_expression', ""))
            fol_gt_points.append(gf['gaze_points_norm'])      
            fol_gt_inouts.append(gf['in_outs'])               
            fol_gt_exprs.append(gf['gaze_point_expressions']) 

        dict_data = {
            'task': 'following',
            'face_img': torch.stack(fol_faces),
            '3D_fol_ground_truth': torch.stack(fol_3d_gts),
            'scene_img': torch.stack(fol_scenes),
            'face_bbox': torch.stack(fol_bboxes),
            'eye_point_norm': torch.stack(fol_eyes),
            'obs_exprs_text': fol_obs_texts,
            'gt_points_list': fol_gt_points,
            'gt_inouts_list': fol_gt_inouts,
            'gt_exprs_list': fol_gt_exprs,
            'width': fol_widths,
            'height': fol_heights
        }
        return dict_data

# ==========================================
# 3. Evaluation Loops
# ==========================================

def eval_estimation(model, dataloader, device, dataset_name):
    model.eval()
    errors = []
    print(f"\n>>> Evaluating {dataset_name} (3D Estimation)...")
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc=dataset_name):
            imgs = batch['face_img'].to(device)
            gts = batch['3D_est_ground_truth'].cpu().numpy()

            # Est Branch Only
            preds_angle, _ = model.estimation_branch(imgs)
            preds_angle = preds_angle.cpu().numpy()
            
            for i in range(len(imgs)):
                # 调用 utils 计算 3D 角度误差
                err = utils.calculate_3d_angular_error(preds_angle[i], gts[i])
                errors.append(err)
            
    mean_err = np.mean(errors)
    print(f"[{dataset_name}] Mean 3D Angular Error: {mean_err:.4f}°")
    return mean_err

def eval_following(model, dataloader, device, tokenizer, dataset_name, args):
    model.eval()
    stats = {
        'est_fol_errs': [],
        'l2_avg': [], 'l2_best': [],
        'bleu_avg': [], 'bleu_best': [],
        'io_preds': [], 'io_gts': [], 
        'auc': []
    }
    
    print(f"\n>>> Evaluating {dataset_name} (Following)...")
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc=dataset_name):
            # 1. Inputs
            face_img = batch['face_img'].to(device)
            scene_img = batch['scene_img'].to(device)
            
            # 原始数据
            face_bbox = batch['face_bbox'].to(device)
            eye_norm = batch['eye_point_norm'].to(device)
            obs_texts = batch['obs_exprs_text']
            obs_ids = tokenizer.encode(obs_texts).to(device)
            w, h = batch['width'], batch['height']

            # [新增] 根据 prompt_type 过滤输入 (与 Train 保持一致)
            if 'bbox' not in batch:
                face_bbox = None
            if 'point' not in batch:
                eye_norm = None
            if 'text' not in batch:
                obs_ids = None

            # 2. Forward
            # A. 2D Estimation (ResNet)
            pred_angles, _ = model.estimation_branch(face_img)

            # B. Following (SAM + GPT) —— 直接走 FollowingBranch.forward
            # 这样当 model.py 内部结构变化时，这里不需要同步重写一整套 pipeline
            fol_out = model.following_branch(
                scene_img=scene_img,
                obs_text_ids=obs_ids,
                face_bbox=face_bbox,
                eye_point=eye_norm,
                target_text_ids=None
            )

            # Heatmap -> (x, y)
            pred_xy_np = utils.get_heatmap_preds(fol_out['pred_gaze_point'])
            pred_map_batch = fol_out['pred_gaze_point']

            # In/Out
            pred_io_logits = fol_out['pred_inout'].flatten()
            pred_io_probs = torch.sigmoid(pred_io_logits).cpu().numpy()

            pred_token_ids = torch.argmax(fol_out['text_logits'], dim=-1).cpu()

            # 3. Calculate Metrics
            batch_size = face_img.size(0)
            for i in range(batch_size):
                # --- 2D Angle Error ---
                gt_3d_vec = batch['3D_fol_ground_truth'][i].numpy()
                err_fol = utils.calculate_3d_angular_error(pred_angles[i], gt_3d_vec)
                stats['est_fol_errs'].append(err_fol)

                # --- Point L2 Distance ---
                gt_points = batch['gt_points_list'][i]
                gt_inouts = batch['gt_inouts_list'][i]
                if isinstance(gt_inouts, torch.Tensor): gt_inouts = gt_inouts.tolist()
                
                is_in_frame = (1 in gt_inouts)
                
                if is_in_frame:
                    l2_avg, l2_min = utils.compute_l2_metrics(pred_xy_np[i], gt_points)
                    stats['l2_avg'].append(l2_avg)
                    stats['l2_best'].append(l2_min)
                
                    # [新增] AUC Calculation
                    # 准备 GT 列表
                    gt_x_list = [p[0] for p in gt_points]
                    gt_y_list = [p[1] for p in gt_points]
                    
                    # 取出单张热图 [56, 56]
                    heatmap_i = pred_map_batch[i].squeeze() 
                    
                    auc_score = utils.gazefollow_auc(heatmap_i, gt_x_list, gt_y_list, h[i], w[i])
                    if auc_score is not None:
                        stats['auc'].append(auc_score)

                # --- In/Out ---
                stats['io_preds'].append(pred_io_probs[i])
                stats['io_gts'].append(1 if is_in_frame else 0)

                # --- Text BLEU ---
                pred_str = tokenizer.decode(pred_token_ids[i])
                gt_strs = batch['gt_exprs_list'][i]
                gt_strs = [s for s in gt_strs if s and len(s) > 0]
                if gt_strs and len(gt_strs) > 0:
                    b_avg, b_max = utils.compute_bleu_metrics(pred_str, gt_strs)
                    stats['bleu_avg'].append(b_avg)
                    stats['bleu_best'].append(b_max)

    # --- Print Summary ---
    print("\n" + "=" * 60)
    print(f" >>> Results for: {dataset_name} <<<")
    if stats['est_fol_errs']:    
        print(f" Estimation (3D Follow)   | Mean Angular Err: {np.mean(stats['est_fol_errs']):.2f}°")
    
    if stats['l2_avg']:
        l2 = np.mean(stats['l2_avg'])
        print(f" Following (Point) | Avg L2 Dist:      {l2:.4f} (Norm)")
        print(f" Following (Point) | Best L2 Dist:     {np.mean(stats['l2_best']):.4f} (Norm)")

    if stats['auc']:
        print(f" Following (Point) | AUC:              {np.mean(stats['auc']):.4f}")

    if stats['io_preds']:
        f1, ap = utils.compute_io_metrics(stats['io_preds'], stats['io_gts'])
        print(f" Following (I/O)   | F1 Score:         {f1:.4f}")
        print(f" Following (I/O)   | AP Score:         {ap:.4f}")
        
    if stats['bleu_avg']:
        print(f" Following (Text)  | BLEU-4 (Avg):     {np.mean(stats['bleu_avg']):.4f}")
    print("=" * 60 + "\n")

# ==========================================
# 4. Main
# ==========================================
def get_args():
    parser = argparse.ArgumentParser(description="Unified Gaze Testing")
    
    parser.add_argument('--gpus', type=str, default='2,3')
    
    # 自动 Checkpoint 逻辑
    parser.add_argument('--checkpoint', type=str, default=None, help="Specific .pth path. If None, auto search.")
    parser.add_argument('--model_dir', type=str, default="/mnt/nvme1n1/lululemon/fm_shijing/foudation_model", help="Dir to search latest checkpoint")
    
    parser.add_argument('--batch_size', type=int, default=64)
    
    # Estimation Datasets
    parser.add_argument('--eth_label', type=str, default='/mnt/nvme1n1/lululemon/xjj/datasets/ETH-Gaze/Label/test_temp.label')
    parser.add_argument('--eth_img', type=str, default='/mnt/nvme1n1/lululemon/xjj/datasets/ETH-Gaze/Image')
    parser.add_argument('--gaze360_label', type=str, default='/mnt/nvme1n1/lululemon/xjj/datasets/Gaze360/Label/test.label')
    parser.add_argument('--gaze360_img', type=str, default='/mnt/nvme1n1/lululemon/xjj/datasets/Gaze360/Image')
    parser.add_argument('--mpii_label', type=str, default="/mnt/nvme1n1/lululemon/xjj/datasets/MPIIGaze/Label")
    parser.add_argument('--mpii_img', type=str, default="/mnt/nvme1n1/lululemon/xjj/datasets/MPIIGaze/Image")
    parser.add_argument('--diap_label', type=str, default="/mnt/nvme1n1/lululemon/xjj/datasets/EyeDiap/Label")
    parser.add_argument('--diap_img', type=str, default="/mnt/nvme1n1/lululemon/xjj/datasets/EyeDiap/Image")
    
    # Following Datasets (支持多个)
    parser.add_argument('--gf_json', type=str, nargs='+', default=["/mnt/nvme1n1/lululemon/xjj/result/information_fusion/merged_GazeFollow_test_EN_Qwen_Qwen3-VL-32B-Instruct.json", "/mnt/nvme1n1/lululemon/xjj/result/information_fusion/merged_VAT_test_EN_Qwen_Qwen3-VL-32B-Instruct.json"], help="List of json paths")
    parser.add_argument('--prompt_type', type=str, default=['bbox'], choices=['bbox', 'text', 'point'])

    args = parser.parse_args()
    return args

def main():
    args = get_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. 确定 Checkpoint
    if args.checkpoint is None:
        try:
            args.checkpoint = find_latest_checkpoint(args.model_dir)
        except Exception as e:
            print(f"Error finding checkpoint: {e}")
            return
    else:
        print(f"Using specified checkpoint: {args.checkpoint}")

    print(f"Loading Model from: {args.checkpoint}")

    # 获取模型所在的文件夹路径
    model_dir = os.path.dirname(args.checkpoint)
    log_path = os.path.join(model_dir, "test.txt")

    # 定义 Logger 类，同时写文件和控制台
    class Logger(object):
        def __init__(self, filename):
            self.terminal = sys.stdout
            self.log = open(filename, "a", encoding='utf-8') 
        def write(self, message):
            self.terminal.write(message)
            self.log.write(message)
            self.log.flush() 
        def flush(self):
            self.terminal.flush()
            self.log.flush()

    # 重定向 stdout
    sys.stdout = Logger(log_path)
    
    print(f"\n" + "="*40)
    print(f"Test Log saved to: {log_path}")
    print(f"Time: {time.ctime()}")
    print("="*40 + "\n")
    # ==========================================

    print(f"Loading Model from: {args.checkpoint}")
    
    # 2. 加载模型
    model = UnifiedGazeModel(args.model_dir).to(device)
    
    # [修改] 增加 weights_only=False 
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    state_dict = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint
    
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k.replace('module.', '')
        new_state_dict[name] = v
    model.load_state_dict(new_state_dict, strict=False)
    
    tokenizer = UnifiedTokenizer()

    # Transforms (Val/Test)
    transform_face = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    transform_scene = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((448, 448)), 
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # === 3. 评估 Estimation 数据集 (分开输出) ===
    est_configs = [
        ("ETH-Gaze", args.eth_label, args.eth_img),
        ("Gaze360", args.gaze360_label, args.gaze360_img),
        ("MPII", args.mpii_label, args.mpii_img),
        ("Diap", args.diap_label, args.diap_img)
    ]

    for name, label_path, img_path in est_configs:
        if label_path and img_path:
            ds = GazeEstimationDataset(label_path, img_path, transform=transform_face, is_train=False)
            dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=4, collate_fn=test_collate_fn)
            eval_estimation(model, dl, device, name)

    # === 4. 评估 Following 数据集 (GF & VAT 分开输出) ===
    if args.gf_json:
        for json_path in args.gf_json:
            # 智能命名逻辑
            if "VAT" in json_path:
                dataset_name = "VideoAttentionTarget (VAT)"
            elif "GazeFollow" in json_path:
                dataset_name = "GazeFollow"
            else:
                dataset_name = os.path.basename(json_path)

            ds = GazeFollowDataset(json_path, transform_face=transform_face, transform_scene=transform_scene, is_train=False)
            dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=4, collate_fn=test_collate_fn)
            
            # 传入 args
            eval_following(model, dl, device, tokenizer, dataset_name, args)

if __name__ == "__main__":
    main()