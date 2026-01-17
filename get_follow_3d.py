import os
import json
import glob
import argparse
import torch
import numpy as np
import random
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.transforms import functional as TF
from tqdm import tqdm
import torch.nn.functional as F
import cv2
import PIL.Image as Image

# === Import Project Modules ===
from model import EstimationBranch

# ==========================================
# 1. 核心几何工具函数
# ==========================================

def angles_to_vectors(angles):
    """ (Yaw, Pitch) -> (x, y, z) """
    # angles: [B, 2]
    yaw = angles[:, 0]
    pitch = angles[:, 1]
    
    # Gaze360 定义: z负为看相机
    x = -torch.cos(pitch) * torch.sin(yaw)
    y = -torch.sin(pitch)
    z = -torch.cos(pitch) * torch.cos(yaw)
    
    return torch.stack([x, y, z], dim=1)

def vectors_to_angles(vectors):
    """ (x, y, z) -> (Yaw, Pitch) """
    vectors = F.normalize(vectors, dim=1)
    x, y, z = vectors[:, 0], vectors[:, 1], vectors[:, 2]
    
    pitch = torch.asin(-y.clamp(-1.0, 1.0))
    yaw = torch.atan2(-x, -z)
    return torch.stack([yaw, pitch], dim=1)

def geometric_projection_torch(pred_3d_vec, gt_2d_vec):
    """ 
    几何投影: 结合模型预测的Z轴幅度(取绝对值) 和 2D GT的方向
    """
    # 1. 强制 Z 为正 (假设GazeFollow场景都是看向远处的物体，即远离相机)
    # 这是解决"背影看相机"问题的关键Patch
    pseudo_z_magnitude = torch.abs(pred_3d_vec[:, 2:]) 
    
    # 2. 归一化 2D GT
    norm_gt = torch.norm(gt_2d_vec, dim=1, keepdim=True) + 1e-8
    unit_gt_2d = gt_2d_vec / norm_gt
    
    # 3. 重构 3D 向量 (利用几何约束: x^2+y^2+z^2=1)
    # 限制 z 最大为 0.99，防止 sqrt 出现负数
    pseudo_z_val = torch.clamp(pseudo_z_magnitude, 0, 0.99)
    k = torch.sqrt(1 - pseudo_z_val**2)
    
    pseudo_xy = unit_gt_2d * k
    
    pseudo_label = torch.cat([pseudo_xy, pseudo_z_val], dim=1)
    return pseudo_label 

def get_2d_gaze_vector_torch(eye_norm, gaze_norm):
    dx = gaze_norm[0] - eye_norm[0]
    dy = gaze_norm[1] - eye_norm[1]
    vec = torch.stack([dx, dy])
    norm = torch.norm(vec) + 1e-8
    return vec / norm

# ==========================================
# 2. 支持高级 TTA 的 Dataset
# ==========================================
class GazeFollowTTADataset(Dataset):
    def __init__(self, json_path, tta_times=8):
        self.tta_times = max(1, tta_times)
        
        # 基础处理 (用于原图和确定性翻转)
        self.base_resize = transforms.Resize((224, 224))
        self.to_tensor = transforms.ToTensor()
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        
        # 随机增强 (用于 TTA 样本)
        # 引入 ColorJitter 应对光照差异
        # 引入 RandomResizedCrop 模拟 BBox 噪声
        self.aug_transform = transforms.Compose([
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.RandomResizedCrop(size=(224, 224), scale=(0.85, 1.0), ratio=(0.9, 1.1))
        ])
        
        with open(json_path, 'r') as f:
            self.data = json.load(f)
            
    def __len__(self):
        return len(self.data)

    def process_image(self, img_pil, transform_type='base'):
        """ 辅助函数：应用变换并标准化 """
        if transform_type == 'base':
            img = self.base_resize(img_pil)
        elif transform_type == 'aug':
            img = self.aug_transform(img_pil)
            
        return self.normalize(self.to_tensor(img))

    def __getitem__(self, idx):
        item = self.data[idx]
        
        # --- 1. 读取并裁剪头部 ---
        img_path = item['img_path']
        full_image = cv2.imread(img_path)
        
        if full_image is None:
            full_image = np.zeros((224, 224, 3), dtype=np.uint8)
            
        full_image = cv2.cvtColor(full_image, cv2.COLOR_BGR2RGB)
        h, w, _ = full_image.shape
        
        bbox_n = item['head_bbox_norm']
        x_min = int(max(0, bbox_n['x_min'] * w))
        y_min = int(max(0, bbox_n['y_min'] * h))
        x_max = int(min(w, bbox_n['x_max'] * w))
        y_max = int(min(h, bbox_n['y_max'] * h))
        
        face_img = full_image[y_min:y_max, x_min:x_max]
        if face_img.size == 0:
            face_img = np.zeros((224, 224, 3), dtype=np.uint8)
            
        # 转为 PIL 以便进行 Transforms
        face_pil = Image.fromarray(face_img)
        
        # --- 2. 生成 TTA 图像列表 ---
        tta_imgs = []
        is_flipped_list = [] 
        
        # (1) 原图 [Deterministic]
        tta_imgs.append(self.process_image(face_pil, 'base'))
        is_flipped_list.append(False)
        
        if self.tta_times > 1:
            # (2) 水平翻转 [Deterministic]
            # 保证至少有一次翻转预测，消除左右不对称偏置
            flip_img = TF.hflip(face_pil)
            tta_imgs.append(self.process_image(flip_img, 'base'))
            is_flipped_list.append(True)
            
            # (3) 剩余 (N-2) 张：随机增强 + 随机翻转
            for _ in range(self.tta_times - 2):
                aug_img = face_pil
                
                # 50% 概率随机翻转
                do_flip = random.random() < 0.5
                if do_flip:
                    aug_img = TF.hflip(aug_img)
                
                # 应用颜色和裁剪抖动
                # 注意：RandomResizedCrop 会自动 Resize 到 224x224
                # 然后再 ToTensor 和 Normalize
                processed_aug = self.normalize(self.to_tensor(self.aug_transform(aug_img)))
                
                tta_imgs.append(processed_aug)
                is_flipped_list.append(do_flip)
                
        # Stack: [TTA, 3, 224, 224]
        faces_tensor = torch.stack(tta_imgs)
        is_flipped_tensor = torch.tensor(is_flipped_list, dtype=torch.bool)
        
        # --- 3. 准备 GT 数据 ---
        # (代码同之前，略去冗余注释)
        eye_n = item['eye_point_norm']
        eye_t = torch.tensor([eye_n['x'], eye_n['y']])
        
        MAX_GAZES = 20
        gaze_vecs = torch.zeros((MAX_GAZES, 2), dtype=torch.float32)
        gaze_indices = torch.zeros((MAX_GAZES), dtype=torch.long)
        mask = torch.zeros((MAX_GAZES), dtype=torch.bool)
        
        count = 0
        for i, g in enumerate(item['gazes']):
            if count >= MAX_GAZES: break
            if g.get('inout', 0) == 1 and 'gaze_point_norm' in g:
                gz = g['gaze_point_norm']
                gz_t = torch.tensor([gz['x'], gz['y']])
                vec_2d = get_2d_gaze_vector_torch(eye_t, gz_t)
                
                gaze_vecs[count] = vec_2d
                gaze_indices[count] = i
                mask[count] = True
                count += 1
                
        return {
            "faces": faces_tensor,       # [TTA, 3, H, W]
            "is_flipped": is_flipped_tensor, # [TTA]
            "gaze_vecs": gaze_vecs,      # [Max, 2]
            "gaze_indices": gaze_indices,
            "mask": mask,
            "json_idx": idx
        }

# ==========================================
# 3. 推理与聚合逻辑
# ==========================================

def process_single_json(json_path, model, device, args):
    print(f"\nProcessing: {json_path} (TTA={args.tta_times})")
    
    # Dataset 内部负责 TTA 增强
    dataset = GazeFollowTTADataset(json_path, tta_times=args.tta_times)
    if len(dataset) == 0: return

    # DataLoader 利用多进程并行处理数据增强
    dataloader = DataLoader(
        dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=args.num_workers, # 关键：利用 CPU 多核并行做 Augmentation
        pin_memory=True
    )
    
    results_map = {} 
    
    model.eval()
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Inference"):
            # batch['faces']: [B, TTA, 3, 224, 224]
            B, TTA, C, H, W = batch['faces'].shape
            
            # 1. 展平为大 Batch 进行推理
            # [B*TTA, 3, 224, 224]
            flat_inputs = batch['faces'].view(-1, C, H, W).to(device)
            
            # 2. 模型前向传播
            pred_angles_flat, _ = model(flat_inputs) # [B*TTA, 2] (Yaw, Pitch)
            
            # 3. 处理翻转 (Test-Time Augmentation Flip Correction)
            # 如果图片翻转了，预测出的 Yaw 是反的，需要由 Yaw' = -Yaw 还原
            is_flipped_flat = batch['is_flipped'].view(-1).to(device) # [B*TTA]
            pred_angles_flat[is_flipped_flat, 0] *= -1
            
            # 4. 投票聚合 (Vector Space Voting)
            # 转为 3D 向量进行平均，避免角度平均在 180 度处的奇异性
            pred_vecs_flat = angles_to_vectors(pred_angles_flat) # [B*TTA, 3]
            pred_vecs = pred_vecs_flat.view(B, TTA, 3)
            
            # 计算平均向量 [B, 3]
            avg_vecs = torch.mean(pred_vecs, dim=1)
            
            # 5. 生成伪标签 (Geometric Projection)
            # 此时 avg_vecs 包含了更鲁棒的头部朝向估计 (Z可能还是负的)
            # 我们通过 geometric_projection_torch 强制 Z 为正并结合 2D GT
            
            MAX_GAZES = batch['gaze_vecs'].shape[1]
            # [B, 3] -> [B, Max, 3]
            heads_expanded = avg_vecs.unsqueeze(1).expand(-1, MAX_GAZES, -1)
            gaze_vecs_gt = batch['gaze_vecs'].to(device) # [B, Max, 2]
            
            # 几何投影计算 (强制 Z 正)
            pseudo_3d = geometric_projection_torch(
                heads_expanded.reshape(B*MAX_GAZES, 3), 
                gaze_vecs_gt.reshape(B*MAX_GAZES, 2)
            )
            
            # 转回角度保存 (Yaw, Pitch)
            pseudo_angles = vectors_to_angles(pseudo_3d) # [B*Max, 2]
            
            # 6. 保存结果 (映射回 JSON 结构)
            mask_flat = batch['mask'].view(-1)
            valid_angles = pseudo_angles[mask_flat]
            
            # 展开索引以匹配 flatten 后的维度
            json_indices_expanded = batch['json_idx'].unsqueeze(1).expand(-1, MAX_GAZES).reshape(-1)
            gaze_indices_expanded = batch['gaze_indices'].reshape(-1)
            
            # 提取有效数据到 CPU
            valid_json_idx = json_indices_expanded[mask_flat].tolist()
            valid_gaze_idx = gaze_indices_expanded[mask_flat].tolist()
            valid_angles_cpu = valid_angles.cpu().tolist()
            
            for i, val in enumerate(valid_angles_cpu):
                j_idx = valid_json_idx[i]
                g_idx = valid_gaze_idx[i]
                
                if j_idx not in results_map:
                    results_map[j_idx] = {}
                results_map[j_idx][g_idx] = val

    # 更新 JSON 文件
    print("Writing results to JSON...")
    with open(json_path, 'r') as f:
        data_list = json.load(f)
        
    update_cnt = 0
    for j_idx, item in enumerate(data_list):
        if j_idx in results_map:
            for g_idx, val in results_map[j_idx].items():
                if g_idx < len(item['gazes']):
                    item['gazes'][g_idx]['pseudo_3d_gaze'] = val
                    update_cnt += 1
    
    if args.output_json_root:
        fname = os.path.basename(json_path)
        save_path = os.path.join(args.output_json_root, fname)
    else:
        save_path = json_path
        
    print(f"Updated {update_cnt} annotations. Saved to {save_path}")
    with open(save_path, 'w') as f:
        json.dump(data_list, f, indent=4)

def find_latest_checkpoint(root_dir, prefix="Pretrain"):
    """
    在 root_dir 下查找所有以 prefix 开头的子文件夹，
    并返回其中修改时间最近的 .pth 文件。
    """
    if not os.path.exists(root_dir):
        raise FileNotFoundError(f"Dir not found: {root_dir}")

    # 构造搜索路径: root_dir/Prefix*/ *.pth
    # 例如: /mnt/.../datasets/UnifiedGaze_2023_.../*.pth
    search_pattern = os.path.join(root_dir, f"{prefix}*", "*.pth")
    
    # 获取所有匹配的文件
    all_files = glob.glob(search_pattern)
    
    # 如果找不到，尝试递归搜索 (以防 .pth 在更深目录)
    if not all_files:
        search_pattern_recursive = os.path.join(root_dir, f"{prefix}*", "**", "*.pth")
        all_files = glob.glob(search_pattern_recursive, recursive=True)

    if not all_files:
        print(f"[Warning] No checkpoints found in {root_dir} starting with '{prefix}'")
        return None

    # 按修改时间排序，取最新的
    latest_file = max(all_files, key=os.path.getmtime)
    print(f"Found latest checkpoint: {latest_file}")
    return latest_file

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_pth', type=str, default=None)
    parser.add_argument('--save_dir', type=str, default='/mnt/nvme1n1/lululemon/fm_shijing/foudation_model')
    parser.add_argument('--input_gf_json_root', type=str, default='/mnt/nvme1n1/lululemon/xjj/result/information_fusion')
    parser.add_argument('--output_json_root', type=str, default=None)
    parser.add_argument('--device', type=str, default='cuda')
    
    # TTA 参数
    parser.add_argument('--tta_times', type=int, default=8, help="Total augmentations per image (Original + Flip + Randoms)")
    parser.add_argument('--batch_size', type=int, default=32, help="Original image batch size (Actual input will be BS * TTA)")
    parser.add_argument('--num_workers', type=int, default=16, help="Parallel workers for augmentation")
    
    args = parser.parse_args()
    
    ckpt_path = args.model_pth
    if ckpt_path is None:
        try: ckpt_path = find_latest_checkpoint(args.save_dir, "Pretrain")
        except: pass
    
    print(f"Loading Model: {ckpt_path}")
    device = torch.device(args.device)
    model = EstimationBranch().to(device)
    
    checkpoint = torch.load(ckpt_path, map_location=device)
    state_dict = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint
    new_state = {k.replace('module.', ''): v for k, v in state_dict.items()}
    model.load_state_dict(new_state, strict=True)
    
    json_files = glob.glob(os.path.join(args.input_gf_json_root, "*.json"))
    
    for jf in json_files:
        try:
            process_single_json(jf, model, device, args)
        except Exception as e:
            print(f"Error processing {jf}: {e}")
            import traceback; traceback.print_exc()

if __name__ == "__main__":
    main()