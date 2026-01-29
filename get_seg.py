import json
import cv2
import os
import torch
import numpy as np
import glob
import shutil
import math
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Resize, ToTensor, Normalize, Compose
from segment_anything import sam_model_registry
from segment_anything.utils.transforms import ResizeLongestSide
import torch.multiprocessing as mp
from tqdm import tqdm

# ================= 配置区域 =================
# 权重与路径
CHECKPOINT_DIR = "/mnt/nvme1n1/lululemon/xjj/checkpoints" 
MODEL_TYPE = "vit_h" 
MODEL_FILENAME = "sam_vit_h_4b8939.pth"
SAM_CHECKPOINT = os.path.join(CHECKPOINT_DIR, MODEL_FILENAME)

JSON_ROOT_DIR = '/mnt/nvme1n1/lululemon/xjj/result/information_fusion'
MASK_ROOT_DIR = '/mnt/nvme1n1/lululemon/xjj/datasets/Gaze_Masks'

# 【新增】可视化检查结果保存路径
CHECK_MASK_DIR = '/mnt/nvme1n1/lululemon/xjj/datasets/check_Gaze_masks'
VIS_INTERVAL = 50  # 每隔多少张图片保存一张可视化结果用于检查

TARGET_GPUS = [2, 3] 

# 【核心参数】
BATCH_SIZE = 8        # H100 上 vit_h 可以尝试 32-48
NUM_WORKERS = 16      # CPU 预处理线程数
SAVE_INTERVAL = 1000 
# ===========================================

# 自动下载权重函数
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
from huggingface_hub import hf_hub_download

def check_and_download_weights():
    if os.path.exists(SAM_CHECKPOINT): return SAM_CHECKPOINT
    print(f"🚀 Downloading weights...")
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    try:
        hf_hub_download(repo_id="ybelkada/segment-anything", filename="checkpoints/sam_vit_h_4b8939.pth", local_dir=CHECKPOINT_DIR, local_dir_use_symlinks=False)
        shutil.move(os.path.join(CHECKPOINT_DIR, "checkpoints/sam_vit_h_4b8939.pth"), SAM_CHECKPOINT)
        return SAM_CHECKPOINT
    except Exception as e: raise RuntimeError(e)

# 路径生成函数
def get_mask_save_path(mask_root_dir, img_path, gaze_idx):
    path_parts = img_path.split(os.sep)
    if 'images' in path_parts:
        idx = len(path_parts) - 1 - path_parts[::-1].index('images')
        relative_path = os.sep.join(path_parts[idx+1:])
    else:
        relative_path = os.path.join(path_parts[-2], path_parts[-1])
    
    file_dir = os.path.dirname(relative_path)
    file_name = os.path.basename(relative_path)
    name_no_ext = os.path.splitext(file_name)[0]
    mask_filename = f"{name_no_ext}_mask_{gaze_idx}.png"
    save_dir = os.path.join(mask_root_dir, file_dir)
    save_path = os.path.join(save_dir, mask_filename)
    return save_dir, save_path

# =========================================================
# 核心修改：Dataset 读取 Eye Point 作为负样本 + 【新增 bbox 传递】
# =========================================================
class SAMInferenceDataset(Dataset):
    def __init__(self, items, mask_root_dir):
        self.items = items
        self.mask_root_dir = mask_root_dir
        self.transform = ResizeLongestSide(1024) 
        self.pixel_mean = torch.tensor([123.675, 116.28, 103.53]).view(-1, 1, 1)
        self.pixel_std = torch.tensor([58.395, 57.12, 57.375]).view(-1, 1, 1)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        # item 结构: (json_idx, gaze_idx, item_dict, gaze_dict)
        original_idx, gaze_idx, item_data, gaze_data = self.items[idx]
        
        img_path = item_data['img_path']
        
        image = cv2.imread(img_path)
        if image is None: return None
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        original_size = image.shape[:2] # h, w
        h, w = original_size

        # 1. SAM 图像预处理
        input_image = self.transform.apply_image(image)
        input_image_torch = torch.as_tensor(input_image).permute(2, 0, 1).float() 
        
        # 2. 坐标提取 (Gaze & Eye)
        # Point 1: Gaze Point (正样本)
        g_norm = gaze_data['gaze_point_norm']
        gx, gy = g_norm['x'] * w, g_norm['y'] * h
        
        # Point 2: Eye Point (负样本，用于排除人脸/身体)
        e_norm = item_data['eye_point_norm']
        ex, ey = e_norm['x'] * w, e_norm['y'] * h
        
        # 组合坐标点: shape (2, 2) -> [[gx, gy], [ex, ey]]
        points_np = np.array([[gx, gy], [ex, ey]])
        
        # 变换坐标以匹配 resize 后的图片
        points_trans = self.transform.apply_coords(points_np, original_size)
        points_torch = torch.as_tensor(points_trans).float()

        # 【新增】获取 BBox 用于可视化 (Normalized [xmin, ymin, xmax, ymax])
        # 假设 json 中 bbox 字段名为 'bbox'
        bbox_norm = torch.tensor(gaze_data.get('bbox', [0, 0, 0, 0]), dtype=torch.float32)
        
        # 【新增】传递原始 Gaze 坐标用于可视化 (Normalized [x, y])
        gaze_center_norm = torch.tensor([g_norm['x'], g_norm['y']], dtype=torch.float32)

        return {
            "image": input_image_torch,
            "original_size": torch.tensor(original_size), 
            "point_coords": points_torch,                 # (2, 2)
            "img_path": img_path,
            "original_idx": original_idx,
            "gaze_idx": gaze_idx,
            "bbox_norm": bbox_norm,           # For Visualization
            "gaze_center_norm": gaze_center_norm # For Visualization
        }

def collate_fn(batch):
    batch = [x for x in batch if x is not None]
    if len(batch) == 0: return None
    
    # Pad Images
    images = [x["image"] for x in batch]
    max_h, max_w = 1024, 1024
    padded_images = []
    
    for img in images:
        h, w = img.shape[-2:]
        padh = max_h - h
        padw = max_w - w
        padded_img = F.pad(img, (0, padw, 0, padh))
        padded_images.append(padded_img)
        
    batch_images = torch.stack(padded_images)
    
    return {
        "image": batch_images, 
        "original_size": torch.stack([x["original_size"] for x in batch]),
        "point_coords": torch.stack([x["point_coords"] for x in batch]), # (B, 2, 2)
        "img_path": [x["img_path"] for x in batch],
        "original_idx": [x["original_idx"] for x in batch],
        "gaze_idx": [x["gaze_idx"] for x in batch],
        "bbox_norm": torch.stack([x["bbox_norm"] for x in batch]),
        "gaze_center_norm": torch.stack([x["gaze_center_norm"] for x in batch])
    }

# =========================================================
# 核心工作进程
# =========================================================
def worker_process(gpu_id, json_files, checkpoint_path):
    torch.cuda.set_device(gpu_id)
    device = torch.device(f"cuda:{gpu_id}")
    
    print(f"[GPU {gpu_id}] Loading SAM Model...")
    sam = sam_model_registry[MODEL_TYPE](checkpoint=checkpoint_path)
    sam.to(device)
    sam.eval() 

    pixel_mean = torch.tensor([123.675, 116.28, 103.53], device=device).view(-1, 1, 1)
    pixel_std = torch.tensor([58.395, 57.12, 57.375], device=device).view(-1, 1, 1)

    # 【新增】确保可视化目录存在
    os.makedirs(CHECK_MASK_DIR, exist_ok=True)

    for json_path in json_files:
        print(f"[GPU {gpu_id}] Processing JSON: {os.path.basename(json_path)}")
        
        output_json_path = json_path.replace(".json", "_with_SEG.json")
        dataset_name_clean = os.path.basename(json_path).replace('.json', '')
        current_mask_root = os.path.join(MASK_ROOT_DIR, dataset_name_clean)

        with open(json_path, 'r') as f:
            data = json.load(f)
            
        # 1. 过滤任务: 只处理 inout=1 的
        tasks = []
        for i, item in enumerate(data):
            if 'gazes' in item:
                for j, gaze in enumerate(item['gazes']):
                    # 检查是否已存在
                    if 'seg_mask_path' in gaze and gaze['seg_mask_path'] and os.path.exists(gaze['seg_mask_path']):
                        continue
                    
                    # 【关键点 1】只添加 inout=1 且包含必要坐标的样本
                    if gaze.get('inout', 0) == 1 and 'gaze_point_norm' in gaze and 'eye_point_norm' in item:
                        tasks.append((i, j, item, gaze))

        if not tasks:
            print(f"[GPU {gpu_id}] No new tasks in {os.path.basename(json_path)}")
            continue

        dataset = SAMInferenceDataset(tasks, current_mask_root)
        loader = DataLoader(
            dataset, 
            batch_size=BATCH_SIZE, 
            shuffle=False, 
            num_workers=NUM_WORKERS, 
            collate_fn=collate_fn,
            pin_memory=True,
            drop_last=False
        )

        print(f"[GPU {gpu_id}] Batch inference on {len(tasks)} items...")
        processed_count = 0
        
        with torch.no_grad():
            for batch in tqdm(loader, desc=f"GPU {gpu_id}", position=gpu_id, leave=False):
                if batch is None: continue

                input_images = (batch["image"].to(device) - pixel_mean) / pixel_std 
                point_coords = batch["point_coords"].to(device) # (B, 2, 2)
                original_sizes = batch["original_size"]
                
                # 【关键点 2】构建 Labels
                # 形状: (B, 2)
                # 每一行都是 [1, 0] -> 1代表Gaze正样本, 0代表Eye负样本
                B = point_coords.shape[0]
                point_labels = torch.tensor([[1, 0]] * B, dtype=torch.int, device=device)

                # --- Batch Encoder ---
                image_embeddings = sam.image_encoder(input_images)

                # --- Batch Prompt Encoder ---
                sparse_embeddings, dense_embeddings = sam.prompt_encoder(
                    points=(point_coords, point_labels),
                    boxes=None,
                    masks=None,
                )
                
                # --- Loop Decoder ---
                low_res_masks_list = []
                
                # 由于 mask_decoder 不支持 batch prompt (在某些版本)，这里还是循环处理比较稳
                # 或者如果你确定显存够，可以直接 batch 喂进去，但这里为了改动最小，保持循环结构
                for i in range(len(image_embeddings)):
                    img_emb = image_embeddings[i:i+1] 
                    sparse_emb = sparse_embeddings[i:i+1]
                    dense_emb = dense_embeddings[i:i+1]
                    
                    # [修正 1] 开启多 Mask 输出
                    low_res_masks_i, iou_preds_i = sam.mask_decoder(
                        image_embeddings=img_emb,
                        image_pe=sam.prompt_encoder.get_dense_pe(),
                        sparse_prompt_embeddings=sparse_emb,
                        dense_prompt_embeddings=dense_emb,
                        multimask_output=True,  # <--- 改为 True
                    )
                    
                    # [修正 2] 挑选分数最高的 Mask
                    # iou_preds_i shape: (1, 3)
                    best_idx = torch.argmax(iou_preds_i, dim=1) # 找到分最高的索引
                    
                    # low_res_masks_i shape: (1, 3, 256, 256)
                    # 取出最好的那个 mask
                    best_mask = low_res_masks_i[0, best_idx] # (1, 256, 256)
                    
                    low_res_masks_list.append(best_mask)
                
                # 堆叠回 Batch
                low_res_masks = torch.cat(low_res_masks_list, dim=0) # (B, 1, 256, 256)

                # --- Post Processing (保持不变) ---
                masks = F.interpolate(
                    low_res_masks,
                    size=(1024, 1024),
                    mode="bilinear",
                    align_corners=False,
                )
                
                masks_np = masks.squeeze(1).cpu().numpy() # (B, 1024, 1024)
                
                # 获取可视化所需的 BBox 和 Gaze 点 (CPU)
                bbox_norms = batch["bbox_norm"].numpy()
                gaze_centers = batch["gaze_center_norm"].numpy()

                for k in range(len(masks_np)):
                    orig_h, orig_w = original_sizes[k]
                    scale = 1024.0 / max(orig_h, orig_w)
                    new_h, new_w = int(orig_h * scale + 0.5), int(orig_w * scale + 0.5)
                    mask_valid = masks_np[k, :new_h, :new_w]
                    mask_final = cv2.resize(mask_valid, (orig_w.item(), orig_h.item()), interpolation=cv2.INTER_NEAREST)
                    mask_binary = (mask_final > 0.0).astype(np.uint8) * 255
                    
                    o_idx = batch['original_idx'][k]
                    g_idx = batch['gaze_idx'][k]
                    img_path = batch['img_path'][k]
                    
                    save_dir, save_path = get_mask_save_path(current_mask_root, img_path, g_idx)
                    os.makedirs(save_dir, exist_ok=True)
                    cv2.imwrite(save_path, mask_binary)
                    
                    data[o_idx]['gazes'][g_idx]['seg_mask_path'] = save_path
                    processed_count += 1

                    # =========================================================
                    # 【新增】可视化检查逻辑
                    # =========================================================
                    if processed_count % VIS_INTERVAL == 0:
                        try:
                            # 重新读取原图 (最安全，不经过 resize/padding 的逆变换)
                            check_img = cv2.imread(img_path)
                            if check_img is not None:
                                h_curr, w_curr = check_img.shape[:2]
                                
                                # 1. 绘制 Observer BBox (蓝色)
                                bx1, by1, bx2, by2 = bbox_norms[k]
                                x1, y1 = int(bx1 * w_curr), int(by1 * h_curr)
                                x2, y2 = int(bx2 * w_curr), int(by2 * h_curr)
                                cv2.rectangle(check_img, (x1, y1), (x2, y2), (255, 0, 0), 3) # Blue
                                cv2.putText(check_img, 'Observer', (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

                                # 2. 绘制 Gaze Point (红色实心圆)
                                gx, gy = gaze_centers[k]
                                cx, cy = int(gx * w_curr), int(gy * h_curr)
                                cv2.circle(check_img, (cx, cy), 10, (0, 0, 255), -1) # Red filled
                                cv2.putText(check_img, 'Gaze', (cx+15, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

                                # 3. 叠加 Mask (绿色透明)
                                # mask_binary 是单通道，需要转 3 通道
                                color_mask = np.zeros_like(check_img)
                                color_mask[:, :, 1] = mask_binary # Green Channel
                                
                                # 仅在 Mask 区域进行加权叠加
                                mask_bool = mask_binary > 0
                                check_img[mask_bool] = cv2.addWeighted(check_img[mask_bool], 0.6, color_mask[mask_bool], 0.4, 0)

                                # 4. 保存检查图
                                file_name = os.path.basename(img_path)
                                check_save_name = f"CHECK_{dataset_name_clean}_{o_idx}_{g_idx}_{file_name}"
                                cv2.imwrite(os.path.join(CHECK_MASK_DIR, check_save_name), check_img)
                        except Exception as e:
                            print(f"[Warning] Vis failed: {e}")
                    # =========================================================
                
                if processed_count % SAVE_INTERVAL == 0:
                    with open(output_json_path, 'w') as f:
                        json.dump(data, f, indent=2)

        with open(output_json_path, 'w') as f:
            json.dump(data, f, indent=2)
            
    print(f"[GPU {gpu_id}] Finished all files.")

# =========================================================
# 主函数
# =========================================================
def main():
    try: mp.set_start_method('spawn', force=True)
    except: pass

    ckpt_path = check_and_download_weights()
    json_files = glob.glob(os.path.join(JSON_ROOT_DIR, "*.json"))
    json_files = [f for f in json_files if "_with_SEG.json" not in f]
    
    if not json_files: return

    num_gpus = len(TARGET_GPUS)
    chunk_size = math.ceil(len(json_files) / num_gpus)
    processes = []
    
    for i, gpu_id in enumerate(TARGET_GPUS):
        start_idx = i * chunk_size
        end_idx = min((i + 1) * chunk_size, len(json_files))
        gpu_files = json_files[start_idx:end_idx]
        if not gpu_files: continue
            
        p = mp.Process(target=worker_process, args=(gpu_id, gpu_files, ckpt_path))
        p.start()
        processes.append(p)

    for p in processes: p.join()
    print("All Done.")

if __name__ == "__main__":
    main()