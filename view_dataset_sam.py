import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cv2
import json
import torch
import random
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

# ==========================================
# 1. Augmentation Functions (Sync Mask & Image)
# ==========================================

def aug_random_crop(img, bbox, eye, gazes, inout):
    # img: (H, W, 4) -> RGB + Mask
    h, w = img.shape[:2]
    bbox_xmin, bbox_ymin, bbox_xmax, bbox_ymax = bbox

    keep_xmin, keep_ymin, keep_xmax, keep_ymax = bbox_xmin, bbox_ymin, bbox_xmax, bbox_ymax
    
    if inout == 1 and len(gazes) > 0:
        gaze_xs = [g[0] for g in gazes]; gaze_ys = [g[1] for g in gazes]
        gaze_buffer = random.randint(30, 80)
        keep_xmin = min(keep_xmin, min(gaze_xs) - gaze_buffer)
        keep_ymin = min(keep_ymin, min(gaze_ys) - gaze_buffer)
        keep_xmax = max(keep_xmax, max(gaze_xs) + gaze_buffer)
        keep_ymax = max(keep_ymax, max(gaze_ys) + gaze_buffer)

    keep_xmin = max(0, int(np.floor(keep_xmin)))
    keep_ymin = max(0, int(np.floor(keep_ymin)))
    keep_xmax = min(w, int(np.ceil(keep_xmax)))
    keep_ymax = min(h, int(np.ceil(keep_ymax)))

    try:
        crop_x1 = random.randint(0, keep_xmin)
        crop_y1 = random.randint(0, keep_ymin)
        crop_x2 = random.randint(keep_xmax, w)
        crop_y2 = random.randint(keep_ymax, h)
    except ValueError:
        return img, bbox, eye, gazes

    if crop_x2 <= crop_x1 or crop_y2 <= crop_y1: return img, bbox, eye, gazes

    img_cropped = img[crop_y1:crop_y2, crop_x1:crop_x2]
    bbox_new = [bbox[0]-crop_x1, bbox[1]-crop_y1, bbox[2]-crop_x1, bbox[3]-crop_y1]
    eye_new = [eye[0]-crop_x1, eye[1]-crop_y1]
    gazes_new = [[g[0]-crop_x1, g[1]-crop_y1] for g in gazes]

    return img_cropped, bbox_new, eye_new, gazes_new

def aug_horiz_flip(img, bbox, eye, gazes, inout):
    h, w = img.shape[:2]
    img_flipped = cv2.flip(img, 1)
    xmin, ymin, xmax, ymax = bbox
    bbox_new = [w - xmax, ymin, w - xmin, ymax]
    eye_new = [w - eye[0], eye[1]]
    
    gazes_new = []
    for gx, gy in gazes:
        if inout: gazes_new.append([w - gx, gy])
        else: gazes_new.append([gx, gy])

    return img_flipped, bbox_new, eye_new, gazes_new

def aug_bbox_jitter(img, bbox):
    h, w = img.shape[:2]
    xmin, ymin, xmax, ymax = bbox
    box_w, box_h = xmax - xmin, ymax - ymin
    jitter = 0.1
    xmin_j = (random.random() * 2 - 1) * jitter * box_w
    xmax_j = (random.random() * 2 - 1) * jitter * box_w
    ymin_j = (random.random() * 2 - 1) * jitter * box_h
    ymax_j = (random.random() * 2 - 1) * jitter * box_h
    return [max(0, xmin+xmin_j), max(0, ymin+ymin_j), min(w, xmax+xmax_j), min(h, ymax+ymax_j)]

# ==========================================
# 2. GazeFollowDataset Class
# ==========================================

class GazeFollowDataset(Dataset):
    def __init__(self, json_path, transform_face=None, transform_scene=None, is_train=True):
        self.transform_face = transform_face
        self.transform_scene = transform_scene 
        self.dataset_name = "GazeFollow_Extended"
        self.is_train = is_train
        
        with open(json_path, 'r') as f:
            self.data = json.load(f)
        print(f"Loaded Dataset: {len(self.data)} samples. Source: {os.path.basename(json_path)}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        img_path = item['img_path']
        
        # 1. Load Image
        full_image = cv2.imread(img_path)
        if full_image is None: raise FileNotFoundError(f"Img not found: {img_path}")
        full_image = cv2.cvtColor(full_image, cv2.COLOR_BGR2RGB)
        h, w, c = full_image.shape

        # 2. Load Mask (Only if it exists in JSON)
        mask_overlay = np.zeros((h, w), dtype=np.uint8)
        has_mask = False
        
        raw_gazes = item.get('gazes', [])
        # Only try to load mask if we have a valid gaze
        mask_path = None
        if len(raw_gazes) > 0:
            mask_path = raw_gazes[0].get('seg_mask_path')
            
        if mask_path and os.path.exists(mask_path):
            mask_img = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if mask_img is not None:
                # Resize robustness check
                if mask_img.shape != (h, w):
                    mask_img = cv2.resize(mask_img, (w, h), interpolation=cv2.INTER_NEAREST)
                mask_overlay = mask_img
                has_mask = True
        
        # Stack Mask as 4th channel
        full_data = np.dstack((full_image, mask_overlay))

        # 3. Coordinates
        bbox_n = item['head_bbox_norm']
        cur_bbox = [bbox_n['x_min']*w, bbox_n['y_min']*h, bbox_n['x_max']*w, bbox_n['y_max']*h]
        eye_n = item['eye_point_norm']
        cur_eye = [eye_n['x']*w, eye_n['y']*h]
        
        cur_gazes_pixel = []
        is_inout = 0
        for g in raw_gazes:
            if "gaze_point_norm" in g and g.get('inout', 0) == 1:
                cur_gazes_pixel.append([g['gaze_point_norm']['x']*w, g['gaze_point_norm']['y']*h])
                is_inout = 1
            else:
                cur_gazes_pixel.append([-1.0, -1.0])

        # 4. Augmentation
        if self.is_train:
            if random.random() <= 0.5:
                full_data, cur_bbox, cur_eye, cur_gazes_pixel = aug_random_crop(
                    full_data, cur_bbox, cur_eye, cur_gazes_pixel, is_inout
                )
            if random.random() <= 0.5:
                full_data, cur_bbox, cur_eye, cur_gazes_pixel = aug_horiz_flip(
                    full_data, cur_bbox, cur_eye, cur_gazes_pixel, is_inout
                )
            if random.random() <= 0.5:
                cur_bbox = aug_bbox_jitter(full_data, cur_bbox)

        # Split back
        full_image_aug = full_data[:, :, :3]
        mask_aug = full_data[:, :, 3]
        h_new, w_new, _ = full_image_aug.shape

        # 5. Transform & Resize
        if self.transform_scene:
            scene_img_tensor = self.transform_scene(full_image_aug)
            # Mask must use NEAREST interpolation
            mask_pil = Image.fromarray(mask_aug)
            mask_resized = mask_pil.resize((448, 448), resample=Image.NEAREST)
            mask_tensor = transforms.ToTensor()(mask_resized) 
        else:
            scene_img_tensor = transforms.ToTensor()(full_image_aug)
            mask_tensor = transforms.ToTensor()(mask_aug)

        # 6. Normalize Coordinates
        final_bbox = torch.tensor([
            cur_bbox[0]/w_new*448, cur_bbox[1]/h_new*448, 
            cur_bbox[2]/w_new*448, cur_bbox[3]/h_new*448
        ], dtype=torch.float32)
        
        final_eye = torch.tensor([
            cur_eye[0]/w_new*448, cur_eye[1]/h_new*448
        ], dtype=torch.float32)
        
        gaze_points_norm_final = []
        for i, (gx, gy) in enumerate(cur_gazes_pixel):
            if raw_gazes[i].get('inout') == 1:
                gaze_points_norm_final.append([gx/w_new, gy/h_new])
            else:
                gaze_points_norm_final.append([-1.0, -1.0])
        
        obs_expr = ""
        if "observer_expression" in item and item['observer_expression'] and "unique" in item['observer_expression']:
             if len(item['observer_expression']['unique']) > 0:
                obs_expr = random.choice(item['observer_expression']['unique'])

        file_name = os.path.basename(img_path)

        return {
            "following_subtask": {
                "scene_img": scene_img_tensor,
                "mask_img": mask_tensor, 
                "has_mask": has_mask,
                "face_bbox_pixel": final_bbox,
                "eye_point_pixel": final_eye, 
                "gaze_points_norm": gaze_points_norm_final,
                "observer_expression": obs_expr,
                "img_path_name": file_name
            }
        }

# ==========================================
# 3. Visualization Function (Full Features)
# ==========================================

def denormalize(tensor, mean, std):
    t = tensor.clone().detach().cpu()
    for i in range(3):
        t[i] = t[i] * std[i] + mean[i]
    img_np = t.permute(1, 2, 0).numpy()
    img_np = np.clip(img_np, 0, 1)
    return img_np

def visualize_complete_sample(batch_data, batch_idx=0):
    """
    Visualizes:
    1. Base Image
    2. Red Mask Overlay (Segmentation)
    3. Green BBox (Observer Head)
    4. Blue Cross (Gaze Point)
    5. Yellow Dashed Line (Eye -> Gaze)
    """
    data = batch_data['following_subtask']
    
    # 1. Base Image
    scene_tensor = data['scene_img'][batch_idx]
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    img_vis = denormalize(scene_tensor, mean, std)
    
    # 2. Red Mask Overlay
    mask_tensor = data['mask_img'][batch_idx]
    mask_np = mask_tensor.squeeze().cpu().numpy() # H, W
    H, W = mask_np.shape
    
    # RGBA Overlay: Red with Alpha based on mask
    overlay = np.zeros((H, W, 4), dtype=np.float32)
    overlay[..., 0] = 1.0 # Red Channel
    overlay[..., 3] = np.where(mask_np > 0.1, 0.4, 0.0) # Alpha Channel (0.4 if mask exists)
    
    # 3. Coordinates
    bbox = data['face_bbox_pixel'][batch_idx].cpu().numpy()
    eye = data['eye_point_pixel'][batch_idx].cpu().numpy()
    gaze_norms = data['gaze_points_norm']
    desc = data['observer_expression'][batch_idx]
    
    # === Plotting ===
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    
    # Draw Image & Mask
    ax.imshow(img_vis)
    ax.imshow(overlay) 
    
    # Draw Observer BBox (Green)
    x1, y1, x2, y2 = bbox
    rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, linewidth=2, edgecolor='#00FF00', facecolor='none', label='Observer')
    ax.add_patch(rect)
    
    # Draw Gaze & Line
    ex, ey = eye[0], eye[1]
    valid_gaze_found = False
    
    for g_tensor in gaze_norms:
        g = g_tensor[batch_idx].cpu().numpy()
        # Check if valid (not -1)
        if g[0] > 0 and g[1] > 0:
            gx, gy = g[0] * W, g[1] * H
            
            # Yellow Dashed Line (Eye -> Gaze)
            ax.plot([ex, gx], [ey, gy], color='yellow', linestyle='--', linewidth=2, alpha=0.9)
            
            # Blue Cross (Gaze Point)
            ax.scatter([gx], [gy], c='blue', s=150, marker='X', linewidth=2.5, edgecolors='white', label='Gaze Target')
            
            valid_gaze_found = True
            
    # Title & Save
    status_text = "Valid Gaze" if valid_gaze_found else "Gaze Out-of-Frame"
    title_str = f"File: {data['img_path_name'][batch_idx]}\nStatus: {status_text} | Mask: True\nDesc: {desc}"
    
    ax.set_title(title_str, fontsize=11, color='black', wrap=True)
    ax.axis('off')
    
    save_dir = './tmp/vis_complete'
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"vis_{data['img_path_name'][batch_idx]}")
    save_path = os.path.splitext(save_path)[0] + ".png"
    
    plt.savefig(save_path, bbox_inches='tight', dpi=100)
    plt.close()
    print(f"Saved visualization: {save_path}")

# ==========================================
# 4. Main Search Loop
# ==========================================

def main():
    transform_face = transforms.Compose([
        transforms.ToPILImage(), transforms.Resize((224, 224)),
        transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    transform_scene = transforms.Compose([
        transforms.ToPILImage(), transforms.Resize((448, 448)),
        transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # JSON File Path
    gf_json = '/mnt/nvme1n1/lululemon/xjj/result/information_fusion/merged_GazeFollow_train_EN_Qwen_Qwen3-VL-32B-Instruct_with_SEG.json'
    
    print("Initializing Dataset...")
    ds = GazeFollowDataset(gf_json, transform_face=transform_face, transform_scene=transform_scene, is_train=True)
    
    # Larger batch size to search faster
    loader = DataLoader(ds, batch_size=16, shuffle=True, num_workers=4)
    
    print("Searching for samples with BOTH Mask and Valid In-Frame Gaze...")
    count = 0
    target = 5
    
    for batch_i, batch in enumerate(loader):
        data = batch['following_subtask']
        has_mask_flags = data['has_mask']
        gaze_norms = data['gaze_points_norm'] # List of tensors
        
        for i in range(len(has_mask_flags)):
            # Condition 1: Must have mask
            has_mask = has_mask_flags[i]
            
            # Condition 2: Must have valid gaze point (inout=1)
            # Check the first annotator's gaze point
            g_xy = gaze_norms[0][i] 
            is_valid_gaze = (g_xy[0] > 0) # if x > 0, it's valid (we set -1 for invalid)

            if has_mask and is_valid_gaze:
                print(f"--> Found Valid Sample! Batch {batch_i}, Index {i}")
                
                # Create single-item batch for visualization
                single_sample = {
                    'following_subtask': {
                        k: v[i:i+1] for k, v in data.items()
                    }
                }
                
                visualize_complete_sample(single_sample, batch_idx=0)
                count += 1
                
                if count >= target:
                    print("Found 5 valid samples. Exiting.")
                    return
        
        print(f"Batch {batch_i} searched. Found so far: {count}")
        if batch_i > 500: break

if __name__ == "__main__":
    main()