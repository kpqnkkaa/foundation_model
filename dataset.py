# # 加载两种数据集



# #第一种gaze_estimation数据集

# #它包括标注文件和图片目录，其中txt可能是单个文件，或者一个文件夹，那就合并下面的所有文件

# #我们需要获得他的图片路径（已近预处理好了只包括人脸，然后resize成224），这是gaze estimation分支的输入

# # 以及标签yaw,pitch标注信息(这是gaze estimation分支的 3D(包含深度信息) 输出ground truth)



# def Decode_ETH(line):

#     anno = edict()

#     anno.face = line[0]d

#     anno.gaze = [float(x) for x in line[1:3]]

#     return anno



# def Decode_Gaze360(line):

#     anno = edict()

#     anno.face = line[0]

#     anno.gaze = [float(x) for x in line[1:3]]

#     return anno



# def Decode_MPIIGaze(line):

#     anno = edict()

#     anno.face = line[0]

#     if ',' in line[-1]: anno.gaze = [float(x) for x in line[-1].split(',')]

#     else: anno.gaze = [float(x) for x in line[-2:]]

#     return anno



# def Decode_EyeDiap(line):

#     anno = edict()

#     anno.face = line[0]

#     anno.gaze = [float(x) for x in line[1:3]]

#     return anno



# def Get_Decode(path_string):

#     s = path_string.lower()

#     if "eth" in s: return Decode_ETH

#     if "gaze360" in s: return Decode_Gaze360

#     if "mpii" in s: return Decode_MPIIGaze

#     if "eyediap" in s or "diap" in s: return Decode_EyeDiap

#     return None



# # 第二种gaze_follow数据集

# # 他的输入是一个json文件

# # 格式如下：   {

#     # "index": "00000015_person0",

#     # "img_path": "/mnt/nvme1n1/lululemon/xjj/datasets/resized/gazefollow_extended/test2/00000000/00000015.jpg",

#     # "head_bbox_norm": {

#     #   "x_min": 0.498,

#     #   "y_min": 0.008,

#     #   "x_max": 0.706,

#     #   "y_max": 0.215

#     # },

#     # "eye_point_norm": {

#     #   "x": 0.538,

#     #   "y": 0.107

#     # },

#     # "picture_scale": [

#     #   640,

#     #   480

#     # ],

#     # "bbox_path": "/mnt/nvme1n1/lululemon/xjj/datasets/resized/bbox/gazefollow_extended/test/00000015_person0.jpg",

#     # "observer_expression": {

#     #   "unique": [

#     #     "the woman in a striped dress",

#     #     "the woman chopping green beans on a wooden counter",

#     #     "the woman wearing glasses and a sleeveless dress"

#     #   ],

#     #   "ambiguous": [],

#     #   "non_exist": [

#     #     "the man wearing a blue shirt",

#     #     "the child standing near the window"

#     #   ],

#     #   "pronouns": "she"

#     # },

#     # "gazes": [

#     #   {

#     #     "index": "00000015_person0_label0",

#     #     "gaze_point_norm": {

#     #       "x": 0.353,

#     #       "y": 0.612

#     #     },

#     #     "gaze_direction": "below",

#     #     "inout": 1,

#     #     "gaze_point_expressions": [

#     #       "the green vegetable she is cutting on the wooden cutting board",

#     #       "the piece of zucchini she is slicing with her knife",

#     #       "the vegetable lying on the cutting board just in front of her hands"

#     #     ]

#     #   },

#     #   {

#     #     "index": "00000015_person0_label1",

#     #     "gaze_point_norm": {

#     #       "x": 0.355,

#     #       "y": 0.614

#     #     },

#     #     "gaze_direction": "below",

#     #     "inout": 1,

#     #     "gaze_point_expressions": [

#     #       "the green vegetable she is cutting on the wooden cutting board",

#     #       "the piece of zucchini she is slicing with a knife on the counter",

#     #       "the vegetable in front of her on the cutting board, just to the left of the bowl of salad"

#     #     ]

#     #   }}}

#     # 我们需要其中的img_path他是输入的图片路径

#     # 同时我们获得head_bbox_norm，根据他裁剪出人脸，然后resize成224用于gaze estimation分支的输入

#     # gazes下面的多个gaze_point_norm，根据它跟eye_point_norm计算出gaze_direction，他的平均值作为gaze estimation分支的2D（平面算出来的，不包含深度信息） ground truth

#     # 同时我们可以获得observer_expression(随机采样一个)，head_bbox_norm, eye_point_norm

#     # 他们可以作为视线追随的SAM prompt encoder的输入 结合 img_path 对应的图片，我们把它resize成1024通过image encoder

#     # 而视线追随分支的输出也有三种形式，描述（gaze_point_expressions）， gaze_point_norm(坐标)，以及现在没有后续我会补充的分割结果（现在没有）

   





#     # 我们训练的时候对于视线追随的数据集，可能有多个，平均采样混合

#     # 对于视线估计数据集也有多个，平均采样混合

#     # 对于视线追随和视线估计每次都同时返回，因为是联合训练

#     # 最终返回的格式是{

#     #     "gaze_estimation": {

#     #         "face_img": face_img,

#     #         "3D_ground_truth": 3D_ground_truth,

#     #         "dataset_name": dataset_name,

#     #     },

#     #     "gaze_following": {

#     #         "estimation_subtask": {

#     #             "face_img": face_img,

#     #             "2D_ground_truth": 2D_ground_truth,

#     #         },

#     #         "following_subtask": {

#     #             "scene_img": scene_img,

#     #             "face_bbox": face_bbox,

#     #             "eye_point_norm": eye_point_norm,

#     #             "observer_expression": observer_expression,

#     #             "gaze_point_norm": [gaze_point_norm1, gaze_point_norm2, ...],

#     #             "gaze_point_expressions": [gaze_point_expressions1, gaze_point_expressions2, ...],

#     #         },

#     #         "dataset_name": dataset_name,

#     #     }

#     # }

# 我更新下了要求，请写出，并给出示例（gaze estimation的数据有eth：/mnt/nvme1n1/lululemon/xjj/datasets/ETH-Gaze/Label/train_temp.label； /mnt/nvme1n1/lululemon/xjj/datasets/ETH-Gaze/Image gaze360： /mnt/nvme1n1/lululemon/xjj/datasets/Gaze360/Label/train.label， /mnt/nvme1n1/lululemon/xjj/datasets/Gaze360/Image， gaze_following的数据集有/mnt/nvme1n1/lululemon/xjj/result/information_fusion/merged_GazeFollow_train_EN_Qwen_Qwen3-VL-32B-Instruct.json）测试dataset加载是否正确成功

import os
import matplotlib.pyplot as plt
import cv2
import json
import torch
import random
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import functional as F_vis
from torchvision import transforms
from PIL import Image
from easydict import EasyDict as edict
import glob
import math

# ==========================================
# 1. 解码与辅助函数 (Decode Functions & Utils)
# ==========================================

def Decode_ETH(line):
    anno = edict()
    anno.face = line[0]
    yaw_pitch_str = line[1]
    ys = yaw_pitch_str.split(",")
    anno.gaze = [float(ys[0]), float(ys[1])]
    return anno

def Decode_Gaze360(line):
    anno = edict()
    anno.face = line[0]
    yaw_pitch_str = line[-1]
    ys = yaw_pitch_str.split(",")
    anno.gaze = [float(ys[0]), float(ys[1])]
    return anno

def Decode_MPIIGaze(line):
    anno = edict()
    anno.face = line[0]
    yaw_pitch_str = line[7]
    ys = yaw_pitch_str.split(",")
    anno.gaze = [float(ys[0]), float(ys[1])]
    return anno

def Decode_EyeDiap(line):
    anno = edict()
    anno.face = line[0]
    yaw_pitch_str = line[6]
    ys = yaw_pitch_str.split(",")
    anno.gaze = [float(ys[0]), float(ys[1])]
    return anno

def Get_Decode(path_string):
    s = path_string.lower()
    if "eth" in s: return Decode_ETH
    if "gaze360" in s: return Decode_Gaze360
    if "mpii" in s: return Decode_MPIIGaze
    if "eyediap" in s or "diap" in s: return Decode_EyeDiap
    return None

def get_2d_gaze_vector(eye_norm, gaze_norm):
    dx = gaze_norm['x'] - eye_norm['x']
    dy = gaze_norm['y'] - eye_norm['y']
    norm = np.sqrt(dx**2 + dy**2)
    if norm < 1e-6: return [0.0, 0.0]
    return [dx / norm, dy / norm]

def get_2d_angles(eye_norm, gaze_norm):
    dx = gaze_norm['x'] - eye_norm['x']
    dy = gaze_norm['y'] - eye_norm['y']
    return np.arctan2(-dy, dx)

def gaze_to_xyz_vector(angles):
    if isinstance(angles, (list, tuple, np.ndarray)):
        yaw, pitch = angles[0], angles[1]
    else:
        yaw, pitch = angles[0].item(), angles[1].item()
    dx = -np.cos(pitch) * np.sin(yaw)
    dy = -np.sin(pitch)
    dz = -np.cos(pitch) * np.cos(yaw)
    return [dx, dy, dz]

# ==========================================
# 2. 视线估计数据集 (Gaze Estimation Dataset)
# ==========================================

class GazeEstimationDataset(Dataset):
    def __init__(self, label_path, image_root, transform=None, is_train=True):
        self.transform = transform
        self.image_root = image_root
        self.is_train = is_train 
        self.lines = []
        self.decode_function = Get_Decode(label_path)
        self.dataset_name = label_path.split('/')[-3] if len(label_path.split('/')) > 3 else "unknown_est"

        if os.path.isdir(label_path):
            files = glob.glob(os.path.join(label_path, '*.label')) + glob.glob(os.path.join(label_path, '*.txt'))
        else:
            files = [label_path]

        for f_path in files:
            with open(f_path, 'r') as f:
                content = f.readlines()[1:] 
                self.lines.extend([l.strip().split() for l in content])

        print(f"Loaded GazeEstimation Dataset ({self.dataset_name}): {len(self.lines)} samples. (Train={is_train})")

        if self.is_train:
            self.color_jitter = transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05)
            self.bbox_jitter = transforms.RandomResizedCrop(size=(224, 224), scale=(0.85, 1.0), ratio=(0.9, 1.1))

    def __len__(self):
        return len(self.lines)

    def __getitem__(self, idx):
        line = self.lines[idx]
        anno = self.decode_function(line)
        img_path = os.path.join(self.image_root, anno.face)        
        
        image = cv2.imread(img_path)
        if image is None: raise FileNotFoundError(f"Image not found: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        yaw = anno.gaze[0]
        pitch = anno.gaze[1]
        
        if self.is_train:
            pil_img = Image.fromarray(image)
            if random.random() < 0.5: pil_img = self.color_jitter(pil_img)
            pil_img = self.bbox_jitter(pil_img)
            
            if random.random() < 0.5:
                pil_img = F_vis.hflip(pil_img)
                yaw = -yaw 
            image = np.array(pil_img)
        
        if self.transform:
            image = self.transform(image)

        return {
            "face_img": image,
            "3D_ground_truth": torch.tensor([yaw, pitch], dtype=torch.float32),
            "dataset_name": self.dataset_name
        }

# ==========================================
# 3. 视线追随数据集 (Gaze Follow Dataset)
# ==========================================

# ... (aug_random_crop, aug_horiz_flip, aug_bbox_jitter 保持不变，直接复制你的代码即可) ...
def aug_random_crop(img, bbox, eye, gazes, inout):
    h, w = img.shape[:2]
    bbox_xmin, bbox_ymin, bbox_xmax, bbox_ymax = bbox
    keep_xmin, keep_ymin, keep_xmax, keep_ymax = bbox_xmin, bbox_ymin, bbox_xmax, bbox_ymax

    if inout == 1 and len(gazes) > 0:
        gaze_xs = [g[0] for g in gazes]; gaze_ys = [g[1] for g in gazes]
        
        # [修改] gaze_buffer 改为原图大小的比例得到
        # 设定一个随机比例，例如图像短边的 5% 到 20%
        # 对于 448x448 的图，这大约是 22px 到 90px，覆盖了原来的 30-80 范围
        buffer_ratio = random.uniform(0.1, 0.2)
        gaze_buffer = int(min(h, w) * buffer_ratio)
        
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

def aug_horiz_flip(img, bbox, eye, gazes, inout, pseudo_3d_gts):
    h, w, _ = img.shape
    img_flipped = cv2.flip(img, 1)
    xmin, ymin, xmax, ymax = bbox
    bbox_new = [w - xmax, ymin, w - xmin, ymax]
    eye_new = [w - eye[0], eye[1]]
    
    gazes_new = []
    for gx, gy in gazes:
        if inout: gazes_new.append([w - gx, gy])
        else: gazes_new.append([gx, gy])

    pseudo_3d_gts_new = []
    for yaw, pitch in pseudo_3d_gts:
        if inout: pseudo_3d_gts_new.append([-yaw, pitch])
        else: pseudo_3d_gts_new.append([yaw, pitch])

    return img_flipped, bbox_new, eye_new, gazes_new, pseudo_3d_gts_new

def aug_bbox_jitter(img, bbox):
    h, w, _ = img.shape
    xmin, ymin, xmax, ymax = bbox
    box_w, box_h = xmax - xmin, ymax - ymin
    jitter = 0.1
    xmin_j = (random.random() * 2 - 1) * jitter * box_w
    xmax_j = (random.random() * 2 - 1) * jitter * box_w
    ymin_j = (random.random() * 2 - 1) * jitter * box_h
    ymax_j = (random.random() * 2 - 1) * jitter * box_h
    return [max(0, xmin+xmin_j), max(0, ymin+ymin_j), min(w, xmax+xmax_j), min(h, ymax+ymax_j)]

class GazeFollowDataset(Dataset):
    def __init__(self, json_path, transform_face=None, transform_scene=None, is_train=True):
        self.transform_face = transform_face
        self.transform_scene = transform_scene # DINOv2 uses 448
        self.dataset_name = "GazeFollow_Extended"
        self.is_train = is_train
        
        # [修改] 增加 ColorJitter (匹配 Gaze-LLE / Standard ViT Training)
        if self.is_train:
            self.color_jitter = transforms.ColorJitter(
                brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1
            )
        
        with open(json_path, 'r') as f:
            self.data = json.load(f)
        print(f"Loaded GazeFollow Dataset: {len(self.data)} samples. (Train={is_train})")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        img_path = item['img_path']
        full_image = cv2.imread(img_path)
        if full_image is None: raise FileNotFoundError
        full_image = cv2.cvtColor(full_image, cv2.COLOR_BGR2RGB)
        h, w, c = full_image.shape

        bbox_n = item['head_bbox_norm']
        cur_bbox = [bbox_n['x_min']*w, bbox_n['y_min']*h, bbox_n['x_max']*w, bbox_n['y_max']*h]
        eye_n = item['eye_point_norm']
        cur_eye = [eye_n['x']*w, eye_n['y']*h]
        
        raw_gazes = item['gazes']
        cur_gazes_pixel = []
        is_inout = 0
        pseudo_3d_gts = []
        
        for g in raw_gazes:
            if "gaze_point_norm" in g and g.get('inout', 0) == 1:
                cur_gazes_pixel.append([g['gaze_point_norm']['x']*w, g['gaze_point_norm']['y']*h])
                pseudo_3d_gts.append(g['pseudo_3d_gaze'])
                is_inout = 1
            else:
                cur_gazes_pixel.append([-1.0, -1.0])
                pseudo_3d_gts.append([-10.0, -10.0])

        origin_bbox = cur_bbox
        
        # === Data Augmentation ===
        if self.is_train:
            # 1. Random Crop
            if random.random() <= 0.5:
                full_image, cur_bbox, cur_eye, cur_gazes_pixel = aug_random_crop(
                    full_image, cur_bbox, cur_eye, cur_gazes_pixel, is_inout
                )
                origin_bbox = cur_bbox
            
            # 2. Horizontal Flip
            if random.random() <= 0.5:
                full_image, cur_bbox, cur_eye, cur_gazes_pixel, pseudo_3d_gts = aug_horiz_flip(
                    full_image, cur_bbox, cur_eye, cur_gazes_pixel, is_inout, pseudo_3d_gts
                )
                origin_bbox = cur_bbox
            
            # 3. BBox Jitter
            if random.random() <= 0.5:
                cur_bbox = aug_bbox_jitter(full_image, cur_bbox)
            
            # # [修改] 4. Color Jitter (新增)
            # if random.random() <= 0.5:
            #     pil_img = Image.fromarray(full_image)
            #     pil_img = self.color_jitter(pil_img)
            #     full_image = np.array(pil_img)

        # Update h, w
        h, w, c = full_image.shape

        # === Post-Processing ===
        x_min, y_min = int(max(0, cur_bbox[0])), int(max(0, cur_bbox[1]))
        x_max, y_max = int(min(w, cur_bbox[2])), int(min(h, cur_bbox[3]))
        face_img = full_image[y_min:y_max, x_min:x_max]
        if face_img.size == 0: face_img = full_image # Fallback

        # Calculate average gaze
        sum_gx, sum_gy, sum_yaw, sum_pitch, valid_cnt = 0, 0, 0, 0, 0
        for i, (gx, gy) in enumerate(cur_gazes_pixel):
            if raw_gazes[i].get('inout') == 1:
                sum_gx += gx; sum_gy += gy
                sum_yaw += pseudo_3d_gts[i][0]; sum_pitch += pseudo_3d_gts[i][1]
                valid_cnt += 1
        
        if valid_cnt > 0:
            avg_gaze_norm_x = (sum_gx / valid_cnt) / w
            avg_gaze_norm_y = (sum_gy / valid_cnt) / h
            eye_norm_x = cur_eye[0] / w
            eye_norm_y = cur_eye[1] / h
            gaze_2d_vector = get_2d_gaze_vector({'x': eye_norm_x, 'y': eye_norm_y}, {'x': avg_gaze_norm_x, 'y': avg_gaze_norm_y})
            gaze_2d_angles = get_2d_angles({'x': eye_norm_x, 'y': eye_norm_y}, {'x': avg_gaze_norm_x, 'y': avg_gaze_norm_y})
            avg_yaw = sum_yaw / valid_cnt
            avg_pitch = sum_pitch / valid_cnt
        else:
            gaze_2d_vector = [-1.0, -1.0]
            gaze_2d_angles = -10.0
            avg_yaw = avg_pitch = -10.0

        if self.transform_face: face_img_tensor = self.transform_face(face_img)
        else: face_img_tensor = transforms.ToTensor()(face_img)

        if self.transform_scene: scene_img_tensor = self.transform_scene(full_image)
        else: scene_img_tensor = transforms.ToTensor()(full_image)

        # Observer Expression
        if "observer_expression" in item and item['observer_expression'] is not None and "unique" in item['observer_expression']:
            obs_expr_pool = item['observer_expression']['unique']
            observer_expression = random.choice(obs_expr_pool) if len(obs_expr_pool) > 0 else ""
        else:
            observer_expression = ""

        # Modality Dropout (Keep all for now or implement logic)
        final_expression = observer_expression
        
        # Prepare Outputs
        new_w, new_h = scene_img_tensor.shape[2], scene_img_tensor.shape[1]
        
        final_bbox = torch.tensor([
            cur_bbox[0]/w*new_w, cur_bbox[1]/h*new_h, cur_bbox[2]/w*new_w, cur_bbox[3]/h*new_h
        ], dtype=torch.float32)
        
        face_bbox_gt = torch.tensor([
            origin_bbox[0]/w, origin_bbox[1]/h, origin_bbox[2]/w, origin_bbox[3]/h
        ], dtype=torch.float32)

        final_eye = torch.tensor([cur_eye[0]/w*new_w, cur_eye[1]/h*new_h], dtype=torch.float32)

        gaze_points_norm_final = []
        for i, (gx, gy) in enumerate(cur_gazes_pixel):
            if raw_gazes[i].get('inout') == 1 and 0 <= gx <= w and 0 <= gy <= h:
                gaze_points_norm_final.append([gx/w, gy/h])
            else:
                gaze_points_norm_final.append([-1.0, -1.0])

        in_outs = [g.get('inout') for g in raw_gazes]
        all_gaze_exprs = [random.choice(g['gaze_point_expressions']) if 'gaze_point_expressions' in g and g['gaze_point_expressions'] else "" for g in raw_gazes]

        return {
            "estimation_subtask": {
                "face_img": face_img_tensor,
                "2D_ground_truth": torch.tensor(gaze_2d_vector, dtype=torch.float32),
                "3D_ground_truth": torch.tensor([avg_yaw, avg_pitch], dtype=torch.float32),
            },
            "following_subtask": {
                "scene_img": scene_img_tensor,
                "width": w,
                "height": h,
                "face_bbox_pixel": final_bbox,
                "eye_point_pixel": final_eye,
                "face_bbox_gt_norm": face_bbox_gt,
                "observer_expression": final_expression,
                "gaze_points_norm": gaze_points_norm_final,
                "gaze_point_expressions": all_gaze_exprs,
                "in_outs": in_outs,
                "2D_ground_truth": torch.tensor(gaze_2d_vector, dtype=torch.float32),
                "2D_angles_ground_truth_rad": torch.tensor(gaze_2d_angles, dtype=torch.float32)
            },
            "dataset_name": self.dataset_name
        }

# ... (JointFusionDataset 保持不变) ...
class JointFusionDataset(Dataset):
    def __init__(self, est_datasets_list, follow_datasets_list, est_batch_multiplier=8):
        self.est_datasets = est_datasets_list
        self.follow_datasets = follow_datasets_list
        self.est_batch_multiplier = est_batch_multiplier
        self.follow_sizes = [len(d) for d in self.follow_datasets]
        self.epoch_length = sum(self.follow_sizes)
        
        est_lens = [len(ds) for ds in self.est_datasets]
        max_est_len = max(est_lens)
        
        self.global_est_pool = []
        for i, length in enumerate(est_lens):
            base_indices = [(i, j) for j in range(length)]
            factor = math.ceil(max_est_len / length)
            balanced_indices = (base_indices * factor)[:max_est_len]
            self.global_est_pool.extend(balanced_indices)
            
        random.shuffle(self.global_est_pool)
        self.est_pointer = 0 
        self.current_epoch_est_indices = []
        self.step_epoch() 

    def step_epoch(self):
        needed = self.epoch_length * self.est_batch_multiplier
        indices = []
        while needed > 0:
            remaining = len(self.global_est_pool) - self.est_pointer
            if remaining >= needed:
                start = self.est_pointer
                end = self.est_pointer + needed
                indices.extend(self.global_est_pool[start:end])
                self.est_pointer += needed
                needed = 0
            else:
                indices.extend(self.global_est_pool[self.est_pointer:])
                needed -= remaining
                random.shuffle(self.global_est_pool)
                self.est_pointer = 0
        self.current_epoch_est_indices = indices
        print(f"Dataset Epoch Update: Pointer at {self.est_pointer}/{len(self.global_est_pool)}")

    def __len__(self):
        return self.epoch_length

    def __getitem__(self, idx):
        dataset_idx = 0
        local_idx = idx
        for size in self.follow_sizes:
            if local_idx < size: break
            local_idx -= size
            dataset_idx += 1
        follow_data = self.follow_datasets[dataset_idx][local_idx]

        indices_slice = self.current_epoch_est_indices[idx * self.est_batch_multiplier: (idx + 1) * self.est_batch_multiplier]
        est_data_list = []
        for est_ds_i, est_img_i in indices_slice:
            est_data_list.append(self.est_datasets[est_ds_i][est_img_i])
        
        return {"gaze_estimation": est_data_list, "gaze_following": follow_data}

# ==========================================
# 5. 测试代码 (Test Execution)
# ==========================================

def main():
    transform_face = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # [修改] Scene resize to 448 (Match Gaze-LLE / DINOv2)
    transform_scene = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((448, 448)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 1. 准备数据集路径
    eth_label = '/mnt/nvme1n1/lululemon/xjj/datasets/ETH-Gaze/Label/train_temp.label'
    eth_img = '/mnt/nvme1n1/lululemon/xjj/datasets/ETH-Gaze/Image'
    gaze360_label = '/mnt/nvme1n1/lululemon/xjj/datasets/Gaze360/Label/train.label'
    gaze360_img = '/mnt/nvme1n1/lululemon/xjj/datasets/Gaze360/Image'
    gf_json = '/mnt/nvme1n1/lululemon/xjj/result/information_fusion/merged_GazeFollow_train_EN_Qwen_Qwen3-VL-32B-Instruct.json'

    print("=== Initializing Datasets ===")
    ds_eth = GazeEstimationDataset(eth_label, eth_img, transform=transform_face)
    ds_g360 = GazeEstimationDataset(gaze360_label, gaze360_img, transform=transform_face)
    ds_gf = GazeFollowDataset(gf_json, transform_face=transform_face, transform_scene=transform_scene)
    
    joint_dataset = JointFusionDataset(
        est_datasets_list=[ds_eth, ds_g360], 
        follow_datasets_list=[ds_gf]
    )
    
    print(f"=== Joint Dataset Created. Epoch Length: {len(joint_dataset)} ===")

    loader = DataLoader(joint_dataset, batch_size=2, shuffle=True, num_workers=0)
    
    print("=== Fetching One Batch ===")
    for i, batch in enumerate(loader):
        gf = batch['gaze_following']
        gf_fol = gf['following_subtask']
        print(f" > Following Task ({gf['dataset_name']}):")
        print(f"   - scene_img Shape: {gf_fol['scene_img'].shape} (Should be 448x448)")
        print(f"   - face_bbox (pixel): {gf_fol['face_bbox_pixel']}")
        break 

if __name__ == "__main__":
    main()