import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import torch
import json
import os
import copy
from PIL import Image
import numpy as np
from transformers import GPT2Tokenizer
import cv2
import gazelle.utils as utils
import random
import glob
from easydict import EasyDict as edict
import torchvision.transforms.functional as F_vis
from torchvision import transforms

# ==========================================
# 1. 解码与辅助函数
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

# ==========================================
# 2. 数据集定义
# ==========================================

def load_data_vat(file, sample_rate):
    sequences = json.load(open(file, "r"))
    data = []
    for i in range(len(sequences)):
        for j in range(0, len(sequences[i]['frames']), sample_rate):
            data.append(sequences[i]['frames'][j])
    return data

def load_data_gazefollow(file):
    data = json.load(open(file, "r"))
    return data

class GazeDataset(torch.utils.data.dataset.Dataset):
    def __init__(self, dataset_name, path, split, transform, in_frame_only=True, sample_rate=1, is_mix_gaze_estimation=False, is_partial_input=False):
        self.dataset_name = dataset_name
        self.path = path
        self.split = split
        self.aug = self.split == "train"
        self.transform = transform
        self.in_frame_only = in_frame_only
        self.sample_rate = sample_rate
        self.expr_tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        self.expr_tokenizer.pad_token = self.expr_tokenizer.eos_token
        
        self.is_mix_gaze_estimation = is_mix_gaze_estimation and self.split == "train"
        self.is_partial_input = is_partial_input
        
        if dataset_name == "gazefollow":
            self.data = load_data_gazefollow(os.path.join(self.path, "{}_preprocessed.json".format(split)))
        elif dataset_name == "videoattentiontarget":
            self.data = load_data_vat(os.path.join(self.path, "{}_preprocessed.json".format(split)), sample_rate=sample_rate)
        else:
            raise ValueError("Invalid dataset: {}".format(dataset_name))

        self.data_idxs = []
        for i in range(len(self.data)):
            for j in range(len(self.data[i]['heads'])):
                if not self.in_frame_only or self.data[i]['heads'][j]['inout'] == 1:
                    self.data_idxs.append((i, j))

    def __getitem__(self, idx):
        img_idx, head_idx = self.data_idxs[idx]
        img_data = self.data[img_idx]
        head_data = copy.deepcopy(img_data['heads'][head_idx])
        bbox_norm = head_data['bbox_norm']
        eye_norm = head_data['eye_norm'] if 'eye_norm' in head_data else None
        gazex_norm = head_data['gazex_norm']
        gazey_norm = head_data['gazey_norm']
        inout = head_data['inout']

        observer_expression_list = head_data.get('observer_expression')
        if observer_expression_list and isinstance(observer_expression_list, list) and len(observer_expression_list) > 0:
            observer_expression = np.random.choice(observer_expression_list)
        else:
            observer_expression = ""
            
        gaze_point_expressions_list = head_data.get('gaze_point_expressions')
        if gaze_point_expressions_list and isinstance(gaze_point_expressions_list, list) and len(gaze_point_expressions_list) > 0:
            gaze_point_expression = np.random.choice(gaze_point_expressions_list)
        else:
            gaze_point_expression = ""

        direction_map = {
            "right": 0, "top-right": 1, "above": 2, "top-left": 3,
            "left": 4, "bottom-left": 5, "below": 6, "bottom-right": 7
        }
        gaze_dir_str = head_data.get('gaze_direction')
        if gaze_dir_str in direction_map:
            gaze_direction = direction_map[gaze_dir_str]
        else:
            gaze_direction = -100 # Ignore index for CE Loss
        
        seg_mask_path = head_data.get('seg_mask_path')
        if seg_mask_path is not None and os.path.exists(seg_mask_path):
            seg_mask = cv2.imread(seg_mask_path, 0)
            seg_mask = cv2.resize(seg_mask, (64, 64))
            seg_mask = seg_mask / 255.0
        else:
            seg_mask = np.zeros((64, 64))
        seg_mask = torch.from_numpy(seg_mask).float()

        img_path = os.path.join(self.path, img_data['path'])
        img = Image.open(img_path)
        img = img.convert("RGB")
        width, height = img.size

        is_face_crop_mode = False

        if self.aug:
            if self.is_mix_gaze_estimation and np.random.sample() <= 0.1:
                is_face_crop_mode = True
                
                raw_bbox = head_data['bbox']
                scale = np.random.uniform(0.8, 1.5)
                b_w = raw_bbox[2] - raw_bbox[0]; b_h = raw_bbox[3] - raw_bbox[1]
                cx = raw_bbox[0] + b_w / 2; cy = raw_bbox[1] + b_h / 2
                new_w = b_w * scale; new_h = b_h * scale
                
                crop_x1 = max(0, int(cx - new_w / 2)); crop_y1 = max(0, int(cy - new_h / 2))
                crop_x2 = min(width, int(cx + new_w / 2)); crop_y2 = min(height, int(cy + new_h / 2))
                
                img = img.crop((crop_x1, crop_y1, crop_x2, crop_y2))
                width, height = img.size 
                
                bbox_norm = [0.0, 0.0, 1.0, 1.0]
                if eye_norm is not None: eye_norm = [0.5, 0.5]
                observer_expression = np.random.choice([
                    "a close-up of a human face", "the face of a person",
                    "head pose and gaze direction", "a cropped face image"
                ])
                seg_mask = torch.zeros((64, 64)).float()
                gaze_point_expression = "outside of image"

            else:
                bbox = head_data['bbox']
                gazex = head_data['gazex']
                gazey = head_data['gazey']

                if np.random.sample() <= 0.5:
                    img, bbox, gazex, gazey = utils.random_crop(img, bbox, gazex, gazey, inout)
                if np.random.sample() <= 0.5:
                    img, bbox, gazex, gazey = utils.horiz_flip(img, bbox, gazex, gazey, inout)
                if np.random.sample() <= 0.5:
                    bbox = utils.random_bbox_jitter(img, bbox)

                width, height = img.size
                bbox_norm = [bbox[0] / width, bbox[1] / height, bbox[2] / width, bbox[3] / height]
                gazex_norm = [x / float(width) for x in gazex]
                gazey_norm = [y / float(height) for y in gazey]

        # Partial Input Logic
        if self.is_partial_input:
            if self.split == "train":
                use_text = True; use_bbox = True; use_eye = True
                text_is_reliable = (len(observer_expression) > 0)
                r = random.random() 
                if text_is_reliable:
                    if r < 0.5: 
                        r_sub = random.random()
                        if r_sub < 0.333: use_bbox = False; use_eye = False 
                        elif r_sub < 0.666: use_text = False; use_eye = False 
                        else: use_text = False; use_bbox = False 
                    elif r < 0.8: 
                        r_sub = random.random()
                        if r_sub < 0.333: use_eye = False 
                        elif r_sub < 0.666: use_bbox = False 
                        else: use_text = False 
                else: 
                    use_text = False 
                    if random.random() < 0.5:
                        if random.random() < 0.5: use_eye = False 
                        else: use_bbox = False 
            else:
                use_text = False; use_eye = False; use_bbox = True

            if not use_bbox: bbox_norm = [0.0, 0.0, 0.0, 0.0] 
            if not use_eye: eye_norm = [-1.0, -1.0] 
            if not use_text: observer_expression = ""

        # Tokenization
        if observer_expression:
            observer_encoded = self.expr_tokenizer(observer_expression, max_length=25, padding="max_length", truncation=True, return_tensors="pt")
            observer_expression_ids = observer_encoded['input_ids'].squeeze(0)
        else:
            observer_expression_ids = torch.full((25,), self.expr_tokenizer.pad_token_id, dtype=torch.long)

        if gaze_point_expression:
            gaze_point_encoded = self.expr_tokenizer(gaze_point_expression, max_length=25, padding="max_length", truncation=True, return_tensors="pt")
            gaze_point_expression_ids = gaze_point_encoded['input_ids'].squeeze(0)
        else:
            gaze_point_expression_ids = torch.full((25,), self.expr_tokenizer.pad_token_id, dtype=torch.long)

        img = self.transform(img)
        
        # [NEW] Gaze3D: GazeFollow 没有 3D 标签，设为 0
        gaze3d = torch.zeros(2).float() 
        has_3d = torch.tensor(0).long() # Flag: 0 means no valid 3D label

        if self.split == "train":
            if is_face_crop_mode:
                heatmap = torch.zeros((64, 64))
            else:
                heatmap = utils.get_heatmap(gazex_norm[0], gazey_norm[0], 64, 64) 
            
            return img, bbox_norm, eye_norm, gazex_norm, gazey_norm, torch.tensor(inout), height, width, heatmap, observer_expression_ids, torch.tensor(gaze_direction), gaze_point_expression_ids, seg_mask, is_face_crop_mode, gaze3d, has_3d
        else:
            return img, bbox_norm, eye_norm, gazex_norm, gazey_norm, torch.tensor(inout), height, width, observer_expression_ids, torch.tensor(gaze_direction), gaze_point_expression_ids, seg_mask, is_face_crop_mode, gaze3d, has_3d

    def __len__(self):
        return len(self.data_idxs)


# [NEW] Gaze Estimation Dataset Class
class GazeEstimationDataset(torch.utils.data.dataset.Dataset):
    def __init__(self, label_path, image_root, transform, is_train=True):
        self.transform = transform
        self.image_root = image_root
        self.is_train = is_train 
        self.lines = []
        self.decode_function = Get_Decode(label_path)
        self.dataset_name = label_path.split('/')[-3] if len(label_path.split('/')) > 3 else "unknown_est"
        self.expr_tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        self.expr_tokenizer.pad_token = self.expr_tokenizer.eos_token

        if os.path.isdir(label_path):
            files = glob.glob(os.path.join(label_path, '*.label')) + glob.glob(os.path.join(label_path, '*.txt'))
        else:
            files = [label_path]
        for f_path in files:
            with open(f_path, 'r') as f:
                content = f.readlines()[1:] 
                self.lines.extend([l.strip().split() for l in content])
        print(f"Loaded GazeEstimation Dataset ({self.dataset_name}): {len(self.lines)} samples.")
        
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
        if image is None: return self.__getitem__(random.randint(0, len(self.lines)-1))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        yaw = anno.gaze[0]; pitch = anno.gaze[1]
        
        pil_img = Image.fromarray(image)
        if self.is_train:
            if random.random() < 0.5: pil_img = self.color_jitter(pil_img)
            pil_img = self.bbox_jitter(pil_img)
            if random.random() < 0.5:
                pil_img = F_vis.hflip(pil_img)
                yaw = -yaw 
        
        img = pil_img
        width, height = img.size

        # Construct Aligned Outputs
        bbox_norm = [0.0, 0.0, 1.0, 1.0] 
        eye_norm = [0.5, 0.5] 
        gazex_norm = [-1.0]; gazey_norm = [-1.0] 
        inout = 0
        heatmap = torch.zeros((64, 64)) # Valid label: target is outside
        seg_mask = torch.zeros((64, 64)).float() # Valid label: no object
        gaze_direction = -100 # No 2D direction label
        is_face_crop_mode = True 
        
        observer_expression = np.random.choice([
            "a close-up of a human face", "the face of a person",
            "head pose and gaze direction", "a cropped face image"
        ])
        gaze_point_expression = "outside of image" # Valid label

        observer_encoded = self.expr_tokenizer(observer_expression, max_length=25, padding="max_length", truncation=True, return_tensors="pt")
        observer_expression_ids = observer_encoded['input_ids'].squeeze(0)

        gaze_point_encoded = self.expr_tokenizer(gaze_point_expression, max_length=25, padding="max_length", truncation=True, return_tensors="pt")
        gaze_point_expression_ids = gaze_point_encoded['input_ids'].squeeze(0)

        if self.transform: img = self.transform(img)

        # [NEW] Gaze3D & Flag
        gaze3d = torch.tensor([yaw, pitch], dtype=torch.float32)
        has_3d = torch.tensor(1).long() # Flag: 1 means has valid 3D label

        return (
            img, bbox_norm, eye_norm, gazex_norm, gazey_norm, torch.tensor(inout), height, width, 
            heatmap, observer_expression_ids, torch.tensor(gaze_direction), gaze_point_expression_ids, 
            seg_mask, is_face_crop_mode, gaze3d, has_3d
        )

def collate_fn(batch):
    transposed = list(zip(*batch))
    return tuple(
        torch.stack(items) if isinstance(items[0], torch.Tensor) else list(items)
        for items in transposed
    )