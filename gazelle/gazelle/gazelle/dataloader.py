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
        
        # [Mix Gaze Estimation Flag]
        self.is_mix_gaze_estimation = is_mix_gaze_estimation and self.split == "train"
        # [Partial Input Flag]
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

        # 1. Prepare Text (Observer Expression)
        observer_expression_list = head_data.get('observer_expression')
        if observer_expression_list and isinstance(observer_expression_list, list) and len(observer_expression_list) > 0:
            observer_expression = np.random.choice(observer_expression_list)
        else:
            observer_expression = ""
            
        # 2. Prepare Gaze Point Text
        gaze_point_expressions_list = head_data.get('gaze_point_expressions')
        if gaze_point_expressions_list and isinstance(gaze_point_expressions_list, list) and len(gaze_point_expressions_list) > 0:
            gaze_point_expression = np.random.choice(gaze_point_expressions_list)
        else:
            gaze_point_expression = ""

        # 3. Prepare Direction
        direction_map = {
            "right": 0, "top-right": 1, "above": 2, "top-left": 3,
            "left": 4, "bottom-left": 5, "below": 6, "bottom-right": 7
        }
        gaze_dir_str = head_data.get('gaze_direction')
        if gaze_dir_str in direction_map:
            gaze_direction = direction_map[gaze_dir_str]
        else:
            gaze_direction = -1
        
        # 4. Prepare Seg Mask
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
            # === Mix Gaze Estimation Logic ===
            if self.is_mix_gaze_estimation and np.random.sample() <= 0.2:
                is_face_crop_mode = True
                
                raw_bbox = head_data['bbox'] # [xmin, ymin, xmax, ymax]
                scale = np.random.uniform(0.8, 1.5)
                
                b_w = raw_bbox[2] - raw_bbox[0]
                b_h = raw_bbox[3] - raw_bbox[1]
                cx = raw_bbox[0] + b_w / 2
                cy = raw_bbox[1] + b_h / 2
                
                new_w = b_w * scale
                new_h = b_h * scale
                
                crop_x1 = max(0, int(cx - new_w / 2))
                crop_y1 = max(0, int(cy - new_h / 2))
                crop_x2 = min(width, int(cx + new_w / 2))
                crop_y2 = min(height, int(cy + new_h / 2))
                
                img = img.crop((crop_x1, crop_y1, crop_x2, crop_y2))
                width, height = img.size 
                
                # Update BBox Norm: Full image
                bbox_norm = [0.0, 0.0, 1.0, 1.0]
                
                # Update Eyes: Approx center
                if eye_norm is not None:
                    eye_norm = [0.5, 0.5]
                
                # Update Text: Face description
                observer_expression = np.random.choice([
                    "a close-up of a human face", "the face of a person",
                    "head pose and gaze direction", "a cropped face image"
                ])

                # [Modified] Update Outputs for Gaze Estimation Mode
                # Seg Mask -> All Zeros
                seg_mask = torch.zeros((64, 64)).float()
                
                # Gaze Point Text -> "outside of image"
                gaze_point_expression = "outside of image"

            else:
                # === Standard Augmentation ===
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

        # === Partial Input Logic (Prompt Dropout) ===
        if self.is_partial_input:
            if self.split == "train":
                # 默认全部保留
                use_text = True
                use_bbox = True
                use_eye = True
                
                # 判断文本是否本身有效
                text_is_reliable = (len(observer_expression) > 0)
                
                # 第一层随机：决定 模态数量 (1个 vs 2个 vs 3个)
                r = random.random() 

                if text_is_reliable:
                    # === 情况 A: 文本可靠 (所有组合都可用) ===
                    if r < 0.5:
                        # === 1. 单模态 (Single) ===
                        r_sub = random.random()
                        if r_sub < 0.333:
                            use_bbox = False; use_eye = False # [纯文本]
                        elif r_sub < 0.666:
                            use_text = False; use_eye = False # [纯框]
                        else:
                            use_text = False; use_bbox = False # [纯点]
                            
                    elif r < 0.8:
                        # === 2. 双模态组合 (Pair) ===
                        r_sub = random.random()
                        if r_sub < 0.333:
                            use_eye = False # [文本 + 框]
                        elif r_sub < 0.666:
                            use_bbox = False # [文本 + 点]
                        else:
                            use_text = False # [框 + 点]
                    else:
                        pass # === 3. 全模态 (All) ===

                else:
                    # === 情况 B: 文本本来就是空的 ===
                    use_text = False 
                    r_geo = random.random()
                    if r_geo < 0.5:
                        # === 单几何 (Single) ===
                        if random.random() < 0.5:
                            use_eye = False # [纯框]
                        else:
                            use_bbox = False # [纯点]
                    else:
                        pass # [框 + 点] 保留两者
            else:
                use_text = False
                use_eye = False
                use_bbox = True

            # Apply Dropout Results (Set to Placeholders)
            if not use_bbox:
                # 占位符: 全0或全-1, 视后续模型处理而定，这里设为 [0,0,0,0] 表示无效框
                bbox_norm = [0.0, 0.0, 0.0, 0.0] 
            
            if not use_eye:
                # 占位符: [-1, -1]
                eye_norm = [-1.0, -1.0] 
            
            if not use_text:
                # 占位符: 空字符串
                observer_expression = ""


        # === Tokenization after prompt modification ===
        # 1. Observer Expression
        if observer_expression:
            observer_encoded = self.expr_tokenizer(observer_expression, max_length=25, padding="max_length", truncation=True, return_tensors="pt")
            observer_expression_ids = observer_encoded['input_ids'].squeeze(0)
        else:
            observer_expression_ids = torch.full((25,), self.expr_tokenizer.pad_token_id, dtype=torch.long)

        # 2. Gaze Point Expression
        if gaze_point_expression:
            gaze_point_encoded = self.expr_tokenizer(gaze_point_expression, max_length=25, padding="max_length", truncation=True, return_tensors="pt")
            gaze_point_expression_ids = gaze_point_encoded['input_ids'].squeeze(0)
        else:
            gaze_point_expression_ids = torch.full((25,), self.expr_tokenizer.pad_token_id, dtype=torch.long)

        img = self.transform(img)
        
        if self.split == "train":
            # [Modified] Heatmap Logic
            if is_face_crop_mode:
                heatmap = torch.zeros((64, 64))
            else:
                heatmap = utils.get_heatmap(gazex_norm[0], gazey_norm[0], 64, 64) 
            
            return img, bbox_norm, eye_norm, gazex_norm, gazey_norm, torch.tensor(inout), height, width, heatmap, observer_expression_ids, torch.tensor(gaze_direction), gaze_point_expression_ids, seg_mask, is_face_crop_mode
        else:
            return img, bbox_norm, eye_norm, gazex_norm, gazey_norm, torch.tensor(inout), height, width, observer_expression_ids, torch.tensor(gaze_direction), gaze_point_expression_ids, seg_mask, is_face_crop_mode

    def __len__(self):
        return len(self.data_idxs)


def collate_fn(batch):
    transposed = list(zip(*batch))
    return tuple(
        torch.stack(items) if isinstance(items[0], torch.Tensor) else list(items)
        for items in transposed
    )