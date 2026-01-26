import torch
import torch.nn as nn
import torchvision
from timm.models.vision_transformer import Block
import math
import os
from gazelle.backbone import DinoV2Backbone, SAMBackboneWrapper, SAMImageEncoder, GazeTextDecoder
import gazelle.utils as utils
import torch.nn.functional as F

class GazeLLE(nn.Module):
    def __init__(self, backbone, inout=False, dim=256, num_layers=3, in_size=(448, 448), out_size=(64, 64), is_multi_output=False):
        super().__init__()
        self.backbone = backbone
        self.dim = dim
        self.num_layers = num_layers
        self.is_multi_output = is_multi_output
        self.inout = inout
        self.in_size = in_size
        self.out_size = out_size

        # 获取 Backbone 输出特征图尺寸
        self.featmap_h, self.featmap_w = backbone.get_out_size(in_size)

        # 1. 检查能力: 是否支持 Prompt Encoder (即是否为 SAM 系列 Backbone)
        self.has_prompt_encoder = hasattr(backbone, 'prompt_encoder')

        # 2. 基础组件 (所有模式共用)
        # 投影层: 适配 Backbone 维度到 Model 维度
        self.linear = nn.Conv2d(backbone.get_dimension(), self.dim, 1)
        
        # 位置编码: 关键！Standard Transformer 必须显式注入位置信息
        self.register_buffer("pos_embed", positionalencoding2d(self.dim, self.featmap_h, self.featmap_w))

        # 3. 定义特殊 Tokens (Learnable Embeddings)
        # 无论是不是 SAM，只要有对应任务，就初始化对应的 Token
        if self.inout:
            self.inout_token = nn.Embedding(1, self.dim)
        
        if self.is_multi_output:
            # 顺序: [Seg, Direction, Text]
            self.multi_output_tokens = nn.Embedding(3, self.dim)

        # Legacy Support: 如果不是 SAM Backbone，还需要旧的 Head Token
        if not self.has_prompt_encoder:
            self.head_token = nn.Embedding(1, self.dim)

        # 4. 核心融合层: Standard Transformer (Self-Attention)
        # 替代了 SAM 的 Two-Way Transformer
        self.transformer = nn.Sequential(*[
            Block(
                dim=self.dim, 
                num_heads=8, 
                mlp_ratio=4, 
                drop_path=0.1)
            for i in range(num_layers)
        ])

        # 5. 输出头 (Heads)
        
        # A. Gaze Heatmap Head (Standard Deconv)
        self.heatmap_head = nn.Sequential(
            nn.ConvTranspose2d(dim, dim, kernel_size=2, stride=2),
            nn.Conv2d(dim, 1, kernel_size=1, bias=False),
            nn.Sigmoid()
        )

        # B. InOut Head
        if self.inout:
             self.inout_head = nn.Sequential(
                nn.Linear(self.dim, 128), nn.ReLU(), nn.Dropout(0.1),
                nn.Linear(128, 1), nn.Sigmoid()
            )

        # C. Multi-Output Heads
        if self.is_multi_output:
            # Segmentation Head (也是一个 Deconv，结构同 Heatmap)
            self.seg_head = nn.Sequential(
                nn.ConvTranspose2d(dim, dim, kernel_size=2, stride=2),
                nn.Conv2d(dim, 1, kernel_size=1, bias=False),
                nn.Sigmoid()
            )
            # Direction Head (8分类)
            self.direction_head = nn.Sequential(
                nn.Linear(self.dim, 128), nn.ReLU(), nn.Dropout(0.1),
                nn.Linear(128, 8)
            )
            # Text Head (GPT2 Decoder)
            self.text_head = GazeTextDecoder(input_dim=self.dim, model_name="gpt2", lora_r=8, max_len=25)

    def forward(self, input):
        # input["images"]: [B, 3, H, W]
        num_ppl_per_img = [len(bbox_list) for bbox_list in input["bboxes"]]
        
        # 1. 提取基础图像特征
        x = self.backbone.forward(input["images"]) # [B, C, H, W]
        
        # 投影到 256 维
        if x.shape[1] != self.dim:
            x = self.linear(x)
            
        # 2. 【关键步骤】注入位置编码
        # 在 Flatten 之前加，确保空间信息被编码进特征值
        x = x + self.pos_embed
        
        # 扩展到 Person 维度 [Total_People, dim, H, W]
        x = utils.repeat_tensors(x, num_ppl_per_img)

        # 3. 准备各种 Tokens
        tokens_to_concat = []

        # [A] 特殊任务 Tokens
        if self.inout:
            # [Total, 1, dim]
            inout_t = self.inout_token.weight.unsqueeze(0).repeat(x.shape[0], 1, 1)
            tokens_to_concat.append(inout_t)
        
        if self.is_multi_output:
            # [Total, 3, dim]
            multi_t = self.multi_output_tokens.weight.unsqueeze(0).repeat(x.shape[0], 1, 1)
            tokens_to_concat.append(multi_t)

        # [B] Prompt Embeddings (SAM 逻辑)
        if self.has_prompt_encoder:
            # 坐标转换: Normalized -> Absolute
            flat_bboxes = []
            for bbox_list in input["bboxes"]:
                for bbox in bbox_list:
                    xmin, ymin, xmax, ymax = bbox
                    flat_bboxes.append([
                        xmin * self.in_size[1], ymin * self.in_size[0], 
                        xmax * self.in_size[1], ymax * self.in_size[0]
                    ])
            flat_eyes = []
            for eye in input["eyes"]:
                flat_eyes.append([
                    eye[0] * self.in_size[1], eye[1] * self.in_size[0]
                ])
            
            bboxes_tensor = torch.tensor(flat_bboxes, device=x.device, dtype=torch.float32)
            eyes_tensor = torch.tensor(flat_eyes, device=x.device, dtype=torch.float32)
            
            # 提取 Prompt 特征 [Total, N_prompts, dim]
            sparse_embeddings = self.backbone.prompt_encoder(
                bboxes_tensor, device=x.device, eyes=eyes_tensor, 
                expr_ids=input.get("observer_expression_ids", None)
            )
            tokens_to_concat.append(sparse_embeddings)
        
        # [C] Legacy Head Map Logic (如果不是 SAM Backbone)
        else:
            # 这种情况下，直接把 head map 加到图像特征里，不作为额外的 token 拼接
            # (保持原有的非 SAM 逻辑一致性)
            head_maps = torch.cat(self.get_input_head_maps(input["bboxes"]), dim=0).to(x.device)
            head_map_embeddings = head_maps.unsqueeze(dim=1) * self.head_token.weight.unsqueeze(-1).unsqueeze(-1)
            x = x + head_map_embeddings
            # sparse_embeddings 为空，用于后续计算索引
            sparse_embeddings = torch.zeros(x.shape[0], 0, self.dim, device=x.device)

        # 4. 序列构建与融合
        # Flatten Image: [Total, dim, H, W] -> [Total, H*W, dim]
        x_flat = x.flatten(start_dim=2).permute(0, 2, 1)
        
        # 拼接所有 Token: [Special..., Prompts..., Image...]
        # Image 放在最后
        tokens_to_concat.append(x_flat)
        x_input = torch.cat(tokens_to_concat, dim=1)
        
        # 5. Standard Transformer 全局交互
        x_out = self.transformer(x_input)

        # 6. 解码与输出 (根据索引切片)
        
        current_idx = 0
        output_dict = {
            "heatmap": None, "inout": None, 
            "seg": None, "direction": None, "text_loss": None
        }

        # [Decode] InOut
        if self.inout:
            inout_token_vec = x_out[:, current_idx, :]
            inout_val = self.inout_head(inout_token_vec).squeeze(dim=-1)
            output_dict["inout"] = utils.split_tensors(inout_val, num_ppl_per_img)
            current_idx += 1

        # [Decode] Multi-Output
        if self.is_multi_output:
            # tokens: 0=Seg, 1=Dir, 2=Text
            seg_token_vec = x_out[:, current_idx, :]
            dir_token_vec = x_out[:, current_idx+1, :]
            text_token_vec = x_out[:, current_idx+2, :]
            current_idx += 3
            
            # Direction
            dir_val = self.direction_head(dir_token_vec)
            output_dict["direction"] = utils.split_tensors(dir_val, num_ppl_per_img)
            
            # Text
            output_dict["text_loss"] = self.text_head(
                fusion_feat=text_token_vec, 
                target_ids=input.get("gaze_point_expression_ids")
            )
            
            # Seg (Pass to image decoding phase or decode here if strictly token based? 
            # In standard ViT, dense pred usually comes from image features, 
            # but we can condition it. For simplicity, we use separate head on image features below)
            pass 

        # [Skip] Prompts
        # 如果是 SAM backbone, 跳过 sparse_embeddings 的长度
        if self.has_prompt_encoder:
            num_prompts = sparse_embeddings.shape[1]
            current_idx += num_prompts

        # [Decode] Image Features -> Heatmap / Seg
        # 剩下的全是 Image Tokens
        x_img = x_out[:, current_idx:, :]
        
        # Reshape: [Total, H*W, dim] -> [Total, dim, H, W]
        x_img = x_img.permute(0, 2, 1).reshape(x_img.shape[0], self.dim, self.featmap_h, self.featmap_w)
        
        # Gaze Heatmap
        heatmap = self.heatmap_head(x_img)
        heatmap = torchvision.transforms.functional.resize(heatmap, self.out_size)
        heatmap = heatmap.squeeze(1)
        output_dict["heatmap"] = utils.split_tensors(heatmap, num_ppl_per_img)
        
        # Seg Heatmap (if multi-output)
        if self.is_multi_output:
            # 使用相同的图像特征，但通过不同的头生成分割图
            seg_map = self.seg_head(x_img)
            seg_map = torchvision.transforms.functional.resize(seg_map, self.out_size)
            seg_map = seg_map.squeeze(1)
            output_dict["seg"] = utils.split_tensors(seg_map, num_ppl_per_img)

        return output_dict

    # 保持辅助函数兼容性
    def get_input_head_maps(self, bboxes):
        head_maps = []
        for bbox_list in bboxes:
            img_head_maps = []
            for bbox in bbox_list:
                if bbox is None:
                    img_head_maps.append(torch.zeros(self.featmap_h, self.featmap_w))
                else:
                    xmin, ymin, xmax, ymax = bbox
                    width, height = self.featmap_w, self.featmap_h
                    xmin = round(xmin * width); ymin = round(ymin * height)
                    xmax = round(xmax * width); ymax = round(ymax * height)
                    head_map = torch.zeros((height, width))
                    # Clamp coordinates
                    xmin = max(0, xmin); ymin = max(0, ymin)
                    xmax = min(width, xmax); ymax = min(height, ymax)
                    if xmax > xmin and ymax > ymin:
                        head_map[ymin:ymax, xmin:xmax] = 1
                    img_head_maps.append(head_map)
            head_maps.append(torch.stack(img_head_maps))
        return head_maps
    
    def get_gazelle_state_dict(self):
        return self.state_dict()

    def load_gazelle_state_dict(self, ckpt_state_dict):
        current_state_dict = self.state_dict()
        keys1 = set(current_state_dict.keys())
        keys2 = set(ckpt_state_dict.keys())
        for k in list(keys1 & keys2):
            current_state_dict[k] = ckpt_state_dict[k]
        self.load_state_dict(current_state_dict, strict=False)

# ... (positionalencoding2d 和 factory functions 保持不变) ...
# 为了完整性，这里列出 positionalencoding2d
def positionalencoding2d(d_model, height, width):
    if d_model % 4 != 0: raise ValueError("Odd dimension")
    pe = torch.zeros(d_model, height, width)
    d_model = int(d_model / 2)
    div_term = torch.exp(torch.arange(0., d_model, 2) * -(math.log(10000.0) / d_model))
    pos_w = torch.arange(0., width).unsqueeze(1)
    pos_h = torch.arange(0., height).unsqueeze(1)
    pe[0:d_model:2, :, :] = torch.sin(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, height, 1)
    pe[1:d_model:2, :, :] = torch.cos(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, height, 1)
    pe[d_model::2, :, :] = torch.sin(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, width)
    pe[d_model + 1::2, :, :] = torch.cos(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, width)
    return pe

# models
def get_gazelle_model(model_name):
    factory = {
        "gazelle_dinov2_vitb14": gazelle_dinov2_vitb14,
        "gazelle_dinov2_vitl14": gazelle_dinov2_vitl14,
        "gazelle_dinov2_vitb14_inout": gazelle_dinov2_vitb14_inout,
        "gazelle_dinov2_vitl14_inout": gazelle_dinov2_vitl14_inout,
        # 新增sam模型
        "gazelle_sam_vitb": gazelle_sam_vitb,
        "gazelle_dinov2_vitb_lora": gazelle_dinov2_vitb_lora,
        # "sam_prompt_dinov2_vitb": sam_prompt_dinov2_vitb,
        "sam_prompt_dinov2_vitb_lora_multi_input": sam_prompt_dinov2_vitb_lora_multi_input,
        "dinov2_vitb_lora_multi_output": dinov2_vitb_lora_multi_output
    }
    assert model_name in factory.keys(), "invalid model name"
    return factory[model_name]()

def gazelle_dinov2_vitb14():
    backbone = DinoV2Backbone('dinov2_vitb14')
    transform = backbone.get_transform((448, 448))
    model = GazeLLE(backbone)
    return model, transform

def gazelle_dinov2_vitl14():
    backbone = DinoV2Backbone('dinov2_vitl14')
    transform = backbone.get_transform((448, 448))
    model = GazeLLE(backbone)
    return model, transform

def gazelle_dinov2_vitb14_inout():
    backbone = DinoV2Backbone('dinov2_vitb14')
    transform = backbone.get_transform((448, 448))
    model = GazeLLE(backbone, inout=True)
    return model, transform

def gazelle_dinov2_vitl14_inout():
    backbone = DinoV2Backbone('dinov2_vitl14')
    transform = backbone.get_transform((448, 448))
    model = GazeLLE(backbone, inout=True)
    return model, transform

def gazelle_sam_vitb():
    backbone = SAMImageEncoder(model_type="vit_b", in_size=(448, 448))
    transform = backbone.get_transform((448, 448))
    model = GazeLLE(backbone, inout=False)
    return model, transform

def gazelle_dinov2_vitb_lora():
    backbone = DinoV2Backbone('dinov2_vitb14', is_lora=True)
    transform = backbone.get_transform((448, 448))
    model = GazeLLE(backbone, inout=False)
    return model, transform

def sam_prompt_dinov2_vitb():
    backbone = SAMBackboneWrapper(model_type="vit_b", in_size=(448, 448), backbone_type="dinov2", is_lora=False, is_multi_input=False)
    transform = backbone.get_transform((448, 448))
    model = GazeLLE(backbone, inout=False, is_sam_prompt=True)
    return model, transform

def sam_prompt_dinov2_vitb_lora():
    backbone = SAMBackboneWrapper(model_type="vit_b", in_size=(448, 448), backbone_type="dinov2", is_lora=True, is_multi_input=False)
    transform = backbone.get_transform((448, 448))
    model = GazeLLE(backbone, inout=False, is_sam_prompt=True)
    return model, transform

def sam_prompt_dinov2_vitb_lora_multi_input():
    backbone = SAMBackboneWrapper(model_type="vit_b", in_size=(448, 448), backbone_type="sam", is_lora=True, is_multi_input=True)
    transform = backbone.get_transform((448, 448))
    model = GazeLLE(backbone, inout=False, is_sam_prompt=True)
    return model, transform
    
def sam_prompt_dinov2_vitb_lora_multi_output_input():
    backbone = SAMBackboneWrapper(model_type="vit_b", in_size=(448, 448), backbone_type="dinov2", is_lora=True, is_multi_input=True, is_sam_prompt=True)
    transform = backbone.get_transform((448, 448))
    model = GazeLLE(backbone, inout=False, is_multi_output=True)
    return model, transform
