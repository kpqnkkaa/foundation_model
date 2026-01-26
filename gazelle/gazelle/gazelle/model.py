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
        self.in_size = in_size
        self.out_size = out_size
        self.inout = inout

        # 获取 Backbone 输出特征图尺寸
        self.featmap_h, self.featmap_w = backbone.get_out_size(in_size)
        
        # 1. 判定是否为 SAM Backbone (用于调用 prompt_encoder)
        self.is_sam = getattr(backbone, 'is_sam', False)

        # 2. 基础投影 (DINOv2 768 -> 256)
        self.linear = nn.Conv2d(backbone.get_dimension(), self.dim, 1)
        
        # 3. 位置编码 (Standard 方案必须显式加位置编码)
        self.register_buffer("pos_embed", positionalencoding2d(self.dim, self.featmap_h, self.featmap_w))

        # 4. Standard Transformer (核心交互层)
        # 这个 Transformer 接收 [Prompts, Image] 拼接序列
        # 它会自动完成: Prompt-to-Prompt (你想要的), Image-to-Image (修补SAM缺陷), Prompt-to-Image (交互)
        self.transformer = nn.Sequential(*[
            Block(
                dim=self.dim, 
                num_heads=8, 
                mlp_ratio=4, 
                drop_path=0.1)
            for i in range(num_layers)
        ])
        
        # 5. Standard Deconv Head (解码层)
        # 这种头最适合 Standard Transformer 输出的空间连续特征
        self.heatmap_head = nn.Sequential(
            nn.ConvTranspose2d(dim, dim, kernel_size=2, stride=2),
            nn.Conv2d(dim, 1, kernel_size=1, bias=False),
            nn.Sigmoid()
        )
        
        # InOut Token (如果有)
        if self.inout: 
            self.inout_token = nn.Embedding(1, self.dim)
            self.inout_head = nn.Sequential(
                nn.Linear(self.dim, 128), nn.ReLU(), nn.Dropout(0.1),
                nn.Linear(128, 1), nn.Sigmoid()
            )

    def forward(self, input):
        num_ppl_per_img = [len(bbox_list) for bbox_list in input["bboxes"]]
        
        # 1. 提取基础图像特征
        x = self.backbone.forward(input["images"])
        
        # 投影: [B, 256, H, W]
        if x.shape[1] != self.dim:
            x = self.linear(x) 
        
        # 2. 【关键】加上位置编码 (对 Standard Transformer 至关重要)
        x = x + self.pos_embed
        
        # 扩展到 Person 维度 [Total_People, 256, H, W]
        x = utils.repeat_tensors(x, num_ppl_per_img)

        # 3. 提取 Prompt Embeddings (利用 SAM 预训练的编码器)
        if hasattr(self.backbone, 'prompt_encoder'):
            # A. 坐标归一化转绝对坐标
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
            
            # B. 获取 Sparse Embeddings [Total_People, N_prompts, 256]
            sparse_embeddings = self.backbone.prompt_encoder(
                bboxes_tensor, 
                device=x.device, 
                eyes=eyes_tensor, 
                expr_ids=input.get("observer_expression_ids", None)
            )
        else:
            # Fallback (如果不应该发生)
            sparse_embeddings = torch.zeros(x.shape[0], 0, self.dim, device=x.device)

        # 4. 序列构建
        
        # Flatten Image: [Total, 256, H, W] -> [Total, H*W, 256]
        # 注意：这里我们要 permute 成 (B, Seq, Dim) 供 Transformer 使用
        x_flat = x.flatten(start_dim=2).permute(0, 2, 1)
        
        # 拼接: [Prompts, Image]
        # 这样 Transformer 就能让 Prompts 和 Image 充分交互
        x_input = torch.cat([sparse_embeddings, x_flat], dim=1)

        # 拼接 InOut Token
        if self.inout:
            inout_t = self.inout_token.weight.unsqueeze(dim=0).repeat(x_input.shape[0], 1, 1)
            x_input = torch.cat([inout_t, x_input], dim=1)
        
        # 5. Transformer 融合 (Standard Self-Attention)
        # 这里包含了你关心的 "Prompt Self-Fusion"，也包含了 "Image Context Reasoning"
        x_out = self.transformer(x_input)

        # 6. 解码
        current_idx = 0
        
        # A. InOut 解码
        inout_preds = None
        if self.inout:
            inout_tokens = x_out[:, 0, :] 
            inout_preds_flat = self.inout_head(inout_tokens).squeeze(dim=-1)
            inout_preds = utils.split_tensors(inout_preds_flat, num_ppl_per_img)
            current_idx += 1 
        
        # B. Heatmap 解码
        # 剥离掉 Prompt Tokens，只保留图像部分的 Tokens
        num_prompts = sparse_embeddings.shape[1]
        start_img_idx = current_idx + num_prompts
        
        x_img = x_out[:, start_img_idx:, :]
        
        # Reshape 回空间维度 [Total, 256, H, W]
        x_img = x_img.permute(0, 2, 1).reshape(x_img.shape[0], self.dim, self.featmap_h, self.featmap_w)
        
        # Standard Deconv Head
        heatmap = self.heatmap_head(x_img)
        
        # 后处理
        heatmap = torchvision.transforms.functional.resize(heatmap, self.out_size)
        heatmap = heatmap.squeeze(1)
        heatmap_preds = utils.split_tensors(heatmap, num_ppl_per_img)

        return {"heatmap": heatmap_preds, 
                "inout": inout_preds,
                "seg": None,
                "direction": None,
                "text_loss": None}

    # ... (辅助函数)
    def get_gazelle_state_dict(self):
        return self.state_dict()
    
    def load_gazelle_state_dict(self, ckpt_state_dict):
        # 标准加载逻辑
        current_state_dict = self.state_dict()
        keys1 = set(current_state_dict.keys())
        keys2 = set(ckpt_state_dict.keys())
        for k in list(keys1 & keys2):
            current_state_dict[k] = ckpt_state_dict[k]
        self.load_state_dict(current_state_dict, strict=False)

def positionalencoding2d(d_model, height, width):
    if d_model % 4 != 0:
        raise ValueError("Cannot use sin/cos positional encoding with odd dimension")
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
        "sam_sam_vitb": sam_sam_vitb,
        "sam_dinov2_vitb": sam_dinov2_vitb,
        "sam_dinov2_vitb_lora": sam_dinov2_vitb_lora,
        "sam_sam_vitb_lora": sam_sam_vitb_lora,
        "sam_dinov2_vitb_lora_multi_input": sam_dinov2_vitb_lora_multi_input,
        "sam_dinov2_vitb_lora_multi_output_input": sam_dinov2_vitb_lora_multi_output_input
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

def sam_dinov2_vitb():
    backbone = SAMBackboneWrapper(model_type="vit_b", in_size=(448, 448), backbone_type="dinov2", is_lora=False, is_multi_input=False)
    transform = backbone.get_transform((448, 448))
    model = GazeLLE(backbone, inout=False)
    return model, transform

def sam_dinov2_vitb_lora():
    backbone = SAMBackboneWrapper(model_type="vit_b", in_size=(448, 448), backbone_type="dinov2", is_lora=True, is_multi_input=False)
    transform = backbone.get_transform((448, 448))
    model = GazeLLE(backbone, inout=False)
    return model, transform

def sam_sam_vitb_lora():
    backbone = SAMBackboneWrapper(model_type="vit_b", in_size=(448, 448), backbone_type="sam", is_lora=True, is_multi_input=False)
    transform = backbone.get_transform((448, 448))
    model = GazeLLE(backbone, inout=False)
    return model, transform

def sam_sam_vitb():
    backbone = SAMBackboneWrapper(model_type="vit_b", in_size=(448, 448), backbone_type="sam", is_lora=False, is_multi_input=False)
    transform = backbone.get_transform((448, 448))
    model = GazeLLE(backbone, inout=False)
    return model, transform

def sam_dinov2_vitb_lora_multi_input():
    backbone = SAMBackboneWrapper(model_type="vit_b", in_size=(448, 448), backbone_type="dinov2", is_lora=True, is_multi_input=True)
    transform = backbone.get_transform((448, 448))
    model = GazeLLE(backbone, inout=False)
    return model, transform
    
def sam_dinov2_vitb_lora_multi_output_input():
    backbone = SAMBackboneWrapper(model_type="vit_b", in_size=(448, 448), backbone_type="dinov2", is_lora=True, is_multi_input=True)
    transform = backbone.get_transform((448, 448))
    model = GazeLLE(backbone, inout=False, is_multi_output=True)
    return model, transform
