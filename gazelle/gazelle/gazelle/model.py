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
    def __init__(self, backbone, inout=False, dim=256, in_size=(448, 448), out_size=(64, 64), is_multi_output=False):
        super().__init__()
        self.backbone = backbone
        self.dim = dim
        self.in_size = in_size
        self.out_size = out_size
        self.inout = inout
        
        self.is_sam = getattr(backbone, 'is_sam', False)
        assert self.is_sam

        # 1. 基础投影
        self.linear = nn.Conv2d(backbone.get_dimension(), self.dim, 1)

        # =========================================================
        # 【关键修复】: 引入额外的 Self-Attention 层 (借用 Step 1 的成功经验)
        # =========================================================
        # 这层 Block 负责让图像像素之间进行全局交流，推断视线方向
        # 我们插入 1 到 2 层标准的 ViT Block
        self.pre_fusion_transformer = nn.Sequential(
            Block(dim=self.dim, num_heads=8, mlp_ratio=4, drop_path=0.1),
            Block(dim=self.dim, num_heads=8, mlp_ratio=4, drop_path=0.1)
        )
        # 为这个额外的 Transformer 准备位置编码
        self.featmap_h, self.featmap_w = backbone.get_out_size(in_size)
        self.register_buffer("extra_pos_embed", positionalencoding2d(self.dim, self.featmap_h, self.featmap_w))


        # 2. SAM Tokens 定义 (保持不变)
        self.num_output_tokens = 2 if self.inout else 1
        self.output_tokens = nn.Embedding(self.num_output_tokens, self.dim)

        self.idx_heatmap = 0
        self.idx_inout = 1 if inout else -1

        if self.inout:
             self.inout_head = nn.Sequential(
                nn.Linear(self.dim, 128), nn.ReLU(), nn.Dropout(0.1),
                nn.Linear(128, 1), nn.Sigmoid()
            )

    def forward(self, input):
        num_ppl_per_img = [len(bbox_list) for bbox_list in input["bboxes"]]
        
        # 1. Image Features
        x = self.backbone.forward(input["images"])
        if x.shape[1] != self.dim: x = self.linear(x)
        
        # [B, C, H, W] -> [Total_People, C, H, W]
        x = utils.repeat_tensors(x, num_ppl_per_img) 

        # =========================================================
        # 【关键修复实施】: 在进入 SAM Fusion 之前，先跑一遍 Self-Attention
        # =========================================================
        # A. 加上位置编码 (这对 Gaze 极其重要)
        x = x + self.extra_pos_embed
        
        # B. Flatten 准备进入 Transformer: [Total, H*W, C]
        x_flat = x.flatten(start_dim=2).permute(0, 2, 1)
        
        # C. 执行全局 Self-Attention
        # 这一步让眼睛的特征能传播到视线落点区域
        x_flat = self.pre_fusion_transformer(x_flat)
        
        # D. 还原回空间维度供 SAM 使用
        x = x_flat.permute(0, 2, 1).reshape(x.shape[0], self.dim, self.featmap_h, self.featmap_w)

        # ---------------------------------------------------------
        # 以下是标准的 SAM 流程 (Step 3)
        # ---------------------------------------------------------

        # 2. Prompt Encoder
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
            flat_eyes.append([eye[0] * self.in_size[1], eye[1] * self.in_size[0]])
        
        bboxes_tensor = torch.tensor(flat_bboxes, device=x.device, dtype=torch.float32)
        eyes_tensor = torch.tensor(flat_eyes, device=x.device, dtype=torch.float32)
        
        sparse_embeddings = self.backbone.prompt_encoder(
            bboxes_tensor, device=x.device, eyes=eyes_tensor, 
            expr_ids=input.get("observer_expression_ids", None)
        )
        
        tokens = self.output_tokens.weight.unsqueeze(0).repeat(x.shape[0], 1, 1)
        tokens = torch.cat((tokens, sparse_embeddings), dim=1)

        # 3. Two-Way Fusion
        # 此时输入的 x 已经包含了全局上下文信息，SAM Fusion 只需要负责“筛选”和“解码”
        hs, src = self.backbone.fusion(image_embeddings=x, tokens=tokens)

        # 4. 解码 (MLP + Dot Product)
        heatmap_token = hs[:, self.idx_heatmap, :] 
        src_upscaled = self.backbone.fusion.output_upscaling(src)
        
        heatmap_mlp = self.backbone.fusion.output_hypernetworks_mlps[0] 
        hyper_weight = heatmap_mlp(heatmap_token)
        
        b, c, h, w = src_upscaled.shape
        masks = (hyper_weight.unsqueeze(1) @ src_upscaled.view(b, c, h * w))
        masks = masks.view(b, 1, h, w)
        
        if masks.shape[-2:] != self.out_size:
            masks = F.interpolate(masks, size=self.out_size, mode='bilinear', align_corners=False)
        
        heatmap = torch.sigmoid(masks).squeeze(1)
        heatmap_preds = utils.split_tensors(heatmap, num_ppl_per_img)

        inout_preds = None
        if self.inout:
             inout_token_out = hs[:, self.idx_inout, :]
             inout_preds_flat = self.inout_head(inout_token_out).squeeze(dim=-1)
             inout_preds = utils.split_tensors(inout_preds_flat, num_ppl_per_img)

        return {"heatmap": heatmap_preds, "inout": inout_preds, 
                "seg": None, "direction": None, "text_loss": None}

# 辅助函数: Positional Encoding (如果之前未定义)
import math
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
