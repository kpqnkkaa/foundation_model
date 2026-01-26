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

        self.featmap_h, self.featmap_w = backbone.get_out_size(in_size)
        
        self.in_size = in_size
        self.out_size = out_size
        self.inout = inout
        
        # 必须是 SAM Backbone 才有 fusion 模块
        self.is_sam = getattr(backbone, 'is_sam', False)
        assert self.is_sam, "Step 2 需要使用 SAM Backbone 及其 Fusion 模块"

        # 基础投影
        self.linear = nn.Conv2d(backbone.get_dimension(), self.dim, 1)
        self.register_buffer("pos_embed", positionalencoding2d(self.dim, self.featmap_h, self.featmap_w))

        # ====================================================
        # Step 2 修改：恢复 SAM 的 Tokens 定义，但保持 Deconv 输出头
        # ====================================================

        # 1. 恢复 Output Tokens (Learnable Queries)
        # 因为 SAM 的 Fusion 需要 (Image, Tokens) 作为输入
        # 我们这里暂时只初始化 1 个 token (heatmap)，忽略 inout 的复杂情况以简化消融
        self.num_output_tokens = 1 
        self.output_tokens = nn.Embedding(self.num_output_tokens, self.dim)
        
        # 2. 【移除】Standard Transformer
        # self.transformer = ... (已移除)

        # 3. 【保持】Standard Deconv Head
        # 我们尝试直接解码 Fusion 输出的 Image Features (src)
        self.heatmap_head = nn.Sequential(
            nn.ConvTranspose2d(dim, dim, kernel_size=2, stride=2),
            nn.Conv2d(dim, 1, kernel_size=1, bias=False),
            nn.Sigmoid()
        )
        
        # Inout Head (如果需要测试 inout 任务)
        if self.inout: 
             self.inout_head = nn.Sequential(
                nn.Linear(self.dim, 128),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(128, 1),
                nn.Sigmoid()
            )

    def forward(self, input):
        num_ppl_per_img = [len(bbox_list) for bbox_list in input["bboxes"]]
        
        # 1. 提取基础图像特征
        x = self.backbone.forward(input["images"])
        
        # 投影并加上位置编码 (SAM Fusion 内部通常也会加位置编码，但这里为了控制变量，我们先在外面处理好维度)
        if x.shape[1] != self.dim:
            x = self.linear(x)
        
        # 扩展到 Person 维度 [Total_People, dim, H, W]
        x = utils.repeat_tensors(x, num_ppl_per_img)
        
        # ==========================================
        #       Step 2: 准备 SAM Tokens & Fusion
        # ==========================================
        
        # A. 准备 Prompts (Step 1 验证过没问题)
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
        
        # B. Sparse Embeddings
        sparse_embeddings = self.backbone.prompt_encoder(
            bboxes_tensor, device=x.device, eyes=eyes_tensor, 
            expr_ids=input.get("observer_expression_ids", None)
        )
        
        # C. 拼接 Tokens: [Learnable_Token, Prompt_Tokens]
        # output_tokens: [1, 1, dim] -> [Total_People, 1, dim]
        tokens = self.output_tokens.weight.unsqueeze(0).repeat(x.shape[0], 1, 1)
        tokens = torch.cat((tokens, sparse_embeddings), dim=1)

        # D. 【关键点】Two-Way Fusion
        # 输入: x (Image), tokens (Prompts)
        # 输出: hs (更新后的Tokens), src (更新后的Image)
        # 这里的 src 经过了 "Image-to-Token" 的 Cross-Attention，理论上应该感知到了 Token 的位置信息
        hs, src = self.backbone.fusion(image_embeddings=x, tokens=tokens)

        # ==========================================
        #       Step 2: 使用 Deconv Head 解码
        # ==========================================
        
        # 我们这里做一个重要的假设测试：
        # 如果 Fusion 工作正常，src (图像特征) 应该被 tokens (Prompt) "激活" 了相关区域。
        # 我们直接把 src 扔进 Deconv 头，看看能不能解出 Heatmap。
        
        # src shape: [Total_People, dim, H, W]
        
        heatmap = self.heatmap_head(src)
        
        heatmap = torchvision.transforms.functional.resize(heatmap, self.out_size)
        heatmap = heatmap.squeeze(1)
        heatmap_preds = utils.split_tensors(heatmap, num_ppl_per_img)

        # InOut 逻辑 (简化处理，如果有的话)
        inout_preds = None
        if self.inout:
             # 简单地对 src 进行 Global Average Pooling 来预测 inout，或者使用 hs 里的 token
             # 为了避免引入复杂变量，这里可以使用 hs 的第一个 token (learnable token)
             inout_token_out = hs[:, 0, :]
             inout_preds_flat = self.inout_head(inout_token_out).squeeze(dim=-1)
             inout_preds = utils.split_tensors(inout_preds_flat, num_ppl_per_img)

        return {"heatmap": heatmap_preds, 
                "inout": inout_preds,
                "seg": None,
                "direction": None,
                "text_loss": None}

    # ... (辅助函数保持不变) ...
    def get_input_head_maps(self, bboxes):
        return [] # 不再使用
    
    def get_gazelle_state_dict(self):
        return self.state_dict()

    def load_gazelle_state_dict(self, ckpt_state_dict):
        current_state_dict = self.state_dict()
        keys1 = current_state_dict.keys()
        keys2 = ckpt_state_dict.keys()
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
