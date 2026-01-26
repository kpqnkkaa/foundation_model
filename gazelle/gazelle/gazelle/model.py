import torch
import torch.nn as nn
import torchvision
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
        
        # 必须确保是 SAM Backbone 且包含 fusion 模块
        self.is_sam = getattr(backbone, 'is_sam', False)
        assert self.is_sam, "Step 3 strictly requires SAM Backbone"

        # 基础投影
        self.linear = nn.Conv2d(backbone.get_dimension(), self.dim, 1)
        
        # 这里不需要显式 self.pos_embed，因为 SAMFusion 内部处理了

        # ==========================================
        # Step 3: 完整的 SAM Token 系统
        # ==========================================
        
        # 定义 Token 索引指针
        self.idx_heatmap = 0
        self.idx_inout = 1 if inout else -1
        
        # 初始化 Tokens: [Heatmap, (Optional InOut)]
        self.num_output_tokens = 2 if self.inout else 1
        self.output_tokens = nn.Embedding(self.num_output_tokens, self.dim)

        # InOut 分类头 (基于 Token)
        if self.inout:
             self.inout_head = nn.Sequential(
                nn.Linear(self.dim, 128), nn.ReLU(), nn.Dropout(0.1),
                nn.Linear(128, 1), nn.Sigmoid()
            )

        # 注意：不再定义 self.heatmap_head (Deconv)
        # 我们将直接使用 self.backbone.fusion.output_hypernetworks_mlps

    def forward(self, input):
        num_ppl_per_img = [len(bbox_list) for bbox_list in input["bboxes"]]
        
        # 1. Image Features
        x = self.backbone.forward(input["images"])
        if x.shape[1] != self.dim: x = self.linear(x)
        x = utils.repeat_tensors(x, num_ppl_per_img) # [Total_People, 256, 32, 32]

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
        
        # 获取 Sparse Embeddings [Total, N_prompts, 256]
        sparse_embeddings = self.backbone.prompt_encoder(
            bboxes_tensor, device=x.device, eyes=eyes_tensor, 
            expr_ids=input.get("observer_expression_ids", None)
        )
        
        # 3. Concatenate Tokens
        # tokens: [Total, Num_Learnable, 256]
        tokens = self.output_tokens.weight.unsqueeze(0).repeat(x.shape[0], 1, 1)
        # 拼接顺序: [Heatmap_Token, InOut_Token(可选), Prompt_Tokens...]
        tokens = torch.cat((tokens, sparse_embeddings), dim=1)

        # 4. Two-Way Fusion (内部会自动加 PE)
        # hs: [Total, All_Tokens, 256]
        # src: [Total, 256, 32, 32] (这里的 src 缺乏 pixel-SA，适合 DotProduct，不适合 Deconv)
        hs, src = self.backbone.fusion(image_embeddings=x, tokens=tokens)

        # 5. 解码逻辑 (Hypernetwork + Dot Product)
        
        # A. 提取对应的 Token
        heatmap_token = hs[:, self.idx_heatmap, :] # [Total, 256]
        
        # B. 上采样图像特征 (SAM 的 output_upscaling 层)
        # src [B, 256, 32, 32] -> [B, 32, 128, 128] (通道减少，尺寸变大)
        src_upscaled = self.backbone.fusion.output_upscaling(src)
        
        # C. 通过 MLP 生成权重 (Hypernetwork)
        # 使用第 0 个 MLP 对应 heatmap token
        # output_hypernetworks_mlps 是一个 ModuleList
        heatmap_mlp = self.backbone.fusion.output_hypernetworks_mlps[0] 
        hyper_weight = heatmap_mlp(heatmap_token) # [Total, 32] (维度通常对应 src_upscaled 的通道数)
        
        # D. Dot Product (点积生成 Mask)
        b, c, h, w = src_upscaled.shape
        # [B, 1, C] @ [B, C, H*W] -> [B, 1, H*W]
        masks = (hyper_weight.unsqueeze(1) @ src_upscaled.view(b, c, h * w))
        masks = masks.view(b, 1, h, w) # [Total, 1, 128, 128]
        
        # E. 后处理
        if masks.shape[-2:] != self.out_size:
            masks = F.interpolate(masks, size=self.out_size, mode='bilinear', align_corners=False)
        
        heatmap = torch.sigmoid(masks).squeeze(1)
        heatmap_preds = utils.split_tensors(heatmap, num_ppl_per_img)

        # 6. InOut 预测 (直接用 Token 预测，不做 Dot Product)
        inout_preds = None
        if self.inout:
             inout_token_out = hs[:, self.idx_inout, :]
             inout_preds_flat = self.inout_head(inout_token_out).squeeze(dim=-1)
             inout_preds = utils.split_tensors(inout_preds_flat, num_ppl_per_img)

        return {"heatmap": heatmap_preds, "inout": inout_preds, 
                "seg": None, "direction": None, "text_loss": None}

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
