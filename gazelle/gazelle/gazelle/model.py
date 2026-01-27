import torch
import torch.nn as nn
import torchvision
from timm.models.vision_transformer import Block
import math
import os
from gazelle.backbone import DinoV2Backbone, SAMBackboneWrapper, SAMImageEncoder, GazeTextDecoder
import gazelle.utils as utils
import torch.nn.functional as F

# ==========================================
# [New] Prompt Fusion Module (处理不全的Prompt)
# ==========================================
class PromptFusionModule(nn.Module):
    def __init__(self, dim, num_heads=4):
        super().__init__()
        # 定义一个统一的 "Gaze Query"
        self.unified_query = nn.Parameter(torch.randn(1, 1, dim)) 
        # 使用 Cross-Attention: Query是统一Token, Key/Value是输入的Prompt(bbox, eye, text)
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.ReLU(),
            nn.Linear(dim * 2, dim)
        )
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, prompt_embeddings):
        """
        prompt_embeddings: [B, N_prompts, Dim] (e.g., BBox + Eye + Text)
        即使某些是占位符，Attention机制也会学会赋予它们极低的权重
        """
        B = prompt_embeddings.shape[0]
        query = self.unified_query.repeat(B, 1, 1) # [B, 1, Dim]
        
        # Attention Fusion
        attn_out, _ = self.attn(query, prompt_embeddings, prompt_embeddings)
        x = self.norm1(query + attn_out)
        
        # FFN
        x = x + self.mlp(x)
        x = self.norm2(x)
        return x # [B, 1, Dim] -> 输出单一、融合后的 Prompt Token

# ==========================================
# [New] Gaze Guide Module (注视方向引导场景)
# ==========================================
class GazeGuideModule(nn.Module):
    def __init__(self, dim):
        super().__init__()
        # Channel Attention: 根据注视方向筛选重要的特征通道
        self.channel_gate = nn.Sequential(
            nn.Linear(dim, dim // 2), nn.ReLU(),
            nn.Linear(dim // 2, dim), nn.Sigmoid()
        )
        # FiLM (Feature-wise Linear Modulation): 强力条件注入
        # 通过 Scale 和 Shift 来调整特征分布
        self.gamma = nn.Linear(dim, dim) # Scale
        self.beta = nn.Linear(dim, dim)  # Shift

    def forward(self, x_img, gaze_token):
        """
        x_img: [B, Dim, H, W]
        gaze_token: [B, Dim] (来源于 Gaze3D 或 Direction)
        """
        # 1. Channel Weighting (SE-Block style)
        chn_weight = self.channel_gate(gaze_token).unsqueeze(-1).unsqueeze(-1)
        x_img = x_img * chn_weight

        # 2. FiLM Modulation
        scale = self.gamma(gaze_token).unsqueeze(-1).unsqueeze(-1)
        shift = self.beta(gaze_token).unsqueeze(-1).unsqueeze(-1)
        
        # ResBlock 风格: Original + Modulated
        return x_img * (1 + scale) + shift


class GazeLLE(nn.Module):
    def __init__(self, backbone, inout=False, dim=256, num_layers=3, in_size=(448, 448), out_size=(64, 64), is_multi_output=False, is_sam_prompt=False, is_mix_gaze_estimation=False):
        super().__init__()
        self.backbone = backbone
        self.dim = dim
        self.num_layers = num_layers
        self.is_multi_output = is_multi_output
        self.inout = inout
        self.in_size = in_size
        self.out_size = out_size
        self.is_sam_prompt = is_sam_prompt
        self.is_mix_gaze_estimation = is_mix_gaze_estimation

        # 获取 Backbone 输出特征图尺寸
        self.featmap_h, self.featmap_w = backbone.get_out_size(in_size)

        # 1. 检查能力: 是否支持 Prompt Encoder (即是否为 SAM 系列 Backbone)
        if is_sam_prompt:
            if not hasattr(backbone, 'prompt_encoder'):
                raise ValueError("SAM Prompt Backbone must have prompt encoder")
            # [新增] Prompt Fusion
            self.prompt_fusion = PromptFusionModule(dim=self.dim)

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
            
            # [新增] Dynamic Seg Head 组件 (MLP + Proj)
            # 用于根据 Seg Token 生成动态卷积核
            self.seg_token_proj = nn.Sequential(nn.Linear(dim, dim), nn.ReLU(), nn.Linear(dim, dim))
            self.seg_feat_proj = nn.Sequential(nn.Conv2d(dim, dim, kernel_size=1), nn.ReLU())

        # Legacy Support: 如果不是 SAM Backbone，还需要旧的 Head Token
        if not is_sam_prompt:
            self.head_token = nn.Embedding(1, self.dim)

        if self.is_mix_gaze_estimation:
            self.gaze3d_token = nn.Embedding(1, self.dim)
            self.gaze3d_head = nn.Sequential(
                nn.Linear(self.dim, 128), nn.ReLU(), nn.Dropout(0.1),
                nn.Linear(128, 2) # 输出 [Yaw, Pitch]
            )

        # [新增] Gaze Guide Module 
        # 只要有方向信息（无论是 3D Gaze 还是 2D Direction），就启用引导
        if self.is_mix_gaze_estimation or self.is_multi_output:
            self.gaze_guide = GazeGuideModule(self.dim)

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
            # Segmentation Head: 已改为 Dynamic Head，不再需要静态 Deconv 定义
            
            # Direction Head (8分类)
            self.direction_head = nn.Sequential(
                nn.Linear(self.dim, 128), nn.ReLU(), nn.Dropout(0.1),
                nn.Linear(128, 8)
            )
            # Text Head (GPT2 Decoder)
            self.text_head = GazeTextDecoder(input_dim=self.dim, model_name="gpt2", lora_r=8, max_len=25)

        # 初始化权重 (Bias trick)
        self._init_weights()

    def _init_weights(self):
        pass

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

        # [A] 特殊任务 Tokens (Output Queries)
        if self.inout:
            # [Total, 1, dim]
            inout_t = self.inout_token.weight.unsqueeze(0).repeat(x.shape[0], 1, 1)
            tokens_to_concat.append(inout_t)
        
        if self.is_multi_output:
            # [Total, 3, dim] (Seg, Dir, Text)
            multi_t = self.multi_output_tokens.weight.unsqueeze(0).repeat(x.shape[0], 1, 1)
            tokens_to_concat.append(multi_t)

        # [New Order] Gaze3D 放在 Image 之前 (优先处理)
        if self.is_mix_gaze_estimation:
            gaze3d_token_vec = self.gaze3d_token.weight.unsqueeze(0).repeat(x.shape[0], 1, 1)
            tokens_to_concat.append(gaze3d_token_vec)

        # [B] Prompt Embeddings (SAM 逻辑)
        sparse_embeddings = torch.zeros(x.shape[0], 0, self.dim, device=x.device) # Placeholder if legacy
        
        if self.is_sam_prompt:
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
            
            # 提取 Raw Prompts [Total, N_prompts, dim]
            raw_prompts = self.backbone.prompt_encoder(
                bboxes_tensor, device=x.device, eyes=eyes_tensor, 
                expr_ids=input.get("observer_expression_ids", None)
            )
            
            # [New] Prompt Fusion: 即使有缺失值，这里也会融合出一个干净的 Token
            fused_prompt = self.prompt_fusion(raw_prompts) # [B, 1, Dim]
            tokens_to_concat.append(fused_prompt)
        
        # [C] Legacy Head Map Logic (如果不是 SAM Backbone)
        else:
            # 这种情况下，直接把 head map 加到图像特征里
            head_maps = torch.cat(self.get_input_head_maps(input["bboxes"]), dim=0).to(x.device)
            head_map_embeddings = head_maps.unsqueeze(dim=1) * self.head_token.weight.unsqueeze(-1).unsqueeze(-1)
            x = x + head_map_embeddings

        # 4. 序列构建与融合
        # Flatten Image: [Total, dim, H, W] -> [Total, H*W, dim]
        x_flat = x.flatten(start_dim=2).permute(0, 2, 1)
        
        # Image 放在最后
        tokens_to_concat.append(x_flat)
        
        x_input = torch.cat(tokens_to_concat, dim=1)
        
        # 5. Standard Transformer 全局交互
        x_out = self.transformer(x_input)

        # 6. 解码与输出 (根据索引切片)
        
        current_idx = 0
        output_dict = {
            "heatmap": None, "inout": None, 
            "seg": None, "direction": None, "text_loss": None, "gaze3d": None
        }

        # [Decode] InOut
        if self.inout:
            inout_token_vec = x_out[:, current_idx, :]
            inout_val = self.inout_head(inout_token_vec).squeeze(dim=-1)
            output_dict["inout"] = utils.split_tensors(inout_val, num_ppl_per_img)
            current_idx += 1

        # ============================================================
        # [Step 1] Prioritize Decode: Gaze Direction / 3D
        # 优先拿到注视方向 Embedding，用于引导后续任务
        # ============================================================
        gaze_guide_embedding = None

        seg_token_vec = None
        dir_token_vec = None
        text_token_vec = None

        if self.is_multi_output:
            # tokens: 0=Seg, 1=Dir, 2=Text
            seg_token_vec = x_out[:, current_idx, :]
            dir_token_vec = x_out[:, current_idx+1, :]
            text_token_vec = x_out[:, current_idx+2, :]
            current_idx += 3
            
            # Direction Output
            dir_val = self.direction_head(dir_token_vec)
            output_dict["direction"] = utils.split_tensors(dir_val, num_ppl_per_img)
            
            # 【关键】如果启用了多任务，默认先使用 2D Direction Token 作为引导
            gaze_guide_embedding = dir_token_vec

        if self.is_mix_gaze_estimation:
            gaze3d_vec = x_out[:, current_idx, :]
            gaze3d_val = self.gaze3d_head(gaze3d_vec)
            output_dict["gaze3d"] = utils.split_tensors(gaze3d_val, num_ppl_per_img)
            current_idx += 1
            
            # 【关键】如果有 3D Gaze，这是更精确的引导源 (覆盖 2D Direction)
            gaze_guide_embedding = gaze3d_vec

        # [Skip] Prompts
        if self.is_sam_prompt:
            current_idx += 1 # 只有一个 fused prompt

        # [Decode] Image Features
        # 剩下的全是 Image Tokens
        x_img_tokens = x_out[:, current_idx:, :]
        
        # Reshape Back: [Total, H*W, dim] -> [Total, dim, H, W]
        x_img = x_img_tokens.permute(0, 2, 1).reshape(x.shape[0], self.dim, self.featmap_h, self.featmap_w)
        
        # ============================================================
        # [Step 2] Apply Gaze Guide
        # 如果有 Gaze 信息 (无论是 3D 还是 2D)，用它来增强图像特征
        # ============================================================
        if hasattr(self, 'gaze_guide') and gaze_guide_embedding is not None:
            # x_img_guided 蕴含了 "沿着这个方向看" 的语义
            x_img_guided = self.gaze_guide(x_img, gaze_guide_embedding)
            x_img_final = x_img + x_img_guided # Residual connection
        else:
            x_img_final = x_img

        # ============================================================
        # [Step 3] Decode Scene Targets (Based on Guided Features)
        # ============================================================
        
        # --- Gaze Heatmap (Static Head) ---
        # 使用引导后的特征 x_img_final
        heatmap = self.heatmap_head(x_img_final)
        heatmap = torchvision.transforms.functional.resize(heatmap, self.out_size)
        heatmap = heatmap.squeeze(1)
        output_dict["heatmap"] = utils.split_tensors(heatmap, num_ppl_per_img)
        
        # --- Multi-Output Targets ---
        if self.is_multi_output:
            # 1. Seg Heatmap (Dynamic Head)
            # 同样使用引导后的特征
            mask_features = self.seg_feat_proj(x_img_final)
            seg_kernel = self.seg_token_proj(seg_token_vec)
            
            # Dynamic Convolution (Dot Product)
            # 只有当图像特征匹配(引导后)且Seg Token指令匹配时才激活
            seg_logits = torch.einsum("bchw, bc -> bhw", mask_features, seg_kernel)
            
            seg_map = seg_logits.unsqueeze(1)
            seg_map = F.interpolate(seg_map, size=self.out_size, mode='bilinear', align_corners=False)
            seg_map = seg_map.squeeze(1)
            output_dict["seg"] = utils.split_tensors(seg_map, num_ppl_per_img)

            # 2. Text Output
            # 将 Gaze Embedding 融合进 Text Token，作为 Condition
            # 语义: "在 [gaze_direction] 方向上的 [text_semantic]"
            if gaze_guide_embedding is not None:
                text_input_feat = text_token_vec + gaze_guide_embedding
            else:
                text_input_feat = text_token_vec

            output_dict["text_loss"] = self.text_head(
                fusion_feat=text_input_feat, 
                target_ids=input.get("gaze_point_expression_ids")
            )

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

# ... (positionalencoding2d and factories) ...
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

# Models factory functions (unchanged)
def get_gazelle_model(model_name):
    factory = {
        "gazelle_dinov2_vitb14": gazelle_dinov2_vitb14,
        "gazelle_dinov2_vitl14": gazelle_dinov2_vitl14,
        "gazelle_dinov2_vitb14_inout": gazelle_dinov2_vitb14_inout,
        "gazelle_dinov2_vitl14_inout": gazelle_dinov2_vitl14_inout,
        "sam_vitb": sam_vitb,
        "dinov2_vitb_lora": dinov2_vitb_lora,
        "dinov2_vitb_multi_input": dinov2_vitb_multi_input,
        "dinov2_vitb_multi_output": dinov2_vitb_multi_output,
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

def sam_vitb():
    backbone = SAMImageEncoder(model_type="vit_b", in_size=(448, 448))
    transform = backbone.get_transform((448, 448))
    model = GazeLLE(backbone, inout=False)
    return model, transform

def dinov2_vitb_lora():
    backbone = DinoV2Backbone('dinov2_vitb14', is_lora=True)
    transform = backbone.get_transform((448, 448))
    model = GazeLLE(backbone, inout=False)
    return model, transform

def dinov2_vitb_multi_input():
    backbone = SAMBackboneWrapper(model_type="vit_b", in_size=(448, 448), backbone_type="dinov2", is_lora=False, is_multi_input=True)
    transform = backbone.get_transform((448, 448))
    model = GazeLLE(backbone, inout=False, is_sam_prompt=True)
    return model, transform
    
def dinov2_vitb_multi_output():
    backbone = SAMBackboneWrapper(model_type="vit_b", in_size=(448, 448), backbone_type="dinov2", is_lora=False, is_multi_input=False,)
    transform = backbone.get_transform((448, 448))
    model = GazeLLE(backbone, inout=False, is_multi_output=True, is_sam_prompt=False)
    return model, transform

def dinov2_vitb_lora_multi_output():
    backbone = SAMBackboneWrapper(model_type="vit_b", in_size=(448, 448), backbone_type="dinov2", is_lora=True, is_multi_input=False,)
    transform = backbone.get_transform((448, 448))
    model = GazeLLE(backbone, inout=False, is_multi_output=True, is_sam_prompt=False)
    return model, transform