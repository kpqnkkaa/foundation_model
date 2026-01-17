import os
# 设置环境变量：让 HuggingFace 使用国内镜像站
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import math
from typing import Optional, Tuple, List, Dict
import glob

# === 依赖库 ===
from segment_anything import sam_model_registry
from transformers import GPT2Model, GPT2LMHeadModel
from peft import LoraConfig, get_peft_model, TaskType

# ==========================================
# 辅助模块: 2D 正弦位置编码 (用于 Gaze-LLE)
# ==========================================
class PositionEmbeddingSine(nn.Module):
    """
    标准的 2D 正弦位置编码，用于给 Transformer 提供绝对空间信息
    """
    def __init__(self, num_pos_feats=128, temperature=10000, normalize=True):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        self.scale = 2 * math.pi

    def forward(self, x):
        # x: [B, C, H, W]
        mask = torch.ones((x.shape[0], x.shape[2], x.shape[3]), device=x.device)
        y_embed = mask.cumsum(1, dtype=torch.float32)
        x_embed = mask.cumsum(2, dtype=torch.float32)
        
        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        
        # Interleave sin/cos
        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
        
        # [B, H, W, C] -> [B, C, H, W]
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        return pos

# ==========================================
# 分支 1: 估计分支 (Estimation Branch)
# ==========================================
class EstimationBranch(nn.Module):
    def __init__(self, save_dir=None, w_est_pretrain=False):
        super().__init__()
        # ImageNet 预训练 ResNet50
        resnet = models.resnet50(pretrained=True)
        self.backbone = nn.Sequential(*list(resnet.children())[:-1]) 
        
        # 1. 特征映射: ResNet(2048) -> 256
        self.proj_256 = nn.Linear(2048, 256)
        
        # 2. 预测头
        self.head = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 2)    # [yaw, pitch]
        )
        self.load_pretrained_model(save_dir, w_est_pretrain)

    def load_pretrained_model(self, save_dir, w_est_pretrain):
        if w_est_pretrain and save_dir is not None:
            ckpt_list = glob.glob(os.path.join(save_dir, "Pretrain_*"))
            if len(ckpt_list) > 0:
                ckpt_list.sort(key=os.path.getmtime)
                ckpt_name = os.path.join(save_dir, ckpt_list[0], "pretrain_latest.pth")
                print(f"Loading Pretrained Model from {ckpt_name}...")
                try:
                    checkpoint = torch.load(ckpt_name, map_location='cpu')
                    state_dict = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint
                    new_state = {k.replace('module.', ''): v for k, v in state_dict.items()}
                    self.load_state_dict(new_state, strict=True)
                except Exception as e:
                    print(f"Failed to load estimation branch: {e}")

    def forward(self, face_img):
        feat = self.backbone(face_img) # [B, 2048, 1, 1]
        feat = torch.flatten(feat, 1)  # [B, 2048]
        feat_256 = self.proj_256(feat) 
        angles = self.head(feat_256)   # [B, 2]
        return angles, feat_256

# ==========================================
# 分支 2: 跟随分支组件
# ==========================================

class SAMImageEncoderReal(nn.Module):
    def __init__(self, checkpoint_path, model_type="vit_b", lora_r=8, img_size=448):
        super().__init__()
        print(f"Loading SAM Image Encoder ({model_type}) from {checkpoint_path}...")
        sam_model = sam_model_registry[model_type](checkpoint=checkpoint_path)
        self.image_encoder = sam_model.image_encoder
        
        if img_size != 1024:
            print(f"Resizing SAM pos_embed for {img_size}x{img_size} input...")
            self.adapt_pos_embed(img_size)

        del sam_model.prompt_encoder
        del sam_model.mask_decoder
        del sam_model
        
        for param in self.image_encoder.parameters():
            param.requires_grad = False
            
        peft_config = LoraConfig(
            r=lora_r, lora_alpha=lora_r * 2, target_modules=["qkv"], 
            lora_dropout=0.1, bias="none"
        )
        self.image_encoder = get_peft_model(self.image_encoder, peft_config)

    def adapt_pos_embed(self, new_img_size):
        old_pos_embed = self.image_encoder.pos_embed.data 
        patch_size = 16 
        new_grid_size = new_img_size // patch_size
        permuted_pos_embed = old_pos_embed.permute(0, 3, 1, 2)
        new_pos_embed = F.interpolate(
            permuted_pos_embed, size=(new_grid_size, new_grid_size),
            mode='bicubic', align_corners=False
        )
        new_pos_embed = new_pos_embed.permute(0, 2, 3, 1)
        self.image_encoder.pos_embed = nn.Parameter(new_pos_embed)

    def forward(self, x):
        return self.image_encoder(x)

class DINOv2ImageEncoder(nn.Module):
    def __init__(self, model_name="dinov2_vitb14", out_dim=256, lora_r=None):
        """
        DINOv2 Encoder 适配器 (支持 LoRA 微调)
        lora_r: 如果 > 0，则应用 LoRA 微调；否则冻结全部参数
        """
        super().__init__()
        print(f"Loading DINOv2 Model: {model_name}...")
        
        # 1. 加载 DINOv2
        self.backbone = torch.hub.load('facebookresearch/dinov2', model_name)
        self.embed_dim = self.backbone.embed_dim
        
        # 2. 配置 LoRA 或 冻结参数
        if lora_r is not None and lora_r > 0:
            print(f"Applying LoRA to DINOv2 (r={lora_r}, target=qkv)...")
            # DINOv2 的 Attention 线性层名字通常叫 'qkv'
            peft_config = LoraConfig(
                r=lora_r, 
                lora_alpha=lora_r * 2, 
                target_modules=["qkv"], 
                lora_dropout=0.1, 
                bias="none"
            )
            self.backbone = get_peft_model(self.backbone, peft_config)
            # get_peft_model 会自动把非 LoRA 部分设为 requires_grad=False
            
            # 打印可训练参数量确认
            self.backbone.print_trainable_parameters()
        else:
            print("Freezing DINOv2 backbone...")
            for param in self.backbone.parameters():
                param.requires_grad = False
        
        # 3. 投影层 (适配通道数 768 -> 256)
        # 这一层始终是要训练的
        self.projector = nn.Sequential(
            nn.Conv2d(self.embed_dim, out_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_dim),
            nn.ReLU()
        )

    def forward(self, x):
        # x: [B, 3, 448, 448]
        B = x.shape[0]
        
        # [Fix] 确保输入连续，防止 CUDA 报错
        x = x.contiguous()
        
        # DINOv2 Forward
        features_dict = self.backbone.forward_features(x)
        patch_tokens = features_dict["x_norm_patchtokens"] # [B, 1024, 768]
        
        # 恢复空间维度
        H, W = x.shape[2] // 14, x.shape[3] // 14
        
        # [Fix] permute 后必须加 .contiguous()
        feat_spatial = patch_tokens.permute(0, 2, 1).contiguous().reshape(B, self.embed_dim, H, W)
        
        # 投影到 256
        feat_projected = self.projector(feat_spatial) # [B, 256, 32, 32]
        
        return feat_projected

class SAMPromptEncoderReal(nn.Module):
    """
    修改版: 鲁棒的 Prompt Encoder
    1. 冻结 SAM 几何部分，微调 Text 部分 (LoRA)
    2. forward 中增加有效性检查，防止无效输入干扰特征融合
    """
    def __init__(self, checkpoint_path, model_type="vit_b", lora_r=8):
        super().__init__()
        print("Loading SAM Prompt Encoder & GPT-2 Text Encoder...")
        
        # 1. SAM Prompt Encoder (Frozen)
        sam_model = sam_model_registry[model_type](checkpoint=checkpoint_path)
        self.sam_prompt_encoder = sam_model.prompt_encoder
        
        # 释放显存
        del sam_model.image_encoder, sam_model.mask_decoder, sam_model
        
        # for param in self.sam_prompt_encoder.parameters():
        #     param.requires_grad = False
            
        # 2. Text Encoder (GPT2 + LoRA)
        self.text_encoder = GPT2Model.from_pretrained("gpt2")
        peft_config = LoraConfig(
            task_type=TaskType.FEATURE_EXTRACTION, 
            r=lora_r, lora_alpha=lora_r * 2, 
            target_modules=["c_attn"], lora_dropout=0.1, bias="none"
        )
        self.text_encoder = get_peft_model(self.text_encoder, peft_config)
        self.text_proj = nn.Linear(768, 256) 

    def forward(self, text_input_ids, face_bbox, eye_point, device):
        bs = face_bbox.shape[0] if face_bbox is not None else text_input_ids.shape[0]
        
        sparse_embeddings_list = []
        
        # --- A. 几何 Prompt (Point/Box) ---
        if face_bbox is not None and eye_point is not None:
            has_eye = (eye_point[:, 0] > -0.5) 
            has_box = (face_bbox[:, 0] > -0.5) 

            # Eye (Label 1)
            e_coord = torch.where(has_eye.unsqueeze(1), eye_point, 0.0).unsqueeze(1)
            e_label = torch.where(has_eye, 1, -1).int().unsqueeze(1)

            # Box (Labels 2, 3)
            b_coord = face_bbox.reshape(bs, 2, 2)
            b_coord = torch.where(has_box.view(bs, 1, 1), b_coord, 0.0)
            b_label_base = torch.tensor([[2, 3]], device=device).repeat(bs, 1)
            b_label = torch.where(has_box.unsqueeze(1), b_label_base, -1).int()

            coords = torch.cat([e_coord, b_coord], dim=1)
            labels = torch.cat([e_label, b_label], dim=1)
            
            # 使用 SAM Encoder 编码
            geom_sparse, _ = self.sam_prompt_encoder(
                points=(coords, labels), boxes=None, masks=None
            )
            sparse_embeddings_list.append(geom_sparse)

        # --- B. 文本 Prompt (Text) ---
        if text_input_ids is not None:
            # 检查是否为有效文本 (非全 padding)
            # 假设 50256 是 GPT2 的 pad_token_id (或 eos)
            is_valid_text = (text_input_ids != 50256).any(dim=1) # [B]
            
            # 即使全 batch 无效也要跑一遍结构保持计算图，但可以通过 mask 屏蔽
            gpt_out = self.text_encoder(text_input_ids)[0] # [B, L, 768]
            text_feat = gpt_out.mean(dim=1) # [B, 768]
            text_embed = self.text_proj(text_feat).unsqueeze(1) # [B, 1, 256]
            
            # 屏蔽无效文本的 Embedding (置为0)
            text_embed = torch.where(is_valid_text.view(bs, 1, 1), text_embed, torch.zeros_like(text_embed))
            sparse_embeddings_list.append(text_embed)

        # --- C. 拼接 ---
        if len(sparse_embeddings_list) > 0:
            # [B, N_total, 256]
            # 这里的 concat 就是最高效的 Transformer 融合：
            # 后续的 Self-Attention 会自动处理不同模态 Token 之间的关系
            return torch.cat(sparse_embeddings_list, dim=1)
        else:
            # Fallback
            return torch.zeros(bs, 1, 256, device=device)

class SAMFusionReal(nn.Module):
    """
    SAM TwoWayTransformer: 负责 Image 和 Prompt 的交互
    """
    def __init__(self, checkpoint_path, model_type="vit_b"):
        super().__init__()
        sam_model = sam_model_registry[model_type](checkpoint=checkpoint_path)
        self.transformer = sam_model.mask_decoder.transformer
        self.pe_layer = sam_model.prompt_encoder.pe_layer 
        del sam_model
        
        for param in self.transformer.parameters():
            param.requires_grad = True
        for param in self.pe_layer.parameters():
            param.requires_grad = False

    def get_dense_pe(self, image_shape_hw):
        h, w = image_shape_hw
        return self.pe_layer((h, w)).unsqueeze(0)

    def forward(self, image_embeddings, sparse_embeddings):
        B, C, H, W = image_embeddings.shape
        image_pe = self.get_dense_pe((H, W)).to(image_embeddings.device).repeat(B, 1, 1, 1)
        
        # TwoWayTransformer:
        # sparse_encoded -> Queries (Prompt) 更新后特征 (代表目标语义)
        # dense_encoded  -> Keys (Image) 更新后特征 (代表注意力图)
        sparse_encoded, dense_encoded = self.transformer(
            point_embedding=sparse_embeddings,
            image_embedding=image_embeddings, 
            image_pe=image_pe 
        )
        return sparse_encoded, dense_encoded

class LLMDecoderReal(nn.Module):
    def __init__(self, input_dim=1024, model_name="gpt2", lora_r=8):
        super().__init__()
        self.gpt2 = GPT2LMHeadModel.from_pretrained(model_name)
        for param in self.gpt2.parameters():
            param.requires_grad = False
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM, inference_mode=False, 
            r=lora_r, lora_alpha=lora_r * 2, lora_dropout=0.1
        )
        self.gpt2 = get_peft_model(self.gpt2, peft_config)
        self.visual_proj = nn.Linear(input_dim, self.gpt2.config.n_embd)

    def forward(self, fusion_feat, target_ids=None):
        visual_embeds = self.visual_proj(fusion_feat).unsqueeze(1)
        if target_ids is not None:
            wte = self.gpt2.base_model.model.transformer.wte
            text_embeds = wte(target_ids)
            inputs_embeds = torch.cat([visual_embeds, text_embeds], dim=1)
            outputs = self.gpt2(inputs_embeds=inputs_embeds)
            return outputs.logits
        else:
            outputs = self.gpt2(inputs_embeds=visual_embeds)
            return outputs.logits

# ==========================================
# 核心分支: Following Branch (Gaze-LLE 架构集成)
# ==========================================

class FollowingBranch(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.ckpt_name = "sam_vit_b_01ec64.pth"
        self.ckpt_url = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"
        self.ensure_checkpoint()
        model_type = "vit_b"
        
        # ====================================================
        # [修改 1] 提升 LoRA Rank (8 -> 64) 以增强语义学习能力
        # ====================================================
        lora_rank = 64
        
        # 1. 基础特征提取 (Frozen/Low-Rank)
        self.scene_encoder = DINOv2ImageEncoder(
            model_name="dinov2_vitb14", 
            out_dim=256, 
            lora_r=lora_rank  # <--- 传入 64，开启微调
        )
        self.prompt_encoder = SAMPromptEncoderReal(self.ckpt_name, model_type, lora_r=lora_rank)
        
        # 2. 特征融合 (SAM Fusion)
        self.sam_fusion = SAMFusionReal(self.ckpt_name, model_type)
        
        # 3. Stage 1 中间监督头 (辅助任务)
        # 预测头部 Mask
        self.head_mask_pred = nn.Sequential(
            nn.Conv2d(256, 64, 3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 1, 1), nn.Sigmoid()
        )
        # 预测视线方向 (辅助特征对齐)
        self.gaze_dir_pred = nn.Sequential(
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, 1)
        )

        # 8分类方向上下左右方向预测
        self.gaze_dir_cls = nn.Sequential(
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, 8)
        )
        # 2分类前后预测
        self.gaze_front_back_pred = nn.Sequential(
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, 2)
        )

        # 4. Stage 2: Gaze-LLE 风格融合与解码
        # (1) 上下文构建层 (Conv): 融合 Scene + Mask*Feat
        # 输入维度: 256 (Scene) + 256 (Dense Refined) = 512
        self.context_fusion = nn.Sequential(
            nn.Conv2d(512, 256, 3, padding=1), 
            nn.BatchNorm2d(256), 
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1), 
            nn.BatchNorm2d(256), 
            nn.ReLU()
        )
        
        # (2) 位置编码 (Gaze-LLE Re-injection)
        self.pos_embed_decoder = PositionEmbeddingSine(num_pos_feats=128, normalize=True)
        
        # (3) Tokens 定义
        # A. 独立的 IO Token (Learnable)
        self.io_token = nn.Parameter(torch.randn(1, 1, 256) * 0.02)
        
        # B. [新增] Head Token Projector
        # 将 SAM 提取的头部向量 feat_256 映射为 Transformer 的 Token
        self.head_token_proj = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256) # 输出用于计算 Cosine Loss 的特征
        )
        
        # (4) Transformer Decoder (用于全局推理)
        # [修改 2] 加深网络层数 (3 -> 6) 以增强推理能力
        decoder_layer = nn.TransformerEncoderLayer(d_model=256, nhead=8, dim_feedforward=1024, batch_first=True)
        self.transformer_decoder = nn.TransformerEncoder(decoder_layer, num_layers=6)
        
        # 5. Final Heads
        # Heatmap Head (全卷积)
        self.heatmap_head = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, 2, 1), nn.BatchNorm2d(128), nn.ReLU(), # 28->56
            nn.Conv2d(128, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 1, 1), nn.Sigmoid()
        )
        
        # IO Head (从 IO Token 解码)
        self.inout_head = nn.Sequential(nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, 1))
        
        # Text Generation (从全局特征解码)
        self.global_adapter = nn.Sequential(
            nn.Linear(256, 1024), nn.LayerNorm(1024), nn.GELU()
        )
        self.llm_decoder = LLMDecoderReal(input_dim=1024)

    def ensure_checkpoint(self):
        if not os.path.exists(self.ckpt_name):
            try:
                torch.hub.download_url_to_file(self.ckpt_url, self.ckpt_name)
            except Exception as e:
                raise e

    def forward(self, scene_img, obs_text_ids, face_bbox, eye_point, target_text_ids=None):
        B = scene_img.shape[0]
        device = scene_img.device
        
        # --- Step 1: Encoder ---
        scene_feat_raw = self.scene_encoder(scene_img) # [B, 256, 28, 28] (对于 448 输入)
        
        # [重要] 确保 face_bbox 是像素坐标 (如果前面没有处理，这里需要确认)
        # 假设 DataLoader 输出已经是 0-448 的像素坐标
        sparse_prompts = self.prompt_encoder(obs_text_ids, face_bbox, eye_point, device)
        
        # --- Step 2: SAM Fusion (Target Perception) ---
        # sparse_refined: [B, N, 256] 
        # dense_refined:  [B, 256, H, W]
        sparse_refined, dense_refined = self.sam_fusion(scene_feat_raw, sparse_prompts)
        
        # 提取头部语义特征 (Head Embedding)
        # 这是非常关键的“指南针”信息，指明了谁在看
        feat_256 = sparse_refined.mean(dim=1) # [B, 256]

        # 变换 dense_refined 回空间维度
        B, C, H, W = scene_feat_raw.shape 
        # dense_refined 已经是 [B, C, H, W] (如果 SAMFusionReal 输出是 map) 
        # 或者如果是 Sequence [B, L, C]，则需要 reshape:
        if dense_refined.dim() == 3: # [B, L, C]
             dense_refined_spatial = dense_refined.permute(0, 2, 1).reshape(B, C, H, W)
        else:
             dense_refined_spatial = dense_refined
        
        # --- Step 3: 中间任务输出 ---
        pred_head_mask = self.head_mask_pred(dense_refined_spatial) # [B, 1, H, W]
        
        # --- Step 4: Context Fusion ---
        # 拼接原始场景和SAM注意力图
        combined_feat = torch.cat([scene_feat_raw, dense_refined_spatial], dim=1) # [B, 512, H, W]
        context_feat = self.context_fusion(combined_feat) # [B, 256, H, W]
        
        # --- Step 5: Gaze-LLE Transformer Decoder ---
        # 1. 注入位置编码
        pos = self.pos_embed_decoder(context_feat) # [B, 256, H, W]
        
        # 2. 展平特征 (Scene Tokens)
        src = (context_feat + pos).flatten(2).permute(0, 2, 1) # [B, HW, 256]
        
        # =======================================================
        # [修改 3] 构造 Tokens 序列: [IO, Head, Scene...]
        # =======================================================
        
        # A. IO Token
        io_tokens = self.io_token.expand(B, -1, -1) # [B, 1, 256]
        
        # B. Head Token (注入头部姿态语义)
        # feat_256: [B, 256] -> [B, 1, 256]
        head_token = self.head_token_proj(feat_256.unsqueeze(1))
        
        # C. 拼接
        # 序列长度: 1 (IO) + 1 (Head) + HW (Scene)
        tokens_in = torch.cat([io_tokens, head_token, src], dim=1) 
        
        # 4. Transformer 推理
        tokens_out = self.transformer_decoder(tokens_in)
        
        # 5. 分离 Token
        # tokens_out: [B, 2 + HW, 256]
        
        io_token_out = tokens_out[:, 0, :]   # 第 0 个是 IO
        head_token_out = tokens_out[:, 1, :] # 第 1 个是 Head (参与 Attention 交互后的 Head 表征)
        
        # 视线方向：按你的要求改为直接输出 head_token_out（而不是 feat_256 -> gaze_dir_pred）
        # 同时保留一个标量角度预测，供现有 loss 使用（避免训练代码大改）
        pred_gaze_dir_angle = self.gaze_dir_pred(head_token_out)  # [B, 1]
        
        pred_gaze_dir_cls = self.gaze_dir_cls(head_token_out)  # [B, 8]
        pred_gaze_front_back = self.gaze_front_back_pred(head_token_out)  # [B, 2]

        # [注意] 空间特征从索引 2 开始
        spatial_tokens_out = tokens_out[:, 2:, :] # [B, HW, 256]
        
        # --- Step 6: Final Outputs ---
        # 1. Heatmap
        spatial_map = spatial_tokens_out.permute(0, 2, 1).view(B, 256, H, W)
        pred_heatmap = self.heatmap_head(spatial_map) # [B, 1, 56, 56]
        
        # 2. In/Out
        pred_inout = self.inout_head(io_token_out) # [B, 1]
        
        # 3. Text
        global_feat = spatial_tokens_out.mean(dim=1) # [B, 256]
        adapter_out = self.global_adapter(global_feat) # [B, 1024]
        text_logits = self.llm_decoder(adapter_out, target_ids=target_text_ids)

        return {
            "text_logits": text_logits,
            "pred_gaze_point": pred_heatmap,
            "pred_inout": pred_inout,
            
            # 辅助输出用于 Loss
            "feat_256": feat_256,           # 与 Estimation 分支做 Cosine Loss
            "pred_head_mask": pred_head_mask, # 与 Head Mask GT 做 Dice/BCE Loss
            "pred_gaze_dir": head_token_out,          # Head token 表征（你要的输出）
            "pred_gaze_dir_angle": pred_gaze_dir_angle, # [B,1] 与 2D angle GT 做 Loss
            "pred_gaze_dir_cls": pred_gaze_dir_cls, # [B, 8] 与 8 分类 GT 做 Loss
            "pred_gaze_front_back": pred_gaze_front_back, # [B, 2] 与 2 分类 GT 做 Loss
        }

# ==========================================
# 联合模型入口
# ==========================================
class UnifiedGazeModel(nn.Module):
    def __init__(self, save_dir=None, w_est_pretrain=False):
        super().__init__()
        # 1. Estimation Branch (保留)
        self.estimation_branch = EstimationBranch(save_dir, w_est_pretrain)
        # 2. Following Branch (Gaze-LLE 集成版)
        self.following_branch = FollowingBranch()

    def forward(self, batch_data):
        outputs = {}
        
        # === Estimation Branch ===
        if 'face_img' in batch_data:
            est_angles, est_feat_256 = self.estimation_branch(batch_data['face_img'])
            outputs['pred_angles'] = est_angles
            outputs['est_feat_aligned'] = est_feat_256

        # === Following Branch ===
        if 'scene_img' in batch_data and batch_data['scene_img'] is not None:
            text_out = self.following_branch(
                scene_img=batch_data['scene_img'], 
                obs_text_ids=batch_data.get('observer_tokens', None), 
                face_bbox=batch_data.get('face_bbox', None), 
                eye_point=batch_data.get('eye_point_norm', None),
                target_text_ids=batch_data.get('gaze_point_expressions_ids', None)
            )
            # 整合输出
            outputs['pred_text_logits'] = text_out['text_logits']
            outputs['pred_gaze_point'] = text_out['pred_gaze_point']
            outputs['pred_inout'] = text_out['pred_inout']
            outputs['follow_feat'] = text_out['feat_256'] # 用于对齐
            outputs['pred_gaze_dir_cls'] = text_out['pred_gaze_dir_cls']
            outputs['pred_gaze_front_back'] = text_out['pred_gaze_front_back']
            outputs['pred_gaze_dir_angle'] = text_out['pred_gaze_dir_angle']
            outputs['pred_gaze_dir'] = text_out['pred_gaze_dir']
            outputs['pred_head_mask'] = text_out['pred_head_mask']

        return outputs