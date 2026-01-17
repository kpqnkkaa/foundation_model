# 读取的数据格式如下：
# [Batch 0] Structure Check:
#   > Estimation Task (['Gaze360', 'Gaze360']):
#     - face_img Shape: torch.Size([2, 3, 224, 224])
#     - 3D_ground_truth GT: tensor([[ 0.5233, -0.1544],
#         [ 0.1896, -0.1991]])
#   > Following Task (['GazeFollow_Extended', 'GazeFollow_Extended']):
#     - [Sub-Est] face_img Shape: torch.Size([2, 3, 224, 224])
#     - [Sub-Est] 2D_ground_truth GT: tensor([[-0.5415,  0.8407],
#         [ 0.9670,  0.2549]])
#     - [Sub-Fol] scene_img Shape: torch.Size([2, 3, 448, 448])
#     - [Sub-Fol] observer_expression: ['the man with short dark hair wearing a brown sweatshirt', '']
#     - [Sub-Fol] face_bbox: tensor([[0.3270, 0.1470, 0.5200, 0.3400],
#         [0.1330, 0.0720, 0.3180, 0.3280]])
#     - [Sub-Fol] eye_point_norm: tensor([[0.3870, 0.2780],
#         [0.2730, 0.2040]])
#     - [Sub-Fol] gaze_points_norm: [[tensor([0.1970, 0.7510], dtype=torch.float64), tensor([0.5730, 0.3300], dtype=torch.float64)]]
#     - [Sub-Fol] gaze_point_expressions: [('the ceramic bowl with a blue pattern he is washing in the sink', 'the woman with auburn hair and a hair clip, standing near the reception desk to his right')]
#     - [Sub-Fol] in_outs: [[tensor([1, 1])]]

# 我们的模型分为两个分支
# 1. 估计分支
# 估计分支以人脸为输入，输出3D gaze方向
# 对应的数据有Estimation Task+Following Task
# 他们的face_img经过image_net预训练的resnet50提取特征，得到2048维的特征，然后映射为256维特征，经过2层全连接层，得到yaw，pitch
# 对于Estimation Task， 3D_ground_truth GT和预测的yaw，pitch计算L1损失
# 对于Following Task， yaw，pitch映射为2维方向
#def get_2d_gaze_vector(eye_norm, gaze_norm):
    # """
    # 计算2D平面上的单位方向向量 (x, y)
    # """
    # dx = gaze_norm['x'] - eye_norm['x']
    # dy = gaze_norm['y'] - eye_norm['y']
    
    # # 计算模长
    # norm = np.sqrt(dx**2 + dy**2)
    
    # # 防止除以0 (极少数情况眼睛和注视点重合)
    # if norm < 1e-6:
    #     return [0.0, 0.0]
    
    # # 归一化，只保留方向信息
    # return [dx / norm, dy / norm]
# 二维方向的计算代码如上，把yaw,pitch转化为这样的2D方向，计算预测2D方向和 2D_ground_truth 的L1损失
# 提示这里的   - yaw > 0: 视线朝图像左侧
    #   - yaw < 0: 视线朝图像右侧
    #   - pitch > 0: 视线朝图像上方
    #   - pitch < 0: 视线朝图像下方
# 2. 跟随分支
# 跟随分支采样following task的样本训练
# 跟随分支以场景(scene_img)和人脸标识符(observer_expression/face_bbox/eye_point_norm)为输入
# 输出gaze标识符(gaze_points_norm/gaze_point_expressions/in_outs)
# 场景图像通过预训练的SAM图像编码器提取特征，得到768维的特征
# 人脸标识符根据类型相应通过 SAM 的 prompt encoder 提取特征，得到256维的特征
# 将场景和人脸标识符的特征拼接，得到1024维的特征
# 将该特征经过几种decoder（采用预训练的LLM decoder预测文字/采用3层全连接层预测gaze point的2维坐标/采用3层全连接层预测in_outs二分类））进行预测
# 采用预训练的LLM decoder预测文字的损失为交叉熵损失
# 采用3层全连接层预测gaze point的2维坐标的损失为L1损失
# 采用3层全连接层预测in_outs二分类的损失为交叉熵损失
# 最终的损失为交叉熵损失+L1损失+交叉熵损失的加权和
# 由于SAM的图像编码器以及LLM的decoder都是预训练的，采用LoRA进行微调
# observer描述可能为""或None此时跳过这个prompt输入，仅采用另外两种人脸标识符
# gaze_point_expressions可能为[""]或None此时计算损失跳过这个输出，计算in_outs+gaze_points_norm的损失
# 3.联合训练
# 拉近估计分支face_img经过resnet50提取的特征和跟随分支的场景和人脸标识符融合后的特征的相似性

import os
# 设置环境变量：让 HuggingFace 使用国内镜像站
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import math
from typing import Optional, Tuple

# === 依赖库 ===
from segment_anything import sam_model_registry
from transformers import GPT2Model, GPT2LMHeadModel, GPT2Config
from peft import LoraConfig, get_peft_model, TaskType
import glob

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

    # 如果w_est_pretrain为True，加载save_dir前缀为Pretrain的最新时间戳的pth
    def load_pretrained_model(self, save_dir, w_est_pretrain):
        if w_est_pretrain and save_dir is not None:
            # 找到save_dir下前缀为Pretrain的Pretrain_ETH-G360_Gaze360_EstBranch_20260111_1813后面那个时间戳最新的
            ckpt_list = glob.glob(os.path.join(save_dir, "Pretrain_*"))
            ckpt_list.sort(key=os.path.getmtime)
            ckpt_name = os.path.join(save_dir, ckpt_list[0], "pretrain_latest.pth")
            print(f"Loading Pretrained Model from {ckpt_name}...")
            checkpoint = torch.load(ckpt_name, map_location='cpu')
            state_dict = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint
            new_state = {k.replace('module.', ''): v for k, v in state_dict.items()}
            self.load_state_dict(new_state, strict=True)

    def forward(self, face_img):
        feat = self.backbone(face_img) # [B, 2048, 1, 1]
        feat = torch.flatten(feat, 1)  # [B, 2048]
        feat_256 = self.proj_256(feat) 
        angles = self.head(feat_256)    # [B, 2]
        return angles, feat_256


# ==========================================
# 分支 2: 跟随分支组件 (SAM + GPT2 + GPT2)
# ==========================================

class SAMImageEncoderReal(nn.Module):
    """
    真实 SAM Image Encoder + LoRA
    修复版：支持分辨率调整 (例如 1024 -> 448)
    """
    def __init__(self, checkpoint_path, model_type="vit_b", lora_r=8, img_size=448):
        super().__init__()
        print(f"Loading SAM Image Encoder ({model_type}) from {checkpoint_path}...")
        
        # 1. 加载 SAM
        sam_model = sam_model_registry[model_type](checkpoint=checkpoint_path)
        self.image_encoder = sam_model.image_encoder
        
        # === 核心修复：调整位置编码 (Positional Embedding) ===
        # SAM 默认是 1024x1024 (patch_size=16 -> 64x64 grid)
        # 如果我们输入 448x448 (patch_size=16 -> 28x28 grid)
        # 我们必须把 pos_embed 从 64x64 插值到 28x28
        if img_size != 1024:
            print(f"Resizing SAM pos_embed for {img_size}x{img_size} input...")
            self.adapt_pos_embed(img_size)

        # 释放不需要的部分
        del sam_model.prompt_encoder
        del sam_model.mask_decoder
        del sam_model
        
        # 2. 冻结原始参数
        for param in self.image_encoder.parameters():
            param.requires_grad = False
            
        # 3. 注入 LoRA
        peft_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_r * 2,
            target_modules=["qkv"], 
            lora_dropout=0.1,
            bias="none"
        )
        self.image_encoder = get_peft_model(self.image_encoder, peft_config)
        self.image_encoder.print_trainable_parameters()

    def adapt_pos_embed(self, new_img_size):
        """
        手动插值 SAM 的位置编码以适应新的分辨率
        SAM pos_embed shape: [1, H_grid, W_grid, C] (e.g., 1, 64, 64, 768)
        """
        # 1. 获取原始 pos_embed
        # shape: [1, 64, 64, 768]
        old_pos_embed = self.image_encoder.pos_embed.data 
        
        # 2. 计算新的 Grid Size
        # SAM ViT patch_size 固定为 16
        patch_size = 16 
        new_grid_size = new_img_size // patch_size # 448/16 = 28
        
        print(f"Interpolating pos_embed: {old_pos_embed.shape} -> {new_grid_size}x{new_grid_size}")
        
        # 3. 插值 (需要 permute 到 [N, C, H, W] 进行 interpolate)
        # [1, 64, 64, 768] -> [1, 768, 64, 64]
        permuted_pos_embed = old_pos_embed.permute(0, 3, 1, 2)
        
        new_pos_embed = F.interpolate(
            permuted_pos_embed,
            size=(new_grid_size, new_grid_size),
            mode='bicubic',
            align_corners=False
        )
        
        # 4. 恢复形状并赋值回去
        # [1, 768, 28, 28] -> [1, 28, 28, 768]
        new_pos_embed = new_pos_embed.permute(0, 2, 3, 1)
        
        # 覆盖原始参数
        self.image_encoder.pos_embed = nn.Parameter(new_pos_embed)

    def forward(self, x):
        # x: [B, 3, 448, 448] 
        return self.image_encoder(x)


class SAMPromptEncoderReal(nn.Module):
    """
    真实 SAM Prompt Encoder (处理点/框) + GPT-2 Text Encoder (处理文本)
    
    策略:
    1. SAM Prompt Encoder: 冻结 (保持几何感知能力)
    2. GPT-2 Text Encoder: LoRA 微调 (适配视觉对齐任务)
    """
    def __init__(self, checkpoint_path, model_type="vit_b", lora_r=8):
        super().__init__()
        print("Loading SAM Prompt Encoder & GPT-2 Text Encoder (with LoRA)...")
        
        # 1. 加载 SAM 原生 Prompt Encoder (用于处理 Points 和 Boxes)
        sam_model = sam_model_registry[model_type](checkpoint=checkpoint_path)
        self.sam_prompt_encoder = sam_model.prompt_encoder
        
        # 释放不需要的部分，节省显存
        del sam_model.image_encoder
        del sam_model.mask_decoder
        del sam_model
        
        # === [策略 1] 冻结 SAM Prompt Encoder ===
        # 它的作用是将坐标 (x,y) 转为 Embedding，这种几何逻辑是通用的，不需要训练
        for param in self.sam_prompt_encoder.parameters():
            param.requires_grad = False
            
        # 2. 加载 GPT-2 Model (作为 Text Encoder)
        self.text_encoder = GPT2Model.from_pretrained("gpt2")
        
        # === [策略 2] 对 GPT-2 使用 LoRA 微调 ===
        # 原始 GPT-2 是纯文本模型，不懂图片。
        # 我们用 LoRA 稍微调整它，让它学会输出"能被视觉模型理解"的特征
        peft_config = LoraConfig(
            task_type=TaskType.FEATURE_EXTRACTION, 
            r=lora_r,
            lora_alpha=lora_r * 2,
            target_modules=["c_attn"], # 对 GPT-2 内部的 Attention 层进行微调
            lora_dropout=0.1,
            bias="none"
        )
        self.text_encoder = get_peft_model(self.text_encoder, peft_config)
        self.text_encoder.print_trainable_parameters()
            
        # 4. 投影层 (768 -> 256)
        # 将 GPT-2 的特征维度 (768) 压缩到 SAM 的特征维度 (256)
        # 这个层必须训练 (Requires Grad)
        self.text_proj = nn.Linear(768, 256) 

    def forward(self, text_input_ids, face_bbox, eye_point):
        """
        Args:
            text_input_ids: [B, SeqLen] GPT-2 Token IDs
            face_bbox:      [B, 4]      人脸框坐标
            eye_point:      [B, 2]      眼球中心点坐标
        Returns:
            sparse_embeddings: [B, N+1, 256] (包含点、框、文本的 token 序列)
            dense_embeddings:  [B, 256, 64, 64] (空间掩码，这里通常是 no-mask embed)
        """

        bs = face_bbox.shape[0]
        device = face_bbox.device

        # === A. 处理几何提示 (SAM 原生逻辑) ===
        # SAM 要求 points 格式为 (coords, labels)
        # coords: [B, N_points, 2]
        # labels: [B, N_points] (1代表前景点，0代表背景点)
        # 1. 准备 Mask (并行判断有效性)
        # 假设 eye_point: [B, 2], face_bbox: [B, 4]
        #只要第一个坐标不是 -1 (比如 > -0.5)，就认为是有效数据
        has_eye = (eye_point[:, 0] > -0.5)  # Shape: [B]
        has_box = (face_bbox[:, 0] > -0.5)  # Shape: [B]

        # 2. 处理 Eye Point (B, 1, 2)
        # 坐标: 有效保持原值，无效置为0
        e_coord = torch.where(has_eye.unsqueeze(1), eye_point, 0.0).unsqueeze(1)
        # Label: 有效设为1, 无效设为-1
        e_label = torch.where(has_eye, 1, -1).int().unsqueeze(1)

        # 3. 处理 Box (B, 2, 2) -> 拆解为左上/右下两个点
        # 先变形: [B, 4] -> [B, 2, 2]
        b_coord_raw = face_bbox.reshape(bs, 2, 2)
        
        # 坐标: 利用 mask 置 0 (注意 view 的维度广播)
        b_coord = torch.where(has_box.view(bs, 1, 1), b_coord_raw, 0.0)
        
        # Label: 构造基础 label [2, 3]，利用 mask 将无效的变为 -1
        b_label_base = torch.tensor([[2, 3]], device=device).repeat(bs, 1)
        b_label = torch.where(has_box.unsqueeze(1), b_label_base, -1).int()

        # 4. 拼接 (Concat)
        # coords shape: [B, 3, 2] (第一个是眼点，后两个是框点)
        final_coords = torch.cat([e_coord, b_coord], dim=1)
        # labels shape: [B, 3]
        final_labels = torch.cat([e_label, b_label], dim=1)
        # print("final_coords: " + str(final_coords))
        # print("final_labels: " + str(final_labels))

        # 5. 输出给 SAM
        points = (final_coords, final_labels)
        boxes = None # 必须为 None

        # 调用 SAM Prompt Encoder
        # sparse_embeddings: [B, N_geo, 256] (N_geo 是点和框产生的 token 数)
        # dense_embeddings:  [B, 256, 64, 64] (no-mask embedding)
        # print(points, boxes)
        sparse_embeddings, dense_embeddings = self.sam_prompt_encoder(
            points=points,
            boxes=boxes,
            masks=None # 我们没有 mask 提示
        )

        # === B. 处理语义提示 (GPT-2 + Projection) ===
        if text_input_ids is not None:
            # 1. GPT-2 提取特征
            gpt_out = self.text_encoder(text_input_ids)[0] # [B, SeqLen, 768]
            
            # 2. 聚合序列特征 (取平均，或者取最后一个 token)
            # 这里使用 Mean Pooling
            text_feat = gpt_out.mean(dim=1) # [B, 768]
            
            # 3. 投影到 256 维
            text_embed = self.text_proj(text_feat) # [B, 256]
            
            # 4. 调整形状以进行拼接: [B, 1, 256]
            text_embed = text_embed.unsqueeze(1) 
            
            # 5. 拼接到 sparse_embeddings 后面
            # 最终 sparse_embeddings 变成了: [Geometry_Tokens, Text_Token]
            sparse_embeddings = torch.cat([sparse_embeddings, text_embed], dim=1)

        return sparse_embeddings, dense_embeddings

class SAMFusionReal(nn.Module):
    """
    真实 SAM TwoWayTransformer
    """
    def __init__(self, checkpoint_path, model_type="vit_b"):
        super().__init__()
        print("Extracting SAM TwoWayTransformer...")
        
        sam_model = sam_model_registry[model_type](checkpoint=checkpoint_path)
        self.transformer = sam_model.mask_decoder.transformer
        self.pe_layer = sam_model.prompt_encoder.pe_layer 
        
        del sam_model

        # 开启 Transformer 训练
        for param in self.transformer.parameters():
            param.requires_grad = True
        # 保持位置编码冻结
        for param in self.pe_layer.parameters():
            param.requires_grad = False

    def get_dense_pe(self, image_shape_hw):
        h, w = image_shape_hw
        return self.pe_layer((h, w)).unsqueeze(0)

    def forward(self, image_embeddings, sparse_embeddings):
        B, C, H, W = image_embeddings.shape
        
        # 准备 PE (4D)
        image_pe = self.get_dense_pe((H, W)).to(image_embeddings.device)
        image_pe = image_pe.repeat(B, 1, 1, 1) 
        
        # Transformer 交互
        sparse_encoded, _ = self.transformer(
            point_embedding=sparse_embeddings,
            image_embedding=image_embeddings, # 4D
            image_pe=image_pe                 # 4D
        )
        
        # 聚合特征
        fusion_feat = sparse_encoded.mean(dim=1) 
        return fusion_feat


class LLMDecoderReal(nn.Module):
    """
    真实 GPT-2 Decoder + LoRA
    """
    def __init__(self, input_dim=1024, model_name="gpt2", lora_r=8):
        super().__init__()
        print(f"Loading GPT-2 Decoder ({model_name}) with LoRA...")
        
        self.gpt2 = GPT2LMHeadModel.from_pretrained(model_name)
        
        for param in self.gpt2.parameters():
            param.requires_grad = False
            
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM, 
            inference_mode=False, 
            r=lora_r, 
            lora_alpha=lora_r * 2, 
            lora_dropout=0.1
        )
        self.gpt2 = get_peft_model(self.gpt2, peft_config)
        self.gpt2.print_trainable_parameters()
        
        self.visual_proj = nn.Linear(input_dim, self.gpt2.config.n_embd)

    def forward(self, fusion_feat, target_ids=None):
        # 1. 视觉特征投影: [Batch, 1, Hidden]
        visual_embeds = self.visual_proj(fusion_feat).unsqueeze(1)
        
        if target_ids is not None:
            # === [核心修复] 训练模式：拼接 Visual + Text ===
            wte = self.gpt2.base_model.model.transformer.wte
            text_embeds = wte(target_ids) # [Batch, SeqLen, Hidden]
            
            # 拼接: [Visual, Token1, Token2, ...] -> 长度 1 + SeqLen
            inputs_embeds = torch.cat([visual_embeds, text_embeds], dim=1)
            
            # 传入 GPT-2 (只回传 logits 由外部算 loss)
            outputs = self.gpt2(inputs_embeds=inputs_embeds)
            return outputs.logits # [Batch, 1 + SeqLen, Vocab]
        else:
            # === 推理模式 (仅给图像) ===
            outputs = self.gpt2(inputs_embeds=visual_embeds)
            return outputs.logits # [Batch, 1, Vocab]

# ==========================================
# 新增：热图预测头 (Deconv Head)
# ==========================================
class GazeHeatmapHead(nn.Module):
    def __init__(self, input_dim=1024):
        super().__init__()
        # 1. 将 1D 向量投影回 2D 特征图 (8x8)
        # 我们选择 8x8 作为起始分辨率
        self.feat_h, self.feat_w = 8, 8
        self.feat_c = 256
        self.fc = nn.Linear(input_dim, self.feat_c * self.feat_h * self.feat_w)
        
        # 2. 上采样层 (Deconv): 8x8 -> 16x16 -> 32x32 -> 64x64
        self.deconv_layers = nn.Sequential(
            # Block 1: 8 -> 16
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            
            # Block 2: 16 -> 32
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            # Block 3: 32 -> 64 (输出单通道热图)
            nn.ConvTranspose2d(64, 1, kernel_size=4, stride=2, padding=1)
        )
        
    def forward(self, x):
        # x: [B, 1024]
        # 1. 投影并 Reshape
        x = self.fc(x)
        x = x.view(-1, self.feat_c, self.feat_h, self.feat_w)
        
        # 2. 上采样生成热图 logits
        heatmap = self.deconv_layers(x) # [B, 1, 64, 64]

        return torch.sigmoid(heatmap.squeeze(1))

# ==========================================
# 主分支整合: Following Branch
# ==========================================

class FollowingBranch(nn.Module):
    def __init__(self):
        super().__init__()
        
        # === 自动下载配置 ===
        self.ckpt_name = "sam_vit_b_01ec64.pth"
        self.ckpt_url = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"
        self.ensure_checkpoint()
        
        model_type = "vit_b"
        
        # 1. Image Encoder (传入 img_size=448 进行自动适配)
        self.scene_encoder = SAMImageEncoderReal(
            checkpoint_path=self.ckpt_name, 
            model_type=model_type, 
            lora_r=8,
            img_size=448  # <--- 必须显式指定为 448
        )
        
        # 2. Prompt Encoder
        self.prompt_encoder = SAMPromptEncoderReal(
            checkpoint_path=self.ckpt_name, 
            model_type=model_type,
            lora_r=8
        )
        
        # 3. Fusion Transformer
        self.sam_fusion = SAMFusionReal(
            checkpoint_path=self.ckpt_name, 
            model_type=model_type
        )
        
        # 4. LLM Decoder
        self.llm_decoder = LLMDecoderReal(input_dim=1024, model_name="gpt2", lora_r=8)
        
        # Adapter融合场景和头部特征
        self.adapter = nn.Sequential(
            nn.Linear(512, 1024),
            nn.LayerNorm(1024),
            nn.GELU(),  # GELU 通常比 ReLU 在 Transformer 架构下表现更好
            nn.Dropout(0.1), # 加一点 Dropout 防止过拟合
            
            nn.Linear(1024, 1024), # 增加一层深度的变换
            nn.LayerNorm(1024),
            nn.GELU()
        )
        
        # 6. Heads
        self.point_head = GazeHeatmapHead(input_dim=1024)
        
        self.inout_head = nn.Sequential(
            nn.Linear(1024, 256), nn.GELU(),
            nn.Linear(256, 1)
        )

    def ensure_checkpoint(self):
        if not os.path.exists(self.ckpt_name):
            print(f"SAM checkpoint '{self.ckpt_name}' not found.")
            try:
                torch.hub.download_url_to_file(self.ckpt_url, self.ckpt_name)
            except Exception as e:
                print(f"Download failed: {e}")
                raise e

    def forward(self, scene_img, obs_text_ids, face_bbox, eye_point, target_text_ids=None):
        # 1. Image Encode
        scene_feat = self.scene_encoder(scene_img) 
        
        # 2. Prompt Encode
        sparse_feat, _ = self.prompt_encoder(obs_text_ids, face_bbox, eye_point) 
        
        # 3. Fusion
        fusion_feat = self.sam_fusion(scene_feat, sparse_feat) 
        
        # 4. Combine
        scene_global = scene_feat.mean(dim=[-2, -1]) 
        final_feat_512 = torch.cat([scene_global, fusion_feat], dim=1)
        final_feat = self.adapter(final_feat_512) 
        
        # 5. Predict
        # [修改] 传入 target_text_ids 给 Decoder 用于拼接
        text_logits = self.llm_decoder(final_feat, target_ids=target_text_ids)
        
        pred_gaze_point = self.point_head(final_feat)
        pred_inout = self.inout_head(final_feat)
        
        return text_logits, pred_gaze_point, pred_inout, fusion_feat


# ==========================================
# 联合模型入口: Unified Gaze Model
# ==========================================

class UnifiedGazeModel(nn.Module):
    def __init__(self, save_dir=None, w_est_pretrain=False):
        super().__init__()
        self.estimation_branch = EstimationBranch(save_dir, w_est_pretrain)
        self.following_branch = FollowingBranch()

    def forward(self, batch_data):
        outputs = {}
        
        # === 1. Estimation Branch ===
        if 'face_img' in batch_data:
            est_angles, est_feat_256 = self.estimation_branch(batch_data['face_img'])
            outputs['pred_angles'] = est_angles
            outputs['est_feat_aligned'] = est_feat_256

        # === 2. Following Branch ===
        if 'scene_img' in batch_data and batch_data['scene_img'] is not None:
            obs_tokens = batch_data.get('observer_tokens', None)
            bbox = batch_data.get('face_bbox', None)
            eye = batch_data.get('eye_point_norm', None)
                    
            tgt_tokens = batch_data.get('gaze_point_expressions_ids', None)
                    
            text_out, point_out, inout_out, follow_feat = self.following_branch(
                batch_data['scene_img'], 
                obs_tokens, 
                bbox, 
                eye,
                target_text_ids=tgt_tokens 
            )
            
            outputs['pred_text_logits'] = text_out
            outputs['pred_gaze_point'] = point_out
            outputs['pred_inout'] = inout_out
            outputs['follow_feat'] = follow_feat

        return outputs