import os
# Set environment variable: Use domestic mirror for HuggingFace
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision.ops import StochasticDepth # [New] Required for Drop Path
import math
from typing import Optional, Tuple, List, Dict
import glob

# === Dependencies ===
from transformers import GPT2Model, GPT2LMHeadModel
from peft import LoraConfig, get_peft_model, TaskType

# ==========================================
# Helper Module: 2D Sinusoidal Position Embedding
# ==========================================
class PositionEmbeddingSine(nn.Module):
    """
    Standard 2D sinusoidal position embedding
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
# [New] Custom Transformer Layer with Drop Path
# ==========================================
class GazeLLETransformerLayer(nn.Module):
    """
    Standard Transformer Encoder Layer but with Stochastic Depth (Drop Path).
    Gaze-LLE paper specifies drop path p=0.1.
    """
    def __init__(self, d_model=256, nhead=8, dim_feedforward=1024, dropout=0.1, drop_path=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        
        # Feed Forward Network
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        # [New] Stochastic Depth (Drop Path)
        # mode='batch' means the entire batch is dropped or kept (common for consistent training)
        # mode='row' drops individual samples. 'batch' is safer if unsure.
        self.drop_path1 = StochasticDepth(p=drop_path, mode='row') if drop_path > 0 else nn.Identity()
        self.drop_path2 = StochasticDepth(p=drop_path, mode='row') if drop_path > 0 else nn.Identity()

        self.activation = nn.ReLU()

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        # 1. Self-Attention Block
        src2 = self.norm1(src)
        q = k = v = src2
        src2 = self.self_attn(q, k, v, attn_mask=src_mask, key_padding_mask=src_key_padding_mask)[0]
        
        # Apply Drop Path here (Residual Connection)
        src = src + self.drop_path1(self.dropout1(src2))
        
        # 2. Feed Forward Block
        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        
        # Apply Drop Path here (Residual Connection)
        src = src + self.drop_path2(self.dropout2(src2))
        
        return src

# ==========================================
# Branch 1: Estimation Branch (Unchanged)
# ==========================================
class EstimationBranch(nn.Module):
    def __init__(self, save_dir=None, w_est_pretrain=False):
        super().__init__()
        resnet = models.resnet50(pretrained=True)
        self.backbone = nn.Sequential(*list(resnet.children())[:-1]) 
        self.proj_256 = nn.Linear(2048, 256)
        self.head = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 2)
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
        feat = self.backbone(face_img)
        feat = torch.flatten(feat, 1)
        feat_256 = self.proj_256(feat) 
        angles = self.head(feat_256)
        return angles, feat_256

# ==========================================
# Component: DINOv2 Encoder
# ==========================================
class DINOv2ImageEncoder(nn.Module):
    def __init__(self, model_name="dinov2_vitb14"):
        super().__init__()
        print(f"Loading DINOv2 Model: {model_name}...")
        self.backbone = torch.hub.load('facebookresearch/dinov2', model_name)
        self.embed_dim = self.backbone.embed_dim
        for param in self.backbone.parameters():
            param.requires_grad = False

    def forward(self, x):
        B = x.shape[0]
        x = x.contiguous()
        features_dict = self.backbone.forward_features(x)
        patch_tokens = features_dict["x_norm_patchtokens"] 
        H, W = x.shape[2] // 14, x.shape[3] // 14
        feat_spatial = patch_tokens.permute(0, 2, 1).contiguous().reshape(B, self.embed_dim, H, W)
        return feat_spatial

# ==========================================
# Component: LLM Decoder
# ==========================================
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
# Core Branch: Following Branch (Gaze-LLE Architecture)
# ==========================================
class FollowingBranch(nn.Module):
    def __init__(self):
        super().__init__()
        
        # 1. Frozen DINOv2 Encoder (ViT-B/14)
        self.scene_encoder = DINOv2ImageEncoder(model_name="dinov2_vitb14")
        
        # 2. Linear Projection
        self.proj = nn.Conv2d(768, 256, kernel_size=1)
        
        # 3. Head Prompting Components
        self.p_head = nn.Parameter(torch.randn(1, 256, 1, 1) * 0.02)
        
        # 4. Positional Encoding
        self.pos_embed = PositionEmbeddingSine(num_pos_feats=128, normalize=True)
        
        # 5. Task Token (In/Out)
        self.task_token = nn.Parameter(torch.randn(1, 1, 256) * 0.02)
        
        # =========================================================
        # [Updated] 6. Gaze-LLE Decoder with Drop Path (p=0.1)
        # =========================================================
        # We manually stack 3 layers of our custom block
        self.layers = nn.ModuleList([
            GazeLLETransformerLayer(
                d_model=256, 
                nhead=8, 
                dim_feedforward=1024, 
                dropout=0.1,    # Standard Dropout
                drop_path=0.1   # [Fix] Drop Path Regularization
            ) for _ in range(3)
        ])
        
        # 7. Prediction Heads
        # (A) Heatmap Decoder (Aligned with Paper Table 13)
        self.heatmap_head = nn.Sequential(
            # ConvTranspose to upsample: 32 -> 64
            nn.ConvTranspose2d(256, 256, kernel_size=2, stride=2), 
            # Removed BN/ReLU to strictly match Table 13
            # nn.BatchNorm2d(256),
            # nn.ReLU(),
            # Final projection to 1 channel
            nn.Conv2d(256, 1, kernel_size=1),
            nn.Sigmoid()
        )
        
        # (B) In/Out Head (MLP)
        self.inout_head = nn.Sequential(
            nn.Linear(256, 128), 
            nn.ReLU(), 
            nn.Linear(128, 1)
        )
        
        # (C) Alignment Head
        self.align_head = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256)
        )
        
        # (D) Text Generation
        self.global_adapter = nn.Sequential(
            nn.Linear(256, 1024), nn.LayerNorm(1024), nn.GELU()
        )
        self.llm_decoder = LLMDecoderReal(input_dim=1024)

    def make_head_mask(self, face_bbox, H, W, device):
        B = face_bbox.shape[0]
        mask = torch.zeros((B, 1, H, W), device=device)
        scale = H / 448.0 
        x1 = (face_bbox[:, 0] * scale).long().clamp(0, W-1)
        y1 = (face_bbox[:, 1] * scale).long().clamp(0, H-1)
        x2 = (face_bbox[:, 2] * scale).long().clamp(0, W)
        y2 = (face_bbox[:, 3] * scale).long().clamp(0, H)
        
        for i in range(B):
            if x2[i] > x1[i] and y2[i] > y1[i]:
                mask[i, 0, y1[i]:y2[i], x1[i]:x2[i]] = 1.0
        return mask

    def forward(self, scene_img, obs_text_ids, face_bbox, eye_point, target_text_ids=None):
        B = scene_img.shape[0]
        device = scene_img.device
        
        # 1. Extract & Project
        features_raw = self.scene_encoder(scene_img)
        x_F = self.proj(features_raw)
        
        # 2. Head Prompting
        H, W = x_F.shape[2], x_F.shape[3]
        
        # Robustness check: if face_bbox is missing, default to no prompt
        if face_bbox is not None:
            M = self.make_head_mask(face_bbox, H, W, device)
            S = x_F + (M * self.p_head)
        else:
            S = x_F
            
        # 3. Transformer Reasoning
        P = self.pos_embed(S)
        S_in = S + P
        src = S_in.flatten(2).permute(0, 2, 1) # [B, HW, C]
        task_token = self.task_token.expand(B, -1, -1)
        
        # Input Token Sequence: [Task, Scene...]
        x = torch.cat([task_token, src], dim=1)
        
        # [Updated] Pass through Custom Transformer Layers
        for layer in self.layers:
            x = layer(x)
        
        tokens_out = x
        
        # 4. Decode
        task_token_out = tokens_out[:, 0, :]   # [B, 256]
        scene_tokens_out = tokens_out[:, 1:, :] # [B, 1024, 256]
        
        # (A) Heatmap
        spatial_map = scene_tokens_out.permute(0, 2, 1).view(B, 256, H, W)
        pred_heatmap = self.heatmap_head(spatial_map) 
        
        # (B) In/Out
        pred_inout = self.inout_head(task_token_out)
        
        # (C) Text & Alignment
        global_feat = scene_tokens_out.mean(dim=1) # [B, 256]
        align_feat = self.align_head(global_feat) 
        
        adapter_out = self.global_adapter(global_feat)
        text_logits = self.llm_decoder(adapter_out, target_ids=target_text_ids)
        
        return {
            "text_logits": text_logits,
            "pred_gaze_point": pred_heatmap,
            "pred_inout": pred_inout,
            "feat_256": None,           
            "pred_head_mask": None, 
            "pred_gaze_dir": None,
            "pred_gaze_dir_angle": None,
            "follow_feat": align_feat 
        }

# ==========================================
# Unified Model Entry Point
# ==========================================
class UnifiedGazeModel(nn.Module):
    def __init__(self, save_dir=None, w_est_pretrain=False):
        super().__init__()
        self.estimation_branch = EstimationBranch(save_dir, w_est_pretrain)
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
            outputs['pred_text_logits'] = text_out['text_logits']
            outputs['pred_gaze_point'] = text_out['pred_gaze_point']
            outputs['pred_inout'] = text_out['pred_inout']
            outputs['follow_feat'] = text_out['follow_feat'] 
            outputs['pred_head_mask'] = text_out['pred_head_mask']
            outputs['pred_gaze_dir_follow'] = text_out['pred_gaze_dir']

        return outputs