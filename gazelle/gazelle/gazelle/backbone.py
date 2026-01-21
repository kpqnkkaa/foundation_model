import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
from abc import ABC, abstractmethod
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torch.nn.functional as F

from segment_anything import sam_model_registry
from transformers import GPT2Model
from peft import LoraConfig, get_peft_model, TaskType

# Abstract Backbone class
class Backbone(nn.Module, ABC):
    def __init__(self):
        super(Backbone, self).__init__()
    
    @abstractmethod
    def forward(self, x):
        pass

    @abstractmethod
    def get_dimension(self):
        pass

    @abstractmethod
    def get_out_size(self, in_size):
        pass

    def get_transform(self):
        pass


# Official DINOv2 backbones from torch hub (https://github.com/facebookresearch/dinov2#pretrained-backbones-via-pytorch-hub)
class DinoV2Backbone(Backbone):
    def __init__(self, model_name, is_lora=False, lora_r=8):
        super(DinoV2Backbone, self).__init__()
        self.model = torch.hub.load('facebookresearch/dinov2', model_name)
        for param in self.model.parameters():
            param.requires_grad = False
        
        if is_lora:
            peft_config = LoraConfig(
                r=lora_r, 
                lora_alpha=lora_r * 2, 
                target_modules=["qkv"],  # SAM ViT 中的 attention 投影层通常叫 qkv
                lora_dropout=0.1, 
                bias="none"
            )   
            self.model = get_peft_model(self.model, peft_config)
            self.model.print_trainable_parameters()

    def forward(self, x):
        b, c, h, w = x.shape
        out_h, out_w = self.get_out_size((h, w))
        x = self.model.forward_features(x)['x_norm_patchtokens']
        x = x.view(x.size(0), out_h, out_w, -1).permute(0, 3, 1, 2) # "b (out_h out_w) c -> b c out_h out_w"
        # x = self.proj(x)
        return x
    
    def get_dimension(self):
        return self.model.embed_dim
    
    def get_out_size(self, in_size):
        h, w = in_size
        return (h // self.model.patch_size, w // self.model.patch_size)
    
    def get_transform(self, in_size):
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485,0.456,0.406],
                std=[0.229,0.224,0.225]
            ),
            transforms.Resize(in_size),
        ])

class SAMImageEncoder(nn.Module):
    def __init__(self,  model_type="vit_b", is_lora=False, lora_r=8, img_size=448):
        super().__init__()
        checkpoint_path = get_sam_checkpoint_path(model_type)
        if sam_model_registry is None: raise ImportError("No segment_anything found.")
        if LoraConfig is None: raise ImportError("No peft found. pip install peft")

        print(f"Loading SAM Image Encoder ({model_type}) from {checkpoint_path}...")
        # 加载完整模型提取 Image Encoder
        sam_model = sam_model_registry[model_type](checkpoint=checkpoint_path)
        self.image_encoder = sam_model.image_encoder
        self.patch_size = sam_model.image_encoder.patch_embed.proj.kernel_size[0] # usually 16

        # 调整位置编码
        if img_size != 1024:
            print(f"Resizing SAM pos_embed for {img_size}x{img_size} input...")
            self.adapt_pos_embed(img_size)

        # 清理不需要的部分
        del sam_model.prompt_encoder
        del sam_model.mask_decoder
        del sam_model
        
        # 冻结参数并注入 LoRA
        for param in self.image_encoder.parameters():
            param.requires_grad = False
            
        if is_lora:
            peft_config = LoraConfig(
                r=lora_r, 
                lora_alpha=lora_r * 2, 
                target_modules=["qkv"],  # SAM ViT 中的 attention 投影层通常叫 qkv
                lora_dropout=0.1, 
                bias="none"
            )
            self.image_encoder = get_peft_model(self.image_encoder, peft_config)
            self.image_encoder.print_trainable_parameters()

    def get_dimension(self):
        return 256

    def get_out_size(self, in_size):
        h, w = in_size
        return (h // self.patch_size, w // self.patch_size)

    def adapt_pos_embed(self, new_img_size):
        old_pos_embed = self.image_encoder.pos_embed.data 
        patch_size = 16 
        new_grid_size = new_img_size // patch_size
        
        # [1, H, W, C] -> [1, C, H, W]
        permuted_pos_embed = old_pos_embed.permute(0, 3, 1, 2)
        new_pos_embed = F.interpolate(
            permuted_pos_embed, size=(new_grid_size, new_grid_size),
            mode='bicubic', align_corners=False
        )
        # [1, C, H, W] -> [1, H, W, C]
        new_pos_embed = new_pos_embed.permute(0, 2, 3, 1)
        self.image_encoder.pos_embed = nn.Parameter(new_pos_embed)

    def forward(self, x):
        # SAM Image Encoder forward
        return self.image_encoder(x)


class SAMPromptEncoder(nn.Module):
    def __init__(self, model_type="vit_b", is_multi_input=False, is_lora=False, lora_r=8):
        super().__init__()
        checkpoint_path = get_sam_checkpoint_path(model_type)
        sam_model = sam_model_registry[model_type](checkpoint=checkpoint_path)
        self.prompt_encoder = sam_model.prompt_encoder
        self.is_multi_input = is_multi_input
        del sam_model.image_encoder
        del sam_model.mask_decoder
        del sam_model
        
        # # 通常 Prompt Encoder 不需要训练，或者是跟随整体微调
        for param in self.prompt_encoder.parameters():
            param.requires_grad = False # 保持冻结，或者是 True 取决于你的策略
        if is_multi_input:
            self.text_encoder = GPT2Model.from_pretrained("gpt2")
            for param in self.text_encoder.parameters():
                param.requires_grad = False
            if is_lora:
                peft_config = LoraConfig(
                    task_type=TaskType.FEATURE_EXTRACTION, 
                    r=lora_r, lora_alpha=lora_r, 
                    target_modules=["c_attn"], lora_dropout=0.1, bias="none"
                )
                self.text_encoder = get_peft_model(self.text_encoder, peft_config)
                print("Text Encoder Trainable Parameters:")
                self.text_encoder.print_trainable_parameters()
            self.text_proj = nn.Linear(768, 256) 

    def forward(self, bboxes, device, eyes=None, expr_ids=None):
        """
        bboxes: Tensor [B, 4] 绝对坐标
        """
        if self.is_multi_input:
            assert expr_ids is not None, "expr_ids are required for multi-input"
        bs = bboxes.shape[0]
        b_coords = bboxes.reshape(bs, 2, 2)
        # 2: top-left, 3: bottom-right
        b_label = torch.tensor([2, 3], device=device).unsqueeze(0).repeat(bs, 1)
        if self.is_multi_input and eyes is not None:
            e_coords = eyes.unsqueeze(1) 
            e_label = torch.ones(bs, 1, device=device) 
            coords = torch.cat([e_coords, b_coords], dim=1)
            label = torch.cat([e_label, b_label], dim=1)
        else:
            coords = b_coords
            label = b_label
        
        sparse_embeddings, _ = self.prompt_encoder(
            points=(coords, label),
            boxes=None,
            masks=None
        )
        
        if self.is_multi_input:
            gpt_out = self.text_encoder(expr_ids)[0] # [B, L, 768]
            text_feat = gpt_out.mean(dim=1) # [B, 768]
            text_embed = self.text_proj(text_feat).unsqueeze(1) # [B, 1, 256]
            sparse_embeddings = torch.cat([sparse_embeddings, text_embed], dim=1)

        return sparse_embeddings


class SAMFusion(nn.Module):
    """
    SAM TwoWayTransformer: 负责 Image 和 Prompt 的交互
    """
    def __init__(self, model_type="vit_b"):
        super().__init__()
        checkpoint_path = get_sam_checkpoint_path(model_type)
        from segment_anything import sam_model_registry # 确保这里能引用到
        sam_model = sam_model_registry[model_type](checkpoint=checkpoint_path)
        self.transformer = sam_model.mask_decoder.transformer
        self.pe_layer = sam_model.prompt_encoder.pe_layer 
        self.output_hypernetworks_mlps = sam_model.mask_decoder.output_hypernetworks_mlps
        
        del sam_model.image_encoder
        del sam_model.prompt_encoder
        del sam_model
        
        for param in self.pe_layer.parameters():
            param.requires_grad = False

    def get_dense_pe(self, image_shape_hw):
        h, w = image_shape_hw
        return self.pe_layer((h, w)).unsqueeze(0)

    def forward(self, image_embeddings, tokens):
        """
        image_embeddings: [B, 256, H, W]
        tokens: [B, N, 256]
        """
        B, C, H, W = image_embeddings.shape
        image_pe = self.get_dense_pe((H, W)).to(image_embeddings.device).repeat(B, 1, 1, 1)
        # print(image_pe.shape)
        # print(image_embeddings.shape)
        
        sparse_encoded, dense_encoded = self.transformer(
            point_embedding=tokens,
            image_embedding=image_embeddings, 
            image_pe=image_pe 
        )
        
        # === 修复代码 ===
        # 原代码: dense_encoded = dense_encoded.permute(1, 2, 0).view(B, C, H, W)
        # 修改为 .reshape()，它会自动处理非连续内存
        dense_encoded = dense_encoded.permute(1, 2, 0).reshape(B, C, H, W)
        
        return sparse_encoded, dense_encoded

# --- Wrapper Class for Model Compatibility ---
class SAMBackboneWrapper(Backbone):
    def __init__(self, model_type="vit_b", lora_r=8, in_size=(448, 448), backbone_type="dinov2", is_lora=False, is_multi_input=False):
        super().__init__()
        self.model_type = model_type
        self.in_size = in_size
        
        # 实例化三个组件
        if backbone_type == "dinov2":
            if model_type == "vit_b":
                self.img_encoder = DinoV2Backbone('dinov2_vitb14', is_lora, lora_r)
            elif model_type == "vit_l":
                self.img_encoder = DinoV2Backbone('dinov2_vitl14', is_lora, lora_r)
        elif backbone_type == "sam":
            self.img_encoder = SAMImageEncoder(model_type, is_lora, lora_r, in_size[0])

        self.prompt_encoder = SAMPromptEncoder(model_type, is_multi_input, is_lora, lora_r)
        self.fusion = SAMFusion(model_type)
        
    def forward(self, x):
        # 仅用于兼容 Backbone 接口，实际逻辑在 model.py 中显式调用各个组件
        return self.img_encoder(x)
        
    def get_dimension(self): return self.img_encoder.get_dimension()
    def get_out_size(self, in_size):
        # SAM patch=16
        # return (in_size[0] // 16, in_size[1] // 16)
        # return 32, 32
        return self.img_encoder.get_out_size(in_size)
    
    def get_transform(self, in_size):
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            transforms.Resize(in_size),
        ])
   
# === 新增：SAM 权重自动下载逻辑 ===

SAM_URLS = {
    "vit_b": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth",
    "vit_l": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth",
    "vit_h": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth",
}

def get_sam_checkpoint_path(model_type):
    """
    检查 PyTorch 缓存中是否有权重，如果没有则自动下载。
    模仿 torch.hub 的行为。
    """
    if model_type not in SAM_URLS:
        raise ValueError(f"Invalid SAM model type: {model_type}. Options: {list(SAM_URLS.keys())}")

    url = SAM_URLS[model_type]
    # 获取 PyTorch Hub 的标准缓存目录 (通常是 ~/.cache/torch/hub/checkpoints)
    model_dir = os.path.join(torch.hub.get_dir(), "checkpoints")
    os.makedirs(model_dir, exist_ok=True)

    filename = os.path.basename(url)
    filepath = os.path.join(model_dir, filename)

    if not os.path.exists(filepath):
        print(f"Downloading SAM {model_type} checkpoint to {filepath} ...")
        # 使用 torch 提供的工具下载，支持进度条
        torch.hub.download_url_to_file(url, filepath)
    else:
        print(f"Found existing SAM checkpoint at {filepath}")

    return filepath
