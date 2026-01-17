from abc import ABC, abstractmethod
import torch
import torch.nn as nn
import torchvision.transforms as transforms

from segment_anything import sam_model_registry
from peft import LoraConfig, get_peft_model

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
    def __init__(self, model_name):
        super(DinoV2Backbone, self).__init__()
        self.model = torch.hub.load('facebookresearch/dinov2', model_name)

    def forward(self, x):
        b, c, h, w = x.shape
        out_h, out_w = self.get_out_size((h, w))
        x = self.model.forward_features(x)['x_norm_patchtokens']
        x = x.view(x.size(0), out_h, out_w, -1).permute(0, 3, 1, 2) # "b (out_h out_w) c -> b c out_h out_w"
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
    def __init__(self, checkpoint_path, model_type="vit_b", lora_r=8, img_size=448):
        super().__init__()
        if sam_model_registry is None: raise ImportError("No segment_anything found.")
        if LoraConfig is None: raise ImportError("No peft found. pip install peft")

        print(f"Loading SAM Image Encoder ({model_type}) from {checkpoint_path}...")
        # 加载完整模型提取 Image Encoder
        sam_model = sam_model_registry[model_type](checkpoint=checkpoint_path)
        self.image_encoder = sam_model.image_encoder
        self.patch_size = sam_model.image_encoder.patch_embed.patch_size[0] # usually 16

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
            
        peft_config = LoraConfig(
            r=lora_r, 
            lora_alpha=lora_r * 2, 
            target_modules=["qkv"],  # SAM ViT 中的 attention 投影层通常叫 qkv
            lora_dropout=0.1, 
            bias="none"
        )
        self.image_encoder = get_peft_model(self.image_encoder, peft_config)
        self.image_encoder.print_trainable_parameters()

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
    def __init__(self, checkpoint_path, model_type="vit_b"):
        super().__init__()
        sam_model = sam_model_registry[model_type](checkpoint=checkpoint_path)
        self.prompt_encoder = sam_model.prompt_encoder
        del sam_model.image_encoder
        del sam_model.mask_decoder
        del sam_model
        
        # 通常 Prompt Encoder 不需要训练，或者是跟随整体微调
        # 如果需要训练，确保 requires_grad = True
        for param in self.prompt_encoder.parameters():
            param.requires_grad = False # 保持冻结，或者是 True 取决于你的策略

    def forward(self, bboxes, device):
        """
        bboxes: Tensor [B, 4] 绝对坐标
        """
        bs = bboxes.shape[0]
        b_coords = bboxes.reshape(bs, 2, 2)
        # 2: top-left, 3: bottom-right
        b_label = torch.tensor([2, 3], device=device).unsqueeze(0).repeat(bs, 1)
        
        sparse_embeddings, _ = self.prompt_encoder(
            points=(b_coords, b_label),
            boxes=None,
            masks=None
        )
        return sparse_embeddings


class SAMFusion(nn.Module):
    """
    SAM TwoWayTransformer: 负责 Image 和 Prompt 的交互
    """
    def __init__(self, checkpoint_path, model_type="vit_b"):
        super().__init__()
        sam_model = sam_model_registry[model_type](checkpoint=checkpoint_path)
        self.transformer = sam_model.mask_decoder.transformer
        # 需要 pe_layer 来生成 dense positional encoding
        self.pe_layer = sam_model.prompt_encoder.pe_layer 
        
        del sam_model.image_encoder
        del sam_model.prompt_encoder
        del sam_model
        
        # Fusion 层通常需要训练以适应 Gaze 任务
        for param in self.transformer.parameters():
            param.requires_grad = True
        for param in self.pe_layer.parameters():
            param.requires_grad = False

    def get_dense_pe(self, image_shape_hw):
        h, w = image_shape_hw
        # SAM 的 pe_layer 输出是 (C, H, W) 或者是 (H, W, C)? 
        # 查看源码 pe_layer(size) 返回 (C, H, W)
        return self.pe_layer((h, w)).unsqueeze(0)

    def forward(self, image_embeddings, sparse_embeddings):
        """
        image_embeddings: [B, 256, H, W]
        sparse_embeddings: [B, N, 256]
        """
        B, C, H, W = image_embeddings.shape
        image_pe = self.get_dense_pe((H, W)).to(image_embeddings.device).repeat(B, 1, 1, 1)
        
        # TwoWayTransformer 接受:
        # point_embedding (sparse) 作为 Queries
        # image_embedding (dense) 作为 Keys/Values
        # 输出:
        # sparse_encoded: 更新后的 Prompt query [B, N, C]
        # dense_encoded: 更新后的 Image feature [B, C, H, W] (这就是我们要的 Heatmap 特征源)
        sparse_encoded, dense_encoded = self.transformer(
            point_embedding=sparse_embeddings,
            image_embedding=image_embeddings, 
            image_pe=image_pe 
        )
        return sparse_encoded, dense_encoded


# --- Wrapper Class for Model Compatibility ---
class SAMBackboneWrapper(Backbone):
    def __init__(self, checkpoint_path, model_type="vit_b", lora_r=8, in_size=(448, 448)):
        super().__init__()
        self.model_type = model_type
        self.checkpoint_path = checkpoint_path
        self.in_size = in_size
        
        # 实例化三个组件
        self.img_encoder = SAMImageEncoder(checkpoint_path, model_type, lora_r, in_size[0])
        self.prompt_encoder = SAMPromptEncoder(checkpoint_path, model_type)
        self.fusion = SAMFusion(checkpoint_path, model_type)
        
    def forward(self, x):
        # 仅用于兼容 Backbone 接口，实际逻辑在 model.py 中显式调用各个组件
        return self.img_encoder(x)
        
    def get_dimension(self): return 256 # SAM vit_b
    def get_out_size(self, in_size):
        # SAM patch=16
        return (in_size[0] // 16, in_size[1] // 16)
    
    def get_transform(self, in_size):
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            transforms.Resize(in_size),
        ])