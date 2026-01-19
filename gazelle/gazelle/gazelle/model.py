import torch
import torch.nn as nn
import torchvision
from timm.models.vision_transformer import Block
import math
import os
from gazelle.backbone import DinoV2Backbone, SAMBackboneWrapper
import gazelle.utils as utils

class GazeLLE(nn.Module):
    def __init__(self, backbone, inout=False, dim=256, num_layers=3, in_size=(448, 448), out_size=(64, 64)):
        super().__init__()
        self.backbone = backbone
        self.dim = dim
        self.num_layers = num_layers
        self.featmap_h, self.featmap_w = backbone.get_out_size(in_size)
        self.in_size = in_size
        self.out_size = out_size
        self.inout = inout
        
        # 判定是否为 SAM Backbone
        self.is_sam = isinstance(backbone, SAMBackboneWrapper)
        # self.is_sam = False
        print("is_sam: ", self.is_sam)
        if self.is_sam:
            self.fusion_scale = nn.Parameter(torch.zeros(1)) # 初始化为0，或者很小的值

        self.linear = nn.Conv2d(backbone.get_dimension(), self.dim, 1)
        self.head_token = nn.Embedding(1, self.dim)
        
        # Positional Embedding
        self.register_buffer("pos_embed", positionalencoding2d(self.dim, self.featmap_h, self.featmap_w).squeeze(dim=0).squeeze(dim=0))
        
        if self.inout: self.inout_token = nn.Embedding(1, self.dim)
        
        self.transformer = nn.Sequential(*[
            Block(
                dim=self.dim, 
                num_heads=8, 
                mlp_ratio=4, 
                drop_path=0.1)
                for i in range(num_layers)
                ])
        
        self.heatmap_head = nn.Sequential(
            nn.ConvTranspose2d(dim, dim, kernel_size=2, stride=2),
            nn.Conv2d(dim, 1, kernel_size=1, bias=False),
            nn.Sigmoid()
        )
        
        if self.inout: 
            self.inout_head = nn.Sequential(
                nn.Linear(self.dim, 128),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(128, 1),
                nn.Sigmoid()
            )

    def forward(self, input):
        # input["images"]: [B, 3, H, W]
        # input["bboxes"]: list of lists of bbox tuples (normalized 0-1)

        num_ppl_per_img = [len(bbox_list) for bbox_list in input["bboxes"]]
        
        # 1. 提取基础图像特征 [B, C, H, W]
        x = self.backbone.forward(input["images"])
        
        x = self.linear(x) # [B or Total_People, dim, H, W]
        x = x + self.pos_embed

        x = utils.repeat_tensors(x, num_ppl_per_img)
        
        # 2. 特征处理分支 (SAM vs Standard)
        if self.is_sam:
            # === SAM 分支 ===
            # 准备 Prompts: 将归一化的 bbox 转换为绝对坐标并 flatten
            flat_bboxes = []
            for i, bbox_list in enumerate(input["bboxes"]):
                for bbox in bbox_list:
                    xmin, ymin, xmax, ymax = bbox
                    flat_bboxes.append([
                        xmin * self.in_size[1], 
                        ymin * self.in_size[0], 
                        xmax * self.in_size[1], 
                        ymax * self.in_size[0]
                    ])
            # print(self.in_size)
            
            bboxes_tensor = torch.tensor(flat_bboxes, device=x.device, dtype=torch.float32)
            
            # 获取 Prompt Embeddings [Total_People, N_sparse, C]
            sparse_embeddings = self.backbone.prompt_encoder(bboxes_tensor, device=x.device)
            
            # 执行 Fusion (TwoWayTransformer)
            # 输出 dense_encoded: [Total_People, C, H, W] - 这是融合了 Head 位置信息的特征图
            _, head_map_embeddings = self.backbone.fusion(image_embeddings=x, sparse_embeddings=sparse_embeddings)
            x = x + (self.fusion_scale * head_map_embeddings)
        else:
            # === Standard 分支后续 ===
            # print(num_ppl_per_img)
            # 生成并叠加 Head Maps
            head_maps = torch.cat(self.get_input_head_maps(input["bboxes"]), dim=0).to(x.device) 
            head_map_embeddings = head_maps.unsqueeze(dim=1) * self.head_token.weight.unsqueeze(-1).unsqueeze(-1)
            x = x + head_map_embeddings

        # --- 以下逻辑保持不变 ---
        
        # Flatten for Transformer: "b c h w -> b (h w) c"
        x = x.flatten(start_dim=2).permute(0, 2, 1)

        if self.inout:
            x = torch.cat([self.inout_token.weight.unsqueeze(dim=0).repeat(x.shape[0], 1, 1), x], dim=1)

        x = self.transformer(x)

        if self.inout:
            inout_tokens = x[:, 0, :] 
            inout_preds = self.inout_head(inout_tokens).squeeze(dim=-1)
            inout_preds = utils.split_tensors(inout_preds, num_ppl_per_img)
            x = x[:, 1:, :] 
        
        x = x.reshape(x.shape[0], self.featmap_h, self.featmap_w, x.shape[2]).permute(0, 3, 1, 2)
        x = self.heatmap_head(x).squeeze(dim=1)
        x = torchvision.transforms.functional.resize(x, self.out_size)
        heatmap_preds = utils.split_tensors(x, num_ppl_per_img)

        return {"heatmap": heatmap_preds, "inout": inout_preds if self.inout else None}

    def get_input_head_maps(self, bboxes):
        # bboxes: [[(xmin, ymin, xmax, ymax)]] - list of list of head bboxes per image
        head_maps = []
        for bbox_list in bboxes:
            img_head_maps = []
            for bbox in bbox_list:
                if bbox is None: # no bbox provided, use empty head map
                    img_head_maps.append(torch.zeros(self.featmap_h, self.featmap_w))
                else:
                    xmin, ymin, xmax, ymax = bbox
                    width, height = self.featmap_w, self.featmap_h
                    xmin = round(xmin * width)
                    ymin = round(ymin * height)
                    xmax = round(xmax * width)
                    ymax = round(ymax * height)
                    head_map = torch.zeros((height, width))
                    head_map[ymin:ymax, xmin:xmax] = 1
                    img_head_maps.append(head_map)
            head_maps.append(torch.stack(img_head_maps))
        return head_maps
    
    def get_gazelle_state_dict(self, include_backbone=False):
        if include_backbone:
            return self.state_dict()
        else:
            return {k: v for k, v in self.state_dict().items() if not k.startswith("backbone")}
        
    def load_gazelle_state_dict(self, ckpt_state_dict, include_backbone=False):
        current_state_dict = self.state_dict()
        keys1 = current_state_dict.keys()
        keys2 = ckpt_state_dict.keys()

        if not include_backbone:
            keys1 = set([k for k in keys1 if not k.startswith("backbone")])
            keys2 = set([k for k in keys2 if not k.startswith("backbone")])
        else:
            keys1 = set(keys1)
            keys2 = set(keys2)

        if len(keys2 - keys1) > 0:
            print("WARNING unused keys in provided state dict: ", keys2 - keys1)
        if len(keys1 - keys2) > 0:
            print("WARNING provided state dict does not have values for keys: ", keys1 - keys2)

        for k in list(keys1 & keys2):
            current_state_dict[k] = ckpt_state_dict[k]
        
        self.load_state_dict(current_state_dict, strict=False)


# From https://github.com/wzlxjtu/PositionalEncoding2D/blob/master/positionalembedding2d.py
def positionalencoding2d(d_model, height, width):
    """
    :param d_model: dimension of the model
    :param height: height of the positions
    :param width: width of the positions
    :return: d_model*height*width position matrix
    """
    if d_model % 4 != 0:
        raise ValueError("Cannot use sin/cos positional encoding with "
                         "odd dimension (got dim={:d})".format(d_model))
    pe = torch.zeros(d_model, height, width)
    # Each dimension use half of d_model
    d_model = int(d_model / 2)
    div_term = torch.exp(torch.arange(0., d_model, 2) *
                         -(math.log(10000.0) / d_model))
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
        # 新增 SAM 模型
        "sam_vitb": sam_vitb,
        "sam_vitb_inout": sam_vitb_inout,
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


# === 更新后的工厂函数 ===

def sam_vitb():
    # 自动获取路径，不存在则下载
    checkpoint = get_sam_checkpoint_path("vit_b")
    
    backbone = SAMBackboneWrapper(checkpoint_path=checkpoint, model_type="vit_b", in_size=(448, 448))
    transform = backbone.get_transform((448, 448))
    model = GazeLLE(backbone, inout=False)
    return model, transform

def sam_vitb_inout():
    # 自动获取路径，不存在则下载
    checkpoint = get_sam_checkpoint_path("vit_b")
    
    backbone = SAMBackboneWrapper(checkpoint_path=checkpoint, model_type="vit_b", in_size=(448, 448))
    transform = backbone.get_transform((448, 448))
    model = GazeLLE(backbone, inout=True)
    return model, transform