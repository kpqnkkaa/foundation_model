import torch
import torch.nn as nn
import torchvision
from timm.models.vision_transformer import Block
import math
import os
from gazelle.backbone import DinoV2Backbone, SAMBackboneWrapper, SAMImageEncoder, GazeTextDecoder
import gazelle.utils as utils

class GazeLLE(nn.Module):
    def __init__(self, backbone, inout=False, dim=256, num_layers=3, in_size=(448, 448), out_size=(64, 64), is_multi_output=False):
        super().__init__()
        self.backbone = backbone
        self.dim = dim
        self.num_layers = num_layers
        self.is_multi_output = is_multi_output

        # 获取 Backbone 输出特征图尺寸 (例如 448输入 -> 14x14 或 28x28)
        self.featmap_h, self.featmap_w = backbone.get_out_size(in_size)
        
        self.in_size = in_size
        self.out_size = out_size
        self.inout = inout
        
        # 1. 判定是否为 SAM Backbone
        # 依赖于 backbone.py 中的定义，或者 getattr 检查
        self.is_sam = getattr(backbone, 'is_sam', False)

        # 2. 基础组件 (所有模式共用)
        # 将 Backbone 特征 (如 DINOv2 的 768) 投影到模型维度 (256)
        self.linear = nn.Conv2d(backbone.get_dimension(), self.dim, 1)
        
        # 位置编码
        self.register_buffer("pos_embed", positionalencoding2d(self.dim, self.featmap_h, self.featmap_w))

        # 3. 分支逻辑构建
        if self.is_sam:
            # ================= SAM 分支初始化 =================
            # Tokens: 如果有 inout，则需要 2 个 token [InOut, Heatmap]，否则 1 个 [Heatmap]
            self.num_output_tokens = 2 if self.inout else 1
            self.output_tokens = nn.Embedding(self.num_output_tokens, self.dim)

            # 要输出分割，方向分类，expression的结果
            if self.is_multi_output:
                self.num_output_tokens = 3
                self.multi_output_tokens = nn.Embedding(self.num_output_tokens, self.dim)
            
            # Upscaling Head: 
            # 负责将融合后的图像特征 (256维) 上采样并降维到 32维
            # 这里的 32维 必须与 SAM 预训练 MLP 输出的权重维度一致
            self.output_upscaling = nn.Sequential(
                nn.ConvTranspose2d(dim, dim // 4, kernel_size=2, stride=2),
                nn.LayerNorm([dim // 4, self.featmap_h * 2, self.featmap_w * 2]),
                nn.GELU(),
                nn.ConvTranspose2d(dim // 4, dim // 8, kernel_size=2, stride=2),
                nn.GELU(),
            )
            # 注意：这里不再初始化 MLP，因为我们会直接使用 backbone.fusion 里的预训练 MLP

            # InOut 分类头 (如果需要)
            if self.inout:
                 self.inout_head = nn.Sequential(
                    nn.Linear(self.dim, 128),
                    nn.ReLU(),
                    nn.Dropout(0.1),
                    nn.Linear(128, 1),
                    nn.Sigmoid()
                )
            if self.is_multi_output:
                # 8分类
                self.direction_head = nn.Sequential(
                    nn.Linear(self.dim, 128),
                    nn.ReLU(),
                    nn.Dropout(0.1),
                    nn.Linear(128, 8)
                )
                self.text_head = GazeTextDecoder(input_dim=self.dim, model_name="gpt2", lora_r=8)

        else:
            # ================= Standard 分支初始化 =================
            self.head_token = nn.Embedding(1, self.dim)
            if self.inout: 
                self.inout_token = nn.Embedding(1, self.dim)
            
            # 标准 Transformer Decoder
            self.transformer = nn.Sequential(*[
                Block(
                    dim=self.dim, 
                    num_heads=8, 
                    mlp_ratio=4, 
                    drop_path=0.1)
                for i in range(num_layers)
            ])
            
            # 简单的反卷积头生成 Heatmap
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
        
        heatmap_preds = []
        inout_preds = None
        seg_preds = None
        direction_preds = None
        expression_preds = None

        # ==========================================
        #              SAM 逻辑分支
        # ==========================================
        if self.is_sam:
            # 将 Batch 维度展开为 Total_People 维度
            # x becomes: [Total_People, dim, H, W]
            x = utils.repeat_tensors(x, num_ppl_per_img)
            # A. 准备 Prompts (将归一化 bbox 转为绝对坐标)
            flat_bboxes = []
            flat_eyes = []
            for bbox_list in input["bboxes"]:
                for bbox in bbox_list:
                    xmin, ymin, xmax, ymax = bbox
                    flat_bboxes.append([
                        xmin * self.in_size[1], 
                        ymin * self.in_size[0], 
                        xmax * self.in_size[1], 
                        ymax * self.in_size[0]
                    ])
                    flat_eyes.append([
                        eye_x * self.in_size[1],
                        eye_y * self.in_size[0]
                    ])
            # [Total_People, 4]
            bboxes_tensor = torch.tensor(flat_bboxes, device=x.device, dtype=torch.float32)
            eyes_tensor = torch.tensor(flat_eyes, device=x.device, dtype=torch.float32)
            
            # B. 获取 Sparse Embeddings (通过 Backbone 的 Prompt Encoder)
            # [Total_People, N_sparse, dim]
            sparse_embeddings = self.backbone.prompt_encoder(bboxes_tensor, device=x.device, eyes=eyes_tensor, expr_ids=input["expr_ids"])
            
            # C. 准备 Learnable Tokens
            # tokens: [Total_People, Num_Tokens, dim]
            tokens = self.output_tokens.weight.unsqueeze(0).repeat(x.shape[0], 1, 1)
            if self.is_multi_output:
                multi_output_tokens = self.multi_output_tokens.weight.unsqueeze(0).repeat(x.shape[0], 1, 1)
                tokens = torch.cat((tokens, multi_output_tokens), dim=1)
            # 拼接: [Tokens, Sparse_Prompts]
            tokens = torch.cat((tokens, sparse_embeddings), dim=1)

            # D. Two-Way Fusion (双向交互)
            # src: 更新后的图像特征 [Total_People, dim, H, W]
            # hs:  更新后的 Tokens [Total_People, N_total, dim]
            # 注意：确保 backbone.py 中的 SAMFusion 返回顺序也是 (image, tokens)
            src, hs = self.backbone.fusion(image_embeddings=x, tokens=tokens)
            
            current_idx = 0 # 使用指针，不管是否有 inout 都能对齐
            
            # 1. InOut 任务
            if self.inout:
                inout_token_out = hs[:, current_idx, :]
                inout_preds_flat = self.inout_head(inout_token_out).squeeze(dim=-1)
                inout_preds = utils.split_tensors(inout_preds_flat, num_ppl_per_img)
                current_idx += 1
            
            # 2. Heatmap Token (必须存在)
            heatmap_token_out = hs[:, current_idx, :]
            current_idx += 1
            
            # 3. Multi Output Tokens
            seg_token_out = None
            if self.is_multi_output:
                seg_token_out = hs[:, current_idx, :]      # Seg
                direction_token_out = hs[:, current_idx+1, :] # Dir
                expression_token_out = hs[:, current_idx+2, :] # Expr
                current_idx += 3 # 虽然不用了，但保持习惯
                
                # 计算标量输出
                dir_flat = self.direction_head(direction_token_out).squeeze(dim=-1)
                expr_flat = self.expression_head(expression_token_out).squeeze(dim=-1)
                direction_preds = utils.split_tensors(dir_flat, num_ppl_per_img)
                expression_preds = utils.split_tensors(expr_flat, num_ppl_per_img)

            # F. 生成 Heatmap (Hypernetwork + Dot Product)
            
            # 1. 上采样图像特征 -> [Total_People, 32, H*4, W*4]
            upscaled_embedding = self.output_upscaling(src)
            b, c, h, w = upscaled_embedding.shape

            # 获取 SAM 内部所有的 MLP 层 (通常有 4 个)
            mlp_layers = self.backbone.fusion.output_hypernetworks_mlps

            # --- 生成 Heatmap (使用第 0 个 MLP) ---
            heatmap_mlp = mlp_layers[0] 
            hyper_weights = heatmap_mlp(heatmap_token_out)
            
            heatmap = (hyper_weights.unsqueeze(1) @ upscaled_embedding.view(b, c, h * w))
            heatmap = heatmap.view(b, 1, h, w)
            if heatmap.shape[-2:] != self.out_size:
                heatmap = F.interpolate(heatmap, size=self.out_size, mode='bilinear', align_corners=False)
            heatmap = torch.sigmoid(heatmap).squeeze(1)
            heatmap_preds = utils.split_tensors(heatmap, num_ppl_per_img)
            
            # --- 生成 Seg (使用第 1 个 MLP) ---
            if self.is_multi_output and seg_token_out is not None:
                # 关键：使用 index 1，与 Heatmap 区分开
                seg_mlp = mlp_layers[1] 
                
                seg_weights = seg_mlp(seg_token_out)
                
                seg = (seg_weights.unsqueeze(1) @ upscaled_embedding.view(b, c, h * w))
                seg = seg.view(b, 1, h, w)
                if seg.shape[-2:] != self.out_size:
                    seg = F.interpolate(seg, size=self.out_size, mode='bilinear', align_corners=False)
                seg = torch.sigmoid(seg).squeeze(1)
                seg_preds = utils.split_tensors(seg, num_ppl_per_img)

        # ==========================================
        #            Standard 逻辑分支
        # ==========================================
        else:
            x = self.linear(x) # [B, dim, H, W]
            x = x + self.pos_embed
            
            x = utils.repeat_tensors(x, num_ppl_per_img)

            # A. 叠加 Head Maps
            head_maps = torch.cat(self.get_input_head_maps(input["bboxes"]), dim=0).to(x.device) 
            head_map_embeddings = head_maps.unsqueeze(dim=1) * self.head_token.weight.unsqueeze(-1).unsqueeze(-1)
            x = x + head_map_embeddings

            # B. Flatten 准备进入 Transformer
            # "b c h w -> b (h w) c"
            x = x.flatten(start_dim=2).permute(0, 2, 1)
            
            if self.inout:
                inout_t = self.inout_token.weight.unsqueeze(dim=0).repeat(x.shape[0], 1, 1)
                x = torch.cat([inout_t, x], dim=1)
            
            # C. Transformer 交互
            x = self.transformer(x)

            # D. 解码 InOut
            if self.inout:
                inout_tokens = x[:, 0, :] 
                inout_preds_flat = self.inout_head(inout_tokens).squeeze(dim=-1)
                inout_preds = utils.split_tensors(inout_preds_flat, num_ppl_per_img)
                x = x[:, 1:, :] 
            
            # E. 解码 Heatmap
            # Reshape 回空间维度
            x = x.permute(0, 2, 1).reshape(x.shape[0], self.dim, self.featmap_h, self.featmap_w)
            
            x = self.heatmap_head(x)
            x = torchvision.transforms.functional.resize(x, self.out_size)
            x = x.squeeze(1)
            heatmap_preds = utils.split_tensors(x, num_ppl_per_img)

        return {"heatmap": heatmap_preds, 
                "inout": inout_preds,
                "seg": seg_preds,
                "direction": direction_preds,
                "expression": expression_preds}

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
    
    def get_gazelle_state_dict(self):
        return self.state_dict()

    def load_gazelle_state_dict(self, ckpt_state_dict):
        current_state_dict = self.state_dict()
        keys1 = current_state_dict.keys()
        keys2 = ckpt_state_dict.keys()

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
        # 新增sam模型
        "gazelle_sam_vitb": gazelle_sam_vitb,
        "sam_dinov2_vitb": sam_dinov2_vitb,
        "sam_dinov2_vitb_lora": sam_dinov2_vitb_lora,
        "sam_sam_vitb_lora": sam_sam_vitb_lora,
        "sam_dinov2_vitb_lora_multi_input": sam_dinov2_vitb_lora_multi_input,
        "sam_dinov2_vitb_lora_multi_input_inout": sam_dinov2_vitb_lora_multi_input_inout,
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
    # 自动获取路径，不存在则下载
    checkpoint = get_sam_checkpoint_path("vit_b")
    
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

def sam_dinov2_vitb_lora_multi_input():
    backbone = SAMBackboneWrapper(model_type="vit_b", in_size=(448, 448), backbone_type="dinov2", is_lora=True, is_multi_input=True)
    transform = backbone.get_transform((448, 448))
    model = GazeLLE(backbone, inout=False)
    return model, transform
    
def sam_dinov2_vitb_lora_multi_input_inout():
    backbone = SAMBackboneWrapper(model_type="vit_b", in_size=(448, 448), backbone_type="dinov2", is_lora=True, is_multi_input=True)
    transform = backbone.get_transform((448, 448))
    model = GazeLLE(backbone, inout=True, is_multi_output=True)
    return model, transform
