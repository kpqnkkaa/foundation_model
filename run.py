import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import numpy as np
from torchvision import transforms

# 引入你的模块
from dataset import GazeEstimationDataset, GazeFollowDataset, JointFusionDataset
from model import UnifiedGazeModel
from loss import GazeSystemLoss

# ==========================================
# 0. GPT-2 Tokenizer (用于处理文本描述)
# ==========================================
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com" # 确保能下载
from transformers import GPT2Tokenizer

class UnifiedTokenizer:
    """
    封装 HuggingFace GPT2Tokenizer，统一处理输入格式。
    """
    def __init__(self, max_len=20):
        self.max_len = max_len
        # 加载预训练的分词器
        print("Loading GPT2 Tokenizer...")
        self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        
        # GPT-2 默认没有 pad_token，需要手动指定 eos_token 作为 pad_token
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # 获取词表大小 (50257)
        self.vocab_size = self.tokenizer.vocab_size

    def encode(self, text_list):
        """
        将文本列表转换为 [Batch, MaxLen] 的 Tensor
        """
        # 数据清洗：处理可能存在的 None 或非字符串
        clean_texts = [t if isinstance(t, str) else "" for t in text_list]
        
        encoded = self.tokenizer(
            clean_texts,
            padding='max_length',
            truncation=True,
            max_length=self.max_len,
            return_tensors='pt'
        )
        return encoded['input_ids']

# 全局 Tokenizer 实例
tokenizer = UnifiedTokenizer(max_len=20)

# ==========================================
# 1. Collate Function (核心数据整理)
# ==========================================
def gaze_collate_fn(batch):
    """
    将 Dataset 返回的 list of dicts 整理成模型需要的 batch dict。
    Dataset 返回格式:
    [
        {
            "gaze_estimation": { ... },
            "gaze_following": { ... }
        },
        ...
    ]
    """
    # 初始化容器
    est_faces = []
    est_3d_gts = []
    
    fol_faces = [] # follow 任务里的人脸 (sub-est)
    fol_scenes = []
    fol_2d_gts = []
    fol_bboxes = []
    fol_eyes = []
    fol_obs_exprs = [] # observer expressions
    fol_gaze_exprs = [] # gaze point expressions (targets)
    fol_gaze_pts = [] # gaze points
    fol_inouts = []

    task_types = [] # 记录样本顺序用于 loss 计算

    for item in batch:
        # --- 处理 Estimation 数据 ---
        ge = item['gaze_estimation']
        est_faces.append(ge['face_img'])
        est_3d_gts.append(ge['3D_ground_truth'])
        task_types.append('Estimation') # 这一部分是 Est 任务

        # --- 处理 Following 数据 ---
        gf = item['gaze_following']
        gf_est = gf['estimation_subtask']
        gf_fol = gf['following_subtask']

        # Following 任务也有人脸输入 (用于 Est 分支)
        fol_faces.append(gf_est['face_img'])
        fol_2d_gts.append(gf_est['2D_ground_truth'])
        
        # Following 任务特有输入
        fol_scenes.append(gf_fol['scene_img'])
        fol_bboxes.append(gf_fol['face_bbox'])
        fol_eyes.append(gf_fol['eye_point_norm'])
        
        # 文本和列表数据处理
        fol_obs_exprs.append(gf_fol['observer_expression'])
        
        # Gaze Points 和 Expressions 可能是列表，这里简化逻辑：
        # 训练时通常取第一个有效的，或者随机取一个。
        # 这里为了 Batch 对齐，我们取列表中的第一个元素作为训练目标
        # (Dataset 中已经做了一次随机 choice, 这里 gaze_point_expressions 已经是 string list)
        fol_gaze_exprs.append(gf_fol['gaze_point_expressions'][0]) 
        fol_gaze_pts.append(torch.tensor(gf_fol['gaze_points_norm'][0], dtype=torch.float32))
        fol_inouts.append(torch.tensor(gf_fol['in_outs'][0][0], dtype=torch.float32))
        
        task_types.append('Following') # 这一部分是 Follow 任务

    # === 拼接数据 ===
    
    # 1. Face Images: 先放 Est 数据，再放 Fol 数据
    # Shape: [Batch_Est + Batch_Fol, 3, 224, 224]
    all_face_imgs = torch.stack(est_faces + fol_faces)

    # 2. Scene Images: 只有 Follow 数据有
    # Shape: [Batch_Fol, 3, 448, 448]
    if len(fol_scenes) > 0:
        all_scene_imgs = torch.stack(fol_scenes)
    else:
        all_scene_imgs = None

    # 3. Ground Truths (需要填充以对齐长度)
    # 3D GT (只有 Est 有，Fol 部分填 0)
    batch_est_len = len(est_faces)
    batch_fol_len = len(fol_faces)
    
    gt_3d = torch.stack(est_3d_gts)
    gt_3d_padded = torch.cat([gt_3d, torch.zeros(batch_fol_len, 2)]) # [Total, 2]

    # 2D GT (只有 Fol 有，Est 部分填 0 - 虽然 Est 也可以算 2D，但为了逻辑清晰先分开)
    gt_2d = torch.stack(fol_2d_gts)
    gt_2d_padded = torch.cat([torch.zeros(batch_est_len, 2), gt_2d]) # [Total, 2]

    # 4. Tokenization (文本转 ID)
    obs_tokens = tokenizer.encode(fol_obs_exprs) # [Batch_Fol, MaxLen]
    gaze_expr_tokens = tokenizer.encode(fol_gaze_exprs) # [Batch_Fol, MaxLen]

    # 5. 其他 Tensor 堆叠
    batch_bboxes = torch.stack(fol_bboxes)
    batch_eyes = torch.stack(fol_eyes)
    batch_gaze_pts = torch.stack(fol_gaze_pts)
    batch_inouts = torch.stack(fol_inouts)

    # === 构建返回字典 ===
    batch_dict = {
        # 通用输入 (Est Branch)
        "face_img": all_face_imgs,
        
        # Follow Branch 输入
        "scene_img": all_scene_imgs,
        "observer_tokens": obs_tokens,
        "face_bbox": batch_bboxes,
        "eye_point_norm": batch_eyes,
        
        # GTs
        "3D_ground_truth": gt_3d_padded,
        "2D_ground_truth": gt_2d_padded,
        "gaze_points_norm": batch_gaze_pts,
        "gaze_point_expressions_ids": gaze_expr_tokens, # 用于 Loss 计算 (LLM target)
        "in_outs": batch_inouts,
    }

    return batch_dict, task_types

# ==========================================
# 2. 主程序
# ==========================================
def main():
    # --- A. 数据集路径配置 ---
    # 请确保路径真实存在，或修改为你的测试路径
    eth_label = '/mnt/nvme1n1/lululemon/xjj/datasets/ETH-Gaze/Label/train_temp.label'
    eth_img = '/mnt/nvme1n1/lululemon/xjj/datasets/ETH-Gaze/Image'
    
    gaze360_label = '/mnt/nvme1n1/lululemon/xjj/datasets/Gaze360/Label/train.label'
    gaze360_img = '/mnt/nvme1n1/lululemon/xjj/datasets/Gaze360/Image'
    
    gf_json = '/mnt/nvme1n1/lululemon/xjj/result/information_fusion/merged_GazeFollow_train_EN_Qwen_Qwen3-VL-32B-Instruct.json'

    # --- B. 定义 Transform ---
    transform_face = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    transform_scene = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((448, 448)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    print("=== 1. 加载数据集 ===")
    try:
        # 1. 实例化子数据集
        ds_eth = GazeEstimationDataset(eth_label, eth_img, transform=transform_face)
        ds_g360 = GazeEstimationDataset(gaze360_label, gaze360_img, transform=transform_face)
        ds_gf = GazeFollowDataset(gf_json, transform_face=transform_face, transform_scene=transform_scene)
        
        # 2. 实例化联合数据集
        joint_dataset = JointFusionDataset(
            est_datasets_list=[ds_eth, ds_g360], 
            follow_datasets_list=[ds_gf]
        )
        print(f"联合数据集加载成功，总长度 (Epoch Length): {len(joint_dataset)}")
        
    except Exception as e:
        print(f"数据集加载失败，请检查路径。错误信息: {e}")
        return

    # --- C. DataLoader ---
    batch_size = 4
    train_loader = DataLoader(
        joint_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=0, # 调试时建议设为0
        collate_fn=gaze_collate_fn # 使用自定义的整理函数
    )

    # --- D. 模型与损失函数 ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=== 2. 初始化模型 (Device: {device}) ===")
    
    model = UnifiedGazeModel().to(device)
    criterion = GazeSystemLoss().to(device)
    
    # 优化器: 建议为不同部分设置不同学习率 (ResNet/SAM backbone 学习率可以低一点)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    # --- E. 训练循环模拟 (Training Loop) ---
    print("=== 3. 开始训练模拟 (Running 1 Batch) ===")
    model.train()
    
    for batch_idx, (batch_data, task_types) in enumerate(train_loader):
        # 1. 数据搬运到 GPU
        for k, v in batch_data.items():
            if isinstance(v, torch.Tensor):
                batch_data[k] = v.to(device)
        
        # 2. 前向传播
        # model 内部会自动处理 Est 和 Follow 的分支逻辑
        outputs = model(batch_data)
        
        # 3. 计算损失
        # loss.py 会根据 task_types 自动屏蔽无效的 loss 计算
        loss, loss_dict = criterion(outputs, batch_data, task_types)
        
        # 4. 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # 5. 打印日志
        print(f"\n[Batch {batch_idx}] Total Loss: {loss.item():.4f}")
        print("详细 Loss 组件:")
        for k, v in loss_dict.items():
            print(f"  - {k}: {v:.4f}")
            
        print("-" * 30)
        
        # 为了演示，只运行一个 Batch
        break

    # --- F. 推理演示 (Inference Demo) ---
    print("=== 4. 推理测试 (Inference) ===")
    model.eval()
    with torch.no_grad():
        # 复用刚才那个 batch 的数据进行推理
        # 假设我们需要对 Following 任务进行推理
        
        # 找出 batch 中属于 Following 的样本索引
        follow_indices = [i for i, t in enumerate(task_types) if t == 'Following']
        
        if follow_indices:
            print(f"检测到 {len(follow_indices)} 个 Following 样本，进行推理...")
            outputs = model(batch_data)
            
            # 获取预测结果 (注意：FollowingBranch 的输出只包含 Follow 样本)
            # 输出形状通常是 [B_fol, ...]
            pred_pts = outputs['pred_gaze_point']
            pred_inout = torch.sigmoid(outputs['pred_inout']) # 转概率
            pred_text_logits = outputs['pred_text_logits'] # [B, Seq, Vocab]
            
            # 简单解码文本 (Argmax)
            pred_token_ids = torch.argmax(pred_text_logits, dim=-1)
            
            for i in range(len(follow_indices)):
                print(f"\n样本 {i+1}:")
                print(f"  > 预测 Gaze Point (Norm): {pred_pts[i].cpu().numpy()}")
                print(f"  > 真实 Gaze Point (Norm): {batch_data['gaze_points_norm'][i].cpu().numpy()}")
                print(f"  > 预测 In/Out 概率: {pred_inout[i].item():.4f}")
                print(f"  > 预测 Text Tokens: {pred_token_ids[i].cpu().numpy()}")
        else:
            print("当前 Batch 中没有 Following 样本。")

if __name__ == "__main__":
    main()