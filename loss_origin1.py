import torch
import torch.nn as nn
import torch.nn.functional as F

class GazeSystemLoss(nn.Module):
    def __init__(self, alpha_est_est=1.0, alpha_est_fol=1.0, alpha_follow_text=1.0, alpha_follow_point=1.0, alpha_follow_io=1.0, alpha_joint=0.5, ignore_index=50256, noise_tolerance=0.8):
        super().__init__()
        # 权重参数
        self.alpha_est_est = alpha_est_est
        self.alpha_est_fol = alpha_est_fol
        self.alpha_follow_text = alpha_follow_text
        self.alpha_follow_point = alpha_follow_point
        self.alpha_follow_io = alpha_follow_io
        self.alpha_joint = alpha_joint
        self.noise_tolerance = noise_tolerance

        # self.heatmap_loss_fn = nn.MSELoss()
        self.l1_loss = nn.L1Loss()
        self.ce_loss = nn.CrossEntropyLoss(ignore_index=ignore_index)
        self.bce_logits_loss = nn.BCEWithLogitsLoss()
    
    # [新函数] 加权 MSE Loss
    def weighted_mse_loss(self, input, target, weight_factor=20.0):
        # input: [B, 64, 64]
        # target: [B, 64, 64] (高斯图)
        
        # 1. 计算基础 MSE (不平均)
        loss = (input - target) ** 2
        
        # 2. 构造权重图
        # 如果 target > 0 (即在该位置有热图响应)，给予更大的权重
        # 这样模型会拼命想把“亮点”预测对，而不仅仅是把背景预测对
        weights = torch.ones_like(target)
        weights[target > 0.1] = weight_factor # 亮点区域权重 x20
        
        # 3. 加权平均
        loss = (loss * weights).mean()
        return loss

    def generate_gaussian_heatmap(self, centers, in_outs, size=64, sigma=3):
        with torch.no_grad():
            B = centers.shape[0]
            device = centers.device

            x = torch.arange(size, device=device).float()
            y = torch.arange(size, device=device).float()
            yy, xx = torch.meshgrid(y, x, indexing='ij') 
            
            cx = (centers[:, 0] * size).reshape(B, 1, 1)
            cy = (centers[:, 1] * size).reshape(B, 1, 1)
            
            dist_sq = (xx.unsqueeze(0) - cx)**2 + (yy.unsqueeze(0) - cy)**2
            heatmaps = torch.exp(-dist_sq / (2 * sigma**2)) 

            mask = in_outs.view(B, 1, 1).float()
            heatmaps = heatmaps * mask
            
            heatmaps = torch.clamp(heatmaps, 0.0, 1.0)
            return heatmaps

    # def angles_to_2d_vector(self, angles):
    #     """
    #     将 (Yaw, Pitch) 转换为 2D 平面投影向量 (dx, dy)
    #     angles: (Batch, 2) -> [yaw, pitch]
    #     """
    #     yaw = angles[:, 0]
    #     pitch = angles[:, 1]

    #     # 1. 按照 3D 几何公式计算投影分量
    #     # dx = -cos(pitch) * sin(yaw)  <-- 关键修正：加上 cos(pitch)
    #     # dy = -sin(pitch)
    #     # (注：负号取决于你的坐标系定义，通常 Gaze 数据集定义看向左/上为负)
        
    #     dx = -torch.cos(pitch) * torch.sin(yaw)
    #     dy = -torch.sin(pitch)

    #     # 2. 堆叠成 2D 向量
    #     vec = torch.stack([dx, dy], dim=1)

    #     # 3. 归一化 (只取方向，不关心长度)
    #     # 加上 1e-6 防止 0 除
    #     norm = torch.norm(vec, dim=1, keepdim=True) + 1e-6
    #     return vec / norm

    def forward(self, outputs, targets, task_types):
        total_loss = 0
        loss_dict = {}

        device = outputs.get('pred_angles', torch.tensor([])).device
        
        is_follow = torch.tensor([1 if t == 'Following' else 0 for t in task_types], device=device).bool()
        is_est = ~is_follow 

        # -----------------------------------------------------------
        # 1. Estimation Branch (3D Gaze)
        # -----------------------------------------------------------
        if 'pred_angles' in outputs:
            pred_angles = outputs['pred_angles']
            
            if '3D_est_ground_truth' in targets and is_est.any():
                pred_3d = pred_angles[is_est]
                gt_3d = targets['3D_est_ground_truth'][is_est]

                l_est_3d = self.l1_loss(pred_3d, gt_3d)

                total_loss += self.alpha_est_est * l_est_3d
                loss_dict['l_est_est'] = l_est_3d.item()

            # if '3D_fol_ground_truth' in targets and is_follow.any():
            #     pred_vec = pred_angles[is_follow]
            #     gt_vec = targets['3D_fol_ground_truth'][is_follow]

            #     # 同时inout得为1的才计算loss
            #     in_outs = targets['in_outs']
            #     valid_mask = in_outs.bool()
            #     if valid_mask.any():
            #         l_est_fol = self.l1_loss(pred_vec[valid_mask], gt_vec[valid_mask])
            #     else:
            #         l_est_fol = 0.0

            #     total_loss += self.alpha_est_fol * l_est_fol
            #     loss_dict['l_est_fol'] = l_est_fol.item()
            #     # loss_dict['l_est_fol'] = 0.0
            if '3D_fol_ground_truth' in targets and is_follow.any():
                # l_est_fol = 0.0
                pred_vec = pred_angles[is_follow]
                gt_vec = targets['3D_fol_ground_truth'][is_follow]
                in_outs = targets['in_outs']
                
                # 基础 Mask: 必须在画面内 (in_out=1)
                valid_mask = in_outs.bool().squeeze() # 确保是一维 [N]
                
                # 二次筛选: 根据 Loss 大小过滤噪声
                if valid_mask.any():
                    # 1. 提取有效样本
                    curr_pred = pred_vec[valid_mask]
                    curr_gt = gt_vec[valid_mask]
                    
                    # 2. 计算 per-sample loss (不要 mean, reduction='none')
                    # shape: [N_valid, 2] -> mean -> [N_valid]
                    raw_losses = F.l1_loss(curr_pred, curr_gt, reduction='none').mean(dim=1)
                    
                    # 3. 动态截断 (Truncation)
                    # 计算要保留的样本数量 (例如保留 Loss 最小的 80%)
                    num_valid = len(raw_losses)
                    num_keep = int(num_valid * self.noise_tolerance)
                    
                    if num_keep > 0:
                        # 排序，取 Loss 最小的前 num_keep 个
                        # topk(largest=False) 等同于取最小值
                        vals, _ = torch.topk(raw_losses, k=num_keep, largest=False)
                        l_est_fol = vals.mean()
                    else:
                        # 如果样本太少，就全部计算（避免除零或空）
                        l_est_fol = raw_losses.mean()
                else:
                    l_est_fol = torch.tensor(0.0, device=device)

                total_loss += self.alpha_est_fol * l_est_fol
                loss_dict['l_est_fol'] = l_est_fol.item()
            else:
                loss_dict['l_est_fol'] = 0.0

        # -----------------------------------------------------------
        # 2. Following Branch (Heatmap & Text)
        # -----------------------------------------------------------
        if 'pred_gaze_point' in outputs: 
            
            # A. Text Loss
            if 'pred_text_logits' in outputs and 'gaze_point_expressions_ids' in targets:
                pred_logits = outputs['pred_text_logits'] # [64, 1+L, V]
                gt_tokens = targets['gaze_point_expressions_ids'] # [64, L]
                
                if pred_logits.size(0) > 0:
                    seq_len = gt_tokens.size(1)
                    if pred_logits.size(1) >= seq_len:
                        # Logit[0](Image) -> GT[0]
                        pred_logits_aligned = pred_logits[:, :seq_len, :].contiguous()
                        gt_tokens_aligned = gt_tokens.contiguous()
                        
                        l_text = self.ce_loss(
                            pred_logits_aligned.view(-1, pred_logits_aligned.size(-1)), 
                            gt_tokens_aligned.view(-1)
                        )
                        total_loss += self.alpha_follow_text * l_text
                        loss_dict['l_text'] = l_text.item()
                    else:
                        loss_dict['l_text'] = 0.0

            # B. Gaze Point Loss (MSE)
            if 'gaze_points_norm' in targets:
                pred_map = outputs['pred_gaze_point'] 
                gt_pts = targets['gaze_points_norm'] 
                in_outs = targets['in_outs'].squeeze()

                target_map = self.generate_gaussian_heatmap(gt_pts, in_outs, size=64, sigma=3)

                
                l_point = self.weighted_mse_loss(pred_map, target_map, weight_factor=20.0)
                
                total_loss += self.alpha_follow_point * l_point
                loss_dict['l_point'] = l_point.item()

            # C. In/Out 二分类 Loss
            if 'in_outs' in targets:
                pred_io = outputs['pred_inout'].squeeze()
                if pred_io.ndim == 0: pred_io = pred_io.unsqueeze(0)
                gt_io = targets['in_outs'].squeeze().float()
                if gt_io.ndim == 0: gt_io = gt_io.unsqueeze(0)
                
                l_io = self.bce_logits_loss(pred_io, gt_io)
                    
                total_loss += self.alpha_follow_io * l_io
                loss_dict['l_io'] = l_io.item()

        # -----------------------------------------------------------
        # 3. Joint Loss (特征对齐)
        # -----------------------------------------------------------
        if 'est_feat_aligned' in outputs and 'follow_feat' in outputs:
            if is_follow.any():
                est_f = outputs['est_feat_aligned'][is_follow] 
                fol_f = outputs['follow_feat'] 
                l_joint = 1.0 - F.cosine_similarity(est_f, fol_f).mean()
                    
                total_loss += self.alpha_joint * l_joint
                loss_dict['l_joint'] = l_joint.item()
        
        return total_loss, loss_dict