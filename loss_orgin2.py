import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import math
class GazeSystemLoss(nn.Module):
    def __init__(self, alpha_est_est=1.0, alpha_est_fol=1.0, alpha_follow_text=1.0, alpha_follow_point=1.0, alpha_follow_io=1.0, alpha_joint=0.5, alpha_head_mask=0.5, alpha_gaze_dir=0.5, alpha_gaze_dir_cls=1.0, alpha_gaze_front_back=1.0, ignore_index=50256, noise_tolerance=0.8):
        super().__init__()
        # 权重参数
        self.alpha_est_est = alpha_est_est
        self.alpha_est_fol = alpha_est_fol
        self.alpha_follow_text = alpha_follow_text
        self.alpha_follow_point = alpha_follow_point
        self.alpha_follow_io = alpha_follow_io
        self.alpha_joint = alpha_joint
        self.alpha_head_mask = alpha_head_mask
        self.alpha_gaze_dir = alpha_gaze_dir
        self.alpha_gaze_dir_cls = alpha_gaze_dir_cls
        self.alpha_gaze_front_back = alpha_gaze_front_back
        self.noise_tolerance = noise_tolerance

        # self.heatmap_loss_fn = nn.MSELoss()
        self.l1_loss = nn.L1Loss()
        self.ce_loss = nn.CrossEntropyLoss(ignore_index=ignore_index)
        self.bce_logits_loss = nn.BCEWithLogitsLoss()
        self.bce_loss = nn.BCELoss()

    def gaze_dir_to_cls(self, gaze_dir):
        # gaze_dir: [B] (radians)
        gaze_angle = gaze_dir * 180 / math.pi
        # 偏移 22.5 度，让 0 度对应 "右" 的中心
        gaze_angle = (gaze_angle + 22.5) % 360
        
        # 直接计算索引，而不是 One-hot
        # floor 除法得到 0~7 的整数
        cls_idx = (gaze_angle // 45).long()
        
        # 防止 360 度边缘情况 (虽然后面取模了通常没事，但为了保险可以 clamp)
        cls_idx = torch.clamp(cls_idx, 0, 7)
            
        return cls_idx
    
    # [新函数] 加权 MSE Loss
    def weighted_mse_loss(self, input, target, weight_factor=20.0):
        # input: [B, 56, 56]
        # target: [B, 56, 56] (高斯图)
        
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
    
    # [New] Helper to generate binary mask from bbox
    def generate_bbox_mask(self, bboxes, size=64):
        with torch.no_grad():
            B = bboxes.shape[0]
            device = bboxes.device
            mask = torch.zeros((B, 1, size, size), device=device)
            
            x1 = (bboxes[:, 0] * size).long().clamp(0, size-1)
            y1 = (bboxes[:, 1] * size).long().clamp(0, size-1)
            x2 = (bboxes[:, 2] * size).long().clamp(0, size)
            y2 = (bboxes[:, 3] * size).long().clamp(0, size)
            
            for i in range(B):
                if x2[i] > x1[i] and y2[i] > y1[i]:
                    mask[i, 0, y1[i]:y2[i], x1[i]:x2[i]] = 1.0
            return mask

    def generate_gaussian_heatmap(self, centers, in_outs, size=56, sigma=3):
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
            
            if '3D_est_ground_truth' in targets and is_est.any() and self.alpha_est_est > 1e-6:
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
            if '3D_fol_ground_truth' in targets and is_follow.any() and self.alpha_est_fol > 1e-6:
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
        # A. Text Loss
        if 'pred_text_logits' in outputs and 'gaze_point_expressions_ids' in targets and self.alpha_follow_text > 1e-6:
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
        if 'pred_gaze_point' in outputs and 'gaze_points_norm' in targets and self.alpha_follow_point > 1e-6:
            pred_map = outputs['pred_gaze_point'] 
            gt_pts = targets['gaze_points_norm'] 
            in_outs = targets['in_outs'].squeeze()

            size = pred_map.shape[-1]

            target_map = self.generate_gaussian_heatmap(gt_pts, in_outs, size=size, sigma=3)

            # # 可视化target_map
            # import matplotlib.pyplot as plt
            # import numpy as np
            # # NOTE: generate_gaussian_heatmap() 返回形状为 [B, H, W]
            # # 这里只做可视化拷贝，不能覆盖 target_map（后面还要用于 loss 计算）
            # target_map_np = target_map.detach().cpu().numpy()
            # os.makedirs('./tmp/gaze_point', exist_ok=True)
            # plt.figure(figsize=(10, 5))
            # if target_map_np.ndim == 4:
            #     # [B, 1, H, W]
            #     plt.imshow(target_map_np[0, 0, :, :], cmap='viridis')
            # elif target_map_np.ndim == 3:
            #     # [B, H, W]
            #     plt.imshow(target_map_np[0, :, :], cmap='viridis')
            # else:
            #     # [H, W] or unexpected
            #     plt.imshow(target_map_np, cmap='viridis')
            # plt.title('Target Gaze Point')
            # plt.savefig('./tmp/gaze_point/target_map.png')

            # pred_map 通常为 [B, 1, H, W]，对齐到 [B, H, W]
            if pred_map.dim() == 4 and pred_map.size(1) == 1:
                pred_map_for_loss = pred_map[:, 0, :, :]
            else:
                pred_map_for_loss = pred_map
            l_point = self.weighted_mse_loss(pred_map_for_loss, target_map, weight_factor=20.0)
            
            total_loss += self.alpha_follow_point * l_point
            loss_dict['l_point'] = l_point.item()

        # C. In/Out 二分类 Loss
        if 'pred_inout' in outputs and 'in_outs' in targets and self.alpha_follow_io > 1e-6:
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
        if 'est_feat_aligned' in outputs and 'follow_feat' in outputs and self.alpha_joint > 1e-6:
            if is_follow.any():
                est_f = outputs['est_feat_aligned'][is_follow] 
                fol_f = outputs['follow_feat'] 
                l_joint = 1.0 - F.cosine_similarity(est_f, fol_f).mean()
                    
                total_loss += self.alpha_joint * l_joint
                loss_dict['l_joint'] = l_joint.item()

        # ==========================================================
        # [New] D. Head Mask Loss (Stage 1 Supervision)
        # ==========================================================
        # Key check: 'pred_head_mask' from model output, 'face_bbox' from batch dict
        if 'pred_head_mask' in outputs and 'face_bbox_gt' in targets and self.alpha_head_mask > 1e-6:
            pred_mask = outputs['pred_head_mask'] 
            face_bbox = targets['face_bbox_gt']

            size = pred_mask.shape[-1]
            
            # Generate GT Mask
            target_mask = self.generate_bbox_mask(face_bbox, size=size)
            
            # 都是0，1，计算BCE Loss
            l_head_mask = self.bce_loss(pred_mask, target_mask)
            
            total_loss += self.alpha_head_mask * l_head_mask
            loss_dict['l_head_mask'] = l_head_mask.item()
            # #可视化head_mask
            # import matplotlib.pyplot as plt
            # import numpy as np
            # target_mask = target_mask.detach().cpu().numpy()
            # plt.figure(figsize=(10, 5))
            # plt.subplot(1, 2, 1)
            # plt.imshow(target_mask[0, 0, :, :], cmap='viridis')
            # plt.title('Predicted Head Mask')
            # # 保存到./tmp/head_mask/
            # os.makedirs('./tmp/head_mask/', exist_ok=True)
            # plt.savefig('./tmp/head_mask/head_mask.png')

        # ==========================================================
        # [修改后] E. Gaze Direction Loss (Stage 1 Supervision)
        # ==========================================================
        # 直接使用 Dataset 中算好的 '2D_fol_ground_truth'
        if 'pred_gaze_dir_follow' in outputs and '2D_angles_fol_ground_truth' in targets and self.alpha_gaze_dir > 1e-6:
            pred_dir = outputs['pred_gaze_dir_follow'] # [B, 1]
            gt_dir = targets['2D_angles_fol_ground_truth']    # [B, 1]
            in_outs = targets['in_outs'].squeeze()
            
            # 筛选有效样本:
            # 通常只对画面内 (in_out=1) 的样本计算方向 Loss，
            # 或者你可以根据你的 Dataset 逻辑，如果画面外也有有效方向 GT，可以去掉这个限制。
            valid_mask = (in_outs > 0.5)
            
            if valid_mask.any():
                l_gaze_dir = self.l1_loss(pred_dir[valid_mask], gt_dir[valid_mask].unsqueeze(1))
                total_loss += self.alpha_gaze_dir * l_gaze_dir
                loss_dict['l_gaze_dir'] = l_gaze_dir.item()
            else:
                loss_dict['l_gaze_dir'] = 0.0

        if 'pred_gaze_dir_cls' in outputs and '2D_angles_fol_ground_truth' in targets and self.alpha_gaze_dir_cls > 1e-6:
            pred_dir_cls = outputs['pred_gaze_dir_cls'] # [B, 8] Logits
            
            gt_dir_cls = self.gaze_dir_to_cls(targets['2D_angles_fol_ground_truth'])
            
            in_outs = targets['in_outs'].squeeze()
            valid_mask = (in_outs > 0.5)
            
            if valid_mask.any():
                l_gaze_dir_cls = self.ce_loss(pred_dir_cls[valid_mask], gt_dir_cls[valid_mask])
                
                total_loss += self.alpha_gaze_dir_cls * l_gaze_dir_cls
                loss_dict['l_gaze_dir_cls'] = l_gaze_dir_cls.item()
            else:
                loss_dict['l_gaze_dir_cls'] = 0.0

        if 'pred_gaze_front_back' in outputs and 'gaze_front_back' in targets and self.alpha_gaze_front_back > 1e-6:
            pred_front_back = outputs['pred_gaze_front_back'] # [B, 2]
            gt_front_back = targets['gaze_front_back'] # [B, 1]
            l_gaze_front_back = self.bce_loss(pred_front_back, gt_front_back)
        
        return total_loss, loss_dict