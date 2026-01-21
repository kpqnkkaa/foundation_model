import argparse
import torch
from PIL import Image
import json
import os
import numpy as np
from sklearn.metrics import average_precision_score
from tqdm import tqdm

# 1. 强制设置可见显卡为 2
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

from gazelle.model import get_gazelle_model
from gazelle.utils import vat_auc, vat_l2

parser = argparse.ArgumentParser()
parser.add_argument("--data_path", type=str, default="/mnt/nvme1n1/lululemon/xjj/datasets/resized/videoattentiontarget")
parser.add_argument("--model_name", type=str, default="gazelle_dinov2_vitb14_inout")
parser.add_argument("--ckpt_path", type=str, default="./experiments/train_gazelle_vitb_vat/2026-01-16_19-24-24/best.pt")
parser.add_argument("--batch_size", type=int, default=64)
parser.add_argument("--output_file", type=str, default="test.txt", help="Path to save the output results")
args = parser.parse_args()

class VideoAttentionTarget(torch.utils.data.Dataset):
    def __init__(self, path, img_transform):
        self.sequences = json.load(open(os.path.join(path, "test_preprocessed.json"), "rb"))
        self.frames = []
        for i in range(len(self.sequences)):
            for j in range(len(self.sequences[i]['frames'])):
                self.frames.append((i, j))
        self.path = path
        self.transform = img_transform

    def __getitem__(self, idx):
        seq_idx, frame_idx = self.frames[idx]
        seq = self.sequences[seq_idx]
        frame = seq['frames'][frame_idx]
        image = self.transform(Image.open(os.path.join(self.path, frame['path'])).convert("RGB"))
        bboxes = [head['bbox_norm'] for head in frame['heads']]
        gazex = [head['gazex_norm'] for head in frame['heads']]
        gazey = [head['gazey_norm'] for head in frame['heads']]
        inout = [head['inout'] for head in frame['heads']]

        return image, bboxes, gazex, gazey, inout

    def __len__(self):
        return len(self.frames)
    
def collate(batch):
    images, bboxes, gazex, gazey, inout = zip(*batch)
    return torch.stack(images), list(bboxes), list(gazex), list(gazey), list(inout)


@torch.no_grad()
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Running on {} (Physical GPU: 2)".format(device))

    model, transform = get_gazelle_model(args.model_name)
    # 使用 map_location='cpu' 确保加载安全
    model.load_gazelle_state_dict(torch.load(args.ckpt_path, map_location='cpu'))
    model.to(device)
    model.eval()

    dataset = VideoAttentionTarget(args.data_path, transform)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, collate_fn=collate)

    aucs = []
    l2s = []
    inout_preds = []
    inout_gts = []

    for _, (images, bboxes, eyes, gazex, gazey, inout) in tqdm(enumerate(dataloader), desc="Evaluating", total=len(dataloader)):
        preds = model.forward({"images": images.to(device), "bboxes": bboxes, "eyes": None, "expr_ids": observer_expressions})
        
        # eval each instance (head)
        for i in range(images.shape[0]): # per image
            for j in range(len(bboxes[i])): # per head
                # 将 Tensor 移回 CPU 进行 metric 计算
                heatmap_pred = preds['heatmap'][i][j].cpu()
                
                if inout[i][j] == 1: # in frame
                    auc = vat_auc(heatmap_pred, gazex[i][j][0], gazey[i][j][0])
                    l2 = vat_l2(heatmap_pred, gazex[i][j][0], gazey[i][j][0])
                    aucs.append(auc)
                    l2s.append(l2)
                
                inout_preds.append(preds['inout'][i][j].item())
                inout_gts.append(inout[i][j])

    # 计算最终指标
    final_auc = np.array(aucs).mean()
    final_l2 = np.array(l2s).mean()
    final_ap = average_precision_score(inout_gts, inout_preds)

    # 格式化输出
    result_str = (
        f"Results for model: {args.model_name}\n"
        f"Checkpoint: {args.ckpt_path}\n"
        f"----------------------------\n"
        f"AUC: {final_auc:.4f}\n"
        f"Avg L2: {final_l2:.4f}\n"
        f"Inout AP: {final_ap:.4f}\n"
    )

    # 打印并写入文件
    print("\n" + result_str)
    
    with open(args.output_file, "w") as f:
        f.write(result_str)

    print(f"Results have been saved to {args.output_file}")

        
if __name__ == "__main__":
    main()