import argparse
import torch
from PIL import Image
import json
import os
import numpy as np
from tqdm import tqdm

from gazelle.model import get_gazelle_model
from gazelle.utils import gazefollow_auc, gazefollow_l2

# 强制设置可见显卡为 2
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

parser = argparse.ArgumentParser()
parser.add_argument("--data_path", type=str, default='/mnt/nvme1n1/lululemon/xjj/datasets/resized/gazefollow_extended')
parser.add_argument("--model_name", type=str, default="sam_vitb")
parser.add_argument("--ckpt_path", type=str, default="./experiments/train_gazelle_dinov2_vitb_sam_prompt_gazefollow/2026-01-19_01-16-03/best.pt")
parser.add_argument("--batch_size", type=int, default=128)
parser.add_argument("--output_file", type=str, default="test.txt", help="Path to save the output results")
args = parser.parse_args()

class GazeFollow(torch.utils.data.Dataset):
    def __init__(self, path, img_transform):
        self.images = json.load(open(os.path.join(path, "test_preprocessed.json"), "rb"))
        self.path = path
        self.transform = img_transform

    def __getitem__(self, idx):
        item = self.images[idx]
        image = self.transform(Image.open(os.path.join(self.path, item['path'])).convert("RGB"))
        height = item['height']
        width = item['width']
        bboxes = [head['bbox_norm'] for head in item['heads']]
        gazex = [head['gazex_norm'] for head in item['heads']]
        gazey = [head['gazey_norm'] for head in item['heads']]

        return image, bboxes, gazex, gazey, height, width

    def __len__(self):
        return len(self.images)
    
def collate(batch):
    images, bboxes, gazex, gazey, height, width = zip(*batch)
    return torch.stack(images), list(bboxes), list(gazex), list(gazey), list(height), list(width)


@torch.no_grad()
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Running on {}".format(device))

    model, transform = get_gazelle_model(args.model_name)
    # 增加 map_location='cpu' 防止权重加载时的设备不匹配
    model.load_gazelle_state_dict(torch.load(args.ckpt_path, map_location='cpu'))
    model.to(device)
    model.eval()

    dataset = GazeFollow(args.data_path, transform)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, collate_fn=collate)

    aucs = []
    min_l2s = []
    avg_l2s = []

    for _, (images, bboxes, gazex, gazey, height, width) in tqdm(enumerate(dataloader), desc="Evaluating", total=len(dataloader)):
        preds = model.forward({"images": images.to(device), "bboxes": bboxes})
        
        # eval each instance (head)
        for i in range(images.shape[0]): # per image
            for j in range(len(bboxes[i])): # per head
                # 确保 tensor 在 CPU 上进行 metric 计算
                heatmap_pred = preds['heatmap'][i][j].cpu()
                
                auc = gazefollow_auc(heatmap_pred, gazex[i][j], gazey[i][j], height[i], width[i])
                avg_l2, min_l2 = gazefollow_l2(heatmap_pred, gazex[i][j], gazey[i][j])
                aucs.append(auc)
                avg_l2s.append(avg_l2)
                min_l2s.append(min_l2)
    
    # 计算最终结果
    final_auc = np.array(aucs).mean()
    final_avg_l2 = np.array(avg_l2s).mean()
    final_min_l2 = np.array(min_l2s).mean()

    # 格式化输出字符串
    result_str = (
        f"Results for model: {args.model_name}\n"
        f"Checkpoint: {args.ckpt_path}\n"
        f"----------------------------\n"
        f"AUC: {final_auc:.4f}\n"
        f"Avg L2: {final_avg_l2:.4f}\n"
        f"Min L2: {final_min_l2:.4f}\n"
    )

    # 1. 打印到控制台
    print("\n" + result_str)

    # 2. 写入到 test.txt，它在ckpt_path的同级目录下
    with open(os.path.join(os.path.dirname(args.ckpt_path), args.output_file), 'w') as f:
        f.write(result_str)
    
    print(f"Results have been saved to {args.output_file}")

        
if __name__ == "__main__":
    main()