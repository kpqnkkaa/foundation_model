import argparse
import glob
from functools import reduce
import os
import pandas as pd
import json
import numpy as np
from PIL import Image

parser = argparse.ArgumentParser()
parser.add_argument("--data_path", type=str, default="/mnt/nvme1n1/lululemon/xjj/datasets/resized/videoattentiontarget")
parser.add_argument("--fusion_path", type=str, default="/mnt/nvme1n1/lululemon/xjj/result/information_fusion")
args = parser.parse_args()

# preprocessing adapted from https://github.com/ejcgt/attention-target-detection/blob/master/dataset.py

def merge_dfs(ls):
    for i, df in enumerate(ls): # give columns unique names
        df.columns = [col if col == "path" else f"{col}_df{i}" for col in df.columns]
    merged_df = reduce(
        lambda left, right: pd.merge(left, right, on=["path"], how="outer"), ls
    )
    merged_df = merged_df.sort_values(by=["path"])
    merged_df = merged_df.reset_index(drop=True)
    return merged_df

def smooth_by_conv(window_size, df, col):
    """Temporal smoothing on labels to match original VideoAttTarget evaluation.
    Adapted from https://github.com/ejcgt/attention-target-detection/blob/acd264a3c9e6002b71244dea8c1873e5c5818500/utils/myutils.py"""
    values = df[col].values
    padded_track = np.concatenate([values[0].repeat(window_size // 2), values, values[-1].repeat(window_size // 2)])
    smoothed_signals = np.convolve(
        padded_track.squeeze(), np.ones(window_size) / window_size, mode="valid"
    )
    return smoothed_signals

def smooth_df(window_size, df):
    df["xmin"] = smooth_by_conv(window_size, df, "xmin")
    df["ymin"] = smooth_by_conv(window_size, df, "ymin")
    df["xmax"] = smooth_by_conv(window_size, df, "xmax")
    df["ymax"] = smooth_by_conv(window_size, df, "ymax")
    return df

def load_fusion_data(fusion_path, json_filename):
    full_path = os.path.join(fusion_path, json_filename)
    print(f"Loading fusion data from {full_path}...")
    with open(full_path, 'r') as f:
        data = json.load(f)
    
    # 构建字典：key=image_path (全局唯一路径), value=list of persons
    fusion_dict = {}
    for item in data:
        # 尝试获取 path 字段 (例如 "images/video1/clip1/00001.jpg")
        # 如果没有 path 字段，则回退到 index (但在VAT中这可能不安全)
        key = item.get('path') 
        if key is None and 'index' in item:
             # 仅作备选，VAT中建议必须有 path
             key = item['index'].split('_person')[0]

        if key:
            if key not in fusion_dict:
                fusion_dict[key] = []
            fusion_dict[key].append(item)
    return fusion_dict

def is_bbox_match(bbox1, bbox2, threshold=0.01):
    # 比较两个归一化 bbox 是否接近
    return all(abs(b1 - b2) < threshold for b1, b2 in zip(bbox1, bbox2))

def main(PATH):
    print("Loading Fusion JSONs...")
    fusion_train = load_fusion_data(args.fusion_path, "merged_VAT_train_EN_Qwen_Qwen3-VL-32B-Instruct_with_SEG.json")
    fusion_test = load_fusion_data(args.fusion_path, "merged_VAT_test_EN_Qwen_Qwen3-VL-32B-Instruct_with_SEG.json")

    # preprocess by sequence and person track
    splits = ["train", "test"]

    for split in splits:
        sequences = []
        max_num_ppl = 0
        seq_idx = 0
        for seq_path in glob.glob(
            os.path.join(PATH, "annotations", split, "*", "*")
        ):
            seq_img_path = os.path.join("images", *seq_path.split("/")[-2:]
            )
            sample_image = os.path.join(PATH, seq_img_path, os.listdir(os.path.join(PATH, seq_img_path))[0])
            width, height = Image.open(sample_image).size
            seq_dict = {"path": seq_img_path, "width": width, "height": height}
            frames = []
            person_files = glob.glob(os.path.join(seq_path, "*"))
            num_ppl = len(person_files)
            if num_ppl > max_num_ppl:
                max_num_ppl = num_ppl
            person_dfs = [
                pd.read_csv(
                    file,
                    header=None,
                    index_col=False,
                    names=["path", "xmin", "ymin", "xmax", "ymax", "gazex", "gazey"],
                )
                for file in person_files
            ]
            # moving-avg smoothing to match original benchmark's evaluation
            window_size = 11
            person_dfs = [smooth_df(window_size, df) for df in person_dfs]
            merged_df = merge_dfs(person_dfs) # merge annotations per person for same frames
            for frame_idx, row in merged_df.iterrows():
                frame_dict = {
                    "path": os.path.join(seq_img_path, row["path"]),
                    "heads": [],
                }
                p_idx = 0
                for i in range(1, num_ppl * 6 + 1, 6):
                    if not np.isnan(row.iloc[i]): # if it's nan lack of continuity (one person leaving the frame for a period of time)
                        xmin, ymin, xmax, ymax, gazex, gazey = row[i: i+6].values.tolist()
                        # match original benchmark's preprocessing of annotations
                        if gazex >=0 and gazey < 0:
                            gazey = 0
                        elif gazey >=0 and gazex < 0:
                            gazex = 0
                        inout = int(gazex >= 0 and gazey >= 0)

                        # move bboxes within frame if necessary
                        xmin = max(xmin, 0)
                        ymin = max(ymin, 0)
                        xmax = min(xmax, width)
                        ymax = min(ymax, height)

                        current_fusion_dict = fusion_train if split == "train" else fusion_test
                        
                        current_frame_path = os.path.join(seq_img_path, row["path"])
                        
                        # 3. 计算当前 bbox_norm 用于匹配
                        current_bbox_norm = [xmin / float(width), ymin / float(height), xmax / float(width), ymax / float(height)]

                        # 4. 初始化默认值
                        observer_expression_unique = []
                        gaze_direction = ""
                        gaze_point_expressions = []
                        seg_mask_path = ""

                        # 5. 查找匹配
                        if current_frame_path in current_fusion_dict:
                            candidates = current_fusion_dict[current_frame_path]
                            for cand in candidates:
                                if 'head_bbox_norm' not in cand:
                                    continue
                                
                                # 处理 BBox 格式 (Dict 转 List)
                                cand_bbox = cand['head_bbox_norm']
                                if isinstance(cand_bbox, dict):
                                    cand_bbox = [cand_bbox['x_min'], cand_bbox['y_min'], cand_bbox['x_max'], cand_bbox['y_max']]

                                if is_bbox_match(cand_bbox, current_bbox_norm):
                                    # 提取数据 (包含 NoneType 检查)
                                    if cand.get('observer_expression') is not None and 'unique' in cand['observer_expression']:
                                        observer_expression_unique = cand['observer_expression']['unique']
                                    
                                    if cand.get('gazes') is not None and len(cand['gazes']) > 0:
                                        gaze_info = cand['gazes'][0]
                                        gaze_direction = gaze_info.get('gaze_direction', "")
                                        gaze_point_expressions = gaze_info.get('gaze_point_expressions', [])
                                        seg_mask_path = gaze_info.get('seg_mask_path', "")
                                    break

                        frame_dict["heads"].append({
                            "bbox": [xmin, ymin, xmax, ymax],
                            "bbox_norm": [xmin / float(width), ymin / float(height), xmax / float(width), ymax / float(height)],
                            "gazex": [gazex],
                            "gazex_norm": [gazex / float(width)],
                            "gazey": [gazey],
                            "gazey_norm": [gazey / float(height)],
                            "inout": inout,
                            'observer_expression': observer_expression_unique,
                            'gaze_direction': gaze_direction,
                            'gaze_point_expressions': gaze_point_expressions,
                            'seg_mask_path': seg_mask_path
                        })
                    p_idx = p_idx + 1

                frames.append(frame_dict)
            seq_dict["frames"] = frames
            sequences.append(seq_dict)
            seq_idx += 1

        print("{} max people per image {}".format(split, max_num_ppl))
        print("{} num unique video sequences {}".format(split, len(sequences)))

        out_file = open(os.path.join(PATH, "{}_preprocessed.json".format(split)), "w")
        json.dump(sequences, out_file)

if __name__ == "__main__":
    main(args.data_path)