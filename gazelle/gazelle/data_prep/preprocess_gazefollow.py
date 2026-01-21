import os
import pandas as pd
import json
from PIL import Image
import argparse

# preprocessing adapted from https://github.com/ejcgt/attention-target-detection/blob/master/dataset.py

parser = argparse.ArgumentParser()
parser.add_argument("--data_path", type=str, default="/mnt/nvme1n1/lululemon/xjj/datasets/resized/gazefollow_extended")
parser.add_argument("--fusion_path", type=str, default="/mnt/nvme1n1/lululemon/xjj/result/information_fusion")
args = parser.parse_args()

def load_fusion_data(fusion_path, json_filename):
    full_path = os.path.join(fusion_path, json_filename)
    print(f"Loading fusion data from {full_path}...")
    with open(full_path, 'r') as f:
        data = json.load(f)
    
    # 构建字典：key=image_id (例如 '00032486'), value=list of persons
    fusion_dict = {}
    for item in data:
        # 提取图片ID，假设 index 格式为 "00032486_person0"
        image_id = item['index'].split('_')[0] 
        if image_id not in fusion_dict:
            fusion_dict[image_id] = []
        fusion_dict[image_id].append(item)
    return fusion_dict

def is_bbox_match(bbox1, bbox2, threshold=0.01):
    return all(abs(b1 - b2) < threshold for b1, b2 in zip(bbox1, bbox2))

def main(DATA_PATH):

    # TRAIN
    train_fusion_file = "merged_GazeFollow_train_EN_Qwen_Qwen3-VL-32B-Instruct_with_SEG.json"
    fusion_dict_train = load_fusion_data(args.fusion_path, train_fusion_file)

    train_csv_path = os.path.join(DATA_PATH, "train_annotations_release.txt")
    column_names = ['path', 'idx', 'body_bbox_x', 'body_bbox_y', 'body_bbox_w', 'body_bbox_h', 'eye_x', 'eye_y',
                                'gaze_x', 'gaze_y', 'bbox_x_min', 'bbox_y_min', 'bbox_x_max', 'bbox_y_max', 'inout', 'source', 'meta']
    df = pd.read_csv(train_csv_path, header=None, names=column_names, index_col=False)
    df = df[df['inout'] != -1]
    df = df.groupby("path").agg(list) # aggregate over frames

    multiperson_ex = 0
    TRAIN_FRAMES = []
    for path, row in df.iterrows():
        img_path = os.path.join(DATA_PATH, path)
        img = Image.open(img_path)
        width, height = img.size

        num_people = len(row['idx'])
        if num_people > 1:
            multiperson_ex += 1
        heads = []

        for i in range(num_people):
            xmin, ymin, xmax, ymax = row['bbox_x_min'][i], row['bbox_y_min'][i], row['bbox_x_max'][i], row['bbox_y_max'][i]
            gazex = row['gaze_x'][i] * float(width)
            gazey = row['gaze_y'][i] * float(height)
            gazex_norm = row['gaze_x'][i]
            gazey_norm = row['gaze_y'][i]


            if xmin > xmax:
                temp = xmin
                xmin = xmax
                xmax = temp
            if ymin > ymax:
                temp = ymin
                ymin = ymax
                ymax = temp

            # move in out of frame bbox annotations
            xmin = max(xmin, 0)
            ymin = max(ymin, 0)
            xmax = min(xmax, width)
            ymax = min(ymax, height)

            current_bbox_norm = [xmin / float(width), ymin / float(height), xmax / float(width), ymax / float(height)]
            observer_expression_unique = []
            gaze_direction = ""
            gaze_point_expressions = []
            seg_mask_path = ""

            image_id_str = os.path.basename(path).split('.')[0]
            if image_id_str in fusion_dict_train:
                candidates = fusion_dict_train[image_id_str]
                for cand in candidates:
                    # [新增] 处理字典格式，转为列表 [xmin, ymin, xmax, ymax]
                    cand_bbox = cand['head_bbox_norm']
                    if isinstance(cand_bbox, dict):
                        cand_bbox = [cand_bbox['x_min'], cand_bbox['y_min'], cand_bbox['x_max'], cand_bbox['y_max']]
                    
                    # 使用转换后的 cand_bbox 进行比较
                    if is_bbox_match(cand_bbox, current_bbox_norm):
                        # 提取数据
                        if 'observer_expression' in cand and 'unique' in cand['observer_expression']:
                            observer_expression_unique = cand['observer_expression']['unique']
                        
                        if 'gazes' in cand and len(cand['gazes']) > 0:
                            gaze_info = cand['gazes'][0]
                            gaze_direction = gaze_info.get('gaze_direction', "")
                            gaze_point_expressions = gaze_info.get('gaze_point_expressions', [])
                            seg_mask_path = gaze_info.get('seg_mask_path', "")
                        break # 找到对应的人，停止查找

            heads.append({
                'bbox': [xmin, ymin, xmax, ymax],
                'bbox_norm': [xmin / float(width), ymin / float(height), xmax / float(width), ymax / float(height)],
                'inout': row['inout'][i],
                'gazex': [gazex], # convert to list for consistency with multi-annotation format
                'gazey': [gazey],
                'gazex_norm': [gazex_norm],
                'gazey_norm': [gazey_norm],
                'head_id': i,
                'observer_expression': observer_expression_unique,
                'gaze_direction': gaze_direction,
                'gaze_point_expressions': gaze_point_expressions,
                'seg_mask_path': seg_mask_path
            })
        TRAIN_FRAMES.append({
            'path': path,
            'heads': heads,
            'num_heads': num_people,
            'width': width,
            'height': height,
        })

    print("Train set: {} frames, {} multi-person".format(len(TRAIN_FRAMES), multiperson_ex))
    out_file = open(os.path.join(DATA_PATH, "train_preprocessed.json"), "w")
    json.dump(TRAIN_FRAMES, out_file)

    # TEST
    test_fusion_file = "merged_GazeFollow_test_EN_Qwen_Qwen3-VL-32B-Instruct_with_SEG.json"
    fusion_dict_test = load_fusion_data(args.fusion_path, test_fusion_file)

    test_csv_path = os.path.join(DATA_PATH, "test_annotations_release.txt")
    column_names = ['path', 'idx', 'body_bbox_x', 'body_bbox_y', 'body_bbox_w', 'body_bbox_h', 'eye_x', 'eye_y',
                                'gaze_x', 'gaze_y', 'bbox_x_min', 'bbox_y_min', 'bbox_x_max', 'bbox_y_max', 'source', 'meta']
    df = pd.read_csv(test_csv_path, header=None, names=column_names, index_col=False)

    TEST_FRAME_DICT = {}
    df = df.groupby(["path", "eye_x"]).agg(list) # aggregate over frames
    for id, row in df.iterrows(): # aggregate by frame
        path, _ = id
        if path in TEST_FRAME_DICT.keys():
            TEST_FRAME_DICT[path].append(row)
        else:
            TEST_FRAME_DICT[path] = [row]

    multiperson_ex = 0
    TEST_FRAMES = []
    for path in TEST_FRAME_DICT.keys():
        img_path = os.path.join(DATA_PATH, path)
        img = Image.open(img_path)
        width, height = img.size

        item = TEST_FRAME_DICT[path]
        num_people = len(item)
        heads = []

        for i in range(num_people):
            row = item[i]
            assert(row['bbox_x_min'].count(row['bbox_x_min'][0]) == len(row['bbox_x_min'])) # quick check that all bboxes are equivalent
            xmin, ymin, xmax, ymax = row['bbox_x_min'][0], row['bbox_y_min'][0], row['bbox_x_max'][0], row['bbox_y_max'][0]
            
            if xmin > xmax:
                temp = xmin
                xmin = xmax
                xmax = temp
            if ymin > ymax:
                temp = ymin
                ymin = ymax
                ymax = temp
            
            # move in out of frame bbox annotations
            xmin = max(xmin, 0)
            ymin = max(ymin, 0)
            xmax = min(xmax, width)
            ymax = min(ymax, height)

            current_bbox_norm = [xmin / float(width), ymin / float(height), xmax / float(width), ymax / float(height)]
            
            observer_expression_unique = []
            gaze_direction = ""
            gaze_point_expressions = []
            seg_mask_path = ""
            
            image_id_str = os.path.basename(path).split('.')[0]

            if image_id_str in fusion_dict_test:
                candidates = fusion_dict_test[image_id_str]
                for cand in candidates:
                    cand_bbox = cand['head_bbox_norm']
                    if isinstance(cand_bbox, dict):
                        cand_bbox = [cand_bbox['x_min'], cand_bbox['y_min'], cand_bbox['x_max'], cand_bbox['y_max']]

                    if is_bbox_match(cand_bbox, current_bbox_norm):
                        if 'observer_expression' in cand and 'unique' in cand['observer_expression']:
                            observer_expression_unique = cand['observer_expression']['unique']
                        
                        if 'gazes' in cand and len(cand['gazes']) > 0:
                            gaze_info = cand['gazes'][0]
                            gaze_direction = gaze_info.get('gaze_direction', "")
                            gaze_point_expressions = gaze_info.get('gaze_point_expressions', [])
                            seg_mask_path = gaze_info.get('seg_mask_path', "")
                        break

            gazex_norm = [x for x in row['gaze_x']]
            gazey_norm = [y for y in row['gaze_y']]
            gazex = [x * float(width) for x in row['gaze_x']]
            gazey = [y * float(height) for y in row['gaze_y']]

            heads.append({
                'bbox': [xmin, ymin, xmax, ymax],
                'bbox_norm': [xmin / float(width), ymin / float(height), xmax / float(width), ymax / float(height)],
                'gazex': gazex,
                'gazey': gazey,
                'gazex_norm': gazex_norm,
                'gazey_norm': gazey_norm,
                'inout': 1, # all test frames are in frame
                'num_annot': len(gazex),
                'head_id': i,
                'observer_expression': observer_expression_unique,
                'gaze_direction': gaze_direction,
                'gaze_point_expressions': gaze_point_expressions,
                'seg_mask_path': seg_mask_path
            })
        
        TEST_FRAMES.append({
            'path': path,
            'heads': heads,
            'num_heads': num_people,
            'width': width,
            'height': height,
        })
        if num_people > 1:
            multiperson_ex += 1

    print("Test set: {} frames, {} multi-person".format(len(TEST_FRAMES), multiperson_ex))
    out_file = open(os.path.join(DATA_PATH, "test_preprocessed.json"), "w")
    json.dump(TEST_FRAMES, out_file)



if __name__ == "__main__":
    main(args.data_path)