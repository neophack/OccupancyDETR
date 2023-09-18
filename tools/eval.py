import os

import numpy as np
from tqdm import tqdm

from tools.common import read_json
from tools.data.semantic_kitti import image_height, image_width


def eval_voxel(output_path, restore_path):
    pred = np.load(output_path)
    gt = np.load(restore_path)
    Is = []
    Us = []
    for i in range(20):
        cur_pred = pred == i
        cur_gt = gt == i
        Is.append(np.sum(cur_pred & cur_gt))
        Us.append(np.sum(cur_pred | cur_gt))
    return Is, Us


def eval_voxels(output_dir, restore_dir):
    Iss, Uss = [], []
    for file in tqdm(os.listdir(restore_dir)):
        output_path = os.path.join(output_dir, file)
        restore_path = os.path.join(restore_dir, file)
        Is, Us = eval_voxel(output_path, restore_path)
        Iss.append(Is)
        Uss.append(Us)
    Iss = np.array(Iss).sum(axis=0)
    Uss = np.array(Uss).sum(axis=0) + 1e-6
    ious = Iss / Uss
    return ious


def calculate_iou(box1, box2):
    min_x1, min_y1, max_x1, max_y1 = box1
    min_x2, min_y2, max_x2, max_y2 = box2
    w1, h1 = max_x1 - min_x1, max_y1 - min_y1
    w2, h2 = max_x2 - min_x2, max_y2 - min_y2

    x_inter_left = max(min_x1, min_x2)
    y_inter_top = max(min_y1, min_y2)
    x_inter_right = min(max_x1, max_x2)
    y_inter_bottom = min(max_y1, max_y2)

    I = max(0, x_inter_right - x_inter_left) * max(0, y_inter_bottom - y_inter_top)
    U = w1 * h1 + w2 * h2 - I
    iou = I / U
    return iou


def calculate_ap(gt_boxes, pred_boxes, scroes, iou_threshold=0.5):
    sorted_indices = np.argsort(scroes)[::-1]
    pred_boxes = pred_boxes[sorted_indices]
    matches = np.zeros(len(pred_boxes), dtype=bool)
    gt_matches = np.zeros(len(gt_boxes), dtype=bool)
    for i, predicted_box in enumerate(pred_boxes):
        best_iou = 0.0
        best_match_index = -1
        for j, gt_box in enumerate(gt_boxes[~gt_matches]):
            iou = calculate_iou(predicted_box, gt_box)
            if iou > best_iou:
                best_iou = iou
                best_match_index = j

        if best_iou >= iou_threshold:
            matches[i] = True
            gt_matches[best_match_index] = True

    tp = np.sum(matches)
    fp = len(matches) - tp
    fn = len(gt_boxes) - tp
    return tp, fp, fn


def eval_2d_obj(output_path, gt_path, iou_threshold=0.5):
    result = read_json(output_path)
    classes = np.array(result["classes"])
    boxes = np.array(result["boxes"])
    scores = np.array(result["scores"])
    boxes[:, [0, 2]] *= image_width
    boxes[:, [1, 3]] *= image_height
    gt = read_json(gt_path)
    classes_gt = []
    boxes_gt = []
    for l in gt["labels"]:
        classes_gt.append(l["label"])
        boxes_gt.append(l["box2d"])
    classes_gt = np.array(classes_gt)
    boxes_gt = np.array(boxes_gt)
    sum_tp, sum_fp, sum_fn = 0, 0, 0
    for i in range(1, 20):
        cls_mask = classes == i
        cls_mask_gt = classes_gt == i
        tp, fp, fn = calculate_ap(boxes_gt[cls_mask_gt], boxes[cls_mask], scores[cls_mask], iou_threshold)
        sum_tp += tp
        sum_fp += fp
        sum_fn += fn
    return sum_tp, sum_fp, sum_fn


def eval_2d_objs(output_dir, preprocessed_dir, iou_threshold=0.5):
    sum_tp, sum_fp, sum_fn = 0, 0, 0
    for file in tqdm(os.listdir(preprocessed_dir)):
        output_path = os.path.join(output_dir, file)
        gt_path = os.path.join(preprocessed_dir, file)
        tp, fp, fn = eval_2d_obj(output_path, gt_path, iou_threshold)
        sum_tp += tp
        sum_fp += fp
        sum_fn += fn
    precision = sum_tp / (sum_tp + sum_fp)
    recall = sum_tp / (sum_tp + sum_fn)
    return precision, recall


def eval_2d_objs_during_training(preds, labels, iou_threshold=0.5):
    sum_tp, sum_fp, sum_fn = 0, 0, 0
    pred_logits = preds[2]
    classes, scores = pred_logits.argmax(-1), pred_logits.max(-1)
    boxes = np.zeros_like(preds[3])
    boxes[:, :2] = preds[3][:, :2] - preds[3][:, 2:] / 2
    boxes[:, 2:] = preds[3][:, :2] + preds[3][:, 2:] / 2
    gt_classes = labels[0]
    gt_boxes = np.zeros_like(labels[1])
    gt_boxes[:, :, :2] = labels[1][:, :, :2] - labels[1][:, :, 2:] / 2
    gt_boxes[:, :, 2:] = labels[1][:, :, :2] + labels[1][:, :, 2:] / 2
    bs = labels[0].shape[0]
    num_preds = preds[5]
    cur_i = 0
    for b in range(bs):
        b_classes = classes[cur_i : cur_i + num_preds[b]]
        b_scores = scores[cur_i : cur_i + num_preds[b]]
        b_boxes = boxes[cur_i : cur_i + num_preds[b]]
        cur_i += num_preds[b]
        b_gt_classes = gt_classes[b]
        b_gt_boxes = gt_boxes[b]
        for i in range(1, 20):
            cls_mask = b_classes == i
            cls_mask_gt = b_gt_classes == i
            tp, fp, fn = calculate_ap(b_gt_boxes[cls_mask_gt], b_boxes[cls_mask], b_scores[cls_mask], iou_threshold)
            sum_tp += tp
            sum_fp += fp
            sum_fn += fn
    precision = sum_tp / (sum_tp + sum_fp + 1e-6)
    recall = sum_tp / (sum_tp + sum_fn)
    return {"accuracy": precision, "recall": recall}
