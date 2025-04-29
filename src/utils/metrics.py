import src.utils.helpers as helpers
from sklearn.metrics import precision_recall_fscore_support, average_precision_score
import numpy as np


def calculate_f1_at_iou(gt_segments, pred_segments, iou_threshold):
    if not pred_segments:
        return 0.0, 0.0, 0.0
    
    pred_segments = sorted(pred_segments, key=lambda x: x['score'], reverse=True)
    
    true_positives = 0
    gt_matched = [False] * len(gt_segments)
    
    for pred in pred_segments:
        pred_segment = pred['segment']
        best_iou = 0
        best_idx = -1
        
        for i, gt_segment in enumerate(gt_segments):
            if not gt_matched[i]:
                iou = helpers.calculate_temporal_iou(pred_segment, gt_segment)
                if iou > best_iou:
                    best_iou = iou
                    best_idx = i
        
        if best_iou >= iou_threshold and best_idx >= 0:
            true_positives += 1
            gt_matched[best_idx] = True
    
    precision = true_positives / len(pred_segments) if pred_segments else 0
    recall = true_positives / len(gt_segments) if gt_segments else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    return precision, recall, f1

def calculate_class_mAP(gt_segments, pred_segments, iou_threshold=0.5):
    if len(gt_segments) == 0:
        return 0.0
    
    if len(pred_segments) == 0:
        return 0.0
    
    pred_segments = sorted(pred_segments, key=lambda x: x['score'], reverse=True)
    
    gt_detected = [False] * len(gt_segments)
    
    y_true = []
    y_score = []
    
    for pred in pred_segments:
        pred_segment = pred['segment']
        score = pred['score']
        
        y_score.append(score)
        
        max_iou = 0
        max_idx = -1
        
        for i, gt_segment in enumerate(gt_segments):
            if gt_detected[i]:
                continue
            
            iou = helpers.calculate_temporal_iou(pred_segment, gt_segment)
            if iou > max_iou:
                max_iou = iou
                max_idx = i
        
        if max_iou >= iou_threshold and max_idx >= 0:
            gt_detected[max_idx] = True
            y_true.append(1)
        else:
            y_true.append(0)
    
    if sum(y_true) > 0:
        ap = average_precision_score(y_true, y_score)
    else:
        ap = 0.0
    
    return ap

def calculate_map_mid(all_action_gt, all_action_preds, num_classes):
    aps = []
    
    for action_id in range(num_classes):
        gt_segments = all_action_gt[action_id]
        pred_segments = all_action_preds[action_id]
        
        if len(gt_segments) == 0:
            continue
        
        pred_segments = sorted(pred_segments, key=lambda x: x['score'], reverse=True)
        
        y_true = []
        y_score = []
        
        gt_detected = [False] * len(gt_segments)
        
        for pred in pred_segments:
            pred_segment = pred['segment']
            score = pred['score']
            
            pred_mid = (pred_segment[0] + pred_segment[1]) / 2
            
            y_score.append(score)
            
            is_correct = False
            for i, gt_segment in enumerate(gt_segments):
                if not gt_detected[i] and gt_segment[0] <= pred_mid <= gt_segment[1]:
                    gt_detected[i] = True
                    is_correct = True
                    break
            
            y_true.append(1 if is_correct else 0)
        
        if sum(y_true) > 0:
            ap = average_precision_score(y_true, y_score)
        else:
            ap = 0.0
        
        aps.append(ap)
    
    return np.mean(aps) if aps else 0.0

def calculate_mAP(all_action_gt, all_action_preds, num_classes, iou_thresholds=[0.3, 0.5, 0.7]):
    aps = []
    
    for action_id in range(num_classes):
        gt_segments = all_action_gt[action_id]
        pred_segments = all_action_preds[action_id]
        
        if len(gt_segments) == 0:
            continue
        
        class_aps = []
        for iou_threshold in iou_thresholds:
            ap = calculate_class_mAP(gt_segments, pred_segments, iou_threshold)
            class_aps.append(ap)
        
        aps.append(np.mean(class_aps))
    
    return np.mean(aps) if aps else 0.0
