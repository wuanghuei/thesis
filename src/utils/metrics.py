from .helpers import calculate_temporal_iou
from sklearn.metrics import precision_recall_fscore_support, average_precision_score
import numpy as np


def calculate_f1_at_iou(gt_segments, pred_segments, iou_threshold):
    """Calculate F1 score at a specific IoU threshold"""
    if not pred_segments:
        return 0.0, 0.0, 0.0  # Precision, Recall, F1
    
    # Sắp xếp predictions theo confidence
    pred_segments = sorted(pred_segments, key=lambda x: x['score'], reverse=True)
    
    true_positives = 0
    gt_matched = [False] * len(gt_segments)
    
    for pred in pred_segments:
        pred_segment = pred['segment']
        best_iou = 0
        best_idx = -1
        
        # Tìm GT có IoU cao nhất với prediction này
        for i, gt_segment in enumerate(gt_segments):
            if not gt_matched[i]:  # Chỉ xét GT chưa được match
                iou = calculate_temporal_iou(pred_segment, gt_segment)
                if iou > best_iou:
                    best_iou = iou
                    best_idx = i
        
        # Nếu IoU >= threshold, đánh dấu là true positive
        if best_iou >= iou_threshold and best_idx >= 0:
            true_positives += 1
            gt_matched[best_idx] = True
    
    # Tính precision, recall, F1
    precision = true_positives / len(pred_segments) if pred_segments else 0
    recall = true_positives / len(gt_segments) if gt_segments else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    return precision, recall, f1

def calculate_class_mAP(gt_segments, pred_segments, iou_threshold=0.5):
    """Calculate AP for a single class"""
    # Handle edge cases
    if len(gt_segments) == 0:
        return 0.0  # No ground truth -> cannot compute AP
    
    if len(pred_segments) == 0:
        return 0.0  # No predictions -> precision is 0
    
    # Sort predictions by confidence
    pred_segments = sorted(pred_segments, key=lambda x: x['score'], reverse=True)
    
    # Mark GT segments as detected or not
    gt_detected = [False] * len(gt_segments)
    
    # Arrays for AP calculation
    y_true = []
    y_score = []
    
    # For each prediction
    for pred in pred_segments:
        pred_segment = pred['segment']
        score = pred['score']
        
        # Add score to y_score
        y_score.append(score)
        
        # Find best matching GT segment
        max_iou = 0
        max_idx = -1
        
        for i, gt_segment in enumerate(gt_segments):
            if gt_detected[i]:
                continue  # Skip already detected GT segments
            
            iou = calculate_temporal_iou(pred_segment, gt_segment)
            if iou > max_iou:
                max_iou = iou
                max_idx = i
        
        # Match if IoU >= threshold
        if max_iou >= iou_threshold and max_idx >= 0:
            gt_detected[max_idx] = True
            y_true.append(1)  # True positive
        else:
            y_true.append(0)  # False positive
    
    # Calculate AP
    if sum(y_true) > 0:
        ap = average_precision_score(y_true, y_score)
    else:
        ap = 0.0
    
    return ap

def calculate_map_mid(all_action_gt, all_action_preds, num_classes):
    """Calculate mAP with midpoint criterion instead of IoU"""
    aps = []
    
    for action_id in range(num_classes):
        gt_segments = all_action_gt[action_id]
        pred_segments = all_action_preds[action_id]
        
        # Skip if no ground truth
        if len(gt_segments) == 0:
            continue
        
        # Sort predictions by confidence
        pred_segments = sorted(pred_segments, key=lambda x: x['score'], reverse=True)
        
        # Arrays for AP calculation
        y_true = []
        y_score = []
        
        # Đánh dấu GT segments đã được phát hiện
        gt_detected = [False] * len(gt_segments)
        
        for pred in pred_segments:
            pred_segment = pred['segment']
            score = pred['score']
            
            # Tính midpoint
            pred_mid = (pred_segment[0] + pred_segment[1]) / 2
            
            # Thêm score
            y_score.append(score)
            
            # Kiểm tra xem midpoint có nằm trong bất kỳ GT nào
            is_correct = False
            for i, gt_segment in enumerate(gt_segments):
                if not gt_detected[i] and gt_segment[0] <= pred_mid <= gt_segment[1]:
                    gt_detected[i] = True
                    is_correct = True
                    break
            
            y_true.append(1 if is_correct else 0)
        
        # Tính AP
        if sum(y_true) > 0:  # Nếu có ít nhất 1 detection đúng
            ap = average_precision_score(y_true, y_score)
        else:
            ap = 0.0
        
        aps.append(ap)
    
    # Tính mAP
    return np.mean(aps) if aps else 0.0

def calculate_mAP(all_action_gt, all_action_preds, num_classes, iou_thresholds=[0.3, 0.5, 0.7]):
    """Calculate mean Average Precision across classes and IoU thresholds"""
    # Calculate AP for each class and IoU threshold
    aps = []
    
    for action_id in range(num_classes):
        gt_segments = all_action_gt[action_id]
        pred_segments = all_action_preds[action_id]
        
        # Skip if no ground truth
        if len(gt_segments) == 0:
            continue
        
        # Calculate AP for each IoU threshold
        class_aps = []
        for iou_threshold in iou_thresholds:
            ap = calculate_class_mAP(gt_segments, pred_segments, iou_threshold)
            class_aps.append(ap)
        
        # Average AP across IoU thresholds
        aps.append(np.mean(class_aps))
    
    # Calculate mAP
    return np.mean(aps) if aps else 0.0
