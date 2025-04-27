from src.utils.helpers import calculate_temporal_iou
from src.utils.metrics import calculate_mAP, calculate_class_mAP, calculate_map_mid, calculate_f1_at_iou
from src.utils.postprocessing import merge_cross_window_detections, resolve_cross_class_overlaps
from collections import defaultdict
import numpy as np
from sklearn.metrics import precision_recall_fscore_support


def compute_final_metrics(all_window_detections, all_window_metadata, all_frame_preds, all_frame_targets, num_classes):
    merged_video_detections = merge_cross_window_detections(
        all_window_detections, 
        all_window_metadata,
        iou_threshold=0.2,
        confidence_threshold=0.15
        )
    merged_video_detections = resolve_cross_class_overlaps(merged_video_detections)
    # Tính F1 dựa trên frame-level predictions
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_frame_targets, all_frame_preds, average='macro', zero_division=0
    )
    
    merged_all_action_preds = {c: [] for c in range(num_classes)}

    for video_dets in merged_video_detections.values():
        for det in video_dets:
            merged_all_action_preds[det['action_id']].append({
                'segment': (det['start_frame'], det['end_frame']),
                'score': det['confidence']
            })
    # Tạo dictionary để lưu ground truth toàn cục theo video_id và action_id
    global_action_gt = defaultdict(lambda: defaultdict(list))

    # Extract ground truth segments from annotations
    for i, meta in enumerate(all_window_metadata):
        video_id = meta['video_id']
        start_idx = meta['start_idx']  # Window's start index in global coordinates
        
        for anno in meta['annotations']:
            action_id = anno['action_id']
            # Chuyển ground truth sang tọa độ global
            global_gt_start = start_idx + anno['start_frame']
            global_gt_end = start_idx + anno['end_frame']
            
            # Thêm vào ground truth toàn cục
            global_action_gt[video_id][action_id].append((global_gt_start, global_gt_end))

    # Chuyển đổi thành format tương thích với hàm calculate_mAP
    all_action_gt_global = {c: [] for c in range(num_classes)}
    for video_id, actions in global_action_gt.items():
        for action_id, segments in actions.items():
            # Loại bỏ các segments trùng lặp
            unique_segments = list(set(segments))
            all_action_gt_global[action_id].extend(unique_segments)

    for c in range(num_classes):
        print(f"Class {c} - Global GT count: {len(all_action_gt_global[c])}")
        if all_action_gt_global[c]:
            print(f"  Sample GT: {all_action_gt_global[c][0]}")
        if merged_all_action_preds[c]:
            print(f"  Sample Pred: {merged_all_action_preds[c][0]}")

    # Trực quan hóa kết quả merger
    for video_id, detections in merged_video_detections.items():
        print(f"\nVideo {video_id}: {len(detections)} detections sau khi kết hợp")
        detections = sorted(detections, key=lambda x: (x['action_id'], x['start_frame']))
    
    # Sử dụng merged_all_action_preds thay vì all_action_preds
    mAP = calculate_mAP(all_action_gt_global, merged_all_action_preds, num_classes)

    merged_all_frame_preds = []
    merged_all_frame_targets = []

    # Thu thập tất cả frames từ các video
    for video_id, detections in merged_video_detections.items():
        # Tìm độ dài video (lấy frame cuối cùng từ detections hoặc ground truth)
        max_frame = 0
        for det in detections:
            max_frame = max(max_frame, det['end_frame'])
        
        for c in range(num_classes):
            if video_id in global_action_gt and c in global_action_gt[video_id]:
                for start, end in global_action_gt[video_id][c]:
                    max_frame = max(max_frame, end)
        
        # Tạo mảng frame targets và predictions cho video này
        video_length = max_frame + 1
        video_targets = np.zeros((video_length, num_classes), dtype=int)
        video_preds = np.zeros((video_length, num_classes), dtype=int)
        
        # Lấp đầy ground truth
        if video_id in global_action_gt:
            for c, segments in global_action_gt[video_id].items():
                for start, end in segments:
                    for t in range(start, end):
                        if t < video_length:
                            video_targets[t, c] = 1
        
        # Lấp đầy predictions
        for det in detections:
            c = det['action_id']
            start = det['start_frame']
            end = det['end_frame']
            for t in range(start, end):
                if t < video_length:
                    video_preds[t, c] = 1
        
        # Chuyển sang dạng flatten để tính F1
        for t in range(video_length):
            for c in range(num_classes):
                merged_all_frame_targets.append(video_targets[t, c])
                merged_all_frame_preds.append(video_preds[t, c])

    # Tính global F1
    merged_precision, merged_recall, merged_f1, _ = precision_recall_fscore_support(
        merged_all_frame_targets, merged_all_frame_preds, average='macro', zero_division=0
    )

    print(f"\n--- F1 Metrics ---")
    print(f"Window-level F1: {f1:.4f} (Precision: {precision:.4f}, Recall: {recall:.4f})")
    print(f"Global F1 after merge: {merged_f1:.4f} (Precision: {merged_precision:.4f}, Recall: {merged_recall:.4f})")

    # Tính F1 cho từng lớp riêng biệt
    merged_class_f1 = []
    for c in range(num_classes):
        class_targets = [merged_all_frame_targets[i] for i in range(len(merged_all_frame_targets)) 
                        if i % num_classes == c]
        class_preds = [merged_all_frame_preds[i] for i in range(len(merged_all_frame_preds))
                    if i % num_classes == c]
        
        if sum(class_targets) > 0:  # Chỉ tính F1 cho các lớp có ground truth
            _, _, class_f1, _ = precision_recall_fscore_support(
                class_targets, class_preds, average='binary', zero_division=0
            )
            merged_class_f1.append(class_f1)
            print(f"Class {c} F1: {class_f1:.4f}")



        # Tính segment accuracy (dùng IoU=0.5)
    total_correct = 0
    total_global_gt_segments = sum(len(all_action_gt_global[c]) for c in range(num_classes))
    total_pred = sum(len(merged_all_action_preds[c]) for c in range(num_classes))

    for c in range(num_classes):
        gt_matched = [False] * len(all_action_gt_global[c])
        for pred in merged_all_action_preds[c]:
            best_iou = 0
            best_idx = -1
            for i, gt in enumerate(all_action_gt_global[c]):
                if not gt_matched[i]:
                    iou = calculate_temporal_iou(pred['segment'], gt)
                    if iou > best_iou:
                        best_iou = iou
                        best_idx = i
            
            if best_iou >= 0.5 and best_idx >= 0:
                total_correct += 1
                gt_matched[best_idx] = True


    # Tính trung bình độ dài segment trước và sau khi kết hợp
    avg_length_before = np.mean([det['end_frame'] - det['start_frame'] for dets in all_window_detections for det in dets] if any(all_window_detections) else [0])
    avg_length_after = np.mean([det['end_frame'] - det['start_frame'] for dets in merged_video_detections.values() for det in dets] if any(merged_video_detections.values()) else [0])
    print(f"Avg length: {avg_length_before:.2f} → {avg_length_after:.2f} frames")
    
    # SỬA LẠI DÒNG PRINT THEO Ý NGHĨA MỚI
    total_merged_predictions = sum(len(dets) for dets in merged_video_detections.values())
    print(f"Total Segments: Detected={total_merged_predictions} / GroundTruth={total_global_gt_segments}")

    # (Có thể giữ lại dòng đếm dự đoán dài nếu muốn so sánh riêng)
    # long_actions_detected = sum(1 for det in merged_video_detections.values() for d in det if d['end_frame']-d['start_frame'] > 32)
    
    # Calculate AP cho từng lớp
    class_ap_dict = {}
    print("\n--- mAP by class ---")
    
    for c in range(num_classes):
        class_ap = calculate_class_mAP(all_action_gt_global[c], merged_all_action_preds[c])
        class_ap_dict[c] = class_ap
        num_gt = len(all_action_gt_global[c])
        num_pred = len(merged_all_action_preds[c])
        print(f"Class {c}: AP={class_ap:.4f} (GT={num_gt}, Pred={num_pred})")
    
    # Tính số lượng false positives và true positives
    for pred, target in zip(all_frame_preds, all_frame_targets):
        if pred == 1 and target == 1:
            true_positives += 1
        elif pred == 1 and target == 0:
            false_positives += 1
    
    print("\n--- Segment-level F1 at different IoU thresholds ---")
    iou_thresholds = [0.1, 0.25, 0.5]
    avg_f1_scores = {} # Store average F1 for returning
    
    for iou in iou_thresholds:
        all_class_f1 = []
        for c in range(num_classes):
            # Check if GT exists for this class to avoid division by zero or misleading F1=0
            if len(all_action_gt_global[c]) == 0: 
                # If no GT, F1 is undefined or arguably 0 if preds exist, 1 if no preds either. 
                # For averaging, skipping might be best unless defined otherwise.
                # Let's skip for averaging to avoid skewing.
                continue # Skip this class if no ground truth
            
            # Tính F1 ở ngưỡng IoU cụ thể
            _, _, class_f1 = calculate_f1_at_iou(all_action_gt_global[c], merged_all_action_preds[c], iou)
            all_class_f1.append(class_f1)
            print(f"Class {c} - F1@{iou:.2f}: {class_f1:.4f}")
        
        avg_f1 = np.mean(all_class_f1) if all_class_f1 else 0.0 # Handle case where no class had GT
        avg_f1_scores[f'avg_f1_iou_{iou:.2f}'.replace('.', '')] = avg_f1 # Store with a key like 'avg_f1_iou_01'
        print(f"Average F1@{iou:.2f}: {avg_f1:.4f}")

    map_mid = calculate_map_mid(all_action_gt_global, merged_all_action_preds, num_classes)
    print(f"mAP@mid: {map_mid:.4f}")

    accuracy = total_correct / max(1, total_global_gt_segments) # Tránh chia cho 0 nếu không có GT
    print(f"\nSegment Accuracy@0.5: {accuracy:.4f} (Correct={total_correct}, GT={total_global_gt_segments}, Pred={total_pred})")
    
    # In thêm về khả năng phát hiện
    total_frames = len(all_frame_preds)
    print(f"False positives: {false_positives}/{total_frames} ({false_positives/total_frames*100:.2f}%)")
    return {
        'mAP': mAP,
        'merged_f1': merged_f1,
        'class_aps': class_ap_dict,
        'map_mid': map_mid,
        'avg_f1_iou_010': avg_f1_scores.get('avg_f1_iou_010', 0.0),
        'avg_f1_iou_025': avg_f1_scores.get('avg_f1_iou_025', 0.0),
        'avg_f1_iou_050': avg_f1_scores.get('avg_f1_iou_050', 0.0)
    }
    
    