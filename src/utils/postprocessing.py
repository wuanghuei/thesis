from collections import defaultdict
import torch
from src.utils.helpers import calculate_temporal_iou

def resolve_cross_class_overlaps(merged_detections):
    for video_id, detections in merged_detections.items():
        detections = sorted(detections, key=lambda x: x['confidence'], reverse=True)
        
        final_detections = []
        
        max_frame = max([det['end_frame'] for det in detections]) if detections else 0
        frames_occupied = [False] * (max_frame + 1)
        
        for det in detections:
            start = det['start_frame']
            end = det['end_frame']
            
            overlap = False
            for t in range(start, end):
                if t < len(frames_occupied) and frames_occupied[t]:
                    overlap = True
                    break
            
            if not overlap:
                for t in range(start, end):
                    if t < len(frames_occupied):
                        frames_occupied[t] = True
                final_detections.append(det)
        
        merged_detections[video_id] = final_detections
    
    return merged_detections

def merge_cross_window_detections(all_window_detections, all_window_metadata, iou_threshold=0.2, confidence_threshold=0.15):
    video_detections = defaultdict(lambda: defaultdict(list))
    
    for window_idx, (window_dets, meta) in enumerate(zip(all_window_detections, all_window_metadata)):
        video_id = meta['video_id']
        start_idx = meta['start_idx']
        
        for det in window_dets:
            action_id = det['action_id']
            global_start = start_idx + det['start_frame']
            global_end = start_idx + det['end_frame']
            confidence = det['confidence']
            
            video_detections[video_id][action_id].append({
                'start_frame': global_start,
                'end_frame': global_end,
                'confidence': confidence,
                'window_idx': window_idx
            })
    
    merged_results = {}
    for video_id, action_dets in video_detections.items():
        merged_results[video_id] = []
        
        for action_id, dets in action_dets.items():
            dets = sorted(dets, key=lambda x: x['start_frame'])
            
            i = 0
            while i < len(dets):
                current = dets[i]
                merged = dict(current)
                
                j = i + 1
                while j < len(dets):
                    next_det = dets[j]
                    
                    overlap = min(merged['end_frame'], next_det['end_frame']) - max(merged['start_frame'], next_det['start_frame'])
                    overlap_ratio = overlap / min(merged['end_frame'] - merged['start_frame'], next_det['end_frame'] - next_det['start_frame'])
                    
                    time_diff = abs(next_det['start_frame'] - merged['end_frame'])
                    
                    if (overlap_ratio >= iou_threshold or time_diff <= 5) and \
                       (merged['confidence'] + next_det['confidence']) / 2 >= confidence_threshold:
                        merged['start_frame'] = min(merged['start_frame'], next_det['start_frame'])
                        merged['end_frame'] = max(merged['end_frame'], next_det['end_frame'])
                        merged['confidence'] = (merged['confidence'] * (merged['end_frame'] - merged['start_frame']) + 
                                             next_det['confidence'] * (next_det['end_frame'] - next_det['start_frame'])) / \
                                             ((merged['end_frame'] - merged['start_frame']) + 
                                             (next_det['end_frame'] - next_det['start_frame']))
                        dets.pop(j)
                    else:
                        j += 1
                
                merged_results[video_id].append({
                    'action_id': action_id,
                    'start_frame': merged['start_frame'],
                    'end_frame': merged['end_frame'],
                    'confidence': merged['confidence']
                })
                
                i += 1
    
    return merged_results

def post_process(action_probs, start_probs, end_probs, class_thresholds, boundary_threshold, nms_threshold, min_segment_length):

    batch_size, seq_len, num_classes = action_probs.shape
    all_detections_batch = []
    
    for b in range(batch_size):
        detections_window = []
        
        for c in range(num_classes):
            action_score_c = action_probs[b, :, c]
            start_score_c = start_probs[b, :, c]
            end_score_c = end_probs[b, :, c]
            class_threshold_c = class_thresholds[c]

            start_indices = torch.where(start_score_c > boundary_threshold)[0]
            end_indices = torch.where(end_score_c > boundary_threshold)[0]

            if c == 2:
                print(f"Class 2 (window {b}): Max Start Score = {start_score_c.max().item():.4f}, Max End Score = {end_score_c.max().item():.4f}")
                print(f"Class 2 (window {b}): Num Start Indices (> {boundary_threshold}) = {len(start_indices)}, Num End Indices (> {boundary_threshold}) = {len(end_indices)}")

            if len(start_indices) == 0 or len(end_indices) == 0:
                continue
            
            proposals_class_c = []
            for start_idx_tensor in start_indices:
                start_idx = start_idx_tensor.item()
                valid_end_indices = end_indices[end_indices > start_idx]

                for end_idx_tensor in valid_end_indices:
                    end_idx = end_idx_tensor.item()

                    if (end_idx - start_idx) >= min_segment_length:
                        segment_action_score = action_score_c[start_idx:end_idx].mean().item()

                        if segment_action_score > class_threshold_c:
                            start_conf = start_score_c[start_idx].item()
                            effective_end_idx = max(start_idx, end_idx - 1)
                            end_conf = end_score_c[effective_end_idx].item()

                            confidence = (segment_action_score + start_conf + end_conf) / 3.0

                            proposals_class_c.append({
                                'action_id': c,
                                'start_frame': start_idx,
                                'end_frame': end_idx,
                                'confidence': confidence
                            })

            detections_window.extend(proposals_class_c)

        detections_window = sorted(detections_window, key=lambda x: x['confidence'], reverse=True)
        detections_window_nms = nms(detections_window, nms_threshold)


        all_detections_batch.append(detections_window_nms)

    return all_detections_batch

def nms(detections, threshold):
    if not detections:
        return []
    
    detections = sorted(detections, key=lambda x: x['confidence'], reverse=True)
    
    keep = []
    while detections:
        current = detections.pop(0)
        keep.append(current)
        
        detections = [
            d for d in detections if 
            calculate_temporal_iou(
                (current['start_frame'], current['end_frame']),
                (d['start_frame'], d['end_frame'])
            ) <= threshold
        ]
    
    return keep
