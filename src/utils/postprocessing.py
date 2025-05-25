from collections import defaultdict
import torch
import src.utils.helpers as helpers

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
    
    global_window_idx = 0
    for batch_idx, (batch_detections, batch_meta_tuple) in enumerate(zip(all_window_detections, all_window_metadata)):
        if len(batch_detections) != len(batch_meta_tuple):
            print(f"Warning: Mismatch between detections ({len(batch_detections)}) and metadata ({len(batch_meta_tuple)}) in batch {batch_idx}. Skipping batch.")
            continue
            
        for window_dets_in_batch, meta in zip(batch_detections, batch_meta_tuple):
            try:
                video_id = meta['video_id']
                start_idx = meta['start_idx']
            except (TypeError, KeyError) as e:
                 print(f"Warning: Invalid metadata format or missing key in batch {batch_idx}: {meta}. Error: {e}. Skipping window.")
                 global_window_idx += 1
                 continue

            for det in window_dets_in_batch: 
                try:
                    action_id = det['action_id']
                    global_start = start_idx + det['start_frame']
                    global_end = start_idx + det['end_frame']
                    confidence = det['confidence']
                    
                    video_detections[video_id][action_id].append({
                        'start_frame': global_start,
                        'end_frame': global_end,
                        'confidence': confidence,
                        'window_idx': global_window_idx
                    })
                except (TypeError, KeyError) as e:
                    print(f"Warning: Invalid detection format or missing key in batch {batch_idx}, window {global_window_idx}: {det}. Error: {e}. Skipping detection.")
            
            global_window_idx += 1
    
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
                    
                    current_len = max(1, merged['end_frame'] - merged['start_frame'])
                    next_len = max(1, next_det['end_frame'] - next_det['start_frame'])
                    
                    overlap = max(0, min(merged['end_frame'], next_det['end_frame']) - max(merged['start_frame'], next_det['start_frame']))
                    overlap_ratio = overlap / min(current_len, next_len) 
                    
                    time_diff = abs(next_det['start_frame'] - merged['end_frame'])
                    
                    if (overlap_ratio >= iou_threshold or time_diff <= 5) and \
                       ((merged['confidence'] + next_det['confidence']) / 2 >= confidence_threshold):
                        
                        new_start = min(merged['start_frame'], next_det['start_frame'])
                        new_end = max(merged['end_frame'], next_det['end_frame'])
                        
                        total_len = current_len + next_len
                        if total_len > 0:
                           merged_confidence = (merged['confidence'] * current_len + next_det['confidence'] * next_len) / total_len
                        else: 
                           merged_confidence = (merged['confidence'] + next_det['confidence']) / 2
                           
                        merged['start_frame'] = new_start
                        merged['end_frame'] = new_end
                        merged['confidence'] = merged_confidence
                        
                        current_len = max(1, new_end - new_start)
                        
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
            helpers.calculate_temporal_iou(
                (current['start_frame'], current['end_frame']),
                (d['start_frame'], d['end_frame'])
            ) <= threshold
        ]
    
    return keep

def labels_to_segments(labels, ignore_label):

    segments = defaultdict(list)
    current_action = -1
    start_frame = -1
    T = len(labels)
    for t in range(T):
        label = labels[t]
        if label != ignore_label and label != current_action:
            if current_action != -1 and start_frame != -1:
                segments[current_action].append({'start_frame': start_frame, 'end_frame': t})
            current_action = label
            start_frame = t
        elif (label == ignore_label or label != current_action) and current_action != -1:
            segments[current_action].append({'start_frame': start_frame, 'end_frame': t})
            current_action = -1
            start_frame = -1
            if label != ignore_label:
                 current_action = label
                 start_frame = t
    if current_action != -1 and start_frame != -1:
        segments[current_action].append({'start_frame': start_frame, 'end_frame': T})
    return segments