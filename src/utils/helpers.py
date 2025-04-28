import numpy as np
import torch
import random
import os
import json
import pickle
from tqdm import tqdm


def find_nearest_subsampled_idx(original_idx, frame_indices):
    differences = np.abs(np.array(frame_indices) - original_idx)
    nearest_idx = np.argmin(differences)
    return int(nearest_idx)  

def gaussian_kernel(center, window_size, sigma=1.0):
    x = torch.arange(window_size).float()
    return torch.exp(-((x - center) ** 2) / (2 * sigma**2))


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    print(f"Set seed={seed}")

def calculate_temporal_iou(pred_segment, gt_segment):
    pred_start, pred_end = pred_segment
    gt_start, gt_end = gt_segment
    
    pred_start, pred_end = min(pred_start, pred_end), max(pred_start, pred_end)
    gt_start, gt_end = min(gt_start, gt_end), max(gt_start, gt_end)
    
    if pred_start == pred_end or gt_start == gt_end:
        return 0.0
    
    intersection = max(0, min(pred_end, gt_end) - max(pred_start, gt_start))
    
    union = max(1, (pred_end - pred_start) + (gt_end - gt_start) - intersection)
    
    return intersection / union

def process_for_evaluation(detections, gt_annotations, window_size, num_classes):
    processed_preds = []
    processed_targets = []
    
    for t in range(window_size):
        for c in range(num_classes):
            is_gt_action = False
            for anno in gt_annotations:
                if anno['action_id'] == c and anno['start_frame'] <= t < anno['end_frame']:
                    is_gt_action = True
                    break
            
            is_pred_action = False
            for det in detections:
                if det['action_id'] == c and det['start_frame'] <= t < det['end_frame']:
                    is_pred_action = True
                    break
            
            processed_preds.append(1 if is_pred_action else 0)
            processed_targets.append(1 if is_gt_action else 0)
    
    return processed_preds, processed_targets

def reconstruct_full_video_probs(video_id, all_raw_preds_loaded, all_batch_meta_loaded, num_classes, window_size):
    video_windows_indices = []
    max_end_frame = 0

    for batch_idx, batch_meta in enumerate(all_batch_meta_loaded):
        for window_idx_in_batch, meta in enumerate(batch_meta):
            if meta['video_id'] == video_id:
                try:
                    current_window_size = all_raw_preds_loaded[batch_idx][0][window_idx_in_batch].shape[0]
                except (IndexError, AttributeError):
                    current_window_size = window_size 
                
                video_windows_indices.append({
                    'batch_idx': batch_idx,
                    'window_idx_in_batch': window_idx_in_batch,
                    'start_frame': meta['start_idx'],
                    'window_len': current_window_size
                })
                max_end_frame = max(max_end_frame, meta['end_idx'])

    if not video_windows_indices:
        return None, None, None, 0

    num_frames = max_end_frame
    if num_frames <= 0:
        return None, None, None, 0

    sum_action_probs = np.zeros((num_frames, num_classes), dtype=np.float64)
    sum_start_probs = np.zeros((num_frames, num_classes), dtype=np.float64)
    sum_end_probs = np.zeros((num_frames, num_classes), dtype=np.float64)
    counts = np.zeros((num_frames, num_classes), dtype=np.int16)

    for window_info in video_windows_indices:
        batch_idx = window_info['batch_idx']
        window_idx = window_info['window_idx_in_batch']
        start_f = window_info['start_frame']
        window_len = window_info['window_len']

        try:
            action_probs_win = all_raw_preds_loaded[batch_idx][0][window_idx].numpy().astype(np.float64)
            start_probs_win = all_raw_preds_loaded[batch_idx][1][window_idx].numpy().astype(np.float64)
            end_probs_win = all_raw_preds_loaded[batch_idx][2][window_idx].numpy().astype(np.float64)
            
            if action_probs_win.shape[0] != window_len or \
               start_probs_win.shape[0] != window_len or \
               end_probs_win.shape[0] != window_len or \
               action_probs_win.shape[1] != num_classes or \
               start_probs_win.shape[1] != num_classes or \
               end_probs_win.shape[1] != num_classes:
                 print(f"Shape mismatch for window {video_id} batch {batch_idx} window {window_idx} Expected T={window_len}, C={num_classes} Skipping window")
                 continue
                 
        except (IndexError, AttributeError) as e:
            print(f"Error accessing raw preds for window {video_id} batch {batch_idx} window {window_idx}: {e} Skipping window")
            continue

        global_start = start_f
        global_end = min(start_f + window_len, num_frames)
        local_end = global_end - global_start

        if local_end > 0:
            sum_action_probs[global_start:global_end, :] += action_probs_win[:local_end, :]
            sum_start_probs[global_start:global_end, :] += start_probs_win[:local_end, :]
            sum_end_probs[global_start:global_end, :] += end_probs_win[:local_end, :]
            counts[global_start:global_end, :] += 1

    counts[counts == 0] = 1 
    avg_action_probs = sum_action_probs / counts
    avg_start_probs = sum_start_probs / counts
    avg_end_probs = sum_end_probs / counts
    
    avg_action_probs = np.clip(avg_action_probs, 0.0, 1.0)
    avg_start_probs = np.clip(avg_start_probs, 0.0, 1.0)
    avg_end_probs = np.clip(avg_end_probs, 0.0, 1.0)

    return avg_action_probs, avg_start_probs, avg_end_probs, num_frames

def generate_target_labels(video_id, anno_dir, num_frames, num_classes):
    if num_frames <= 0:
        return None
        
    anno_filename = f"{video_id}_annotations.json"
    anno_filepath = os.path.join(anno_dir, anno_filename)

    if not os.path.exists(anno_filepath):
        print(f"Annotation file not found for {video_id} at {anno_filepath}")
        return None

    try:
        with open(anno_filepath, 'r') as f:
            anno_data = json.load(f)
        
        annotations = anno_data['annotations']
        
        frame_labels = np.full(num_frames, fill_value=num_classes, dtype=int)
        
        annotations.sort(key=lambda x: x['start_frame'])
        
        for anno in annotations:
            action_id = anno['action_id']
            start = anno['start_frame']
            end = anno['end_frame']
            
            start = max(0, start)
            end = min(num_frames, end)
            
            if start < end and 0 <= action_id < num_classes:
                frame_labels[start:end] = action_id
            elif action_id >= num_classes:
                pass
                 
        return frame_labels
        
    except Exception as e:
        print(f"processing annotation file {anno_filepath}: {e}")
        return None

def process_predictions_for_rnn(all_raw_preds, all_batch_meta, num_classes, window_size, anno_dir, output_dir):
    print(f"\nProcessing")
    
    video_ids = sorted(list(set(meta['video_id'] for batch_meta in all_batch_meta for meta in batch_meta)))
    print(f"Found {len(video_ids)} unique video IDs")

    processed_count = 0
    skipped_count = 0
    for video_id in tqdm(video_ids, desc=f"Processing videos"):
        avg_action_probs, avg_start_probs, avg_end_probs, num_frames = reconstruct_full_video_probs(
            video_id, all_raw_preds, all_batch_meta, num_classes, window_size
        )
        
        if num_frames is None or num_frames <= 0:
            skipped_count += 1
            continue
            
        target_labels = generate_target_labels(video_id, anno_dir, num_frames, num_classes)
        
        if target_labels is None:
            skipped_count += 1
            continue
            
        input_features = np.concatenate([avg_action_probs, avg_start_probs, avg_end_probs], axis=1)
        
        output_path = os.path.join(output_dir, f"{video_id}.npz")
        try:
            np.savez_compressed(output_path, features=input_features, labels=target_labels)
            processed_count += 1
        except Exception as e:
            print(f"saving {output_path}: {e}")
            skipped_count += 1
            
    print(f"Processed: {processed_count}, Skipped: {skipped_count}")


