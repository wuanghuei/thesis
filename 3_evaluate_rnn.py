import torch
import torch.nn as nn
import torch.optim as optim # Needed if optimizer state is loaded
from torch.utils.data import DataLoader
import os
import json
import argparse
import pickle
import copy
import numpy as np
from tqdm import tqdm
from collections import defaultdict
from sklearn.metrics import precision_recall_fscore_support, average_precision_score
import cv2 # For visualization

# Import base model and potentially dataloaders if running inference
try:
    from model_fixed import TemporalActionDetector
except ImportError:
    print("Error: Could not import TemporalActionDetector from model_fixed.py.")
    exit()
try:
    from dataloader import get_val_loader, get_test_loader # Only need val loader for evaluation
except ImportError:
    print("Error: Could not import get_val_loader from dataloader.py.")
    exit()
# Import the RNN model
try:
    from models.rnn_postprocessor import RNNPostProcessor
except ImportError:
    print("Error: Could not import RNNPostProcessor from models/rnn_postprocessor.py")
    exit()

# ====== Configuration ======
NUM_CLASSES = 5
WINDOW_SIZE = 32 # Should match base model training
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_MIXED_PRECISION = True # For base model inference
BACKGROUND_LABEL = NUM_CLASSES # Consistent label for background state
DEFAULT_BASE_CHECKPOINT = "checkpoints/interim_model_epoch18.pth"
DEFAULT_RNN_CHECKPOINT = "best/rnn_checkpoints/best_rnn_model.pth"
DEFAULT_INFERENCE_PKL = "best/val_inference_raw.pkl" # Default pkl file for validation data
DEFAULT_TEST_PKL = "best/test_inference_raw.pkl" # Default pkl file for test data

# ====== Helper Functions (Copied/Adapted from eval_hmm.py) ======

# --- Metric Calculation Functions ---
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

def calculate_f1_at_iou(gt_segments, pred_segments, iou_threshold):
    if not pred_segments:
        return 0.0, 0.0, 0.0
    pred_segments = sorted(pred_segments, key=lambda x: x.get('score', 0.0), reverse=True) # Use get for safety
    true_positives = 0
    gt_matched = [False] * len(gt_segments)
    for pred in pred_segments:
        pred_segment = pred['segment']
        best_iou = 0
        best_idx = -1
        for i, gt_segment in enumerate(gt_segments):
            if not gt_matched[i]:
                iou = calculate_temporal_iou(pred_segment, gt_segment)
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
    if len(gt_segments) == 0: return 0.0
    if len(pred_segments) == 0: return 0.0
    pred_segments = sorted(pred_segments, key=lambda x: x.get('score', 0.0), reverse=True)
    gt_detected = [False] * len(gt_segments)
    y_true = []
    y_score = []
    for pred in pred_segments:
        pred_segment = pred['segment']
        score = pred.get('score', 0.0)
        y_score.append(score)
        max_iou = 0
        max_idx = -1
        for i, gt_segment in enumerate(gt_segments):
            if gt_detected[i]: continue
            iou = calculate_temporal_iou(pred_segment, gt_segment)
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

def calculate_map_mid(all_action_gt, all_action_preds):
    aps = []
    for action_id in range(NUM_CLASSES):
        gt_segments = all_action_gt.get(action_id, []) 
        pred_segments = all_action_preds.get(action_id, [])
        if len(gt_segments) == 0: continue
        try:
            pred_segments = sorted(pred_segments, key=lambda x: x.get('score', 0.0), reverse=True)
        except AttributeError:
             continue 
        y_true = []
        y_score = []
        gt_detected = [False] * len(gt_segments)
        for pred in pred_segments:
            if not isinstance(pred, dict) or 'segment' not in pred or 'score' not in pred: continue
            pred_segment = pred['segment']
            score = pred['score']
            if not isinstance(pred_segment, (tuple, list)) or len(pred_segment) != 2: continue
            try: pred_mid = (pred_segment[0] + pred_segment[1]) / 2
            except TypeError: continue
            y_score.append(score)
            is_correct = False
            for i, gt_segment in enumerate(gt_segments):
                 if not isinstance(gt_segment, (tuple, list)) or len(gt_segment) != 2: continue
                 try:
                      if not gt_detected[i] and gt_segment[0] <= pred_mid <= gt_segment[1]:
                           gt_detected[i] = True
                           is_correct = True
                           break
                 except TypeError: continue
            y_true.append(1 if is_correct else 0)
        if sum(y_true) > 0:
            try: ap = average_precision_score(y_true, y_score)
            except ValueError: ap = 0.0
        else: ap = 0.0
        aps.append(ap)
    return np.mean(aps) if aps else 0.0

def calculate_mAP(all_action_gt, all_action_preds, iou_thresholds=[0.3, 0.5, 0.7]):
    aps = []
    for action_id in range(NUM_CLASSES):
        gt_segments = all_action_gt.get(action_id, [])
        pred_segments = all_action_preds.get(action_id, [])
        if len(gt_segments) == 0: continue
        class_aps_across_iou = []
        for iou_threshold in iou_thresholds:
            ap = calculate_class_mAP(gt_segments, pred_segments, iou_threshold)
            class_aps_across_iou.append(ap)
        avg_class_ap = np.mean(class_aps_across_iou) if class_aps_across_iou else 0.0
        aps.append(avg_class_ap)
    return np.mean(aps) if aps else 0.0

# --- Probability Reconstruction Function ---
def reconstruct_full_video_probs(video_id, all_raw_preds_loaded, all_batch_meta_loaded):
    # This function is identical to the one in 1_generate_rnn_data.py
    # It reconstructs action, start, end probs for a full video
    video_windows_indices = []
    max_end_frame = 0
    global WINDOW_SIZE # Access global WINDOW_SIZE as fallback

    for batch_idx, batch_meta in enumerate(all_batch_meta_loaded):
        for window_idx_in_batch, meta in enumerate(batch_meta):
            if meta['video_id'] == video_id:
                try:
                    current_window_size = all_raw_preds_loaded[batch_idx][0][window_idx_in_batch].shape[0]
                except (IndexError, AttributeError):
                    current_window_size = WINDOW_SIZE 
                video_windows_indices.append({
                    'batch_idx': batch_idx,
                    'window_idx_in_batch': window_idx_in_batch,
                    'start_frame': meta['start_idx'],
                    'window_len': current_window_size
                })
                max_end_frame = max(max_end_frame, meta['end_idx'])
    if not video_windows_indices: return None, None, None, 0
    num_frames = max_end_frame
    if num_frames <= 0: return None, None, None, 0
    sum_action_probs = np.zeros((num_frames, NUM_CLASSES), dtype=np.float64)
    sum_start_probs = np.zeros((num_frames, NUM_CLASSES), dtype=np.float64)
    sum_end_probs = np.zeros((num_frames, NUM_CLASSES), dtype=np.float64)
    counts = np.zeros((num_frames, NUM_CLASSES), dtype=np.int16)
    for window_info in video_windows_indices:
        batch_idx, window_idx, start_f, window_len = window_info['batch_idx'], window_info['window_idx_in_batch'], window_info['start_frame'], window_info['window_len']
        try:
            action_probs_win = all_raw_preds_loaded[batch_idx][0][window_idx].numpy().astype(np.float64)
            start_probs_win = all_raw_preds_loaded[batch_idx][1][window_idx].numpy().astype(np.float64)
            end_probs_win = all_raw_preds_loaded[batch_idx][2][window_idx].numpy().astype(np.float64)
            if action_probs_win.shape[0] != window_len or start_probs_win.shape[0] != window_len or end_probs_win.shape[0] != window_len or \
               action_probs_win.shape[1] != NUM_CLASSES or start_probs_win.shape[1] != NUM_CLASSES or end_probs_win.shape[1] != NUM_CLASSES:
                 continue
        except (IndexError, AttributeError): continue
        global_start, global_end = start_f, min(start_f + window_len, num_frames)
        local_end = global_end - global_start
        if local_end > 0:
            sum_action_probs[global_start:global_end, :] += action_probs_win[:local_end, :]
            sum_start_probs[global_start:global_end, :] += start_probs_win[:local_end, :]
            sum_end_probs[global_start:global_end, :] += end_probs_win[:local_end, :]
            counts[global_start:global_end, :] += 1
    counts[counts == 0] = 1 
    avg_action_probs = np.clip(sum_action_probs / counts, 0.0, 1.0)
    avg_start_probs = np.clip(sum_start_probs / counts, 0.0, 1.0)
    avg_end_probs = np.clip(sum_end_probs / counts, 0.0, 1.0)
    return avg_action_probs, avg_start_probs, avg_end_probs, num_frames

# --- Label to Segment Function (Can reuse hmm_labels_to_segments logic) ---
def labels_to_segments(labels, ignore_label=BACKGROUND_LABEL):
    """
    Converts frame-level label sequence into a list of action segments.
    Args:
        labels (np.array): Frame-level labels (T,).
        ignore_label (int): Label to ignore (background).
    Returns:
        dict: {action_id: [{'start_frame': s, 'end_frame': e}, ...]}
    """
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
            if label != ignore_label: # Start new segment immediately if current label is an action
                 current_action = label
                 start_frame = t
    if current_action != -1 and start_frame != -1:
        segments[current_action].append({'start_frame': start_frame, 'end_frame': T})
    return segments

# --- Visualization Function (Can adapt later if needed) ---
def visualize_rnn_predictions(video_id, frames_npz_path, output_video_path, fps, 
                               global_gt_data_by_video, rnn_preds_by_video, 
                               num_classes):
    print(f"\n--- Starting Visualization for Video: {video_id} ---")
    print(f"  Loading frames from: {frames_npz_path}")
    print(f"  Saving output to: {output_video_path}")

    if not os.path.exists(frames_npz_path):
        print(f"Error: Frames file not found at {frames_npz_path}")
        return

    try:
        frames_data = np.load(frames_npz_path)
        # Common key is 'frames', but check for others just in case
        if 'frames' in frames_data:
            frames = frames_data['frames']
        elif len(frames_data.files) == 1:
            frames = frames_data[frames_data.files[0]]
        else:
            print(f"Error: Could not determine the correct key for frames in {frames_npz_path}")
            return
    except Exception as e:
        print(f"Error loading frames from {frames_npz_path}: {e}")
        return

    if frames.ndim != 4 or frames.shape[3] != 3:
         print(f"Error: Unexpected frame shape {frames.shape}. Expected (T, H, W, 3).")
         return

    num_frames, height, width, _ = frames.shape
    print(f"  Loaded {num_frames} frames ({height}x{width})")

    # Ensure output directory exists
    output_dir = os.path.dirname(output_video_path)
    if output_dir and not os.path.exists(output_dir):
        try:
            os.makedirs(output_dir)
            print(f"  Created output directory: {output_dir}")
        except OSError as e:
            print(f"Error creating output directory {output_dir}: {e}")
            return

    # Initialize Video Writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
    if not out.isOpened():
        print(f"Error: Could not open video writer for {output_video_path}")
        return

    # Define Colors (Example: Use a colormap or define manually)
    # Using a simple approach for 5 classes + GT + Pred
    colors = [
        (255, 0, 0),    # Class 0 - Red
        (0, 255, 0),    # Class 1 - Lime
        (0, 0, 255),    # Class 2 - Blue
        (255, 255, 0),  # Class 3 - Yellow
        (0, 255, 255),  # Class 4 - Cyan
        (255, 0, 255),  # Background/Other - Magenta (Not used for drawing segments)
    ]
    gt_color = (200, 200, 200) # White/Gray for GT
    pred_color_base = (180, 105, 255) # Pink/Purple for Pred base

    bar_height = 25 # Height of the GT/Pred bars
    gt_y_pos = 10
    pred_y_pos = gt_y_pos + bar_height + 5
    text_y_offset = 18 # Offset for text inside the bar
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    font_thickness = 1

    # Pre-process predictions for easier lookup per frame
    rnn_preds_this_video = rnn_preds_by_video.get(video_id, defaultdict(list))

    for frame_idx in tqdm(range(num_frames), desc="Processing Frames"): 
        frame = frames[frame_idx].copy() # Work on a copy

        # Draw GT segments
        gt_labels_on_frame = []
        for action_id, gt_segments in global_gt_data_by_video.get(video_id, {}).items():
            for start, end in gt_segments:
                if start <= frame_idx < end:
                    label_text = f"GT Cls {action_id}"
                    gt_labels_on_frame.append((label_text, colors[action_id % len(colors)]))
                    # Draw GT bar segment
                    cv2.rectangle(frame, (0, gt_y_pos), (width-1, gt_y_pos + bar_height), gt_color, -1) # Background bar
                    cv2.rectangle(frame, (0, gt_y_pos), (width-1, gt_y_pos + bar_height), colors[action_id % len(colors)], 2) # Border
                    cv2.putText(frame, label_text, (5, gt_y_pos + text_y_offset), font, font_scale, (0,0,0), font_thickness, cv2.LINE_AA)
                    break # Only draw one GT label per class on the bar
            if gt_labels_on_frame: # Optimization: if any GT is found, stop checking other classes for this frame
                 pass # Allow multiple GTs potentially 

        # Draw RNN Predicted segments
        pred_labels_on_frame = []
        for action_id, pred_segments in rnn_preds_this_video.items():
            for segment_info in pred_segments:
                start, end = segment_info['segment']
                score = segment_info.get('score', 0.0)
                if start <= frame_idx < end:
                    label_text = f"Pred Cls {action_id} ({score:.2f})"
                    pred_labels_on_frame.append((label_text, colors[action_id % len(colors)]))
                    # Draw Pred bar segment
                    cv2.rectangle(frame, (0, pred_y_pos), (width-1, pred_y_pos + bar_height), pred_color_base, -1) # Background bar
                    cv2.rectangle(frame, (0, pred_y_pos), (width-1, pred_y_pos + bar_height), colors[action_id % len(colors)], 2) # Border
                    cv2.putText(frame, label_text, (5, pred_y_pos + text_y_offset), font, font_scale, (0,0,0), font_thickness, cv2.LINE_AA)
                    break # Only draw one Pred label per class on the bar
            if pred_labels_on_frame:
                 pass # Allow multiple preds potentially

        # Write Frame Index
        frame_text = f"Frame: {frame_idx}/{num_frames}"
        cv2.putText(frame, frame_text, (width - 150, height - 15), font, font_scale, (255, 255, 255), font_thickness, cv2.LINE_AA)

        # Write frame to video
        out.write(frame)

    # Release video writer
    out.release()
    print(f"--- Visualization finished for Video: {video_id} ---")
    print(f"  Output saved to: {output_video_path}")

# ====== Main Evaluation Function Structure ======
def main_evaluate(args):
    print(f"Using device: {DEVICE}")
    print(f"Evaluating base model checkpoint: {args.base_checkpoint_path}")
    print(f"Using RNN post-processor checkpoint: {args.rnn_checkpoint_path}")
    
    # --- Load Base Model Inference Results (Step 0/1) ---
    all_raw_preds = None
    all_batch_meta = None
    if args.skip_inference:
        if args.inference_output_path and os.path.exists(args.inference_output_path):
            print(f"\n-- Attempting to load pre-computed inference results from: {args.inference_output_path} --")
            try:
                with open(args.inference_output_path, 'rb') as f:
                    inference_results = pickle.load(f)
                all_raw_preds = inference_results['all_raw_preds']
                all_batch_meta = inference_results['all_batch_meta']
                # avg_val_loss = inference_results.get('avg_val_loss', None) # Optional loss value
                print(f"Successfully loaded inference results for {len(all_raw_preds)} batches.")
            except Exception as e:
                print(f"Error loading inference results: {e}. Need to run inference.")
                all_raw_preds = None # Force inference run
        else:
             print(f"Error: --skip_inference flag set, but file not found: {args.inference_output_path}")
             return # Cannot proceed without inference data

    if all_raw_preds is None: # Need to run inference
        print("\nRunning Base Model Inference...")
        # Initialize base model
        base_model = TemporalActionDetector(num_classes=NUM_CLASSES, window_size=WINDOW_SIZE) 
        if os.path.exists(args.base_checkpoint_path):
            print(f"Loading base checkpoint...")
            try:
                checkpoint = torch.load(args.base_checkpoint_path, map_location=DEVICE)
                if 'model_state_dict' in checkpoint: state_dict = checkpoint['model_state_dict']
                elif 'state_dict' in checkpoint: state_dict = checkpoint['state_dict']
                else: state_dict = checkpoint
                base_model.load_state_dict(state_dict)
                print(f"Loaded base model weights from epoch {checkpoint.get('epoch', 'N/A')}")
            except Exception as e:
                print(f"Error loading base checkpoint: {e}")
                return
        else:
            print(f"Error: Base checkpoint file not found: {args.base_checkpoint_path}")
            return
        base_model.to(DEVICE)
        base_model.eval()
        
        # Get dataloader
        val_loader = get_test_loader(batch_size=args.batch_size, shuffle=False) # Use eval batch size
        
        # Run inference (reuse function from 1_generate_rnn_data.py? Need to import or copy)
        # For simplicity, let's copy the core logic here
        all_raw_preds = []
        all_batch_meta = []
        print(f"Running inference on {len(val_loader.dataset)} validation samples...")
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Base Model Inference"):
                try: frames, pose_data, _, _, _, _, metadata = batch
                except ValueError: print("Error: Batch structure mismatch."); exit()
                frames = frames.to(DEVICE)
                if pose_data is not None: pose_data = pose_data.to(DEVICE)
                with torch.cuda.amp.autocast(enabled=USE_MIXED_PRECISION):
                    predictions = base_model(frames, pose_data)
                action_probs = torch.sigmoid(predictions['action_scores']).cpu().detach()
                start_probs = torch.sigmoid(predictions['start_scores']).cpu().detach()
                end_probs = torch.sigmoid(predictions['end_scores']).cpu().detach()
                all_raw_preds.append((action_probs, start_probs, end_probs))
                all_batch_meta.append(copy.deepcopy(metadata))
        print("Base model inference complete.")
        # Optionally save these results
        if args.inference_output_path:
             print(f"Saving inference results to {args.inference_output_path}...")
             try:
                 # Ensure directory exists
                 output_dir = os.path.dirname(args.inference_output_path)
                 if output_dir and not os.path.exists(output_dir):
                     os.makedirs(output_dir)
                 with open(args.inference_output_path, 'wb') as f:
                     pickle.dump({'all_raw_preds': all_raw_preds, 'all_batch_meta': all_batch_meta}, f)
                 print("Saved inference results.")
             except Exception as e:
                 print(f"Error saving inference results: {e}")

    # --- Calculate Global Ground Truth (Step 2 - Same as before) ---
    print("\nStep 2: Calculating Global Ground Truth...")
    global_action_gt_by_video = defaultdict(lambda: defaultdict(list))
    all_metadata_flat = [meta for batch_meta in all_batch_meta for meta in batch_meta]
    unique_video_ids_from_meta = set()
    for meta in all_metadata_flat:
         video_id = meta['video_id']
         unique_video_ids_from_meta.add(video_id)
         start_idx = meta['start_idx']
         # Ensure annotations exist and are iterable
         if isinstance(meta.get('annotations'), list):
             for anno in meta['annotations']:
                 # Basic validation of annotation structure
                 if isinstance(anno, dict) and all(k in anno for k in ('action_id', 'start_frame', 'end_frame')):
                     action_id = anno['action_id']
                     global_gt_start = start_idx + anno['start_frame']
                     global_gt_end = start_idx + anno['end_frame']
                     if 0 <= action_id < NUM_CLASSES: # Ensure valid action_id
                          global_action_gt_by_video[video_id][action_id].append((global_gt_start, global_gt_end))
                 else:
                      print(f"Warning: Invalid annotation format found in metadata for video {video_id}: {anno}")
         elif 'annotations' in meta: # If key exists but is not a list
              print(f"Warning: Invalid 'annotations' type in metadata for video {video_id}: {type(meta['annotations'])}")

    final_global_gt = defaultdict(list)
    for video_id, actions in global_action_gt_by_video.items():
         for action_id, segments in actions.items():
             unique_segments_video_action = sorted(list(set(segments)))
             final_global_gt[action_id].extend(unique_segments_video_action)
    final_global_gt = {c: sorted(list(set(final_global_gt.get(c, [])))) for c in range(NUM_CLASSES)}
    total_global_gt_segments = sum(len(v) for v in final_global_gt.values())
    print(f"Global GT calculated. Total unique GT segments: {total_global_gt_segments}")
    for c in range(NUM_CLASSES):
        print(f"  Class {c} GT count: {len(final_global_gt.get(c, []))}")

    # --- Load RNN Model (NEW Step 3) ---
    print("\nStep 3: Loading RNN Post-Processor Model...")
    if not os.path.exists(args.rnn_checkpoint_path):
        print(f"Error: RNN checkpoint not found at {args.rnn_checkpoint_path}")
        return
        
    try:
        rnn_checkpoint = torch.load(args.rnn_checkpoint_path, map_location=DEVICE)
        rnn_args = rnn_checkpoint.get('args', None) # Get args used during RNN training
        if rnn_args is None:
             print("Warning: RNN checkpoint does not contain training arguments. Using default/current args for model init.")
             # Use current script defaults or args if provided, need consistency
             # For simplicity, let's assume the necessary args are available or use fixed values
             # This is risky, best practice is to save args with the RNN checkpoint
             # Re-create args dict based on known defaults/structure
             rnn_args = {
                 'rnn_type': 'lstm', # Assuming default if not found
                 'hidden_size': 128,  # Assuming default
                 'num_layers': 2,    # Assuming default
                 'dropout_prob': 0.5, # Assuming default
                 'bidirectional': True # Assuming default
             }
             print(f"Using assumed RNN params: {rnn_args}")

        # Determine input size (should be fixed based on NUM_CLASSES)
        rnn_input_size = 3 * NUM_CLASSES 
        # Determine output classes (should be fixed)
        rnn_num_classes_out = NUM_CLASSES + 1
        
        rnn_model = RNNPostProcessor(
            input_size=rnn_input_size,
            hidden_size=rnn_args['hidden_size'],
            num_layers=rnn_args['num_layers'],
            num_classes=rnn_num_classes_out,
            rnn_type=rnn_args['rnn_type'],
            dropout_prob=rnn_args['dropout_prob'], # Use dropout from saved args
            bidirectional=rnn_args['bidirectional']
        ).to(DEVICE)
        
        rnn_model.load_state_dict(rnn_checkpoint['model_state_dict'])
        rnn_model.eval() # Set to evaluation mode
        print(f"Successfully loaded RNN model from epoch {rnn_checkpoint.get('epoch', 'N/A')}")
        print(f"  RNN Val Loss during training: {rnn_checkpoint.get('val_loss', 'N/A'):.4f}")
        
    except Exception as e:
        print(f"Error loading RNN checkpoint: {e}")
        return

    # --- Run RNN Post-Processing (NEW Step 4) ---
    print("\nStep 4: Running RNN Post-Processing...")
    rnn_predictions_by_video = defaultdict(lambda: defaultdict(list))
    rnn_all_action_preds_flat = defaultdict(list)
    
    # Use video IDs found in the metadata from inference run
    unique_video_ids_to_process = sorted(list(unique_video_ids_from_meta))
    if not unique_video_ids_to_process:
         print("Warning: No unique video IDs found in metadata. Cannot run RNN post-processing.")
         return

    for video_id in tqdm(unique_video_ids_to_process, desc="RNN Processing Videos"):
        # 1. Reconstruct input features (probabilities) for this video
        avg_action_probs, avg_start_probs, avg_end_probs, num_frames = reconstruct_full_video_probs(
            video_id, all_raw_preds, all_batch_meta
        )
        
        if num_frames is None or num_frames <= 0:
            # print(f"Skipping RNN for {video_id}: Could not reconstruct probabilities.")
            continue

        # 2. Combine features and prepare for RNN input
        input_features = np.concatenate([avg_action_probs, avg_start_probs, avg_end_probs], axis=1)
        input_tensor = torch.tensor(input_features, dtype=torch.float32).unsqueeze(0).to(DEVICE) # Add batch dim
        
        # 3. Run RNN model
        with torch.no_grad():
            logits = rnn_model(input_tensor) # Shape: (1, T, num_classes_out)
            # Calculate softmax probabilities ONCE per video
            probs = torch.softmax(logits.squeeze(0), dim=1) # Shape: (T, num_classes_out)

        # 4. Get predicted labels
        predicted_labels = torch.argmax(logits.squeeze(0), dim=1).cpu().numpy() # Shape: (T,)

        # 5. Convert labels to segments
        video_segments = labels_to_segments(predicted_labels, ignore_label=BACKGROUND_LABEL)

        # 6. Store segments WITH calculated confidence scores
        # Calculate score based on average max probability within the segment
        for action_id, segments in video_segments.items():
            processed_segments = []
            for s in segments:
                start, end = s['start_frame'], s['end_frame']
                if end > start: # Ensure segment is not empty
                    # Extract probabilities for the frames in this segment
                    segment_probs = probs[start:end, :] # Shape: (segment_len, num_classes_out)
                    if segment_probs.numel() > 0:
                         # Get the probabilities of the specific action_id for frames in the segment
                         probs_of_action_id = segment_probs[:, action_id] # Shape: (segment_len,)
                         # Calculate the mean probability as the score
                         segment_score = torch.mean(probs_of_action_id).item()
                    else:
                         segment_score = 0.0 # Assign 0 score if segment is somehow empty after all

                    processed_segments.append({'segment': (start, end), 'score': segment_score})

            if processed_segments:
                rnn_predictions_by_video[video_id][action_id] = processed_segments
                # Ensure the segment info added here contains the calculated score
                rnn_all_action_preds_flat[action_id].extend(processed_segments)

    # --- Calculate Metrics using RNN Results (NEW Step 5, 6, 7) ---
    print("\nStep 5, 6, 7: Calculating metrics for RNN Post-Processed results...")

    # Calculate mAP (IoU based)
    rnn_mAP = calculate_mAP(final_global_gt, rnn_all_action_preds_flat)
    
    # Calculate class-wise AP
    rnn_class_aps = {}
    print("Calculating Average Precision (AP) for each class (RNN)...")
    for c in range(NUM_CLASSES):
        gt_c = final_global_gt.get(c, []) 
        preds_c = rnn_all_action_preds_flat.get(c, [])
        rnn_class_aps[c] = calculate_class_mAP(gt_c, preds_c)
        print(f"  Class {c} AP: {rnn_class_aps[c]:.4f}")
        
    # Calculate mAP@mid 
    rnn_map_mid = calculate_map_mid(final_global_gt, rnn_all_action_preds_flat)
    
    # Calculate F1 scores
    iou_thresholds_list = [0.1, 0.25, 0.5]
    rnn_avg_f1_scores = {} 
    rnn_class_f1_scores_by_iou = {}
    print("Calculating F1 scores at different IoU thresholds (RNN)...")
    for threshold in iou_thresholds_list:
        all_class_f1_for_this_iou = []
        current_iou_class_f1s = {}
        print(f"  Threshold: {threshold:.2f}")
        for c in range(NUM_CLASSES):
            gt_c = final_global_gt.get(c, []) 
            preds_c = rnn_all_action_preds_flat.get(c, [])
            if not gt_c: class_f1 = 0.0 
            else: _, _, class_f1 = calculate_f1_at_iou(gt_c, preds_c, threshold)
            all_class_f1_for_this_iou.append(class_f1)
            current_iou_class_f1s[c] = class_f1 
        avg_f1_for_this_iou = np.mean(all_class_f1_for_this_iou) if all_class_f1_for_this_iou else 0.0
        key_name = f'avg_f1_iou_{int(threshold*100):03d}'
        rnn_avg_f1_scores[key_name] = avg_f1_for_this_iou
        rnn_class_f1_scores_by_iou[threshold] = current_iou_class_f1s 
        print(f"  Average F1 @ {threshold:.2f}: {avg_f1_for_this_iou:.4f}")

    # Calculate Global Frame-Level metrics
    rnn_all_frame_targets_flat = []
    rnn_all_frame_preds_flat = []
    all_involved_videos_rnn = set(global_action_gt_by_video.keys()) | set(rnn_predictions_by_video.keys())
    print(f"Calculating Global Frame-Level metrics for {len(all_involved_videos_rnn)} videos (RNN)...")
    for video_id in tqdm(all_involved_videos_rnn, desc="Global F1 Calc (RNN)"):
        video_max_frame = 0
        if video_id in global_action_gt_by_video:
            for segments in global_action_gt_by_video[video_id].values():
                for _, end in segments: video_max_frame = max(video_max_frame, end)
        if video_id in rnn_predictions_by_video:
            for segments in rnn_predictions_by_video[video_id].values():
                 for seg_info in segments: _, end = seg_info['segment']; video_max_frame = max(video_max_frame, end)
        video_length = video_max_frame + 1 
        if video_length <= 1 : continue
        video_targets = np.zeros((video_length, NUM_CLASSES), dtype=int)
        video_preds = np.zeros((video_length, NUM_CLASSES), dtype=int)
        if video_id in global_action_gt_by_video:
            for c, segments in global_action_gt_by_video[video_id].items():
                for start, end in set(segments):
                    if end > start and 0 <= c < NUM_CLASSES: video_targets[start:min(end, video_length), c] = 1
        if video_id in rnn_predictions_by_video:
             for c, segments in rnn_predictions_by_video[video_id].items():
                  for seg_info in segments:
                       start, end = seg_info['segment']
                       if end > start and 0 <= c < NUM_CLASSES: video_preds[start:min(end, video_length), c] = 1
        rnn_all_frame_targets_flat.extend(video_targets.flatten())
        rnn_all_frame_preds_flat.extend(video_preds.flatten())
    if rnn_all_frame_targets_flat: 
        rnn_merged_precision, rnn_merged_recall, rnn_merged_f1, _ = precision_recall_fscore_support(
            rnn_all_frame_targets_flat, rnn_all_frame_preds_flat, average='macro', zero_division=0
        )
        # Calculate frame-wise accuracy
        rnn_frame_accuracy = np.mean(np.array(rnn_all_frame_targets_flat) == np.array(rnn_all_frame_preds_flat)) if rnn_all_frame_targets_flat else 0.0
        print(f"RNN Global Frame-Level Metrics: Precision={rnn_merged_precision:.4f}, Recall={rnn_merged_recall:.4f}, F1={rnn_merged_f1:.4f}")
        print(f"RNN Frame-wise Accuracy: {rnn_frame_accuracy:.4f}") # Print the new accuracy here as well
    else:
        rnn_merged_precision, rnn_merged_recall, rnn_merged_f1 = 0.0, 0.0, 0.0
        rnn_frame_accuracy = 0.0 # Ensure it's defined even if no data
        print("Warning: No frame data available for global frame-level metrics.")

    # --- Print Final Results (Step 8 - Adjusted) ---
    print("\n\n--- RNN Post-Processing Evaluation Results ---")
    # avg_val_loss_base = # Load if available from inference pickle
    # print(f"Base Avg Validation Loss (from inference): {avg_val_loss_base:.4f}") # Optional
    print(f"RNN mAP (IoU 0.3,0.5,0.7): {rnn_mAP:.4f}")
    print(f"RNN mAP@mid: {rnn_map_mid:.4f}")
    print(f"RNN Global Frame-level -> Precision: {rnn_merged_precision:.4f}, Recall: {rnn_merged_recall:.4f}, F1: {rnn_merged_f1:.4f}")
    print(f"RNN Frame-wise Accuracy: {rnn_frame_accuracy:.4f}") # Replace with this line
    print(f"RNN Avg F1@0.10: {rnn_avg_f1_scores.get('avg_f1_iou_010', 0.0):.4f}")
    print(f"RNN Avg F1@0.25: {rnn_avg_f1_scores.get('avg_f1_iou_025', 0.0):.4f}")
    print(f"RNN Avg F1@0.50: {rnn_avg_f1_scores.get('avg_f1_iou_050', 0.0):.4f}")
    header = "Class | AP@0.5 | Preds | F1@0.1 | F1@0.5"
    print(header)
    print("-" * len(header))
    for c in range(NUM_CLASSES):
        ap = rnn_class_aps.get(c, 0.0)
        preds_count = len(rnn_all_action_preds_flat.get(c, []))
        f1_010 = rnn_class_f1_scores_by_iou.get(0.1, {}).get(c, 0.0)
        f1_050 = rnn_class_f1_scores_by_iou.get(0.5, {}).get(c, 0.0)
        print(f"{c:<5} | {ap:.4f} | {preds_count:<5} | {f1_010:.4f} | {f1_050:.4f}")
    print("-" * len(header))
    # Calculate total predictions again here, as the previous calculation was removed
    total_rnn_preds = sum(len(v) for v in rnn_all_action_preds_flat.values())
    # Recalculate total GT segments as well
    total_global_gt_segments_recalc = sum(len(v) for v in final_global_gt.values())
    print(f"Total RNN Pred Segments: {total_rnn_preds}") 
    print(f"Total GT Segments: {total_global_gt_segments_recalc}")

    # --- Optional Visualization (Step 9) ---
    if args.visualize_video_id:
        print(f"\n--- Preparing for Visualization of Video ID: {args.visualize_video_id} ---")
        if not args.frames_npz_path_template:
             print("Error: --frames_npz_path_template is required for visualization.")
        elif not args.output_video_path:
             print("Error: --output_video_path is required for visualization.")
        else:
            # Construct the specific NPZ path for the requested video
            try:
                vis_npz_path = args.frames_npz_path_template.format(video_id=args.visualize_video_id)
            except Exception as e:
                 print(f"Error formatting frames_npz_path_template: {e}")
                 vis_npz_path = None

            if vis_npz_path and os.path.exists(vis_npz_path):
                visualize_rnn_predictions(
                    video_id=args.visualize_video_id,
                    frames_npz_path=vis_npz_path,
                    output_video_path=args.output_video_path,
                    fps=args.fps,
                    global_gt_data_by_video=global_action_gt_by_video, # Use the per-video GT
                    rnn_preds_by_video=rnn_predictions_by_video,      # Use the per-video Preds
                    num_classes=NUM_CLASSES
                )
            elif vis_npz_path:
                 print(f"Error: Visualization frames file not found: {vis_npz_path}")
            # No need for an else here, previous error messages cover it

# ====== Argument Parser ======
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Base Model + RNN Post-Processor")
    parser.add_argument("--base_checkpoint_path", type=str, default=DEFAULT_BASE_CHECKPOINT, help="Path to the base model checkpoint (.pth)")
    parser.add_argument("--rnn_checkpoint_path", type=str, default=DEFAULT_RNN_CHECKPOINT, help="Path to the trained RNN post-processor checkpoint (.pth)")
    parser.add_argument("--inference_output_path", type=str, default=DEFAULT_TEST_PKL, help="Path to load/save base model inference results (.pkl) for the validation set")
    parser.add_argument("--skip_inference", action="store_true", help="Skip base model inference and load results from --inference_output_path")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for base model inference if run")
    parser.add_argument("--num_workers", type=int, default=0, help="Number of workers for dataloader")
    # Add visualization args if adapting visualization later
    parser.add_argument("--visualize_video_id", type=str, default=None, help="Optional video ID to visualize")
    parser.add_argument("--frames_npz_path_template", type=str, default=None, help="Path template for frame NPZ files")
    parser.add_argument("--output_video_path", type=str, default=None, help="Path to save visualization video")
    parser.add_argument("--fps", type=int, default=15, help="FPS for visualization video")

    args = parser.parse_args()
    main_evaluate(args) 