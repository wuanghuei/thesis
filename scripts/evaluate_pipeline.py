import torch
import os
import argparse
import pickle
import numpy as np
from tqdm import tqdm
from collections import defaultdict

# Updated imports to use moved functions/classes
from src.utils.helpers import reconstruct_full_video_probs, calculate_global_gt
from src.utils.postprocessing import labels_to_segments
from src.utils.visualization import visualize_rnn_predictions
# Import the consolidated metrics calculation function
from src.evaluation import compute_final_metrics 
# Import only necessary models
try:
    from src.models.rnn_postprocessor import RNNPostProcessor
except ImportError:
    print("Error: Could not import RNNPostProcessor from src/models/rnn_postprocessor.py")
    exit()

# ====== Configuration ======
# Constants that might be moved to a config file later
NUM_CLASSES = 5
BACKGROUND_LABEL = NUM_CLASSES # Consistent label for background state
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Default paths (can be overridden by args)
DEFAULT_RNN_CHECKPOINT = "best/rnn_checkpoints/best_rnn_model.pth"
DEFAULT_INFERENCE_PKL = "best/test_inference_raw.pkl" # Default PKL for evaluation

# ====== Helper Function for RNN Processing Loop ======
def _run_rnn_on_all_videos(rnn_model, all_raw_preds, all_batch_meta, global_action_gt_by_video, device, num_classes, background_label):
    """Runs the loaded RNN model on reconstructed features for all videos."""
    print("\nStep 4: Running RNN Post-Processing...")
    rnn_predictions_by_video = defaultdict(lambda: defaultdict(list))
    rnn_all_action_preds_flat = defaultdict(list)
    
    # Use video IDs found in the metadata from inference run
    unique_video_ids_to_process = sorted(list(global_action_gt_by_video.keys()))
    if not unique_video_ids_to_process:
         print("Warning: No unique video IDs found in metadata. Cannot run RNN post-processing.")
         return rnn_predictions_by_video, rnn_all_action_preds_flat

    for video_id in tqdm(unique_video_ids_to_process, desc="RNN Processing Videos"):
        # 1. Reconstruct input features (probabilities) for this video
        # Assumes reconstruct_full_video_probs now takes num_classes and window_size if needed implicitly from data dims
        avg_action_probs, avg_start_probs, avg_end_probs, num_frames = reconstruct_full_video_probs(
            video_id, all_raw_preds, all_batch_meta
        )
        
        if num_frames is None or num_frames <= 0:
            continue

        # 2. Combine features and prepare for RNN input
        # Assuming input size is 3 * num_classes
        input_features = np.concatenate([avg_action_probs, avg_start_probs, avg_end_probs], axis=1)
        input_tensor = torch.tensor(input_features, dtype=torch.float32).unsqueeze(0).to(device) # Add batch dim
        
        # 3. Run RNN model
        with torch.no_grad():
            logits = rnn_model(input_tensor) # Shape: (1, T, num_classes_out)
            # Calculate softmax probabilities ONCE per video
            probs = torch.softmax(logits.squeeze(0), dim=1) # Shape: (T, num_classes_out)

        # 4. Get predicted labels
        predicted_labels = torch.argmax(logits.squeeze(0), dim=1).cpu().numpy() # Shape: (T,)

        # 5. Convert labels to segments
        video_segments = labels_to_segments(predicted_labels, ignore_label=background_label)

        # 6. Store segments WITH calculated confidence scores
        for action_id, segments in video_segments.items():
            processed_segments = []
            for s in segments:
                start, end = s['start_frame'], s['end_frame']
                if end > start: # Ensure segment is not empty
                    segment_probs = probs[start:end, :] # Shape: (segment_len, num_classes_out)
                    if segment_probs.numel() > 0:
                         probs_of_action_id = segment_probs[:, action_id] # Shape: (segment_len,)
                         segment_score = torch.mean(probs_of_action_id).item()
                    else:
                         segment_score = 0.0 

                    processed_segments.append({'segment': (start, end), 'score': segment_score})

            if processed_segments:
                rnn_predictions_by_video[video_id][action_id] = processed_segments
                rnn_all_action_preds_flat[action_id].extend(processed_segments)
                
    return rnn_predictions_by_video, rnn_all_action_preds_flat


# ====== Main Evaluation Function Structure ======
def main_evaluate(args):
    print(f"Using device: {DEVICE}")
    print(f"Evaluating using RNN post-processor checkpoint: {args.rnn_checkpoint_path}")
    print(f"Loading base model inference results from: {args.inference_output_path}")
    
    # --- Load Base Model Inference Results (Step 1) ---
    # No longer runs inference, directly loads or fails.
    if not args.inference_output_path or not os.path.exists(args.inference_output_path):
        print(f"Error: Inference results file not found or not specified: {args.inference_output_path}")
        return

    print(f"\n-- Loading pre-computed inference results from: {args.inference_output_path} --")
    try:
        with open(args.inference_output_path, 'rb') as f:
            inference_results = pickle.load(f)
        all_raw_preds = inference_results['all_raw_preds']
        all_batch_meta = inference_results['all_batch_meta']
        print(f"Successfully loaded inference results for {len(all_raw_preds)} batches.")
    except Exception as e:
        print(f"Error loading inference results: {e}.")
        return

    # --- Calculate Global Ground Truth (Step 2) ---
    print("\nStep 2: Calculating Global Ground Truth...")
    final_global_gt, total_global_gt_segments, global_action_gt_by_video = calculate_global_gt(all_batch_meta, NUM_CLASSES)
    print(f"Global GT calculated. Total unique GT segments: {total_global_gt_segments}")
    for c in range(NUM_CLASSES):
        print(f"  Class {c} GT count: {len(final_global_gt.get(c, []))}")

    # --- Load RNN Model (Step 3) ---
    print("\nStep 3: Loading RNN Post-Processor Model...")
    if not os.path.exists(args.rnn_checkpoint_path):
        print(f"Error: RNN checkpoint not found at {args.rnn_checkpoint_path}")
        return
        
    try:
        rnn_checkpoint = torch.load(args.rnn_checkpoint_path, map_location=DEVICE)
        rnn_args = rnn_checkpoint.get('args', None) 
        if rnn_args is None:
             print("Warning: RNN checkpoint does not contain training arguments. Assuming defaults.")
             rnn_args = { 'rnn_type': 'lstm', 'hidden_size': 128, 'num_layers': 2, 'dropout_prob': 0.5, 'bidirectional': True }
             print(f"Using assumed RNN params: {rnn_args}")

        rnn_input_size = 3 * NUM_CLASSES 
        rnn_num_classes_out = NUM_CLASSES + 1
        
        rnn_model = RNNPostProcessor(
            input_size=rnn_input_size,
            hidden_size=rnn_args['hidden_size'],
            num_layers=rnn_args['num_layers'],
            num_classes=rnn_num_classes_out,
            rnn_type=rnn_args['rnn_type'],
            dropout_prob=rnn_args['dropout_prob'], 
            bidirectional=rnn_args['bidirectional']
        ).to(DEVICE)
        
        rnn_model.load_state_dict(rnn_checkpoint['model_state_dict'])
        rnn_model.eval() 
        print(f"Successfully loaded RNN model from epoch {rnn_checkpoint.get('epoch', 'N/A')}")
        
    except Exception as e:
        print(f"Error loading RNN checkpoint: {e}")
        return

    # --- Run RNN Post-Processing using Helper Function (Step 4) ---
    rnn_predictions_by_video, rnn_all_action_preds_flat = _run_rnn_on_all_videos(
        rnn_model, all_raw_preds, all_batch_meta, global_action_gt_by_video, 
        DEVICE, NUM_CLASSES, BACKGROUND_LABEL
    )

    # --- Calculate Final Metrics using Imported Function (Step 5, 6, 7) ---
    print("\nStep 5, 6, 7: Calculating final metrics...")
    
    # Prepare data for compute_final_metrics (it needs global frame preds/targets)
    # Re-calculate flattened frame predictions/targets based on RNN output
    rnn_all_frame_targets_flat = []
    rnn_all_frame_preds_flat_for_metric = [] # Renamed to avoid confusion
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
        
        # Populate targets
        if video_id in global_action_gt_by_video:
            for c, segments in global_action_gt_by_video[video_id].items():
                for start, end in set(segments): # Use set to avoid double counting from overlapping windows
                    if end > start and 0 <= c < NUM_CLASSES: 
                         video_targets[start:min(end, video_length), c] = 1
                         
        # Populate predictions based on RNN segments
        if video_id in rnn_predictions_by_video:
             for c, segments in rnn_predictions_by_video[video_id].items():
                  for seg_info in segments:
                       start, end = seg_info['segment']
                       if end > start and 0 <= c < NUM_CLASSES: 
                            video_preds[start:min(end, video_length), c] = 1
                            
        rnn_all_frame_targets_flat.extend(video_targets.flatten())
        rnn_all_frame_preds_flat_for_metric.extend(video_preds.flatten()) # Use the renamed list

    # Now call the centralized metric calculation function
    final_metrics = compute_final_metrics(
        global_action_gt_global=final_global_gt, # Use globally unique GT
        merged_all_action_preds=rnn_all_action_preds_flat, # Use flattened RNN preds
        merged_all_frame_targets=rnn_all_frame_targets_flat, # Use flattened frame targets
        merged_all_frame_preds=rnn_all_frame_preds_flat_for_metric, # Use flattened frame preds
        num_classes=NUM_CLASSES
    )

    # --- Print Final Results (Step 8 - Adjusted) ---
    print("\n\n--- RNN Post-Processing Evaluation Results ---")
    print(f"RNN mAP (IoU 0.3,0.5,0.7): {final_metrics.get('mAP', 0.0):.4f}")
    print(f"RNN mAP@mid: {final_metrics.get('map_mid', 0.0):.4f}")
    print(f"RNN Global Frame-level -> F1: {final_metrics.get('merged_f1', 0.0):.4f}") # Assuming compute_final_metrics returns 'merged_f1'
    # Add other frame-level metrics if returned by compute_final_metrics
    # print(f"RNN Frame-wise Accuracy: {final_metrics.get('frame_accuracy', 0.0):.4f}") 
    print(f"RNN Avg F1@0.10: {final_metrics.get('avg_f1_iou_010', 0.0):.4f}")
    print(f"RNN Avg F1@0.25: {final_metrics.get('avg_f1_iou_025', 0.0):.4f}")
    print(f"RNN Avg F1@0.50: {final_metrics.get('avg_f1_iou_050', 0.0):.4f}")
    
    header = "Class | AP@0.5 | Preds | F1@0.1 | F1@0.5" # Assuming AP is @0.5 from compute_final_metrics class_aps
    print(header)
    print("-" * len(header))
    # Access class-specific results from the dictionary returned by compute_final_metrics
    class_aps = final_metrics.get('class_aps', {})
    # Need class F1 scores if compute_final_metrics provides them, otherwise cannot print here
    # class_f1_010 = final_metrics.get('class_f1_iou_010', {}) 
    # class_f1_050 = final_metrics.get('class_f1_iou_050', {})
    for c in range(NUM_CLASSES):
        ap = class_aps.get(c, 0.0)
        preds_count = len(rnn_all_action_preds_flat.get(c, [])) # Keep preds count calculation here
        # f1_010 = class_f1_010.get(c, 0.0) # Get class F1 if available
        # f1_050 = class_f1_050.get(c, 0.0) # Get class F1 if available
        # Print only available metrics - adjust format if class F1s aren't available
        print(f"{c:<5} | {ap:.4f} | {preds_count:<5} | N/A    | N/A") # Placeholder if class F1 not returned
    print("-" * len(header))
    total_rnn_preds = sum(len(v) for v in rnn_all_action_preds_flat.values())
    print(f"Total RNN Pred Segments: {total_rnn_preds}") 
    print(f"Total GT Segments: {total_global_gt_segments}") # Use value calculated earlier

    # --- Optional Visualization (Step 9) ---
    if args.visualize_video_id:
        print(f"\n--- Preparing for Visualization of Video ID: {args.visualize_video_id} ---")
        if not args.frames_npz_path_template:
             print("Error: --frames_npz_path_template is required for visualization.")
        elif not args.output_video_path:
             print("Error: --output_video_path is required for visualization.")
        else:
            try:
                vis_npz_path = args.frames_npz_path_template.format(video_id=args.visualize_video_id)
            except Exception as e:
                 print(f"Error formatting frames_npz_path_template: {e}")
                 vis_npz_path = None

            if vis_npz_path and os.path.exists(vis_npz_path):
                # Ensure the visualization function uses the correct variables
                visualize_rnn_predictions(
                    video_id=args.visualize_video_id,
                    frames_npz_path=vis_npz_path,
                    output_video_path=args.output_video_path,
                    fps=args.fps,
                    global_gt_data_by_video=global_action_gt_by_video, # Use the per-video GT
                    rnn_preds_by_video=rnn_predictions_by_video,      # Use the per-video Preds from RNN helper
                    num_classes=NUM_CLASSES
                )
            elif vis_npz_path:
                 print(f"Error: Visualization frames file not found: {vis_npz_path}")

# ====== Argument Parser ======
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Base Model + RNN Post-Processor Pipeline")
    # Removed args related to base model inference
    parser.add_argument("--rnn_checkpoint_path", type=str, required=True, help="Path to the trained RNN post-processor checkpoint (.pth)")
    parser.add_argument("--inference_output_path", type=str, required=True, help="Path to load base model inference results (.pkl)")
    
    # Visualization args remain
    parser.add_argument("--visualize_video_id", type=str, default=None, help="Optional video ID to visualize")
    parser.add_argument("--frames_npz_path_template", type=str, default=None, help="Path template for frame NPZ files (e.g., 'Data/full_videos/test/frames/{video_id}_frames.npz')")
    parser.add_argument("--output_video_path", type=str, default=None, help="Path to save visualization video")
    parser.add_argument("--fps", type=int, default=15, help="FPS for visualization video")

    args = parser.parse_args()
    main_evaluate(args) 