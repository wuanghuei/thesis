import torch
import os
import argparse
import pickle
import numpy as np
from tqdm import tqdm
from collections import defaultdict
import yaml

from src.utils.helpers import reconstruct_full_video_probs, calculate_global_gt
from src.utils.postprocessing import labels_to_segments
from src.utils.visualization import visualize_rnn_predictions
from src.evaluation import compute_final_metrics
try:
    from src.models.rnn_postprocessor import RNNPostProcessor
except ImportError:
    print("Error: Could not import RNNPostProcessor from src/models/rnn_postprocessor.py")
    exit()
    

def _run_rnn_on_all_videos(rnn_model, all_raw_preds, all_batch_meta, global_action_gt_by_video, device, num_classes, background_label):
    print("\nStep 4: Running RNN Post-Processing...")
    rnn_predictions_by_video = defaultdict(lambda: defaultdict(list))
    rnn_all_action_preds_flat = defaultdict(list)
    
    unique_video_ids_to_process = sorted(list(global_action_gt_by_video.keys()))
    if not unique_video_ids_to_process:
         print("No unique video IDs found in metadata. Cannot run RNN post-processing.")
         return rnn_predictions_by_video, rnn_all_action_preds_flat

    for video_id in tqdm(unique_video_ids_to_process, desc="RNN Processing Videos"):
        avg_action_probs, avg_start_probs, avg_end_probs, num_frames = reconstruct_full_video_probs(
            video_id, all_raw_preds, all_batch_meta
        )      
        if num_frames is None or num_frames <= 0:
            continue
        input_features = np.concatenate([avg_action_probs, avg_start_probs, avg_end_probs], axis=1)
        input_tensor = torch.tensor(input_features, dtype=torch.float32).unsqueeze(0).to(device)
        
        with torch.no_grad():
            logits = rnn_model(input_tensor)
            probs = torch.softmax(logits.squeeze(0), dim=1)
        predicted_labels = torch.argmax(logits.squeeze(0), dim=1).cpu().numpy()
        video_segments = labels_to_segments(predicted_labels, ignore_label=background_label)
        for action_id, segments in video_segments.items():
            processed_segments = []
            for s in segments:
                start, end = s['start_frame'], s['end_frame']
                if end > start:
                    segment_probs = probs[start:end, :]
                    if segment_probs.numel() > 0:
                         probs_of_action_id = segment_probs[:, action_id]
                         segment_score = torch.mean(probs_of_action_id).item()
                    else:
                         segment_score = 0.0 
                    processed_segments.append({'segment': (start, end), 'score': segment_score})
            if processed_segments:
                rnn_predictions_by_video[video_id][action_id] = processed_segments
                rnn_all_action_preds_flat[action_id].extend(processed_segments)

    return rnn_predictions_by_video, rnn_all_action_preds_flat

def main_evaluate(cfg, args):
    if cfg['global']['device'] == 'auto':
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(cfg['global']['device'])
    
    num_classes = cfg['global']['num_classes']
    background_label = num_classes
    rnn_checkpoint_path = args.rnn_checkpoint_path if args.rnn_checkpoint_path else cfg['pipeline_evaluation']['rnn_checkpoint_to_use']
    inference_output_path = args.inference_output_path if args.inference_output_path else cfg['pipeline_evaluation']['inference_results_pkl']

    print(f"Using device: {device}")
    print(f"Evaluating using RNN post-processor checkpoint: {rnn_checkpoint_path}")
    print(f"Loading base model inference results from: {inference_output_path}")
    
    if not inference_output_path or not os.path.exists(inference_output_path):
        print(f"Error: Inference results file not found or not specified: {inference_output_path}")
        return

    print(f"\n-- Loading pre-computed inference results from: {inference_output_path} --")
    try:
        with open(inference_output_path, 'rb') as f:
            inference_results = pickle.load(f)
        all_raw_preds = inference_results['all_raw_preds']
        all_batch_meta = inference_results['all_batch_meta']
        print(f"Successfully loaded inference results for {len(all_raw_preds)} batches.")
    except Exception as e:
        print(f"Error loading inference results: {e}.")
        return

    print("\nStep 2: Calculating Global Ground Truth...")
    final_global_gt, total_global_gt_segments, global_action_gt_by_video = calculate_global_gt(all_batch_meta, num_classes)
    print(f"Global GT calculated. Total unique GT segments: {total_global_gt_segments}")
    for c in range(num_classes):
        print(f"  Class {c} GT count: {len(final_global_gt.get(c, []))}")

    print("\nStep 3: Loading RNN Post-Processor Model...")
    if not os.path.exists(rnn_checkpoint_path):
        print(f"Error: RNN checkpoint not found at {rnn_checkpoint_path}")
        return
        
    try:
        rnn_checkpoint = torch.load(rnn_checkpoint_path, map_location=device)
        rnn_args_loaded = rnn_checkpoint.get('args', None)
        if rnn_args_loaded:
             print("Using RNN hyperparameters from checkpoint.")
             rnn_model_cfg = rnn_args_loaded
        else:
             print("Warning: RNN checkpoint does not contain training arguments. Using config file.")
             rnn_model_cfg = cfg['rnn_training']['model']

        rnn_input_size = 3 * num_classes 
        rnn_num_classes_out = num_classes + 1

        rnn_model = RNNPostProcessor(
            input_size=rnn_input_size,
            hidden_size=rnn_model_cfg['hidden_size'],
            num_layers=rnn_model_cfg['num_layers'],
            num_classes=rnn_num_classes_out,
            rnn_type=rnn_model_cfg['type'],
            dropout_prob=rnn_model_cfg['dropout_prob'], 
            bidirectional=rnn_model_cfg['bidirectional']
        ).to(device)
        
        rnn_model.load_state_dict(rnn_checkpoint['model_state_dict'])
        rnn_model.eval() 
        print(f"Successfully loaded RNN model from epoch {rnn_checkpoint.get('epoch', 'N/A')}")
        
    except Exception as e:
        print(f"Error loading RNN checkpoint: {e}")
        return

    rnn_predictions_by_video, rnn_all_action_preds_flat = _run_rnn_on_all_videos(
        rnn_model, all_raw_preds, all_batch_meta, global_action_gt_by_video, 
        device, num_classes, background_label
    )

    print("\nStep 5, 6, 7: Calculating final metrics...")

    rnn_all_frame_targets_flat = []
    rnn_all_frame_preds_flat_for_metric = []
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
        
        video_targets = np.zeros((video_length, num_classes), dtype=int)
        video_preds = np.zeros((video_length, num_classes), dtype=int)

        if video_id in global_action_gt_by_video:
            for c, segments in global_action_gt_by_video[video_id].items():
                for start, end in set(segments):
                    if end > start and 0 <= c < num_classes: 
                         video_targets[start:min(end, video_length), c] = 1

        if video_id in rnn_predictions_by_video:
             for c, segments in rnn_predictions_by_video[video_id].items():
                  for seg_info in segments:
                       start, end = seg_info['segment']
                       if end > start and 0 <= c < num_classes: 
                            video_preds[start:min(end, video_length), c] = 1
                            
        rnn_all_frame_targets_flat.extend(video_targets.flatten())
        rnn_all_frame_preds_flat_for_metric.extend(video_preds.flatten()) 

    final_metrics = compute_final_metrics(
        global_action_gt_global=final_global_gt, 
        merged_all_action_preds=rnn_all_action_preds_flat, 
        merged_all_frame_targets=rnn_all_frame_targets_flat, 
        merged_all_frame_preds=rnn_all_frame_preds_flat_for_metric, 
        num_classes=num_classes
    )

    print("\n\n--- RNN Post-Processing Evaluation Results ---")
    print(f"RNN mAP (IoU 0.3,0.5,0.7): {final_metrics.get('mAP', 0.0):.4f}")
    print(f"RNN mAP@mid: {final_metrics.get('map_mid', 0.0):.4f}")
    print(f"RNN Global Frame-level -> F1: {final_metrics.get('merged_f1', 0.0):.4f}") 
    # print(f"RNN Frame-wise Accuracy: {final_metrics.get('frame_accuracy', 0.0):.4f}") 
    print(f"RNN Avg F1@0.10: {final_metrics.get('avg_f1_iou_010', 0.0):.4f}")
    print(f"RNN Avg F1@0.25: {final_metrics.get('avg_f1_iou_025', 0.0):.4f}")
    print(f"RNN Avg F1@0.50: {final_metrics.get('avg_f1_iou_050', 0.0):.4f}")
    
    header = "Class | AP@0.5 | Preds | F1@0.1 | F1@0.5"
    print(header)
    print("-" * len(header))
    class_aps = final_metrics.get('class_aps', {})
    for c in range(num_classes):
        ap = class_aps.get(c, 0.0)
        preds_count = len(rnn_all_action_preds_flat.get(c, []))
        print(f"{c:<5} | {ap:.4f} | {preds_count:<5} | N/A    | N/A") 
    print("-" * len(header))
    total_rnn_preds = sum(len(v) for v in rnn_all_action_preds_flat.values())
    print(f"Total RNN Pred Segments: {total_rnn_preds}") 
    print(f"Total GT Segments: {total_global_gt_segments}") 

    vis_cfg = cfg['pipeline_evaluation']['visualization']
    vis_video_id = args.visualize_video_id if args.visualize_video_id else vis_cfg['video_id']
    vis_enabled = args.visualize_video_id is not None or vis_cfg['enabled']
    
    if vis_enabled and vis_video_id:
        print(f"\n--- Preparing for Visualization of Video ID: {vis_video_id} ---")
        frames_template = args.frames_npz_path_template if args.frames_npz_path_template else vis_cfg['frames_npz_template']
        output_video_path = args.output_video_path if args.output_video_path else vis_cfg['output_video_path']
        fps = args.fps if args.fps is not None else vis_cfg['fps']
        
        if not frames_template:
             print("Error: --frames_npz_path_template or config['pipeline_evaluation']['visualization']['frames_npz_template'] is required.")
        elif not output_video_path:
             print("Error: --output_video_path or config['pipeline_evaluation']['visualization']['output_video_path'] is required.")
        else:
            try:
                vis_npz_path = frames_template.format(video_id=vis_video_id)
                formatted_output_video_path = output_video_path.format(video_id=vis_video_id)
                output_dir = os.path.dirname(formatted_output_video_path)
                if output_dir and not os.path.exists(output_dir):
                    os.makedirs(output_dir)
            except Exception as e:
                 print(f"Error formatting visualization paths: {e}")
                 vis_npz_path = None
                 formatted_output_video_path = None

            if vis_npz_path and os.path.exists(vis_npz_path) and formatted_output_video_path:
                visualize_rnn_predictions(
                    video_id=vis_video_id,
                    frames_npz_path=vis_npz_path,
                    output_video_path=formatted_output_video_path,
                    fps=fps,
                    global_gt_data_by_video=global_action_gt_by_video, 
                    rnn_preds_by_video=rnn_predictions_by_video,      
                    num_classes=num_classes
                )
            elif vis_npz_path and not os.path.exists(vis_npz_path):
                 print(f"Error: Visualization frames file not found: {vis_npz_path}")
            elif not formatted_output_video_path:
                 print("Error: Could not determine visualization output path.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Base Model + RNN Post-Processor Pipeline")
    parser.add_argument('--config', default='configs/config.yaml', help='Path to configuration file')
    parser.add_argument("--rnn_checkpoint_path", type=str, default=None, help="Override RNN checkpoint path from config")
    parser.add_argument("--inference_output_path", type=str, default=None, help="Override inference results pkl path from config")
    
    #Visualization
    parser.add_argument("--visualize_video_id", type=str, default=None, help="Optional video ID to visualize (overrides config)")
    parser.add_argument("--frames_npz_path_template", type=str, default=None, help="Override frame NPZ path template from config")
    parser.add_argument("--output_video_path", type=str, default=None, help="Override visualization output video path from config")
    parser.add_argument("--fps", type=int, default=None, help="Override FPS for visualization video from config")

    args = parser.parse_args()

    try:
        with open(args.config, 'r') as f:
            cfg = yaml.safe_load(f)
    except FileNotFoundError:
        print(f"Error: Config file not found at {args.config}")
        exit()
    except Exception as e:
        print(f"Error loading config file: {e}")
        exit()
    main_evaluate(cfg, args) 
