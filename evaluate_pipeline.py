import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import nms
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
import yaml # For configuration loading
import src.utils.helpers as helpers
import src.utils.metrics as metrics
import src.utils.postprocessing as postprocessing
from src.utils import visualization

# Import base model and potentially dataloaders if running inference
try:
    from src.models.TemporalActionDetector_v5_1 import TemporalActionDetector
except ImportError:
    print("Error: Could not import TemporalActionDetector from model_fixed.py.")
    exit()
try:
    from src.dataloader import get_feature_val_loader, get_feature_test_loader
except ImportError:
    print("Error: Could not import get_val_loader from dataloader.py.")
    exit()
try:
    from src.models.rnn_postprocessor import RNNPostProcessor
except ImportError:
    print("Error: Could not import RNNPostProcessor from models/rnn_postprocessor.py")
    exit()

NUM_CLASSES = 5
WINDOW_SIZE = 32
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_MIXED_PRECISION = True
BACKGROUND_LABEL = NUM_CLASSES

DEFAULT_BASE_CHECKPOINT = "trained_models/best_action_detector_v2_3_external.pth"
DEFAULT_RNN_CHECKPOINT = r"rnn_checkpoints\sav\best\V2_1\best_rnn_model_128_2.pth"
DEFAULT_INFERENCE_PKL = "val_inference_raw.pkl"
DEFAULT_TEST_PKL = r"test_inference_raw.pkl"
DEFAUT_FORMAT = "visualization_{video_id}.mp4"
FRAME_TEMPLATE = "Data/full_videos/test/frames/{video_id}_frames.npz"

class CFG:
    class_thr       = [0.5, 0.5, 0.12, 0.2, 0.4]
    boundary_thr    = [0.2, 0.2, 0.08, 0.08, 0.2]
    min_seg_len     = 3
    nms_thr         = 0.4
    iou_merge_thr   = 0.3
    max_gap         = 2
    iou_xclass_thr  = 0.4
    smoothing_window= 3
    num_classes     = 5


def main_evaluate(args):
    print(f"Using device: {DEVICE}")
    print(f"Evaluating base model checkpoint: {args.base_checkpoint_path}")
    print(f"Using RNN post-processor checkpoint: {args.rnn_checkpoint_path}")
    
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
                print(f"Successfully loaded inference results for {len(all_raw_preds)} batches.")
            except Exception as e:
                print(f"Error loading inference results: {e}. Need to run inference.")
                all_raw_preds = None
        else:
             print(f"Error: --skip_inference flag set, but file not found: {args.inference_output_path}")
             return

    if all_raw_preds is None:
        print("\nRunning Base Model Inference...")
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
        
        val_loader = get_feature_val_loader(batch_size=args.batch_size, shuffle=False)
        
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
        if args.inference_output_path:
             print(f"Saving inference results to {args.inference_output_path}...")
             try:
                 output_dir = os.path.dirname(args.inference_output_path)
                 if output_dir and not os.path.exists(output_dir):
                     os.makedirs(output_dir)
                 with open(args.inference_output_path, 'wb') as f:
                     pickle.dump({'all_raw_preds': all_raw_preds, 'all_batch_meta': all_batch_meta}, f)
                 print("Saved inference results.")
             except Exception as e:
                 print(f"Error saving inference results: {e}")

    global_action_gt_by_video = defaultdict(lambda: defaultdict(list))
    all_metadata_flat = [meta for batch_meta in all_batch_meta for meta in batch_meta]
    unique_video_ids_from_meta = set()
    for meta in all_metadata_flat:
        video_id = meta['video_id']
        unique_video_ids_from_meta.add(video_id)
        start_idx = meta['slice_start_idx']
        if isinstance(meta.get('annotations'), list):
            for anno in meta['annotations']:
                if isinstance(anno, dict) and all(k in anno for k in ('action_id', 'start_frame', 'end_frame')):
                    action_id = anno['action_id']
                    global_gt_start = start_idx + anno['start_frame']
                    global_gt_end = start_idx + anno['end_frame']
                    if 0 <= action_id < NUM_CLASSES:
                        global_action_gt_by_video[video_id][action_id].append((global_gt_start, global_gt_end))
                else:
                    print(f"Warning: Invalid annotation format found in metadata for video {video_id}: {anno}")
        elif 'annotations' in meta:
            print(f"Warning: Invalid 'annotations' type in metadata for video {video_id}: {type(meta['annotations'])}")
    for vid, cls_dict in global_action_gt_by_video.items():
        for cls, segs in cls_dict.items():
            unique_segs = sorted(set(segs))
            cls_dict[cls] = helpers.merge_segments(unique_segs)
    
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

    if not os.path.exists(args.rnn_checkpoint_path):
        print(f"Error: RNN checkpoint not found at {args.rnn_checkpoint_path}")
        return
        
    try:
        rnn_checkpoint = torch.load(args.rnn_checkpoint_path, map_location=DEVICE)
        rnn_args = rnn_checkpoint.get('args', None)
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
        print("Hidden size:", rnn_args['hidden_size'])
        print("RNN Type:", rnn_args['rnn_type'])
        print("Number of layers:", rnn_args['num_layers'])
        print(f"Successfully loaded RNN model from epoch {rnn_checkpoint.get('epoch', 'N/A')}")
        print(f"  RNN Val Loss during training: {rnn_checkpoint.get('val_loss', 'N/A'):.4f}")
        
    except Exception as e:
        print(f"Error loading RNN checkpoint: {e}")
        return

    rnn_predictions_by_video = defaultdict(lambda: defaultdict(list))
    rnn_all_action_preds_flat = defaultdict(list)
    
    unique_video_ids_to_process = sorted(list(unique_video_ids_from_meta))
    if not unique_video_ids_to_process:
         print("Warning: No unique video IDs found in metadata. Cannot run RNN post-processing.")
         return

    for video_id in tqdm(unique_video_ids_to_process, desc="RNN Processing Videos"):
        avg_action_probs, avg_start_probs, avg_end_probs, num_frames = helpers.reconstruct_full_video_probs(
            video_id, all_raw_preds, all_batch_meta
        )
        
        if num_frames is None or num_frames <= 0:
            continue

        input_features = np.concatenate([avg_action_probs, avg_start_probs, avg_end_probs], axis=1)
        input_tensor = torch.tensor(input_features, dtype=torch.float32).unsqueeze(0).to(DEVICE)
        
        with torch.no_grad():
            logits = rnn_model(input_tensor)
            probs = torch.softmax(logits.squeeze(0), dim=1)
        predicted_labels = torch.argmax(logits.squeeze(0), dim=1).cpu().numpy()
        video_segments = helpers.labels_to_segments(predicted_labels, ignore_label=BACKGROUND_LABEL)
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

    print("Heuristic Post-Processing")
    cfg = CFG()
    postprocessing.evaluate_and_print(all_raw_preds, all_batch_meta, final_global_gt, cfg, model_name="Heurisitcs Post-Processing")
    
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
            if not gt_c:
                class_prec, class_rec, class_f1 = 0.0, 0.0, 0.0
            else:
                class_prec, class_rec, class_f1 = metrics.calculate_f1_at_iou(gt_c, preds_c, threshold)
            all_class_f1_for_this_iou.append(class_f1)
            current_iou_class_f1s[c] = class_f1
            print(f"    Class {c}: Precision={class_prec:.4f}, Recall={class_rec:.4f}, F1={class_f1:.4f}")
        avg_f1_for_this_iou = np.mean(all_class_f1_for_this_iou) if all_class_f1_for_this_iou else 0.0
        key_name = f'avg_f1_iou_{int(threshold*100):03d}'
        rnn_avg_f1_scores[key_name] = avg_f1_for_this_iou
        rnn_class_f1_scores_by_iou[threshold] = current_iou_class_f1s
        print(f"  Average F1 @ {threshold:.2f}: {avg_f1_for_this_iou:.4f}")

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
                    if end > start and 0 <= c < NUM_CLASSES: 
                         video_targets[start:min(end, video_length), c] = 1

        if video_id in rnn_predictions_by_video:
             for c, segments in rnn_predictions_by_video[video_id].items():
                  for seg_info in segments:
                       start, end = seg_info['segment']
                       if end > start and 0 <= c < NUM_CLASSES: 
                            video_preds[start:min(end, video_length), c] = 1
                            
        rnn_all_frame_targets_flat.extend(video_targets.flatten())
        rnn_all_frame_preds_flat.extend(video_preds.flatten()) 

    mAP_avg = metrics.calculate_mAP(
        all_action_gt=final_global_gt, 
        all_action_preds=rnn_all_action_preds_flat, 
        num_classes=NUM_CLASSES, 
        iou_thresholds=[0.3, 0.5, 0.7]
    )
    
    class_aps_at_05 = {}
    for c in range(NUM_CLASSES):
        class_aps_at_05[c] = metrics.calculate_class_mAP(
            gt_segments=final_global_gt.get(c, []), 
            pred_segments=rnn_all_action_preds_flat.get(c, []),
            iou_threshold=0.5
        )
        
    map_mid = metrics.calculate_map_mid(
        all_action_gt=final_global_gt, 
        all_action_preds=rnn_all_action_preds_flat, 
        num_classes=NUM_CLASSES
    )

    iou_thresholds_f1 = [0.1, 0.25, 0.5]

    class_f1_scores       = {
        f"f1_iou_{iou:.2f}".replace('.', ''): {} 
        for iou in iou_thresholds_f1
    }
    avg_f1_scores         = {}
    avg_precision_scores = {}
    avg_recall_scores    = {}
    class_precision_scores = {
        f"prec_iou_{iou:.2f}".replace('.', ''): {} 
        for iou in iou_thresholds_f1
    }
    class_recall_scores = {
        f"rec_iou_{iou:.2f}".replace('.', ''): {} 
        for iou in iou_thresholds_f1
    }

    for iou in iou_thresholds_f1:
        all_class_f1        = []
        all_class_precision = []
        all_class_recall    = []

        avg_key = f"avg_f1_iou_{iou:.2f}".replace('.', '')
        prec_key = f"avg_prec_iou_{iou:.2f}".replace('.', '')
        rec_key  = f"avg_rec_iou_{iou:.2f}".replace('.', '')
        class_f1_key = f"f1_iou_{iou:.2f}".replace('.', '')
        class_prec_key = f"prec_iou_{iou:.2f}".replace('.', '')
        class_rec_key  = f"rec_iou_{iou:.2f}".replace('.', '')

        for c in range(NUM_CLASSES):
            gts = final_global_gt.get(c, [])
            if len(gts) == 0:
                class_f1_scores[class_f1_key][c] = 0.0
                class_precision_scores[class_prec_key][c] = 0.0
                class_recall_scores[class_rec_key][c]    = 0.0
                continue

            preds_c = rnn_all_action_preds_flat.get(c, [])
            p, r, f1 = metrics.calculate_f1_at_iou(gts, preds_c, iou)

            class_precision_scores[class_prec_key][c] = p
            class_recall_scores[class_rec_key][c] = r
            class_f1_scores[class_f1_key][c] = f1

            all_class_precision.append(p)
            all_class_recall.append(r)
            all_class_f1.append(f1)

        avg_precision_scores[prec_key] = np.mean(all_class_precision) if all_class_precision else 0.0
        avg_recall_scores[rec_key]  = np.mean(all_class_recall)if all_class_recall else 0.0
        avg_f1_scores[avg_key]  = np.mean(all_class_f1) if all_class_f1 else 0.0

    merged_precision, merged_recall, merged_f1, _ = precision_recall_fscore_support(
        rnn_all_frame_targets_flat,
        rnn_all_frame_preds_flat,
        average='macro',
        zero_division=0
    )
    rnn_frame_accuracy = (
        np.mean(np.array(rnn_all_frame_targets_flat) == np.array(rnn_all_frame_preds_flat))
        if rnn_all_frame_targets_flat else 0.0
    )
    final_metrics = {
        'mAP':               mAP_avg,
        'class_aps_05':      class_aps_at_05,
        'map_mid':           map_mid,
        'merged_precision':  merged_precision,
        'merged_recall':     merged_recall,
        'merged_f1':         merged_f1,
        **avg_precision_scores,
        **avg_recall_scores,
        **avg_f1_scores,
    }
    print("\n\nRNN Post-Processing Evaluation Results")
    print(f"RNN mAP: {final_metrics['mAP']:.4f}")
    print(f"RNN mAP@mid:{final_metrics['map_mid']:.4f}\n")

    print(f"Global Frame-level Precision (macro): {final_metrics['merged_precision']:.4f}")
    print(f"Global Frame-level Recall (macro): {final_metrics['merged_recall']:.4f}")
    print(f"Global Frame-level F1 (macro): {final_metrics['merged_f1']:.4f}")
    print(f"RNN Frame-wise Accuracy: {rnn_frame_accuracy:.4f}\n")

    for iou in iou_thresholds_f1:
        key_p  = f"avg_prec_iou_{iou:.2f}".replace('.', '')
        key_r  = f"avg_rec_iou_{iou:.2f}".replace('.', '')
        key_f1 = f"avg_f1_iou_{iou:.2f}".replace('.', '')
        print(f"IoU={iou:.2f}: Prec {final_metrics[key_p]:.4f} | "
            f"Rec {final_metrics[key_r]:.4f} | "
            f"F1 {final_metrics[key_f1]:.4f}")
        header = "Class | AP@0.5 | Preds | F1@0.1 | F1@0.5"
    print(header)
    print("-" * len(header))

    class_aps_print = final_metrics.get('class_aps_05', {})
    class_f1_01 = class_f1_scores.get('f1_iou_010', {})
    class_f1_05 = class_f1_scores.get('f1_iou_050', {})

    for c in range(NUM_CLASSES):
        ap = class_aps_print.get(c, 0.0)
        preds_count = len(rnn_all_action_preds_flat.get(c, []))
        f1_01 = class_f1_01.get(c, 0.0)
        f1_05 = class_f1_05.get(c, 0.0)
        print(f"{c:<5} | {ap:.4f} | {preds_count:<5} | {f1_01:.4f} | {f1_05:.4f}")

    total_rnn_preds = sum(len(v) for v in rnn_all_action_preds_flat.values())
    total_global_gt_segments = sum(len(v) for v in final_global_gt.values())
    print(f"Total RNN Pred Segments: {total_rnn_preds}") 
    print(f"Total GT Segments: {total_global_gt_segments}")

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
                output_video_path = args.output_video_path.format(video_id=args.visualize_video_id)
                visualization.visualize_rnn_predictions(
                    video_id=args.visualize_video_id,
                    frames_npz_path=vis_npz_path,
                    output_video_path=output_video_path,
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
    try:
        with open('configs/config.yaml', 'r') as f:
            cfg = yaml.safe_load(f)
    except FileNotFoundError:
        print(f"Config file not found at configs/config.yaml")
        exit()
    except Exception as e:
        print(f"loading config file: {e}")

    cfg_eval = cfg.get('pipeline_evaluation', {})

    parser = argparse.ArgumentParser(description="Evaluate Base Model + RNN Post-Processor")
    parser.add_argument("--base_checkpoint_path", type=str, default=DEFAULT_BASE_CHECKPOINT, help="Path to the base model checkpoint (.pth)")
    parser.add_argument("--rnn_checkpoint_path", type=str, default=cfg_eval.get('rnn_checkpoint_to_use', 'rnn_checkpoints/best_rnn_model.pth'), help="Path to the trained RNN post-processor checkpoint (.pth)")
    parser.add_argument("--inference_output_path", type=str, default=cfg_eval.get('inference_results_pkl', 'test_inference_raw.pkl '), help="Path to load/save base model inference results (.pkl) for the validation set")
    parser.add_argument("--skip_inference", action="store_true", help="Skip base model inference and load results from --inference_output_path")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for base model inference if run")
    parser.add_argument("--num_workers", type=int, default=0, help="Number of workers for dataloader")
    # Add visualization args if adapting visualization later
    parser.add_argument("--visualize_video_id", type=str, default=None, help="Optional video ID to visualize")
    parser.add_argument("--frames_npz_path_template", type=str, default=cfg_eval.get('frames_npz_template', 'data/full_videos/test/frames/{video_id}_frames.npz'), help="Path template for frame NPZ files")
    parser.add_argument("--output_video_path", type=str, default=cfg_eval.get('output_video_path', 'logs/visualization_{video_id}.mp4'), help="Path to save visualization video")
    parser.add_argument("--fps", type=int, default=15, help="FPS for visualization video")

    args = parser.parse_args()
    main_evaluate(args) 