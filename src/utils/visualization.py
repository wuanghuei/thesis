import os
import cv2
import numpy as np
from tqdm import tqdm
from collections import defaultdict
import src.utils.metrics as metrics

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

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
    if not out.isOpened():
        print(f"Error: Could not open video writer for {output_video_path}")
        return

    colors = [
        (255, 0, 0),
        (0, 255, 0),
        (0, 0, 255),
        (255, 255, 0),
        (0, 255, 255),
        (255, 0, 255),
    ]
    gt_color = (200, 200, 200)
    pred_color_base = (180, 105, 255)

    bar_height = 25
    gt_y_pos = 10
    pred_y_pos = gt_y_pos + bar_height + 5
    text_y_offset = 18
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    font_thickness = 1

    rnn_preds_this_video = rnn_preds_by_video.get(video_id, defaultdict(list))

    for frame_idx in tqdm(range(num_frames), desc="Processing Frames"): 
        frame = frames[frame_idx].copy()

        gt_labels_on_frame = []
        for action_id, gt_segments in global_gt_data_by_video.get(video_id, {}).items():
            for start, end in gt_segments:
                if start <= frame_idx < end:
                    label_text = f"GT Cls {action_id}"
                    gt_labels_on_frame.append((label_text, colors[action_id % len(colors)]))
                    cv2.rectangle(frame, (0, gt_y_pos), (width-1, gt_y_pos + bar_height), gt_color, -1)
                    cv2.rectangle(frame, (0, gt_y_pos), (width-1, gt_y_pos + bar_height), colors[action_id % len(colors)], 2)
                    cv2.putText(frame, label_text, (5, gt_y_pos + text_y_offset), font, font_scale, (0,0,0), font_thickness, cv2.LINE_AA)
                    break
            if gt_labels_on_frame:
                 pass

        pred_labels_on_frame = []
        for action_id, pred_segments in rnn_preds_this_video.items():
            for segment_info in pred_segments:
                start, end = segment_info['segment']
                score = segment_info.get('score', 0.0)
                if start <= frame_idx < end:
                    label_text = f"Pred Cls {action_id} ({score:.2f})"
                    pred_labels_on_frame.append((label_text, colors[action_id % len(colors)]))
                    cv2.rectangle(frame, (0, pred_y_pos), (width-1, pred_y_pos + bar_height), pred_color_base, -1)
                    cv2.rectangle(frame, (0, pred_y_pos), (width-1, pred_y_pos + bar_height), colors[action_id % len(colors)], 2)
                    cv2.putText(frame, label_text, (5, pred_y_pos + text_y_offset), font, font_scale, (0,0,0), font_thickness, cv2.LINE_AA)
                    break
            if pred_labels_on_frame:
                 pass

        frame_text = f"Frame: {frame_idx}/{num_frames}"
        cv2.putText(frame, frame_text, (width - 150, height - 15), font, font_scale, (255, 255, 255), font_thickness, cv2.LINE_AA)

        out.write(frame)

    out.release()
    print(f"--- Visualization finished for Video: {video_id} ---")
    print(f"  Output saved to: {output_video_path}")


    gt_segments_all = []
    for segs in global_gt_data_by_video.get(video_id, {}).values():
        gt_segments_all.extend(segs)
    pred_segments_all = []
    for pred_list in rnn_preds_this_video.values():
        pred_segments_all.extend(pred_list)

    for thr in [0.1, 0.25, 0.5]:
        precision, recall, f1 = metrics.calculate_f1_at_iou(gt_segments_all, pred_segments_all, thr)
        print(f"Precision, Recall, F1 at {int(thr*100)}% IOU: {precision:.3f}, {recall:.3f}, {f1:.3f}")
    print("\nAll GT segments:")
    for idx, (start, end) in enumerate(gt_segments_all, 1):
        print(f"{idx}: ({start}, {end})")

    print("\nAll Pred segments:")
    for idx, pred in enumerate(pred_segments_all, 1):
        s, e = pred['segment']
        score = pred.get('score', 0.0)
        print(f"{idx}: ({s}, {e})  score={score:.2f}")

    tp_segments = []
    fp_segments = []
    gt_matched = [False] * len(gt_segments_all)
    for pred in pred_segments_all:
        best_iou = 0
        best_i = -1
        for i, gt in enumerate(gt_segments_all):
            if not gt_matched[i]:
                iou = metrics.calculate_temporal_iou(pred['segment'], gt)
                if iou > best_iou:
                    best_iou = iou
                    best_i = i
        if best_iou >= 0.5:
            tp_segments.append(pred['segment'])
            gt_matched[best_i] = True
        else:
            fp_segments.append(pred['segment'])

    print("\nTrue Positive (TP) segments (>=50% IOU):")
    for seg in tp_segments:
        print(seg)
    print("\nFalse Positive (FP) segments (<50% IOU):")
    for seg in fp_segments:
        print(seg)

    for cls, segs in global_gt_data_by_video[video_id].items():
        for idx, (s,e) in enumerate(segs,1):
            print(f"Action {cls} – #{idx}: ({s},{e})")
