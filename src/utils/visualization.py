import os
import cv2
import numpy as np
from tqdm import tqdm
from collections import defaultdict

def visualize_rnn_predictions(video_id, frames_npz_path, output_video_path, fps, 
                               global_gt_data_by_video, rnn_preds_by_video, 
                               num_classes):
    print(f"Load frames from: {frames_npz_path}")
    print(f"Save output to: {output_video_path}")

    if not os.path.exists(frames_npz_path):
        print(f"frames file not found at {frames_npz_path}")
        return

    try:
        frames_data = np.load(frames_npz_path)
        frames = frames_data['frames']

    except Exception as e:
        print(f"Error loading frames from {frames_npz_path}: {e}")
        return

    if frames.ndim != 4 or frames.shape[3] != 3:
         print(f"Error: Unexpected frame shape {frames.shape}. Expected (T, H, W, 3).")
         return

    num_frames, height, width, _ = frames.shape
    print(f"  Loaded {num_frames} frames ({height}x{width})")

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

    bar_height = 25
    gt_y_pos = 10
    text_y_offset = 18
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    font_thickness = 1

    for frame_idx in tqdm(range(num_frames), desc="Processing Frames"): 
        frame = frames[frame_idx].copy() 

        gt_labels_on_frame = []
        for action_id, gt_segments in global_gt_data_by_video.get(video_id, {}).items():
            for start, end in gt_segments:
                if start <= frame_idx < end:
                    label_text = f"GT Cls {action_id}"
                    gt_labels_on_frame.append((label_text, colors[action_id % len(colors)]))
                    cv2.rectangle(frame, (0, gt_y_pos), (width-1, gt_y_pos + bar_height), gt_color, -1)
                    cv2.rectangle(frame, (0, gt_y_pos), (width-1, gt_y_pos + bar_height), colors[action_id % len(colors)])
                    cv2.putText(frame, label_text, (5, gt_y_pos + text_y_offset), font, font_scale, (0,0,0), font_thickness, cv2.LINE_AA)
                    break 

        pred_labels_on_frame = []
        for action_id, pred_segments in rnn_preds_this_video.items():
            for segment_info in pred_segments:
                start, end = segment_info['segment']
                score = segment_info.get('score', 0.0)
                if start <= frame_idx < end:
                    label_text = f"Pred Cls {action_id} ({score:.2f})"
                    pred_labels_on_frame.append((label_text, colors[action_id % len(colors)]))
                    cv2.rectangle(frame, (0, pred_y_pos), (width-1, pred_y_pos + bar_height), pred_color_base, -1) # Background bar
                    cv2.rectangle(frame, (0, pred_y_pos), (width-1, pred_y_pos + bar_height), colors[action_id % len(colors)], 2) # Border
                    cv2.putText(frame, label_text, (5, pred_y_pos + text_y_offset), font, font_scale, (0,0,0), font_thickness, cv2.LINE_AA)
                    break 

        frame_text = f"Frame: {frame_idx}/{num_frames}"
        cv2.putText(frame, frame_text, (width - 150, height - 15), font, font_scale, (255, 255, 255), font_thickness, cv2.LINE_AA)
        out.write(frame)