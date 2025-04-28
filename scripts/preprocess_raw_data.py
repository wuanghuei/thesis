import os
import cv2
import numpy as np
from scipy.io import loadmat
import json
import argparse
import yaml
from pathlib import Path
from tqdm import tqdm
from src.utils.helpers import find_nearest_subsampled_idx

def prepare_full_video(video_path, label_path, output_dir_split, frame_size, subsample_factor):
    try:
        tlabs = loadmat(label_path)["tlabs"].ravel()
    except FileNotFoundError:
        print(f"Label file not found: {label_path}")
        return 0, 0, False
    except Exception as e:
        print(f"loading label file {label_path}: {e}")
        return 0, 0, False
        
    video_id = Path(video_path).stem.replace("_crop", "")

    video_folder = output_dir_split / "frames"
    annotation_folder = output_dir_split / "annotations"
    video_folder.mkdir(parents=True, exist_ok=True)
    annotation_folder.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"Cannot open video file: {video_path}")
        return 0, 0, False
        
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(cap.get(cv2.CAP_PROP_FPS))

    frames = []
    frame_indices = []
    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if idx % subsample_factor == 0:
            try:
                 resized_frame = cv2.resize(frame, (frame_size, frame_size))
                 frames.append(resized_frame)
                 frame_indices.append(int(idx))
            except Exception as e:
                 print(f" resizing frame {idx} for video {video_id}: {e}")
                 pass 

        idx += 1
    cap.release()
    
    if not frames:
         print(f"No frames extracted for video {video_id} Skip")
         return 0, 0, False

    try:
        video_array = np.stack(frames)
    except ValueError as e:
         print(f" stacking frames for {video_id} (likely inconsistent shapes): {e}")
         return 0, 0, False

    npz_path = video_folder / f"{video_id}_frames.npz"
    try:
        np.savez_compressed(npz_path, frames=video_array)
    except Exception as e:
        print(f" saving frames NPZ for {video_id}: {e}")
        return 0, 0, False

    action_annotations = []
    try:
        for action_idx, segments in enumerate(tlabs, start=1):
            if not isinstance(segments, (np.ndarray, list)) or len(segments) == 0:
                 continue
                 
            for segment_pair in segments:
                 if not isinstance(segment_pair, (np.ndarray, list)) or len(segment_pair) != 2:
                      continue
                      
                 start, end = segment_pair
                 start_frame = int(start)
                 end_frame = int(end)

                 if not frame_indices:
                     print(f"frame_indices empty for {video_id} Cannot map segments")
                     break
                     
                 sub_start = find_nearest_subsampled_idx(start_frame, frame_indices)
                 sub_end = find_nearest_subsampled_idx(end_frame, frame_indices)

                 if sub_end > sub_start:
                    action_annotations.append(
                        {
                            "action_id": int(action_idx - 1),
                            "action_name": f"action{action_idx}",
                            "start_frame": int(sub_start),
                            "end_frame": int(sub_end),
                            "start_time": float(start_frame / max(fps, 1e-6)),
                            "end_time": float(end_frame / max(fps, 1e-6)),
                            "original_start": int(start_frame),
                            "original_end": int(end_frame),
                        }
                    )
            if not frame_indices: break

    except Exception as e:
         print(f"processing segments for {video_id}: {e}")

    annotation_data = {
        "video_id": video_id,
        "num_frames": int(len(frames)),
        "original_frames": int(total_frames),
        "fps": float(fps),
        "subsample_factor": int(subsample_factor),
        "frame_indices": [int(idx) for idx in frame_indices],
        "annotations": action_annotations,
        "frames_file": f"{video_id}_frames.npz",
    }
    
    annotation_path = annotation_folder / f"{video_id}_annotations.json"
    try:
        with open(annotation_path, "w") as f:
            json.dump(annotation_data, f, indent=2)
    except Exception as e:
        print(f"saving annotation JSON for {video_id}: {e}")
        return len(frames), len(action_annotations), False

    return len(frames), len(action_annotations), True

def process_split(split_name, video_dir, label_dir, output_dir_split, frame_size, subsample_factor):
    print(f"\nProcessing {split_name} split")
    print(f"Video dir: {video_dir}")
    print(f"Label dir: {label_dir}")
    print(f"Output dir: {output_dir_split}")
    
    if not video_dir.exists() or not label_dir.exists():
        print(f"Input directories for {split_name} not found Skipping split")
        return 0, 0, 0, 0

    output_dir_split.mkdir(parents=True, exist_ok=True)
    
    total_videos = 0
    total_frames = 0
    total_actions = 0
    error_count = 0
    processed_videos = 0

    label_files = sorted(list(label_dir.glob("*.mat")))
    
    if not label_files:
        print(f"No mat files found in {label_dir} Skip")
        return 0, 0, 0, 0
        
    for label_path in tqdm(label_files, desc=f"Processing {split_name}"):
        mat_file = label_path.name
        base_name = label_path.stem
        if "_label" in base_name:
            base_name = base_name.replace("_label", "")

        video_file = f"{base_name}_crop.mp4"
        video_path = video_dir / video_file

        if not video_path.exists():
            print(f"Missing video file {video_file} for label {mat_file} Skip")
            error_count += 1
            continue
        
        total_videos += 1
        
        num_frames, num_actions, success = prepare_full_video(
            video_path, label_path, output_dir_split, frame_size, subsample_factor
        )
        
        if success:
             processed_videos += 1
             total_frames += num_frames
             total_actions += num_actions
        else:
             error_count += 1
             
    print(f"{split_name} split processing complete")
    print(f"Processed videos: {processed_videos}/{total_videos}")
    print(f"Errors encountered: {error_count}")
    
    dataset_stats = {
        "split": split_name,
        "processed_videos": int(processed_videos),
        "total_videos_in_split" : int(total_videos),
        "total_frames_processed": int(total_frames),
        "total_actions_found": int(total_actions),
        "frame_size": int(frame_size),
        "subsample_factor": int(subsample_factor),
        "file_format": "npz_compressed"
    }

    stats_path = output_dir_split / "dataset_stats.json"
    try:
        with open(stats_path, "w") as f:
            json.dump(dataset_stats, f, indent=2)
        print(f"Saved dataset stats for {split_name} to {stats_path}")
    except Exception as e:
        print(f"saving dataset stats for {split_name}: {e}")
        
    return processed_videos, total_frames, total_actions, error_count

def main():
    parser = argparse.ArgumentParser(description="Preprocess MERL Shopping dataset videos.")
    parser.add_argument('--config', default='configs/config.yaml', help='Path to configuration file')
    parser.add_argument(
        "--split", 
        type=str, 
        choices=["train", "val", "test", "all"],
        default="all", 
        help="Dataset split(s) to process."
    )
    args = parser.parse_args()

    try:
        with open(args.config, 'r') as f:
            cfg = yaml.safe_load(f)
    except FileNotFoundError:
        print(f"Config file not found at {args.config}")
        return
    except Exception as e:
        print(f"loading config file: {e}")
        return

    data_cfg = cfg['data']
    prep_cfg = cfg['preprocessing']
    frame_size = prep_cfg['frame_size']
    subsample_factor = prep_cfg['subsample_factor']
    
    base_raw_video_dir = Path(data_cfg['raw_video_dir'])
    base_raw_label_dir = Path(data_cfg['raw_label_dir'])
    base_processed_dir = Path(data_cfg['processed_dir'])
    
    splits_to_process = []
    if args.split == "train" or args.split == "all":
        splits_to_process.append("train")
    if args.split == "val" or args.split == "all":
        splits_to_process.append("val")
    if args.split == "test" or args.split == "all":
        splits_to_process.append("test")
        
    if not splits_to_process:
        print("No valid split selected")
        return
        
    grand_total_videos = 0
    grand_total_frames = 0
    grand_total_actions = 0
    grand_total_errors = 0

    for split in splits_to_process:
        video_dir = base_raw_video_dir / split
        label_dir = base_raw_label_dir / split
        output_dir_split = base_processed_dir / split
        
        processed, frames, actions, errors = process_split(
            split, video_dir, label_dir, output_dir_split, frame_size, subsample_factor
        )
        grand_total_videos += processed
        grand_total_frames += frames
        grand_total_actions += actions
        grand_total_errors += errors

    print("\nOverall Preprocessing Summary")
    print(f"Total videos processed across all selected splits: {grand_total_videos}")
    print(f"Total frames generated: {grand_total_frames}")
    print(f"Total action segments found: {grand_total_actions}")
    print(f"Total errors encountered: {grand_total_errors}")

if __name__ == "__main__":
    main()
