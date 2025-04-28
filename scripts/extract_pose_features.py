import os
import numpy as np
import mediapipe as mp
import argparse
from tqdm import tqdm
import glob
import yaml
from pathlib import Path
from src.utils.feature_extraction import extract_pose_features

mp_pose = mp.solutions.pose

def process_video(video_id, frames_dir, pose_dir, pose_config):
    npz_path = Path(frames_dir) / f"{video_id}_frames.npz"
    
    if not npz_path.exists():
        print(f"Frames file not found: {npz_path}")
        return False
    
    try:
        frames_data = np.load(npz_path)
        frames = frames_data['frames']
    except Exception as e:
        print(f"loading frames from {npz_path}: {e}")
        return False
        
    num_frames = frames.shape[0]
       
    with mp_pose.Pose(
        static_image_mode=False, 
        model_complexity=pose_config['model_complexity'],
        min_detection_confidence=pose_config['min_detection_confidence']) as pose_detector:
        
        pose_features = np.zeros((num_frames, 99))  #33 landmarks × 3 coordinates
        
        for i in tqdm(range(num_frames), desc=f"Processing {video_id}"):
            frame = frames[i]
            pose_features[i] = extract_pose_features(frame, pose_detector)
            
    
    pose_output_path = Path(pose_dir) / f"{video_id}_pose.npz"
    try:
        pose_output_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(pose_output_path, pose=pose_features)
    except Exception as e:
        print(f"saving pose features to {pose_output_path}: {e}")
        return False

    return True

def process_dataset(split_name, frames_dir, pose_dir, pose_config):
    print(f"Processing {split_name} data...")
    print(f"  Frames dir: {frames_dir}")
    print(f"  Pose output dir: {pose_dir}")
    
    if not Path(frames_dir).exists():
        print(f"Frames directory not found: {frames_dir} Skipping split")
        return 0, 0
        
    frame_files = glob.glob(str(Path(frames_dir) / "*_frames.npz"))
    
    if not frame_files:
        print(f"No *_frames.npz files found in {frames_dir} Skipping split")
        return 0, 0
        
    success_count = 0
    error_count = 0
    skipped_count = 0
    
    Path(pose_dir).mkdir(parents=True, exist_ok=True)
    
    for frame_file in frame_files:
        video_id = Path(frame_file).stem.replace("_frames", "")
        
        pose_output_path = Path(pose_dir) / f"{video_id}_pose.npz"

        if pose_output_path.exists():
            skipped_count += 1
            success_count += 1
            continue
            
        try:
            result = process_video(video_id, frames_dir, pose_dir, pose_config)
            if result:
                success_count += 1
            else:
                error_count += 1
        except Exception as e:
            print(f"processing {video_id}: {str(e)}")
            error_count += 1
            
    print(f"{split_name} data: {success_count} processed ({skipped_count} skipped), {error_count} errors")
    return success_count, error_count

def main():
    parser = argparse.ArgumentParser(description="Extract pose features from videos")
    parser.add_argument('--config', default='configs/config.yaml', help='Path to configuration file')
    parser.add_argument(
        "--split", 
        type=str, 
        choices=["train", "val", "test", "all"],
        default="all", 
        help="Dataset split to process (overrides config potentially, used for selection)"
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
    feature_cfg = cfg['feature_extraction']
    pose_cfg = feature_cfg['pose']
    
    base_processed_dir = Path(data_cfg['processed_dir'])

    splits_to_process = []
    if args.split == "train" or args.split == "all":
        splits_to_process.append("train")
    if args.split == "val" or args.split == "all":
        splits_to_process.append("val")
    if args.split == "test" or args.split == "all":
        splits_to_process.append("test")
        
    total_success = 0
    total_errors = 0
    
    for split in splits_to_process:
        frames_dir = base_processed_dir / split / "frames"
        pose_dir = base_processed_dir / split / "pose"
        
        success, errors = process_dataset(split, frames_dir, pose_dir, pose_cfg)
        total_success += success
        total_errors += errors
    
    print(f"Feature extraction complete Total processed: {total_success}, Total errors: {total_errors}")

if __name__ == "__main__":
    main() 