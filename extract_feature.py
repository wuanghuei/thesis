import os
import torch
import numpy as np
from pathlib import Path
import json
import argparse
from tqdm import tqdm
import zipfile
import time
import pandas as pd

# Import feature extractor
from src.models.feature_extractor.mvit_32 import MViT32FeatureExtractor

def extract_features_for_video(video_id, frames_dir, anno_dir, feature_extractor, device, window_size=32, stride=8):
    """
    Extract features for a single video by loading frames directly
    
    Parameters:
    -----------
    video_id : str
        ID of the video to process
    frames_dir : Path
        Directory containing frame files
    anno_dir : Path
        Directory containing annotation files
    feature_extractor : nn.Module
        Feature extraction model
    device : torch.device
        Device to run model on
    window_size : int
        Size of sliding window
    stride : int
        Stride for sliding window
        
    Returns:
    --------
    dict
        Dictionary with features and metadata
    """
    frames_path = frames_dir / f"{video_id}_frames.npz"
    anno_path = anno_dir / f"{video_id}_annotations.json"
    
    # Check if files exist
    if not frames_path.exists() or not anno_path.exists():
        print(f"Missing files for {video_id}. Skipping.")
        return None
    
    # Load annotations
    try:
        with open(anno_path, 'r') as f:
            annotations = json.load(f)
    except Exception as e:
        print(f"Error loading annotations for {video_id}: {e}")
        return None
    
    # Load frames
    try:
        npz_data = np.load(frames_path)
        all_frames = npz_data['frames']
    except (zipfile.BadZipFile, KeyError, OSError, ValueError) as e:
        print(f"Error loading frames for {video_id}: {e}")
        return None
    
    num_frames = len(all_frames)
    
    # Prepare sliding windows
    windows = []
    window_indices = []
    
    # If video is shorter than window size, use the whole video
    if num_frames <= window_size:
        # Pad with copies of the last frame if needed
        frames = all_frames
        if len(frames) < window_size:
            padding = np.repeat(frames[-1:], window_size - len(frames), axis=0)
            frames = np.concatenate([frames, padding], axis=0)
        windows.append(frames)
        window_indices.append((0, min(window_size, num_frames)))
    else:
        # Create sliding windows
        for start in range(0, num_frames - window_size + 1, stride):
            end = start + window_size
            window_frames = all_frames[start:end]
            windows.append(window_frames)
            window_indices.append((start, end))
        
        # Add final window if needed
        if (num_frames - window_size) % stride != 0:
            start = num_frames - window_size
            end = num_frames
            window_frames = all_frames[start:end]
            windows.append(window_frames)
            window_indices.append((start, end))
    
    # Extract features for each window
    features = []
    with torch.no_grad():
        for window in tqdm(windows, desc=f"Extracting features for {video_id}", leave=False):
            # Convert to tensor and normalize
            frames = torch.from_numpy(window).float() / 255.0
            frames = frames.permute(3, 0, 1, 2)  # NHWC -> CTHW
            frames = frames.unsqueeze(0)  # Add batch dimension
            frames = frames.to(device)
            
            # Extract features
            feature = feature_extractor.extract_features(frames)
            features.append(feature.cpu().numpy())
    
    # Stack features
    features = np.vstack(features)
    window_indices = np.array(window_indices)
    

    # Return dictionary with features and metadata
    return {
        "video_id": video_id,
        "clip_features": features,
        "window_indices": window_indices,
        "metadata": annotations
    }

def extract_features_for_split(split, base_dir, output_dir, device=None):
    """
    Extract features for all videos in a split
    
    Parameters:
    -----------
    split : str
        Dataset split ('train', 'val', 'test')
    base_dir : str or Path
        Base directory containing dataset
    output_dir : str or Path
        Output directory for features
    device : str or None
        Device to run model on
    """
    base_dir = Path(base_dir)
    output_dir = Path(output_dir)
    
    # Setup directories
    frames_dir = base_dir / split / "frames"
    anno_dir = base_dir / split / "annotations"
    out_split_dir = output_dir / split
    out_split_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize device
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)
    
    # Initialize feature extractor
    feature_extractor = MViT32FeatureExtractor(device=device)
    feature_extractor.eval()
    print(f"Using device: {device}")
    
    # Get all video IDs
    video_ids = []
    for frame_file in frames_dir.glob("*_frames.npz"):
        video_id = frame_file.stem.replace("_frames", "")
        anno_path = anno_dir / f"{video_id}_annotations.json"
        if anno_path.exists():
            video_ids.append(video_id)
    
    print(f"Found {len(video_ids)} videos in {split} split")
    
    # Create overall metadata
    metadata = {
        "split": split,
        "feature_dim": feature_extractor.feature_dimension,
        "extraction_date": pd.Timestamp.now().isoformat(),
        "num_videos": len(video_ids)
    }
    
    with open(out_split_dir / "metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)
    
    # Process each video
    successful_videos = 0
    failed_videos = 0
    
    for video_id in tqdm(video_ids, desc=f"Processing {split} videos"):
        # Extract features
        result = extract_features_for_video(
            video_id, frames_dir, anno_dir, feature_extractor, device
        )
        
        if result is None:
            failed_videos += 1
            continue
        
        # Save features
        output_file = out_split_dir / f"{video_id}_features.npz"
        try:
            # Convert metadata to JSON string
            metadata_json = json.dumps(result["metadata"])
            
            # Save to NPZ file
            np.savez_compressed(
                output_file,
                clip_features=result["clip_features"],
                window_indices=result["window_indices"],
                metadata=metadata_json
            )
            successful_videos += 1
        except Exception as e:
            print(f"Error saving features for {video_id}: {e}")
            failed_videos += 1
    
    print(f"Processed {successful_videos}/{len(video_ids)} videos successfully")
    print(f"Failed to process {failed_videos} videos")
    
    return out_split_dir

def main():
    parser = argparse.ArgumentParser(description="Extract features from videos directly")
    
    parser.add_argument("--base_dir", type=str, default="Data/full_videos",
                       help="Base directory containing dataset")
    parser.add_argument("--output_dir", type=str, default="features",
                       help="Output directory for features")
    parser.add_argument("--split", type=str, choices=["train", "val", "test", "all"], default="all",
                       help="Which data split to process")
    parser.add_argument("--device", type=str, default=None, choices=["cuda", "cpu"],
                       help="Device to run model on (cuda or cpu)")
    
    args = parser.parse_args()
    
    # Process specified splits
    splits_to_process = ["train", "val", "test"] if args.split == "all" else [args.split]
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Track split directories
    split_dirs = {}
    
    # Process each split
    for split in splits_to_process:
        print(f"\nProcessing {split} split...")
        split_dir = extract_features_for_split(split, args.base_dir, output_dir, args.device)
        split_dirs[split] = str(split_dir)
    
    # Save info about splits
    with open(output_dir / "splits_info.json", 'w') as f:
        json.dump(split_dirs, f, indent=2)
    
    print("\nFeature extraction completed")

if __name__ == "__main__":
    main() 