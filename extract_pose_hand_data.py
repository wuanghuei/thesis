import os
import cv2
import numpy as np
import mediapipe as mp
import argparse
from tqdm import tqdm
import json
import glob
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image

# MediaPipe solutions
mp_pose = mp.solutions.pose
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

# ====== Configuration ======
BASE_DIR = "Data"
TRAIN_FRAMES_DIR = os.path.join(BASE_DIR, "full_videos", "train", "frames")
TEST_FRAMES_DIR = os.path.join(BASE_DIR, "full_videos", "test", "frames")
TRAIN_ANNO_DIR = os.path.join(BASE_DIR, "full_videos", "train", "annotations")
TEST_ANNO_DIR = os.path.join(BASE_DIR, "full_videos", "test", "annotations")

# Output directories
TRAIN_POSE_DIR = os.path.join(BASE_DIR, "full_videos", "train", "pose")
TEST_POSE_DIR = os.path.join(BASE_DIR, "full_videos", "test", "pose")
TRAIN_HAND_DIR = os.path.join(BASE_DIR, "full_videos", "train", "hand")
TEST_HAND_DIR = os.path.join(BASE_DIR, "full_videos", "test", "hand")

# Models directory
MODELS_DIR = "models"
os.makedirs(MODELS_DIR, exist_ok=True)

# Create output directories if they don't exist
os.makedirs(TRAIN_POSE_DIR, exist_ok=True)
os.makedirs(TEST_POSE_DIR, exist_ok=True)


def extract_pose_features(frame, pose_detector):
    """Extract pose keypoints using MediaPipe Pose"""
    results = pose_detector.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    
    if not results.pose_landmarks:
        # Return zeros if no pose detected
        return np.zeros(33 * 3), None, None  # 33 landmarks, each with x, y, z
    
    # Extract pose landmarks
    pose_landmarks = []
    for landmark in results.pose_landmarks.landmark:
        pose_landmarks.extend([landmark.x, landmark.y, landmark.z])

    return np.array(pose_landmarks)

def process_video(video_id, frames_dir, pose_dir, anno_dir):
    """Process a single video to extract pose and hand features"""
    # Load frames
    npz_path = os.path.join(frames_dir, f"{video_id}_frames.npz")
    anno_path = os.path.join(anno_dir, f"{video_id}_annotations.json")
    
    # Check if files exist
    if not os.path.exists(npz_path):
        print(f"Frames file not found: {npz_path}")
        return False
    
    if not os.path.exists(anno_path):
        print(f"Annotation file not found: {anno_path}")
        return False
    
    # Load frames
    frames_data = np.load(npz_path)
    frames = frames_data['frames']
    
    # Load annotations to get number of frames and action info
    with open(anno_path, 'r') as f:
        anno_data = json.load(f)
    
    num_frames = anno_data["num_frames"]
    annotations = anno_data.get("annotations", [])
    
    print(f"Processing {video_id}: {num_frames} frames, {len(annotations)} actions")
   
    # Initialize detectors
    with mp_pose.Pose(
        static_image_mode=False,  # Set to False for video sequence
        model_complexity=1,
        min_detection_confidence=0.5) as pose_detector:
        

        # Initialize arrays to store features
        pose_features = np.zeros((num_frames, 99))  # 33 landmarks × 3 coordinates
        
        # Process each frame
        for i in tqdm(range(min(num_frames, len(frames))), desc=f"Processing {video_id}"):
            frame = frames[i]

            # Extract pose features and wrist landmarks
            pose_features[i]= extract_pose_features(frame, pose_detector)
            
    
    # Save pose features
    pose_output_path = os.path.join(pose_dir, f"{video_id}_pose.npz")
    np.savez_compressed(pose_output_path, pose=pose_features)
    
    
    print(f"✅ Saved features for {video_id}:")
    print(f"  Pose: {pose_features.shape}")
    
    return True

def process_dataset(frames_dir, anno_dir, pose_dir, hand_dir):
    """Process all videos in a dataset split"""
    # Get all frame files
    frame_files = glob.glob(os.path.join(frames_dir, "*_frames.npz"))
    
    success_count = 0
    error_count = 0
    
    for frame_file in frame_files:
        video_id = os.path.basename(frame_file).replace("_frames.npz", "")
        
        # Check if pose and hand data already exist
        pose_output_path = os.path.join(pose_dir, f"{video_id}_pose.npz")
        hand_output_path = os.path.join(hand_dir, f"{video_id}_hand.npz")
        
        if os.path.exists(pose_output_path) and os.path.exists(hand_output_path):
            print(f"Skipping {video_id} - features already extracted")
            success_count += 1
            continue
        
        try:
            result = process_video(video_id, frames_dir, pose_dir, anno_dir)
            if result:
                success_count += 1
            else:
                error_count += 1
        except Exception as e:
            print(f"❌ Error processing {video_id}: {str(e)}")
            error_count += 1
    
    return success_count, error_count

def main():
    parser = argparse.ArgumentParser(description="Extract pose and hand features from videos")
    parser.add_argument(
        "--split", 
        type=str, 
        choices=["train", "test", "all"],
        default="all", 
        help="Dataset split to process"
    )
    args = parser.parse_args()
    

    if args.split == "train" or args.split == "all":
        print("Processing training data...")
        train_success, train_error = process_dataset(
            TRAIN_FRAMES_DIR, TRAIN_ANNO_DIR, TRAIN_POSE_DIR, TRAIN_HAND_DIR
        )
        print(f"Training data: {train_success} videos processed, {train_error} errors")
    
    if args.split == "test" or args.split == "all":
        print("Processing testing data...")
        test_success, test_error = process_dataset(
            TEST_FRAMES_DIR, TEST_ANNO_DIR, TEST_POSE_DIR, TEST_HAND_DIR
        )
        print(f"Testing data: {test_success} videos processed, {test_error} errors")
    
    print("Feature extraction complete!")

if __name__ == "__main__":
    main() 