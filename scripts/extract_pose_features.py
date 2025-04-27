import os
import numpy as np
import mediapipe as mp
import argparse
from tqdm import tqdm
import glob
from src.utils.feature_extraction import extract_pose_features

mp_pose = mp.solutions.pose

BASE_DIR = "Data"
TRAIN_FRAMES_DIR = os.path.join(BASE_DIR, "full_videos", "train", "frames")
VAL_FRAMES_DIR = os.path.join(BASE_DIR, "full_videos", "val", "frames")
TEST_FRAMES_DIR = os.path.join(BASE_DIR, "full_videos", "test", "frames")

TRAIN_ANNO_DIR = os.path.join(BASE_DIR, "full_videos", "train", "annotations")
VAL_ANNO_DIR = os.path.join(BASE_DIR, "full_videos", "val", "annotations")
TEST_ANNO_DIR = os.path.join(BASE_DIR, "full_videos", "test", "annotations")

TRAIN_POSE_DIR = os.path.join(BASE_DIR, "full_videos", "train", "pose")
VAL_POSE_DIR = os.path.join(BASE_DIR, "full_videos", "val", "pose")
TEST_POSE_DIR = os.path.join(BASE_DIR, "full_videos", "test", "pose")


os.makedirs(TRAIN_POSE_DIR, exist_ok=True)
os.makedirs(VAL_POSE_DIR, exist_ok=True)
os.makedirs(TEST_POSE_DIR, exist_ok=True)


def process_video(video_id, frames_dir, pose_dir, anno_dir):
    """Process a single video to extract pose and hand features"""
    npz_path = os.path.join(frames_dir, f"{video_id}_frames.npz")
    
    if not os.path.exists(npz_path):
        print(f"Frames file not found: {npz_path}")
        return False
    
    frames_data = np.load(npz_path)
    frames = frames_data['frames']
    num_frames = frames.shape[0]
       
    with mp_pose.Pose(
        static_image_mode=False, 
        model_complexity=1,
        min_detection_confidence=0.5) as pose_detector:
        
        pose_features = np.zeros((num_frames, 99))  # 33 landmarks × 3 coordinates
        
        for i in tqdm(range(num_frames), desc=f"Processing {video_id}"):
            frame = frames[i]

            pose_features[i]= extract_pose_features(frame, pose_detector)
            
    
    pose_output_path = os.path.join(pose_dir, f"{video_id}_pose.npz")
    np.savez_compressed(pose_output_path, pose=pose_features)

    print(f"saved features for {video_id}, shape: {pose_features.shape}")
    
    return True

def process_dataset(frames_dir, anno_dir, pose_dir):
    """Process all videos in a dataset split"""
    frame_files = glob.glob(os.path.join(frames_dir, "*_frames.npz"))
    
    success_count = 0
    error_count = 0
    
    for frame_file in frame_files:
        video_id = os.path.basename(frame_file).replace("_frames.npz", "")
        
        pose_output_path = os.path.join(pose_dir, f"{video_id}_pose.npz")

        if os.path.exists(pose_output_path):
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
            print(f"Error processing {video_id}: {str(e)}")
            error_count += 1
    
    return success_count, error_count

def main():
    parser = argparse.ArgumentParser(description="Extract pose features from videos")
    parser.add_argument(
        "--split", 
        type=str, 
        choices=["train", "val", "test", "all"],
        default="all", 
        help="Dataset split to process"
    )
    args = parser.parse_args()
    

    if args.split == "train" or args.split == "all":
        print("Processing training data...")
        train_success, train_error = process_dataset(
            TRAIN_FRAMES_DIR, TRAIN_ANNO_DIR, TRAIN_POSE_DIR
        )
        print(f"Training data: {train_success} videos processed, {train_error} errors")
    
    if args.split == "val" or args.split == "all":
        print("Processing validation data...")
        val_success, val_error = process_dataset(
            VAL_FRAMES_DIR, VAL_ANNO_DIR, VAL_POSE_DIR
        )
        print(f"Validation data: {val_success} videos processed, {val_error} errors")

    if args.split == "test" or args.split == "all":
        print("Processing testing data...")
        test_success, test_error = process_dataset(
            TEST_FRAMES_DIR, TEST_ANNO_DIR, TEST_POSE_DIR,)
        print(f"Testing data: {test_success} videos processed, {test_error} errors")
    
    print("Feature extraction complete!")

if __name__ == "__main__":
    main() 