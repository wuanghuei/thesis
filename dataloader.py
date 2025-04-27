import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import json
import random

# ====== CẤU HÌNH CỐ ĐỊNH ======

NUM_CLASSES = 5
WINDOW_SIZE = 32  # Sliding window size

# PATHS - cập nhật đường dẫn cho train, val, test
BASE_DIR = "Data"
# Training data (Subjects 1-20)
TRAIN_FRAMES_DIR = os.path.join(BASE_DIR, "full_videos", "train", "frames")
TRAIN_ANNO_DIR = os.path.join(BASE_DIR, "full_videos", "train", "annotations")
TRAIN_POSE_DIR = os.path.join(BASE_DIR, "full_videos", "train", "pose")

# Validation data (Subjects 21-26)
VAL_FRAMES_DIR = os.path.join(BASE_DIR, "full_videos", "val", "frames")
VAL_ANNO_DIR = os.path.join(BASE_DIR, "full_videos", "val", "annotations")
VAL_POSE_DIR = os.path.join(BASE_DIR, "full_videos", "val", "pose")

# Testing data (Subjects 27-41)
TEST_FRAMES_DIR = os.path.join(BASE_DIR, "full_videos", "test", "frames")
TEST_ANNO_DIR = os.path.join(BASE_DIR, "full_videos", "test", "annotations")
TEST_POSE_DIR = os.path.join(BASE_DIR, "full_videos", "test", "pose")

# ====== Helper Functions ======
def compute_velocity(pose_data):
    """
    Tính velocity features từ pose data
    
    Args:
        pose_data: Numpy array shape (T, 99) - 33 keypoints x 3 coords
        
    Returns:
        velocity: Numpy array shape (T, 99) - velocity for each keypoint coordinate
    """
    # Khởi tạo mảng velocity với kích thước giống pose_data
    T, D = pose_data.shape
    velocity = np.zeros_like(pose_data)
    
    # Frame đầu tiên có velocity = 0 (đã zero-initialized)
    # Tính velocity cho các frame tiếp theo bằng cách lấy sự khác biệt giữa frame hiện tại và trước đó
    if T > 1:
        velocity[1:] = pose_data[1:] - pose_data[:-1]
    
    # Chuẩn hóa velocity để tránh các giá trị quá lớn
    # Sử dụng robust scaling thay vì z-normalization
    max_abs_val = np.max(np.abs(velocity)) + 1e-10  # Tránh chia cho 0
    velocity = velocity / max_abs_val
    
    return velocity

# ====== Full Video Dataset ======
class FullVideoDataset(Dataset):
    def __init__(self, frames_dir, anno_dir, pose_dir, mode='train', window_size=WINDOW_SIZE):
        """
        Dataset for full video temporal localization
        
        Args:
            frames_dir: Directory containing video frame .npz files
            anno_dir: Directory containing annotation .json files
            pose_dir: Directory containing pose data .npz files
            mode: 'train', 'val', or 'test'
            window_size: Size of sliding window for processing
        """
        self.frames_dir = frames_dir
        self.anno_dir = anno_dir
        self.pose_dir = pose_dir
        self.mode = mode
        self.window_size = window_size
        self.samples = []

        # Get all available videos
        video_ids = []
        frame_files = os.listdir(frames_dir) if os.path.exists(frames_dir) else []
        
        for fname in frame_files:
            if fname.endswith("_frames.npz"):
                video_id = fname.replace("_frames.npz", "")
                anno_path = os.path.join(anno_dir, f"{video_id}_annotations.json")
                pose_path = os.path.join(pose_dir, f"{video_id}_pose.npz")
                
                # Check if required files exist
                if os.path.exists(anno_path) and os.path.exists(pose_path):
                    video_ids.append(video_id)
                else:
                    missing = []
                    if not os.path.exists(anno_path): missing.append("annotations")
                    if not os.path.exists(pose_path): missing.append("pose")
                    print(f"Skipping {video_id} - missing {', '.join(missing)}")
        
        # Process all videos
        for video_id in video_ids:
            self._process_video(video_id)
                
        print(f"[{mode}] Loaded {len(self.samples)} sliding windows from {len(video_ids)} videos")
        
    def _process_video(self, video_id):
        """Process a single video and generate sliding window samples"""
        # Load annotations
        anno_path = os.path.join(self.anno_dir, f"{video_id}_annotations.json")
        with open(anno_path, 'r') as f:
            anno = json.load(f)
        
        num_frames = anno["num_frames"]
        annotations = anno["annotations"]
        
        # If video is shorter than window size, we'll pad it
        if num_frames < self.window_size:
            # Create one window covering the entire video
            self._add_window(video_id, 0, num_frames, annotations)
            return
            
        # For longer videos, create sliding windows with overlap
        stride = self.window_size // 2  # 50% overlap
        
        for start in range(0, num_frames - self.window_size + 1, stride):
            end = start + self.window_size
            self._add_window(video_id, start, end, annotations)
            
        # Add final window if needed
        if (num_frames - self.window_size) % stride != 0:
            start = num_frames - self.window_size
            end = num_frames
            self._add_window(video_id, start, end, annotations)
            
    def _add_window(self, video_id, start_idx, end_idx, all_annotations):
        """Add a sliding window with corresponding annotations to samples"""
        window_annos = []
        
        # Find all actions that overlap with this window
        for anno in all_annotations:
            action_start = anno["start_frame"]
            action_end = anno["end_frame"]
            
            # Check if action overlaps with window
            if action_end < start_idx or action_start >= end_idx:
                continue  # No overlap
                
            # Convert to window coordinates
            rel_start = max(0, action_start - start_idx)
            rel_end = min(end_idx - start_idx, action_end - start_idx)
            
            if rel_end > rel_start:  # Ensure valid segment
                window_annos.append({
                    "action_id": anno["action_id"],
                    "start_frame": rel_start,
                    "end_frame": rel_end,
                    "original_start": anno["original_start"],
                    "original_end": anno["original_end"]
                })
                
        # Only add windows with at least one action for training
        # For testing, include all windows

        self.samples.append({
            "video_id": video_id,
            "start_idx": start_idx,
            "end_idx": end_idx,
            "annotations": window_annos
            })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        video_id = sample["video_id"]
        start_idx = sample["start_idx"]
        end_idx = sample["end_idx"]
        annotations = sample["annotations"]
        
        # Load video frames (.npz format)
        frames_path = os.path.join(self.frames_dir, f"{video_id}_frames.npz")
        npz_data = np.load(frames_path)
        all_frames = npz_data['frames']
        
        # Load pose data (.npz format)
        pose_path = os.path.join(self.pose_dir, f"{video_id}_pose.npz")
        pose_npz = np.load(pose_path)
        all_pose = pose_npz['pose']  # Assuming this contains pose keypoints
        
        # Extract window frames
        if end_idx <= all_frames.shape[0]:
            frames = all_frames[start_idx:end_idx]
        else:
            # Padding needed
            frames = np.zeros((self.window_size, *all_frames.shape[1:]), dtype=all_frames.dtype)
            actual_frames = all_frames[start_idx:end_idx]
            frames[:actual_frames.shape[0]] = actual_frames
            # Repeat last frame if needed
            if actual_frames.shape[0] > 0:
                frames[actual_frames.shape[0]:] = actual_frames[-1]
        
        # Extract window pose data
        if end_idx <= all_pose.shape[0]:
            pose_data = all_pose[start_idx:end_idx]
        else:
            # Padding needed
            pose_data = np.zeros((self.window_size, all_pose.shape[1]), dtype=all_pose.dtype)
            actual_pose = all_pose[start_idx:end_idx]
            pose_data[:actual_pose.shape[0]] = actual_pose
            # Repeat last pose if needed
            if actual_pose.shape[0] > 0:
                pose_data[actual_pose.shape[0]:] = actual_pose[-1]
        
        # Compute velocity features for pose data
        velocity_data = compute_velocity(pose_data)
        
        # Concatenate pose and velocity features
        pose_with_velocity = np.concatenate([pose_data, velocity_data], axis=1)  # (T, 198)
                
        # Convert frames to RGB tensor
        frames = torch.from_numpy(frames).float() / 255.0  # (T, H, W, 3)
        frames = frames.permute(3, 0, 1, 2)  # (3, T, H, W)
        
        # Convert pose data to tensor
        pose_with_velocity = torch.from_numpy(pose_with_velocity).float()  # (T, 198)
        
        # Create action masks for each class
        # Shape: (num_classes, window_size)
        action_masks = torch.zeros((NUM_CLASSES, self.window_size), dtype=torch.float32)
        
        # Create start and end points masks for each class
        # Shape: (num_classes, window_size)
        start_mask = torch.zeros((NUM_CLASSES, self.window_size), dtype=torch.float32)
        end_mask = torch.zeros((NUM_CLASSES, self.window_size), dtype=torch.float32)
        
        # Apply Gaussian smoothing around start/end points
        def gaussian_kernel(center, sigma=1.0):
            x = torch.arange(self.window_size).float()
            return torch.exp(-((x - center) ** 2) / (2 * sigma ** 2))
        
        # Fill action masks
        has_action = False
        for anno in annotations:
            action_id = anno["action_id"]
            s, e = anno["start_frame"], anno["end_frame"]
            
            # Action segment mask (1 where action is occurring)
            action_masks[action_id, s:e] = 1.0
            has_action = True
            
            # Start/end point masks with Gaussian smoothing
            start_mask[action_id] += gaussian_kernel(s, sigma=2.0)
            end_mask[action_id] += gaussian_kernel(e-1, sigma=2.0)  # -1 because end is exclusive
            
        # Normalize Gaussian peaks
        start_mask = torch.clamp(start_mask, 0, 1)
        end_mask = torch.clamp(end_mask, 0, 1)
        
        # Create metadata for evaluation
        metadata = {
            "video_id": video_id,
            "start_idx": start_idx,
            "end_idx": end_idx,
            "annotations": annotations,
            "has_action": has_action
        }
        
        # We no longer use hand data, but keep the format compatible with the model 
        # by returning a dummy tensor with the same shape as before (T, 2)
        dummy_hand = torch.zeros((self.window_size, 2), dtype=torch.float32)
        
        return frames, pose_with_velocity, dummy_hand, action_masks, start_mask, end_mask, metadata

# Hàm collate_fn tùy chỉnh để xử lý metadata không đồng nhất
def custom_collate_fn(batch):
    """
    Custom collate function để xử lý khi các phần tử trong batch có kích thước khác nhau
    (đặc biệt là phần annotations trong metadata)
    """
    # Chia batch thành các thành phần
    frames, pose_data, hand_data, action_masks, start_masks, end_masks, metadata = zip(*batch)
    
    # Stack các tensor có kích thước cố định
    frames = torch.stack(frames)
    pose_data = torch.stack(pose_data)
    hand_data = torch.stack(hand_data)
    action_masks = torch.stack(action_masks)
    start_masks = torch.stack(start_masks)
    end_masks = torch.stack(end_masks)
    
    # Giữ metadata dưới dạng list (không stack)
    # Vì metadata chứa annotations có số lượng khác nhau giữa các mẫu
    
    return frames, pose_data, hand_data, action_masks, start_masks, end_masks, metadata

# ====== Get Loaders ======
def get_train_loader(batch_size=1, shuffle=True):
    dataset = FullVideoDataset(
        TRAIN_FRAMES_DIR, 
        TRAIN_ANNO_DIR, 
        TRAIN_POSE_DIR,
        mode='train'
    )
    return DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=shuffle, 
        num_workers=4,
        collate_fn=custom_collate_fn
    )

def get_val_loader(batch_size=1, shuffle=False):
    dataset = FullVideoDataset(
        VAL_FRAMES_DIR, 
        VAL_ANNO_DIR, 
        VAL_POSE_DIR,
        mode='val'
    )
    return DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=shuffle, 
        num_workers=4,
        collate_fn=custom_collate_fn
    )

def get_test_loader(batch_size=1, shuffle=False):
    dataset = FullVideoDataset(
        TEST_FRAMES_DIR, 
        TEST_ANNO_DIR, 
        TEST_POSE_DIR,
        mode='test'
    )
    return DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=shuffle, 
        num_workers=4,
        collate_fn=custom_collate_fn
    )
