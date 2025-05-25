import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import json
import random
import src.utils.feature_extraction as feature_extraction

NUM_CLASSES = 5
WINDOW_SIZE = 32
MVIT_STRIDE = 16

BASE_DIR = "Data"

MVIT_FEATURES_BASE_DIR = os.path.join(BASE_DIR, "features", "mvit_32f_16s")
RESNET_FEATURES_BASE_DIR = os.path.join(BASE_DIR, "features", "resnet18_per_frame")

ANNOTATIONS_BASE_DIR = os.path.join(BASE_DIR, "full_videos")
POSE_BASE_DIR = os.path.join(BASE_DIR, "full_videos")

TRAIN_MVIT_DIR = os.path.join(MVIT_FEATURES_BASE_DIR, "train")
TRAIN_RESNET_DIR = os.path.join(RESNET_FEATURES_BASE_DIR, "train")
TRAIN_ANNO_DIR = os.path.join(ANNOTATIONS_BASE_DIR, "train", "annotations")
TRAIN_POSE_DIR = os.path.join(POSE_BASE_DIR, "train", "pose")

VAL_MVIT_DIR = os.path.join(MVIT_FEATURES_BASE_DIR, "val")
VAL_RESNET_DIR = os.path.join(RESNET_FEATURES_BASE_DIR, "val")
VAL_ANNO_DIR = os.path.join(ANNOTATIONS_BASE_DIR, "val", "annotations")
VAL_POSE_DIR = os.path.join(POSE_BASE_DIR, "val", "pose")

TEST_MVIT_DIR = os.path.join(MVIT_FEATURES_BASE_DIR, "test")
TEST_RESNET_DIR = os.path.join(RESNET_FEATURES_BASE_DIR, "test")
TEST_ANNO_DIR = os.path.join(ANNOTATIONS_BASE_DIR, "test", "annotations")
TEST_POSE_DIR = os.path.join(POSE_BASE_DIR, "test", "pose")


class FeatureVideoDataset(Dataset):
    def __init__(self, mvit_dir, resnet_dir, anno_dir, pose_dir, mode='train', window_size=WINDOW_SIZE):
        self.mvit_dir = mvit_dir
        self.resnet_dir = resnet_dir
        self.anno_dir = anno_dir
        self.pose_dir = pose_dir
        self.mode = mode
        self.window_size = window_size
        self.mvit_stride = MVIT_STRIDE
        self.samples = []

        video_ids = []
        if os.path.exists(self.mvit_dir):
            for fname in os.listdir(self.mvit_dir):
                if fname.endswith("_features.npz"):
                    video_id = fname.replace("_features.npz", "")
                    resnet_path = os.path.join(self.resnet_dir, f"{video_id}_resnet_features.npz")
                    anno_path = os.path.join(self.anno_dir, f"{video_id}_annotations.json")
                    pose_path = os.path.join(self.pose_dir, f"{video_id}_pose.npz")

                    if os.path.exists(resnet_path) and os.path.exists(anno_path) and os.path.exists(pose_path):
                        video_ids.append(video_id)
                    else:
                        missing = []
                        if not os.path.exists(resnet_path): missing.append(f"ResNet features ({resnet_path})")
                        if not os.path.exists(anno_path): missing.append(f"annotations ({anno_path})")
                        if not os.path.exists(pose_path): missing.append(f"pose ({pose_path})")
                        print(f"skip {video_id} in '{mode}' because missing: {', '.join(missing)}")
        else:
            print(f"Error: Folder not found '{mode}': {self.mvit_dir}")
            
        for video_id in video_ids:
            self._process_video(video_id)
                
        print(f"[{mode}] Loaded {len(self.samples)} sliding windows from {len(video_ids)} videos.")
        if len(video_ids) > 0 and len(self.samples) == 0:
            print(f"[{mode}] No valid sliding windows found")

    def _process_video(self, video_id):
        anno_path = os.path.join(self.anno_dir, f"{video_id}_annotations.json")
        try:
            with open(anno_path, 'r') as f:
                anno_data = json.load(f)
        except Exception as e:
            print(f"Error while loading videos {video_id}: {e}")
            return
        
        num_video_frames = anno_data["num_frames"]
        annotations = anno_data["annotations"]

        mvit_feature_path = os.path.join(self.mvit_dir, f"{video_id}_features.npz")
        try:
            mvit_npz = np.load(mvit_feature_path)
            # Giả sử 'features' luôn tồn tại nếu file .npz được load thành công
            num_mvit_windows_in_file = mvit_npz['features'].shape[0]
        except Exception as e:
            print(f"Cannot load video {video_id}")
            return

        mvit_window_idx_counter = 0
        for i in range(num_mvit_windows_in_file):
            start_frame_of_window = i * self.mvit_stride
            end_frame_for_slicing = start_frame_of_window + self.window_size
            
            self._add_window(video_id, start_frame_of_window, end_frame_for_slicing, annotations, mvit_window_idx_counter)
            mvit_window_idx_counter += 1
            
            if self.mvit_stride == 0:
                break
            if (i + 1) * self.mvit_stride >= num_video_frames and start_frame_of_window < (i+1) * self.mvit_stride:
                 break

    def _add_window(self, video_id, start_idx, end_idx_for_slicing, all_annotations, mvit_window_idx):
        window_annos = []
        for anno in all_annotations:
            action_start = anno["start_frame"]
            action_end = anno["end_frame"]
            if action_end < start_idx or action_start >= end_idx_for_slicing:
                continue
            rel_start = max(0, action_start - start_idx)
            rel_end = min(self.window_size, action_end - start_idx)
            if rel_end > rel_start:
                window_annos.append({
                    "action_id": anno["action_id"]-1,
                    "start_frame": rel_start,
                    "end_frame": rel_end,
                })
        self.samples.append({
            "video_id": video_id,
            "mvit_window_idx": mvit_window_idx,
            "slice_start_idx": start_idx,
            "annotations": window_annos
        })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        video_id = sample["video_id"]
        mvit_window_idx_to_load = sample["mvit_window_idx"]
        slice_start_idx = sample["slice_start_idx"]
        annotations = sample["annotations"]
        
        try:
            mvit_feature_path = os.path.join(self.mvit_dir, f"{video_id}_features.npz")
            mvit_npz = np.load(mvit_feature_path)
            mvit_feature = mvit_npz['features'][mvit_window_idx_to_load]
            mvit_feature_tensor = torch.from_numpy(mvit_feature).float()

            resnet_feature_path = os.path.join(self.resnet_dir, f"{video_id}_resnet_features.npz")
            resnet_npz = np.load(resnet_feature_path)
            all_resnet_features = resnet_npz['features']
            
            resnet_window_data = np.zeros((self.window_size, all_resnet_features.shape[1]), dtype=all_resnet_features.dtype)
            actual_end_idx = min(slice_start_idx + self.window_size, all_resnet_features.shape[0])
            available_resnet_features = all_resnet_features[slice_start_idx:actual_end_idx]
            num_available_resnet_frames = available_resnet_features.shape[0]
            
            if num_available_resnet_frames > 0:
                resnet_window_data[:num_available_resnet_frames] = available_resnet_features
                if num_available_resnet_frames < self.window_size:
                    resnet_window_data[num_available_resnet_frames:] = available_resnet_features[-1]
            resnet_features_tensor = torch.from_numpy(resnet_window_data).float()

            pose_path = os.path.join(self.pose_dir, f"{video_id}_pose.npz")
            pose_npz = np.load(pose_path)
            all_pose_raw = pose_npz['pose']
            
            pose_window_raw = np.zeros((self.window_size, all_pose_raw.shape[1]), dtype=all_pose_raw.dtype)
            actual_end_idx_pose = min(slice_start_idx + self.window_size, all_pose_raw.shape[0])
            available_pose_raw = all_pose_raw[slice_start_idx:actual_end_idx_pose]
            num_available_pose_frames = available_pose_raw.shape[0]

            if num_available_pose_frames > 0:
                pose_window_raw[:num_available_pose_frames] = available_pose_raw
                if num_available_pose_frames < self.window_size:
                    pose_window_raw[num_available_pose_frames:] = available_pose_raw[-1]
            
            velocity_data = feature_extraction.compute_velocity(pose_window_raw)
            pose_with_velocity = np.concatenate([pose_window_raw, velocity_data], axis=1)
            pose_with_velocity_tensor = torch.from_numpy(pose_with_velocity).float()
            
            action_masks = torch.zeros((NUM_CLASSES, self.window_size), dtype=torch.float32)
            start_mask = torch.zeros((NUM_CLASSES, self.window_size), dtype=torch.float32)
            end_mask = torch.zeros((NUM_CLASSES, self.window_size), dtype=torch.float32)
            
            def gaussian_kernel(center, sigma=1.0):
                x = torch.arange(self.window_size).float()
                return torch.exp(-((x - center) ** 2) / (2 * sigma ** 2))
            
            has_action = False
            for anno in annotations:
                action_id = anno["action_id"]
                s, e = anno["start_frame"], anno["end_frame"]
                action_masks[action_id, s:e] = 1.0
                has_action = True
                start_mask[action_id] += gaussian_kernel(s, sigma=2.0)
                end_mask[action_id] += gaussian_kernel(e - 1 if e > s else s, sigma=2.0)
                
            start_mask = torch.clamp(start_mask, 0, 1)
            end_mask = torch.clamp(end_mask, 0, 1)
            
            metadata = {
                "video_id": video_id,
                "slice_start_idx": slice_start_idx,
                "mvit_window_idx": mvit_window_idx_to_load,
                "has_action": has_action,
                "annotations": annotations
            }
            
            return mvit_feature_tensor, resnet_features_tensor, pose_with_velocity_tensor, \
                   action_masks, start_mask, end_mask, metadata

        except Exception as e:
            raise e


def custom_feature_collate_fn(batch):
    batch = [b for b in batch if b is not None]
    if not batch:
        return None 

    mvit_features, resnet_features, pose_data, \
    action_masks, start_masks, end_masks, metadata = zip(*batch)
    
    mvit_features = torch.stack(mvit_features)
    resnet_features = torch.stack(resnet_features)
    pose_data = torch.stack(pose_data)
    action_masks = torch.stack(action_masks)
    start_masks = torch.stack(start_masks)
    end_masks = torch.stack(end_masks)
    
    return mvit_features, resnet_features, pose_data, \
           action_masks, start_masks, end_masks, metadata

def get_feature_train_loader(batch_size=1, shuffle=True, num_workers=0):
    dataset = FeatureVideoDataset(
        TRAIN_MVIT_DIR,
        TRAIN_RESNET_DIR, 
        TRAIN_ANNO_DIR, 
        TRAIN_POSE_DIR,
        mode='train'
    )
    if len(dataset) == 0:
        print("Training dataset empty")
        return None
    return DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=shuffle, 
        num_workers=num_workers,
        collate_fn=custom_feature_collate_fn,
        pin_memory=True if num_workers > 0 else False
    )

def get_feature_val_loader(batch_size=1, shuffle=False, num_workers=0):
    dataset = FeatureVideoDataset(
        VAL_MVIT_DIR,
        VAL_RESNET_DIR,
        VAL_ANNO_DIR, 
        VAL_POSE_DIR,
        mode='val'
    )
    if len(dataset) == 0:
        print("Validation dataset empty")
        return None
    return DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=shuffle, 
        num_workers=num_workers,
        collate_fn=custom_feature_collate_fn,
        pin_memory=True if num_workers > 0 else False
    )

def get_feature_test_loader(batch_size=1, shuffle=False, num_workers=0):
    dataset = FeatureVideoDataset(
        TEST_MVIT_DIR,
        TEST_RESNET_DIR,
        TEST_ANNO_DIR, 
        TEST_POSE_DIR,
        mode='test'
    )
    if len(dataset) == 0:
        print("Test dataset empty")
        return None
    return DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=shuffle, 
        num_workers=num_workers,
        collate_fn=custom_feature_collate_fn,
        pin_memory=True if num_workers > 0 else False
    )