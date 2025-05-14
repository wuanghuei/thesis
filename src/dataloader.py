import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import json
from pathlib import Path
import src.utils.helpers as helpers

class FeatureDataset(Dataset):
    def __init__(self, features_dir, anno_dir, num_classes, mode='train', max_seq_len=600, stride=8, window_size=16):
        self.features_dir = Path(features_dir)
        self.anno_dir = Path(anno_dir)
        self.num_classes = num_classes
        self.mode = mode
        self.samples = []
        self.max_seq_len = max_seq_len
        self.stride = stride
        self.window_size = window_size
        self.background_class_idx = 0
        self.ignore_index = -100

        # Get all feature files
        feature_files = [f for f in self.features_dir.iterdir() if f.name.endswith('_features.npz')]
        
        for feature_file in feature_files:
            video_id = feature_file.name.replace('_features.npz', '')
            anno_path = self.anno_dir / f"{video_id}_annotations.json"
            
            if anno_path.exists():
                self._process_video(video_id)
            else:
                print(f"Skipping {video_id} - missing annotations")
                
        print(f"[{mode}] Loaded {len(self.samples)} samples from {len(feature_files)} videos")
        
    def _process_video(self, video_id):
        # Load annotations
        anno_path = self.anno_dir / f"{video_id}_annotations.json"
        with open(anno_path, 'r') as f:
            anno = json.load(f)
            
        # Load features
        feature_path = self.features_dir / f"{video_id}_features.npz"
        features = np.load(feature_path)
        clip_features = features['clip_features']  # Shape: [num_frames, feature_dim]
        
        # Create sample
        self.samples.append({
            'video_id': video_id,
            'features': clip_features,
            'annotations': anno
        })

    def _compute_iou(self, window_start, window_end, action_start, action_end):
        # window_end và action_end là frame CUỐI CÙNG CÓ trong đoạn
        intersection_start = max(window_start, action_start)
        intersection_end = min(window_end, action_end)

        if intersection_end < intersection_start: # Không có overlap
            return 0.0

        intersection_length = intersection_end - intersection_start + 1
        window_length = window_end - window_start + 1
        action_length = action_end - action_start + 1
        union_length = window_length + action_length - intersection_length

        return intersection_length / union_length if union_length > 0 else 0.0

    def _get_window_center(self, token_idx):
        """Get the center frame of the window represented by this token."""
        window_start = token_idx * self.stride
        window_end = window_start + self.window_size - 1
        return (window_start + window_end) / 2

    def __len__(self):
        return len(self.samples)
        
    def __getitem__(self, idx):
        sample = self.samples[idx]
        features = sample['features']  # [T_video, D_backbone]
        annotations = sample['annotations']
        
        # Convert features to tensor
        features = torch.from_numpy(features).float()
        T_video = features.shape[0]
        
        # Initialize ground truth tensors
        gt_classes = torch.full((self.max_seq_len,), self.ignore_index, dtype=torch.long)
        gt_offsets = torch.zeros((self.max_seq_len, 2), dtype=torch.float)
        offset_loss_mask = torch.zeros(self.max_seq_len, dtype=torch.bool)
        gt_actionness = torch.zeros(self.max_seq_len, dtype=torch.float)
        attention_padding_mask = torch.ones(self.max_seq_len, dtype=torch.bool)
        
        # Set padding mask for actual data
        attention_padding_mask[:T_video] = False
        
        # Process each token
        for token_idx in range(min(T_video, self.max_seq_len)):
            window_start = token_idx * self.stride
            window_end = window_start + self.window_size - 1
            window_center = self._get_window_center(token_idx)
            # Find overlapping actions
            overlapping_actions = []
            for anno in annotations['annotations']:
                action_start = anno['start_frame']
                action_end = anno['end_frame']
                action_class = anno['action_id']
                
                iou = self._compute_iou(window_start, window_end, action_start, action_end)
                if iou > 0.1:  # Significant overlap threshold
                    overlapping_actions.append({
                        'class': action_class,
                        'start': action_start,
                        'end': action_end,
                        'iou': iou,
                        'center_dist': abs((action_start + action_end) / 2 - window_center)
                    })
            
            if not overlapping_actions:
                # No significant overlap - background
                gt_classes[token_idx] = self.background_class_idx
                gt_actionness[token_idx] = 0.0
                offset_loss_mask[token_idx] = False
            elif len(overlapping_actions) == 1:
                # Single action overlap - clear case
                action = overlapping_actions[0]
                gt_classes[token_idx] = action['class']
                gt_actionness[token_idx] = 1.0
                
                # Compute target offsets
                d_left = max(0, window_center - action['start'])
                d_right = max(0, action['end'] - window_center)
                gt_offsets[token_idx] = torch.tensor([d_left, d_right])
                offset_loss_mask[token_idx] = True
            else:
                # Multiple actions - ambiguous case
                # Choose action with highest IoU
                best_action = max(overlapping_actions, key=lambda x: x['iou'])
                gt_classes[token_idx] = best_action['class']
                gt_actionness[token_idx] = 1.0
                offset_loss_mask[token_idx] = False  # Don't train offsets for ambiguous cases
        if T_video < self.max_seq_len:
            # Chỉ gán ignore_index cho các token thực sự là padding
            # Những token < T_video đã được gán là action hoặc background thật sự rồi
            gt_classes[T_video:] = self.ignore_index
        # Pad features to max_seq_len
        padded_features = torch.zeros((self.max_seq_len, features.shape[1]), dtype=features.dtype)
        padded_features[:T_video] = features[:self.max_seq_len]
        
        return {
            'feature_sequence': padded_features,         # [max_seq_len, D_backbone]
            'gt_classes': gt_classes,                    # [max_seq_len]
            'gt_offsets': gt_offsets,                    # [max_seq_len, 2]
            'offset_loss_mask': offset_loss_mask,        # [max_seq_len]
            'gt_actionness': gt_actionness,              # [max_seq_len]
            'attention_padding_mask': attention_padding_mask,  # [max_seq_len]
            'metadata': {
                'video_id': sample['video_id'],
                'num_frames': annotations['num_frames']
            }
        }

def custom_collate_fn(batch):
    # Stack all tensors
    feature_sequences = torch.stack([item['feature_sequence'] for item in batch])
    gt_classes = torch.stack([item['gt_classes'] for item in batch])
    gt_offsets = torch.stack([item['gt_offsets'] for item in batch])
    offset_loss_masks = torch.stack([item['offset_loss_mask'] for item in batch])
    gt_actionness = torch.stack([item['gt_actionness'] for item in batch])
    attention_padding_masks = torch.stack([item['attention_padding_mask'] for item in batch])
    
    # Collect metadata
    metadata = [item['metadata'] for item in batch]
    
    return {
        'feature_sequence': feature_sequences,      # [batch_size, max_seq_len, D_backbone]
        'gt_classes': gt_classes,                   # [batch_size, max_seq_len]
        'gt_offsets': gt_offsets,                   # [batch_size, max_seq_len, 2]
        'offset_loss_mask': offset_loss_masks,      # [batch_size, max_seq_len]
        'gt_actionness': gt_actionness,             # [batch_size, max_seq_len]
        'attention_padding_mask': attention_padding_masks,  # [batch_size, max_seq_len]
        'metadata': metadata
    }

def get_train_loader(cfg, shuffle=True):
    data_cfg = cfg.get('data', {})
    train_cfg = cfg.get('base_model_training', {})
    global_cfg = cfg.get('global', {})

    features_dir = Path(data_cfg.get('base_dir', 'data')) / 'features/train'
    anno_dir = Path(data_cfg.get('base_dir', 'data')) / 'full_videos/train/annotations'
    num_classes = global_cfg.get('num_classes', 6)  # Now default to 6 classes (background + 5 actions)
    batch_size = train_cfg.get('dataloader', {}).get('batch_size', 1)
    num_workers = train_cfg.get('dataloader', {}).get('num_workers', 4)
    max_seq_len = global_cfg.get('max_seq_len', 600)
    stride = global_cfg.get('stride', 8)
    window_size = global_cfg.get('window_size', 16)

    dataset = FeatureDataset(
        features_dir,
        anno_dir,
        num_classes=num_classes,
        mode='train',
        max_seq_len=max_seq_len,
        stride=stride,
        window_size=window_size
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=custom_collate_fn
    )

def get_val_loader(cfg, shuffle=False):
    data_cfg = cfg.get('data', {})
    train_cfg = cfg.get('base_model_training', {})
    global_cfg = cfg.get('global', {})

    features_dir = Path(data_cfg.get('base_dir', 'data')) / 'features/val'
    anno_dir = Path(data_cfg.get('base_dir', 'data')) / 'full_videos/val/annotations'
    num_classes = global_cfg.get('num_classes', 6)  # Now default to 6 classes (background + 5 actions)
    batch_size = train_cfg.get('dataloader', {}).get('batch_size', 1)
    num_workers = train_cfg.get('dataloader', {}).get('num_workers', 4)
    max_seq_len = global_cfg.get('max_seq_len', 600)
    stride = global_cfg.get('stride', 8)
    window_size = global_cfg.get('window_size', 16)

    dataset = FeatureDataset(
        features_dir,
        anno_dir,
        num_classes=num_classes,
        mode='val',
        max_seq_len=max_seq_len,
        stride=stride,
        window_size=window_size
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=custom_collate_fn
    )

def get_test_loader(cfg, shuffle=False):
    data_cfg = cfg.get('data', {})
    train_cfg = cfg.get('base_model_training', {})
    global_cfg = cfg.get('global', {})

    features_dir = Path(data_cfg.get('base_dir', 'data')) / 'features/test'
    anno_dir = Path(data_cfg.get('base_dir', 'data')) / 'full_videos/test/annotations'
    num_classes = global_cfg.get('num_classes', 6)  # Now default to 6 classes (background + 5 actions)
    batch_size = train_cfg.get('dataloader', {}).get('batch_size', 1)
    num_workers = train_cfg.get('dataloader', {}).get('num_workers', 4)
    max_seq_len = global_cfg.get('max_seq_len', 600)
    stride = global_cfg.get('stride', 8)
    window_size = global_cfg.get('window_size', 16)

    dataset = FeatureDataset(
        features_dir,
        anno_dir,
        num_classes=num_classes,
        mode='test',
        max_seq_len=max_seq_len,
        stride=stride,
        window_size=window_size
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=custom_collate_fn
    )
