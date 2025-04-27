import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from model_fixed import TemporalActionDetector
from dataloader import get_train_loader, get_val_loader, get_test_loader
from sklearn.metrics import precision_recall_fscore_support, average_precision_score
from tqdm import tqdm
import numpy as np
import os
import json
import matplotlib.pyplot as plt
from datetime import datetime
from torch.cuda.amp import autocast, GradScaler
from collections import defaultdict


# ====== Config ======
NUM_CLASSES = 5  # Đảm bảo khớp với định nghĩa trong prepare_segments.py và dataloader.py
WINDOW_SIZE = 32  # Kích thước sliding window, phải khớp với WINDOW_SIZE trong dataloader.py
EPOCHS = 100
BATCH_SIZE = 1
GRADIENT_ACCUMULATION_STEPS = 4
LR = 1e-5  # Giảm từ 5e-5 xuống 2e-5 để ổn định loss
WEIGHT_DECAY = 1e-4  # Tăng weight decay để tránh overfitting
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CHECKPOINT_DIR = "checkpoints"
CHECKPOINT = os.path.join(CHECKPOINT_DIR, "best_model_velocity.pth")  # Dùng checkpoint mới cho phiên bản với velocity
RESUME_CHECKPOINT = os.path.join(CHECKPOINT_DIR, "interim_model_epoch15.pth")  # Dùng checkpoint mới cho phiên bản với velocity
LOG_DIR = "logs"
USE_MIXED_PRECISION = True
RESUME_TRAINING = True  # Continue from checkpoint
BOUNDARY_THRESHOLD = 0.11  # Giảm từ 0.15 xuống 0.08
DEBUG_DETECTION = True  # Enable detection debugging
MAX_GRAD_NORM = 7.0  # Tăng từ 1.0 lên 5.0 để ổn định gradient
FINAL_EVALUATION = True  # Chạy đánh giá cuối cùng trên test set
WARMUP_EPOCHS = 7  # Giảm số epoch cho learning rate warmup từ 5 xuống 3
WARMUP_FACTOR = 2.5  # Tăng LR trong warmup lên 2.5x (tăng từ 2.0)

# Sử dụng threshold riêng cho từng lớp - điều chỉnh theo đặc tính của từng action class
CLASS_THRESHOLDS = [0.15, 0.15, 0.01, 0.08, 0.15]  # Giảm thresholds để phù hợp với phân phối xác suất hiện tại

# Trọng số cho loss components - điều chỉnh để tập trung mạnh hơn vào action classification
ACTION_WEIGHT = 1.5  # Tăng từ 2.0 lên 3.0 để tập trung hơn vào action classification
START_WEIGHT = 1.5  # Giảm từ 1.5 xuống 1.0
END_WEIGHT = 1.5  # Giảm từ 1.5 xuống 1.0

# ====== THÊM CONFIG CHO POST-PROCESSING ======
MIN_SEGMENT_LENGTH = 3  # Độ dài tối thiểu của segment (frames) - giảm từ 10 xuống 8
MIN_CONFIDENT_RATIO = 0.15  # Giảm từ 0.2 xuống 0.15 để cho phép nhiều detections hơn
NMS_THRESHOLD = 0.4  # Tăng từ 0.3 lên 0.5

# Tạo thư mục lưu trữ
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)


def set_seed(seed=42):
    """Đặt seed cho tất cả các generator ngẫu nhiên để đảm bảo tính tái lập"""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # if you use multi-GPU
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    print(f"Đã thiết lập seed={seed} cho tất cả generator ngẫu nhiên")

# ====== Debug Functions ======
def debug_detection_stats(batch_detections, batch_size, metadata):
    """Print detection statistics for debugging"""
    total_dets = sum(len(dets) for dets in batch_detections)
    if total_dets == 0:
        print("⚠️ WARNING: No detections in batch!")
        return
    
    print(f"Detections in batch: {total_dets} (avg {total_dets/batch_size:.1f} per sample)")
    
    # Count detections per class
    class_counts = {}
    for i, dets in enumerate(batch_detections):
        video_id = metadata[i]["video_id"] if i < len(metadata) else "unknown"
        print(f"  Sample {i} (video {video_id}): {len(dets)} detections")
        
        for det in dets:
            action_id = det["action_id"]
            if action_id not in class_counts:
                class_counts[action_id] = 0
            class_counts[action_id] += 1
    
    # Print class statistics
    for action_id, count in sorted(class_counts.items()):
        print(f"  Class {action_id}: {count} detections")
    
    # Print detection details for first few detections
    if total_dets > 0:
        print("\nDetection details (first 3):")
        count = 0
        for i, dets in enumerate(batch_detections):
            if len(dets) > 0:
                for det in dets[:min(3, len(dets))]:
                    print(f"  Det {count}: Class {det['action_id']}, Start: {det['start_frame']}, End: {det['end_frame']}, Conf: {det['confidence']:.4f}")
                    count += 1
                    if count >= 3:
                        break
            if count >= 3:
                break

def debug_raw_predictions(action_probs, start_probs, end_probs):
    """Analyze raw prediction values before thresholding"""
    # Check variance in predictions (helpful to detect potential collapse)
    action_variance = torch.var(action_probs).item()
    print(f"Action prediction variance: {action_variance:.6f}")
    
    # Check per-class stats
    for c in range(action_probs.shape[2]):  # For each class
        class_probs = action_probs[:, :, c]
        print(f"  Class {c}: min={class_probs.min().item():.4f}, max={class_probs.max().item():.4f}, mean={class_probs.mean().item():.4f}", end = " - ")
    print("\n")

# ====== Loss Functions ======
class ActionDetectionLoss(nn.Module):
    def __init__(self, action_weight=ACTION_WEIGHT, start_weight=START_WEIGHT, end_weight=END_WEIGHT, label_smoothing=0.1):
        """
        Loss function for temporal action detection
        
        Args:
            action_weight: Trọng số cho action segmentation loss
            start_weight: Trọng số cho start point detection loss
            end_weight: Trọng số cho end point detection loss
            label_smoothing: Amount of label smoothing to apply (0.0 to 0.5)
        """
        super().__init__()
        self.action_weight = action_weight
        self.start_weight = start_weight
        self.end_weight = end_weight
        self.label_smoothing = label_smoothing
        
        # BCE for action segmentation
        self.action_criterion = nn.BCEWithLogitsLoss(reduction='none')
        
        # BCE for start/end detection
        self.boundary_criterion = nn.BCEWithLogitsLoss(reduction='none')
        
        # Trọng số cho từng lớp (tăng trọng số cho Class 2 và 3)
        self.class_weights = torch.ones(NUM_CLASSES, device=DEVICE)
        self.class_weights[0] = 1.0 
        self.class_weights[1] = 1.5 
        self.class_weights[2] = 7.0  # Tăng mạnh từ 3.3 lên 7.0
        self.class_weights[3] = 2.0  # Thêm trọng số cho Class 3
        self.class_weights[4] = 1.0 

        # Trọng số riêng cho boundary loss của từng lớp
        self.boundary_weights = torch.ones(NUM_CLASSES, device=DEVICE)
        # Tăng mạnh trọng số boundary cho Class 2
        self.boundary_weights[2] = 5.0  # Thêm trọng số đáng kể cho boundary Class 2

    def smooth_labels(self, targets):
        """Apply label smoothing to targets"""
        if self.label_smoothing <= 0:
            return targets
        return targets * (1.0 - self.label_smoothing) + 0.5 * self.label_smoothing
        
    def forward(self, predictions, targets):
        """
        Calculate loss for temporal action detection
        
        Args:
            predictions: dict với keys 'action_scores', 'start_scores', 'end_scores'
                  - action_scores: (B, T, C) - scores cho mỗi frame và class
                  - start_scores: (B, T, C) - scores cho start points
                  - end_scores: (B, T, C) - scores cho end points
                  
            targets: dict với keys 'action_masks', 'start_masks', 'end_masks'
                  - action_masks: (B, C, T) - binary masks cho actions
                  - start_masks: (B, C, T) - Gaussian smoothed masks cho start points
                  - end_masks: (B, C, T) - Gaussian smoothed masks cho end points
            
        Returns:
            Dict với 'total' loss và các individual components
        """
        action_scores = predictions['action_scores']  # (B, T, C)
        start_scores = predictions['start_scores']    # (B, T, C)
        end_scores = predictions['end_scores']        # (B, T, C)
        
        action_masks = targets['action_masks']  # (B, C, T)
        start_masks = targets['start_masks']    # (B, C, T) - Đã áp dụng Gaussian smoothing trong dataloader
        end_masks = targets['end_masks']        # (B, C, T) - Đã áp dụng Gaussian smoothing trong dataloader
        
        # Transpose targets to match predictions
        action_masks = action_masks.transpose(1, 2)  # (B, T, C)
        start_masks = start_masks.transpose(1, 2)    # (B, T, C)
        end_masks = end_masks.transpose(1, 2)        # (B, T, C)
        
        # Apply label smoothing if enabled
        if self.label_smoothing > 0:
            action_masks = self.smooth_labels(action_masks)
            start_masks = self.smooth_labels(start_masks)
            end_masks = self.smooth_labels(end_masks)
        
        # Calculate action loss với class weights
        action_loss = self.action_criterion(action_scores, action_masks)
        
        # Áp dụng class weights
        action_loss = action_loss * self.class_weights.view(1, 1, -1)
        action_loss = action_loss.mean()
        
        # Only consider regions with actions for start/end loss
        # Tạo mask cho những frame có ít nhất một action
        valid_regions = (action_masks.sum(dim=2, keepdim=True) > 0).float()
        
        # Calculate start/end loss với class weights
        start_loss = self.boundary_criterion(start_scores, start_masks)
        end_loss = self.boundary_criterion(end_scores, end_masks)
        
        # Áp dụng class weights cho start/end loss
        start_loss = start_loss * self.boundary_weights.view(1, 1, -1) # Sử dụng boundary_weights
        end_loss = end_loss * self.boundary_weights.view(1, 1, -1)   # Sử dụng boundary_weights
        
        # Apply valid regions mask and normalize
        if valid_regions.sum() > 0:
            # Chỉ tính loss trên những frame có action
            start_loss = (start_loss * valid_regions).sum() / (valid_regions.sum() + 1e-6)
            end_loss = (end_loss * valid_regions).sum() / (valid_regions.sum() + 1e-6)
        else:
            # Fallback khi không có frame nào có action
            start_loss = start_loss.mean()
            end_loss = end_loss.mean()
        
        # Total loss với weighted components
        total_loss = (
            self.action_weight * action_loss + 
            self.start_weight * start_loss + 
            self.end_weight * end_loss
        )
        
        return {
            'total': total_loss,
            'action': action_loss.detach(),
            'start': start_loss.detach(),
            'end': end_loss.detach()
        }

def calculate_f1_at_iou(gt_segments, pred_segments, iou_threshold):
    """Calculate F1 score at a specific IoU threshold"""
    if not pred_segments:
        return 0.0, 0.0, 0.0  # Precision, Recall, F1
    
    # Sắp xếp predictions theo confidence
    pred_segments = sorted(pred_segments, key=lambda x: x['score'], reverse=True)
    
    true_positives = 0
    gt_matched = [False] * len(gt_segments)
    
    for pred in pred_segments:
        pred_segment = pred['segment']
        best_iou = 0
        best_idx = -1
        
        # Tìm GT có IoU cao nhất với prediction này
        for i, gt_segment in enumerate(gt_segments):
            if not gt_matched[i]:  # Chỉ xét GT chưa được match
                iou = calculate_temporal_iou(pred_segment, gt_segment)
                if iou > best_iou:
                    best_iou = iou
                    best_idx = i
        
        # Nếu IoU >= threshold, đánh dấu là true positive
        if best_iou >= iou_threshold and best_idx >= 0:
            true_positives += 1
            gt_matched[best_idx] = True
    
    # Tính precision, recall, F1
    precision = true_positives / len(pred_segments) if pred_segments else 0
    recall = true_positives / len(gt_segments) if gt_segments else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    return precision, recall, f1

def resolve_cross_class_overlaps(merged_detections):
    """Giải quyết chồng lấp giữa các lớp sau khi merge, không cho phép bất kỳ frame nào bị chồng lấn"""
    for video_id, detections in merged_detections.items():
        # Sắp xếp theo confidence giảm dần
        detections = sorted(detections, key=lambda x: x['confidence'], reverse=True)
        
        # Khởi tạo mảng detections mới không có chồng lấp
        final_detections = []
        
        # Tìm frame cuối cùng trong tất cả detections
        max_frame = max([det['end_frame'] for det in detections]) if detections else 0
        frames_occupied = [False] * (max_frame + 1)
        
        for det in detections:
            start = det['start_frame']
            end = det['end_frame']
            
            # Kiểm tra overlap - nếu bất kỳ frame nào đã bị chiếm, bỏ qua detection này
            overlap = False
            for t in range(start, end):
                if t < len(frames_occupied) and frames_occupied[t]:
                    overlap = True
                    break
            
            if not overlap:
                # Thêm detection và đánh dấu tất cả frame của nó là đã chiếm
                for t in range(start, end):
                    if t < len(frames_occupied):
                        frames_occupied[t] = True
                final_detections.append(det)
        
        # Cập nhật lại danh sách detections cho video này
        merged_detections[video_id] = final_detections
    
    return merged_detections

def merge_cross_window_detections(all_window_detections, all_window_metadata, iou_threshold=0.2, confidence_threshold=0.15):
    """
    Kết hợp detections từ các sliding window chồng lấp
    
    Args:
        all_window_detections: List các detections từ mỗi window [window_idx][detection_idx]
        all_window_metadata: Thông tin về mỗi window (video_id, start_idx, end_idx)
        iou_threshold: Ngưỡng IoU để kết hợp các detections liên tiếp
        confidence_threshold: Ngưỡng confidence để chấp nhận kết hợp
        
    Returns:
        merged_detections: Danh sách các detections đã kết hợp xuyên window
    """
    # Tổ chức các detections theo video_id và action_id
    video_detections = defaultdict(lambda: defaultdict(list))
    
    for window_idx, (window_dets, meta) in enumerate(zip(all_window_detections, all_window_metadata)):
        video_id = meta['video_id']
        start_idx = meta['start_idx']  # Vị trí bắt đầu của window trong video
        
        for det in window_dets:
            action_id = det['action_id']
            # Chuyển coordinates từ window-relative sang global video coordinates
            global_start = start_idx + det['start_frame']
            global_end = start_idx + det['end_frame']
            confidence = det['confidence']
            
            video_detections[video_id][action_id].append({
                'start_frame': global_start,
                'end_frame': global_end,
                'confidence': confidence,
                'window_idx': window_idx
            })
    
    # Kết hợp các detections thuộc cùng một action trong mỗi video
    merged_results = {}
    for video_id, action_dets in video_detections.items():
        merged_results[video_id] = []
        
        for action_id, dets in action_dets.items():
            # Sắp xếp theo vị trí bắt đầu
            dets = sorted(dets, key=lambda x: x['start_frame'])
            
            # Kết hợp các detections bị ngắt do window size
            i = 0
            while i < len(dets):
                current = dets[i]
                merged = dict(current)  # Copy để không thay đổi detection gốc
                
                j = i + 1
                while j < len(dets):
                    next_det = dets[j]
                    
                    # Kiểm tra xem hai detections có khả năng là một hành động bị cắt không
                    overlap = min(merged['end_frame'], next_det['end_frame']) - max(merged['start_frame'], next_det['start_frame'])
                    overlap_ratio = overlap / min(merged['end_frame'] - merged['start_frame'], next_det['end_frame'] - next_det['start_frame'])
                    
                    time_diff = abs(next_det['start_frame'] - merged['end_frame'])
                    
                    # Điều kiện để kết hợp: có overlap hoặc cách nhau không quá xa
                    if (overlap_ratio >= iou_threshold or time_diff <= 5) and \
                       (merged['confidence'] + next_det['confidence']) / 2 >= confidence_threshold:
                        # Mở rộng detection hiện tại
                        merged['start_frame'] = min(merged['start_frame'], next_det['start_frame'])
                        merged['end_frame'] = max(merged['end_frame'], next_det['end_frame'])
                        merged['confidence'] = (merged['confidence'] * (merged['end_frame'] - merged['start_frame']) + 
                                             next_det['confidence'] * (next_det['end_frame'] - next_det['start_frame'])) / \
                                             ((merged['end_frame'] - merged['start_frame']) + 
                                             (next_det['end_frame'] - next_det['start_frame']))
                        dets.pop(j)  # Loại bỏ detection đã kết hợp
                    else:
                        j += 1
                
                merged_results[video_id].append({
                    'action_id': action_id,
                    'start_frame': merged['start_frame'],
                    'end_frame': merged['end_frame'],
                    'confidence': merged['confidence']
                })
                
                i += 1
    
    return merged_results

# ====== Tính IoU between prediction and ground truth ======
def calculate_temporal_iou(pred_segment, gt_segment):
    """Calculate temporal IoU between prediction and ground truth segments"""
    pred_start, pred_end = pred_segment
    gt_start, gt_end = gt_segment
    
    # Đảm bảo end > start cho cả predicted và ground truth segments
    pred_start, pred_end = min(pred_start, pred_end), max(pred_start, pred_end)
    gt_start, gt_end = min(gt_start, gt_end), max(gt_start, gt_end)
    
    # Kiểm tra trường hợp segments bị thoái hóa
    if pred_start == pred_end or gt_start == gt_end:
        return 0.0
    
    # Calculate intersection
    intersection = max(0, min(pred_end, gt_end) - max(pred_start, gt_start))
    
    # Calculate union
    union = max(1, (pred_end - pred_start) + (gt_end - gt_start) - intersection)
    
    return intersection / union

# ====== Training Function ======
def train(model, train_loader, val_loader, criterion, optimizer, scheduler, epochs, device, start_epoch=0, best_map=0):
    """Train the model"""
    initial_action_weight = ACTION_WEIGHT  # Lưu trọng số ban đầu
    initial_start_weight = START_WEIGHT
    initial_end_weight = END_WEIGHT
    start_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(LOG_DIR, f"training_log_fixed_{start_time}.csv")
    
    # Write header to log file - ADDED NEW METRICS
    with open(log_file, 'w') as f:
        f.write("epoch,train_loss,val_loss,val_map,val_f1,map_mid,f1_iou_01,f1_iou_025,f1_iou_05,class0_ap,class1_ap,class2_ap,class3_ap,class4_ap\n")
    
    losses = {'train': [], 'val': []}
    maps = []
    class_aps = {c: [] for c in range(NUM_CLASSES)}
    
    # Khởi tạo GradScaler cho mixed precision
    scaler = GradScaler(enabled=USE_MIXED_PRECISION)
    
    for epoch in range(start_epoch, epochs):
        if epoch >= 30:  # Tăng từ 20 lên 30 epochs để tập trung vào action classification lâu hơn
            # Gradually transition to more balanced weights
            progress = min(1.0, (epoch - 30) / 20)  # Transition over 20 epochs 
            criterion.action_weight = initial_action_weight * (1 - 0.3 * progress)  # Giảm dần 30%
            criterion.start_weight = initial_start_weight * (1 + 0.5 * progress)  # Tăng dần 50%
            criterion.end_weight = initial_end_weight * (1 + 0.5 * progress)  # Tăng dần 50%
            print(f"Epoch {epoch+1}: Adjusted weights - Action: {criterion.action_weight:.2f}, Start: {criterion.start_weight:.2f}, End: {criterion.end_weight:.2f}")
        # Training
        model.train()
        train_loss = 0
        
        # Reset gradient accumulation
        optimizer.zero_grad()
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        batch_count = 0
        
        # Theo dõi gradient
        grad_norms = []
        
        # Apply learning rate warmup if in warmup phase
        if epoch < WARMUP_EPOCHS:
            warmup_start = LR/2.5
            current_lr = warmup_start + (LR - warmup_start) * (epoch + 1) / WARMUP_EPOCHS
            for param_group in optimizer.param_groups:
                param_group['lr'] = current_lr
            print(f"Warmup LR: {current_lr:.8f}")
        
        for batch_idx, batch in enumerate(progress_bar):
            # Unpack the batch with RGB and Pose+Velocity streams
            frames, pose_data, hand_data, action_masks, start_masks, end_masks, _ = batch
            
            frames = frames.to(device)
            pose_data = pose_data.to(device)
            action_masks = action_masks.to(device)
            start_masks = start_masks.to(device)
            end_masks = end_masks.to(device)
            


            
            # Forward pass with mixed precision
            with autocast(enabled=USE_MIXED_PRECISION):
                # Forward pass với RGB và Pose+Velocity
                predictions = model(frames, pose_data)
                
                # Debug raw predictions
                if batch_idx % 100 == 0:
                    # For classification model, adjust debugging
                    if 'classification' in predictions:
                        print(f"Classification logits: min={predictions['classification'].min().item():.4f}, max={predictions['classification'].max().item():.4f}")
                    else:
                        action_logits = predictions['action_scores']
                        print("\n")
                        print(f"Action logits: min={action_logits.min().item():.4f}, max={action_logits.max().item():.4f}")
                
                # Calculate loss
                targets = {
                    'action_masks': action_masks,
                    'start_masks': start_masks,
                    'end_masks': end_masks
                }
                
                loss_dict = criterion(predictions, targets)
                loss = loss_dict['total']
                
                # Normalize loss for gradient accumulation
                loss = loss / GRADIENT_ACCUMULATION_STEPS
            
            # Backward pass with gradient scaling
            scaler.scale(loss).backward()
            
            # Update metrics
            train_loss += loss.item() * GRADIENT_ACCUMULATION_STEPS
            batch_count += 1
            
            # Gradient accumulation
            if (batch_idx + 1) % GRADIENT_ACCUMULATION_STEPS == 0 or (batch_idx + 1) == len(train_loader):
                # Unscale gradients before clipping
                scaler.unscale_(optimizer)
                
                # Gradient clipping để ngăn chặn gradient explosion
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=MAX_GRAD_NORM)
                grad_norms.append(grad_norm.item())
                
                # Update weights
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                
                # Update progress bar
                progress_bar.set_postfix({
                    'loss': f"{loss_dict['total'].item():.4f}",
                    'action': f"{loss_dict['action'].item():.4f}",
                    'start': f"{loss_dict['start'].item():.4f}",
                    'end': f"{loss_dict['end'].item():.4f}",
                    'grad': f"{grad_norm:.2f}"
                })
        
        # Print gradient statistics
        if grad_norms:
            print(f"Gradient stats: min={min(grad_norms):.4f}, max={max(grad_norms):.4f}, mean={np.mean(grad_norms):.4f}")
        
        # Average loss
        train_loss /= batch_count
        losses['train'].append(train_loss)
        
        
        
        # Validation - UPDATED to receive metrics dict
        val_metrics = evaluate(model, val_loader, criterion, device)
        val_loss = val_metrics['val_loss']
        val_map = val_metrics['mAP']
        val_f1 = val_metrics['merged_f1']
        class_ap_dict = val_metrics['class_aps']
        map_mid = val_metrics['map_mid']
        avg_f1_iou_01 = val_metrics['avg_f1_iou_010']
        avg_f1_iou_025 = val_metrics['avg_f1_iou_025']
        avg_f1_iou_05 = val_metrics['avg_f1_iou_050']
        
        losses['val'].append(val_loss)
        maps.append(val_map)
        
        # Update learning rate if not in warmup phase
        if epoch >= WARMUP_EPOCHS:
            # Cập nhật scheduler với validation loss
            scheduler.step(val_loss)
            current_lr = optimizer.param_groups[0]['lr']
            print(f"Current LR: {current_lr:.8f}")
        
        # Lưu AP của từng lớp
        for c in range(NUM_CLASSES):
            class_aps[c].append(class_ap_dict[c])
        
        # Log results - UPDATED to print new metrics
        print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val mAP: {val_map:.4f}, Val F1: {val_f1:.4f}")
        print(f"  Extra Metrics: mAP@mid={map_mid:.4f}, F1@0.1={avg_f1_iou_01:.4f}, F1@0.25={avg_f1_iou_025:.4f}, F1@0.5={avg_f1_iou_05:.4f}")
        print(f"  Class AP: {', '.join([f'C{c}={class_ap_dict[c]:.4f}' for c in range(NUM_CLASSES)])}")
        
        # Write to log file - UPDATED to write new metrics
        with open(log_file, 'a') as f:
            f.write(f"{epoch+1},{train_loss},{val_loss},{val_map},{val_f1},{map_mid},{avg_f1_iou_01},{avg_f1_iou_025},{avg_f1_iou_05}")
            for c in range(NUM_CLASSES):
                f.write(f",{class_ap_dict[c]}")
            f.write("\n")
        
        # Save best model
        if val_map > best_map:
            best_map = val_map
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_map': val_map,
                'val_f1': val_f1,
                'class_aps': class_ap_dict
            }, CHECKPOINT)
            print(f"✅ Saved best model with mAP: {val_map:.4f}")
            
            # Checkpoint filename with epoch and mAP
            epoch_checkpoint = os.path.join(CHECKPOINT_DIR, f"model_fixed_epoch{epoch+1}_map{val_map:.4f}.pth")
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_map': val_map,
                'val_f1': val_f1,
                'class_aps': class_ap_dict
            }, epoch_checkpoint)
        
        # Lưu checkpoint định kỳ mỗi 5 epochs hoặc khi có detections đầu tiên
        save_interim = False
        if (epoch + 1) % 1 == 0:  # Lưu mỗi 5 epochs
            save_interim = True
            print(f"💾 Lưu checkpoint định kỳ tại epoch {epoch+1}")
        elif val_map > 0 and best_map == val_map:  # Lưu khi có detection đầu tiên
            save_interim = True
            print(f"🔍 Lưu checkpoint khi có detection đầu tiên: mAP = {val_map:.4f}")
        
        if save_interim:
            interim_checkpoint = os.path.join(CHECKPOINT_DIR, f"interim_model_epoch{epoch+1}.pth")
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_map': val_map,
                'val_f1': val_f1,
                'class_aps': class_ap_dict
            }, interim_checkpoint)
    
    # Plot training curves
    plt.figure(figsize=(15, 10))
    
    plt.subplot(2, 2, 1)
    plt.plot(losses['train'], label='Train Loss')
    plt.plot(losses['val'], label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss')
    
    plt.subplot(2, 2, 2)
    plt.plot(maps, label='Val mAP')
    plt.xlabel('Epoch')
    plt.ylabel('mAP')
    plt.title('Validation mAP')
    plt.legend()
    
    plt.subplot(2, 2, 3)
    for c in range(NUM_CLASSES):
        plt.plot(class_aps[c], label=f'Class {c}')
    plt.xlabel('Epoch')
    plt.ylabel('AP')
    plt.title('Class AP')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(LOG_DIR, f"training_curves_fixed_{start_time}.png"))
    
    return best_map

# ====== Evaluation Function ======
def evaluate(model, val_loader, criterion, device):
    """Evaluate the model"""
    model.eval()
    val_loss = 0

    # Lưu tất cả detections từ mọi window để kết hợp sau
    all_window_detections = []
    all_window_metadata = []
    
    # Các biến để tính mAP
    all_action_gt = {c: [] for c in range(NUM_CLASSES)}
    all_action_preds = {c: [] for c in range(NUM_CLASSES)}
    
    # Biến để tính F1 thông thường (frame-level)
    all_frame_preds = []
    all_frame_targets = []
    
    # Thêm theo dõi false positives
    true_positives = 0
    false_positives = 0
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validation"):
            # Unpack the batch with RGB and Pose+Velocity streams
            frames, pose_data, hand_data, action_masks, start_masks, end_masks, metadata = batch
            
            frames = frames.to(device)
            pose_data = pose_data.to(device)
            action_masks = action_masks.to(device)
            start_masks = start_masks.to(device)
            end_masks = end_masks.to(device)
            
            # Forward pass with mixed precision
            with autocast(enabled=USE_MIXED_PRECISION):
                # Forward pass với RGB và Pose+Velocity
                predictions = model(frames, pose_data)
                
                # Kiểm tra predictions
                if 'classification' in predictions:
                    class_logits = predictions['classification']
                    print(f"Val Classification logits: min={class_logits.min().item():.4f}, max={class_logits.max().item():.4f}")
                
                # Calculate loss
                targets = {
                    'action_masks': action_masks,
                    'start_masks': start_masks,
                    'end_masks': end_masks
                }
                
                loss_dict = criterion(predictions, targets)
                loss = loss_dict['total']
            
            # Update metrics
            val_loss += loss.item()
            
            # Post-process để lấy action segments
            if 'classification' in predictions:
                # For TwoStreamActionNet, use classification scores
                action_probs = torch.sigmoid(predictions['classification']).unsqueeze(1).repeat(1, WINDOW_SIZE, 1)
                start_probs = torch.sigmoid(predictions['start_logits'].unsqueeze(-1).repeat(1, 1, NUM_CLASSES))
                end_probs = torch.sigmoid(predictions['end_logits'].unsqueeze(-1).repeat(1, 1, NUM_CLASSES))
            else:
                # For TemporalActionDetector, use action_scores
                action_probs = torch.sigmoid(predictions['action_scores'])
                start_probs = torch.sigmoid(predictions['start_scores'])
                end_probs = torch.sigmoid(predictions['end_scores'])
            
            # Debug raw probability distributions
            debug_raw_predictions(action_probs, start_probs, end_probs)
            
            # Custom post-process function với threshold riêng cho từng lớp
            batch_detections = custom_post_process(
                model, 
                action_probs,
                start_probs,
                end_probs,
                class_thresholds=CLASS_THRESHOLDS,
                boundary_threshold=BOUNDARY_THRESHOLD,
                nms_threshold=NMS_THRESHOLD
            )
            
            if DEBUG_DETECTION:
                debug_detection_stats(batch_detections, frames.shape[0], metadata)
            
            # Process each sample in batch
            for i, (detections, meta) in enumerate(zip(batch_detections, metadata)):
                window_size = frames.shape[2]  # Temporal dimension
                video_id = meta['video_id']
                start_idx = meta['start_idx']
                all_window_detections.append(detections)
                all_window_metadata.append(meta)
                
                # Extract ground truth segments from annotations
                for anno in meta['annotations']:
                    action_id = anno['action_id']
                    gt_start = anno['start_frame']
                    gt_end = anno['end_frame']
                    
                    # Add to GT segments với window-relative coordinates
                    all_action_gt[action_id].append((gt_start, gt_end))
                
                # Process detections with window-relative coordinates
                for det in detections:
                    action_id = det['action_id']
                    # KHÔNG thêm start_idx để tránh coordinate mismatch
                    start_frame = det['start_frame']
                    end_frame = det['end_frame']
                    confidence = det['confidence']
                    
                    # Đảm bảo end_frame > start_frame
                    if end_frame <= start_frame:
                        end_frame = start_frame + 1
                    
                    # Add to predictions
                    all_action_preds[action_id].append({
                        'segment': (start_frame, end_frame),
                        'score': confidence
                    })
                
                # Process frame-level predictions for F1 score
                processed_preds, processed_targets = process_for_evaluation(
                    detections,
                    meta['annotations'],
                    action_masks[i].cpu(),
                    window_size
                )
                
                all_frame_preds.extend(processed_preds)
                all_frame_targets.extend(processed_targets)
    
    merged_video_detections = merge_cross_window_detections(
    all_window_detections, 
    all_window_metadata,
    iou_threshold=0.2,
    confidence_threshold=0.15
    )
    merged_video_detections = resolve_cross_class_overlaps(merged_video_detections)
    # Tính F1 dựa trên frame-level predictions
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_frame_targets, all_frame_preds, average='macro', zero_division=0
    )
    
    merged_all_action_preds = {c: [] for c in range(NUM_CLASSES)}

    for video_dets in merged_video_detections.values():
        for det in video_dets:
            merged_all_action_preds[det['action_id']].append({
                'segment': (det['start_frame'], det['end_frame']),
                'score': det['confidence']
            })
    # Tạo dictionary để lưu ground truth toàn cục theo video_id và action_id
    global_action_gt = defaultdict(lambda: defaultdict(list))

    # Extract ground truth segments from annotations
    for i, meta in enumerate(all_window_metadata):
        video_id = meta['video_id']
        start_idx = meta['start_idx']  # Window's start index in global coordinates
        
        for anno in meta['annotations']:
            action_id = anno['action_id']
            # Chuyển ground truth sang tọa độ global
            global_gt_start = start_idx + anno['start_frame']
            global_gt_end = start_idx + anno['end_frame']
            
            # Thêm vào ground truth toàn cục
            global_action_gt[video_id][action_id].append((global_gt_start, global_gt_end))

    # Chuyển đổi thành format tương thích với hàm calculate_mAP
    all_action_gt_global = {c: [] for c in range(NUM_CLASSES)}
    for video_id, actions in global_action_gt.items():
        for action_id, segments in actions.items():
            # Loại bỏ các segments trùng lặp
            unique_segments = list(set(segments))
            all_action_gt_global[action_id].extend(unique_segments)

    for c in range(NUM_CLASSES):
        print(f"Class {c} - Global GT count: {len(all_action_gt_global[c])}")
        if all_action_gt_global[c]:
            print(f"  Sample GT: {all_action_gt_global[c][0]}")
        if merged_all_action_preds[c]:
            print(f"  Sample Pred: {merged_all_action_preds[c][0]}")

    # Trực quan hóa kết quả merger
    for video_id, detections in merged_video_detections.items():
        print(f"\nVideo {video_id}: {len(detections)} detections sau khi kết hợp")
        detections = sorted(detections, key=lambda x: (x['action_id'], x['start_frame']))
    
    # Sử dụng merged_all_action_preds thay vì all_action_preds
    mAP = calculate_mAP(all_action_gt_global, merged_all_action_preds)

    merged_all_frame_preds = []
    merged_all_frame_targets = []

    # Thu thập tất cả frames từ các video
    for video_id, detections in merged_video_detections.items():
        # Tìm độ dài video (lấy frame cuối cùng từ detections hoặc ground truth)
        max_frame = 0
        for det in detections:
            max_frame = max(max_frame, det['end_frame'])
        
        for c in range(NUM_CLASSES):
            if video_id in global_action_gt and c in global_action_gt[video_id]:
                for start, end in global_action_gt[video_id][c]:
                    max_frame = max(max_frame, end)
        
        # Tạo mảng frame targets và predictions cho video này
        video_length = max_frame + 1
        video_targets = np.zeros((video_length, NUM_CLASSES), dtype=int)
        video_preds = np.zeros((video_length, NUM_CLASSES), dtype=int)
        
        # Lấp đầy ground truth
        if video_id in global_action_gt:
            for c, segments in global_action_gt[video_id].items():
                for start, end in segments:
                    for t in range(start, end):
                        if t < video_length:
                            video_targets[t, c] = 1
        
        # Lấp đầy predictions
        for det in detections:
            c = det['action_id']
            start = det['start_frame']
            end = det['end_frame']
            for t in range(start, end):
                if t < video_length:
                    video_preds[t, c] = 1
        
        # Chuyển sang dạng flatten để tính F1
        for t in range(video_length):
            for c in range(NUM_CLASSES):
                merged_all_frame_targets.append(video_targets[t, c])
                merged_all_frame_preds.append(video_preds[t, c])

    # Tính global F1
    merged_precision, merged_recall, merged_f1, _ = precision_recall_fscore_support(
        merged_all_frame_targets, merged_all_frame_preds, average='macro', zero_division=0
    )

    print(f"\n--- F1 Metrics ---")
    print(f"Window-level F1: {f1:.4f} (Precision: {precision:.4f}, Recall: {recall:.4f})")
    print(f"Global F1 after merge: {merged_f1:.4f} (Precision: {merged_precision:.4f}, Recall: {merged_recall:.4f})")

    # Tính F1 cho từng lớp riêng biệt
    merged_class_f1 = []
    for c in range(NUM_CLASSES):
        class_targets = [merged_all_frame_targets[i] for i in range(len(merged_all_frame_targets)) 
                        if i % NUM_CLASSES == c]
        class_preds = [merged_all_frame_preds[i] for i in range(len(merged_all_frame_preds))
                    if i % NUM_CLASSES == c]
        
        if sum(class_targets) > 0:  # Chỉ tính F1 cho các lớp có ground truth
            _, _, class_f1, _ = precision_recall_fscore_support(
                class_targets, class_preds, average='binary', zero_division=0
            )
            merged_class_f1.append(class_f1)
            print(f"Class {c} F1: {class_f1:.4f}")



        # Tính segment accuracy (dùng IoU=0.5)
    total_correct = 0
    total_global_gt_segments = sum(len(all_action_gt_global[c]) for c in range(NUM_CLASSES))
    total_pred = sum(len(merged_all_action_preds[c]) for c in range(NUM_CLASSES))

    for c in range(NUM_CLASSES):
        gt_matched = [False] * len(all_action_gt_global[c])
        for pred in merged_all_action_preds[c]:
            best_iou = 0
            best_idx = -1
            for i, gt in enumerate(all_action_gt_global[c]):
                if not gt_matched[i]:
                    iou = calculate_temporal_iou(pred['segment'], gt)
                    if iou > best_iou:
                        best_iou = iou
                        best_idx = i
            
            if best_iou >= 0.5 and best_idx >= 0:
                total_correct += 1
                gt_matched[best_idx] = True


    # Tính trung bình độ dài segment trước và sau khi kết hợp
    avg_length_before = np.mean([det['end_frame'] - det['start_frame'] for dets in all_window_detections for det in dets] if any(all_window_detections) else [0])
    avg_length_after = np.mean([det['end_frame'] - det['start_frame'] for dets in merged_video_detections.values() for det in dets] if any(merged_video_detections.values()) else [0])
    print(f"Avg length: {avg_length_before:.2f} → {avg_length_after:.2f} frames")
    
    # SỬA LẠI DÒNG PRINT THEO Ý NGHĨA MỚI
    total_merged_predictions = sum(len(dets) for dets in merged_video_detections.values())
    print(f"Total Segments: Detected={total_merged_predictions} / GroundTruth={total_global_gt_segments}")

    # (Có thể giữ lại dòng đếm dự đoán dài nếu muốn so sánh riêng)
    # long_actions_detected = sum(1 for det in merged_video_detections.values() for d in det if d['end_frame']-d['start_frame'] > 32)
    
    # Calculate AP cho từng lớp
    class_ap_dict = {}
    print("\n--- mAP by class ---")
    
    for c in range(NUM_CLASSES):
        class_ap = calculate_class_mAP(all_action_gt_global[c], merged_all_action_preds[c])
        class_ap_dict[c] = class_ap
        num_gt = len(all_action_gt_global[c])
        num_pred = len(merged_all_action_preds[c])
        print(f"Class {c}: AP={class_ap:.4f} (GT={num_gt}, Pred={num_pred})")
    
    # Tính số lượng false positives và true positives
    for pred, target in zip(all_frame_preds, all_frame_targets):
        if pred == 1 and target == 1:
            true_positives += 1
        elif pred == 1 and target == 0:
            false_positives += 1
    
    print("\n--- Segment-level F1 at different IoU thresholds ---")
    iou_thresholds = [0.1, 0.25, 0.5]
    avg_f1_scores = {} # Store average F1 for returning
    
    for iou in iou_thresholds:
        all_class_f1 = []
        for c in range(NUM_CLASSES):
            # Check if GT exists for this class to avoid division by zero or misleading F1=0
            if len(all_action_gt_global[c]) == 0: 
                # If no GT, F1 is undefined or arguably 0 if preds exist, 1 if no preds either. 
                # For averaging, skipping might be best unless defined otherwise.
                # Let's skip for averaging to avoid skewing.
                continue # Skip this class if no ground truth
            
            # Tính F1 ở ngưỡng IoU cụ thể
            _, _, class_f1 = calculate_f1_at_iou(all_action_gt_global[c], merged_all_action_preds[c], iou)
            all_class_f1.append(class_f1)
            print(f"Class {c} - F1@{iou:.2f}: {class_f1:.4f}")
        
        avg_f1 = np.mean(all_class_f1) if all_class_f1 else 0.0 # Handle case where no class had GT
        avg_f1_scores[f'avg_f1_iou_{iou:.2f}'.replace('.', '')] = avg_f1 # Store with a key like 'avg_f1_iou_01'
        print(f"Average F1@{iou:.2f}: {avg_f1:.4f}")

    map_mid = calculate_map_mid(all_action_gt_global, merged_all_action_preds)
    print(f"mAP@mid: {map_mid:.4f}")

    accuracy = total_correct / max(1, total_global_gt_segments) # Tránh chia cho 0 nếu không có GT
    print(f"\nSegment Accuracy@0.5: {accuracy:.4f} (Correct={total_correct}, GT={total_global_gt_segments}, Pred={total_pred})")
    
    # In thêm về khả năng phát hiện
    total_frames = len(all_frame_preds)
    print(f"False positives: {false_positives}/{total_frames} ({false_positives/total_frames*100:.2f}%)")

    
    # Tính loss trung bình
    val_loss /= len(val_loader)
    
    # Return a dictionary of all calculated metrics
    return {
        'val_loss': val_loss,
        'mAP': mAP,
        'merged_f1': merged_f1,
        'class_aps': class_ap_dict,
        'map_mid': map_mid,
        'avg_f1_iou_010': avg_f1_scores.get('avg_f1_iou_010', 0.0),
        'avg_f1_iou_025': avg_f1_scores.get('avg_f1_iou_025', 0.0),
        'avg_f1_iou_050': avg_f1_scores.get('avg_f1_iou_050', 0.0) # Use the correct key 'avg_f1_iou_050'
    }

# Custom post-process function với threshold riêng cho từng lớp
def custom_post_process(model, action_probs, start_probs, end_probs, class_thresholds, boundary_threshold=BOUNDARY_THRESHOLD, nms_threshold=NMS_THRESHOLD, min_segment_length=MIN_SEGMENT_LENGTH):
    """Post-processing đơn giản hóa:
    1. Tìm start/end candidates > boundary_threshold.
    2. Tạo tất cả các cặp (start, end) hợp lệ (đúng min_length).
    3. Lọc các cặp dựa trên action_score trung bình > class_threshold.
    4. Tính confidence kết hợp.
    5. Áp dụng NMS cho cùng lớp trong cửa sổ.
    (Loại bỏ xử lý overlap khác lớp trong cửa sổ)
    """
    batch_size, seq_len, num_classes = action_probs.shape
    all_detections_batch = []
    
    for b in range(batch_size):
        detections_window = [] # Detections cho cửa sổ hiện tại
        
        for c in range(num_classes):
            action_score_c = action_probs[b, :, c]  # (T,)
            start_score_c = start_probs[b, :, c]    # (T,)
            end_score_c = end_probs[b, :, c]        # (T,)
            class_threshold_c = class_thresholds[c]

            # 1. Tìm start/end candidates
            start_indices = torch.where(start_score_c > boundary_threshold)[0]
            end_indices = torch.where(end_score_c > boundary_threshold)[0]

            # === ADD DEBUG FOR CLASS 2 ===
            if c == 2:
                print(f"DEBUG Class 2 (window {b}): Max Start Score = {start_score_c.max().item():.4f}, Max End Score = {end_score_c.max().item():.4f}")
                print(f"DEBUG Class 2 (window {b}): Num Start Indices (> {boundary_threshold}) = {len(start_indices)}, Num End Indices (> {boundary_threshold}) = {len(end_indices)}")
            # === END DEBUG ===

            if len(start_indices) == 0 or len(end_indices) == 0:
                continue
            
            proposals_class_c = []
            # 2. Tạo tất cả các cặp (start, end) hợp lệ
            for start_idx_tensor in start_indices:
                start_idx = start_idx_tensor.item()
                # Chỉ xét end_indices sau start_idx
                valid_end_indices = end_indices[end_indices > start_idx]

                for end_idx_tensor in valid_end_indices:
                    end_idx = end_idx_tensor.item()

                    # Kiểm tra độ dài tối thiểu
                    if (end_idx - start_idx) >= min_segment_length:
                        # 3. Lọc dựa trên action score trung bình
                        segment_action_score = action_score_c[start_idx:end_idx].mean().item()

                        if segment_action_score > class_threshold_c:
                            # 4. Tính confidence kết hợp
                            start_conf = start_score_c[start_idx].item()
                            # Lấy điểm end của frame cuối cùng TRONG segment (end_idx là exclusive)
                            # Đảm bảo end_idx-1 không nhỏ hơn start_idx
                            effective_end_idx = max(start_idx, end_idx - 1)
                            end_conf = end_score_c[effective_end_idx].item()

                            confidence = (segment_action_score + start_conf + end_conf) / 3.0

                            proposals_class_c.append({
                        'action_id': c,
                                'start_frame': start_idx,
                                'end_frame': end_idx, # end_idx là exclusive
                                'confidence': confidence
                            })

            # Thêm proposals của lớp này vào danh sách chung của window
            detections_window.extend(proposals_class_c)

        # 5. Áp dụng NMS cho TẤT CẢ detections trong window (chỉ loại bỏ cùng lớp)
        # Sắp xếp trước khi vào NMS để đảm bảo tính nhất quán
        detections_window = sorted(detections_window, key=lambda x: x['confidence'], reverse=True)
        detections_window_nms = nms(detections_window, nms_threshold) # nms chỉ xử lý overlap CÙNG LỚP

        # KHÔNG còn xử lý overlap khác lớp ở đây
        # KHÔNG còn gọi validate_detections ở đây

        all_detections_batch.append(detections_window_nms)

    return all_detections_batch

def nms(detections, threshold):
    """Non-maximum suppression for action detections (cho cùng một lớp)"""
    if not detections:
        return []
    
    # Sort by confidence (descending)
    detections = sorted(detections, key=lambda x: x['confidence'], reverse=True)
    
    # Apply NMS
    keep = []
    while detections:
        current = detections.pop(0)
        keep.append(current)
        
        # Remove overlapping detections with IoU > threshold
        detections = [
            d for d in detections if 
            calculate_temporal_iou(
                (current['start_frame'], current['end_frame']),
                (d['start_frame'], d['end_frame'])
            ) <= threshold
        ]
    
    return keep

def calculate_class_mAP(gt_segments, pred_segments, iou_threshold=0.5):
    """Calculate AP for a single class"""
    # Handle edge cases
    if len(gt_segments) == 0:
        return 0.0  # No ground truth -> cannot compute AP
    
    if len(pred_segments) == 0:
        return 0.0  # No predictions -> precision is 0
    
    # Sort predictions by confidence
    pred_segments = sorted(pred_segments, key=lambda x: x['score'], reverse=True)
    
    # Mark GT segments as detected or not
    gt_detected = [False] * len(gt_segments)
    
    # Arrays for AP calculation
    y_true = []
    y_score = []
    
    # For each prediction
    for pred in pred_segments:
        pred_segment = pred['segment']
        score = pred['score']
        
        # Add score to y_score
        y_score.append(score)
        
        # Find best matching GT segment
        max_iou = 0
        max_idx = -1
        
        for i, gt_segment in enumerate(gt_segments):
            if gt_detected[i]:
                continue  # Skip already detected GT segments
            
            iou = calculate_temporal_iou(pred_segment, gt_segment)
            if iou > max_iou:
                max_iou = iou
                max_idx = i
        
        # Match if IoU >= threshold
        if max_iou >= iou_threshold and max_idx >= 0:
            gt_detected[max_idx] = True
            y_true.append(1)  # True positive
        else:
            y_true.append(0)  # False positive
    
    # Calculate AP
    if sum(y_true) > 0:
        ap = average_precision_score(y_true, y_score)
    else:
        ap = 0.0
    
    return ap

def calculate_map_mid(all_action_gt, all_action_preds):
    """Calculate mAP with midpoint criterion instead of IoU"""
    aps = []
    
    for action_id in range(NUM_CLASSES):
        gt_segments = all_action_gt[action_id]
        pred_segments = all_action_preds[action_id]
        
        # Skip if no ground truth
        if len(gt_segments) == 0:
            continue
        
        # Sort predictions by confidence
        pred_segments = sorted(pred_segments, key=lambda x: x['score'], reverse=True)
        
        # Arrays for AP calculation
        y_true = []
        y_score = []
        
        # Đánh dấu GT segments đã được phát hiện
        gt_detected = [False] * len(gt_segments)
        
        for pred in pred_segments:
            pred_segment = pred['segment']
            score = pred['score']
            
            # Tính midpoint
            pred_mid = (pred_segment[0] + pred_segment[1]) / 2
            
            # Thêm score
            y_score.append(score)
            
            # Kiểm tra xem midpoint có nằm trong bất kỳ GT nào
            is_correct = False
            for i, gt_segment in enumerate(gt_segments):
                if not gt_detected[i] and gt_segment[0] <= pred_mid <= gt_segment[1]:
                    gt_detected[i] = True
                    is_correct = True
                    break
            
            y_true.append(1 if is_correct else 0)
        
        # Tính AP
        if sum(y_true) > 0:  # Nếu có ít nhất 1 detection đúng
            ap = average_precision_score(y_true, y_score)
        else:
            ap = 0.0
        
        aps.append(ap)
    
    # Tính mAP
    return np.mean(aps) if aps else 0.0

def calculate_mAP(all_action_gt, all_action_preds, iou_thresholds=[0.3, 0.5, 0.7]):
    """Calculate mean Average Precision across classes and IoU thresholds"""
    # Calculate AP for each class and IoU threshold
    aps = []
    
    for action_id in range(NUM_CLASSES):
        gt_segments = all_action_gt[action_id]
        pred_segments = all_action_preds[action_id]
        
        # Skip if no ground truth
        if len(gt_segments) == 0:
            continue
        
        # Calculate AP for each IoU threshold
        class_aps = []
        for iou_threshold in iou_thresholds:
            ap = calculate_class_mAP(gt_segments, pred_segments, iou_threshold)
            class_aps.append(ap)
        
        # Average AP across IoU thresholds
        aps.append(np.mean(class_aps))
    
    # Calculate mAP
    return np.mean(aps) if aps else 0.0

def process_for_evaluation(detections, gt_annotations, action_masks, window_size):
    """Process predictions and ground truth for evaluation metrics"""
    processed_preds = []
    processed_targets = []
    
    # Process each frame
    for t in range(window_size):
        for c in range(NUM_CLASSES):
            # Check if frame t belongs to class c in ground truth
            is_gt_action = False
            for anno in gt_annotations:
                if anno['action_id'] == c and anno['start_frame'] <= t < anno['end_frame']:
                    is_gt_action = True
                    break
            
            # Check if frame t belongs to class c in predictions
            is_pred_action = False
            for det in detections:
                if det['action_id'] == c and det['start_frame'] <= t < det['end_frame']:
                    is_pred_action = True
                    break
            
            processed_preds.append(1 if is_pred_action else 0)
            processed_targets.append(1 if is_gt_action else 0)
    
    return processed_preds, processed_targets

# ====== Main Function ======
def main():
    """Main training function"""
    print(f"Using device: {DEVICE}")
    
    # Log GPU memory stats
    if torch.cuda.is_available():
        set_seed(42) 
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Total GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        print(f"Mixed precision: {'Enabled' if USE_MIXED_PRECISION else 'Disabled'}")
        print(f"Detection thresholds: Boundary={BOUNDARY_THRESHOLD}")
        print(f"Class-specific thresholds: {CLASS_THRESHOLDS}")
        print(f"Loss weights: Action={ACTION_WEIGHT}, Start={START_WEIGHT}, End={END_WEIGHT}")
        print(f"Class 2 weight: {2.0}x, Class 3 weight: {2.0}x")
        print(f"Learning rate: {LR}, Peak warmup: {LR*WARMUP_FACTOR}, Weight decay: {WEIGHT_DECAY}")
        print(f"Gradient clipping: {MAX_GRAD_NORM}")
        print(f"Warmup epochs: {WARMUP_EPOCHS}")
        print(f"MIN_CONFIDENT_RATIO: {MIN_CONFIDENT_RATIO}")
    
    # Get dataloaders
    train_loader = get_train_loader(batch_size=BATCH_SIZE)
    val_loader = get_val_loader(batch_size=BATCH_SIZE)  # Use validation loader instead of test
    
    # Initialize model - use TemporalActionDetector for multi-stream processing
    model = TemporalActionDetector(num_classes=NUM_CLASSES, window_size=WINDOW_SIZE)
    
    # Chuyển model to device trước
    model = model.to(DEVICE)
    
    # Initialize optimizer and scheduler
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY, eps=1e-4)
    
    # Thay đổi scheduler từ CosineAnnealingLR sang ReduceLROnPlateau
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',        # Giảm LR khi metric giảm
        factor=0.2,        # Giảm LR 50% mỗi lần
        patience=3,        # Đợi 3 epochs không cải thiện
        min_lr=1e-6,       # LR tối thiểu
        verbose=True       # In thông báo khi LR thay đổi
    )
    
    # Khởi tạo training state
    start_epoch = 0
    best_map = 0
    
    # Resume training nếu RESUME_TRAINING=True và file checkpoint tồn tại
    if RESUME_TRAINING and os.path.exists(RESUME_CHECKPOINT):
        print(f"Resuming training from checkpoint: {RESUME_CHECKPOINT}")
        checkpoint = torch.load(RESUME_CHECKPOINT, map_location=DEVICE)
        
        # Load model state
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # Đảm bảo tất cả params đều ở GPU sau khi load state dict
        model = model.to(DEVICE)
        
        # Khởi tạo optimizer mới trên cùng device
        optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY, eps=1e-4)
        
        # Load optimizer state
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # Chuyển tất cả state tensors của optimizer lên GPU
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(DEVICE)
        
        # Khởi tạo scheduler mới
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.2,
            patience=3,
            min_lr=1e-6,
            verbose=True
        )
        # Load scheduler state if available in checkpoint
        if 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            print("Loaded scheduler state from checkpoint.")
        else:
            print("Scheduler state not found in checkpoint, initializing new scheduler.")
        
        # Set training state
        start_epoch = checkpoint['epoch']
        best_map = checkpoint['val_map']
        
        print(f"Loaded checkpoint from epoch {start_epoch} with mAP: {best_map:.4f}")
    else:
        print("No checkpoint found, starting from scratch")
    
    # Print model summary
    print(f"Model: TemporalActionDetector with {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M parameters")
    print(f"Streams: RGB + Pose with Velocity (198 dims, bỏ Hand stream)")
    print(f"Scheduler: ReduceLROnPlateau (factor=0.5, patience=3, min_lr=1e-6)")
    
    # Initialize loss function với class weights và label smoothing
    criterion = ActionDetectionLoss(action_weight=ACTION_WEIGHT, start_weight=START_WEIGHT, 
                                   end_weight=END_WEIGHT, label_smoothing=0.1)
    
    
    # Train model
    try:
        best_map = train(model, train_loader, val_loader, criterion, optimizer, scheduler, EPOCHS, DEVICE, 
                          start_epoch=start_epoch, best_map=best_map)
        print(f"\n✅ Training complete! Best validation mAP: {best_map:.4f}")
        print(f"Best model saved to {CHECKPOINT}")
        
        # Final evaluation on test set if requested
        if FINAL_EVALUATION:
            print("\n=== Final Evaluation on Test Set ===")
            # Load best model
            checkpoint = torch.load(CHECKPOINT, map_location=DEVICE)
            model.load_state_dict(checkpoint['model_state_dict'])
            model.eval()
            
            # Get test loader
            test_loader = get_test_loader(batch_size=BATCH_SIZE)
            
            # Evaluate on test set
            test_loss, test_map, test_f1, test_class_ap_dict = evaluate(model, test_loader, criterion, DEVICE)
            
            print(f"Test Loss: {test_loss:.4f}, Test mAP: {test_map:.4f}, Test F1: {test_f1:.4f}")
            print(f"Test Class AP: {', '.join([f'C{c}={test_class_ap_dict[c]:.4f}' for c in range(NUM_CLASSES)])}")
            
            # Save test results
            test_results = {
                'test_map': test_map,
                'test_f1': test_f1,
                'test_loss': test_loss,
                'class_aps': test_class_ap_dict
            }
            
            with open(os.path.join(LOG_DIR, 'test_results.json'), 'w') as f:
                json.dump(test_results, f, indent=2)
                
    except Exception as e:
        print(f"\n❌ Error during training: {str(e)}")
        import traceback
        traceback.print_exc()
        # Try to free up memory
        torch.cuda.empty_cache()
        raise e

if __name__ == "__main__":
    torch.cuda.empty_cache()  # Clear cache before starting
    main() 
