import torch
import torch.optim as optim
from src.models.base_detector import TemporalActionDetector
from src.dataloader import get_train_loader, get_val_loader, get_test_loader
from tqdm import tqdm
import numpy as np
import os
import json
from datetime import datetime
from torch.cuda.amp import autocast, GradScaler
from src.utils.helpers import set_seed, process_for_evaluation
from src.utils.debugging import debug_detection_stats, debug_raw_predictions
from src.losses import ActionDetectionLoss
from src.utils.postprocessing import post_process
from src.evaluation import compute_final_metrics


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
    
    
    return best_map

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
            
            # debug_raw_predictions(action_probs)
            
            #post-process function với threshold riêng cho từng lớp
            batch_detections = post_process(
                model, 
                action_probs,
                start_probs,
                end_probs,
                class_thresholds=CLASS_THRESHOLDS,
                boundary_threshold=BOUNDARY_THRESHOLD,
                nms_threshold=NMS_THRESHOLD
            )
            
            # if DEBUG_DETECTION:
            #     debug_detection_stats(batch_detections, frames.shape[0], metadata)
            
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
                    window_size,
                    NUM_CLASSES
                )
                
                all_frame_preds.extend(processed_preds)
                all_frame_targets.extend(processed_targets)
    
    # Tính toán các metrics cuối cùng
    final_metrics = compute_final_metrics(
        all_window_detections,
        all_window_metadata,
        all_frame_preds,
        all_frame_targets,
        NUM_CLASSES
    )

    # Tính loss trung bình
    val_loss /= len(val_loader)
    
    # Return a dictionary of all calculated metrics
    return {
        'val_loss': val_loss,
        'mAP': final_metrics['mAP'],
        'merged_f1': final_metrics['merged_f1'],
        'class_aps': final_metrics['class_aps'],
        'map_mid': final_metrics['map_mid'],
        'avg_f1_iou_010': final_metrics['avg_f1_iou_010'],
        'avg_f1_iou_025': final_metrics['avg_f1_iou_025'],
        'avg_f1_iou_050': final_metrics['avg_f1_iou_050']
    }

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
                                   end_weight=END_WEIGHT, device=DEVICE, num_classes=NUM_CLASSES, label_smoothing=0.1)
    
    
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
