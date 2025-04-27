import torch
import torch.optim as optim
from tqdm import tqdm
import numpy as np
import os
import json
from datetime import datetime
from torch.cuda.amp import autocast, GradScaler
import yaml # Import YAML
from pathlib import Path # Import Path
import argparse # Import argparse
from collections import defaultdict

# Import components from src
# Ensure model import is correct
from src.models.base_detector import TemporalActionDetector 
from src.dataloader import get_train_loader, get_val_loader, get_test_loader
from src.utils.helpers import set_seed, calculate_global_gt # Assuming process_for_evaluation moved or handled by compute_final_metrics
from src.utils.debugging import debug_detection_stats, debug_raw_predictions
from src.losses import ActionDetectionLoss
# Assuming post_process is now part of the model or called within compute_final_metrics
# from src.utils.postprocessing import post_process # Remove if not called directly
from src.evaluation import compute_final_metrics


# Removed the large block of hardcoded CONFIG constants

def train(
    model, train_loader, val_loader, criterion, optimizer, scheduler, 
    cfg, # Pass the relevant config section (e.g., base_model_training)
    device, start_epoch=0, best_map=0
):
    """Train the model using parameters from config."""
    
    # Extract necessary training parameters from cfg
    epochs = cfg['epochs']
    log_dir = Path(cfg['data']['logs'])
    checkpoint_dir = Path(cfg['data']['base_model_checkpoints'])
    best_checkpoint_path = checkpoint_dir / cfg['data']['base_best_checkpoint_name']
    use_mixed_precision = cfg['use_mixed_precision']
    gradient_accumulation_steps = cfg['gradient_accumulation_steps']
    max_grad_norm = cfg['gradient_clipping']['max_norm']
    warmup_epochs = cfg['warmup']['epochs']
    base_lr = cfg['optimizer']['lr']
    warmup_factor = cfg['warmup']['factor']
    # Loss weights might be adjusted, get initial values
    initial_loss_weights = cfg['loss']
    # Debug setting
    debug_detection_enabled = cfg['debugging']['debug_detection_enabled']
    
    # --- Initialization ---
    # Dynamic weight adjustment setup (optional, based on config?)
    # adjust_weights = cfg.get('adjust_loss_weights_during_train', False) # Example: make it configurable
    adjust_weights = True # Keep current behavior for now
    if adjust_weights:
        initial_action_weight = initial_loss_weights['action_weight']
        initial_start_weight = initial_loss_weights['start_weight']
        initial_end_weight = initial_loss_weights['end_weight']
        
    start_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"training_log_base_{start_time}.csv"
    log_dir.mkdir(parents=True, exist_ok=True) # Ensure log dir exists
    
    # Write header to log file
    with open(log_file, 'w') as f:
        f.write("epoch,train_loss,val_loss,val_map,val_f1,map_mid,f1_iou_010,f1_iou_025,f1_iou_050,class0_ap,class1_ap,class2_ap,class3_ap,class4_ap\n")
    
    losses = {'train': [], 'val': []}
    maps = []
    # Get num_classes from the main config section if needed here, or pass it
    num_classes = cfg['global']['num_classes'] 
    class_aps = {c: [] for c in range(num_classes)}
    
    scaler = GradScaler(enabled=use_mixed_precision)
    
    # --- Training Loop ---
    for epoch in range(start_epoch, epochs):
        # Optional: Adjust loss weights dynamically
        if adjust_weights and epoch >= 30: 
            progress = min(1.0, (epoch - 30) / 20)  
            criterion.action_weight = initial_action_weight * (1 - 0.3 * progress)  
            criterion.start_weight = initial_start_weight * (1 + 0.5 * progress)  
            criterion.end_weight = initial_end_weight * (1 + 0.5 * progress)  
            print(f"Epoch {epoch+1}: Adjusted weights - Action: {criterion.action_weight:.2f}, Start: {criterion.start_weight:.2f}, End: {criterion.end_weight:.2f}")
        
        # Apply learning rate warmup
        if epoch < warmup_epochs:
            warmup_start = base_lr / warmup_factor # Adjust start LR based on factor
            current_lr = warmup_start + (base_lr - warmup_start) * (epoch + 1) / warmup_epochs
            for param_group in optimizer.param_groups:
                param_group['lr'] = current_lr
            print(f"Warmup LR: {current_lr:.8f}")
        
        # --- Train One Epoch ---
        model.train()
        train_loss = 0
        optimizer.zero_grad()
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        batch_count = 0
        grad_norms = []
        
        for batch_idx, batch in enumerate(progress_bar):
            try:
                # Assuming dataloader structure (adjust if hand_data was removed)
                frames, pose_data, _, action_masks, start_masks, end_masks, _ = batch
            except ValueError:
                 print("Warning: Batch structure mismatch. Trying simplified unpack (frames, pose, masks..., meta). Check dataloader.")
                 try: frames, pose_data, action_masks, start_masks, end_masks, _ = batch
                 except ValueError: print("Fatal: Cannot determine batch structure. Exiting."); exit()
            
            frames = frames.to(device)
            if pose_data is not None: pose_data = pose_data.to(device)
            action_masks = action_masks.to(device)
            start_masks = start_masks.to(device)
            end_masks = end_masks.to(device)

            with autocast(enabled=use_mixed_precision):
                predictions = model(frames, pose_data)
                
                targets = {
                    'action_masks': action_masks,
                    'start_masks': start_masks,
                    'end_masks': end_masks
                }
                loss_dict = criterion(predictions, targets)
                loss = loss_dict['total']
                loss = loss / gradient_accumulation_steps
            
            scaler.scale(loss).backward()
            
            train_loss += loss.item() * gradient_accumulation_steps # Use accumulated loss
            batch_count += 1 # Count batches processed before optimizer step
            
            if (batch_idx + 1) % gradient_accumulation_steps == 0 or (batch_idx + 1) == len(train_loader):
                scaler.unscale_(optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
                grad_norms.append(grad_norm.item())
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                
                progress_bar.set_postfix({
                    'loss': f"{loss_dict['total'].item():.4f}", # Log non-accumulated loss
                    'action': f"{loss_dict['action'].item():.4f}",
                    'start': f"{loss_dict['start'].item():.4f}",
                    'end': f"{loss_dict['end'].item():.4f}",
                    'grad': f"{grad_norm:.2f}"
                })
        
        if grad_norms: print(f"Gradient stats: min={min(grad_norms):.4f}, max={max(grad_norms):.4f}, mean={np.mean(grad_norms):.4f}")
        train_loss /= batch_count # Average loss over batches where optimizer stepped
        losses['train'].append(train_loss)
        
        # --- Validation --- #
        # Pass relevant parts of config to evaluate
        val_metrics = evaluate(
            model, val_loader, criterion, device, 
            eval_cfg=cfg['base_model_training'], # Pass the training config section for thresholds etc.
            num_classes=num_classes,
            use_mixed_precision=use_mixed_precision
        )
        val_loss = val_metrics['val_loss']
        val_map = val_metrics['mAP']
        val_f1 = val_metrics['merged_f1'] # Assuming this key exists
        class_ap_dict = val_metrics['class_aps']
        map_mid = val_metrics.get('map_mid', 0.0) # Use .get for safety
        avg_f1_iou_010 = val_metrics.get('avg_f1_iou_010', 0.0)
        avg_f1_iou_025 = val_metrics.get('avg_f1_iou_025', 0.0)
        avg_f1_iou_050 = val_metrics.get('avg_f1_iou_050', 0.0)
        
        losses['val'].append(val_loss)
        maps.append(val_map)
        for c in range(num_classes): class_aps[c].append(class_ap_dict.get(c, 0.0))
        
        # Update LR scheduler
        if epoch >= warmup_epochs:
            scheduler.step(val_loss)
            current_lr = optimizer.param_groups[0]['lr']
            print(f"Current LR: {current_lr:.8f}")
        
        # --- Logging --- #
        print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val mAP: {val_map:.4f}, Val F1: {val_f1:.4f}")
        print(f"  Extra Metrics: mAP@mid={map_mid:.4f}, F1@0.1={avg_f1_iou_010:.4f}, F1@0.25={avg_f1_iou_025:.4f}, F1@0.5={avg_f1_iou_050:.4f}")
        print(f"  Class AP: {', '.join([f'C{c}={class_ap_dict.get(c, 0.0):.4f}' for c in range(num_classes)])}")
        with open(log_file, 'a') as f:
            f.write(f"{epoch+1},{train_loss},{val_loss},{val_map},{val_f1},{map_mid},{avg_f1_iou_010},{avg_f1_iou_025},{avg_f1_iou_050}")
            for c in range(num_classes): f.write(f",{class_ap_dict.get(c, 0.0)}")
            f.write("\n")
        
        # --- Save Checkpoint --- #
        is_best = val_map > best_map
        if is_best:
            best_map = val_map
            print(f"✅ Saving best model with mAP: {best_map:.4f} to {best_checkpoint_path}")
            
        # Save best or potentially interim checkpoint
        save_interim = (epoch + 1) % 5 == 0 # Example: save every 5 epochs
        if is_best or save_interim:
            checkpoint_data = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_map': val_map,
                'val_f1': val_f1,
                'class_aps': class_ap_dict,
                'cfg': cfg # Optionally save config used for this run
            }
            current_checkpoint_path = best_checkpoint_path if is_best else checkpoint_dir / f"interim_model_epoch{epoch+1}.pth"
            try:
                 torch.save(checkpoint_data, current_checkpoint_path)
                 if not is_best and save_interim:
                      print(f"💾 Saved interim checkpoint to {current_checkpoint_path}")
            except Exception as e:
                 print(f"Error saving checkpoint to {current_checkpoint_path}: {e}")
                 
    # Optional: Plotting can be moved to a separate function/utility
    # plot_training_curves(losses, maps, class_aps, log_dir, start_time) 
    
    return best_map

def evaluate(model, val_loader, criterion, device, eval_cfg, num_classes, use_mixed_precision):
    """Evaluate the model using parameters from config."""
    model.eval()
    val_loss = 0.0
    
    # Get postprocessing settings from config
    pp_cfg = eval_cfg['postprocessing']
    boundary_threshold = pp_cfg['boundary_threshold']
    class_thresholds = pp_cfg['class_thresholds']
    nms_threshold = pp_cfg['nms_threshold']
    # min_segment_length = pp_cfg['min_segment_length'] # If needed by post_process
    
    # Lists to store results for final metric calculation
    all_window_detections = []
    all_window_metadata = []
    all_frame_preds_flat = [] # For frame-level F1
    all_frame_targets_flat = [] # For frame-level F1
    all_action_preds_flat = defaultdict(list) # For mAP
    final_global_gt = defaultdict(list) # For mAP
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validation"):
            try:
                 frames, pose_data, _, action_masks, start_masks, end_masks, metadata = batch
            except ValueError:
                 print("Warning: Batch structure mismatch in validation. Check dataloader.")
                 try: frames, pose_data, action_masks, start_masks, end_masks, metadata = batch
                 except ValueError: print("Fatal: Cannot determine batch structure. Exiting."); exit()
            
            frames = frames.to(device)
            if pose_data is not None: pose_data = pose_data.to(device)
            action_masks = action_masks.to(device)
            start_masks = start_masks.to(device)
            end_masks = end_masks.to(device)
            
            with autocast(enabled=use_mixed_precision):
                predictions = model(frames, pose_data)
                targets = {
                    'action_masks': action_masks,
                    'start_masks': start_masks,
                    'end_masks': end_masks
                }
                loss_dict = criterion(predictions, targets)
                loss = loss_dict['total']
            
            val_loss += loss.item()
            
            # --- Post-processing --- # 
            # Assuming the model itself has the post_process method now,
            # or we call a utility function from src.utils.postprocessing
            # For now, let's assume model.post_process exists as cleaned previously
            action_probs = torch.sigmoid(predictions['action_scores'])
            start_probs = torch.sigmoid(predictions['start_scores'])
            end_probs = torch.sigmoid(predictions['end_scores'])
        
            # Option 1: If model has post_process method
            try:
                 batch_detections = model.post_process(
                     action_scores=action_probs, # Pass probs if method expects scores post-sigmoid
                     start_scores=start_probs,
                     end_scores=end_probs,
                     action_threshold=class_thresholds, # Pass class-specific if method supports it
                     boundary_threshold=boundary_threshold,
                     nms_threshold=nms_threshold
                     # Pass other args like min_segment_length if needed
                 )
            except AttributeError:
                 print("Error: model.post_process not found. Ensure post-processing logic is correctly placed.")
                 # Option 2: Call a utility function (needs import)
                 # from src.utils.postprocessing import post_process # Example import
                 batch_detections = [[] for _ in range(frames.shape[0])] # Placeholder
            except TypeError as e:
                 print(f"Error calling model.post_process (likely argument mismatch): {e}")
                 batch_detections = [[] for _ in range(frames.shape[0])] # Placeholder
                 
            if eval_cfg['debugging']['debug_detection_enabled']:
                debug_detection_stats(batch_detections, frames.shape[0], metadata)
            
            # --- Accumulate results for metric calculation --- #
            # This part needs to be aligned with what compute_final_metrics expects
            # It involves calculating global GT and flattening predictions/targets
            # This logic might be complex and could be part of compute_final_metrics itself
            
            # Placeholder: Assume compute_final_metrics handles aggregation internally for now
            # We need to pass the raw batch-level info it needs
            all_window_detections.extend(batch_detections) # Detections per window
            all_window_metadata.extend(metadata) # Metadata per window
            
            # Frame-level accumulation (might be done inside compute_final_metrics too)
            # from src.utils.helpers import process_for_evaluation # Needs import
            # for i, (dets, meta) in enumerate(zip(batch_detections, metadata)):
            #     preds, targets = process_for_evaluation(dets, meta['annotations'], action_masks[i].cpu(), frames.shape[2], num_classes)
            #     all_frame_preds_flat.extend(preds)
            #     all_frame_targets_flat.extend(targets)

    # --- Calculate Final Metrics --- #
    # Call the centralized function
    # Note: compute_final_metrics needs adjustment to accept window-level data 
    # and perform the merging/flattening internally OR this evaluate function 
    # needs to do the preparation before calling it.
    
    # Let's assume evaluate prepares the final flattened lists as before for simplicity now.
    # This requires re-implementing the GT/Pred processing loop here temporarily until
    # compute_final_metrics is fully refactored to handle window data. 
    
    # --- TEMPORARY: Re-implement GT/Pred processing for compute_final_metrics input --- #
    print("\nTemporary: Preparing data for compute_final_metrics...")
    temp_global_gt, _, _ = calculate_global_gt(all_window_metadata, num_classes) 
    
    # Flatten predictions similar to how evaluate_pipeline does for RNN output
    from src.utils.postprocessing import merge_cross_window_detections, resolve_cross_class_overlaps # Need these imports
    merged_video_detections = merge_cross_window_detections(all_window_detections, all_window_metadata, iou_threshold=0.2, confidence_threshold=0.15)
    merged_video_detections = resolve_cross_class_overlaps(merged_video_detections)
    merged_all_action_preds = defaultdict(list)
    for video_dets in merged_video_detections.values():
        for det in video_dets:
            merged_all_action_preds[det['action_id']].append({'segment': (det['start_frame'], det['end_frame']), 'score': det['confidence']})
    
    # Calculate flattened frame targets/preds (based on merged detections)
    temp_all_frame_targets_flat = []
    temp_all_frame_preds_flat = []
    # (Need the loop similar to evaluate_pipeline to create frame-level lists based on merged_video_detections and temp_global_gt)
    # This is complex and ideally lives within compute_final_metrics
    # For now, pass empty lists for frame metrics as placeholder
    print("Warning: Frame-level F1 calculation needs refactoring within compute_final_metrics or evaluate.")
    # --- End Temporary Section --- # 
    
    final_metrics = compute_final_metrics(
        global_action_gt_global=temp_global_gt, 
        merged_all_action_preds=merged_all_action_preds, 
        merged_all_frame_targets=temp_all_frame_targets_flat, # Placeholder
        merged_all_frame_preds=temp_all_frame_preds_flat,   # Placeholder
        num_classes=num_classes
    )

    avg_val_loss = val_loss / len(val_loader) # Calculate average loss
    final_metrics['val_loss'] = avg_val_loss # Add loss to the results dict
    
    return final_metrics

def main():
    """Main training function, loads config and initiates training."""
    parser = argparse.ArgumentParser(description="Train Temporal Action Detection Base Model")
    parser.add_argument('--config', default='configs/config.yaml', help='Path to configuration file')
    # Add arg to override resume behavior from config
    parser.add_argument('--resume', action=argparse.BooleanOptionalAction, help="Override config resume_training setting")
    parser.add_argument('--checkpoint', type=str, default=None, help="Specific checkpoint to resume from (overrides config default resume checkpoint)")

    args = parser.parse_args()

    # Load config
    try:
        with open(args.config, 'r') as f:
            cfg = yaml.safe_load(f)
    except FileNotFoundError: print(f"Error: Config file not found at {args.config}"); return
    except Exception as e: print(f"Error loading config file: {e}"); return

    # --- Setup from Config --- #
    global_cfg = cfg['global']
    data_cfg = cfg['data']
    train_cfg = cfg['base_model_training']
    opt_cfg = train_cfg['optimizer']
    sched_cfg = train_cfg['scheduler']
    loss_cfg = train_cfg['loss']

    set_seed(global_cfg['seed'])

    if global_cfg['device'] == 'auto': device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else: device = torch.device(global_cfg['device'])
    
    print(f"Using device: {device}")
    if torch.cuda.is_available(): print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Using config file: {args.config}")

    # --- Dataloaders --- #
    # Batch size for loader comes from training config
    train_loader = get_train_loader(batch_size=train_cfg['batch_size'], shuffle=True) 
    val_loader = get_val_loader(batch_size=train_cfg['batch_size'], shuffle=False)

    # --- Model --- #
    model = TemporalActionDetector(
        num_classes=global_cfg['num_classes'], 
        window_size=global_cfg['window_size']
        # Add dropout from config if applicable to model init
        # dropout=train_cfg.get('dropout', 0.3) 
    ).to(device)
    print(f"Model: TemporalActionDetector with {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M parameters")

    # --- Loss Function --- #
    criterion = ActionDetectionLoss(
        action_weight=loss_cfg['action_weight'], 
        start_weight=loss_cfg['start_weight'], 
        end_weight=loss_cfg['end_weight'], 
        num_classes=global_cfg['num_classes'],
        label_smoothing=loss_cfg['label_smoothing'],
        device=device # Pass device explicitly
    )

    # --- Optimizer --- #
    if opt_cfg['type'].lower() == 'adamw':
        optimizer = optim.AdamW(
            model.parameters(), lr=opt_cfg['lr'], 
            weight_decay=opt_cfg['weight_decay'], eps=opt_cfg['eps']
        )
    else:
        # Add other optimizers if needed
        print(f"Error: Unsupported optimizer type {opt_cfg['type']}")
        return
        
    # --- Scheduler --- #
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min', 
        factor=sched_cfg['factor'], 
        patience=sched_cfg['patience'], 
        min_lr=sched_cfg['min_lr'], 
        verbose=True 
    )

    # --- Checkpoint Loading / Resume Logic --- #
    start_epoch = 0
    best_map = 0.0
    checkpoint_dir = Path(data_cfg['base_model_checkpoints'])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # Determine if resuming and which checkpoint to use
    resume_training = args.resume if args.resume is not None else train_cfg['resume_training']
    resume_checkpoint_path = None
    if resume_training:
        if args.checkpoint:
            resume_checkpoint_path = Path(args.checkpoint)
            print(f"Attempting to resume from specified checkpoint: {resume_checkpoint_path}")
        else:
            default_resume_name = data_cfg.get('base_resume_checkpoint_name')
            if default_resume_name:
                 resume_checkpoint_path = checkpoint_dir / default_resume_name
                 print(f"Attempting to resume from default checkpoint in config: {resume_checkpoint_path}")
            else:
                 print("Resume enabled but no specific or default checkpoint specified in config.")

    if resume_checkpoint_path and resume_checkpoint_path.exists():
        print(f"Resuming training from checkpoint: {resume_checkpoint_path}")
        try:
            checkpoint = torch.load(resume_checkpoint_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            
                # Re-init optimizer before loading state for device compatibility
            if opt_cfg['type'].lower() == 'adamw':
                    optimizer = optim.AdamW(model.parameters(), lr=opt_cfg['lr'], weight_decay=opt_cfg['weight_decay'], eps=opt_cfg['eps'])
                # Add other optimizers if needed

            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                # Ensure optimizer state is on the correct device
            for state in optimizer.state.values():
                for k, v in state.items():
                        if isinstance(v, torch.Tensor): state[k] = v.to(device)

                # Re-init scheduler before loading state
                scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=sched_cfg['factor'], patience=sched_cfg['patience'], min_lr=sched_cfg['min_lr'], verbose=True)
                if 'scheduler_state_dict' in checkpoint: scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

                start_epoch = checkpoint.get('epoch', 0)
                best_map = checkpoint.get('val_map', 0.0)
                print(f"Loaded checkpoint from epoch {start_epoch}. Previous best mAP: {best_map:.4f}")
                # Optionally load and restore config if saved in checkpoint? cfg = checkpoint.get('cfg', cfg)
        except Exception as e:
             print(f"Error loading checkpoint {resume_checkpoint_path}: {e}. Starting from scratch.")
             start_epoch = 0
             best_map = 0.0
    elif resume_training:
        print(f"Resume enabled but checkpoint not found at {resume_checkpoint_path}. Starting from scratch.")
    else:
        print("Starting training from scratch.")

    # --- Start Training --- #
    try:
        final_best_map = train(
            model, train_loader, val_loader, criterion, optimizer, scheduler, 
            cfg, # Pass full config dictionary
            device, start_epoch=start_epoch, best_map=best_map
        )
        print(f"\n✅ Training complete! Best validation mAP: {final_best_map:.4f}")
        best_model_path = checkpoint_dir / data_cfg['base_best_checkpoint_name']
        print(f"Best model saved to {best_model_path}")
        
        # --- Final Evaluation on Test Set --- #
        if train_cfg['evaluation']['run_final_evaluation_on_test']:
            print("\n=== Final Evaluation on Test Set ===")
            best_model_path = checkpoint_dir / data_cfg['base_best_checkpoint_name']
            if best_model_path.exists():
                print(f"Loading best model from {best_model_path}")
                checkpoint = torch.load(best_model_path, map_location=device)
                model.load_state_dict(checkpoint['model_state_dict'])

                test_loader = get_test_loader(batch_size=train_cfg['batch_size'], shuffle=False)

                # Call evaluate for the test set
                test_metrics = evaluate(
                    model, test_loader, criterion, device,
                    eval_cfg=train_cfg, # Use same eval settings as validation
                    num_classes=global_cfg['num_classes'],
                    use_mixed_precision=train_cfg['use_mixed_precision']
                )

                print(f"Test Loss: {test_metrics.get('val_loss', -1):.4f}, Test mAP: {test_metrics.get('mAP', -1):.4f}, Test F1: {test_metrics.get('merged_f1', -1):.4f}")
                class_aps_test = test_metrics.get('class_aps', {})
                print(f"Test Class AP: {', '.join([f'C{c}={class_aps_test.get(c, 0.0):.4f}' for c in range(global_cfg['num_classes'])])}")
            
            # Save test results
                log_dir = Path(data_cfg['logs'])
                test_results_path = log_dir / 'test_results_base_model.json'
                try:
                     with open(test_results_path, 'w') as f: json.dump(test_metrics, f, indent=2)
                     print(f"Saved test results to {test_results_path}")
                except Exception as e: print(f"Error saving test results: {e}")
            else:
                print(f"Error: Best model checkpoint not found at {best_model_path} for final test evaluation.")
                
    except Exception as e:
        print(f"\n❌ Error during training: {str(e)}")
        import traceback
        traceback.print_exc()
        # Try to free up memory
        torch.cuda.empty_cache()
        # raise e # Optional: re-raise after cleanup

if __name__ == "__main__":
    torch.cuda.empty_cache()  
    main() 
