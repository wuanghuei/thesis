import torch
import torch.optim as optim
from tqdm import tqdm
import numpy as np
import json
from datetime import datetime
from torch.cuda.amp import autocast, GradScaler
import yaml
from pathlib import Path
import argparse

import src.models.base_detector as base_detector
import src.dataloader as dataloader
import src.utils.helpers as helpers
import src.utils.debugging as debugging
import src.losses as losses
import src.evaluation as evaluation
import src.utils.postprocessing as postprocessing


def train(
    model, train_loader, val_loader, criterion, optimizer, scheduler, 
    cfg,
    device, start_epoch=0, best_map=0
):
    epochs = cfg['base_model_training']['epochs']
    log_dir = Path(cfg['data']['logs'])
    checkpoint_dir = Path(cfg['data']['base_model_checkpoints'])
    best_checkpoint_path = checkpoint_dir / cfg['data']['base_best_checkpoint_name']
    use_mixed_precision = cfg['base_model_training']['use_mixed_precision']
    gradient_accumulation_steps = cfg['base_model_training']['gradient_accumulation_steps']
    max_grad_norm = cfg['base_model_training']['gradient_clipping']['max_norm']
    warmup_epochs = cfg['base_model_training']['warmup']['epochs']
    base_lr = float(cfg['base_model_training']['optimizer']['lr'])
    warmup_factor = float(cfg['base_model_training']['warmup']['factor'])
    initial_loss_weights = cfg['base_model_training']['loss']
    debug_detection_enabled = cfg['base_model_training']['debugging']['debug_detection_enabled']
    
    adjust_weights = True
    if adjust_weights:
        initial_action_weight = initial_loss_weights['action_weight']
        initial_start_weight = initial_loss_weights['start_weight']
        initial_end_weight = initial_loss_weights['end_weight']
        
    start_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"training_log_base_{start_time}.csv"
    log_dir.mkdir(parents=True, exist_ok=True)
    
    with open(log_file, 'w') as f:
        f.write("epoch,train_loss,val_loss,val_map,val_f1,map_mid,f1_iou_010,f1_iou_025,f1_iou_050,class0_ap,class1_ap,class2_ap,class3_ap,class4_ap\n")
    
    losses = {'train': [], 'val': []}
    maps = []
    num_classes = cfg['global']['num_classes'] 
    class_aps = {c: [] for c in range(num_classes)}
    
    scaler = GradScaler(enabled=use_mixed_precision)
    
    for epoch in range(start_epoch, epochs):
        if adjust_weights and epoch >= 30: 
            progress = min(1.0, (epoch - 30) / 20)  
            criterion.action_weight = initial_action_weight * (1 - 0.3 * progress)  
            criterion.start_weight = initial_start_weight * (1 + 0.5 * progress)  
            criterion.end_weight = initial_end_weight * (1 + 0.5 * progress)  
            print(f"Epoch {epoch+1}: Adjusted weights - Action: {criterion.action_weight:.2f}, Start: {criterion.start_weight:.2f}, End: {criterion.end_weight:.2f}")
        
        if epoch < warmup_epochs:
            warmup_start = base_lr / warmup_factor
            current_lr = warmup_start + (base_lr - warmup_start) * (epoch + 1) / warmup_epochs
            for param_group in optimizer.param_groups:
                param_group['lr'] = current_lr
            print(f"Warmup LR: {current_lr:.8f}")
        
        model.train()
        train_loss = 0
        optimizer.zero_grad()
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        batch_count = 0
        grad_norms = []
        
        for batch_idx, batch in enumerate(progress_bar):
            try:
                frames, pose_data, _, action_masks, start_masks, end_masks, _ = batch
            except ValueError:
                 print("Batch structure mismatch Trying simplified unpack (frames, pose, masks..., meta) Check dataloader")
                 try: frames, pose_data, action_masks, start_masks, end_masks, _ = batch
                 except ValueError: print("Cannot determine batch structure, exiting"); exit()
            
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
            
            train_loss += loss.item() * gradient_accumulation_steps
            batch_count += 1
            
            if (batch_idx + 1) % gradient_accumulation_steps == 0 or (batch_idx + 1) == len(train_loader):
                scaler.unscale_(optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
                grad_norms.append(grad_norm.item())
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                
                progress_bar.set_postfix({
                    'loss': f"{loss_dict['total'].item():.4f}",
                    'action': f"{loss_dict['action'].item():.4f}",
                    'start': f"{loss_dict['start'].item():.4f}",
                    'end': f"{loss_dict['end'].item():.4f}",
                    'grad': f"{grad_norm:.2f}"
                })
        
        if grad_norms: print(f"Gradient stats: min={min(grad_norms):.4f}, max={max(grad_norms):.4f}, mean={np.mean(grad_norms):.4f}")
        train_loss /= batch_count
        losses['train'].append(train_loss)
    
        val_metrics = evaluate(
            model, val_loader, criterion, device, 
            eval_cfg=cfg['base_model_training'],
            num_classes=num_classes,
            use_mixed_precision=use_mixed_precision
        )
        val_loss = val_metrics['val_loss']
        val_map = val_metrics['mAP']
        val_f1 = val_metrics['merged_f1']
        class_ap_dict = val_metrics['class_aps']
        map_mid = val_metrics.get('map_mid', 0.0)
        avg_f1_iou_010 = val_metrics.get('avg_f1_iou_010', 0.0)
        avg_f1_iou_025 = val_metrics.get('avg_f1_iou_025', 0.0)
        avg_f1_iou_050 = val_metrics.get('avg_f1_iou_050', 0.0)
        
        losses['val'].append(val_loss)
        maps.append(val_map)
        for c in range(num_classes): class_aps[c].append(class_ap_dict.get(c, 0.0))
        
        if epoch >= warmup_epochs:
            scheduler.step(val_loss)
            current_lr = optimizer.param_groups[0]['lr']
            print(f"Current LR: {current_lr:.8f}")
        
        print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val mAP: {val_map:.4f}, Val F1: {val_f1:.4f}")
        print(f"  Extra Metrics: mAP@mid={map_mid:.4f}, F1@0.1={avg_f1_iou_010:.4f}, F1@0.25={avg_f1_iou_025:.4f}, F1@0.5={avg_f1_iou_050:.4f}")
        print(f"  Class AP: {', '.join([f'C{c}={class_ap_dict.get(c, 0.0):.4f}' for c in range(num_classes)])}")
        with open(log_file, 'a') as f:
            f.write(f"{epoch+1},{train_loss},{val_loss},{val_map},{val_f1},{map_mid},{avg_f1_iou_010},{avg_f1_iou_025},{avg_f1_iou_050}")
            for c in range(num_classes): f.write(f",{class_ap_dict.get(c, 0.0)}")
            f.write("\n")
        
        is_best = val_map > best_map
        if is_best:
            best_map = val_map
            print(f"Saving best model with mAP: {best_map:.4f} to {best_checkpoint_path}")
            
        save_interim = (epoch + 1) % 5 == 0
        if is_best or save_interim:
            checkpoint_data = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_map': val_map,
                'val_f1': val_f1,
                'class_aps': class_ap_dict,
                'cfg': cfg
            }
            current_checkpoint_path = best_checkpoint_path if is_best else checkpoint_dir / f"interim_model_epoch{epoch+1}.pth"
            try:
                 torch.save(checkpoint_data, current_checkpoint_path)
                 if not is_best and save_interim:
                      print(f"Saved interim checkpoint to {current_checkpoint_path}")
            except Exception as e:
                 print(f"Error saving checkpoint to {current_checkpoint_path}: {e}")

    
    return best_map

def evaluate(model, val_loader, criterion, device, eval_cfg, num_classes, use_mixed_precision):
    model.eval()
    val_loss = 0.0
    
    pp_cfg = eval_cfg['postprocessing']
    boundary_threshold = pp_cfg['boundary_threshold']
    class_thresholds = pp_cfg['class_thresholds']
    nms_threshold = pp_cfg['nms_threshold']
    
    all_window_detections = []
    all_window_metadata = []
    all_frame_preds_flat = []
    all_frame_targets_flat = []
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validation"):
            try:
                 frames, pose_data, _, action_masks, start_masks, end_masks, metadata = batch
            except ValueError:
                 print("Batch structure mismatch in validation Check dataloader")
                 try: frames, pose_data, action_masks, start_masks, end_masks, metadata = batch
                 except ValueError: print("Cannot determine batch structure, exiting"); exit()
            
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
            

            action_probs = torch.sigmoid(predictions['action_scores'])
            start_probs = torch.sigmoid(predictions['start_scores'])
            end_probs = torch.sigmoid(predictions['end_scores'])
        

            batch_detections = postprocessing.post_process(
                action_probs=action_probs,
                start_probs=start_probs,
                end_probs=end_probs,
                class_thresholds=class_thresholds,
                boundary_threshold=boundary_threshold,
                nms_threshold=nms_threshold
                )

                 
            if eval_cfg['debugging']['debug_detection_enabled']:
                debugging.debug_detection_stats(batch_detections, frames.shape[0], metadata)
            
            
            all_window_detections.extend(batch_detections)
            all_window_metadata.extend(metadata)
            
            for i, (dets, meta) in enumerate(zip(batch_detections, metadata)):
                preds, targets = helpers.process_for_evaluation(dets, meta['annotations'], action_masks[i].cpu(), frames.shape[2], num_classes)
                all_frame_preds_flat.extend(preds)
                all_frame_targets_flat.extend(targets)


    final_metrics = evaluation.compute_final_metrics(
        all_window_detections,
        all_window_metadata,
        all_frame_preds_flat,
        all_frame_targets_flat,
        num_classes
    )

    avg_val_loss = val_loss / len(val_loader)
    final_metrics['val_loss'] = avg_val_loss
    
    return final_metrics

def main():
    """Main training function, loads config and initiates training."""
    parser = argparse.ArgumentParser(description="Train Temporal Action Detection Base Model")
    parser.add_argument('--config', default='configs/config.yaml', help='Path to configuration file')
    parser.add_argument('--resume', action=argparse.BooleanOptionalAction, help="Override config resume_training setting")
    parser.add_argument('--checkpoint', type=str, default=None, help="Specific checkpoint to resume from (overrides config default resume checkpoint)")

    args = parser.parse_args()

    try:
        with open(args.config, 'r') as f:
            cfg = yaml.safe_load(f)
    except FileNotFoundError: print(f"Config file not found at {args.config}, returning"); return
    except Exception as e: print(f"loading config file: {e}, returning"); return

    global_cfg = cfg['global']
    data_cfg = cfg['data']
    train_cfg = cfg['base_model_training']
    opt_cfg = train_cfg['optimizer']
    sched_cfg = train_cfg['scheduler']
    loss_cfg = train_cfg['loss']

    helpers.set_seed(global_cfg['seed'])

    device = torch.device(global_cfg['device'])
    
    print(f"Using device: {device}")
    if torch.cuda.is_available(): print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Using config file: {args.config}")

    train_loader = dataloader.get_train_loader(cfg, shuffle=True) 
    val_loader = dataloader.get_val_loader(cfg, shuffle=False)

    model = base_detector.TemporalActionDetector(
        num_classes=global_cfg['num_classes'], 
        window_size=global_cfg['window_size']
    ).to(device)
    print(f"Model: TemporalActionDetector with {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M parameters")

    criterion = losses.ActionDetectionLoss(
        action_weight=loss_cfg['action_weight'], 
        start_weight=loss_cfg['start_weight'], 
        end_weight=loss_cfg['end_weight'], 
        num_classes=global_cfg['num_classes'],
        label_smoothing=loss_cfg['label_smoothing'],
        device=device
    )

    if opt_cfg['type'].lower() == 'adamw':
        optimizer = optim.AdamW(
            model.parameters(), lr=float(opt_cfg['lr']), 
            weight_decay=float(opt_cfg['weight_decay']), eps=float(opt_cfg['eps'])
        )
    else:
        print(f"Unsupported optimizer type {opt_cfg['type']}")
        return
        
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min', 
        factor=sched_cfg['factor'], 
        patience=sched_cfg['patience'], 
        min_lr=sched_cfg['min_lr'], 
        verbose=True 
    )

    start_epoch = 0
    best_map = 0.0
    checkpoint_dir = Path(data_cfg['base_model_checkpoints'])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
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
                 print("Resume enabled but no specific or default checkpoint specified in config")

    if resume_checkpoint_path and resume_checkpoint_path.exists():
        print(f"Resuming training from checkpoint: {resume_checkpoint_path}")
        try:
            checkpoint = torch.load(resume_checkpoint_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            
            if opt_cfg['type'].lower() == 'adamw':
                    optimizer = optim.AdamW(model.parameters(), lr=opt_cfg['lr'], weight_decay=opt_cfg['weight_decay'], eps=opt_cfg['eps'])

            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            for state in optimizer.state.values():
                for k, v in state.items():
                        if isinstance(v, torch.Tensor): state[k] = v.to(device)

                scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=sched_cfg['factor'], patience=sched_cfg['patience'], min_lr=sched_cfg['min_lr'], verbose=True)
                if 'scheduler_state_dict' in checkpoint: scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

                start_epoch = checkpoint.get('epoch', 0)
                best_map = checkpoint.get('val_map', 0.0)
                print(f"Loaded checkpoint from epoch {start_epoch} Previous best mAP: {best_map:.4f}")
        except Exception as e:
             print(f"loading checkpoint {resume_checkpoint_path}: {e}, Starting from scratch")
             start_epoch = 0
             best_map = 0.0
    elif resume_training:
        print(f"Resume enabled but checkpoint not found at {resume_checkpoint_path}, Starting from scratch")
    else:
        print("Starting training from scratch")

    try:
        final_best_map = train(
            model, train_loader, val_loader, criterion, optimizer, scheduler, 
            cfg, 
            device, start_epoch=start_epoch, best_map=best_map
        )
        print(f"Training complete Best validation mAP: {final_best_map:.4f}")
        best_model_path = checkpoint_dir / data_cfg['base_best_checkpoint_name']
        print(f"Best model saved to {best_model_path}")
        
        if train_cfg['evaluation']['run_final_evaluation_on_test']:
            print("\n=== Final Evaluation on Test Set ===")
            best_model_path = checkpoint_dir / data_cfg['base_best_checkpoint_name']
            if best_model_path.exists():
                print(f"Loading best model from {best_model_path}")
                checkpoint = torch.load(best_model_path, map_location=device)
                model.load_state_dict(checkpoint['model_state_dict'])

                test_loader = dataloader.get_test_loader(cfg, shuffle=False)

                test_metrics = evaluate(
                    model, test_loader, criterion, device,
                    eval_cfg=train_cfg, 
                    num_classes=global_cfg['num_classes'],
                    use_mixed_precision=train_cfg['use_mixed_precision']
                )

                print(f"Test Loss: {test_metrics.get('val_loss', -1):.4f}, Test mAP: {test_metrics.get('mAP', -1):.4f}, Test F1: {test_metrics.get('merged_f1', -1):.4f}")
                class_aps_test = test_metrics.get('class_aps', {})
                print(f"Test Class AP: {', '.join([f'C{c}={class_aps_test.get(c, 0.0):.4f}' for c in range(global_cfg['num_classes'])])}")
            
                log_dir = Path(data_cfg['logs'])
                test_results_path = log_dir / 'test_results_base_model.json'
                try:
                     with open(test_results_path, 'w') as f: json.dump(test_metrics, f, indent=2)
                     print(f"Saved test results to {test_results_path}")
                except Exception as e: print(f"saving test results: {e}")
            else:
                print(f"Best model checkpoint not found at {best_model_path} for final test evaluation")
                
    except Exception as e:
        print(f"Error during training: {str(e)}")
        import traceback
        traceback.print_exc()
        torch.cuda.empty_cache()

if __name__ == "__main__":
    torch.cuda.empty_cache()  
    main() 
