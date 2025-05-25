import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import torch.nn.functional as F
from contextlib import nullcontext
import numpy as np
from pathlib import Path
import os
import random
from tqdm import tqdm
from models.concat import TemporalActionDetector
from torch.optim.lr_scheduler import ReduceLROnPlateau
import src.utils.helpers as helpers


try:
    from src.dataloader import get_feature_train_loader, get_feature_val_loader
    from src.dataloader import WINDOW_SIZE as DL_WINDOW_SIZE
    from src.dataloader import NUM_CLASSES as DL_NUM_CLASSES
except ImportError:
    print("Error: cannot import from src.dataloader")
    DL_WINDOW_SIZE = 32


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LEARNING_RATE = 1e-4
BATCH_SIZE = 256
NUM_EPOCHS = 50
WEIGHT_DECAY = 1e-5
MODEL_SAVE_PATH = "trained_models"
os.makedirs(MODEL_SAVE_PATH, exist_ok=True)
BEST_MODEL_NAME = os.path.join(MODEL_SAVE_PATH, "v5_3.pth")

criterion = nn.BCEWithLogitsLoss()


total_duration_raw = np.array([30582, 31021, 18183, 39308, 52011], dtype=np.float32)


class_weights_by_duration = (np.sum(total_duration_raw) / total_duration_raw)
class_weights_by_duration = class_weights_by_duration / np.mean(class_weights_by_duration)
print(f"Class weights by duration: {class_weights_by_duration}")
CLASS_WEIGHTS = torch.tensor(class_weights_by_duration, device=DEVICE, dtype=torch.float32)

def calculate_loss(outputs, targets_action, targets_start, targets_end, class_weights):

    pred_action = outputs['action_scores'].permute(0, 2, 1)
    pred_start = outputs['start_scores'].permute(0, 2, 1)
    pred_end = outputs['end_scores'].permute(0, 2, 1)

    loss_action_elements = criterion(pred_action, targets_action)
    loss_start_elements = criterion(pred_start, targets_start)
    loss_end_elements = criterion(pred_end, targets_end)

    weights_reshaped = class_weights.view(1, -1, 1)

    weighted_loss_action = loss_action_elements * weights_reshaped
    weighted_loss_start = loss_start_elements * weights_reshaped
    weighted_loss_end = loss_end_elements * weights_reshaped

    final_loss_action = torch.mean(weighted_loss_action)
    final_loss_start = torch.mean(weighted_loss_start)
    final_loss_end = torch.mean(weighted_loss_end)

    total_loss = final_loss_action + final_loss_start + final_loss_end

    return total_loss, final_loss_action, final_loss_start, final_loss_end

def train_one_epoch(model, dataloader, optimizer, device):
    model.train()
    total_loss_epoch = 0
    total_l_action, total_l_start, total_l_end = 0, 0, 0

    progress_bar = tqdm(dataloader, desc="Training Epoch", leave=False)
    for batch_idx, batch_data in enumerate(progress_bar):
        if batch_data is None: continue

        mvit_feat, resnet_feat, pose_feat, action_mask, start_mask, end_mask, _ = batch_data
        
        mvit_feat = mvit_feat.to(device)
        resnet_feat = resnet_feat.to(device)
        pose_feat = pose_feat.to(device)
        action_mask = action_mask.to(device)
        start_mask = start_mask.to(device)
        end_mask = end_mask.to(device)

        optimizer.zero_grad()
        outputs = model(mvit_feat, resnet_feat, pose_feat)
        loss, l_act, l_start, l_end = calculate_loss(outputs, action_mask, start_mask, end_mask, CLASS_WEIGHTS)
        
        loss.backward()
        optimizer.step()
        
        total_loss_epoch += loss.item()
        total_l_action += l_act.item()
        total_l_start += l_start.item()
        total_l_end += l_end.item()

        progress_bar.set_postfix(loss=loss.item())

    avg_loss = total_loss_epoch / len(dataloader)
    avg_l_action = total_l_action / len(dataloader)
    avg_l_start = total_l_start / len(dataloader)
    avg_l_end = total_l_end / len(dataloader)
    
    return avg_loss, avg_l_action, avg_l_start, avg_l_end

def validate_one_epoch(model, dataloader, device):
    model.eval()
    total_loss_epoch = 0
    total_l_action, total_l_start, total_l_end = 0, 0, 0
    
    progress_bar = tqdm(dataloader, desc="Validation Epoch", leave=False)
    with torch.no_grad():
        for batch_idx, batch_data in enumerate(progress_bar):
            if batch_data is None: continue

            mvit_feat, resnet_feat, pose_feat, action_mask, start_mask, end_mask, _ = batch_data

            mvit_feat = mvit_feat.to(device)
            resnet_feat = resnet_feat.to(device)
            pose_feat = pose_feat.to(device)
            action_mask = action_mask.to(device)
            start_mask = start_mask.to(device)
            end_mask = end_mask.to(device)

            outputs = model(mvit_feat, resnet_feat, pose_feat)
            loss, l_act, l_start, l_end = calculate_loss(outputs, action_mask, start_mask, end_mask, CLASS_WEIGHTS)
            
            total_loss_epoch += loss.item()
            total_l_action += l_act.item()
            total_l_start += l_start.item()
            total_l_end += l_end.item()
            progress_bar.set_postfix(loss=loss.item())

    avg_loss = total_loss_epoch / len(dataloader)
    avg_l_action = total_l_action / len(dataloader)
    avg_l_start = total_l_start / len(dataloader)
    avg_l_end = total_l_end / len(dataloader)

    return avg_loss, avg_l_action, avg_l_start, avg_l_end

if __name__ == "__main__":
    patience = 0
    helpers.set_seed(42)
    print(f"Device using: {DEVICE}")

    print("Loading data")
    train_loader = get_feature_train_loader(batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = get_feature_val_loader(batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    if train_loader is None or val_loader is None:
        print("Error: Cannot load train or validation data")
        exit()
    
    print(f"Train batch: {len(train_loader)}")
    print(f"Val batch: {len(val_loader)}")


    model = TemporalActionDetector(
        num_classes=DL_NUM_CLASSES, 
        window_size=DL_WINDOW_SIZE, 
        dropout=0.3, 
    ).to(DEVICE)
    
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3, verbose=True)

    best_val_loss = float('inf')

    print(f"\n Start training for {NUM_EPOCHS} epochs...")
    for epoch in range(1, NUM_EPOCHS + 1):
        print(f"\n--- Epoch {epoch}/{NUM_EPOCHS} ---")
        
        train_loss, tr_act, tr_start, tr_end = train_one_epoch(model, train_loader, optimizer, DEVICE)
        print(f"Train Loss: {train_loss:.4f} (Action: {tr_act:.4f}, Start: {tr_start:.4f}, End: {tr_end:.4f})")
        
        val_loss, v_act, v_start, v_end = validate_one_epoch(model, val_loader, DEVICE)
        print(f"Validation Loss: {val_loss:.4f} (Action: {v_act:.4f}, Start: {v_start:.4f}, End: {v_end:.4f})")
        
        scheduler.step(val_loss)
        print(f"Current LR: {scheduler.get_last_lr()[0]:.6f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), BEST_MODEL_NAME)
            print(f"Best model saved {BEST_MODEL_NAME}")
            patience = 0
        else:
            patience += 1
            print(f"Patience: {patience}/{10}")
            if patience >= 10:
                print("Reached early stopping. Stopping training")
                break

    print("\nFinished training")
    print(f"Best model saved: {BEST_MODEL_NAME} - Val Loss: {best_val_loss:.4f}")