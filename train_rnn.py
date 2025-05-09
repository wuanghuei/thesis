import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import numpy as np
import os
import glob
import argparse
from tqdm import tqdm
import yaml
from pathlib import Path

try:
    # Adjust path if your model file is elsewhere
    import src.models.rnn_postprocessor as rnn_postprocessor
except ImportError:
    print("Could not import RNNPostProcessor from src/models/rnn_postprocessor py")
    print("Please ensure the file exists and is in the correct directory")
    exit()

# ====== Dataset Class ======
class RNNDataset(Dataset):
    """Dataset to load preprocessed features and labels for RNN training."""
    def __init__(self, data_dir):
        """
        Args:
            data_dir (str): Path to the directory containing .npz files.
        """
        self.data_files = glob.glob(os.path.join(data_dir, "*.npz"))
        if not self.data_files:
            raise FileNotFoundError(f"No npz files found in directory: {data_dir}")
        print(f"Found {len(self.data_files)} data files in {data_dir}")

    def __len__(self):
        return len(self.data_files)

    def __getitem__(self, idx):
        filepath = self.data_files[idx]
        try:
            data = np.load(filepath)
            features = torch.tensor(data['features'], dtype=torch.float32)
            labels = torch.tensor(data['labels'], dtype=torch.long)
            return features, labels
        except Exception as e:
            print(f"loading or processing file {filepath}: {e}")
            raise e

def collate_fn(batch):
    """Pads sequences within a batch and returns features, labels, and lengths."""

    features_list = [item[0] for item in batch]
    labels_list = [item[1] for item in batch]
    
    lengths = torch.tensor([len(seq) for seq in features_list], dtype=torch.long)
 
    features_padded = pad_sequence(features_list, batch_first=True, padding_value=0.0)

    labels_padded = pad_sequence(labels_list, batch_first=True, padding_value=-100)
    
    return features_padded, labels_padded, lengths

def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    
    progress_bar = tqdm(dataloader, desc="Training", leave=False)
    for features, labels, lengths in progress_bar:
        if features is None: continue
        
        features, labels = features.to(device), labels.to(device)
        
        optimizer.zero_grad()
        
        logits = model(features, lengths)

        loss = criterion(logits.view(-1, logits.shape[-1]), labels.view(-1))
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        progress_bar.set_postfix(loss=loss.item())
        
    return total_loss / len(dataloader)

def validate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    
    progress_bar = tqdm(dataloader, desc="Validation", leave=False)
    with torch.no_grad():
        for features, labels, lengths in progress_bar:
            if features is None: continue
            
            features, labels = features.to(device), labels.to(device)
            
            logits = model(features, lengths)
            
            loss = criterion(logits.view(-1, logits.shape[-1]), labels.view(-1))
            
            total_loss += loss.item()
            progress_bar.set_postfix(loss=loss.item())
            
    return total_loss / len(dataloader)

def main(args):

    config_path = Path(args.config)
    if not config_path.is_file():
        print(f"Config file not found at {config_path}")
        exit(1)
    try:
        with open(config_path, 'r') as f:
            cfg = yaml.safe_load(f)
            print(f"Loaded configuration from {config_path}")
    except Exception as e:
        print(f"loading config file {config_path}: {e}")
        exit(1)

    global_cfg = cfg.get('global', {})
    data_cfg = cfg.get('data', {})
    rnn_cfg = cfg.get('rnn_training', {})

    train_data_dir = Path(data_cfg.get('rnn_processed_data')/ 'train')
    val_data_dir = Path(data_cfg.get('rnn_processed_data')/ 'val')
    checkpoint_dir = Path(data_cfg.get('rnn_model_checkpoints'))
    best_checkpoint_name = data_cfg.get('rnn_best_checkpoint_name', 'best_rnn_model.pth')


    device_str = global_cfg.get('device', 'auto').lower()
    if device_str == 'auto':
        DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        DEVICE = torch.device(device_str)

    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    print(f"Using device: {DEVICE}")
    print("--- Data Configuration ---")
    print(f"  Train Data: {train_data_dir}")
    print(f"  Val Data: {val_data_dir}")
    print(f"  Checkpoints: {checkpoint_dir}")
    print("--- Model Hyperparameters ---")
    print(f"  RNN Type: {rnn_cfg['model']['type']}")
    print(f"  Input Size: {rnn_cfg['model']['input_size']}")
    print(f"  Num Classes: {rnn_cfg['model']['num_classes']}")
    print(f"  Hidden Size: {rnn_cfg['model']['hidden_size']}")
    print(f"  Num Layers: {rnn_cfg['model']['num_layers']}")
    print(f"  Bidirectional: {rnn_cfg['model']['bidirectional']}")
    print(f"  Dropout: {rnn_cfg['model']['dropout_prob']}")
    print("--- Training Hyperparameters ---")
    print(f"  Batch Size: {rnn_cfg['batch_size']}")
    print(f"  Learning Rate: {rnn_cfg['optimizer']['lr']}")
    print(f"  Epochs: {rnn_cfg['epochs']}")
    print(f"  Patience: {rnn_cfg['early_stopping']['patience']}")
    print(f"  Num Workers: {rnn_cfg['dataloader']['num_workers']}")
    print(f"-----------------------------")

    try:
        # Pass paths directly
        train_dataset = RNNDataset(str(train_data_dir))
        val_dataset = RNNDataset(str(val_data_dir))
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please ensure Phase 1 (1_generate_rnn_data py) completed successfully and data exists")
        exit()

    # Create DataLoaders
    # Make validation batch size configurable or keep fixed factor?
    val_batch_size = rnn_cfg.get('val_batch_size', rnn_cfg['batch_size'] * 2)
    train_loader = DataLoader(train_dataset,
                              batch_size=rnn_cfg['batch_size'],
                              shuffle=True,
                              collate_fn=collate_fn,
                              num_workers=rnn_cfg['num_workers'])
    val_loader = DataLoader(val_dataset,
                            batch_size=val_batch_size,
                            shuffle=False,
                            collate_fn=collate_fn,
                            num_workers=rnn_cfg['num_workers'])

    # Initialize Model
    model = rnn_postprocessor.RNNPostProcessor(
        input_size=rnn_cfg['model']['input_size'],
        hidden_size=rnn_cfg['model']['hidden_size'],
        num_layers=rnn_cfg['model']['num_layers'],
        num_classes=rnn_cfg['model']['num_classes'],
        rnn_type=rnn_cfg['model']['type'],
        dropout_prob=rnn_cfg['model']['dropout_prob'],
        bidirectional=rnn_cfg['model']['bidirectional']
    ).to(DEVICE)

    # Loss and Optimizer
    # Use ignore_index=-100 for padded values in labels
    criterion = nn.CrossEntropyLoss(ignore_index=-100)
    optimizer = optim.AdamW(model.parameters(), lr=rnn_cfg['optimizer']['lr'])

    # Optional: Learning rate scheduler - make config more detailed if needed
    scheduler_factor = rnn_cfg['scheduler']['factor']
    # Calculate scheduler patience based on training patience
    scheduler_patience = rnn_cfg['scheduler']['patience']
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min',
                                                     factor=scheduler_factor,
                                                     patience=scheduler_patience,
                                                     verbose=True)

    # Training Loop
    best_val_loss = float('inf')
    epochs_no_improve = 0

    print("Starting Training...")
    for epoch in range(rnn_cfg['epochs']): # Use config epochs
        print(f"--- Epoch {epoch+1}/{rnn_cfg['epochs']} ---")

        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, DEVICE)
        print(f"  Average Training Loss: {train_loss:.4f}")

        val_loss = validate(model, val_loader, criterion, DEVICE)
        print(f"  Average Validation Loss: {val_loss:.4f}")

        # Update LR scheduler
        scheduler.step(val_loss)

        # Save best model
        if val_loss < best_val_loss:
            print(f"  Validation loss improved ({best_val_loss:.4f} -> {val_loss:.4f}) Saving model")
            best_val_loss = val_loss
            # Use Path object and config name
            checkpoint_path = checkpoint_dir / best_checkpoint_name
            save_dict = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': best_val_loss,
                # Store relevant config sections for reproducibility
                'config': {
                    'rnn_training': rnn_cfg,
                    'data': data_cfg # Include data paths used
                 }
            }
            torch.save(save_dict, checkpoint_path)
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            print(f"  Validation loss did not improve Best loss: {best_val_loss:.4f} ({epochs_no_improve}/{rnn_cfg['patience']} epochs without improvement)")

        # Early stopping
        if epochs_no_improve >= rnn_cfg['patience']: # Use config patience
            print(f"Early stopping triggered after {rnn_cfg['patience']} epochs without improvement")
            break

    print("Training finished")
    print(f"Best validation loss achieved: {best_val_loss:.4f}")
    # Use Path object and config name
    print(f"Best model saved to: {checkpoint_dir / best_checkpoint_name}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 3: Train RNN Post-Processor")

    # --- Config File Argument ---
    parser.add_argument("--config", type=str, default="configs/config.yaml",
                        help="Path to the YAML configuration file.")


    args = parser.parse_args()
    main(args) 