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

# Import the RNN model definition
try:
    # Adjust path if your model file is elsewhere
    from models.rnn_postprocessor import RNNPostProcessor 
except ImportError:
    print("Error: Could not import RNNPostProcessor from models/rnn_postprocessor.py")
    print("Please ensure the file exists and is in the correct directory.")
    exit()

# ====== Configuration & Default Hyperparameters ======
# Data paths (assuming data generated in Phase 1)
TRAIN_DATA_DIR = "rnn_processed_data/train"
VAL_DATA_DIR = "rnn_processed_data/val"
CHECKPOINT_DIR = "rnn_checkpoints"

# Model Hyperparameters (Defaults)
INPUT_SIZE = 15       # 3 * NUM_CLASSES (assuming NUM_CLASSES=5)
NUM_CLASSES_OUT = 6 # NUM_CLASSES + 1 (for background)
HIDDEN_SIZE = 128
NUM_LAYERS = 2
DROPOUT_PROB = 0.5
RNN_TYPE = 'lstm'     # 'lstm' or 'gru'
BIDIRECTIONAL = True

# Training Hyperparameters (Defaults)
LEARNING_RATE = 1e-3
BATCH_SIZE = 16      # Adjust based on GPU memory
NUM_EPOCHS = 50
PATIENCE = 5        # For early stopping
NUM_WORKERS = 1      # Dataloader workers

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Ensure checkpoint directory exists
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

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
            raise FileNotFoundError(f"No .npz files found in directory: {data_dir}")
        print(f"Found {len(self.data_files)} data files in {data_dir}")

    def __len__(self):
        return len(self.data_files)

    def __getitem__(self, idx):
        filepath = self.data_files[idx]
        try:
            data = np.load(filepath)
            # Ensure data types are correct for tensors
            features = torch.tensor(data['features'], dtype=torch.float32) # (T, input_size)
            labels = torch.tensor(data['labels'], dtype=torch.long)     # (T,)
            return features, labels
        except Exception as e:
            print(f"Error loading or processing file {filepath}: {e}")
            # Return dummy data or skip? For simplicity, let's return None and handle in collate_fn
            # Or raise error? Raising might be better during debugging.
            raise # Reraise the exception to stop training if a file is corrupted

# ====== Collate Function for Padding ======
def collate_fn(batch):
    """Pads sequences within a batch and returns features, labels, and lengths."""
    # Filter out potential None items if __getitem__ were to return None on error
    # batch = [item for item in batch if item is not None]
    # if not batch: return None, None, None # Handle empty batch
    
    # Separate features and labels
    features_list = [item[0] for item in batch]
    labels_list = [item[1] for item in batch]
    
    # Get sequence lengths BEFORE padding
    lengths = torch.tensor([len(seq) for seq in features_list], dtype=torch.long)
    
    # Pad features (batch_first=True means output shape is B x T x F)
    # Use 0.0 for padding features
    features_padded = pad_sequence(features_list, batch_first=True, padding_value=0.0)
    
    # Pad labels
    # Use -100 for padding labels, as CrossEntropyLoss ignores this index by default
    labels_padded = pad_sequence(labels_list, batch_first=True, padding_value=-100)
    
    return features_padded, labels_padded, lengths

# ====== Training and Validation Functions ======
def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    
    progress_bar = tqdm(dataloader, desc="Training", leave=False)
    for features, labels, lengths in progress_bar:
        if features is None: continue # Skip if collate_fn returned None
        
        features, labels = features.to(device), labels.to(device)
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass
        # Pass lengths only if model uses packing (RNNPostProcessor does)
        logits = model(features, lengths) # Shape: (B, T, C)
        
        # Calculate loss
        # CrossEntropyLoss expects logits as (N, C) and labels as (N)
        # N = total number of non-padded elements = B * T (in padded terms)
        # We need to filter based on padding later if needed, but CEL handles ignore_index
        loss = criterion(logits.view(-1, logits.shape[-1]), labels.view(-1))
        
        # Backward pass and optimize
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
            
            # Forward pass
            logits = model(features, lengths)
            
            # Calculate loss
            loss = criterion(logits.view(-1, logits.shape[-1]), labels.view(-1))
            
            total_loss += loss.item()
            progress_bar.set_postfix(loss=loss.item())
            
    return total_loss / len(dataloader)

# ====== Main Training Script ======
def main(args):
    print(f"Using device: {DEVICE}")
    print("--- Hyperparameters ---")
    print(f"  RNN Type: {args.rnn_type}")
    print(f"  Hidden Size: {args.hidden_size}")
    print(f"  Num Layers: {args.num_layers}")
    print(f"  Bidirectional: {args.bidirectional}")
    print(f"  Dropout: {args.dropout_prob}")
    print(f"  Batch Size: {args.batch_size}")
    print(f"  Learning Rate: {args.lr}")
    print(f"  Epochs: {args.epochs}")
    print(f"-----------------------")

    # Load Datasets
    try:
        train_dataset = RNNDataset(TRAIN_DATA_DIR)
        val_dataset = RNNDataset(VAL_DATA_DIR)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please ensure Phase 1 (1_generate_rnn_data.py) completed successfully and data exists.")
        exit()
        
    # Create DataLoaders
    train_loader = DataLoader(train_dataset, 
                              batch_size=args.batch_size, 
                              shuffle=True, 
                              collate_fn=collate_fn, 
                              num_workers=args.num_workers)
    val_loader = DataLoader(val_dataset, 
                            batch_size=args.batch_size * 2, # Often use larger batch for validation
                            shuffle=False, 
                            collate_fn=collate_fn, 
                            num_workers=args.num_workers)

    # Initialize Model
    model = RNNPostProcessor(
        input_size=INPUT_SIZE,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        num_classes=NUM_CLASSES_OUT,
        rnn_type=args.rnn_type,
        dropout_prob=args.dropout_prob,
        bidirectional=args.bidirectional
    ).to(DEVICE)

    # Loss and Optimizer
    # Use ignore_index=-100 for padded values in labels
    criterion = nn.CrossEntropyLoss(ignore_index=-100) 
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    # Optional: Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=args.patience // 2, verbose=True)

    # Training Loop
    best_val_loss = float('inf')
    epochs_no_improve = 0

    print("\nStarting Training...")
    for epoch in range(args.epochs):
        print(f"\n--- Epoch {epoch+1}/{args.epochs} ---")
        
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, DEVICE)
        print(f"  Average Training Loss: {train_loss:.4f}")
        
        val_loss = validate(model, val_loader, criterion, DEVICE)
        print(f"  Average Validation Loss: {val_loss:.4f}")
        
        # Update LR scheduler
        scheduler.step(val_loss)
        
        # Save best model
        if val_loss < best_val_loss:
            print(f"  Validation loss improved ({best_val_loss:.4f} -> {val_loss:.4f}). Saving model...")
            best_val_loss = val_loss
            checkpoint_path = os.path.join(CHECKPOINT_DIR, 'best_rnn_model.pth')
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': best_val_loss,
                # Store hyperparameters used for this checkpoint
                'args': vars(args) 
            }, checkpoint_path)
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            print(f"  Validation loss did not improve. Best loss: {best_val_loss:.4f} ({epochs_no_improve}/{args.patience} epochs without improvement)")
            
        # Early stopping
        if epochs_no_improve >= args.patience:
            print(f"\nEarly stopping triggered after {args.patience} epochs without improvement.")
            break

    print("\nTraining finished.")
    print(f"Best validation loss achieved: {best_val_loss:.4f}")
    print(f"Best model saved to: {os.path.join(CHECKPOINT_DIR, 'best_rnn_model.pth')}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 3: Train RNN Post-Processor")
    
    # Model args
    parser.add_argument("--rnn_type", type=str, default=RNN_TYPE, choices=['lstm', 'gru'], help="Type of RNN layer")
    parser.add_argument("--hidden_size", type=int, default=HIDDEN_SIZE, help="Number of hidden units in RNN")
    parser.add_argument("--num_layers", type=int, default=NUM_LAYERS, help="Number of RNN layers")
    parser.add_argument("--dropout_prob", type=float, default=DROPOUT_PROB, help="Dropout probability")
    parser.add_argument("--bidirectional", action=argparse.BooleanOptionalAction, default=BIDIRECTIONAL, help="Use bidirectional RNN")

    # Training args
    parser.add_argument("--lr", type=float, default=LEARNING_RATE, help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE, help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=NUM_EPOCHS, help="Maximum number of training epochs")
    parser.add_argument("--patience", type=int, default=PATIENCE, help="Patience for early stopping")
    parser.add_argument("--num_workers", type=int, default=NUM_WORKERS, help="Number of dataloader workers")

    args = parser.parse_args()
    main(args) 