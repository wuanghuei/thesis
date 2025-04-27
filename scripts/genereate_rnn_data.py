import torch
import os
import argparse
import pickle
from tqdm import tqdm
import numpy as np
from collections import defaultdict
import copy
import torch.nn as nn # Might be needed for loading model
import json # Needed for loading annotations
from src.utils.helpers import process_predictions_for_rnn

# Import necessary components from other project files
# Assuming these files are in the same root directory
try:
    from src.models.base_detector import TemporalActionDetector
except ImportError:
    print("Error: Could not import TemporalActionDetector from model_fixed.py. Make sure the file exists.")
    exit()
try:
    from src.dataloader import get_train_loader, get_val_loader, get_test_loader # Need both loaders
except ImportError:
    print("Error: Could not import get_train_loader/get_val_loader from dataloader.py.")
    exit()
try:
    # Import Loss class, might be needed for checkpoint loading if criterion state was saved
    from src.losses import ActionDetectionLoss 
except ImportError:
    # If not found, create a dummy class to avoid errors if checkpoint includes it
    print("Warning: ActionDetectionLoss not found in train_fixed.py. Using a dummy class.")
    class ActionDetectionLoss: pass 

# ====== Configuration ======
# These should ideally match the settings used during training/evaluation
NUM_CLASSES = 5
WINDOW_SIZE = 32 # Or load from checkpoint if saved there
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_MIXED_PRECISION = True # Should match training
DEFAULT_CHECKPOINT = "checkpoints/interim_model_epoch18.pth"
TRAIN_OUTPUT_PKL = "train_inference_raw.pkl"
VAL_OUTPUT_PKL = "val_inference_raw.pkl"
TEST_OUTPUT_PKL = "test_inference_raw.pkl"
# Dataloader params (adjust if needed)
TRAIN_BATCH_SIZE = 4 # Can use a larger batch size for inference
VAL_BATCH_SIZE = 8  
TEST_BATCH_SIZE = 8
NUM_WORKERS = 0 # Set based on your system
# Constants needed for label generation

# --- Helper Functions for Data Generation (Phase 1.2) ---





# --- End Helper Functions ---

def run_inference(model, data_loader, device, use_mixed_precision):
    """Runs inference on the provided dataloader and returns raw predictions and metadata."""
    model.eval()
    all_raw_preds = []
    all_batch_meta = []
    
    print(f"Running inference on {len(data_loader.dataset)} samples...")
    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Model Inference"):
            # Assuming batch structure: frames, pose, hand, masks..., metadata
            # Adjust unpacking based on your actual dataloader output
            try:
                frames, pose_data, _, _, _, _, metadata = batch # Unpack only what's needed for model input + metadata
            except ValueError:
                print("Error: Batch structure mismatch in dataloader. Check dataloader output.")
                # Attempt to unpack differently if structure changed
                try: 
                     frames, pose_data, metadata = batch # Example if only these 3 are returned
                     print("Warning: Assuming simplified batch structure (frames, pose, metadata)")
                except ValueError:
                     print("Fatal: Cannot determine batch structure. Exiting.")
                     exit()


            frames = frames.to(device)
            if pose_data is not None: # Handle cases where pose might be optional
                 pose_data = pose_data.to(device)

            with torch.cuda.amp.autocast(enabled=use_mixed_precision):
                predictions = model(frames, pose_data) # Pass only necessary inputs

            # Detach, move to CPU, convert to numpy (optional here, can keep as tensors)
            # Storing as tensors might save memory if converting to numpy is not immediately needed
            action_probs = torch.sigmoid(predictions['action_scores']).cpu().detach()
            start_probs = torch.sigmoid(predictions['start_scores']).cpu().detach()
            end_probs = torch.sigmoid(predictions['end_scores']).cpu().detach()

            all_raw_preds.append((action_probs, start_probs, end_probs))
            all_batch_meta.append(copy.deepcopy(metadata)) # Deep copy metadata
            
    return all_raw_preds, all_batch_meta

def main(args):
    print(f"Using device: {DEVICE}")
    print(f"Loading checkpoint: {args.checkpoint_path}")

    # Initialize model
    model = TemporalActionDetector(num_classes=NUM_CLASSES, window_size=WINDOW_SIZE) 
    model = model.to(DEVICE)

    # Load checkpoint
    if os.path.exists(args.checkpoint_path):
        print(f"Loading checkpoint...")
        try:
            # Add map_location to handle loading CUDA-trained models on CPU-only systems
            checkpoint = torch.load(args.checkpoint_path, map_location=DEVICE) 
            
            # Handle different checkpoint saving conventions
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            elif 'state_dict' in checkpoint:
                 state_dict = checkpoint['state_dict']
            else:
                 state_dict = checkpoint # Assume the checkpoint object IS the state_dict

            # Load the state dict
            model.load_state_dict(state_dict)
            print(f"Successfully loaded model weights from epoch {checkpoint.get('epoch', 'N/A')}")
        except Exception as e:
            print(f"Error loading checkpoint: {e}")
            print("Please ensure the checkpoint file is valid and compatible with the model.")
            exit()
    else:
        print(f"Error: Checkpoint file not found at {args.checkpoint_path}")
        exit()

    # --- Run Inference on Training Set ---
    print("\nPreparing Training Set Inference...")
    train_loader = get_train_loader(batch_size=TRAIN_BATCH_SIZE, shuffle=False)
    train_raw_preds, train_batch_meta = run_inference(model, train_loader, DEVICE, USE_MIXED_PRECISION)
    
    # Save training results
    print(f"\nSaving training inference results to: {TRAIN_OUTPUT_PKL}...")
    train_inference_results = {
        'all_raw_preds': train_raw_preds,
        'all_batch_meta': train_batch_meta,
    }
    try:
        with open(TRAIN_OUTPUT_PKL, 'wb') as f:
            pickle.dump(train_inference_results, f)
        print("Successfully saved training inference results.")
    except Exception as e:
        print(f"Error saving training inference results: {e}")

    # --- Run Inference on Test Set ---
    print("\nPreparing Test Set Inference...")
    test_loader = get_test_loader(batch_size=TEST_BATCH_SIZE, shuffle=False)
    test_raw_preds, test_batch_meta = run_inference(model, test_loader, DEVICE, USE_MIXED_PRECISION)

    # Save test results
    print(f"\nSaving test inference results to: {TEST_OUTPUT_PKL}...")  
    test_inference_results = {
        'all_raw_preds': test_raw_preds,
        'all_batch_meta': test_batch_meta,
    }
    try:
        with open(TEST_OUTPUT_PKL, 'wb') as f:
            pickle.dump(test_inference_results, f)
        print("Successfully saved test inference results.")
    except Exception as e:
        print(f"Error saving test inference results: {e}")

    # --- Run Inference on Validation Set ---
    print("\nPreparing Validation Set Inference...")
    val_loader = get_val_loader(batch_size=VAL_BATCH_SIZE, shuffle=False)
    val_raw_preds, val_batch_meta = run_inference(model, val_loader, DEVICE, USE_MIXED_PRECISION)

    # Save validation results
    print(f"\nSaving validation inference results to: {VAL_OUTPUT_PKL}...")
    val_inference_results = {
        'all_raw_preds': val_raw_preds,
        'all_batch_meta': val_batch_meta,
    }
    try:
        with open(VAL_OUTPUT_PKL, 'wb') as f:
            pickle.dump(val_inference_results, f)
        print("Successfully saved validation inference results.")
    except Exception as e:
        print(f"Error saving validation inference results: {e}")

    print("\nPhase 1.1 (Inference) Completed.")

    # ====== Phase 1.2: Process and Save Data for RNN ====== 
    print("\nStarting Phase 1.2: Processing raw data into RNN input format...")

    # Define annotation directories (confirm these are correct)
    TRAIN_ANNO_DIR = "Data/full_videos/train/annotations"
    VAL_ANNO_DIR = "Data/full_videos/val/annotations"
    TEST_ANNO_DIR = "Data/full_videos/test/annotations"

    # Define output directories for processed data
    RNN_TRAIN_DATA_DIR = "rnn_processed_data/train"
    RNN_VAL_DATA_DIR = "rnn_processed_data/val"
    RNN_TEST_DATA_DIR = "rnn_processed_data/test"

    os.makedirs(RNN_TRAIN_DATA_DIR, exist_ok=True)
    os.makedirs(RNN_VAL_DATA_DIR, exist_ok=True)
    os.makedirs(RNN_TEST_DATA_DIR, exist_ok=True)

    # --- Process Training Data ---
    process_predictions_for_rnn(train_raw_preds, NUM_CLASSES, WINDOW_SIZE, TRAIN_OUTPUT_PKL, TRAIN_ANNO_DIR, RNN_TRAIN_DATA_DIR)
    process_predictions_for_rnn(val_raw_preds, NUM_CLASSES, WINDOW_SIZE, VAL_OUTPUT_PKL, VAL_ANNO_DIR, RNN_VAL_DATA_DIR)
    process_predictions_for_rnn(test_raw_preds, NUM_CLASSES, WINDOW_SIZE, TEST_OUTPUT_PKL, TEST_ANNO_DIR, RNN_TEST_DATA_DIR)

    print("\nPhase 1.2 (Data Generation for RNN) Completed.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 1.1: Run inference on train/val sets to generate raw data for RNN Post-Processor.")
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default=DEFAULT_CHECKPOINT,
        help=f"Path to the base model checkpoint file (.pth). Default: {DEFAULT_CHECKPOINT}"
    )
    # Add other arguments if needed (e.g., batch_size, num_workers)
    
    args = parser.parse_args()
    main(args) 