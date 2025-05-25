import torch
import argparse
import pickle
from tqdm import tqdm
import copy
import yaml
from pathlib import Path

import src.utils.helpers as helpers

try:
    from src.models.TemporalActionDetector_v5_1 import TemporalActionDetector
except ImportError:
    print("Could not import TemporalActionDetector from src models model_fixed Make sure the file exists")
    exit()
try:
    import src.dataloader as dataloader
except ImportError:
    print("Could not import get_train_loader/get_val_loader/get_test_loader from src dataloader")
    exit()
try:
    import src.losses as losses
except ImportError:
    print("ActionDetectionLoss not found")
    exit()

def run_inference(model, data_loader, device, use_mixed_precision):
    model.eval()
    all_raw_preds = []
    all_batch_meta = []
    
    print(f"Running inference on {len(data_loader.dataset)} samples...")
    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Model Inference"):
            try:
                 mvit_feat, resnet_feat, pose_feat, _, _, _, metadata = batch
            except ValueError:
                print("Error: Batch structure mismatch in dataloader. Check dataloader output.")

            mvit_feat = mvit_feat.to(device)
            resnet_feat = resnet_feat.to(device)
            if pose_feat is not None:
                pose_feat = pose_feat.to(device)
            with torch.cuda.amp.autocast(enabled=use_mixed_precision):
                predictions = model(mvit_feat, resnet_feat, pose_feat)

            action_probs = torch.sigmoid(predictions['action_scores']).cpu().detach()
            start_probs = torch.sigmoid(predictions['start_scores']).cpu().detach()
            end_probs = torch.sigmoid(predictions['end_scores']).cpu().detach()

            all_raw_preds.append((action_probs, start_probs, end_probs))
            all_batch_meta.append(copy.deepcopy(metadata))
            
    return all_raw_preds, all_batch_meta

def main(cfg, args):
    if cfg['global']['device'] == 'auto':
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(cfg['global']['device'])

    num_classes = cfg['global']['num_classes']
    window_size = cfg['global']['window_size']
    dl_cfg = cfg['rnn_data_generation']['dataloader']
    data_cfg = cfg['data']

    checkpoint_path = args.checkpoint_path if args.checkpoint_path else cfg['rnn_data_generation']['base_checkpoint_to_use']

    log_dir = Path(data_cfg['logs'])
    train_pkl_path = log_dir / data_cfg['train_inference_raw_name']
    val_pkl_path = log_dir / data_cfg['val_inference_raw_name']
    test_pkl_path = log_dir / data_cfg['test_inference_raw_name']
    processed_dir = Path(data_cfg['processed_dir'])
    train_anno_dir = processed_dir / "train" / "annotations"
    val_anno_dir = processed_dir / "val" / "annotations"
    test_anno_dir = processed_dir / "test" / "annotations"
    rnn_base_dir = Path(data_cfg['rnn_processed_data'])
    rnn_train_data_dir = rnn_base_dir / "train"
    rnn_val_data_dir = rnn_base_dir / "val"
    rnn_test_data_dir = rnn_base_dir / "test"

    log_dir.mkdir(parents=True, exist_ok=True)
    rnn_train_data_dir.mkdir(parents=True, exist_ok=True)
    rnn_val_data_dir.mkdir(parents=True, exist_ok=True)
    rnn_test_data_dir.mkdir(parents=True, exist_ok=True)

    print(f"Using device: {device}")
    print(f"Loading base model checkpoint: {checkpoint_path}")

    model = TemporalActionDetector(num_classes=num_classes, window_size=window_size)
    model = model.to(device)

    if Path(checkpoint_path).exists():
        print(f"Loading checkpoint")
        try:
            checkpoint = torch.load(checkpoint_path, map_location=device)
            if 'model_state_dict' in checkpoint: state_dict = checkpoint['model_state_dict']
            elif 'state_dict' in checkpoint: state_dict = checkpoint['state_dict']
            else: state_dict = checkpoint
            model.load_state_dict(state_dict)
            print(f"Successfully loaded model weights from epoch {checkpoint.get('epoch', 'N/A')}")
        except Exception as e:
            print(f"loading checkpoint: {e} Ensure valid path and compatible model")
            exit()
    else:
        print(f"Checkpoint file not found at {checkpoint_path}")
        exit()
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}")
    print("Preparing Training Set Inference")

    train_loader = dataloader.get_train_loader(cfg)
    train_raw_preds, train_batch_meta = run_inference(model, train_loader, device)
    print(f"\nSaving training inference results to: {train_pkl_path}")
    try:
        with open(train_pkl_path, 'wb') as f:
            pickle.dump({'all_raw_preds': train_raw_preds, 'all_batch_meta': train_batch_meta}, f)
        print("Successfully saved training inference results")
    except Exception as e: print(f"saving training inference results: {e}")

    print("Preparing Validation Set Inference")
    val_loader = dataloader.get_val_loader(cfg)
    val_raw_preds, val_batch_meta = run_inference(model, val_loader, device)
    print(f"\nSaving validation inference results to: {val_pkl_path}")
    try:
        with open(val_pkl_path, 'wb') as f:
            pickle.dump({'all_raw_preds': val_raw_preds, 'all_batch_meta': val_batch_meta}, f)
        print("Successfully saved validation inference results")
    except Exception as e: print(f"saving validation inference results: {e}")

    print("Preparing Test Set Inference")
    test_loader = dataloader.get_test_loader(cfg)
    test_raw_preds, test_batch_meta = run_inference(model, test_loader, device)
    print(f"\nSaving test inference results to: {test_pkl_path}")
    try:
        with open(test_pkl_path, 'wb') as f:
            pickle.dump({'all_raw_preds': test_raw_preds, 'all_batch_meta': test_batch_meta}, f)
        print("Successfully saved test inference results")
    except Exception as e: print(f"saving test inference results: {e}")

    print("Processing raw data into RNN input format")

    print("Processing Training Data for RNN")
    helpers.process_predictions_for_rnn(
        num_classes=num_classes,
        window_size=window_size,
        output_pkl=train_pkl_path,
        anno_dir=train_anno_dir,
        rnn_data_dir=rnn_train_data_dir
    )

    print("Processing Validation Data for RNN")
    helpers.process_predictions_for_rnn(
        num_classes=num_classes,
        window_size=window_size,
        output_pkl=val_pkl_path,
        anno_dir=val_anno_dir,
        rnn_data_dir=rnn_val_data_dir
    )
    
    print("Processing Test Data for RNN")
    helpers.process_predictions_for_rnn(
        num_classes=num_classes,
        window_size=window_size,
        output_pkl=test_pkl_path,
        anno_dir=test_anno_dir,
        rnn_data_dir=rnn_test_data_dir
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate raw data for RNN Post-Processor using a trained base model.")
    parser.add_argument('--config', default='configs/config.yaml', help='Path to configuration file')
    parser.add_argument("--checkpoint_path", type=str, default=None, help="Override base model checkpoint path from config")
    
    args = parser.parse_args()
    
    try:
        with open(args.config, 'r') as f:
            cfg = yaml.safe_load(f)
    except FileNotFoundError:
        print(f"Config file not found at {args.config}")
        exit()
    except Exception as e:
        print(f"loading config file: {e}")
        exit()
        
    main(cfg, args) 