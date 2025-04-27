# Temporal Action Detection Project

## Introduction

This project aims to detect temporal action segments in videos. It utilizes a multi-stream architecture combining information from RGB (using MViT and a 2D CNN) and Pose data (using LSTM). Subsequently, an RNN model (LSTM/GRU) is employed for post-processing the predictions from the main model to refine the final results.

This project is based on the MERL Shopping Dataset.

## Directory Structure

```
.
├── configs/                  # Configuration files
│   └── config.yaml           # Main configuration file (paths, hyperparameters)
├── Data/                     # Input and processed data (requires specific structure)
│   ├── Videos_MERL_Shopping_Dataset/ # Original videos (train/test)
│   ├── Labels_MERL_Shopping_Dataset/ # Original labels (train/test)
│   └── full_videos/          # Processed data (frames npz, annotations json, pose npz)
│       ├── train/
│       └── test/ (or val/)
├── checkpoints/              # Stores base model checkpoint files
├── rnn_checkpoints/          # Stores RNN model checkpoint files
├── rnn_processed_data/       # Processed data for RNN (train/val)
├── logs/                     # Stores training logs and test results
├── src/                      # Main source code (model definitions, dataloader, utils, ...)
│   ├── models/               # Model class definitions
│   ├── utils/                # Utility functions (helpers, metrics, postprocessing, ...)
│   ├── dataloader.py         # Dataset and DataLoader definitions for the base model
│   ├── evaluation.py         # Function for calculating aggregate metrics
│   └── losses.py             # Loss function class definitions
├── scripts/                  # Scripts to run different steps of the pipeline
│   ├── prepare_segments.py   # Preprocess original videos -> frames npz, annotations json
│   ├── extract_pose_features.py # Extract pose features -> pose npz
│   ├── train_base_model.py   # Train the base TemporalActionDetector model
│   ├── generate_rnn_data.py  # Generate RNN input data from base model results
│   ├── train_rnn.py          # Train the RNN post-processor model
│   └── evaluate_pipeline.py  # Evaluate the entire pipeline (base + RNN)
├── requirements.txt          # List of required Python libraries
└── README.md                 # This file
```

## Installation

1.  **Create a virtual environment (Recommended):**
    ```bash
    python -m venv venv
    # Windows
    venv\Scripts\activate
    # macOS/Linux
    source venv/bin/activate
    ```
2.  **Install libraries:**
    ```bash
    pip install -r requirements.txt
    ```
3.  **Prepare data:**
    *   Place the original MERL Shopping Dataset videos and labels into `Data/Videos_MERL_Shopping_Dataset` and `Data/Labels_MERL_Shopping_Dataset` respectively, following the train/test structure.
    *   The preprocessing scripts will generate data in `Data/full_videos`.
4.  **Configure the pipeline:**
    *   Edit `configs/config.yaml` to set paths, hyperparameters, and other settings for the different components (data loading, base model, RNN model, training parameters). Refer to the comments within the file for guidance.

## Usage

Run the scripts in the `scripts/` directory in the following order:

1.  **Preprocess Data:** Convert original videos to the required format.
    ```bash
    python scripts/prepare_segments.py
    ```
    *(Note: This script currently hardcodes paths for the training set. You might need to modify it to handle test/val sets or add command-line arguments based on `configs/config.yaml` if desired).*

2.  **Extract Pose Features:**
    ```bash
    python scripts/extract_pose_features.py --split all # Or --split train/test (Reads config for paths)
    ```

3.  **Train Base Model:** Uses settings from `configs/config.yaml`.
    ```bash
    python scripts/train_base_model.py # Optional: --config path/to/alternative/config.yaml
    ```
    *   Checkpoints are saved based on the `checkpoint_dir` in the config.

4.  **Generate RNN Data:** Use the base model checkpoint to create input for the RNN. Reads paths from config.
    ```bash
    python scripts/generate_rnn_data.py # Optional: --config path/to/config.yaml --base_checkpoint_path path/to/model.pth
    ```
    *   Requires `base_checkpoint_path` to be specified either in the config or via CLI.
    *   Data will be saved in `rnn_processed_data/`. A pickle file containing raw inference results will also be created (e.g., `train_inference_raw.pkl`, `val_inference_raw.pkl`).

5.  **Train RNN Post-processor:** Uses settings from `configs/config.yaml`.
    ```bash
    python scripts/train_rnn.py # Optional: --config path/to/config.yaml
    ```
    *   Checkpoints are saved based on the `checkpoint_dir` in the config (under `data`).

6.  **Evaluate Pipeline:** Evaluate the combined performance of the base model and RNN.
    ```bash
    python scripts/evaluate_pipeline.py # Optional: --config path/to/config.yaml --rnn_checkpoint_path path/to/rnn_model.pth --inference_output_path path/to/inference.pkl
    ```
    *   This script requires the inference results file (`.pkl`) from step 4 and the RNN checkpoint from step 5, specified either in the config or via CLI.
    *   Parameters can be added to visualize predictions for a specific video.

## Configuration

Project configuration (data paths, hyperparameters for base and RNN models, training settings) is managed through the `configs/config.yaml` file.

Key sections include:
*   `global`: Global settings like number of classes.
*   `data`: Paths for datasets, checkpoints, logs, etc.
*   `base_model`: Architecture details for the base temporal action detector.
*   `base_model_training`: Hyperparameters for training the base model.
*   `rnn_model`: Architecture details for the RNN post-processor.
*   `rnn_training`: Hyperparameters for training the RNN model.

Command-line arguments can be used with some scripts (e.g., `train_base_model.py`, `train_rnn.py`) to override specific settings defined in the config file. Run scripts with `-h` or `--help` to see available command-line options.
