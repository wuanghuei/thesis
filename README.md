# Temporal Action Detection Project

## Introduction

This project aims to detect temporal action segments in videos. It utilizes a multi-stream architecture combining information from RGB (using MViT and a 2D CNN) and Pose data (using LSTM). Subsequently, an RNN model (LSTM/GRU) is employed for post-processing the predictions from the main model to refine the final results.

This project is based on the MERL Shopping Dataset.

## Directory Structure

```
.
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

## Usage

Run the scripts in the `scripts/` directory in the following order:

1.  **Preprocess Data:** Convert original videos to the required format.
    ```bash
    python scripts/prepare_segments.py
    ```
    *(Note: This script currently hardcodes paths for the training set. You might need to modify it to handle test/val sets or add command-line arguments).*

2.  **Extract Pose Features:**
    ```bash
    python scripts/extract_pose_features.py --split all # Or --split train/test
    ```

3.  **Train Base Model:**
    ```bash
    python scripts/train_base_model.py # Parameters can be customized within the file
    ```
    *   The best checkpoint will be saved in `checkpoints/`.

4.  **Generate RNN Data:** Use the base model checkpoint to create input for the RNN.
    ```bash
    python scripts/generate_rnn_data.py --base_checkpoint_path checkpoints/best_model_velocity.pth # Or another checkpoint
    ```
    *   Data will be saved in `rnn_processed_data/`. A pickle file containing raw inference results will also be created (e.g., `train_inference_raw.pkl`, `val_inference_raw.pkl`).

5.  **Train RNN Post-processor:**
    ```bash
    python scripts/train_rnn.py # Parameters can be customized via command line or in the file
    ```
    *   The best RNN checkpoint will be saved in `rnn_checkpoints/`.

6.  **Evaluate Pipeline:** Evaluate the combined performance of the base model and RNN.
    ```bash
    python scripts/evaluate_pipeline.py --rnn_checkpoint_path rnn_checkpoints/best_rnn_model.pth --inference_output_path val_inference_raw.pkl # Or the test set pkl file
    ```
    *   This script requires the inference results file (`.pkl`) from step 4 and the RNN checkpoint from step 5.
    *   Parameters can be added to visualize predictions for a specific video.

## Configuration

Currently, many configurations (paths, hyperparameters) are set directly within the script files. Consider moving them to separate configuration files (e.g., YAML) in the future for easier management.
