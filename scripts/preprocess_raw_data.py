import os
import cv2
import numpy as np
from scipy.io import loadmat
import json
from src.utils.helpers import find_nearest_subsampled_idx

VIDEO_DIR = r"Data\Videos_MERL_Shopping_Dataset\train"
LABEL_DIR = r"Data\Labels_MERL_Shopping_Dataset\train"
OUTPUT_DIR = r"Data\full_videos\train"
FRAME_SIZE = 224
SUBSAMPLE = 2

def prepare_full_video(video_path, label_path):
    """Process full video and save with temporal action annotations"""

    tlabs = loadmat(label_path)["tlabs"].ravel() 
    video_id = os.path.basename(video_path)[:-4]

    video_folder = os.path.join(OUTPUT_DIR, "frames")
    annotation_folder = os.path.join(OUTPUT_DIR, "annotations")
    os.makedirs(video_folder, exist_ok=True)
    os.makedirs(annotation_folder, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(cap.get(cv2.CAP_PROP_FPS))


    print(
        f"[INFO] Video {video_id} has {total_frames} frames. Subsampling by factor of {SUBSAMPLE}."
    )

    frames = []
    frame_indices = []
    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if idx % SUBSAMPLE == 0:
            resized_frame = cv2.resize(frame, (FRAME_SIZE, FRAME_SIZE))
            frames.append(resized_frame)
            frame_indices.append(int(idx))

        idx += 1
    cap.release()

    video_array = np.stack(frames)

    npz_path = os.path.join(video_folder, f"{video_id}_frames.npz")
    np.savez_compressed(npz_path, frames=video_array)

    action_annotations = []
    for action_idx, segments in enumerate(tlabs, start=1):
        for start, end in segments:
            start_frame = int(start)
            end_frame = int(end)

            sub_start = find_nearest_subsampled_idx(start_frame, frame_indices)
            sub_end = find_nearest_subsampled_idx(end_frame, frame_indices)

            if sub_end > sub_start:
                action_annotations.append(
                    {
                        "action_id": int(action_idx - 1),
                        "action_name": f"action{action_idx}",
                        "start_frame": int(sub_start),
                        "end_frame": int(sub_end),
                        "start_time": float(start_frame / fps),
                        "end_time": float(end_frame / fps),
                        "original_start": int(start_frame),
                        "original_end": int(end_frame),
                    }
                )

    annotation_data = {
        "video_id": video_id,
        "num_frames": int(len(frames)),
        "original_frames": int(total_frames),
        "fps": float(fps),
        "subsample_factor": int(SUBSAMPLE),
        "frame_indices": [int(idx) for idx in frame_indices],
        "annotations": action_annotations,
        "frames_file": f"{video_id}_frames.npz",
    }

    with open(os.path.join(annotation_folder, f"{video_id}_annotations.json"), "w") as f:
        json.dump(annotation_data, f, indent=2)

    # Calculate file size reduction
    video_size_mb = os.path.getsize(npz_path) / (1024 * 1024)
    print(f"[INFO] Compressed video size: {video_size_mb:.2f} MB")

    return len(frames), len(action_annotations)

def main():
    """Process all videos in the dataset"""
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    total_videos = 0
    total_frames = 0
    total_actions = 0

    for mat in sorted(os.listdir(LABEL_DIR)):
        if not mat.endswith(".mat"):
            continue

        base_name = os.path.splitext(mat)[0]
        if "_label" in base_name:
            base_name = base_name.replace("_label", "")

        video_file = f"{base_name}_crop.mp4"
        video_path = os.path.join(VIDEO_DIR, video_file)
        label_path = os.path.join(LABEL_DIR, mat)

        if not os.path.exists(video_path):
            print(f"missing video: {video_file}")
            continue

        num_frames, num_actions = prepare_full_video(video_path, label_path)

       
        total_videos += 1
        total_frames += num_frames
        total_actions += num_actions

    dataset_stats = {
        "total_videos": int(total_videos),
        "total_frames": int(total_frames),
        "total_actions": int(total_actions),
        "frame_size": int(FRAME_SIZE),
        "file_format": "npz_compressed",
    }

    with open(os.path.join(OUTPUT_DIR, "dataset_stats.json"), "w") as f:
        json.dump(dataset_stats, f, indent=2)

if __name__ == "__main__":
    main()
