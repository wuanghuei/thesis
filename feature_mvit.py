import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import os
from tqdm import tqdm
import src.utils.helpers as helpers

# --- Cấu hình ---
BASE_DATA_DIR = Path('Data/full_videos/') # Điều chỉnh đến thư mục dữ liệu chính của bạn
FRAMES_SUBDIR = 'frames' # Thư mục con chứa các frame video (ví dụ: video_id_frames.npz)
OUTPUT_FEATURES_DIR = Path('Data/features/mvit_32f_8s/') # Thư mục để lưu các đặc trưng MViT đã trích xuất
WINDOW_SIZE = 32
STRIDE = 16
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_video_frames(video_id, frames_dir):
    frames_path = frames_dir / f"{video_id}_frames.npz"
    if not frames_path.exists():
        print(f" Cant find {video_id} at {frames_path}")
        return None
    try:
        data = np.load(frames_path)
        frames = data['frames']
        if frames.ndim == 4 and frames.shape[-1] in [1, 3]: # (T, H, W, C)
            frames = np.transpose(frames, (3, 0, 1, 2)) # (C, T, H, W)
        elif frames.ndim == 4 and frames.shape[1] in [1,3] and frames.shape[0] > frames.shape[1]: # (T, C, H, W)
             frames = np.transpose(frames, (1, 0, 2, 3))
        elif frames.ndim == 3:
            frames = np.expand_dims(frames, axis=0)
        else:
            raise ValueError(f"{video_id} required shapeshape: {frames.shape}")

        frames_tensor = torch.from_numpy(frames).float() / 255.0
        return frames_tensor # Hình dạng: (C, Total_Frames, H, W)
    except Exception as e:
        print(f"Error while loadingloading {video_id}: {e}")
        return None

def main():
    print(f"Sử dụng thiết bị: {DEVICE}")
    OUTPUT_FEATURES_DIR.mkdir(parents=True, exist_ok=True)

    print("Đang tải mô hình MViT...")
    # Khởi tạo model đầy đủ để dễ dàng truy cập backbone
    # Bạn có thể cần điều chỉnh num_classes hoặc các tham số khác nếu chúng ảnh hưởng đến việc tải backbone,
    # tuy nhiên, đối với việc trích xuất đặc trưng, chỉ phần backbone được sử dụng.
    mvit_backbone = torch.hub.load('facebookresearch/pytorchvideo', "mvit_base_32x3", pretrained=True)
    mvit_backbone.head = nn.Identity()
    mvit_backbone.eval()
    mvit_backbone.to(DEVICE)
    print("Đã tải mô hình MViT.")

    # Xử lý cho các tập train, val, test
    for dataset_type in ['train', 'val', 'test']: # Thêm 'val', 'test' nếu có
        frames_dir = BASE_DATA_DIR / dataset_type / FRAMES_SUBDIR
        output_dir_type = OUTPUT_FEATURES_DIR / dataset_type
        output_dir_type.mkdir(parents=True, exist_ok=True)

        if not frames_dir.exists():
            print(f"Thư mục frames {frames_dir} không tồn tại. Bỏ qua.")
            continue

        video_frame_files = [f for f in frames_dir.iterdir() if f.name.endswith("_frames.npz")]
        if not video_frame_files:
            print(f"Không tìm thấy tệp frame video nào trong {frames_dir}")
            continue

        print(f"Tìm thấy {len(video_frame_files)} video để xử lý trong {frames_dir}.")

        for frame_file_path in tqdm(video_frame_files, desc=f"Extracting for ({dataset_type})"):
            video_id = frame_file_path.name.replace("_frames.npz", "")
            output_feature_path = output_dir_type / f"{video_id}_features.npz"

            if output_feature_path.exists():
                print(f"Đặc trưng cho {video_id} ({dataset_type}) đã tồn tại. Bỏ qua.")
                continue

            video_frames_tensor = load_video_frames(video_id, frames_dir)
            if video_frames_tensor is None:
                continue

            C, total_frames, H, W = video_frames_tensor.shape
            video_features_list = []

            with torch.no_grad():
                for start_idx in tqdm(range(0, total_frames, STRIDE), desc=f"Each windows") :
                    end_idx = start_idx + WINDOW_SIZE
                    if end_idx > total_frames: # Nếu cửa sổ cuối cùng ngắn hơn WINDOW_SIZE
                        # Tùy chọn: bỏ qua, đệm (padding), hoặc xử lý cửa sổ ngắn hơn
                        # Ở đây, chúng ta sẽ đệm frame cuối cùng để đủ WINDOW_SIZE
                        if start_idx >= total_frames : # đã xử lý hết
                            continue
                        actual_window_frames = video_frames_tensor[:, start_idx:total_frames, :, :]
                        num_actual_frames = actual_window_frames.shape[1]
                        if num_actual_frames == 0:
                            continue
                        padding_needed = WINDOW_SIZE - num_actual_frames
                        if padding_needed > 0:
                            last_frame = actual_window_frames[:, -1:, :, :].clone() # Lấy frame cuối cùng
                            padding = last_frame.repeat(1, padding_needed, 1, 1)
                            window_frames = torch.cat((actual_window_frames, padding), dim=1)
                        else: # trường hợp num_actual_frames == WINDOW_SIZE (xảy ra khi stride làm start_idx_cuối + WIN_SIZE > total_frames, nhưng total_frames - start_idx_cuối == WIN_SIZE)
                             window_frames = actual_window_frames
                    else:
                        window_frames = video_frames_tensor[:, start_idx:end_idx, :, :]

                    if window_frames.shape[1] != WINDOW_SIZE: # Kiểm tra lại kích thước cửa sổ
                         print(f"Kích thước cửa sổ không chính xác cho {video_id} ({start_idx}-{end_idx}): {window_frames.shape[1]}, dự kiến {WINDOW_SIZE}. Bỏ qua cửa sổ này.")
                         continue


                    window_frames_batch = window_frames.unsqueeze(0).to(DEVICE) # (1, C, WINDOW_SIZE, H, W)

                    try:
                        features = mvit_backbone(window_frames_batch) # Output shape ví dụ: (1, 768)
                        pooled_features = features.mean(dim=1)
                        video_features_list.append(pooled_features.cpu().numpy())
                    except Exception as e:
                        print(f"Lỗi khi xử lý cửa sổ cho {video_id} ({start_idx}-{end_idx}): {e}")


            if not video_features_list:
                print(f"Không có đặc trưng nào được trích xuất cho {video_id} ({dataset_type}). Video có thể quá ngắn hoặc có lỗi.")
                continue

            video_features_np = np.array(video_features_list) # (num_windows, feature_dim)
            np.savez_compressed(output_feature_path, features=video_features_np,
                                frame_indices=[(s, s + WINDOW_SIZE) for s in range(0, total_frames - WINDOW_SIZE + 1, STRIDE) if s + WINDOW_SIZE <= total_frames]
                                # Lưu thêm thông tin về frame_indices nếu cần cho việc khớp nối sau này
                               )
            print(f"Đã lưu đặc trưng MViT cho {video_id} ({dataset_type}) vào {output_feature_path}, shape: {video_features_np.shape}")

    print("Hoàn tất trích xuất đặc trưng MViT.")

if __name__ == '__main__':
    main()