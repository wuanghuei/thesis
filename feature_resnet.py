import torch
import numpy as np
from pathlib import Path
import os
from tqdm import tqdm
from torchvision import transforms
from torchvision.models import resnet18
import torch.nn as nn


# --- Cấu hình ---
BASE_DATA_DIR = Path('Data/full_videos/') # Điều chỉnh đến thư mục dữ liệu chính của bạn
FRAMES_SUBDIR = 'frames' # Thư mục con chứa các frame video (ví dụ: video_id_frames.npz)
OUTPUT_FEATURES_DIR = Path('Data/features/resnet18_per_frame/') # Thư mục để lưu các đặc trưng ResNet đã trích xuất
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
FRAME_BATCH_SIZE = 32 # Xử lý các frame theo batch để tăng hiệu quả

# --- Hàm trợ giúp để tải Frames ---
def load_video_frames_resnet(video_id, frames_dir):
    """Tải tất cả các frame cho một video_id nhất định cho ResNet."""
    frames_path = frames_dir / f"{video_id}_frames.npz"
    if not frames_path.exists():
        print(f"Không tìm thấy tệp frames cho {video_id} tại {frames_path}")
        return None
    try:
        data = np.load(frames_path)
        # ResNet mong đợi (B, C, H, W) cho mỗi frame
        frames = data['frames'] # Ví dụ: (Total_Frames, H, W, 3)
        if frames.ndim == 4 and frames.shape[-1] in [1, 3]: # (T, H, W, C)
             # Không cần thay đổi ở đây nếu xử lý từng frame, sẽ hoán vị sau
             pass
        elif frames.ndim == 4 and frames.shape[1] in [1,3] and frames.shape[0] > frames.shape[1]: # (T, C, H, W)
             frames = np.transpose(frames, (0, 2, 3, 1)) # Chuyển thành (T, H, W, C) để nhất quán trước khi tạo batch
        elif frames.ndim == 3: # (T, H, W) cho video thang độ xám
            frames = np.expand_dims(frames, axis=-1) # (T, H, W, 1)
        else:
            raise ValueError(f"Hình dạng frames không mong muốn cho {video_id}: {frames.shape}")

        # Chuẩn hóa và biến đổi ToTensor sẽ được áp dụng cho mỗi batch
        return frames # Hình dạng: (Total_Frames, H, W, C)
    except Exception as e:
        print(f"Lỗi khi tải frames cho {video_id}: {e}")
        return None

# Tiền xử lý điển hình của ResNet
preprocess = transforms.Compose([
    transforms.ToPILImage(), # Nếu frame là numpy H,W,C
    transforms.ToTensor()
])


def main():
    print(f"Sử dụng thiết bị: {DEVICE}")
    OUTPUT_FEATURES_DIR.mkdir(parents=True, exist_ok=True)

    print("Đang tải mô hình ResNet-18...")
    resnet_extractor = resnet18(weights='IMAGENET1K_V1')
    resnet_extractor = nn.Sequential(*list(resnet_extractor.children())[:-1])
    resnet_extractor.eval()
    resnet_extractor.to(DEVICE)
    print("Đã tải mô hình ResNet-18.")

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

        for frame_file_path in tqdm(video_frame_files, desc=f"Trích xuất đặc trưng ResNet ({dataset_type})"):
            video_id = frame_file_path.name.replace("_frames.npz", "")
            output_feature_path = output_dir_type / f"{video_id}_resnet_features.npz"

            if output_feature_path.exists():
                print(f"Đặc trưng cho {video_id} ({dataset_type}) đã tồn tại. Bỏ qua.")
                continue

            video_frames_np = load_video_frames_resnet(video_id, frames_dir)
            if video_frames_np is None:
                continue

            total_frames, H, W, C = video_frames_np.shape
            if total_frames == 0:
                print(f"Video {video_id} không có frame nào. Bỏ qua.")
                continue
            video_features_list = []

            with torch.no_grad():
                for i in range(0, total_frames, FRAME_BATCH_SIZE):
                    batch_frames_np = video_frames_np[i:i+FRAME_BATCH_SIZE] # (batch_slice, H, W, C)

                    # Tiền xử lý batch: danh sách các tensor (C,H,W)
                    # Đảm bảo frame_img là uint8 nếu ToPILImage mong đợi điều đó
                    if batch_frames_np.dtype != np.uint8:
                        # Chuyển đổi nếu frames của bạn không phải là uint8 (ví dụ: float 0-255)
                        # Hoặc điều chỉnh preprocess cho phù hợp
                        # Ở đây giả sử chúng là uint8 hoặc có thể được chuyển đổi an toàn
                        processed_batch = []
                        for frame_img_np in batch_frames_np:
                            if frame_img_np.shape[-1] == 1: # Nếu là ảnh xám
                                frame_img_np = np.squeeze(frame_img_np, axis=-1) # (H, W)
                            # ToPILImage mong đợi (H, W) hoặc (H, W, C)
                            processed_batch.append(preprocess(frame_img_np))

                        batch_frames_tensor = torch.stack(processed_batch).to(DEVICE) # (batch_slice, C, H, W)
                    else:
                         batch_frames_tensor = torch.stack(
                            [preprocess(frame_img) for frame_img in batch_frames_np]
                        ).to(DEVICE)


                    try:
                        features = resnet_extractor(batch_frames_tensor) # Output (batch_slice, 512, 1, 1)
                        # Ép các chiều không gian và chuyển sang CPU
                        features = features.squeeze(-1).squeeze(-1).cpu().numpy() # (batch_slice, 512)
                        video_features_list.append(features)
                    except Exception as e:
                        print(f"Lỗi khi xử lý batch frame cho {video_id} (frames {i} đến {i+len(batch_frames_np)-1}): {e}")


            if not video_features_list:
                print(f"Không có đặc trưng nào được trích xuất cho {video_id} ({dataset_type}). Video có thể trống hoặc có lỗi.")
                continue

            video_features_np_all = np.concatenate(video_features_list, axis=0) # (Total_Frames, feature_dim)

            np.savez_compressed(output_feature_path, features=video_features_np_all)
            print(f"Đã lưu đặc trưng ResNet cho {video_id} ({dataset_type}) vào {output_feature_path}, shape: {video_features_np_all.shape}")

    print("Hoàn tất trích xuất đặc trưng ResNet.")

if __name__ == '__main__':
    main()