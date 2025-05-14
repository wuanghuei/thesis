import torch
import torch.nn as nn
from torchvision.models.video import mvit_v2_s, MViT_V2_S_Weights

class MViT16FeatureExtractor(nn.Module):
    """
    Module trích xuất đặc trưng sử dụng mô hình MViT với đầu vào 16 frames.
    Lấy CLS token làm vector đặc trưng đại diện cho đoạn video.
    """
    
    def __init__(self, model_name="mvit_base_16x4", device=None):
        """
        Khởi tạo MViT Feature Extractor
        
        Parameters:
        -----------
        model_name : str
            Tên mô hình MViT từ PyTorchVideo
        device : torch.device or None
            Device để chạy mô hình. Nếu None, sẽ dùng GPU nếu có hoặc CPU.
        """
        super().__init__()
        
        # Xác định device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
        
        
        video_backbone_pretrained = mvit_v2_s(
            weights=MViT_V2_S_Weights.KINETICS400_V1
        ).to(self.device)
        self.feature_dimension = 768
        
        if hasattr(video_backbone_pretrained, 'head') and isinstance(video_backbone_pretrained.head, nn.Sequential):
            video_backbone_pretrained.head = nn.Identity()
        self.model = video_backbone_pretrained

    def extract_features(self, video_tensor):
        """
        Trích xuất đặc trưng từ video tensor
        
        Parameters:
        -----------
        video_tensor : torch.Tensor
            Tensor đầu vào với shape [B, C, T, H, W]
            B: batch size
            C: channels (3 cho RGB)
            T: thời gian (số frames, ở đây là 32)
            H, W: chiều cao và rộng (thường là 224x224)
            
        Returns:
        --------
        torch.Tensor
            CLS token feature với shape [B, feature_dimension]
        """
        # Đảm bảo model ở chế độ eval
        self.model.eval()
        
        # Chuyển tensor sang device của model
        video_tensor = video_tensor.to(self.device)
        
        # Trích xuất đặc trưng mà không tính gradient
        with torch.no_grad():
            # shape của output là [B, NumTokens, feature_dimension]
            token_sequence = self.model(video_tensor)
            
            # Lấy CLS token (token đầu tiên)
            cls_token_features = token_sequence
            
        return cls_token_features
    

if __name__ == "__main__":
    extractor = MViT16FeatureExtractor()
    video_tensor = torch.randn(1, 3, 16, 224, 224)
    features = extractor.extract_features(video_tensor)
    print(extractor.feature_dimension)
    print(features.shape)
    


