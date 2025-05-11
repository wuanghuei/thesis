import torch
import torch.nn as nn

class MViT32FeatureExtractor(nn.Module):
    """
    Module trích xuất đặc trưng sử dụng mô hình MViT với đầu vào 32 frames.
    Lấy CLS token làm vector đặc trưng đại diện cho đoạn video.
    """
    
    def __init__(self, model_name="mvit_base_32x3", device=None):
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
        
        # Tải mô hình pretrained
        try:
            self.model = torch.hub.load('facebookresearch/pytorchvideo', model_name, pretrained=True)
            print(f"Model {model_name} loaded successfully from PyTorch Hub!")
        except Exception as e:
            raise RuntimeError(f"Error loading model from PyTorch Hub: {e}")
        
        # Chuyển model sang device chỉ định
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # Thay thế head để lấy token embeddings thay vì class scores
        try:
            # Xác định số chiều đặc trưng của token
            if hasattr(self.model, 'head') and hasattr(self.model.head, 'proj') and isinstance(self.model.head.proj, nn.Linear):
                self.feature_dimension = self.model.head.proj.in_features
            else:
                print("Cannot directly infer feature_dimension from head.proj. Assuming 768 for MViT-Base.")
                self.feature_dimension = 768  # Giả định cho MViT-Base
                
            # Thay thế head bằng Identity
            self.model.head = nn.Identity()  # Output sẽ là [B, NumTokens, FeatureDimPerToken]
            print(f"Replaced model.head with nn.Identity(). Output will be a sequence of tokens.")
            print(f"Feature dimension per token: {self.feature_dimension}")
            
        except AttributeError as e:
            raise RuntimeError(f"Could not replace model.head. The model structure might be different than expected: {e}")
    
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
            cls_token_features = token_sequence[:, 0, :]
            
        return cls_token_features
