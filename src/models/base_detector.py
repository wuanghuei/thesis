import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.video import mvit_v2_s, MViT_V2_S_Weights
from torchvision.models import resnet18
from contextlib import nullcontext

class TemporalActionDetector(nn.Module):
    def __init__(self, num_classes=5, window_size=32, dropout=0.3):
        super().__init__()
        self.num_classes = num_classes
        self.window_size = window_size
        
        # Load pretrained MViT_V2_S model
        video_backbone_pretrained = mvit_v2_s(
            weights=MViT_V2_S_Weights.KINETICS400_V1
        )
        
        # The feature dimension for MViT_V2_S (small) before the head is 768
        self.rgb_feature_dim = 768 # This is generally the case for MViT_V2_S
        
        # Replace the head of the MViT_V2_S model with an identity layer to get features
        # MViT's head is typically named 'head' and is an nn.Sequential(nn.Dropout(...), nn.Linear(...))
        # We need to ensure this attribute name is correct or find the actual feature dimension.
        # Based on typical torchvision MViT structure:
        if hasattr(video_backbone_pretrained, 'head') and isinstance(video_backbone_pretrained.head, nn.Sequential):
            # Assuming the Linear layer is the last module in the head sequential block
            # and its in_features is what we need for rgb_feature_dim.
            # For MViT_V2_S, the head is nn.Sequential(nn.Dropout(p=..., inplace=True), nn.Linear(in_features=768, out_features=..., bias=True))
            # So, self.rgb_feature_dim = 768 is likely correct.
            video_backbone_pretrained.head = nn.Identity()
        else:
            # Fallback or error if head structure is not as expected
            # This might require inspecting the model structure further if it fails.
            print("Warning: MViT_V2_S head structure not as expected. Feature extraction might be incorrect.")
            # As a robust measure, we could try to get the in_features of the last linear layer if it exists
            # For now, we proceed assuming the head removal is successful and dim is 768.
        self.video_backbone = self._adapt_mvit_for_32_frames(video_backbone_pretrained)
        
        # The rest of the parameters from the original MViT initialization that were commented out
        # are either implicitly handled by mvit_v2_s (like spatial_size, num_heads, num_layers, hidden_dim, mlp_dim for the specific variant 's')
        # or need to be compatible with the pretrained model's configuration (like temporal_size).
        # The KINETICS400_V1 weights for MViT_V2_S are trained with specific input resolution and temporal length (e.g., 16 frames for temporal_size).
        # If window_size (e.g., 32) is different, this needs careful handling, potentially by adjusting model parameters if allowed,
        # or by ensuring input preprocessing matches what the fine-tuning expects.
        # mvit_v2_s does not take temporal_size directly as an argument when loading pretrained_weights.
        # It expects a certain input shape. The `window_size` here will dictate the input tensor's temporal dimension.

        self.frame_extractor_2d = resnet18(weights='IMAGENET1K_V1')
        self.frame_extractor_2d = nn.Sequential(*list(self.frame_extractor_2d.children())[:-1])
        self.local_feature_dim = 512
        groups = 64
        self.local_temporal_refiner = nn.Sequential(
            nn.Conv1d(
                self.local_feature_dim,
                self.local_feature_dim,
                kernel_size=3,
                padding=1,
                groups=groups,
            ),
            nn.BatchNorm1d(self.local_feature_dim),
            nn.ReLU(),
            nn.Conv1d(
                self.local_feature_dim,
                self.local_feature_dim,
                kernel_size=3,
                padding=1,
                groups=groups,
            ),
            nn.BatchNorm1d(self.local_feature_dim),
            nn.ReLU(),
        )
        self.local_to_global_projection = nn.Linear(self.local_feature_dim, self.rgb_feature_dim)
        self.feature_gate = nn.Sequential(
            nn.Linear(self.rgb_feature_dim * 2, self.rgb_feature_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(self.rgb_feature_dim // 2, self.rgb_feature_dim),
            nn.Sigmoid(),
        )
        self.pose_lstm = nn.LSTM(
            input_size=198,
            hidden_size=128,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=dropout,
        )
        self.pose_fc = nn.Sequential(nn.Linear(256, 128), nn.ReLU(), nn.Dropout(dropout))
        self.fusion_weights = nn.Parameter(torch.ones(2) / 2.0)
        self.fusion_dim = 512
        self.rgb_projection = nn.Linear(self.rgb_feature_dim, self.fusion_dim)
        self.pose_projection = nn.Linear(128, self.fusion_dim)
        self.tcn = nn.Sequential(
            nn.Conv1d(self.fusion_dim, 512, kernel_size=5, padding=2),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(512, 512, kernel_size=5, padding=2),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.action_classifier = nn.Conv1d(512, num_classes, kernel_size=1)
        self.start_detector = nn.Conv1d(512, num_classes, kernel_size=1)
        self.end_detector = nn.Conv1d(512, num_classes, kernel_size=1)
    def _adapt_mvit_for_32_frames(self, model):
        """Điều chỉnh MViT model để xử lý 32 frames input"""
        # Cập nhật temporal_size trong positional encoding
        model.pos_encoding.temporal_size = self.window_size // 2  # 16 = 32/2 (sau conv_proj)
        return model
    def _extract_frame_features(self, x):
        batch_size, channels, T, H, W = x.shape
        mvit_context = nullcontext() if self.training else torch.no_grad()
        with mvit_context:
            global_clip_features_raw = self.video_backbone(x)
        local_features_list = []
        frame_batch_size = 8
        self.frame_extractor_2d.to(x.device)
        with torch.no_grad():
            self.frame_extractor_2d.eval()
            for t_start in range(0, T, frame_batch_size):
                t_end = min(t_start + frame_batch_size, T)
                current_frames_count = t_end - t_start
                frames_batch = x[:, :, t_start:t_end, :, :].permute(0, 2, 1, 3, 4)
                frames_batch = frames_batch.reshape(-1, channels, H, W)
                local_features_flat = self.frame_extractor_2d(frames_batch)
                local_features_flat = local_features_flat.squeeze(-1).squeeze(-1)
                local_features_batch = local_features_flat.view(
                    batch_size, current_frames_count, self.local_feature_dim
                )
                local_features_list.append(local_features_batch)
        local_features_raw = torch.cat(local_features_list, dim=1)
        local_features_temporal_input = local_features_raw.permute(0, 2, 1)
        self.local_temporal_refiner.to(x.device)
        refiner_context = nullcontext() if self.training else torch.no_grad()
        with refiner_context:
            refined_local_features_temporal = self.local_temporal_refiner(
                local_features_temporal_input
            )
        refined_local_features = refined_local_features_temporal.permute(0, 2, 1)
        self.local_to_global_projection.to(x.device)
        projection_context = nullcontext() if self.training else torch.no_grad()
        with projection_context:
            projected_refined_local_features = self.local_to_global_projection(
                refined_local_features
            )
        expanded_global_features = global_clip_features_raw.unsqueeze(1).expand(-1, T, -1)
        gate_input = torch.cat(
            [expanded_global_features.detach(), projected_refined_local_features], dim=2
        )
        self.feature_gate.to(x.device)
        gate_context = nullcontext() if self.training else torch.no_grad()
        with gate_context:
            gate_weights = self.feature_gate(gate_input)
        combined_features = (
            gate_weights * projected_refined_local_features
            + (1 - gate_weights) * expanded_global_features.detach()
        )
        return combined_features.permute(0, 2, 1)
    def _process_pose_stream(self, pose_data):
        batch_size, seq_len, _ = pose_data.shape
        pose_output, _ = self.pose_lstm(pose_data)
        pose_features = self.pose_fc(pose_output)
        pose_features = pose_features.transpose(1, 2)
        return pose_features
    def forward(self, frames, pose=None):
        batch_size = frames.shape[0]
        T = frames.shape[2]
        rgb_features = self._extract_frame_features(frames)
        rgb_features = self.rgb_projection(rgb_features.transpose(1, 2))
        if pose is not None:
            pose_features = self._process_pose_stream(pose)
            pose_features = self.pose_projection(pose_features.transpose(1, 2))
        else:
            pose_features = torch.zeros(batch_size, T, self.fusion_dim, device=frames.device)
        fusion_weights = F.softmax(self.fusion_weights, dim=0)

        fused_features = fusion_weights[0] * rgb_features + fusion_weights[1] * pose_features

        fused_features = fused_features.transpose(1, 2)

        temporal_features = self.tcn(fused_features)

        action_scores = self.action_classifier(temporal_features)
        start_scores = self.start_detector(temporal_features)
        end_scores = self.end_detector(temporal_features)

        action_scores = action_scores.transpose(1, 2)
        start_scores = start_scores.transpose(1, 2)
        end_scores = end_scores.transpose(1, 2)

        return {
            'action_scores': action_scores,
            'start_scores': start_scores,
            'end_scores': end_scores,
        }
