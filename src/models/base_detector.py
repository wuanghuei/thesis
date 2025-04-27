import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.video.mvit import MViT
from torchvision.models import resnet18
from contextlib import nullcontext

class TemporalActionDetector(nn.Module):
    def __init__(self, num_classes=5, window_size=32, dropout=0.3):
        super().__init__()
        self.num_classes = num_classes
        self.window_size = window_size
        self.video_backbone = MViT(
            spatial_size=(224, 224),
            temporal_size=window_size,
            num_heads=8,
            num_layers=16,
            hidden_dim=768,
            mlp_dim=3072,
            num_classes=num_classes,
            dropout=dropout,
            attention_dropout=dropout,
            stochastic_depth_prob=0.2,
            norm_layer=nn.LayerNorm,
        )
        self.rgb_feature_dim = 768
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
