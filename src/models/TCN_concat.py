import torch
import torch.nn as nn
import torch.nn.functional as F

class TemporalBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation, dropout):
        super().__init__()
        padding = dilation * (kernel_size - 1) // 2
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size,
                               padding=padding, dilation=dilation)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.dropout1 = nn.Dropout(dropout)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size,
                               padding=padding, dilation=dilation)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.dropout2 = nn.Dropout(dropout)
        self.downsample = nn.Conv1d(in_channels, out_channels, 1) \
            if in_channels != out_channels else None
        self.relu = nn.ReLU()
    def forward(self, x):
        out = self.conv1(x)
        out = self.relu(self.bn1(out))
        out = self.dropout1(out)
        out = self.conv2(out)
        out = self.relu(self.bn2(out))
        out = self.dropout2(out)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)

class TemporalActionDetector(nn.Module):
    def __init__(self, num_classes=5, window_size=32, dropout=0.3, pose_feature_dim=198):
        super().__init__()
        self.num_classes = num_classes
        self.window_size = window_size

        self.rgb_feature_dim   = 768
        self.local_feature_dim = 512
        self.fusion_dim        = 512

        self.local_to_global_projection = nn.Linear(self.local_feature_dim, self.rgb_feature_dim)

        self.feature_gate = nn.Sequential(
            nn.Linear(self.rgb_feature_dim * 2, self.rgb_feature_dim // 2), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(self.rgb_feature_dim // 2, self.rgb_feature_dim), nn.Sigmoid()
        )

        self.pose_feature_dim = pose_feature_dim
        self.pose_lstm = nn.LSTM(input_size=self.pose_feature_dim, hidden_size=128, num_layers=2,
                                 batch_first=True, bidirectional=True, dropout=dropout)
        self.pose_fc = nn.Sequential(nn.Linear(128 * 2, 128), nn.ReLU(), nn.Dropout(dropout))

        self.rgb_projection  = nn.Linear(self.rgb_feature_dim, self.fusion_dim)
        self.pose_projection = nn.Linear(128, self.fusion_dim)
        self.fuse_conv       = nn.Conv1d(self.fusion_dim * 2, self.fusion_dim, kernel_size=1)

        self.tcn = nn.Sequential(
            TemporalBlock(self.fusion_dim, self.fusion_dim, kernel_size=3, dilation=1, dropout=dropout),
            TemporalBlock(self.fusion_dim, self.fusion_dim, kernel_size=3, dilation=2, dropout=dropout),
            TemporalBlock(self.fusion_dim, self.fusion_dim, kernel_size=3, dilation=4, dropout=dropout),
            TemporalBlock(self.fusion_dim, self.fusion_dim, kernel_size=3, dilation=8, dropout=dropout),
        )

        self.action_classifier = nn.Conv1d(self.fusion_dim, num_classes, kernel_size=1)
        self.start_detector    = nn.Conv1d(self.fusion_dim, num_classes, kernel_size=1)
        self.end_detector      = nn.Conv1d(self.fusion_dim, num_classes, kernel_size=1)

    def _process_pose_stream(self, pose_features):
        out, _ = self.pose_lstm(pose_features)
        out = self.pose_fc(out)
        return out

    def forward(self, mvit_features, resnet_features, pose_features=None):
        B, T = mvit_features.size(0), resnet_features.size(1)

        x_local = resnet_features
        proj_local = self.local_to_global_projection(x_local)
        x_global = mvit_features.expand(-1, T, -1)

        gate = self.feature_gate(torch.cat([x_global.detach(), proj_local], dim=2))
        fused_rgb = gate * proj_local + (1-gate) * x_global.detach()
        rgb_proj = self.rgb_projection(fused_rgb)

        if pose_features is not None:
            pose_out = self._process_pose_stream(pose_features)
            pose_proj = self.pose_projection(pose_out)
        else:
            pose_proj = torch.zeros(B, T, self.fusion_dim, device=mvit_features.device)

        concat = torch.cat([rgb_proj, pose_proj], dim=2)
        fused = self.fuse_conv(concat.permute(0, 2, 1))

        tcn_out = self.tcn(fused)

        action_scores = self.action_classifier(tcn_out).transpose(1, 2)
        start_scores  = self.start_detector(tcn_out).transpose(1, 2)
        end_scores    = self.end_detector(tcn_out).transpose(1, 2)

        return {
            'action_scores': action_scores,
            'start_scores':  start_scores,
            'end_scores':    end_scores
        }
