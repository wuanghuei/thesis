import torch
import torch.nn as nn
import torch.nn.functional as F
from contextlib import nullcontext
import numpy as np
from pathlib import Path
import os
import random


class TemporalActionDetector(nn.Module):
    def __init__(self, num_classes=5, window_size=32, dropout=0.3, pose_feature_dim=198): 
        super().__init__()
        self.num_classes = num_classes
        self.window_size = window_size

        self.rgb_feature_dim = 768
        self.local_feature_dim = 512

        self.local_to_global_projection = nn.Linear(self.local_feature_dim, self.rgb_feature_dim)
        
        self.feature_gate = nn.Sequential(
            nn.Linear(self.rgb_feature_dim * 2, self.rgb_feature_dim // 2), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(self.rgb_feature_dim // 2, self.rgb_feature_dim), nn.Sigmoid()
        )

        self.pose_feature_dim = pose_feature_dim
        self.pose_lstm = nn.LSTM(input_size=self.pose_feature_dim, hidden_size=128, num_layers=2,
                                  batch_first=True, bidirectional=True, dropout=dropout)
        self.pose_fc = nn.Sequential(nn.Linear(128 * 2, 128), nn.ReLU(), nn.Dropout(dropout))

        self.fusion_weights = nn.Parameter(torch.ones(2) / 2.0)
        self.fusion_dim = 512
        self.rgb_projection = nn.Linear(self.rgb_feature_dim, self.fusion_dim)
        self.pose_projection = nn.Linear(128, self.fusion_dim)

        self.action_classifier = nn.Conv1d(512, num_classes, kernel_size=1)
        self.start_detector    = nn.Conv1d(512, num_classes, kernel_size=1)
        self.end_detector      = nn.Conv1d(512, num_classes, kernel_size=1)
    
    def _process_pose_stream(self, pose_features):
        out, _ = self.pose_lstm(pose_features) 
        out = self.pose_fc(out) 
        return out.transpose(1, 2)

    def forward(self, mvit_features, resnet_features, pose_features=None):
        B = mvit_features.shape[0]
        T = resnet_features.shape[1]

        projected_local_to_global_dim = self.local_to_global_projection(resnet_features)
        expanded_global_features = mvit_features.expand(-1, T, -1)

        gate_input = torch.cat([expanded_global_features.detach(), projected_local_to_global_dim], dim=2)
        gate_weights = self.feature_gate(gate_input)
        
        fused_rgb_for_projection = gate_weights * projected_local_to_global_dim + \
                                   (1 - gate_weights) * expanded_global_features.detach()
        
        projected_rgb_feats = self.rgb_projection(fused_rgb_for_projection)

        if pose_features is not None:
            processed_pose_feats_transposed = self._process_pose_stream(pose_features)
            projected_pose_feats = self.pose_projection(processed_pose_feats_transposed.transpose(1,2))
        else:
            projected_pose_feats = torch.zeros(B, T, self.fusion_dim, device=mvit_features.device)
        
        w = F.softmax(self.fusion_weights, dim=0)
        final_fused_feats = w[0] * projected_rgb_feats + w[1] * projected_pose_feats

        feats = final_fused_feats.transpose(1, 2)

        action_scores = self.action_classifier(feats).transpose(1, 2)
        start_scores  = self.start_detector(feats).transpose(1, 2)
        end_scores    = self.end_detector(feats).transpose(1, 2)
        
        return {
            'action_scores': action_scores,
            'start_scores': start_scores,
            'end_scores': end_scores
        }
