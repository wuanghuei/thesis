import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.video.mvit import MViT, MSBlockConfig
from torchvision.models import resnet18
from contextlib import nullcontext


class CustomMViT(MViT):
    def __init__(self, *args, **kwargs):
        kwargs.pop("num_classes", None)
        kwargs.pop("head", None)
        super().__init__(*args, **kwargs)
        self.head = nn.Identity()

    def forward(self, x):
        features = super().forward(x)
        return features


class TemporalActionDetector(nn.Module):
    
    def __init__(self, num_classes=5, window_size=32, dropout=0.3):
        super().__init__()
        self.num_classes = num_classes
        self.window_size = window_size
        
        
        config = {
            "num_heads":        [1,   2, 2,   4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 8, 8],
            "input_channels":   [96,  96, 192, 192, 384,384,384,384,384,384,384,384,384,384,384,768],
            "output_channels":  [96, 192, 192, 384, 384,384,384,384,384,384,384,384,384,384,768,768],
            "kernel_q":  [[3,3,3]] * 16,
            "kernel_kv": [[3,3,3]] * 16,
            "stride_q": [
                [1,1,1], [1,2,2], [1,1,1], [1,2,2],
                *[[1,1,1]] * 10, [1,2,2], [1,1,1]
            ],
            "stride_kv": [
                [1,8,8], [1,4,4], [1,4,4], [1,2,2],
                *[[1,2,2]] * 10, [1,1,1], [1,1,1]
            ]
        }
        block_setting = [
            MSBlockConfig(
                num_heads=config["num_heads"][i],
                input_channels=config["input_channels"][i],
                output_channels=config["output_channels"][i],
                kernel_q=config["kernel_q"][i],
                kernel_kv=config["kernel_kv"][i],
                stride_q=config["stride_q"][i],
                stride_kv=config["stride_kv"][i],
            )
            for i in range(16)
        ]
        self.video_backbone = CustomMViT(
            spatial_size=(224, 224),
            temporal_size=window_size,
            block_setting=block_setting,
            residual_pool=True,
            residual_with_cls_embed=False,
            rel_pos_embed=True,
            proj_after_attn=True,
            stochastic_depth_prob=0.2,
            norm_layer=nn.LayerNorm
        )
        
        
        self.rgb_feature_dim = 768
        
        
        
        self.frame_extractor_2d = resnet18(weights='IMAGENET1K_V1')
        
        self.frame_extractor_2d = nn.Sequential(*list(self.frame_extractor_2d.children())[:-1])
        self.local_feature_dim = 512 

        
        
        
        groups = 64 
        self.local_temporal_refiner = nn.Sequential(
            nn.Conv1d(self.local_feature_dim, self.local_feature_dim, kernel_size=3, padding=1, groups=groups),
            nn.BatchNorm1d(self.local_feature_dim),
            nn.ReLU(),
            nn.Conv1d(self.local_feature_dim, self.local_feature_dim, kernel_size=3, padding=1, groups=groups),
            nn.BatchNorm1d(self.local_feature_dim),
            nn.ReLU()
        )

        
        self.local_to_global_projection = nn.Linear(self.local_feature_dim, self.rgb_feature_dim)

        
        
        
        self.feature_gate = nn.Sequential(
            nn.Linear(self.rgb_feature_dim * 2, self.rgb_feature_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout), 
            nn.Linear(self.rgb_feature_dim // 2, self.rgb_feature_dim),
            nn.Sigmoid() 
        )
        
        
        
        self.pose_lstm = nn.LSTM(
            input_size=198,  
            hidden_size=128,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=dropout
        )
        self.pose_fc = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        
        
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
            nn.Dropout(dropout)
        )
        
        
        
        self.action_classifier = nn.Conv1d(512, num_classes, kernel_size=1)
        
        
        self.start_detector = nn.Conv1d(512, num_classes, kernel_size=1)
        
        
        self.end_detector = nn.Conv1d(512, num_classes, kernel_size=1)
    
    def _extract_frame_features(self, x):
        
        batch_size, channels, T, H, W = x.shape 
        final_combined_features = torch.zeros((batch_size, self.rgb_feature_dim, T), device=x.device)

        
        
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

                
                local_features_batch = local_features_flat.view(batch_size, current_frames_count, self.local_feature_dim) 
                local_features_list.append(local_features_batch)

        local_features_raw = torch.cat(local_features_list, dim=1) 

        
        
        local_features_temporal_input = local_features_raw.permute(0, 2, 1) 

        
        
        self.local_temporal_refiner.to(x.device)
        
        refiner_context = nullcontext() if self.training else torch.no_grad()
        with refiner_context:
            refined_local_features_temporal = self.local_temporal_refiner(local_features_temporal_input) 

        
        refined_local_features = refined_local_features_temporal.permute(0, 2, 1) 

        
        self.local_to_global_projection.to(x.device)
        
        projection_context = nullcontext() if self.training else torch.no_grad()
        with projection_context:
            projected_refined_local_features = self.local_to_global_projection(refined_local_features) 

        
        
        expanded_global_features = global_clip_features_raw.unsqueeze(1).expand(-1, T, -1) 

        
        gate_input = torch.cat([expanded_global_features.detach(), projected_refined_local_features], dim=2) 
        

        
        
        self.feature_gate.to(x.device)
        gate_context = nullcontext() if self.training else torch.no_grad()
        with gate_context:
            gate_weights = self.feature_gate(gate_input) 

        
        
        combined_features = gate_weights * projected_refined_local_features + (1 - gate_weights) * expanded_global_features.detach() 

        
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
        
        
        fused_features = (
            fusion_weights[0] * rgb_features + 
            fusion_weights[1] * pose_features
        )  
        
        
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
            'end_scores': end_scores          
        }
        
    def post_process(self, action_scores, start_scores, end_scores, 
                     action_threshold=0.5, boundary_threshold=0.5, nms_threshold=0.5):
        
        batch_size = action_scores.shape[0]
        T = action_scores.shape[1]
        
        batch_detections = []
        
        for b in range(batch_size):
            
            sample_detections = []
            
            
            for c in range(self.num_classes):
                
                action_score = action_scores[b, :, c]  
                start_score = start_scores[b, :, c]    
                end_score = end_scores[b, :, c]        
                
                
                start_candidates = (start_score > boundary_threshold).nonzero().flatten()
                end_candidates = (end_score > boundary_threshold).nonzero().flatten()
                
                proposals = []
                
                
                for start_idx in start_candidates:
                    for end_idx in end_candidates:
                        
                        if start_idx >= end_idx:
                            continue
                            
                        
                        segment_score = action_score[start_idx:end_idx].mean()
                        
                        
                        if segment_score > action_threshold:
                            
                            boundary_score = (start_score[start_idx] + end_score[end_idx]) / 2
                            
                            
                            confidence = (segment_score + boundary_score) / 2
                            
                            proposals.append({
                                'action_id': c,
                                'start_frame': start_idx.item(),
                                'end_frame': end_idx.item(), 
                                'confidence': confidence.item()
                            })
                            
                
                proposals = self._nms(proposals, nms_threshold)
                
                
                sample_detections.extend(proposals)
                
            
            sample_detections = sorted(sample_detections, key=lambda x: x['confidence'], reverse=True)
            batch_detections.append(sample_detections)
            
        return batch_detections
    
    def _nms(self, proposals, threshold=0.5):
        
        if not proposals:
            return []
            
        
        proposals = sorted(proposals, key=lambda x: x['confidence'], reverse=True)
        
        
        kept_proposals = []
        
        for prop in proposals:
            
            keep = True
            
            for kept_prop in kept_proposals:
                
                if prop['action_id'] != kept_prop['action_id']:
                    continue
                    
                
                s1, e1 = prop['start_frame'], prop['end_frame']
                s2, e2 = kept_prop['start_frame'], kept_prop['end_frame']
                
                inter = max(0, min(e1, e2) - max(s1, s2))
                union = max(1, (e1 - s1) + (e2 - s2) - inter)  
                iou = inter / union
                
                if iou > threshold:
                    keep = False
                    break
                    
            if keep:
                kept_proposals.append(prop)
                
        return kept_proposals 