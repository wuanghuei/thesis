import torch
import torch.nn as nn

class ActionDetectionLoss(nn.Module):
    def __init__(self, action_weight, start_weight, end_weight, device, num_classes, label_smoothing=0.1):
        """
        Loss function for temporal action detection
        
        Args:
            action_weight: Trọng số cho action segmentation loss
            start_weight: Trọng số cho start point detection loss
            end_weight: Trọng số cho end point detection loss
            label_smoothing: Amount of label smoothing to apply (0.0 to 0.5)
        """
        super().__init__()
        self.action_weight = action_weight
        self.start_weight = start_weight
        self.end_weight = end_weight
        self.label_smoothing = label_smoothing
        
        # BCE for action segmentation
        self.action_criterion = nn.BCEWithLogitsLoss(reduction='none')
        
        # BCE for start/end detection
        self.boundary_criterion = nn.BCEWithLogitsLoss(reduction='none')
        
        # Trọng số cho từng lớp (tăng trọng số cho Class 2 và 3)
        self.class_weights = torch.ones(num_classes, device=device)
        self.class_weights[0] = 1.0 
        self.class_weights[1] = 1.5 
        self.class_weights[2] = 7.0  # Tăng mạnh từ 3.3 lên 7.0
        self.class_weights[3] = 2.0  # Thêm trọng số cho Class 3
        self.class_weights[4] = 1.0 

        # Trọng số riêng cho boundary loss của từng lớp
        self.boundary_weights = torch.ones(num_classes, device=device)
        # Tăng mạnh trọng số boundary cho Class 2
        self.boundary_weights[2] = 5.0  # Thêm trọng số đáng kể cho boundary Class 2

    def smooth_labels(self, targets):
        """Apply label smoothing to targets"""
        if self.label_smoothing <= 0:
            return targets
        return targets * (1.0 - self.label_smoothing) + 0.5 * self.label_smoothing
        
    def forward(self, predictions, targets):
        """
        Calculate loss for temporal action detection
        
        Args:
            predictions: dict với keys 'action_scores', 'start_scores', 'end_scores'
                  - action_scores: (B, T, C) - scores cho mỗi frame và class
                  - start_scores: (B, T, C) - scores cho start points
                  - end_scores: (B, T, C) - scores cho end points
                  
            targets: dict với keys 'action_masks', 'start_masks', 'end_masks'
                  - action_masks: (B, C, T) - binary masks cho actions
                  - start_masks: (B, C, T) - Gaussian smoothed masks cho start points
                  - end_masks: (B, C, T) - Gaussian smoothed masks cho end points
            
        Returns:
            Dict với 'total' loss và các individual components
        """
        action_scores = predictions['action_scores']  # (B, T, C)
        start_scores = predictions['start_scores']    # (B, T, C)
        end_scores = predictions['end_scores']        # (B, T, C)
        
        action_masks = targets['action_masks']  # (B, C, T)
        start_masks = targets['start_masks']    # (B, C, T) - Đã áp dụng Gaussian smoothing trong dataloader
        end_masks = targets['end_masks']        # (B, C, T) - Đã áp dụng Gaussian smoothing trong dataloader
        
        # Transpose targets to match predictions
        action_masks = action_masks.transpose(1, 2)  # (B, T, C)
        start_masks = start_masks.transpose(1, 2)    # (B, T, C)
        end_masks = end_masks.transpose(1, 2)        # (B, T, C)
        
        # Apply label smoothing if enabled
        if self.label_smoothing > 0:
            action_masks = self.smooth_labels(action_masks)
            start_masks = self.smooth_labels(start_masks)
            end_masks = self.smooth_labels(end_masks)
        
        # Calculate action loss với class weights
        action_loss = self.action_criterion(action_scores, action_masks)
        
        # Áp dụng class weights
        action_loss = action_loss * self.class_weights.view(1, 1, -1)
        action_loss = action_loss.mean()
        
        # Only consider regions with actions for start/end loss
        # Tạo mask cho những frame có ít nhất một action
        valid_regions = (action_masks.sum(dim=2, keepdim=True) > 0).float()
        
        # Calculate start/end loss với class weights
        start_loss = self.boundary_criterion(start_scores, start_masks)
        end_loss = self.boundary_criterion(end_scores, end_masks)
        
        # Áp dụng class weights cho start/end loss
        start_loss = start_loss * self.boundary_weights.view(1, 1, -1) # Sử dụng boundary_weights
        end_loss = end_loss * self.boundary_weights.view(1, 1, -1)   # Sử dụng boundary_weights
        
        # Apply valid regions mask and normalize
        if valid_regions.sum() > 0:
            # Chỉ tính loss trên những frame có action
            start_loss = (start_loss * valid_regions).sum() / (valid_regions.sum() + 1e-6)
            end_loss = (end_loss * valid_regions).sum() / (valid_regions.sum() + 1e-6)
        else:
            # Fallback khi không có frame nào có action
            start_loss = start_loss.mean()
            end_loss = end_loss.mean()
        
        # Total loss với weighted components
        total_loss = (
            self.action_weight * action_loss + 
            self.start_weight * start_loss + 
            self.end_weight * end_loss
        )
        
        return {
            'total': total_loss,
            'action': action_loss.detach(),
            'start': start_loss.detach(),
            'end': end_loss.detach()
        }
