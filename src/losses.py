import torch
import torch.nn as nn

class ActionDetectionLoss(nn.Module):
    def __init__(self, action_weight, start_weight, end_weight, device, num_classes, label_smoothing=0.1):
        super().__init__()
        self.action_weight = action_weight
        self.start_weight = start_weight
        self.end_weight = end_weight
        self.label_smoothing = label_smoothing
        
        self.action_criterion = nn.BCEWithLogitsLoss(reduction='none')
        
        self.boundary_criterion = nn.BCEWithLogitsLoss(reduction='none')
        
        self.class_weights = torch.ones(num_classes, device=device)
        self.class_weights[0] = 1.0 
        self.class_weights[1] = 1.5 
        self.class_weights[2] = 7.0 
        self.class_weights[3] = 2.0 
        self.class_weights[4] = 1.0 

        self.boundary_weights = torch.ones(num_classes, device=device)
        self.boundary_weights[2] = 5.0 

    def smooth_labels(self, targets):
        if self.label_smoothing <= 0:
            return targets
        return targets * (1.0 - self.label_smoothing) + 0.5 * self.label_smoothing
        
    def forward(self, predictions, targets):

        action_scores = predictions['action_scores']
        start_scores = predictions['start_scores']    
        end_scores = predictions['end_scores']        
        
        action_masks = targets['action_masks'] 
        start_masks = targets['start_masks']    
        end_masks = targets['end_masks']        
        
        action_masks = action_masks.transpose(1, 2) 
        start_masks = start_masks.transpose(1, 2)    
        end_masks = end_masks.transpose(1, 2)        
        
        if self.label_smoothing > 0:
            action_masks = self.smooth_labels(action_masks)
            start_masks = self.smooth_labels(start_masks)
            end_masks = self.smooth_labels(end_masks)
        
        action_loss = self.action_criterion(action_scores, action_masks)
        
        action_loss = action_loss * self.class_weights.view(1, 1, -1)
        action_loss = action_loss.mean()
        
        valid_regions = (action_masks.sum(dim=2, keepdim=True) > 0).float()
        
        start_loss = self.boundary_criterion(start_scores, start_masks)
        end_loss = self.boundary_criterion(end_scores, end_masks)
        
        start_loss = start_loss * self.boundary_weights.view(1, 1, -1) 
        end_loss = end_loss * self.boundary_weights.view(1, 1, -1)  
        
        if valid_regions.sum() > 0:
            start_loss = (start_loss * valid_regions).sum() / (valid_regions.sum() + 1e-6)
            end_loss = (end_loss * valid_regions).sum() / (valid_regions.sum() + 1e-6)
        else:
            start_loss = start_loss.mean()
            end_loss = end_loss.mean()
        
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
