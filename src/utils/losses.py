"""
Loss Functions for Cephalometric Landmark Detection
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class AdaptiveWingLoss(nn.Module):
    """
    Adaptive Wing Loss for landmark detection
    Better than MSE for handling outliers and imbalanced data
    
    Reference: Wang et al. "Adaptive Wing Loss for Robust Face Alignment via Heatmap Regression"
    """
    
    def __init__(self, omega=14, theta=0.5, epsilon=1, alpha=2.1):
        super(AdaptiveWingLoss, self).__init__()
        self.omega = omega
        self.theta = theta
        self.epsilon = epsilon
        self.alpha = alpha
    
    def forward(self, pred, target):
        """
        Args:
            pred: Predicted heatmaps (B, NUM_LANDMARKS, H, W)
            target: Target heatmaps (B, NUM_LANDMARKS, H, W)
        
        Returns:
            loss: Adaptive wing loss
        """
        delta = (target - pred).abs()
        
        A = self.omega * (1 / (1 + torch.pow(self.theta / self.epsilon, self.alpha - target))) * \
            (self.alpha - target) * torch.pow(self.theta / self.epsilon, self.alpha - target - 1) * \
            (1 / self.epsilon)
        C = self.theta * A - self.omega * torch.log(1 + torch.pow(self.theta / self.epsilon, self.alpha - target))
        
        loss = torch.where(
            delta < self.theta,
            self.omega * torch.log(1 + torch.pow(delta / self.epsilon, self.alpha - target)),
            A * delta - C
        )
        
        return loss.mean()


class WingLoss(nn.Module):
    """
    Wing Loss for landmark detection
    Simpler version of Adaptive Wing Loss
    
    Reference: Feng et al. "Wing Loss for Robust Facial Landmark Localisation with CNNs"
    """
    
    def __init__(self, omega=10, epsilon=2):
        super(WingLoss, self).__init__()
        self.omega = omega
        self.epsilon = epsilon
        self.C = self.omega - self.omega * torch.log(torch.tensor(1.0 + self.omega / self.epsilon))
    
    def forward(self, pred, target):
        """
        Args:
            pred: Predicted heatmaps (B, NUM_LANDMARKS, H, W)
            target: Target heatmaps (B, NUM_LANDMARKS, H, W)
        
        Returns:
            loss: Wing loss
        """
        delta = (target - pred).abs()
        
        loss = torch.where(
            delta < self.omega,
            self.omega * torch.log(1 + delta / self.epsilon),
            delta - self.C
        )
        
        return loss.mean()


class HeatmapMSELoss(nn.Module):
    """
    MSE Loss for heatmap regression
    """
    
    def __init__(self):
        super(HeatmapMSELoss, self).__init__()
        self.mse = nn.MSELoss()
    
    def forward(self, pred, target):
        """
        Args:
            pred: Predicted heatmaps (B, NUM_LANDMARKS, H, W)
            target: Target heatmaps (B, NUM_LANDMARKS, H, W)
        
        Returns:
            loss: MSE loss
        """
        return self.mse(pred, target)


class CombinedLoss(nn.Module):
    """
    Combined loss for multi-task learning (landmarks + CVM classification)
    """
    
    def __init__(
        self,
        landmark_loss_type='wing',
        landmark_weight=1.0,
        cvm_weight=0.1,
        omega=10,
        epsilon=2
    ):
        """
        Args:
            landmark_loss_type: 'mse', 'wing', or 'adaptive_wing'
            landmark_weight: Weight for landmark loss
            cvm_weight: Weight for CVM classification loss
            omega: Wing loss parameter
            epsilon: Wing loss parameter
        """
        super(CombinedLoss, self).__init__()
        
        self.landmark_weight = landmark_weight
        self.cvm_weight = cvm_weight
        
        # Landmark loss
        if landmark_loss_type == 'mse':
            self.landmark_loss = HeatmapMSELoss()
        elif landmark_loss_type == 'wing':
            self.landmark_loss = WingLoss(omega=omega, epsilon=epsilon)
        elif landmark_loss_type == 'adaptive_wing':
            self.landmark_loss = AdaptiveWingLoss()
        else:
            raise ValueError(f"Unknown landmark loss type: {landmark_loss_type}")
        
        # CVM classification loss
        self.cvm_loss = nn.CrossEntropyLoss()
    
    def forward(self, pred_heatmaps, target_heatmaps, pred_cvm=None, target_cvm=None):
        """
        Args:
            pred_heatmaps: Predicted heatmaps (B, NUM_LANDMARKS, H, W)
            target_heatmaps: Target heatmaps (B, NUM_LANDMARKS, H, W)
            pred_cvm: Predicted CVM logits (B, NUM_CVM_STAGES) or None
            target_cvm: Target CVM one-hot (B, NUM_CVM_STAGES) or None
        
        Returns:
            total_loss: Combined loss
            loss_dict: Dictionary of individual losses
        """
        # Landmark loss
        landmark_loss = self.landmark_loss(pred_heatmaps, target_heatmaps)
        total_loss = self.landmark_weight * landmark_loss
        
        loss_dict = {
            'landmark_loss': landmark_loss.item(),
            'total_loss': total_loss.item()
        }
        
        # CVM loss (if provided)
        if pred_cvm is not None and target_cvm is not None:
            # Convert one-hot to class indices
            target_cvm_indices = torch.argmax(target_cvm, dim=1)
            cvm_loss = self.cvm_loss(pred_cvm, target_cvm_indices)
            total_loss += self.cvm_weight * cvm_loss
            
            loss_dict['cvm_loss'] = cvm_loss.item()
            loss_dict['total_loss'] = total_loss.item()
        
        return total_loss, loss_dict


def build_loss_function(config):
    """
    Build loss function based on configuration
    
    Args:
        config: Configuration object
    
    Returns:
        loss_fn: Loss function
    """
    loss_fn = CombinedLoss(
        landmark_loss_type='wing',  # Wing loss works well for landmarks
        landmark_weight=config.LANDMARK_LOSS_WEIGHT,
        cvm_weight=config.CVM_LOSS_WEIGHT
    )
    
    return loss_fn
