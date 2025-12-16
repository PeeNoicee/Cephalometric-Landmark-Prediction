"""
Improved Loss Functions for Cephalometric Landmark Detection
Includes Adaptive Wing Loss for better localization accuracy
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class AdaptiveWingLoss(nn.Module):
    """
    Adaptive Wing Loss for Heatmap Regression
    
    Reference: Adaptive Wing Loss for Robust Face Alignment via Heatmap Regression
    https://arxiv.org/abs/1904.07399
    
    This loss is specifically designed for heatmap regression and handles
    small errors better than MSE, leading to more precise landmark localization.
    """
    def __init__(self, omega=14, theta=0.5, epsilon=1, alpha=2.1):
        super().__init__()
        self.omega = omega
        self.theta = theta
        self.epsilon = epsilon
        self.alpha = alpha
    
    def forward(self, pred, target):
        """
        Args:
            pred: Predicted heatmaps (B, C, H, W)
            target: Target heatmaps (B, C, H, W)
        """
        delta = (target - pred).abs()
        
        # Compute A and C constants
        A = self.omega * (1 / (1 + torch.pow(self.theta / self.epsilon, self.alpha - target))) * \
            (self.alpha - target) * torch.pow(self.theta / self.epsilon, self.alpha - target - 1) * (1 / self.epsilon)
        C = self.theta * A - self.omega * torch.log(1 + torch.pow(self.theta / self.epsilon, self.alpha - target))
        
        # Compute loss
        losses = torch.where(
            delta < self.theta,
            self.omega * torch.log(1 + torch.pow(delta / self.epsilon, self.alpha - target)),
            A * delta - C
        )
        
        return losses.mean()


class WingLoss(nn.Module):
    """
    Wing Loss for precise localization
    
    Reference: Wing Loss for Robust Facial Landmark Localisation with CNNs
    """
    def __init__(self, omega=10, epsilon=2):
        super().__init__()
        self.omega = omega
        self.epsilon = epsilon
        self.C = self.omega - self.omega * math.log(1 + self.omega / self.epsilon)
    
    def forward(self, pred, target):
        delta = (target - pred).abs()
        
        losses = torch.where(
            delta < self.omega,
            self.omega * torch.log(1 + delta / self.epsilon),
            delta - self.C
        )
        
        return losses.mean()


class WeightedMSELoss(nn.Module):
    """
    Weighted MSE Loss with higher weight for difficult landmarks
    
    Difficult landmarks (cranial/posterior) get higher weights to force
    the model to focus on them.
    """
    def __init__(self, landmark_weights=None):
        super().__init__()
        self.landmark_weights = landmark_weights
    
    def forward(self, pred, target):
        """
        Args:
            pred: Predicted heatmaps (B, NUM_LANDMARKS, H, W)
            target: Target heatmaps (B, NUM_LANDMARKS, H, W)
        """
        mse = (pred - target) ** 2
        
        if self.landmark_weights is not None:
            # Apply per-landmark weights
            weights = self.landmark_weights.view(1, -1, 1, 1).to(pred.device)
            mse = mse * weights
        
        return mse.mean()


class FocalMSELoss(nn.Module):
    """
    Focal MSE Loss - focuses more on hard examples
    """
    def __init__(self, gamma=2.0):
        super().__init__()
        self.gamma = gamma
    
    def forward(self, pred, target):
        mse = (pred - target) ** 2
        focal_weight = (1 - torch.exp(-mse)) ** self.gamma
        return (focal_weight * mse).mean()


class CombinedLandmarkLoss(nn.Module):
    """
    Combined loss function for cephalometric landmark detection
    
    Combines:
    - Adaptive Wing Loss for heatmap regression
    - Cross Entropy for CVM classification (if multitask)
    - Optional weighted loss for difficult landmarks
    """
    def __init__(
        self,
        use_multitask=True,
        landmark_weight=1.0,
        cvm_weight=0.1,
        loss_type='adaptive_wing',
        difficult_landmark_indices=None
    ):
        super().__init__()
        self.use_multitask = use_multitask
        self.landmark_weight = landmark_weight
        self.cvm_weight = cvm_weight
        
        # Landmark loss
        if loss_type == 'adaptive_wing':
            self.landmark_loss = AdaptiveWingLoss()
        elif loss_type == 'wing':
            self.landmark_loss = WingLoss()
        elif loss_type == 'mse':
            self.landmark_loss = nn.MSELoss()
        elif loss_type == 'focal_mse':
            self.landmark_loss = FocalMSELoss()
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")
        
        # Create landmark weights if difficult landmarks specified
        self.landmark_weights = None
        if difficult_landmark_indices is not None:
            # Higher weight for difficult landmarks
            weights = torch.ones(29)
            for idx in difficult_landmark_indices:
                weights[idx] = 2.0  # Double weight for difficult landmarks
            self.register_buffer('landmark_weights_buffer', weights)
        
        # CVM loss
        if use_multitask:
            self.cvm_loss = nn.CrossEntropyLoss()
    
    def forward(self, pred_heatmaps, target_heatmaps, pred_cvm=None, target_cvm=None):
        """
        Args:
            pred_heatmaps: (B, NUM_LANDMARKS, H, W)
            target_heatmaps: (B, NUM_LANDMARKS, H, W)
            pred_cvm: (B, NUM_CVM_STAGES) if multitask
            target_cvm: (B, NUM_CVM_STAGES) one-hot if multitask
        
        Returns:
            total_loss: Combined loss
            loss_dict: Dictionary with individual losses
        """
        # Landmark heatmap loss
        lm_loss = self.landmark_loss(pred_heatmaps, target_heatmaps)
        
        # Apply per-landmark weights if available
        if hasattr(self, 'landmark_weights_buffer'):
            # Compute weighted loss per landmark channel
            per_lm_loss = ((pred_heatmaps - target_heatmaps) ** 2).mean(dim=(0, 2, 3))
            weights = self.landmark_weights_buffer.to(pred_heatmaps.device)
            weighted_lm_loss = (per_lm_loss * weights).mean()
            lm_loss = 0.5 * lm_loss + 0.5 * weighted_lm_loss
        
        total_loss = self.landmark_weight * lm_loss
        loss_dict = {
            'landmark_loss': lm_loss.item(),
            'total_loss': total_loss.item()
        }
        
        # CVM classification loss
        if self.use_multitask and pred_cvm is not None and target_cvm is not None:
            cvm_loss = self.cvm_loss(pred_cvm, target_cvm.argmax(dim=1))
            total_loss = total_loss + self.cvm_weight * cvm_loss
            loss_dict['cvm_loss'] = cvm_loss.item()
            loss_dict['total_loss'] = total_loss.item()
        
        return total_loss, loss_dict


# Difficult landmarks based on analysis (cranial/posterior region)
DIFFICULT_LANDMARK_INDICES = [
    4,   # N (Nasion)
    5,   # Or (Orbitale)
    7,   # PNS
    9,   # R (Ramus)
    10,  # S (Sella)
    11,  # Ar (Articulare)
    12,  # Co (Condylion)
    14,  # Go (Gonion)
    15,  # Po (Porion)
    26,  # N' (Soft Tissue Nasion)
]


def get_improved_loss(config):
    """Get improved loss function based on config"""
    return CombinedLandmarkLoss(
        use_multitask=config.USE_MULTITASK,
        landmark_weight=config.LANDMARK_LOSS_WEIGHT,
        cvm_weight=config.CVM_LOSS_WEIGHT,
        loss_type='adaptive_wing',
        difficult_landmark_indices=DIFFICULT_LANDMARK_INDICES
    )
