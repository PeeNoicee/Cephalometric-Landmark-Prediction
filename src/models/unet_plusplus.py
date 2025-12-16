"""
U-Net++ Model for Cephalometric Landmark Detection
"""
import torch
import torch.nn as nn
import segmentation_models_pytorch as smp


class UNetPlusPlusLandmarkDetector(nn.Module):
    """
    U-Net++ based model for landmark detection with optional CVM classification
    """
    
    def __init__(
        self,
        num_landmarks=29,
        num_cvm_stages=6,
        encoder_name='resnet34',
        encoder_weights='imagenet',
        in_channels=1,
        use_multitask=True
    ):
        """
        Args:
            num_landmarks: Number of landmarks to detect
            num_cvm_stages: Number of CVM stages for classification
            encoder_name: Encoder backbone name
            encoder_weights: Pretrained weights ('imagenet' or None)
            in_channels: Number of input channels (1 for grayscale)
            use_multitask: Enable multi-task learning (landmarks + CVM)
        """
        super(UNetPlusPlusLandmarkDetector, self).__init__()
        
        self.num_landmarks = num_landmarks
        self.num_cvm_stages = num_cvm_stages
        self.use_multitask = use_multitask
        
        # U-Net++ for landmark heatmap prediction
        self.unet = smp.UnetPlusPlus(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=num_landmarks,
            activation=None  # We'll apply sigmoid in forward
        )
        
        # CVM classification head (if multitask)
        if self.use_multitask:
            # Get encoder output channels
            encoder_channels = self.unet.encoder.out_channels
            
            self.cvm_head = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(encoder_channels[-1], 256),
                nn.ReLU(inplace=True),
                nn.Dropout(0.3),
                nn.Linear(256, 128),
                nn.ReLU(inplace=True),
                nn.Dropout(0.3),
                nn.Linear(128, num_cvm_stages)
            )
    
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: Input image tensor (B, 1, H, W)
        
        Returns:
            heatmaps: Landmark heatmaps (B, num_landmarks, H, W)
            cvm_logits: CVM classification logits (B, num_cvm_stages) if multitask
        """
        # Use the full U-Net++ forward pass for heatmaps
        heatmaps = self.unet(x)
        heatmaps = torch.sigmoid(heatmaps)
        
        if self.use_multitask:
            # Get encoder features for CVM classification
            features = self.unet.encoder(x)
            cvm_logits = self.cvm_head(features[-1])
            return heatmaps, cvm_logits
        else:
            return heatmaps


class HRNetLandmarkDetector(nn.Module):
    """
    HR-Net based model for landmark detection (placeholder for future implementation)
    """
    
    def __init__(
        self,
        num_landmarks=29,
        num_cvm_stages=6,
        use_multitask=True
    ):
        super(HRNetLandmarkDetector, self).__init__()
        raise NotImplementedError("HRNet implementation coming soon. Use UNetPlusPlus for now.")


def build_model(config):
    """
    Build model based on configuration
    
    Args:
        config: Configuration object
    
    Returns:
        model: PyTorch model
    """
    if config.MODEL_NAME == 'UNetPlusPlus':
        model = UNetPlusPlusLandmarkDetector(
            num_landmarks=config.NUM_LANDMARKS,
            num_cvm_stages=config.NUM_CVM_STAGES,
            encoder_name=config.BACKBONE,
            encoder_weights='imagenet' if config.PRETRAINED else None,
            in_channels=1,
            use_multitask=config.USE_MULTITASK
        )
    elif config.MODEL_NAME == 'HRNet':
        model = HRNetLandmarkDetector(
            num_landmarks=config.NUM_LANDMARKS,
            num_cvm_stages=config.NUM_CVM_STAGES,
            use_multitask=config.USE_MULTITASK
        )
    else:
        raise ValueError(f"Unknown model name: {config.MODEL_NAME}")
    
    return model
