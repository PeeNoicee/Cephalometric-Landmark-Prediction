"""Training Configuration for Cephalometric Landmark Detection

Model: HRNet with multi-task learning (landmark + CVM)
Input: 864x768 rectangular images (preserves aspect ratio)
Output: 29 cephalometric landmarks
Performance: MRE ~0.83mm, SDR@2mm ~94.69%
"""
import torch


class TrainConfig:
    # Dataset paths
    DATASET_ROOT = r"c:\Users\clyde\Special Problem\AI Cephalometric Model_Xenodent\dataset"
    
    # Model parameters
    NUM_LANDMARKS = 29
    NUM_CVM_STAGES = 6
    INPUT_SIZE = (864, 768)  # Rectangular (H, W) to preserve aspect ratio (~0.89)
    HEATMAP_SIZE = (216, 192)  # Proportional to input (divide by 4)
    SIGMA = 6.0  # Slightly larger sigma for higher resolution
    
    # Model architecture: 'unet++' or 'hrnet'
    MODEL_ARCH = 'hrnet'
    
    # Training parameters - improved
    BATCH_SIZE = 3  # Safe for RTX 3060 12GB at 768x768
    NUM_EPOCHS = 300  # More epochs for convergence
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-4
    
    # Optimizer
    OPTIMIZER = 'AdamW'  # AdamW for better regularization
    
    # Learning rate scheduler - cosine annealing
    LR_SCHEDULER = 'CosineAnnealingWarmRestarts'
    LR_T0 = 50  # First restart period
    LR_T_MULT = 2  # Multiply period after each restart
    LR_ETA_MIN = 1e-6  # Minimum learning rate
    
    # Warmup
    WARMUP_EPOCHS = 5
    WARMUP_LR = 1e-6
    
    # Loss function
    LOSS_TYPE = 'adaptive_wing'  # 'adaptive_wing', 'wing', 'mse', 'focal_mse'
    LANDMARK_LOSS_WEIGHT = 1.0
    CVM_LOSS_WEIGHT = 0.1
    
    # Multi-task learning
    USE_MULTITASK = True
    
    # Data augmentation
    USE_AUGMENTATION = True
    USE_CLAHE = True
    CLAHE_CLIP_LIMIT = 2.0
    CLAHE_TILE_GRID_SIZE = (8, 8)
    
    # Pixel spacing - use per-image values from CSV
    USE_PER_IMAGE_PIXEL_SPACING = True
    DEFAULT_PIXEL_SPACING = 0.1  # Fallback
    
    # Evaluation metrics (in mm)
    SDR_THRESHOLDS = [2.0, 2.5, 3.0, 4.0]  # Standard thresholds in mm
    
    # Coordinate extraction
    COORD_EXTRACTION_METHOD = 'argmax'  # 'argmax' or 'soft_argmax' (argmax is more robust)
    
    # Checkpointing
    CHECKPOINT_DIR = 'checkpoints'
    SAVE_EVERY = 10
    
    # Logging
    LOG_DIR = 'logs'
    
    # Device
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Number of workers
    NUM_WORKERS = 0  # Windows compatibility


# Backward compatibility alias
TrainConfigV2 = TrainConfig
