"""
Evaluation Metrics for Cephalometric Landmark Detection
"""
import torch
import numpy as np


def extract_coordinates_from_heatmaps(heatmaps, method='argmax'):
    """
    Extract landmark coordinates from heatmaps
    
    Args:
        heatmaps: Heatmap tensor (B, NUM_LANDMARKS, H, W)
        method: 'argmax' or 'soft_argmax'
    
    Returns:
        coordinates: Landmark coordinates (B, NUM_LANDMARKS, 2) in (x, y) format
    """
    batch_size, num_landmarks, height, width = heatmaps.shape
    
    if method == 'argmax':
        # Flatten spatial dimensions
        heatmaps_flat = heatmaps.view(batch_size, num_landmarks, -1)
        
        # Get argmax indices
        max_indices = torch.argmax(heatmaps_flat, dim=2)
        
        # Convert to (x, y) coordinates
        y_coords = max_indices // width
        x_coords = max_indices % width
        
        coordinates = torch.stack([x_coords, y_coords], dim=2).float()
        
    elif method == 'soft_argmax':
        # Soft-argmax for sub-pixel accuracy
        coordinates = soft_argmax_2d(heatmaps)
    
    else:
        raise ValueError(f"Unknown method: {method}")
    
    return coordinates


def soft_argmax_2d(heatmaps, temperature=1.0):
    """
    Differentiable soft-argmax for sub-pixel coordinate extraction
    
    Args:
        heatmaps: Heatmap tensor (B, NUM_LANDMARKS, H, W)
        temperature: Temperature for softmax
    
    Returns:
        coordinates: Landmark coordinates (B, NUM_LANDMARKS, 2)
    """
    batch_size, num_landmarks, height, width = heatmaps.shape
    
    # Flatten spatial dimensions
    heatmaps_flat = heatmaps.view(batch_size, num_landmarks, -1)
    
    # Apply softmax
    softmax_heatmaps = torch.softmax(heatmaps_flat / temperature, dim=2)
    softmax_heatmaps = softmax_heatmaps.view(batch_size, num_landmarks, height, width)
    
    # Create coordinate grids
    x_coords = torch.arange(width, device=heatmaps.device).float()
    y_coords = torch.arange(height, device=heatmaps.device).float()
    
    x_grid = x_coords.view(1, 1, 1, width).expand(batch_size, num_landmarks, height, width)
    y_grid = y_coords.view(1, 1, height, 1).expand(batch_size, num_landmarks, height, width)
    
    # Compute expected coordinates
    x_expected = torch.sum(softmax_heatmaps * x_grid, dim=[2, 3])
    y_expected = torch.sum(softmax_heatmaps * y_grid, dim=[2, 3])
    
    coordinates = torch.stack([x_expected, y_expected], dim=2)
    
    return coordinates


def calculate_mre(pred_coords, gt_coords, pixel_spacing=0.1):
    """
    Calculate Mean Radial Error (MRE)
    
    Args:
        pred_coords: Predicted coordinates (B, NUM_LANDMARKS, 2) or (NUM_LANDMARKS, 2)
        gt_coords: Ground truth coordinates (B, NUM_LANDMARKS, 2) or (NUM_LANDMARKS, 2)
        pixel_spacing: Pixel spacing in mm/pixel
    
    Returns:
        mre: Mean Radial Error in mm
    """
    if isinstance(pred_coords, torch.Tensor):
        pred_coords = pred_coords.detach().cpu().numpy()
    if isinstance(gt_coords, torch.Tensor):
        gt_coords = gt_coords.detach().cpu().numpy()
    
    # Calculate Euclidean distance
    distances = np.sqrt(np.sum((pred_coords - gt_coords) ** 2, axis=-1))
    
    # Convert to mm
    distances_mm = distances * pixel_spacing
    
    # Mean across all landmarks and samples
    mre = np.mean(distances_mm)
    
    return mre


def calculate_sdr(pred_coords, gt_coords, thresholds=[2.0, 2.5, 3.0, 4.0], pixel_spacing=0.1):
    """
    Calculate Success Detection Rate (SDR) at multiple thresholds
    
    Args:
        pred_coords: Predicted coordinates (B, NUM_LANDMARKS, 2) or (NUM_LANDMARKS, 2)
        gt_coords: Ground truth coordinates (B, NUM_LANDMARKS, 2) or (NUM_LANDMARKS, 2)
        thresholds: List of distance thresholds in mm
        pixel_spacing: Pixel spacing in mm/pixel
    
    Returns:
        sdr_dict: Dictionary of SDR values for each threshold
    """
    if isinstance(pred_coords, torch.Tensor):
        pred_coords = pred_coords.detach().cpu().numpy()
    if isinstance(gt_coords, torch.Tensor):
        gt_coords = gt_coords.detach().cpu().numpy()
    
    # Calculate Euclidean distance
    distances = np.sqrt(np.sum((pred_coords - gt_coords) ** 2, axis=-1))
    
    # Convert to mm
    distances_mm = distances * pixel_spacing
    
    # Calculate SDR for each threshold
    sdr_dict = {}
    for threshold in thresholds:
        successful = distances_mm <= threshold
        sdr = np.mean(successful) * 100  # Percentage
        sdr_dict[f'SDR@{threshold}mm'] = sdr
    
    return sdr_dict


def calculate_per_landmark_error(pred_coords, gt_coords, pixel_spacing=0.1):
    """
    Calculate error for each landmark separately
    
    Args:
        pred_coords: Predicted coordinates (B, NUM_LANDMARKS, 2)
        gt_coords: Ground truth coordinates (B, NUM_LANDMARKS, 2)
        pixel_spacing: Pixel spacing in mm/pixel
    
    Returns:
        errors: Per-landmark mean error in mm (NUM_LANDMARKS,)
    """
    if isinstance(pred_coords, torch.Tensor):
        pred_coords = pred_coords.detach().cpu().numpy()
    if isinstance(gt_coords, torch.Tensor):
        gt_coords = gt_coords.detach().cpu().numpy()
    
    # Calculate Euclidean distance
    distances = np.sqrt(np.sum((pred_coords - gt_coords) ** 2, axis=-1))
    
    # Convert to mm
    distances_mm = distances * pixel_spacing
    
    # Mean across batch for each landmark
    per_landmark_error = np.mean(distances_mm, axis=0)
    
    return per_landmark_error


class MetricsTracker:
    """
    Track metrics during training and evaluation
    """
    
    def __init__(self, sdr_thresholds=[2.0, 2.5, 3.0, 4.0]):
        self.sdr_thresholds = sdr_thresholds
        self.reset()
    
    def reset(self):
        """Reset all metrics"""
        self.all_pred_coords = []
        self.all_gt_coords = []
        self.pixel_spacings = []
    
    def update(self, pred_coords, gt_coords, pixel_spacing=0.1):
        """
        Update metrics with new batch
        
        Args:
            pred_coords: Predicted coordinates (B, NUM_LANDMARKS, 2)
            gt_coords: Ground truth coordinates (B, NUM_LANDMARKS, 2)
            pixel_spacing: Pixel spacing in mm/pixel (scalar or array of length B)
        """
        if isinstance(pred_coords, torch.Tensor):
            pred_coords = pred_coords.detach().cpu().numpy()
        if isinstance(gt_coords, torch.Tensor):
            gt_coords = gt_coords.detach().cpu().numpy()
        
        self.all_pred_coords.append(pred_coords)
        self.all_gt_coords.append(gt_coords)
        
        # Handle pixel spacing
        if isinstance(pixel_spacing, (int, float)):
            pixel_spacing = np.full(len(pred_coords), pixel_spacing)
        self.pixel_spacings.extend(pixel_spacing)
    
    def compute(self):
        """
        Compute final metrics
        
        Returns:
            metrics: Dictionary of metrics
        """
        if not self.all_pred_coords:
            return {}
        
        # Concatenate all predictions and ground truths
        pred_coords = np.concatenate(self.all_pred_coords, axis=0)
        gt_coords = np.concatenate(self.all_gt_coords, axis=0)
        pixel_spacings = np.array(self.pixel_spacings)
        
        # Calculate distances for each sample
        distances = np.sqrt(np.sum((pred_coords - gt_coords) ** 2, axis=-1))
        
        # Convert to mm using per-sample pixel spacing
        distances_mm = distances * pixel_spacings[:, np.newaxis]
        
        # MRE
        mre = np.mean(distances_mm)
        std = np.std(distances_mm)
        
        # SDR
        sdr_dict = {}
        for threshold in self.sdr_thresholds:
            successful = distances_mm <= threshold
            sdr = np.mean(successful) * 100
            sdr_dict[f'SDR@{threshold}mm'] = sdr
        
        # Per-landmark error
        per_landmark_error = np.mean(distances_mm, axis=0)
        
        metrics = {
            'MRE': mre,
            'STD': std,
            **sdr_dict,
            'per_landmark_error': per_landmark_error
        }
        
        return metrics
    
    def get_summary_string(self):
        """Get formatted summary string"""
        metrics = self.compute()
        
        if not metrics:
            return "No metrics available"
        
        summary = f"MRE: {metrics['MRE']:.2f} ± {metrics['STD']:.2f} mm"
        
        for key, value in metrics.items():
            if key.startswith('SDR'):
                summary += f" | {key}: {value:.2f}%"
        
        return summary
