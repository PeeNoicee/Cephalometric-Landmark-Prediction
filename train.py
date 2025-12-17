"""
Training Script for Cephalometric Landmark Detection

Features:
- HRNet architecture (maintains high-resolution features)
- Adaptive Wing Loss (precise landmark localization)
- Per-image pixel spacing (proper mm conversion)
- Cosine annealing with warm restarts
- Multi-task learning (landmarks + CVM staging)

Usage:
    python train.py
"""
import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import numpy as np
from pathlib import Path

try:
    import pynvml
except Exception:
    pynvml = None

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.config.config import TrainConfig
from src.data.dataset_enhanced import get_dataloader
from src.data.pixel_spacing import load_pixel_spacing_dict
from src.models.hrnet import build_hrnet
from src.models.unet_plusplus import build_model as build_unetpp
from src.utils.losses_improved import CombinedLandmarkLoss, DIFFICULT_LANDMARK_INDICES
from src.utils.metrics import MetricsTracker, extract_coordinates_from_heatmaps


class TrainerV2:
    """Improved trainer with HRNet and Adaptive Wing Loss"""
    
    def __init__(self, config):
        self.config = config
        self.device = config.DEVICE

        self._nvml_available = False
        self._nvml_handle = None
        if pynvml is not None and torch.cuda.is_available() and str(self.device).startswith('cuda'):
            try:
                pynvml.nvmlInit()
                self._nvml_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                self._nvml_available = True
            except Exception:
                self._nvml_available = False
                self._nvml_handle = None
        
        # Create directories
        os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
        os.makedirs(config.LOG_DIR, exist_ok=True)
        
        # Load pixel spacing
        print("Loading pixel spacing data...")
        self.pixel_spacing_loader = load_pixel_spacing_dict(config.DATASET_ROOT)
        self.pixel_spacing_loader.print_stats()
        
        # Build model
        print(f"\nBuilding {config.MODEL_ARCH} model...")
        if config.MODEL_ARCH == 'hrnet':
            self.model = build_hrnet(config).to(self.device)
        else:
            self.model = build_unetpp(config).to(self.device)

        if torch.cuda.is_available() and str(self.device).startswith('cuda'):
            try:
                props = torch.cuda.get_device_properties(0)
                total_gb = props.total_memory / (1024 ** 3)
                print(f"GPU: {props.name} ({total_gb:.1f} GB)")
            except Exception:
                pass
        
        # Count parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"Total parameters: {total_params / 1e6:.2f}M")
        print(f"Trainable parameters: {trainable_params / 1e6:.2f}M")
        
        # Loss function
        print(f"\nUsing {config.LOSS_TYPE} loss...")
        self.loss_fn = CombinedLandmarkLoss(
            use_multitask=config.USE_MULTITASK,
            landmark_weight=config.LANDMARK_LOSS_WEIGHT,
            cvm_weight=config.CVM_LOSS_WEIGHT,
            loss_type=config.LOSS_TYPE,
            difficult_landmark_indices=DIFFICULT_LANDMARK_INDICES
        )
        
        # Optimizer
        if config.OPTIMIZER == 'AdamW':
            self.optimizer = optim.AdamW(
                self.model.parameters(),
                lr=config.LEARNING_RATE,
                weight_decay=config.WEIGHT_DECAY
            )
        else:
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=config.LEARNING_RATE,
                weight_decay=config.WEIGHT_DECAY
            )
        
        # Learning rate scheduler
        if config.LR_SCHEDULER == 'CosineAnnealingWarmRestarts':
            self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
                self.optimizer,
                T_0=config.LR_T0,
                T_mult=config.LR_T_MULT,
                eta_min=config.LR_ETA_MIN
            )
        else:
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=0.5,
                patience=10
            )
        
        # Data loaders
        print("\nLoading datasets...")
        self.train_loader = get_dataloader(config, mode='train', return_pixel_spacing=False)
        self.val_loader = get_dataloader(config, mode='valid', return_pixel_spacing=True)
        print(f"Train samples: {len(self.train_loader.dataset)}")
        print(f"Valid samples: {len(self.val_loader.dataset)}")
        
        # TensorBoard
        self.writer = SummaryWriter(config.LOG_DIR)
        
        # Best metrics
        self.best_mre = float('inf')
        self.best_epoch = 0

    def _get_gpu_stats(self):
        if not (torch.cuda.is_available() and str(self.device).startswith('cuda')):
            return None

        try:
            props = torch.cuda.get_device_properties(0)
            total_gb = props.total_memory / (1024 ** 3)
            alloc_gb = torch.cuda.memory_allocated(0) / (1024 ** 3)
            reserved_gb = torch.cuda.memory_reserved(0) / (1024 ** 3)
            peak_alloc_gb = torch.cuda.max_memory_allocated(0) / (1024 ** 3)
        except Exception:
            return None

        util = None
        mem_util = None
        if self._nvml_available and self._nvml_handle is not None:
            try:
                u = pynvml.nvmlDeviceGetUtilizationRates(self._nvml_handle)
                util = int(u.gpu)
                mem_util = int(u.memory)
            except Exception:
                util = None
                mem_util = None

        return {
            'total_gb': total_gb,
            'alloc_gb': alloc_gb,
            'reserved_gb': reserved_gb,
            'peak_alloc_gb': peak_alloc_gb,
            'util': util,
            'mem_util': mem_util,
        }
    
    def warmup_lr(self, epoch, batch_idx, num_batches):
        """Linear warmup learning rate"""
        if epoch < self.config.WARMUP_EPOCHS:
            warmup_iters = self.config.WARMUP_EPOCHS * num_batches
            cur_iter = epoch * num_batches + batch_idx
            lr = self.config.WARMUP_LR + (self.config.LEARNING_RATE - self.config.WARMUP_LR) * cur_iter / warmup_iters
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
            return lr
        return self.optimizer.param_groups[0]['lr']
    
    def train_epoch(self, epoch):
        """Train for one epoch"""
        self.model.train()
        train_losses = []
        lm_losses = []

        if torch.cuda.is_available() and str(self.device).startswith('cuda'):
            try:
                torch.cuda.reset_peak_memory_stats(0)
            except Exception:
                pass
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.config.NUM_EPOCHS}")
        
        for batch_idx, batch_data in enumerate(pbar):
            # Warmup LR
            lr = self.warmup_lr(epoch, batch_idx, len(self.train_loader))
            
            if self.config.USE_MULTITASK:
                images, target_heatmaps, landmarks, target_cvm = batch_data
                target_cvm = target_cvm.to(self.device)
            else:
                images, target_heatmaps, landmarks = batch_data
                target_cvm = None
            
            images = images.to(self.device)
            target_heatmaps = target_heatmaps.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            
            if self.config.USE_MULTITASK:
                pred_heatmaps, pred_cvm = self.model(images)
            else:
                pred_heatmaps = self.model(images)
                pred_cvm = None
            
            # Resize heatmaps if needed
            if pred_heatmaps.shape[-2:] != target_heatmaps.shape[-2:]:
                pred_heatmaps = nn.functional.interpolate(
                    pred_heatmaps,
                    size=target_heatmaps.shape[-2:],
                    mode='bilinear',
                    align_corners=False
                )
            
            # Loss
            loss, loss_dict = self.loss_fn(
                pred_heatmaps, target_heatmaps,
                pred_cvm, target_cvm
            )
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            train_losses.append(loss_dict['total_loss'])
            lm_losses.append(loss_dict['landmark_loss'])

            postfix = {
                'loss': f"{loss_dict['total_loss']:.4f}",
                'lm_loss': f"{loss_dict['landmark_loss']:.4f}",
                'lr': f"{lr:.2e}",
            }
            stats = self._get_gpu_stats()
            if stats is not None:
                postfix['vram'] = f"{stats['alloc_gb']:.1f}/{stats['total_gb']:.0f}G"
                postfix['peak'] = f"{stats['peak_alloc_gb']:.1f}G"
                if stats['util'] is not None:
                    postfix['gpu'] = f"{stats['util']}%"

            pbar.set_postfix(postfix)
        
        # Update scheduler (after warmup)
        if epoch >= self.config.WARMUP_EPOCHS:
            if isinstance(self.scheduler, optim.lr_scheduler.CosineAnnealingWarmRestarts):
                self.scheduler.step()
        
        avg_loss = np.mean(train_losses)
        avg_lm_loss = np.mean(lm_losses)
        
        return avg_loss, avg_lm_loss
    
    @torch.no_grad()
    def validate(self, epoch):
        """Validate model"""
        self.model.eval()
        val_losses = []
        
        # Metrics tracker with mm thresholds
        metrics_tracker = MetricsTracker(sdr_thresholds=self.config.SDR_THRESHOLDS)
        
        pbar = tqdm(self.val_loader, desc="Validation")
        
        for batch_idx, batch_data in enumerate(pbar):
            # Unpack batch with pixel spacing
            if self.config.USE_MULTITASK:
                images, target_heatmaps, landmarks, target_cvm, pixel_spacings = batch_data
                target_cvm = target_cvm.to(self.device)
            else:
                images, target_heatmaps, landmarks, pixel_spacings = batch_data
                target_cvm = None
            
            images = images.to(self.device)
            target_heatmaps = target_heatmaps.to(self.device)
            
            # Forward pass
            if self.config.USE_MULTITASK:
                pred_heatmaps, pred_cvm = self.model(images)
            else:
                pred_heatmaps = self.model(images)
                pred_cvm = None
            
            # Resize heatmaps
            pred_heatmaps = nn.functional.interpolate(
                pred_heatmaps,
                size=self.config.HEATMAP_SIZE,
                mode='bilinear',
                align_corners=False
            )
            
            # Resize target heatmaps if different
            if target_heatmaps.shape[-2:] != self.config.HEATMAP_SIZE:
                target_heatmaps = nn.functional.interpolate(
                    target_heatmaps,
                    size=self.config.HEATMAP_SIZE,
                    mode='bilinear',
                    align_corners=False
                )
            
            # Loss
            loss, loss_dict = self.loss_fn(
                pred_heatmaps, target_heatmaps,
                pred_cvm, target_cvm
            )
            val_losses.append(loss_dict['total_loss'])
            
            # Extract coordinates
            pred_coords = extract_coordinates_from_heatmaps(
                pred_heatmaps,
                method=self.config.COORD_EXTRACTION_METHOD
            )
            
            # Scale to input size
            scale_x = self.config.INPUT_SIZE[1] / self.config.HEATMAP_SIZE[1]
            scale_y = self.config.INPUT_SIZE[0] / self.config.HEATMAP_SIZE[0]
            pred_coords[:, :, 0] *= scale_x
            pred_coords[:, :, 1] *= scale_y
            
            # Update metrics with per-image pixel spacing
            for i in range(len(images)):
                metrics_tracker.update(
                    pred_coords[i:i+1], 
                    landmarks[i:i+1], 
                    pixel_spacing=pixel_spacings[i].item()
                )
            
            # Debug output for first batch
            if batch_idx == 0:
                ps = pixel_spacings[0].item()
                print(f"\n  DEBUG - Sample predictions vs ground truth (pixel_spacing={ps:.4f} mm/px):")
                for lm_idx in range(min(3, pred_coords.shape[1])):
                    pred = pred_coords[0, lm_idx].cpu().numpy()
                    gt = landmarks[0, lm_idx].cpu().numpy()
                    dist = np.sqrt((pred[0] - gt[0])**2 + (pred[1] - gt[1])**2)
                    dist_mm = dist * ps
                    print(f"    LM{lm_idx}: Pred=({pred[0]:.1f}, {pred[1]:.1f}) GT=({gt[0]:.1f}, {gt[1]:.1f}) Dist={dist:.1f}px ({dist_mm:.2f}mm)")
        
        # Compute metrics
        metrics = metrics_tracker.compute()
        avg_loss = np.mean(val_losses)
        
        return avg_loss, metrics
    
    def save_checkpoint(self, epoch, metrics, is_best=False):
        """Save checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'mre': metrics['MRE'],
            'best_mre': self.best_mre,
            'best_epoch': self.best_epoch,
            'config': {
                'MODEL_ARCH': self.config.MODEL_ARCH,
                'LOSS_TYPE': self.config.LOSS_TYPE,
                'INPUT_SIZE': self.config.INPUT_SIZE,
                'HEATMAP_SIZE': self.config.HEATMAP_SIZE,
            }
        }
        
        # Save periodic checkpoint
        if (epoch + 1) % self.config.SAVE_EVERY == 0:
            path = os.path.join(self.config.CHECKPOINT_DIR, f'checkpoint_epoch_{epoch+1}.pth')
            torch.save(checkpoint, path)
        
        # Save best model
        if is_best:
            path = os.path.join(self.config.CHECKPOINT_DIR, 'best_model.pth')
            torch.save(checkpoint, path)
            best_epoch_path = os.path.join(self.config.CHECKPOINT_DIR, f'best_model_epoch_{epoch+1}.pth')
            torch.save(checkpoint, best_epoch_path)
            print(f"Saved best model with MRE: {metrics['MRE']:.2f} mm")
    
    def train(self, start_epoch: int = 0):
        """Full training loop"""
        print("\n" + "="*80)
        print("TRAINING V2 - Improved Settings for Benchmark Performance")
        print("="*80)
        print(f"Model: {self.config.MODEL_ARCH}")
        print(f"Loss: {self.config.LOSS_TYPE}")
        print(f"Epochs: {self.config.NUM_EPOCHS}")
        print(f"Batch size: {self.config.BATCH_SIZE}")
        print(f"Learning rate: {self.config.LEARNING_RATE}")
        print(f"Coord extraction: {self.config.COORD_EXTRACTION_METHOD}")
        print("="*80 + "\n")

        if start_epoch < 0:
            start_epoch = 0
        if start_epoch >= self.config.NUM_EPOCHS:
            print(f"Start epoch {start_epoch} is >= NUM_EPOCHS ({self.config.NUM_EPOCHS}). Nothing to do.")
            return

        for epoch in range(start_epoch, self.config.NUM_EPOCHS):
            # Train
            train_loss, train_lm_loss = self.train_epoch(epoch)
            print(f"\nEpoch {epoch+1} - Train Loss: {train_loss:.4f}, Landmark Loss: {train_lm_loss:.4f}")

            stats = self._get_gpu_stats()
            if stats is not None:
                print(
                    f"GPU Stats - alloc: {stats['alloc_gb']:.2f} GB | reserved: {stats['reserved_gb']:.2f} GB | peak alloc: {stats['peak_alloc_gb']:.2f} GB"
                )
            
            # Validate
            val_loss, metrics = self.validate(epoch)
            
            # Print results
            print(f"\nValidation Results:")
            print(f"  Loss: {val_loss:.4f}")
            print(f"  MRE: {metrics['MRE']:.2f} ± {metrics['STD']:.2f} mm")
            for key, value in metrics.items():
                if key.startswith('SDR'):
                    print(f"  {key}: {value:.2f}%")
            
            # Check if best
            is_best = metrics['MRE'] < self.best_mre
            if is_best:
                self.best_mre = metrics['MRE']
                self.best_epoch = epoch + 1
            
            # Save checkpoint
            self.save_checkpoint(epoch, metrics, is_best)
            
            print(f"Best MRE so far: {self.best_mre:.2f} mm (epoch {self.best_epoch})")
            print("-"*80)
            
            # Log to TensorBoard
            self.writer.add_scalar('Loss/train', train_loss, epoch)
            self.writer.add_scalar('Loss/val', val_loss, epoch)
            self.writer.add_scalar('MRE/val', metrics['MRE'], epoch)
            self.writer.add_scalar('LR', self.optimizer.param_groups[0]['lr'], epoch)
            for key, value in metrics.items():
                if key.startswith('SDR'):
                    self.writer.add_scalar(f'SDR/{key}', value, epoch)
        
        self.writer.close()
        
        print("\n" + "="*80)
        print("TRAINING COMPLETE")
        print(f"Best MRE: {self.best_mre:.2f} mm at epoch {self.best_epoch}")
        print("="*80)


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Train Cephalometric Landmark Detection Model')
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint to resume')
    args = parser.parse_args()
    
    # Load config
    config = TrainConfig()
    print(f"Training config: {config.NUM_EPOCHS} epochs, batch size {config.BATCH_SIZE}")
    
    # Create trainer
    trainer = TrainerV2(config)

    start_epoch = 0
    
    # Resume if specified
    if args.resume:
        print(f"Resuming from: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=config.DEVICE, weights_only=False)
        trainer.model.load_state_dict(checkpoint['model_state_dict'])
        trainer.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if checkpoint['scheduler_state_dict']:
            trainer.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = int(checkpoint.get('epoch', -1)) + 1
        trainer.best_mre = checkpoint.get('best_mre', checkpoint.get('mre', float('inf')))
        trainer.best_epoch = checkpoint.get('best_epoch', start_epoch)

        # If a separate best_model exists, use its MRE to seed best tracking so we don't
        # overwrite a better checkpoint after resuming.
        best_path = os.path.join(config.CHECKPOINT_DIR, 'best_model.pth')
        if os.path.exists(best_path):
            try:
                best_ckpt = torch.load(best_path, map_location='cpu', weights_only=False)
                best_mre = best_ckpt.get('mre', None)
                if best_mre is not None and float(best_mre) < float(trainer.best_mre):
                    trainer.best_mre = float(best_mre)
                    trainer.best_epoch = int(best_ckpt.get('epoch', -1)) + 1
            except Exception:
                pass

        print(f"Resumed from epoch {start_epoch}, best MRE so far: {trainer.best_mre:.2f} mm")
    
    # Train
    trainer.train(start_epoch=start_epoch)


if __name__ == '__main__':
    main()
