"""
Evaluation Script for Cephalometric Landmark Detection

Features:
- HRNet model evaluation
- Argmax coordinate extraction
- Per-image pixel spacing from CSV
- Metrics: MRE ± SD, SDR@{2.0,2.5,3.0,4.0}mm
- Comparison with Khan et al. (2025) baseline

Usage:
    python evaluate.py
    python evaluate.py --checkpoint checkpoints/best_model.pth
"""

import argparse
import os
import sys

import torch
from tqdm import tqdm

# Ensure repo root and src are importable
sys.path.insert(0, os.path.dirname(__file__))

from src.config.config import TrainConfig
from src.data.dataset_enhanced import get_dataloader
from src.models.hrnet import HRNet
from src.utils.metrics import MetricsTracker, extract_coordinates_from_heatmaps


KHAN_2025_BASELINE = {
    "MRE": 1.69,
    "SDR@2.0mm": 81.18,
    "SDR@2.5mm": 87.28,
    "SDR@3.0mm": 90.82,
    "SDR@4.0mm": 94.82,
}


def build_model(config: TrainConfig) -> torch.nn.Module:
    model = HRNet(num_landmarks=config.NUM_LANDMARKS, use_multitask=config.USE_MULTITASK)
    return model


def load_checkpoint(model: torch.nn.Module, checkpoint_path: str, device: torch.device) -> dict:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = checkpoint["model_state_dict"] if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint else checkpoint
    model.load_state_dict(state_dict)
    return checkpoint if isinstance(checkpoint, dict) else {}


def evaluate(config: TrainConfig, checkpoint_path: str, split: str = "test") -> dict:
    device = torch.device(config.DEVICE)

    model = build_model(config).to(device)
    ckpt_meta = load_checkpoint(model, checkpoint_path, device)
    model.eval()

    dataloader = get_dataloader(config, mode=split, return_pixel_spacing=True)

    tracker = MetricsTracker(sdr_thresholds=config.SDR_THRESHOLDS)

    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f"Evaluating {split}"):
            # dataset_enhanced returns (image, heatmaps, landmarks, cvm, pixel_spacing)
            images, _target_heatmaps, landmarks, _cvm, pixel_spacings = batch
            images = images.to(device)

            outputs = model(images)
            pred_heatmaps = outputs[0] if isinstance(outputs, (tuple, list)) else outputs

            # Extract coordinates at heatmap resolution
            pred_coords = extract_coordinates_from_heatmaps(pred_heatmaps, method="argmax")

            # Scale 128 -> 512 (matches train_v2.py)
            scale_x = config.INPUT_SIZE[1] / config.HEATMAP_SIZE[1]
            scale_y = config.INPUT_SIZE[0] / config.HEATMAP_SIZE[0]
            pred_coords[:, :, 0] *= scale_x
            pred_coords[:, :, 1] *= scale_y

            # Update per-sample pixel spacing
            # tracker.update expects scalar or array-like length B
            tracker.update(pred_coords, landmarks, pixel_spacing=pixel_spacings.numpy())

    metrics = tracker.compute()

    # Attach checkpoint metadata if present
    if ckpt_meta:
        metrics["checkpoint_epoch"] = ckpt_meta.get("epoch", None)
        metrics["checkpoint_best_mre"] = ckpt_meta.get("best_mre", None)

    return metrics


def print_comparison(metrics: dict):
    def _get(key: str):
        return metrics.get(key, None)

    print("\n" + "=" * 80)
    print("EVALUATION RESULTS - TEST SET")
    print("=" * 80)

    epoch = _get("checkpoint_epoch")
    best_mre = _get("checkpoint_best_mre")
    if epoch is not None:
        print(f"Checkpoint epoch: {epoch}")
    if best_mre is not None:
        try:
            print(f"Checkpoint stored best MRE: {best_mre:.3f} mm")
        except Exception:
            print("Checkpoint stored best MRE: (unavailable)")

    print("\nYour model:")
    print(f"  MRE: {_get('MRE'):.3f} ± {_get('STD'):.3f} mm")
    for t in [2.0, 2.5, 3.0, 4.0]:
        k = f"SDR@{t}mm"
        print(f"  {k}: {_get(k):.2f}%")

    print("\nKhan et al. (2025) baseline (GitHub):")
    print(f"  MRE: {KHAN_2025_BASELINE['MRE']:.2f} mm")
    for k in ["SDR@2.0mm", "SDR@2.5mm", "SDR@3.0mm", "SDR@4.0mm"]:
        print(f"  {k}: {KHAN_2025_BASELINE[k]:.2f}%")

    print("\nDelta (Your - Khan2025):")
    print(f"  MRE: {_get('MRE') - KHAN_2025_BASELINE['MRE']:+.3f} mm")
    for t in [2.0, 2.5, 3.0, 4.0]:
        k_eval = f"SDR@{t}mm"
        k_ref = f"SDR@{t:.1f}mm"
        print(f"  {k_ref}: {_get(k_eval) - KHAN_2025_BASELINE[k_ref]:+.2f}%")

    print("=" * 80)

    # Per-landmark MRE
    per_lm = metrics.get("per_landmark_error", None)
    if per_lm is not None:
        LANDMARK_SHORT = [
            "A", "ANS", "B", "Me", "N", "Or", "Pog", "PNS", "Prn", "R",
            "S", "Ar", "Co", "Gn", "Go", "Po", "L5", "L1", "L6",
            "U5", "U1A", "U1", "U6", "L1A", "Li", "Ls", "N'", "Pog'", "Sn",
        ]
        LANDMARK_NAMES = [
            "A-point", "Anterior Nasal Spine", "B-point", "Menton", "Nasion",
            "Orbitale", "Pogonion", "Posterior Nasal Spine", "Pronasale", "Ramus",
            "Sella", "Articulare", "Condylion", "Gnathion", "Gonion",
            "Porion", "Lower 2nd PM Cusp Tip", "Lower Incisor Tip", "Lower Molar Cusp Tip",
            "Upper 2nd PM Cusp Tip", "Upper Incisor Apex", "Upper Incisor Tip",
            "Upper Molar Cusp Tip", "Lower Incisor Apex", "Labrale inferius",
            "Labrale superius", "Soft Tissue Nasion", "Soft Tissue Pogonion", "Subnasale",
        ]
        print("\nPer-Landmark MRE (mm):")
        print(f"  {'#':<4} {'Short':<6} {'Landmark':<28} {'MRE (mm)':>10}")
        print(f"  {'─'*4} {'─'*6} {'─'*28} {'─'*10}")
        for i, err in enumerate(per_lm):
            short = LANDMARK_SHORT[i] if i < len(LANDMARK_SHORT) else f"L{i}"
            name = LANDMARK_NAMES[i] if i < len(LANDMARK_NAMES) else f"Landmark {i}"
            print(f"  {i+1:<4} {short:<6} {name:<28} {err:>10.3f}")
        print(f"\n  Overall MRE: {_get('MRE'):.3f} mm")
        print("=" * 80)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=os.path.join("checkpoints", "best_model.pth"),
    )
    parser.add_argument("--split", type=str, default="test", choices=["train", "valid", "test"])
    args = parser.parse_args()

    config = TrainConfig()

    metrics = evaluate(config=config, checkpoint_path=args.checkpoint, split=args.split)
    print_comparison(metrics)


if __name__ == "__main__":
    main()
