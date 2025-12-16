# Cephalometric Landmark Detection

AI-powered automatic detection of 29 cephalometric landmarks from lateral cephalometric radiographs using HRNet architecture.

## Performance

| Metric | This Model | Khan et al. (2025) |
|--------|-----------|-------------------|
| **MRE** | **0.83 ± 0.64 mm** | 1.69 ± 3.36 mm |
| **SDR@2mm** | **94.69%** | 81.18% |
| **SDR@2.5mm** | **97.52%** | 87.28% |
| **SDR@3mm** | **98.83%** | 90.82% |
| **SDR@4mm** | **99.63%** | 94.82% |

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd AI-Cephalometric-Model

# Install dependencies
pip install -r requirements.txt
```

## Project Structure

```
├── train.py              # Training script
├── evaluate.py           # Evaluation script
├── gui.py                # GUI application
├── requirements.txt      # Python dependencies
├── checkpoints/          # Model checkpoints
├── logs/                 # TensorBoard logs
├── dataset/              # Dataset directory
└── src/
    ├── config/
    │   └── config.py     # Training configuration
    ├── data/
    │   ├── dataset_enhanced.py
    │   └── pixel_spacing.py
    ├── models/
    │   ├── hrnet.py      # HRNet architecture
    │   └── unet_plusplus.py
    └── utils/
        ├── losses.py
        ├── losses_improved.py
        └── metrics.py
```

## Usage

### Training

```bash
# Train from scratch
python train.py

# Resume from checkpoint
python train.py --resume checkpoints/checkpoint_epoch_50.pth
```

### Evaluation

```bash
# Evaluate on test set
python evaluate.py

# Evaluate specific checkpoint
python evaluate.py --checkpoint checkpoints/best_model.pth
```

### GUI Application

```bash
python gui.py
```

The GUI allows you to:
- Load cephalometric images
- Run inference with the trained model
- Compare predictions with ground truth
- Calculate MRE and SDR metrics
- Switch between argmax and soft_argmax coordinate extraction

## Landmarks (29 total)

| # | Short | Full Name |
|---|-------|-----------|
| 1 | A | A-point |
| 2 | ANS | Anterior Nasal Spine |
| 3 | B | B-point |
| 4 | Me | Menton |
| 5 | N | Nasion |
| 6 | Or | Orbitale |
| 7 | Pog | Pogonion |
| 8 | PNS | Posterior Nasal Spine |
| 9 | Prn | Pronasale |
| 10 | R | Ramus |
| 11 | S | Sella |
| 12 | Ar | Articulare |
| 13 | Co | Condylion |
| 14 | Gn | Gnathion |
| 15 | Go | Gonion |
| 16 | Po | Porion |
| 17 | L5 | Lower 2nd PM Cusp Tip |
| 18 | L1 | Lower Incisor Tip |
| 19 | L6 | Lower Molar Cusp Tip |
| 20 | U5 | Upper 2nd PM Cusp Tip |
| 21 | U1A | Upper Incisor Apex |
| 22 | U1 | Upper Incisor Tip |
| 23 | U6 | Upper Molar Cusp Tip |
| 24 | L1A | Lower Incisor Apex |
| 25 | Li | Labrale inferius |
| 26 | Ls | Labrale superius |
| 27 | N' | Soft Tissue Nasion |
| 28 | Pog' | Soft Tissue Pogonion |
| 29 | Sn | Subnasale |

## Configuration

Edit `src/config/config.py` to modify:
- Input size (default: 864x768)
- Batch size (default: 3)
- Learning rate (default: 1e-4)
- Number of epochs (default: 300)
- Loss function (default: adaptive_wing)

## Dataset

Based on the [Aariz Dataset](https://doi.org/10.6084/m9.figshare.27986417.v1):
- 1000 lateral cephalometric radiographs
- 29 annotated landmarks per image
- 6 CVM stages
- Train/Valid/Test split

## Citation

```bibtex
@article{khalid2025benchmark,
  title={A Benchmark Dataset for Automatic Cephalometric Landmark Detection and CVM Stage Classification},
  author={Khalid, Muhammad Anwaar and others},
  journal={Scientific Data},
  volume={12},
  pages={1336},
  year={2025},
  publisher={Nature Publishing Group}
}
```

## License

See [LICENSE](LICENSE) file.
