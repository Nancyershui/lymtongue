# DenseMamba Tongue Image Classification

This repository contains PyTorch scripts for binary tongue image classification based on DenseNet121 and Coord-Mamba style sequence modeling. The code was organized for paper submission and public release.

## Project Layout

```text
.
├── scripts/
│   ├── train/          # Main training scripts
│   ├── ablation/       # Ablation experiments
│   ├── visualization/  # Attention / CAM visualization scripts
│   └── analysis/       # Parameter counting and receptive field analysis
├── archive/
│   └── prototypes/     # Earlier exploratory versions kept for traceability
├── checkpoints/        # Local model checkpoints, ignored by git
├── docs/
│   └── FILE_INVENTORY.md
├── requirements.txt
└── README.md
```

## Data Format

The training scripts use `torchvision.datasets.ImageFolder` and expect this layout:

```text
DATA_ROOT/
├── train/
│   ├── class_0/
│   └── class_1/
├── val/
│   ├── class_0/
│   └── class_1/
└── test/
    ├── class_0/
    └── class_1/
```

Update `data_root` or pass `--data-root` before running. Placeholder paths such as `/path/to/tongue_dataset_split` are used for anonymized release.

## Main Scripts

- `scripts/train/train_proposed_densemamba.py`: final proposed DenseNet121 + 14x14 feature map + mixed dilation + Coord-Mamba experiment. Runs five independent experiments and reports mean/std over all runs.
- `scripts/train/train_densenet_baseline.py`: DenseNet121 baseline with five independent runs.
- `scripts/train/train_baselines.py`: unified baseline runner for VMamba, ViT-Base, Swin-Tiny, VGG16, ResNet18, InceptionV3, EfficientNet-B0, and DenseNet121 using the paper protocol.
- `scripts/ablation/`: ablation settings used to isolate the contribution of Coord-Mamba, mixed dilation, higher feature resolution, and preprocessing.
- `scripts/visualization/`: attention map and Grad-CAM style visualization utilities.
- `scripts/analysis/`: parameter count and receptive field analysis utilities.

For the current scripts, run commands from the repository root so relative output paths such as `checkpoints/` are created in the expected location.

## Example Usage

```bash
python scripts/train/train_proposed_densemamba.py
python scripts/train/train_densenet_baseline.py
```

Unified baseline training:

```bash
python scripts/train/train_baselines.py --model resnet18 --data-root /path/to/dataset
python scripts/train/train_baselines.py --model efficientnet_b0 --data-root /path/to/dataset
python scripts/train/train_baselines.py --model vit_base --data-root /path/to/dataset
```

Grad-CAM comparison for the paper-style visualization:

```bash
python scripts/visualization/visualize_gradcam_comparison.py \
  --data-root /path/to/dataset \
  --densenet-checkpoint checkpoints/densenet121_run1.pth \
  --lymtongue-checkpoint checkpoints/densenet_mamba_coord_run1.pth
```

## Release Notes

- `checkpoints/`, result JSON files, CAM outputs, and Python cache files are ignored by `.gitignore`.
- Early exploratory scripts are preserved under `archive/prototypes/` instead of being removed.
- Placeholder dataset paths are used to avoid exposing local experiment environments.
