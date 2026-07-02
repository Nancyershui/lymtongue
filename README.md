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



# File Inventory

This document records what each script is for after repository cleanup.

## Main Training

| File | Purpose |
| --- | --- |
| `scripts/train/train_proposed_densemamba.py` | Final proposed model. DenseNet121 is modified to output 14x14 features, denseblock4 uses mixed dilation, and Coord-Mamba is used for feature modeling. Uses the paper protocol: 224x224 inputs, 30 epochs, Adam with lr=1e-4, and five independent runs summarized by mean/std. |
| `scripts/train/train_densenet_baseline.py` | DenseNet121 baseline with the same five-run paper protocol. |
| `scripts/train/train_baselines.py` | Unified baseline runner. The `build_baseline_model()` function covers DenseNet121, ResNet18, VGG16, InceptionV3, EfficientNet-B0, ViT-Base, Swin-Tiny, and VMamba under one shared training/evaluation pipeline. |

## Ablation Experiments

| File | Purpose |
| --- | --- |
| `scripts/ablation/train_no_coord_mamba.py` | Ablation without Coord-Mamba; keeps 14x14 feature output and mixed dilation. |
| `scripts/ablation/train_no_mixed_dilation.py` | Ablation without mixed dilation; keeps 14x14 feature output and Coord-Mamba. |
| `scripts/ablation/train_mixed_dilation_only.py` | Ablation with mixed dilation only; no 14x14 feature output modification and no Coord-Mamba. |
| `scripts/ablation/train_coord_mamba_only.py` | Ablation with Coord-Mamba only; no A2-style 14x14 feature output and no mixed dilation. |

## Visualization

| File | Purpose |
| --- | --- |
| `scripts/visualization/visualize_attention_peak_region.py` | Main attention visualization script using attention pooling, peak connected region filtering, and tongue-region constrained heatmaps. |
| `scripts/visualization/visualize_gradcam_comparison.py` | Grad-CAM comparison script for trained DenseNet121 and LymTongue checkpoints, intended to reproduce the paper-style visualization comparison. |
| `scripts/visualization/visualize_proposed_fig9_v1.py` | Earlier Fig.9-style visualization for top discriminative regions and masked response maps. |
| `scripts/visualization/visualize_gradcam_imagenet.py` | Generic DenseNet121 Grad-CAM utility using ImageNet pretrained weights. Useful as a sanity-check visualization script, not the proposed model visualization. |

## Analysis Utilities

| File | Purpose |
| --- | --- |
| `scripts/analysis/count_parameters.py` | Counts model parameters for the proposed DenseNet + Coord-Mamba architecture and fallback variants. |
| `scripts/analysis/receptive_field.py` | Computes and visualizes receptive fields for the modified DenseNet121 backbone. |

## Archived Prototypes

| File | Purpose |
| --- | --- |
| `archive/prototypes/proposed_v1.py` | Early proposed-model training script. |
| `archive/prototypes/proposed_visual_v3.py` | Later visualization-oriented proposed-model variant with tightened heatmap display. |
| `archive/prototypes/train_dense_mamba_early.py` | Earlier DenseNet + Mamba training attempt. |
| `archive/prototypes/train_proposed_384_keep_ratio_pad.py` | Exploratory 384 input keep-ratio padding experiment. Archived because the paper protocol uses 224x224 inputs. |

## Generated Or Local Files

| Path | Purpose |
| --- | --- |
| `checkpoints/` | Local model checkpoints. Do not commit large checkpoint files unless the journal or release plan requires them. |
| `__pycache__/` | Python cache files. Safe to delete locally and ignored by git. |
