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
