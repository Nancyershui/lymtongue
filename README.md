# LymTongue

PyTorch code for binary tongue image classification experiments. The repository is organized around DenseNet121, the proposed DenseMamba/Coord-Mamba model, common classification baselines, ablation studies, and receptive-field analysis.

## Repository Layout

```text
.
├── scripts/
│   ├── train/
│   │   ├── train_proposed_densemamba.py
│   │   ├── train_densenet_baseline.py
│   │   └── train_baselines.py
│   ├── ablation/
│   │   ├── train_coord_mamba_only.py
│   │   ├── train_mixed_dilation_only.py
│   │   ├── train_no_coord_mamba.py
│   │   └── train_no_mixed_dilation.py
│   └── analysis/
│       └── receptive_field.py
├── prototypes/
│   ├── proposed_v1.py
│   ├── proposed_visual_v3.py
│   └── train_dense_mamba_early.py
├── requirements.txt
├── LICENSE
└── README.md
```

## Installation

Use a separate Python environment. Install PyTorch according to your local CUDA setup, then install the remaining dependencies:

```bash
pip install -r requirements.txt
```

Key dependencies include `torch`, `torchvision`, `scikit-learn`, `timm`, `mamba-ssm`, `causal-conv1d`, `einops`, `opencv-python`, `matplotlib`, and `grad-cam`.

## Dataset Format

The training scripts use `torchvision.datasets.ImageFolder`. The expected dataset layout is:

```text
DATA_ROOT/
├── train/
│   ├── lymphoma/
│   └── normal/
├── val/
│   ├── lymphoma/
│   └── normal/
└── test/
    ├── lymphoma/
    └── normal/
```

The default dataset path in the scripts is the anonymized placeholder `/path/to/tongue_dataset_split`. Replace it with your local dataset path before running, or pass `--data-root` when using scripts that support command-line arguments.

## Training Scripts

| File | Purpose |
| --- | --- |
| `scripts/train/train_proposed_densemamba.py` | Final proposed model. DenseNet121 outputs 14x14 features, denseblock4 uses mixed dilation, and Coord-Mamba is used for feature modeling. The default protocol uses 224x224 inputs, batch size 16, 30 epochs, Adam with lr=1e-4, and 5 independent runs summarized by mean and standard deviation. |
| `scripts/train/train_densenet_baseline.py` | DenseNet121 baseline using the same five-run experimental protocol as the proposed model. |
| `scripts/train/train_baselines.py` | Unified baseline runner for `densenet121`, `resnet18`, `vgg16`, `inception_v3`, `efficientnet_b0`, `vit_base`, `swin_tiny`, and `vmamba`. |

`train_proposed_densemamba.py` and `train_densenet_baseline.py` configure the dataset path through the `data_root` constant near the top of each script. `train_baselines.py` supports command-line arguments.

## Ablation Scripts

| File | Purpose |
| --- | --- |
| `scripts/ablation/train_no_coord_mamba.py` | Removes Coord-Mamba while keeping the 14x14 feature output and mixed dilation. |
| `scripts/ablation/train_no_mixed_dilation.py` | Removes mixed dilation while keeping the 14x14 feature output and Coord-Mamba. |
| `scripts/ablation/train_mixed_dilation_only.py` | Keeps mixed dilation only, without the 14x14 feature-output modification or Coord-Mamba. |
| `scripts/ablation/train_coord_mamba_only.py` | Keeps Coord-Mamba only, without the 14x14 feature-output modification or mixed dilation. |

The ablation scripts configure `data_root`, `save_path`, `batch_size`, `num_epochs`, and related settings as constants near the top of each file.

## Analysis and Prototypes

| File | Purpose |
| --- | --- |
| `scripts/analysis/receptive_field.py` | Computes the theoretical receptive field of the modified DenseNet121 backbone and can draw selected receptive-field boxes on a 224x224 input image. |
| `prototypes/proposed_v1.py` | Early training version of the proposed model, kept for traceability. |
| `prototypes/proposed_visual_v3.py` | Earlier visualization-oriented version of the proposed model with a more complete heatmap workflow. |
| `prototypes/train_dense_mamba_early.py` | Earlier DenseNet + Mamba training attempt. |

Files under `prototypes/` are not the current main experiment entry points. They are retained to document exploratory development.

## Usage Examples

Run commands from the repository root. For scripts without command-line dataset arguments, update `data_root` inside the script first.

```bash
python scripts/train/train_proposed_densemamba.py
python scripts/train/train_densenet_baseline.py
```

Unified baseline examples:

```bash
python scripts/train/train_baselines.py \
  --model resnet18 \
  --data-root /path/to/tongue_dataset_split

python scripts/train/train_baselines.py \
  --model efficientnet_b0 \
  --data-root /path/to/tongue_dataset_split \
  --output-dir outputs/baselines
```

Ablation examples:

```bash
python scripts/ablation/train_no_coord_mamba.py
python scripts/ablation/train_no_mixed_dilation.py
python scripts/ablation/train_mixed_dilation_only.py
python scripts/ablation/train_coord_mamba_only.py
```

Receptive-field analysis:

```bash
python scripts/analysis/receptive_field.py
```

To draw receptive fields on a real image, update `image_path` near the bottom of `scripts/analysis/receptive_field.py`.

## Outputs

The training scripts write result files either to the repository root or to the configured output directory:

- `checkpoints/`: model checkpoints for the proposed model, DenseNet baseline, and ablation runs.
- `densenet_mamba_coord_run*_results.json` and `densenet_mamba_coord_5runs_summary.*`: per-run and five-run summary outputs for the proposed model.
- `densenet121_run*_results.json` and `densenet121_5runs_summary.*`: per-run and five-run summary outputs for the DenseNet121 baseline.
- `outputs/baselines/<model>/`: checkpoints and summary JSON files from the unified baseline runner.
- `rf_db4_center_on_resized_input.png` and `rf_multi_on_resized_input.png`: example receptive-field visualization outputs.

Do not commit local datasets, model checkpoints, or large experimental outputs unless they are required for a specific release.

## Notes

- Run scripts from the repository root so relative output paths are created in the expected location.
- The public repository does not include the dataset or trained weights.
- The `vmamba` baseline expects a local `Baseline/VMamba` code directory. If that directory is unavailable, run the built-in baseline models first.
- Some scripts use ImageNet pretrained weights. The first run may need network access to download those weights.
