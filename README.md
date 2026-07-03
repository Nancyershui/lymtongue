# LymTongue

PyTorch code for binary tongue image classification experiments. The repository contains the proposed DenseMamba/Coord-Mamba model, classification baselines, ablation studies, receptive-field analysis, segmentation-quality analysis, and robustness evaluation.

## Repository Layout

```text
.
├── scripts/
│   ├── train/
│   │   ├── train_proposed_densemamba.py
│   │   ├── train_densenet_baseline.py
│   │   └── train_baselines.py
│   ├── ablation/
│   │   ├── component_ablation_common.py
│   │   ├── train_coord_mamba_only.py
│   │   ├── train_densenet_plain.py
│   │   ├── train_full_densemamba.py
│   │   ├── train_mixed_dilation_only.py
│   │   ├── train_no_coord_mamba.py
│   │   └── train_no_mixed_dilation.py
│   └── analysis/
│       └── receptive_field.py
├── seg/
│   ├── seg_tool/
│   │   ├── seg.py
│   │   └── GroundingDINO_SwinT_OGC.py
│   └── eval/
│       ├── compare_tongue_seg_quality.py
│       ├── plot.py
│       └── seg_quality_results/
├── robust/
│   ├── generate_corruptions.py
│   ├── evaluate_robustness_densemamba.py
│   ├── evaluate_robustness_densenet121.py
│   ├── com.py
│   ├── generate.sh
│   ├── evaluate.sh
│   ├── robustness_eval_results/
│   └── robustness_eval_results_baseline/
├── requirements.txt
├── LICENSE
└── README.md
```

## Installation

Use a separate Python environment. Install PyTorch according to your local CUDA setup, then install the remaining dependencies:

```bash
pip install -r requirements.txt
```

Key dependencies include `torch`, `torchvision`, `scikit-learn`, `timm`, `mamba-ssm`, `causal-conv1d`, `einops`, `opencv-python`, `matplotlib`, `pandas`, `scipy`, `scikit-image`, and `grad-cam`.

Segmentation generation in `seg/seg_tool/` additionally expects local GroundingDINO and Segment Anything installations plus their checkpoints.

## Code Status

The public code has been cleaned for release. Source-code comments have been removed, repeated blank lines in code files have been compressed, and local machine paths have been replaced by anonymous placeholders such as `/path/to/...`. The repository does not include local datasets, model checkpoints, or external GroundingDINO/SAM/VMamba weights.

## Dataset Format

The training and robustness scripts use `torchvision.datasets.ImageFolder`. The expected dataset layout is:

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

The default dataset path in the scripts is the placeholder `/path/to/tongue_dataset_split`. Replace it with your local dataset path before running, or pass `--data-root` when using scripts that support command-line arguments.

## Training Scripts

| File | Purpose |
| --- | --- |
| `scripts/train/train_proposed_densemamba.py` | Final proposed model. DenseNet121 outputs 14x14 features, denseblock4 uses mixed dilation, and Coord-Mamba is used for feature modeling. The default protocol uses 224x224 inputs, batch size 16, 30 epochs, Adam with lr=1e-4, and 5 independent runs summarized by mean and standard deviation. |
| `scripts/train/train_densenet_baseline.py` | DenseNet121 baseline using the same five-run experimental protocol as the proposed model. |
| `scripts/train/train_baselines.py` | Unified baseline runner for `densenet121`, `resnet18`, `vgg16`, `inception_v3`, `efficientnet_b0`, `vit_base`, `swin_tiny`, and `vmamba`. |

`train_proposed_densemamba.py` and `train_densenet_baseline.py` configure the dataset path through the `data_root` constant near the top of each script. `train_baselines.py` supports command-line arguments.

For the VMamba baseline, `train_baselines.py` imports `VSSM` from a `vmamba` module. It searches the repository root, `VMamba/`, `Baseline/VMamba/`, and the active Python environment.

## Ablation Scripts

| File | Purpose |
| --- | --- |
| `scripts/ablation/component_ablation_common.py` | Shared runner for component-combination ablations over 14x14 feature resolution, mixed dilation, and Coord-Mamba. |
| `scripts/ablation/train_densenet_plain.py` | Plain DenseNet121 ablation without 14x14 feature resolution, mixed dilation, or Coord-Mamba. |
| `scripts/ablation/train_mixed_dilation_only.py` | Keeps mixed dilation only, without the 14x14 feature-output modification or Coord-Mamba. |
| `scripts/ablation/train_coord_mamba_only.py` | Keeps Coord-Mamba only, without the 14x14 feature-output modification or mixed dilation. |
| `scripts/ablation/train_no_coord_mamba.py` | Removes Coord-Mamba while keeping the 14x14 feature output and mixed dilation. |
| `scripts/ablation/train_no_mixed_dilation.py` | Removes mixed dilation while keeping the 14x14 feature output and Coord-Mamba. |
| `scripts/ablation/train_full_densemamba.py` | Full ablation-table model with 14x14 feature resolution, mixed dilation, and Coord-Mamba. |

`train_densenet_plain.py` and `train_full_densemamba.py` call the shared component runner and support command-line arguments such as `--data-root`, `--save-path`, `--img-size`, `--batch-size`, and `--num-epochs`. The standalone ablation scripts configure `data_root`, `save_path`, `batch_size`, `num_epochs`, and related settings as constants near the top of each file.

## Segmentation Analysis

| File | Purpose |
| --- | --- |
| `seg/seg_tool/seg.py` | Grounded-SAM based tongue segmentation pipeline. It loads GroundingDINO and SAM, segments the prompted target, converts white background to black, crops black borders, and writes segmented outputs. |
| `seg/seg_tool/GroundingDINO_SwinT_OGC.py` | GroundingDINO Swin-T configuration used by the segmentation pipeline. |
| `seg/eval/compare_tongue_seg_quality.py` | Unsupervised comparison of GroundedSAM and TongueSAM segmentation outputs. It extracts foreground masks, computes shape and boundary metrics, pairs samples, and writes CSV summaries. |
| `seg/eval/plot.py` | Plots segmentation metric boxplots from `all_metrics.csv`. |
| `seg/eval/seg_quality_results/` | Saved segmentation-quality CSV results and pairing notes. |

Example segmentation-quality evaluation:

```bash
python seg/eval/compare_tongue_seg_quality.py \
  --groundedsam_root /path/to/groundedsam/results \
  --tonguesam_root /path/to/tonguesam/results \
  --output_dir seg/eval/seg_quality_results
```

## Robustness Analysis

| File | Purpose |
| --- | --- |
| `robust/generate_corruptions.py` | Generates corrupted ImageFolder-style test sets with brightness, blur, Gaussian noise, and JPEG compression perturbations. |
| `robust/evaluate_robustness_densemamba.py` | Evaluates a trained DenseMamba checkpoint on clean and corrupted test sets. |
| `robust/evaluate_robustness_densenet121.py` | Evaluates a trained DenseNet121 baseline checkpoint on clean and corrupted test sets. |
| `robust/com.py` | Compares DenseMamba and DenseNet121 robustness CSVs and plots an AUC comparison curve. |
| `robust/generate.sh` and `robust/evaluate.sh` | Example shell commands for generating corruptions and running robustness evaluation. |
| `robust/robustness_eval_results*/` | Saved robustness CSV results. |

Generate default corruptions:

```bash
python robust/generate_corruptions.py \
  --input_root /path/to/tongue_dataset_split/test \
  --output_root /path/to/test_corruptions \
  --mode all
```

Evaluate robustness:

```bash
python robust/evaluate_robustness_densemamba.py \
  --clean_test_root /path/to/tongue_dataset_split/test \
  --corruption_root /path/to/test_corruptions \
  --checkpoint /path/to/densemamba_checkpoint.pth \
  --img_size 320 \
  --batch_size 16 \
  --output_dir robust/robustness_eval_results
```

`robust/evaluate.sh` provides paired DenseMamba and DenseNet121 examples using `224x224` inputs. The Python evaluation scripts default to `320x320` unless `--img_size` is provided.

## Other Analysis

| File | Purpose |
| --- | --- |
| `scripts/analysis/receptive_field.py` | Computes the theoretical receptive field of the modified DenseNet121 backbone and can draw selected receptive-field boxes on a 224x224 input image. |

To draw receptive fields on a real image, update `image_path` near the bottom of `scripts/analysis/receptive_field.py`.

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
  --model vmamba \
  --data-root /path/to/tongue_dataset_split \
  --batch-size 4 \
  --output-dir outputs/baselines
```

Ablation examples:

```bash
python scripts/ablation/train_densenet_plain.py --data-root /path/to/tongue_dataset_split
python scripts/ablation/train_no_coord_mamba.py
python scripts/ablation/train_no_mixed_dilation.py
python scripts/ablation/train_mixed_dilation_only.py
python scripts/ablation/train_coord_mamba_only.py
python scripts/ablation/train_full_densemamba.py --data-root /path/to/tongue_dataset_split
```

Receptive-field analysis:

```bash
python scripts/analysis/receptive_field.py
```

## Outputs

The scripts write result files either to the repository root or to the configured output directory:

- `checkpoints/`: model checkpoints for the proposed model, DenseNet baseline, and ablation runs.
- `densenet_mamba_coord_run*_results.json` and `densenet_mamba_coord_5runs_summary.*`: per-run and five-run summary outputs for the proposed model.
- `densenet121_run*_results.json` and `densenet121_5runs_summary.*`: per-run and five-run summary outputs for the DenseNet121 baseline.
- `outputs/baselines/<model>/`: checkpoints and summary JSON files from the unified baseline runner.
- `seg/eval/seg_quality_results/`: segmentation metric CSV outputs.
- `robust/robustness_eval_results/` and `robust/robustness_eval_results_baseline/`: robustness metric CSV outputs.
- `rf_db4_center_on_resized_input.png` and `rf_multi_on_resized_input.png`: example receptive-field visualization outputs.

Do not commit local datasets, model checkpoints, or large experimental outputs unless they are required for a specific release.

## Notes

- Run scripts from the repository root so relative output paths are created in the expected location.
- The public repository does not include the dataset or trained weights.
- Some scripts use ImageNet pretrained weights. The first run may need network access to download those weights.
- Segmentation generation needs external GroundingDINO/SAM code and checkpoints; segmentation evaluation can run from existing black-background segmentation outputs.
