python evaluate_robustness_densemamba.py \
  --clean_test_root /path/to/tongue_dataset_split/test \
  --corruption_root /path/to/test_corruptions \
  --checkpoint /path/to/checkpoints/densemamba.pth \
  --img_size 224 \
  --batch_size 16 \
  --output_dir ./robustness_eval_results

python evaluate_robustness_densenet121.py \
  --clean_test_root /path/to/tongue_dataset_split/test \
  --corruption_root /path/to/test_corruptions \
  --checkpoint /path/to/checkpoints/densenet121.pth \
  --img_size 224 \
  --batch_size 16 \
  --output_dir ./robustness_eval_results_baseline
