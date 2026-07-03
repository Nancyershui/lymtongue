from component_ablation_common import run_component_ablation

if __name__ == "__main__":
    run_component_ablation(
        variant_name="plain_densenet121",
        use_14x14=False,
        use_mixed_dilation=False,
        use_coord_mamba=False,
        description="Ablation: plain DenseNet121 without 14x14 feature resolution, mixed dilation, or Coord-Mamba.",
        default_save_name="densenet_plain_ablation.pth",
    )
