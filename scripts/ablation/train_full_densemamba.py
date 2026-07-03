from component_ablation_common import run_component_ablation

if __name__ == "__main__":
    run_component_ablation(
        variant_name="full_14x14_mixed_dilation_coord_mamba",
        use_14x14=True,
        use_mixed_dilation=True,
        use_coord_mamba=True,
        description="Ablation table full model: 14x14 feature resolution + mixed dilation + Coord-Mamba.",
        default_save_name="densenet_mamba_coord_full_ablation.pth",
    )
