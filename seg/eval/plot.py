import pandas as pd
import matplotlib.pyplot as plt

csv_path = "seg/eval/seg_quality_results/all_metrics.csv"
metrics = [
    "composite_quality",
    "symmetry",
    "border_touch_ratio",
    "norm_perimeter",
    "boundary_roughness",
    "width_smoothness",
]
title_map = {
    "composite_quality": "composite_quality",
    "symmetry": "symmetry",
    "border_touch_ratio": "border_touch_ratio",
    "norm_perimeter": "norm_perimeter",
    "boundary_roughness": "boundary_roughness",
    "width_smoothness": "width_smoothness",
}
method_name_map = {
    "GroundedSAM": "GDINO-guided SAM",
    "GDINO-guided SAM": "GDINO-guided SAM",
    "TongueSAM": "TongueSAM",
}
method_order = ["GDINO-guided SAM", "TongueSAM"]
df = pd.read_csv(csv_path)
df["method"] = df["method"].replace(method_name_map)
df = df[df["method"].isin(method_order)].copy()
for metric in metrics:
    df[metric] = pd.to_numeric(df[metric], errors="coerce")
method_counts = df.groupby("sample_key")["method"].nunique()
paired_keys = method_counts[method_counts == 2].index
df_paired = df[df["sample_key"].isin(paired_keys)].copy()
print(f"Number of paired samples: {df_paired['sample_key'].nunique()}")
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.titlesize": 13,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "axes.linewidth": 1.0,
    "figure.dpi": 150,
})
fig, axes = plt.subplots(2, 3, figsize=(14, 7.2))
axes = axes.ravel()
for ax, metric in zip(axes, metrics):
    plot_data = []
    labels = []
    for method in method_order:
        values = df_paired.loc[df_paired["method"] == method, metric].dropna().values
        plot_data.append(values)
        labels.append(method)
    ax.boxplot(
        plot_data,
        labels=labels,
        widths=0.18,
        patch_artist=False,
        showfliers=False,
        boxprops=dict(linewidth=1.1),
        whiskerprops=dict(linewidth=1.1),
        capprops=dict(linewidth=1.1),
        medianprops=dict(linewidth=1.2),
    )
    ax.set_title(title_map[metric], pad=7)
    ax.grid(False)
    ax.tick_params(axis="x", rotation=0)
    ax.margins(x=0.28)
ylim_map = {
    "composite_quality": (0.84, 0.99),
    "symmetry": (0.72, 1.005),
    "border_touch_ratio": (-0.01, 0.35),
    "norm_perimeter": (3.65, 5.10),
    "boundary_roughness": (0.0014, 0.0071),
    "width_smoothness": (-0.00002, 0.00105),
}
for ax, metric in zip(axes, metrics):
    if metric in ylim_map:
        ax.set_ylim(ylim_map[metric])
plt.subplots_adjust(
    wspace=0.32,
    hspace=0.38
)
plt.savefig("segmentation_metrics_boxplots.png", dpi=400, bbox_inches="tight")
plt.savefig("segmentation_metrics_boxplots.pdf", bbox_inches="tight")
plt.savefig("segmentation_metrics_boxplots.tiff", dpi=400, bbox_inches="tight")
plt.show()
