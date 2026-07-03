import pandas as pd
import matplotlib.pyplot as plt

ours_csv = "robustness_results.csv"
baseline_csv = "robustness_results_densenet121.csv"
df_ours = pd.read_csv(ours_csv)
df_base = pd.read_csv(baseline_csv)
df_ours["Model"] = "Proposed model"
df_base["Model"] = "DenseNet121"
df = pd.concat([df_base, df_ours], ignore_index=True)

def make_condition(row):
    corruption = str(row["Corruption"]).strip().lower()
    if corruption == "clean":
        return "clean"
    severity = str(row["Severity"]).strip()
    return f"{corruption}_{severity}"
df["Condition"] = df.apply(make_condition, axis=1)
condition_order = [
    "clean",
    "blur_3", "blur_5", "blur_7",
    "brightness_0.8", "brightness_0.9", "brightness_1.1", "brightness_1.2",
    "jpeg_50", "jpeg_70", "jpeg_90",
    "noise_0.02", "noise_0.05", "noise_0.08"
]
df["Condition"] = pd.Categorical(df["Condition"], categories=condition_order, ordered=True)
df = df.sort_values(["Condition", "Model"])
plot_base = df[df["Model"] == "DenseNet121"].sort_values("Condition")
plot_ours = df[df["Model"] == "Proposed model"].sort_values("Condition")
plt.figure(figsize=(14, 6))
plt.plot(
    plot_base["Condition"],
    plot_base["AUC"],
    marker="o",
    linewidth=2,
    label="DenseNet121"
)
plt.plot(
    plot_ours["Condition"],
    plot_ours["AUC"],
    marker="o",
    linewidth=2,
    label="Proposed model"
)
plt.xticks(rotation=45, ha="right")
plt.xlabel("Corruption condition")
plt.ylabel("AUC")
plt.title("Robustness comparison under different corruption conditions")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig("robustness_auc_comparison_from_csv.png", dpi=300, bbox_inches="tight")
plt.show()
