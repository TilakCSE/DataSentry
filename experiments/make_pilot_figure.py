from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

source = Path(r"D:\Development\Projects\DataSentry\experiments\pilot_student_results.csv")
output = Path("pilot_rmse_delta.pdf")
data = pd.read_csv(source)
reference = data[(data.missing_level == 0) & (data.recipe == "median")][
    ["seed", "model", "rmse"]
].rename(columns={"rmse": "reference_rmse"})
data = data.merge(reference, on=["seed", "model"])
data["delta_rmse"] = data.rmse - data.reference_rmse

palette = {"median": "#27678e", "knn": "#b65b35", "adaptive": "#4b7d56"}
labels = {"median": "Median / mode", "knn": "KNN numeric", "adaptive": "Pilot rule"}
fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.2), sharey=True)
for axis, model in zip(axes, ("ridge", "forest")):
    subset = data[data.model == model]
    for recipe in ("median", "knn", "adaptive"):
        points = subset[subset.recipe == recipe].groupby("missing_level").delta_rmse.agg(
            ["mean", "sem"]
        )
        x = points.index.to_numpy() * 100
        axis.errorbar(x, points["mean"], yerr=points["sem"], marker="o",
                      linewidth=1.8, markersize=4, capsize=2.5,
                      color=palette[recipe], label=labels[recipe])
    axis.axhline(0, color="#555555", linewidth=0.8, linestyle="--")
    axis.set_title("Ridge" if model == "ridge" else "Random forest")
    axis.set_xticks([0, 5, 15, 30])
    axis.set_xlabel("Training feature cells masked (%)")
    axis.grid(axis="y", color="#dddddd", linewidth=0.5)
axes[0].set_ylabel("Change in RMSE (grade points)")
axes[1].legend(loc="upper left", frameon=False, fontsize=7)
fig.tight_layout()
fig.savefig(output, bbox_inches="tight")
print(output.resolve())
