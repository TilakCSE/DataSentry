"""Validate saved benchmark runs and create the paper's measured summaries.

Run only after the three prespecified full grids have completed. Figures are
derived from recorded held-out results; no values are embedded in this script.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np
import pandas as pd


DATASETS = {
    "student": "Student performance",
    "wine": "Red wine quality",
    "energy": "Energy heating load",
}
MECHANISMS = ("mcar", "mar", "extreme")
MODELS = ("ridge", "forest", "mlp")
STRATEGIES = ("median", "knn", "iqr", "adaptive")
COLORS = {"median": "#263b62", "knn": "#168a8a", "iqr": "#b75b29", "adaptive": "#7154a4"}


def load_checked(folder: Path) -> pd.DataFrame:
    frames = []
    key = ["seed", "mechanism", "level", "model"]
    for short in DATASETS:
        frame = pd.read_csv(folder / f"{short}_full.csv")
        expected_rows = 5 * 3 * 4 * 4 * 3
        if len(frame) != expected_rows:
            raise ValueError(f"{short}: expected {expected_rows} rows, found {len(frame)}")
        if frame[key + ["strategy"]].duplicated().any():
            raise ValueError(f"{short}: duplicate experiment arm")
        if frame.groupby(key)["strategy"].nunique().ne(4).any():
            raise ValueError(f"{short}: a paired condition lacks a strategy")
        if frame.groupby(key)[["split_sha256", "mask_sha256"]].nunique().gt(1).any().any():
            raise ValueError(f"{short}: strategies received different splits or masks")
        if set(frame.seed) != {11, 23, 37, 53, 71}:
            raise ValueError(f"{short}: unexpected split seeds")
        if set(frame.mechanism) != set(MECHANISMS) or set(frame.model) != set(MODELS):
            raise ValueError(f"{short}: experiment factors differ from protocol")
        if set(np.round(frame.level, 4)) != {0, 0.05, 0.15, 0.30}:
            raise ValueError(f"{short}: corruption levels differ from protocol")
        frame["dataset"] = short
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def paired_summary(all_runs: pd.DataFrame) -> pd.DataFrame:
    keys = ["dataset", "seed", "mechanism", "level", "model"]
    baseline = all_runs.loc[all_runs.strategy.eq("median"), keys + ["rmse"]]
    baseline = baseline.rename(columns={"rmse": "median_rmse"})
    joined = all_runs.merge(baseline, on=keys, validate="many_to_one")
    joined["paired_rmse_delta"] = joined.rmse - joined.median_rmse
    joined["paired_percent_delta"] = 100 * joined.paired_rmse_delta / joined.median_rmse
    return joined.groupby(["dataset", "mechanism", "level", "model", "strategy"],
                          as_index=False).agg(
        mean_rmse=("rmse", "mean"), sd_rmse=("rmse", "std"),
        mean_mae=("mae", "mean"), mean_r2=("r2", "mean"),
        mean_paired_rmse_delta=("paired_rmse_delta", "mean"),
        sd_paired_rmse_delta=("paired_rmse_delta", "std"),
        mean_paired_percent_delta=("paired_percent_delta", "mean"),
        mean_runtime_seconds=("runtime_seconds", "mean"),
    )


def plot_heatmap(summary: pd.DataFrame, output: Path) -> None:
    selected = summary.loc[summary.strategy.eq("adaptive") & summary.level.eq(0.30)]
    limit = max(1.0, float(selected.mean_paired_percent_delta.abs().max()))
    fig, axes = plt.subplots(1, 3, figsize=(11.3, 3.5), constrained_layout=True)
    norm = TwoSlopeNorm(vmin=-limit, vcenter=0, vmax=limit)
    image = None
    for axis, dataset in zip(axes, DATASETS):
        rows = selected.loc[selected.dataset.eq(dataset)].pivot(
            index="mechanism", columns="model", values="mean_paired_percent_delta"
        ).reindex(index=MECHANISMS, columns=MODELS)
        image = axis.imshow(rows.to_numpy(), cmap="RdBu_r", norm=norm, aspect="auto")
        axis.set_title(DATASETS[dataset], fontsize=10, weight="bold")
        axis.set_xticks(range(3), ["Ridge", "Forest", "MLP"])
        axis.set_yticks(range(3), ["Random missing", "Feature-dependent", "Numeric extremes"])
        axis.tick_params(labelsize=8)
        for i in range(3):
            for j in range(3):
                value = rows.iloc[i, j]
                axis.text(j, i, f"{value:+.1f}%", ha="center", va="center", fontsize=8,
                          color="white" if abs(value) > 0.55 * limit else "#182333")
    fig.colorbar(image, ax=axes, shrink=0.82, pad=0.015,
                 label="Policy minus median/mode: paired RMSE change (%)")
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


def plot_mcar_curves(summary: pd.DataFrame, output: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(11.3, 3.45), constrained_layout=True)
    for axis, dataset in zip(axes, DATASETS):
        subset = summary.loc[summary.dataset.eq(dataset) & summary.mechanism.eq("mcar")
                             & summary.model.eq("ridge")]
        for strategy in STRATEGIES:
            rows = subset.loc[subset.strategy.eq(strategy)].sort_values("level")
            axis.errorbar(100 * rows.level, rows.mean_rmse, yerr=rows.sd_rmse / np.sqrt(5),
                          marker="o", markersize=3.5, linewidth=1.5, capsize=2.5,
                          color=COLORS[strategy], label=strategy.title())
        axis.set_title(DATASETS[dataset], fontsize=10, weight="bold")
        axis.set_xlabel("Training cells masked (%)")
        axis.grid(axis="y", color="#d9dde5", linewidth=0.6)
        axis.tick_params(labelsize=8)
    axes[0].set_ylabel("Held-out RMSE (dataset target units)")
    handles, labels = axes[-1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=4, fontsize=8)
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=Path("paper/results"))
    parser.add_argument("--output-dir", type=Path, default=Path("paper"))
    args = parser.parse_args()
    runs = load_checked(args.input_dir)
    summary = paired_summary(runs)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.output_dir / "study_summary.csv", index=False)
    plot_heatmap(summary, args.output_dir / "policy_heatmap.pdf")
    plot_mcar_curves(summary, args.output_dir / "mcar_curves.pdf")
    policy = summary.loc[summary.strategy.eq("adaptive") & summary.level.gt(0)]
    print(f"Validated {len(runs)} fitted arms across {len(DATASETS)} datasets.")
    print(f"Policy lower/tied/higher RMSE than median across condition means: "
          f"{(policy.mean_paired_rmse_delta < -1e-9).sum()}/"
          f"{(policy.mean_paired_rmse_delta.abs() <= 1e-9).sum()}/"
          f"{(policy.mean_paired_rmse_delta > 1e-9).sum()}")


if __name__ == "__main__":
    main()
