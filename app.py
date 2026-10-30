"""Interactive local demo for DataSentry AI.

Run with ``streamlit run app.py`` from the repository root.
"""

from __future__ import annotations

import hashlib
from io import BytesIO
from pathlib import Path

import pandas as pd
import streamlit as st

from datasentry.experiment import ExperimentConfig, run_experiment
from datasentry.quality import profile_dataframe


ROOT = Path(__file__).resolve().parent
DEMO_CSV = ROOT / "data" / "student-por.csv"
MODEL_NAMES = {"ridge": "Ridge regression", "forest": "Random forest", "mlp": "Neural MLP"}
MECHANISM_NAMES = {
    "mcar": "Random missing cells",
    "mar": "Feature-dependent missingness",
    "extreme": "Numeric extremes",
}
STRATEGY_NAMES = {
    "median": "Median / mode",
    "knn": "KNN numeric imputation",
    "iqr": "Median / mode + IQR clipping",
    "adaptive": "Training-guided selection",
    "dedup": "Remove exact training duplicates (exploratory)",
    "normalize": "Normalize categorical text (exploratory)",
}
PAPER_STRATEGIES = ("median", "knn", "iqr", "adaptive")


st.set_page_config(page_title="DataSentry AI", page_icon="🔎", layout="wide")


def load_csv(source: BytesIO | Path, delimiter: str) -> pd.DataFrame:
    """Read an uploaded or bundled CSV without modifying its contents."""
    separator = {"Auto detect": None, "Comma": ",", "Semicolon": ";", "Tab": "\t"}[delimiter]
    frame = pd.read_csv(source, sep=separator, engine="python" if separator is None else "c")
    if frame.empty or frame.shape[1] < 2:
        raise ValueError("The CSV needs at least one row and two columns (a feature and a target).")
    if frame.columns.has_duplicates:
        raise ValueError("The CSV has repeated column names. Rename them before comparison.")
    return frame


def show_profile(frame: pd.DataFrame, target: str) -> None:
    report = profile_dataframe(frame, target=target)
    overview = report["overview"]
    cols = st.columns(4)
    cols[0].metric("Rows", f"{overview['rows']:,}")
    cols[1].metric("Columns", f"{overview['columns']:,}")
    cols[2].metric("Missing cells", f"{overview['missing_cells']:,} ({overview['missing_pct']:.1f}%)")
    cols[3].metric("Exact duplicate rows", f"{overview['duplicate_rows']:,} ({overview['duplicate_pct']:.1f}%)")

    column_table = pd.DataFrame(report["columns"])
    if not column_table.empty:
        st.subheader("Column quality")
        st.dataframe(column_table, use_container_width=True, hide_index=True)
        missing = column_table.loc[column_table["missing_count"] > 0, ["name", "missing_pct"]]
        if not missing.empty:
            st.bar_chart(missing.set_index("name")["missing_pct"], y_label="Missing cells (%)")

    left, right = st.columns(2)
    with left:
        st.subheader("Potential numeric outliers")
        outliers = pd.DataFrame(report["outliers"])
        if outliers.empty:
            st.info("No values were flagged by the profiling rule.")
        else:
            st.dataframe(outliers, use_container_width=True, hide_index=True)
        st.caption("A statistical flag is a review prompt, not proof of a data error.")
    with right:
        st.subheader("Inconsistent text patterns")
        patterns = report["inconsistent_patterns"]
        if not patterns:
            st.info("No case or whitespace variants were flagged.")
        else:
            for item in patterns:
                variants = ", ".join(f"{part['value']} ({part['count']})" for part in item["variants"])
                st.markdown(f"**{item['column']}** · {item['affected_rows']} affected rows")
                st.caption(f"Normalized form: {item['canonical']} · Variants: {variants}")
    for note in report.get("notes", []):
        st.caption(str(note))


def show_results(results: pd.DataFrame) -> None:
    required = {"seed", "mechanism", "level", "strategy", "model", "rmse", "mae", "r2"}
    if results.empty or not required.issubset(results.columns):
        st.error("The experiment returned no usable metric rows.")
        return
    st.success(f"Completed {len(results):,} fitted model comparisons. Metrics below use held-out rows.")
    if set(results["strategy"]) - set(PAPER_STRATEGIES):
        st.info("Duplicate removal and text normalization are exploratory analyses. They are outside the four-strategy paper grid.")
    comparisons = results.copy()
    baseline = (
        comparisons.loc[comparisons["strategy"] == "median",
                        ["seed", "mechanism", "level", "model", "rmse"]]
        .rename(columns={"rmse": "median_rmse"})
    )
    comparisons = comparisons.merge(baseline, on=["seed", "mechanism", "level", "model"],
                                    how="left", validate="many_to_one")
    comparisons["rmse_change_vs_median"] = comparisons["rmse"] - comparisons["median_rmse"]
    summary = (
        comparisons.groupby(["mechanism", "level", "strategy", "model"], as_index=False)
        .agg(rmse_mean=("rmse", "mean"), rmse_sd=("rmse", "std"),
             mae_mean=("mae", "mean"), r2_mean=("r2", "mean"),
             rmse_change_vs_median=("rmse_change_vs_median", "mean"))
        .sort_values(["mechanism", "level", "model", "rmse_mean"])
    )
    st.subheader("Comparison by condition")
    st.dataframe(summary.round(4), use_container_width=True, hide_index=True)
    st.caption("RMSE change is paired to median / mode on the same split and corruption mask. Negative values favor the listed strategy. Standard deviation is blank for a single seed.")

    view_model = st.selectbox("Chart model", sorted(results["model"].unique()),
                              format_func=lambda value: MODEL_NAMES.get(value, value))
    view_mechanism = st.selectbox("Chart corruption", sorted(results["mechanism"].unique()),
                                  format_func=lambda value: MECHANISM_NAMES.get(value, value))
    chart_rows = summary.loc[(summary["model"] == view_model) &
                             (summary["mechanism"] == view_mechanism)]
    chart = chart_rows.pivot(index="level", columns="strategy", values="rmse_mean")
    chart.columns = [STRATEGY_NAMES.get(column, column) for column in chart.columns]
    st.line_chart(chart, x_label="Corrupted training cells (fraction)", y_label="Held-out RMSE")
    st.caption("Lower RMSE is better. The held-out test rows stay untouched in this comparison.")

    with st.expander("Individual runs and selected actions"):
        st.dataframe(comparisons, use_container_width=True, hide_index=True)
        st.download_button("Download run records (CSV)", comparisons.to_csv(index=False),
                           file_name="datasentry_experiment_results.csv", mime="text/csv")


def main() -> None:
    st.title("DataSentry AI")
    st.markdown("**Detecting unreliable data before machine learning**")
    st.caption("Inspect CSV quality and compare cleaning strategies by their effect on held-out prediction.")

    with st.sidebar:
        st.header("Dataset")
        source_choice = st.radio("Source", ["Bundled student example", "Upload a CSV"])
        upload = st.file_uploader("Choose a CSV", type="csv") if source_choice == "Upload a CSV" else None
        delimiter = st.selectbox("Delimiter", ["Auto detect", "Comma", "Semicolon", "Tab"],
                                 index=1 if source_choice == "Bundled student example" else 0)

    if source_choice == "Upload a CSV" and upload is None:
        st.info("Upload a CSV to begin, or choose the bundled student example.")
        return
    try:
        frame = load_csv(upload if upload is not None else DEMO_CSV, delimiter)
    except (OSError, UnicodeError, ValueError, pd.errors.ParserError) as exc:
        st.error(f"Could not read the CSV: {exc}")
        return
    source_bytes = upload.getvalue() if upload is not None else DEMO_CSV.read_bytes()
    source_sha256 = hashlib.sha256(source_bytes).hexdigest()

    default_target = frame.columns.get_loc("G3") if "G3" in frame.columns else len(frame.columns) - 1
    with st.sidebar:
        target = st.selectbox("Prediction target", list(frame.columns), index=int(default_target))
        default_exclusions = [name for name in ("G1", "G2", "absences") if name in frame.columns and name != target]
        exclusions = st.multiselect("Exclude unavailable or leakage-prone features",
                                    [name for name in frame.columns if name != target],
                                    default=default_exclusions)
        st.caption("For the student example, later-period grades and absences are excluded from the early-prediction task.")

    tabs = st.tabs(["Quality profile", "Cleaning experiment", "Protocol"])
    with tabs[0]:
        st.subheader("Raw dataset preview")
        st.dataframe(frame.head(20), use_container_width=True)
        try:
            show_profile(frame, target)
        except (TypeError, ValueError) as exc:
            st.error(f"Could not profile this CSV: {exc}")
    with tabs[1]:
        if not pd.api.types.is_numeric_dtype(frame[target]):
            st.warning("This research comparison currently requires a numeric regression target.")
        elif frame[target].isna().any():
            st.warning("The selected target contains missing values. Choose a complete target or resolve its missing labels first.")
        else:
            st.markdown("Every recipe uses the **same splits and corruption masks**. Cleaning is fitted on training rows; evaluation uses untouched held-out rows.")
            left, right = st.columns(2)
            with left:
                mechanisms = st.multiselect("Training corruption", list(MECHANISM_NAMES),
                                            default=["mcar", "extreme"],
                                            format_func=lambda value: MECHANISM_NAMES[value])
                levels_pct = st.multiselect("Corruption levels (%)", [0, 5, 15, 30], default=[0, 15])
                strategies = st.multiselect("Cleaning strategies", list(STRATEGY_NAMES),
                                            default=list(PAPER_STRATEGIES),
                                            format_func=lambda value: STRATEGY_NAMES[value])
                st.caption("The four selected strategies form the paper comparison. Duplicate removal and text normalization are optional exploratory checks for uploaded CSVs.")
            with right:
                models = st.multiselect("Predictive models", list(MODEL_NAMES),
                                        default=["ridge", "forest"],
                                        format_func=lambda value: MODEL_NAMES[value])
                seed_count = st.slider("Paired split seeds", min_value=1, max_value=5, value=1)
                st.caption("A five-seed run provides a more stable estimate but takes longer.")
            run_count = seed_count * len(mechanisms) * len(levels_pct) * len(strategies) * len(models)
            st.caption(f"Planned model fits: {run_count:,}. Results are measured only after you run them.")
            run_spec = (source_sha256, target, tuple(exclusions), tuple(mechanisms),
                        tuple(sorted(levels_pct)), tuple(strategies), tuple(models), seed_count)
            if st.button("Run held-out comparison", type="primary", disabled=run_count == 0):
                config = ExperimentConfig(
                    seeds=(11, 23, 37, 53, 71)[:seed_count],
                    levels=tuple(value / 100 for value in sorted(levels_pct)),
                    mechanisms=tuple(mechanisms),
                    strategies=tuple(strategies),
                    models=tuple(models),
                    drop_columns=tuple(exclusions),
                )
                with st.spinner("Fitting and evaluating paired comparisons…"):
                    try:
                        st.session_state["results"] = run_experiment(frame, target, config=config)
                        st.session_state["results_spec"] = run_spec
                    except (TypeError, ValueError) as exc:
                        st.error(f"Experiment could not run: {exc}")
                if st.session_state.get("results_spec") == run_spec:
                    show_results(st.session_state["results"])
            elif st.session_state.get("results_spec") == run_spec:
                st.info("Showing the last completed run for these settings.")
                show_results(st.session_state["results"])
    with tabs[2]:
        st.subheader("What the comparison measures")
        st.markdown(
            """1. Split rows into training and held-out test sets; identical feature rows stay in one split.
2. Corrupt selected **training features** at controlled levels. Never corrupt the target.
3. Fit each cleaning pipeline and model on training rows only.
4. Predict the same untouched test rows and compare RMSE, MAE, and R².
5. Keep dataset identity, split seed, corruption, and chosen strategy in the run records."""
        )
        st.caption("This app is a local research demonstration; flagged outliers or text variants require contextual review.")


if __name__ == "__main__":
    main()
