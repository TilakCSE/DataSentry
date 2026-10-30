# DataSentry AI

**Detecting unreliable data before machine learning**

DataSentry is a local Python research application for inspecting a CSV and measuring how controlled training-data corruption and cleaning choices affect regression on untouched held-out rows. It reports missing cells, exact duplicate rows, potential numeric outliers, and conservative text-pattern variants. The experiment compares fixed and training-guided preprocessing with Ridge regression, random forest, and a small neural multilayer perceptron.

The current research question is whether a transparent rule based on the available training features lowers held-out RMSE relative to a fixed median/mode recipe across several tabular regression tasks and corruption settings. The software reports measured results, including cases where cleaning hurts or the rule ties a fixed strategy. A statistical outlier flag alone does not establish an error.

## Install and open the app

Python 3.11 or newer is required. In PowerShell, from this folder:

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Open the local URL printed by Streamlit, normally `http://localhost:8501`. Select the bundled student example or upload a CSV, choose a numeric target, review the quality profile, and run a cleaning comparison. For the early student-grade demonstration, the app excludes later-period grades `G1` and `G2` and `absences` by default. For another CSV, review the feature availability before running an experiment.

## Reproduce the study

The three committed demonstration tables and their official UCI sources, licenses, transformations, and hashes are described in [data/README.md](data/README.md). Run each study separately:

```powershell
.\.venv\Scripts\python.exe experiments\run_study.py data\student-por.csv --target G3 --drop G1 G2 absences --output results\student.csv
.\.venv\Scripts\python.exe experiments\run_study.py data\winequality-red.csv --target quality --output results\wine.csv
.\.venv\Scripts\python.exe experiments\run_study.py data\energy-heating.csv --target heating_load --output results\energy.csv
```

The default research grid uses five fixed split seeds; 0%, 5%, 15%, and 30% corruption; random and feature-dependent missingness plus numeric extremes; four cleaning strategies; and three models. Each dataset produces 720 metric rows. For a short installation check, add `--quick`; its output is a smoke test, not the full paper benchmark. The runner also accepts `--seeds`, `--mechanisms`, `--levels`, `--strategies`, and `--models` for separate batches. Keep each batch in a distinct CSV. A JSON manifest beside each CSV records the dataset hash and configuration.

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

## Experimental boundary

Identical feature rows are grouped before each 80/20 split. Corruption is injected only into training features, never the target or held-out rows. All strategies receive the same split and corruption mask for a given condition. Imputation, encoding, scaling, clipping bounds, the training-guided rule, and models are fit only on the training partition. The held-out rows are used to calculate RMSE, MAE, and R². Run records include the split and mask hashes, affected-cell count, selected recipe, runtime, and scores.

The `median` arm is the uncorrupted reference at 0% and the minimal model-required repair at positive missingness. A literally unprocessed missing-value table cannot be given to estimators that reject missing values. `knn` substitutes numeric KNN imputation; `iqr` adds train-fitted clipping; `adaptive` chooses among these simple recipes from training features by a fixed heuristic. The rule is a measured baseline, not an assumed improvement.

## Scope and limitations

The research runner currently handles **numeric regression targets**. The quality profile accepts mixed CSV feature types. Missingness and numeric extremes are controlled synthetic interventions, so their conclusions do not automatically transfer to naturally erroneous data. Potential outliers may be valid, and exact duplicate rows may represent legitimate observations. The three datasets have different target scales; interpret RMSE within each dataset and use paired differences for comparisons. The study does not claim a generally optimal cleaning algorithm.

The [paper workspace](paper/README.md) contains the 10-page IEEE manuscript, its BibTeX library, measured results, and vector figures. The complete three-dataset study has run: 720 fitted arms per dataset, 2,160 total. The frozen training-guided policy has lower RMSE than fixed median/mode in 31 of 81 positive-corruption condition means, ties in 36, and has higher RMSE in 14. Interpret this as a bounded empirical result, not a universal benefit. Optional duplicate-removal and text-normalization arms are available in the app for exploration but are not included in the four-arm paper benchmark.
