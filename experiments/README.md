# Student Performance missingness pilot

This is a **pilot for method checking**, not a final paper result. It uses one
dataset, five grouped train/test splits, two classical regression models, and
MCAR training-feature missingness only. It does not test outliers, MAR
missingness, categorical inconsistencies, duplicate injection, or a neural
model. The adaptive rule is provisional and should not be promoted as a
validated contribution.

## Data provenance

The dataset is the 649-row Portuguese-course Student Performance table from
the UCI repository (DOI 10.24432/C5TG7T). The CSV used here is a comma-separated
GitHub mirror of the UCI table:
https://github.com/shauryauppal/student-performance-uci-dataset/blob/main/student-por.csv

The script asserts 649 rows and 33 columns. The file SHA-256 is
`5be7b02f13f5800fddd66e2c8d834cd00ece14c552fb404def2397b31b63236b`.
Before citing numeric results in the paper, verify this mirror against the
original UCI file and record any conversion differences.

## Run

```powershell
python experiments/pilot_student.py data/student-por.csv --output experiments/pilot_student_results.csv
```

The script excludes G1, G2, and absences for an early prediction of G3. It
groups exact duplicate feature rows before each split, corrupts training
features only, fits preprocessing on training rows, and evaluates every arm on
the same untouched test rows. It records 120 runs (5 seeds x 4 levels x 3
recipes x 2 models).

## Initial observation

The adaptive pilot rule did not show a meaningful improvement over the fixed
median/mode recipe. Across five seeds at 15% random missingness, its mean RMSE
difference was +0.0005 for the forest and -0.0098 for Ridge; at 0%, 5%, and
30%, it chose the median/mode recipe and therefore tied it. This is too narrow
to support a general claim. The paper's final comparison needs multiple
datasets, missingness mechanisms, outlier conditions, a neural model, and
uncertainty analysis.
