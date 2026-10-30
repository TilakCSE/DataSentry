# Study datasets and provenance

The committed CSVs permit offline reproduction. `python data/fetch_uci.py`
downloads the authoritative UCI archives, verifies the source-file SHA-256
checksums below, checks the existing student CSV against the source by parsed
values, and regenerates the two additional CSVs. The conversion uses pandas;
CSV byte hashes can differ by operating-system line endings even when parsed
values are identical.

| Local CSV | Rows | Predictors | Target | Official source file SHA-256 |
| --- | ---: | ---: | --- | --- |
| `student-por.csv` | 649 | 32 supplied, 29 for the early-prediction study | `G3` | `a7594a11d7771c0efe1a740824e0e833da9c4cad07c39a9766a874575563fb3f` |
| `winequality-red.csv` | 1,599 | 11 | `quality` | `4a402cf041b025d4566d954c3b9ba8635a3a8a01e039005d97d6a710278cf05e` |
| `energy-heating.csv` | 768 | 8 | `heating_load` | `0089fcffc1415e41e2ff63730cca5280efe54fc43d722dfde1b0aaa808e35dc4` |

## Student Performance, Portuguese course

Source: [UCI Student Performance, dataset 320](https://archive.ics.uci.edu/dataset/320/student+performance), DOI [10.24432/C5TG7T](https://doi.org/10.24432/C5TG7T), CC BY 4.0. The source file is `student.zip/student-por.csv` inside UCI's `student+performance.zip` archive. It uses a semicolon separator; the committed CSV uses a comma separator. On 30 October 2026, all 649 rows, 33 columns, column order and parsed values matched the official UCI file exactly. The early-prediction study drops `G1` and `G2` (later-period grades strongly associated with `G3`) and `absences` (availability depends on prediction time). This exclusion is a study design decision, not a change to the source CSV.

## Red Wine Quality

Source: [UCI Wine Quality, dataset 186](https://archive.ics.uci.edu/dataset/186/wine+quality), DOI [10.24432/C56S3T](https://doi.org/10.24432/C56S3T), CC BY 4.0. The source is `winequality-red.csv` in `wine+quality.zip`; the committed copy changes only the delimiter from semicolon to comma. The 11 physicochemical measurements predict the `quality` rating. That rating is ordinal; treating it as numeric regression is a deliberate evaluation choice, and RMSE must be interpreted on its rating scale.

## Energy Efficiency

Source: [UCI Energy Efficiency, dataset 242](https://archive.ics.uci.edu/dataset/242/energy+efficiency), DOI [10.24432/C51307](https://doi.org/10.24432/C51307), CC BY 4.0. The source is `ENB2012_data.xlsx` in `energy+efficiency.zip`. The committed CSV retains `X1`–`X8`, renames `Y1` to `heating_load`, and removes the second response `Y2` (`cooling_load`) so it cannot become a predictor. These samples describe simulated buildings. The 768 rows are a designed parameter grid; repeated or unusual feature values should not automatically be called bad data.

## Evaluation cautions

All three datasets are used as distinct regression tasks with the same corruption and model protocol. Results must be reported separately because target scales and domains differ. Any combined comparison should use paired within-dataset differences or standardized changes, not average raw RMSE across targets. The UCI licenses allow redistribution with attribution. Source checksums above refer to the original UCI files; the repository's normalized CSVs are derived copies.
