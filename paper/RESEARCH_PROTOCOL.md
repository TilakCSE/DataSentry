# DataSentry AI research protocol (working draft)

## Paper-first decision

The first deliverable is an IEEE-format research paper. A small reproducible
experiment must run before the results section is finalized. The full
interactive application follows the paper study. No result will be invented.

## Research question

For tabular regression under controlled missing-value and numeric feature
corruption, does a simple cleaning policy based only on training data
characteristics reduce held-out prediction error compared with fixed cleaning
recipes?

The study is an empirical comparison of an interpretable policy. It does not
claim a new general data-cleaning algorithm.

## Primary demonstration

Use UCI Student Performance, Portuguese course, predicting final grade G3.
Exclude G1 and G2 from the first prediction task because these are grades from
later periods and strongly predict G3. Exclude absences in the early-prediction
version unless the prediction point is explicitly moved later. Retain a fixed
feature-availability table. Use other UCI tabular regression datasets to assess
whether observations extend beyond one small educational dataset.

## Hypotheses

* H1: Higher training corruption generally increases held-out RMSE, but the
  size and direction of the effect vary by dataset and model.
* H2: A frozen, training-only policy can improve paired held-out RMSE over a
  prespecified median/mode recipe in some conditions; it is not assumed to win
  everywhere.
* H3: Outlier clipping can harm performance when flagged extremes represent
  valid data.

## Experimental arms

* Uncorrupted reference with minimal model-required preprocessing.
* Corrupted training data with fixed median/mode imputation and no clipping.
* Corrupted training data with fixed KNN numeric imputation and no clipping.
* Corrupted training data with fixed median/mode imputation and IQR clipping.
* Corrupted training data with a transparent characteristic-based policy
  selecting among the same operations. Freeze its decision rules before final
  held-out evaluation.

Compare Ridge regression and Random Forest regression with a small multilayer
perceptron as the neural-network comparison required by the course's technical
rigor. Keep model settings fixed or tune only within training folds.

## Corruption and evaluation

Inject missing numeric/categorical feature cells and numeric extremes at 0%,
5%, 15%, and 30% of eligible cells. Include random missingness and a
feature-dependent mechanism. Do not corrupt the target. Generate identical
corruption masks for all competing cleaning arms. Treat synthetic affected
cells as known errors; do not label naturally extreme observations as errors
without domain evidence. Exact duplicates and categorical representation
inconsistencies are secondary experiments after the primary comparison.

Create five seeded 80/20 splits per dataset. Group identical feature rows
before splitting so repeated records cannot cross the train/test boundary.
Inject the primary corruption into training data only. Keep the same untouched
test rows for all arms in a split. Fit encoders, imputers, clipping thresholds,
scalers, selection rules, and estimators only inside each training fold using
scikit-learn pipelines. Record dataset and code versions, seeds, split IDs,
corruption masks, selected actions, and run times.

Primary metric: held-out RMSE. Secondary metrics: MAE, R-squared, paired RMSE
change relative to the fixed baseline, runtime, and affected-value count.
Report each dataset separately and summarize paired differences across
datasets. Do not treat repeated seeds as independent datasets. Add a separate
deployment stress test with corrupted held-out features if time permits; label
it separately from the primary training-corruption study.

## Paper structure and citation rules

Target at least 10 IEEE-formatted pages and at least 30 distinct cited papers.
Use the attached paper outline as the section order: Abstract; Introduction;
Literature Review; Research Direction; Proposed Model; Model Implementation;
Results and Discussion; Conclusion; References. The typed instructions and
screenshot conflict on exact section lengths (notably the abstract and
references), so aim for the typed minimums of a half-page abstract, 1.5-page
introduction, 3-page review, 2-page experiment setup, 1-page results, and
2-page references while retaining the screenshot's detailed section headings.
The full paper may therefore exceed 10 pages. Add an explicit experiment-setup
subsection spanning the proposed-model and implementation sections.
Use IEEE numbered citations and Mendeley-managed references. Cite at least 10
papers in the introduction, at least 20 in the literature review, and around
five in the proposed-model discussion; references must correspond to actual
claims. Do not cite sources in the abstract. Every figure/table needs a number,
caption, and a reference in surrounding text. The final results section must
be written from recorded experiment output only.

## Course-project implications

The PDF rubric calls for EDA, preprocessing, feature engineering, baseline
comparison, a neural-network comparison, reproducibility, a GitHub repository,
and a live demo. The dates shown are 30 October for model/results work and
2 November for documentation/presentation. This creates a short initial phase:
paper protocol and limited measured experiments first, then the fuller app and
demo. The four listed team members match the maximum group size.

## Team responsibilities (proposed)

* Tilaksinh Chauhan (23000740): research protocol, experiment integration,
  leakage review, paper assembly.
* Kavya Jariwala (23000543): dataset provenance, EDA, quality profiling,
  feature-availability table.
* Bhargav Panchal (23000502): baseline and neural models, inner-fold tuning,
  reproducible training code.
* Om Nathani (23000883): corruption generators, metrics, paired comparisons,
  paper figures and reference audit.

## Initial primary sources

* Li et al., CleanML, ICDE 2021, DOI: 10.1109/ICDE51399.2021.00009.
* Neutatz et al., Data Cleaning and AutoML, Datenbank-Spektrum 2022,
  DOI: 10.1007/s13222-022-00413-2.
* Jager and Biessmann, Automated Cleaning of Tabular Data, AISTATS 2024,
  https://proceedings.mlr.press/v238/jager24a.html.
* Mohammed et al., Effects of Data Quality on ML Performance on Tabular Data,
  Information Systems 2025, DOI: 10.1016/j.is.2025.102549.
* Jarrett et al., HyperImpute, ICML 2022,
  https://proceedings.mlr.press/v162/jarrett22a.html.
* UCI Student Performance dataset, DOI: 10.24432/C5TG7T.
* scikit-learn, Common pitfalls and recommended practices,
  https://scikit-learn.org/stable/common_pitfalls.
