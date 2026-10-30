"""Focused checks for reproducibility, pairing, and leakage boundaries."""

import unittest

import numpy as np
import pandas as pd

from datasentry.corruption import corrupt_training_features
from datasentry.experiment import (ExperimentConfig, make_group_split,
                                  normalize_feature_missing, normalize_categorical_values,
                                  deduplicate_training_rows, run_experiment)
from datasentry.preprocessing import (CategoricalModeImputer, IQRClipper,
                                      build_preprocessor, select_strategy)


class ExperimentEngineTests(unittest.TestCase):
    def setUp(self):
        self.frame = pd.DataFrame({
            "numeric": np.arange(40, dtype=float),
            "category": ["a", "b", "c", "d"] * 10,
            "target": np.arange(40, dtype=float) * 0.5,
        })

    def test_identical_feature_rows_stay_in_one_partition(self):
        x = self.frame.drop(columns="target").copy()
        x.loc[39] = x.loc[0]
        train, test = make_group_split(x, self.frame.target, 11, 0.2)
        self.assertFalse(set(train).intersection(test))
        self.assertEqual(0 in train, 39 in train)

    def test_whitespace_only_categories_are_missing_before_group_split(self):
        frame = self.frame.copy(deep=True)
        frame.loc[0, "category"] = "   "
        frame.loc[39, ["numeric", "category"]] = [frame.loc[0, "numeric"], np.nan]
        normalized = normalize_feature_missing(frame.drop(columns="target"))
        self.assertTrue(pd.isna(normalized.loc[0, "category"]))
        self.assertTrue(pd.isna(normalized.loc[39, "category"]))
        train, test = make_group_split(normalized, frame.target, 11, 0.2)
        self.assertEqual(0 in train, 39 in train)
        self.assertFalse(pd.isna(frame.loc[0, "category"]))
        config = ExperimentConfig(seeds=(11,), levels=(0.0,),
                                  mechanisms=("mcar",), strategies=("median",),
                                  models=("ridge",))
        result = run_experiment(frame, "target", config)
        self.assertEqual(len(result), 1)
        self.assertTrue(np.isfinite(result.rmse).all())

    def test_corruption_is_repeatable_and_does_not_mutate_source(self):
        x = self.frame.drop(columns="target")
        before = x.copy(deep=True)
        a = corrupt_training_features(x, "mcar", 0.25, 123)
        b = corrupt_training_features(x, "mcar", 0.25, 123)
        pd.testing.assert_frame_equal(x, before)
        pd.testing.assert_frame_equal(a.frame, b.frame)
        np.testing.assert_array_equal(a.mask, b.mask)
        self.assertEqual(a.affected_cells, 20)

    def test_mar_keeps_its_driver_observed(self):
        x = self.frame.drop(columns="target")
        result = corrupt_training_features(x, "mar", 0.3, 42)
        self.assertFalse(result.mask[:, 0].any())
        self.assertGreater(result.affected_cells, 0)

    def test_clipping_threshold_comes_from_fit_data(self):
        clip = IQRClipper().fit(np.array([[0.], [1.], [2.], [3.]]))
        predicted = clip.transform(np.array([[100.]]))
        self.assertAlmostEqual(float(predicted[0, 0]), 4.5)

    def test_categorical_mode_comes_from_training_and_empty_column_is_safe(self):
        train = pd.DataFrame({"color": ["red", "red", "blue", None],
                              "empty": [None, None, None, None]})
        held_out = pd.DataFrame({"color": [None, "green"],
                                 "empty": [None, None]})
        imputer = CategoricalModeImputer().fit(train)
        self.assertEqual(imputer.transform(held_out).tolist(),
                         [["red", "<MISSING>"], ["green", "<MISSING>"]])
        prep = build_preprocessor(train, "median").fit(train)
        transformed = prep.transform(held_out)
        self.assertEqual(transformed.shape[0], 2)
        self.assertTrue(np.isfinite(transformed).all())

    def test_static_categorical_normalization(self):
        source = pd.DataFrame({"category": [" Ａ\t B  ", "a  b", None],
                               "amount": [1, 2, 3]})
        result = normalize_categorical_values(source)
        self.assertEqual(result.category.tolist()[:2], ["a b", "a b"])
        self.assertTrue(pd.isna(result.category.iloc[2]))
        self.assertEqual(result.amount.tolist(), [1, 2, 3])
        self.assertEqual(source.category.iloc[0], " Ａ\t B  ")

    def test_dedup_uses_features_and_target_and_only_training_rows(self):
        features = pd.DataFrame({"x": [1, 1, 1, 2], "c": ["a", "a", "a", "b"]})
        target = pd.Series([3.0, 3.0, 4.0, 5.0])
        reduced_x, reduced_y, removed = deduplicate_training_rows(features, target)
        self.assertEqual(removed, 1)
        self.assertEqual(len(reduced_x), 3)
        self.assertEqual(reduced_y.tolist(), [3.0, 4.0, 5.0])
        self.assertEqual(len(features), 4)

    def test_exploratory_arms_reuse_split_and_report_dedup_count(self):
        source = pd.DataFrame({"x": np.repeat(np.arange(20), 2),
                               "c": [" Ａ ", " Ａ "] * 20,
                               "target": np.repeat(np.arange(20), 2).astype(float)})
        config = ExperimentConfig(seeds=(11,), levels=(0.0,),
                                  mechanisms=("mcar",),
                                  strategies=("median", "dedup", "normalize"),
                                  models=("ridge",))
        result = run_experiment(source, "target", config)
        self.assertEqual(result.split_sha256.nunique(), 1)
        self.assertEqual(result.mask_sha256.nunique(), 1)
        self.assertGreater(result.loc[result.strategy.eq("dedup"), "dedup_rows_removed"].iloc[0], 0)
        self.assertTrue(result.loc[result.strategy.eq("normalize"), "preprocessing_recipe"].eq("median").all())

    def test_policy_reads_training_missingness(self):
        frame = pd.DataFrame({"x": np.arange(500, dtype=float),
                              "z": np.arange(500, dtype=float)})
        frame.loc[:149, "x"] = np.nan
        self.assertEqual(select_strategy(frame), "knn")

    def test_result_rows_share_split_and_mask_across_arms(self):
        before = self.frame.copy(deep=True)
        config = ExperimentConfig(seeds=(11,), levels=(0.0, 0.2),
                                  mechanisms=("mcar",), strategies=("median", "iqr"),
                                  models=("ridge",), drop_columns=())
        result = run_experiment(self.frame, "target", config)
        pd.testing.assert_frame_equal(self.frame, before)
        self.assertEqual(len(result), 4)
        self.assertEqual(result.split_sha256.nunique(), 1)
        self.assertTrue(result.groupby("level").mask_sha256.nunique().eq(1).all())
        self.assertEqual(result.loc[result.level.eq(0), "affected_cells"].iloc[0], 0)
        self.assertTrue(np.isfinite(result.rmse).all())


if __name__ == "__main__":
    unittest.main()
