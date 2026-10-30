"""Focused checks for reproducibility, pairing, and leakage boundaries."""

import unittest

import numpy as np
import pandas as pd

from datasentry.corruption import corrupt_training_features
from datasentry.experiment import ExperimentConfig, make_group_split, run_experiment
from datasentry.preprocessing import IQRClipper, select_strategy


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
