"""Focused tests for descriptive quality checks on mixed CSV-like data."""

import json
import unittest

import pandas as pd
from pandas.testing import assert_frame_equal

from datasentry.quality import profile_dataframe


class QualityProfileTests(unittest.TestCase):
    def test_mixed_quality_flags_are_counts_not_mutations(self):
        frame = pd.DataFrame(
            {
                "amount": [1, 2, 3, 4, 5, 6, 100],
                "label": ["yes", " YES ", "no", "NO", " ", "yes", None],
            }
        )
        original = frame.copy(deep=True)
        report = profile_dataframe(frame)
        assert_frame_equal(frame, original)
        self.assertEqual(report["overview"]["rows"], 7)
        self.assertEqual(report["overview"]["missing_cells"], 2)
        self.assertEqual(report["overview"]["duplicate_rows"], 0)
        self.assertEqual(report["outliers"][0]["column"], "amount")
        self.assertEqual(report["outliers"][0]["count"], 1)
        variants = report["inconsistent_patterns"]
        self.assertEqual({item["canonical"] for item in variants}, {"yes", "no"})
        self.assertEqual({item["affected_rows"] for item in variants}, {1})
        json.dumps(report, allow_nan=False)

    def test_duplicate_rows_and_target_annotation(self):
        frame = pd.DataFrame({"x": [1, 1, 2], "target": [8, 8, 9]})
        report = profile_dataframe(frame, target="target")
        self.assertEqual(report["overview"]["duplicate_rows"], 1)
        self.assertTrue(report["columns"][1]["is_target"])

    def test_empty_data_and_invalid_target(self):
        report = profile_dataframe(pd.DataFrame({"x": pd.Series(dtype=float)}))
        self.assertEqual(report["overview"]["missing_pct"], 0.0)
        with self.assertRaises(ValueError):
            profile_dataframe(pd.DataFrame({"x": [1]}), target="absent")

    def test_constant_numeric_does_not_produce_spurious_iqr_flag(self):
        report = profile_dataframe(pd.DataFrame({"x": [3, 3, 3, 3, 3]}))
        self.assertEqual(report["outliers"], [])


if __name__ == "__main__":
    unittest.main()
