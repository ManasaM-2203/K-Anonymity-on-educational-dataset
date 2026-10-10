"""
Unit Tests for Multi-K Evaluator and Dynamic Recommendation Engine.
"""

import unittest
import pandas as pd
import numpy as np

from src.k_evaluator import evaluate_k_range, KEvaluationSummary
from src.data_generator import generate_student_dataset
from src.k_anonymity import apply_k_anonymity


class TestKEvaluator(unittest.TestCase):

    def setUp(self):
        self.df = generate_student_dataset(n_records=400, random_seed=42)
        self.direct_ids = ["Student_ID", "Name"]
        self.qis = ["Age", "Gender", "Department", "City"]

    def test_multi_k_evaluation_range(self):
        k_min = 2
        k_max = 6
        summary = evaluate_k_range(
            df=self.df,
            quasi_identifiers=self.qis,
            direct_identifiers=self.direct_ids,
            k_min=k_min,
            k_max=k_max,
            generalization_level=1,
            suppress_outliers=True,
            privacy_weight=0.5,
            utility_weight=0.5
        )

        self.assertIsInstance(summary, KEvaluationSummary)
        self.assertEqual(len(summary.results_by_k), k_max - k_min + 1)
        self.assertEqual(len(summary.comparison_df), k_max - k_min + 1)
        for k in range(k_min, k_max + 1):
            self.assertIn(k, summary.results_by_k)
            self.assertTrue(summary.results_by_k[k].is_k_satisfied)

    def test_automatic_k_recommendation_logic(self):
        summary = evaluate_k_range(
            df=self.df,
            quasi_identifiers=self.qis,
            direct_identifiers=self.direct_ids,
            k_min=2,
            k_max=8,
            privacy_weight=0.5,
            utility_weight=0.5
        )

        self.assertGreaterEqual(summary.recommended_k, 2)
        self.assertLessEqual(summary.recommended_k, 8)
        self.assertIn("Balance Score", summary.recommendation_reason)

        # Check recommended K has the highest or tied balance score among valid Ks
        rec_m = summary.results_by_k[summary.recommended_k]
        for k, m in summary.results_by_k.items():
            if m.is_k_satisfied:
                self.assertGreaterEqual(rec_m.balance_score, m.balance_score - 1e-6)

    def test_weight_tuning_influences_recommendation(self):
        # With extreme privacy weight (1.0 privacy, 0.0 utility), higher K should be favored
        summary_priv = evaluate_k_range(
            df=self.df,
            quasi_identifiers=self.qis,
            direct_identifiers=self.direct_ids,
            k_min=2,
            k_max=8,
            privacy_weight=1.0,
            utility_weight=0.0
        )

        # With extreme utility weight (0.0 privacy, 1.0 utility), lower K (with higher retention/less loss) should be favored
        summary_util = evaluate_k_range(
            df=self.df,
            quasi_identifiers=self.qis,
            direct_identifiers=self.direct_ids,
            k_min=2,
            k_max=8,
            privacy_weight=0.0,
            utility_weight=1.0
        )

        self.assertGreaterEqual(summary_priv.recommended_k, summary_util.recommended_k)

    def test_invalid_k_values(self):
        # k_min < 2 should be clamped to 2
        summary = evaluate_k_range(
            df=self.df,
            quasi_identifiers=self.qis,
            k_min=0,
            k_max=4
        )
        self.assertEqual(summary.k_min, 2)

        # k_max < k_min should be adjusted
        summary2 = evaluate_k_range(
            df=self.df,
            quasi_identifiers=self.qis,
            k_min=5,
            k_max=2
        )
        self.assertEqual(summary2.k_max, 5)

    def test_k_greater_than_dataset_size(self):
        small_df = pd.DataFrame([
            {"Age": 20, "Gender": "F", "City": "Bengaluru"},
            {"Age": 20, "Gender": "F", "City": "Bengaluru"}
        ])
        summary = evaluate_k_range(
            df=small_df,
            quasi_identifiers=["Age", "Gender", "City"],
            k_min=5,
            k_max=6,
            suppress_outliers=True
        )
        # Should handle gracefully without crashing
        self.assertIsInstance(summary, KEvaluationSummary)

    def test_empty_dataset(self):
        empty_df = pd.DataFrame(columns=["Age", "Gender", "City"])
        summary = evaluate_k_range(
            df=empty_df,
            quasi_identifiers=["Age", "Gender", "City"],
            k_min=2,
            k_max=5
        )
        self.assertEqual(len(summary.results_by_k), 0)
        self.assertEqual(summary.recommended_k, 2)


if __name__ == "__main__":
    unittest.main()
