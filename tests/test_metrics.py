"""
Unit Tests for Information Loss and Data Utility Metrics.
"""

import unittest
import pandas as pd
import numpy as np

from src.metrics import (
    calculate_generalization_loss,
    calculate_information_loss,
    calculate_utility_score,
    calculate_privacy_score,
    compute_metrics_bundle,
    UtilityMetrics
)
from src.data_generator import generate_student_dataset
from src.k_anonymity import apply_k_anonymity


class TestMetrics(unittest.TestCase):

    def setUp(self):
        self.raw_df = generate_student_dataset(n_records=100, random_seed=42)
        self.qis = ["Age", "Gender", "Department", "City"]

    def test_zero_suppression_and_no_generalization(self):
        # When dataset is unmodified, gen_loss is 0, supp_loss is 0, total_il is 0
        gen_loss = calculate_generalization_loss(self.raw_df, self.raw_df, self.qis, generalization_level=1)
        self.assertEqual(gen_loss, 0.0)

        supp_loss, ret_rate, total_il = calculate_information_loss(100, 100, 0, gen_loss)
        self.assertEqual(supp_loss, 0.0)
        self.assertEqual(ret_rate, 1.0)
        self.assertEqual(total_il, 0.0)

        utility = calculate_utility_score(total_il)
        self.assertEqual(utility, 1.0)

    def test_total_suppression(self):
        # When all records are suppressed, total_il must be 1.0, utility 0.0
        supp_loss, ret_rate, total_il = calculate_information_loss(100, 0, 100, 0.5)
        self.assertEqual(supp_loss, 1.0)
        self.assertEqual(ret_rate, 0.0)
        self.assertEqual(total_il, 1.0)

        utility = calculate_utility_score(total_il)
        self.assertEqual(utility, 0.0)

    def test_privacy_score_bounds(self):
        # Privacy score = 1 - 1/k
        self.assertAlmostEqual(calculate_privacy_score(2, True), 0.5)
        self.assertAlmostEqual(calculate_privacy_score(5, True), 0.8)
        self.assertAlmostEqual(calculate_privacy_score(10, True), 0.9)
        self.assertEqual(calculate_privacy_score(0, False), 0.0)

        # Violation penalty
        self.assertAlmostEqual(calculate_privacy_score(5, False), 0.4)

    def test_metrics_bundle_computation(self):
        anon_res = apply_k_anonymity(
            df=self.raw_df,
            quasi_identifiers=self.qis,
            direct_identifiers=["Student_ID", "Name"],
            k=3,
            generalization_level=1,
            suppress_outliers=True
        )

        bundle = compute_metrics_bundle(
            raw_df=self.raw_df,
            anonymized_df=anon_res.anonymized_df,
            quasi_identifiers=self.qis,
            k_requested=3,
            k_achieved=anon_res.final_min_ec_size,
            is_k_satisfied=anon_res.is_k_satisfied,
            initial_records=anon_res.initial_records_count,
            retained_records=anon_res.final_records_count,
            suppressed_records=anon_res.num_suppressed_records,
            generalization_level=1,
            privacy_weight=0.6,
            utility_weight=0.4
        )

        self.assertIsInstance(bundle, UtilityMetrics)
        self.assertTrue(0.0 <= bundle.total_information_loss <= 1.0)
        self.assertTrue(0.0 <= bundle.data_utility_score <= 1.0)
        self.assertTrue(0.0 <= bundle.privacy_score <= 1.0)
        self.assertTrue(0.0 <= bundle.balance_score <= 1.0)
        self.assertEqual(bundle.k_requested, 3)
        self.assertTrue(bundle.is_k_satisfied)


if __name__ == "__main__":
    unittest.main()
