"""
Unit Tests for K-Anonymity Transformation Engine.
"""

import unittest
import pandas as pd
import numpy as np

from src.k_anonymity import apply_k_anonymity, AnonymizationResult
from src.data_generator import generate_student_dataset
from src.risk_analysis import calculate_risk_metrics


class TestKAnonymity(unittest.TestCase):

    def setUp(self):
        self.df = generate_student_dataset(n_records=500, random_seed=42)
        self.direct_ids = ["Student_ID", "Name"]
        self.qis = ["Age", "Gender", "Department", "Year", "City"]

    def test_direct_identifiers_removed(self):
        result = apply_k_anonymity(
            df=self.df,
            quasi_identifiers=self.qis,
            direct_identifiers=self.direct_ids,
            k=3,
            generalization_level=1,
            suppress_outliers=True
        )

        for direct_id in self.direct_ids:
            self.assertNotIn(direct_id, result.anonymized_df.columns)

    def test_k_anonymity_achieved(self):
        for target_k in [2, 3, 5, 8]:
            result = apply_k_anonymity(
                df=self.df,
                quasi_identifiers=self.qis,
                direct_identifiers=self.direct_ids,
                k=target_k,
                generalization_level=1,
                suppress_outliers=True
            )

            # Check that K is strictly satisfied
            self.assertTrue(result.is_k_satisfied, f"Failed for k={target_k}")
            self.assertGreaterEqual(result.final_min_ec_size, target_k)

            # Verify with risk_analysis module independently
            post_risk = calculate_risk_metrics(
                df=result.anonymized_df,
                quasi_identifiers=self.qis,
                k=target_k
            )
            self.assertEqual(post_risk.num_violating_records, 0)
            self.assertEqual(post_risk.num_singletons, 0)
            self.assertGreaterEqual(post_risk.min_equivalence_class_size, target_k)
            self.assertLessEqual(post_risk.max_reidentification_risk, 1.0 / target_k)

    def test_age_generalization_format(self):
        result = apply_k_anonymity(
            df=self.df,
            quasi_identifiers=["Age"],
            direct_identifiers=self.direct_ids,
            k=2,
            generalization_level=1
        )
        unique_ages = set(result.anonymized_df["Age"].unique())
        # Expected formats like '18-20', '21-23', etc.
        self.assertTrue(all("-" in a or "+" in a for a in unique_ages))

    def test_suppression_accounting(self):
        result = apply_k_anonymity(
            df=self.df,
            quasi_identifiers=self.qis,
            direct_identifiers=self.direct_ids,
            k=5,
            generalization_level=1,
            suppress_outliers=True
        )

        self.assertEqual(
            result.initial_records_count,
            result.final_records_count + result.num_suppressed_records
        )
        self.assertEqual(len(result.suppressed_df), result.num_suppressed_records)


if __name__ == "__main__":
    unittest.main()
