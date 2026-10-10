"""
Unit Tests for Re-identification Risk Analysis.
"""

import unittest
import pandas as pd
import numpy as np
from src.risk_analysis import (
    get_equivalence_classes,
    calculate_risk_metrics,
    annotate_dataset_with_risk
)
from src.data_generator import generate_student_dataset


class TestRiskAnalysis(unittest.TestCase):

    def setUp(self):
        # Create a deterministic mock dataset with known equivalence classes
        # EC 1: (20, 'F', 'CSE') -> 1 row (Singleton, size 1)
        # EC 2: (21, 'M', 'ECE') -> 2 rows (size 2)
        # EC 3: (22, 'F', 'ME')  -> 3 rows (size 3)
        # EC 4: (23, 'M', 'CSE') -> 4 rows (size 4)
        # Total rows = 1 + 2 + 3 + 4 = 10 rows
        data = [
            # EC 1
            {"Age": 20, "Gender": "F", "Dept": "CSE", "Score": 8.5},
            # EC 2
            {"Age": 21, "Gender": "M", "Dept": "ECE", "Score": 7.0},
            {"Age": 21, "Gender": "M", "Dept": "ECE", "Score": 7.5},
            # EC 3
            {"Age": 22, "Gender": "F", "Dept": "ME", "Score": 6.5},
            {"Age": 22, "Gender": "F", "Dept": "ME", "Score": 6.8},
            {"Age": 22, "Gender": "F", "Dept": "ME", "Score": 7.2},
            # EC 4
            {"Age": 23, "Gender": "M", "Dept": "CSE", "Score": 9.1},
            {"Age": 23, "Gender": "M", "Dept": "CSE", "Score": 9.0},
            {"Age": 23, "Gender": "M", "Dept": "CSE", "Score": 8.9},
            {"Age": 23, "Gender": "M", "Dept": "CSE", "Score": 9.4},
        ]
        self.df_mock = pd.DataFrame(data)
        self.qis = ["Age", "Gender", "Dept"]

    def test_equivalence_class_computation(self):
        ec = get_equivalence_classes(self.df_mock, self.qis)
        self.assertEqual(len(ec), 4)  # 4 equivalence classes
        self.assertEqual(sorted(ec.values), [1, 2, 3, 4])

    def test_risk_metrics_k3(self):
        # With k = 3:
        # Classes of size < 3 are: size 1 (1 record) and size 2 (2 records) -> total 3 violating records
        report = calculate_risk_metrics(self.df_mock, self.qis, k=3)

        self.assertEqual(report.total_records, 10)
        self.assertEqual(report.num_equivalence_classes, 4)
        self.assertEqual(report.num_singletons, 1)
        self.assertEqual(report.num_violating_records, 3)
        self.assertEqual(report.min_equivalence_class_size, 1)
        self.assertEqual(report.max_equivalence_class_size, 4)
        self.assertAlmostEqual(report.avg_equivalence_class_size, 10 / 4, places=2)
        self.assertAlmostEqual(report.pct_violating_records, 30.0, places=1)
        self.assertAlmostEqual(report.max_reidentification_risk, 1.0, places=4)  # 1 / 1
        self.assertAlmostEqual(report.avg_reidentification_risk, 4 / 10, places=4)  # 4 ECs / 10 records = 0.4

    def test_risk_metrics_k2(self):
        # With k = 2:
        # Only singleton (size 1) violates k=2 -> 1 record
        report = calculate_risk_metrics(self.df_mock, self.qis, k=2)
        self.assertEqual(report.num_violating_records, 1)
        self.assertAlmostEqual(report.pct_violating_records, 10.0, places=1)

    def test_risk_metrics_k5(self):
        # With k = 5:
        # All classes (1, 2, 3, 4) are < 5 -> all 10 records violate k=5
        report = calculate_risk_metrics(self.df_mock, self.qis, k=5)
        self.assertEqual(report.num_violating_records, 10)
        self.assertAlmostEqual(report.pct_violating_records, 100.0, places=1)

    def test_annotated_dataset(self):
        annotated = annotate_dataset_with_risk(self.df_mock, self.qis)
        self.assertIn("EC_Size", annotated.columns)
        self.assertIn("Individual_Risk", annotated.columns)

        # Check the singleton row (row 0)
        self.assertEqual(annotated.iloc[0]["EC_Size"], 1)
        self.assertEqual(annotated.iloc[0]["Individual_Risk"], 1.0)

        # Check an EC 4 row (row 6)
        self.assertEqual(annotated.iloc[6]["EC_Size"], 4)
        self.assertAlmostEqual(annotated.iloc[6]["Individual_Risk"], 0.25)

    def test_on_synthetic_student_dataset(self):
        df_students = generate_student_dataset(n_records=500, random_seed=42)
        qis = ["Age", "Gender", "Department", "Year", "City"]
        report = calculate_risk_metrics(df_students, qis, k=5)

        self.assertEqual(report.total_records, 500)
        self.assertGreater(report.num_equivalence_classes, 0)
        self.assertGreaterEqual(report.num_singletons, 0)
        self.assertGreaterEqual(report.max_reidentification_risk, 0.0)
        self.assertLessEqual(report.max_reidentification_risk, 1.0)


if __name__ == "__main__":
    unittest.main()
