"""
Re-identification Risk Analysis Engine for K-Anonymity.

This module computes Equivalence Classes (ECs) and derives mathematical
re-identification risk metrics from arbitrary sets of Quasi-Identifiers.
All calculations are dynamic and deterministic.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
import pandas as pd
import numpy as np


@dataclass
class RiskReport:
    """Dataclass holding comprehensive re-identification risk metrics."""
    total_records: int
    num_equivalence_classes: int
    num_singletons: int
    num_violating_records: int
    min_equivalence_class_size: int
    max_equivalence_class_size: int
    avg_equivalence_class_size: float
    pct_violating_records: float
    max_reidentification_risk: float
    avg_reidentification_risk: float
    k_value: int
    quasi_identifiers: List[str]
    size_distribution: Dict[int, int] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to a standard serializable dictionary."""
        return {
            "total_records": self.total_records,
            "num_equivalence_classes": self.num_equivalence_classes,
            "num_singletons": self.num_singletons,
            "num_violating_records": self.num_violating_records,
            "min_equivalence_class_size": self.min_equivalence_class_size,
            "max_equivalence_class_size": self.max_equivalence_class_size,
            "avg_equivalence_class_size": round(self.avg_equivalence_class_size, 2),
            "pct_violating_records": round(self.pct_violating_records, 2),
            "max_reidentification_risk": round(self.max_reidentification_risk, 4),
            "avg_reidentification_risk": round(self.avg_reidentification_risk, 4),
            "k_value": self.k_value,
            "quasi_identifiers": self.quasi_identifiers,
            "size_distribution": self.size_distribution
        }


def get_equivalence_classes(df: pd.DataFrame, quasi_identifiers: List[str]) -> pd.Series:
    """
    Computes the size of each equivalence class defined by the given quasi-identifiers.

    Parameters:
        df (pd.DataFrame): Dataset to analyze.
        quasi_identifiers (List[str]): Columns forming the quasi-identifier tuple.

    Returns:
        pd.Series: Indexed by the unique QI tuples, values represent class size (|EC|).
    """
    if df.empty:
        return pd.Series(dtype=int)

    missing_cols = [col for col in quasi_identifiers if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Quasi-identifier columns not found in dataset: {missing_cols}")

    if not quasi_identifiers:
        # If no QIs selected, entire dataset forms one equivalence class
        return pd.Series([len(df)], index=["All Records"])

    # Group by quasi-identifiers to get counts of each equivalence class
    ec_sizes = df.groupby(quasi_identifiers, observed=False, dropna=False).size()
    return ec_sizes


def annotate_dataset_with_risk(
    df: pd.DataFrame,
    quasi_identifiers: List[str]
) -> pd.DataFrame:
    """
    Annotates each record in the dataframe with:
      - `EC_Size`: The size of the equivalence class the row belongs to.
      - `Individual_Risk`: 1 / EC_Size (re-identification probability).

    Returns:
        pd.DataFrame: A copy of df with additional risk columns.
    """
    if df.empty or not quasi_identifiers:
        annotated = df.copy()
        annotated["EC_Size"] = len(df)
        annotated["Individual_Risk"] = 1.0 / len(df) if len(df) > 0 else 0.0
        return annotated

    annotated = df.copy()
    # Compute class size per row via transform
    annotated["EC_Size"] = annotated.groupby(quasi_identifiers, observed=False, dropna=False)[quasi_identifiers[0]].transform("count")
    annotated["Individual_Risk"] = 1.0 / annotated["EC_Size"]
    return annotated


def calculate_risk_metrics(
    df: pd.DataFrame,
    quasi_identifiers: List[str],
    k: int = 5
) -> RiskReport:
    """
    Performs full re-identification risk evaluation on a dataset for a target K value.

    Parameters:
        df (pd.DataFrame): The student dataset (pre- or post-anonymization).
        quasi_identifiers (List[str]): List of quasi-identifier column names.
        k (int): Target k-anonymity threshold (default: 5).

    Returns:
        RiskReport: Complete mathematical risk summary.
    """
    if df.empty:
        return RiskReport(
            total_records=0,
            num_equivalence_classes=0,
            num_singletons=0,
            num_violating_records=0,
            min_equivalence_class_size=0,
            max_equivalence_class_size=0,
            avg_equivalence_class_size=0.0,
            pct_violating_records=0.0,
            max_reidentification_risk=0.0,
            avg_reidentification_risk=0.0,
            k_value=k,
            quasi_identifiers=quasi_identifiers,
            size_distribution={}
        )

    # Calculate Equivalence Classes
    ec_series = get_equivalence_classes(df, quasi_identifiers)
    total_records = len(df)
    num_ecs = len(ec_series)

    # 1. Minimum and Maximum class sizes
    min_ec_size = int(ec_series.min()) if num_ecs > 0 else 0
    max_ec_size = int(ec_series.max()) if num_ecs > 0 else 0

    # 2. Number of singletons (classes of size 1, where risk = 100%)
    num_singletons = int((ec_series == 1).sum())

    # 3. Violating records: Total rows belonging to ECs with size < k
    # (Each class with size c < k contributes c violating records)
    violating_classes = ec_series[ec_series < k]
    num_violating_records = int(violating_classes.sum())

    # 4. Average Equivalence Class Size
    # Formula: total_records / num_equivalence_classes
    avg_ec_size = float(total_records / num_ecs) if num_ecs > 0 else 0.0

    # 5. Percentage of records at risk (violating k-anonymity)
    pct_violating = (num_violating_records / total_records) * 100.0 if total_records > 0 else 0.0

    # 6. Maximum Re-identification Risk: 1 / min(|EC|)
    max_risk = (1.0 / min_ec_size) if min_ec_size > 0 else 0.0

    # 7. Average Re-identification Risk: (1 / N) * sum(1 / |EC_r|)
    # For an equivalence class of size c, there are c records each having risk 1/c.
    # Therefore, each EC contributes c * (1/c) = 1 to the sum of risks.
    # Total sum of risks across all N records = Number of Equivalence Classes (num_ecs).
    # Hence: Average Risk = num_ecs / total_records = 1 / avg_ec_size.
    avg_risk = float(num_ecs / total_records) if total_records > 0 else 0.0

    # 8. Class size distribution for visual plotting
    # Map EC size -> count of equivalence classes of that size
    size_dist_counts = ec_series.value_counts().sort_index().to_dict()
    size_distribution = {int(size): int(count) for size, count in size_dist_counts.items()}

    return RiskReport(
        total_records=total_records,
        num_equivalence_classes=num_ecs,
        num_singletons=num_singletons,
        num_violating_records=num_violating_records,
        min_equivalence_class_size=min_ec_size,
        max_equivalence_class_size=max_ec_size,
        avg_equivalence_class_size=avg_ec_size,
        pct_violating_records=pct_violating,
        max_reidentification_risk=max_risk,
        avg_reidentification_risk=avg_risk,
        k_value=k,
        quasi_identifiers=quasi_identifiers,
        size_distribution=size_distribution
    )
