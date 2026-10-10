"""
K-Anonymity Transformation Engine.

Applies generalization and suppression to transform educational datasets
into K-Anonymous sanitized datasets while preserving analytical utility.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
import pandas as pd
import numpy as np

from src.hierarchies import (
    generalize_age,
    generalize_cgpa,
    generalize_city,
    generalize_department,
    generalize_year
)
from src.risk_analysis import get_equivalence_classes, calculate_risk_metrics


@dataclass
class AnonymizationResult:
    """Dataclass holding the outcome of the K-Anonymity transformation."""
    anonymized_df: pd.DataFrame
    suppressed_df: pd.DataFrame
    transformation_info: List[Dict[str, str]]
    initial_records_count: int
    final_records_count: int
    num_suppressed_records: int
    num_generalized_records: int
    final_min_ec_size: int
    final_max_ec_size: int
    is_k_satisfied: bool
    k_target: int
    retention_rate_pct: float

    def summary(self) -> str:
        """Returns a concise text summary of the anonymization result."""
        status = "ACHIEVED" if self.is_k_satisfied else "NOT FULLY ACHIEVED"
        return (
            f"K-Anonymity Status: {status} (Target K={self.k_target})\n"
            f"Initial Records: {self.initial_records_count} -> Final Records: {self.final_records_count}\n"
            f"Suppressed Outliers: {self.num_suppressed_records} ({100 - self.retention_rate_pct:.1f}% data loss)\n"
            f"Final Minimum Equivalence Class Size: {self.final_min_ec_size}\n"
            f"Generalizations Applied: {len(self.transformation_info)}"
        )


def _generalize_column(
    series: pd.Series,
    col_name: str,
    level: int = 1
) -> Tuple[pd.Series, Optional[str]]:
    """
    Applies the appropriate generalization hierarchy to a column.

    Returns:
        Tuple[pd.Series, Optional[str]]: Transformed series and explanation string.
    """
    col_lower = col_name.lower()

    if "age" in col_lower:
        desc = f"Age Generalized: Transformed exact ages into {3 if level == 1 else 5}-year academic intervals."
        return series.apply(lambda x: generalize_age(x, level=level)), desc

    elif "cgpa" in col_lower or "gpa" in col_lower or "score" in col_lower:
        if pd.api.types.is_numeric_dtype(series):
            desc = "CGPA Generalized: Categorized continuous scores into academic grade bands."
            return series.apply(lambda x: generalize_cgpa(x, level=level)), desc

    elif "city" in col_lower or "town" in col_lower or "location" in col_lower:
        desc = f"City Generalized: Mapped specific municipalities to {'State Zones' if level == 1 else 'Macro Regions'}."
        return series.apply(lambda x: generalize_city(x, level=level)), desc

    elif "department" in col_lower or "dept" in col_lower or "branch" in col_lower or "major" in col_lower:
        desc = f"Department Generalized: Grouped specialized departments into {'Academic Faculties' if level == 1 else 'Broad Disciplines'}."
        return series.apply(lambda x: generalize_department(x, level=level)), desc

    elif "year" in col_lower:
        desc = "Year of Study Generalized: Grouped into Junior / Senior cohorts."
        return series.apply(lambda x: generalize_year(x, level=level)), desc

    # Generic Numeric column fallback: Equal width binning
    elif pd.api.types.is_numeric_dtype(series) and series.nunique() > 5:
        bins = min(5, series.nunique())
        try:
            binned = pd.cut(series, bins=bins, precision=1).astype(str)
            desc = f"Numeric Generalized: Continuous values binned into {bins} intervals."
            return binned, desc
        except Exception:
            return series.astype(str), None

    return series.astype(str), None


def apply_k_anonymity(
    df: pd.DataFrame,
    quasi_identifiers: List[str],
    direct_identifiers: Optional[List[str]] = None,
    k: int = 5,
    generalization_level: int = 1,
    suppress_outliers: bool = True
) -> AnonymizationResult:
    """
    Transforms a student dataset to satisfy K-Anonymity through
    Direct Identifier Redaction, Quasi-Identifier Generalization, and Outlier Suppression.

    Parameters:
        df (pd.DataFrame): Raw student dataset.
        quasi_identifiers (List[str]): Columns chosen as quasi-identifiers.
        direct_identifiers (List[str]): Columns to remove completely (e.g. Student_ID, Name).
        k (int): Target k-anonymity parameter (k >= 2).
        generalization_level (int): Depth of hierarchy (1 = moderate, 2 = deep).
        suppress_outliers (bool): Whether to suppress residual equivalence classes < k.

    Returns:
        AnonymizationResult: Complete report and anonymized dataframe.
    """
    if df.empty:
        return AnonymizationResult(
            anonymized_df=df.copy(),
            suppressed_df=pd.DataFrame(),
            transformation_info=[],
            initial_records_count=0,
            final_records_count=0,
            num_suppressed_records=0,
            num_generalized_records=0,
            final_min_ec_size=0,
            final_max_ec_size=0,
            is_k_satisfied=False,
            k_target=k,
            retention_rate_pct=0.0
        )

    working_df = df.copy()
    initial_count = len(working_df)
    transformations_log: List[Dict[str, str]] = []

    # -------------------------------------------------------------
    # Step 1: Direct Identifier Removal (Redaction)
    # -------------------------------------------------------------
    if direct_identifiers:
        cols_to_remove = [col for col in direct_identifiers if col in working_df.columns]
        if cols_to_remove:
            working_df = working_df.drop(columns=cols_to_remove)
            transformations_log.append({
                "stage": "Direct Identifier Redaction",
                "column": ", ".join(cols_to_remove),
                "details": f"Completely stripped {len(cols_to_remove)} direct identifier column(s) to prevent single-attribute lookup attacks."
            })

    # -------------------------------------------------------------
    # Step 2: Quasi-Identifier Generalization
    # -------------------------------------------------------------
    generalized_qis = [col for col in quasi_identifiers if col in working_df.columns]

    for col in generalized_qis:
        transformed_col, explanation = _generalize_column(
            working_df[col],
            col_name=col,
            level=generalization_level
        )
        working_df[col] = transformed_col
        if explanation:
            transformations_log.append({
                "stage": "Quasi-Identifier Generalization",
                "column": col,
                "details": explanation
            })

    # -------------------------------------------------------------
    # Step 3: Equivalence Class Evaluation & Outlier Suppression
    # -------------------------------------------------------------
    suppressed_df = pd.DataFrame()
    num_suppressed = 0

    if generalized_qis:
        # Calculate post-generalization equivalence class sizes
        class_sizes = working_df.groupby(generalized_qis, observed=False, dropna=False)[generalized_qis[0]].transform("count")
        violating_mask = class_sizes < k

        if violating_mask.any():
            if suppress_outliers:
                suppressed_df = working_df[violating_mask].copy()
                working_df = working_df[~violating_mask].copy()
                num_suppressed = int(violating_mask.sum())
                transformations_log.append({
                    "stage": "Outlier Suppression",
                    "column": "Multiple QIs",
                    "details": (
                        f"Suppressed {num_suppressed} outlier record(s) belonging to equivalence "
                        f"classes with size < {k} to guarantee strict K={k} anonymity."
                    )
                })
            else:
                transformations_log.append({
                    "stage": "Warning: Outliers Retained",
                    "column": "Multiple QIs",
                    "details": f"Found {violating_mask.sum()} records with EC size < {k}. Suppression was disabled."
                })

    # -------------------------------------------------------------
    # Step 4: Final Risk and K-Anonymity Verification
    # -------------------------------------------------------------
    final_count = len(working_df)
    retention_rate = (final_count / initial_count * 100.0) if initial_count > 0 else 0.0

    if final_count > 0 and generalized_qis:
        final_ec = get_equivalence_classes(working_df, generalized_qis)
        final_min_ec = int(final_ec.min())
        final_max_ec = int(final_ec.max())
        is_satisfied = final_min_ec >= k
    elif final_count == 0:
        final_min_ec = 0
        final_max_ec = 0
        is_satisfied = False
    else:
        final_min_ec = final_count
        final_max_ec = final_count
        is_satisfied = final_min_ec >= k

    return AnonymizationResult(
        anonymized_df=working_df,
        suppressed_df=suppressed_df,
        transformation_info=transformations_log,
        initial_records_count=initial_count,
        final_records_count=final_count,
        num_suppressed_records=num_suppressed,
        num_generalized_records=initial_count,
        final_min_ec_size=final_min_ec,
        final_max_ec_size=final_max_ec,
        is_k_satisfied=is_satisfied,
        k_target=k,
        retention_rate_pct=round(retention_rate, 2)
    )
