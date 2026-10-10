"""
Multi-K Evaluation and Data-Driven K Recommendation Engine.

Evaluates K-Anonymity across a range of K values, computes privacy-utility
trade-off metrics, produces comparative tabular data, and automatically
recommends the optimal K value based on configurable privacy vs. utility weights.
"""

from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple
import pandas as pd
import numpy as np

from src.k_anonymity import apply_k_anonymity, AnonymizationResult
from src.risk_analysis import calculate_risk_metrics, get_equivalence_classes
from src.metrics import compute_metrics_bundle, UtilityMetrics


@dataclass
class KEvaluationSummary:
    """Dataclass holding complete multi-K evaluation results and recommendation."""
    recommended_k: int
    recommendation_reason: str
    results_by_k: Dict[int, UtilityMetrics]
    anonymized_results_by_k: Dict[int, AnonymizationResult]
    comparison_df: pd.DataFrame
    privacy_weight: float
    utility_weight: float
    k_min: int
    k_max: int


def evaluate_k_range(
    df: pd.DataFrame,
    quasi_identifiers: List[str],
    direct_identifiers: Optional[List[str]] = None,
    k_min: int = 2,
    k_max: int = 10,
    generalization_level: int = 1,
    suppress_outliers: bool = True,
    privacy_weight: float = 0.5,
    utility_weight: float = 0.5
) -> KEvaluationSummary:
    """
    Evaluates educational dataset across a continuous range of K values [k_min, k_max].

    For each K:
    1. Applies K-Anonymity (generalization + suppression).
    2. Measures achieved minimum equivalence class size and verifies compliance.
    3. Calculates post-transformation re-identification risk metrics.
    4. Computes normalized Information Loss and Data Utility scores.
    5. Calculates the Privacy-Utility Balance Score.

    Then determines the optimal recommended K value.

    Parameters:
        df (pd.DataFrame): Educational dataset.
        quasi_identifiers (List[str]): Quasi-identifier column names.
        direct_identifiers (List[str]): Direct identifiers to redact.
        k_min (int): Minimum K to test (>= 2).
        k_max (int): Maximum K to test (>= k_min).
        generalization_level (int): Hierarchy depth level (1 or 2).
        suppress_outliers (bool): Whether to suppress residual outlier classes < K.
        privacy_weight (float): Importance weight for privacy (default 0.5).
        utility_weight (float): Importance weight for utility (default 0.5).

    Returns:
        KEvaluationSummary: Structured results and recommendation.
    """
    # Validation of parameters
    if k_min < 2:
        k_min = 2
    if k_max < k_min:
        k_max = k_min

    results_by_k: Dict[int, UtilityMetrics] = {}
    anon_results_by_k: Dict[int, AnonymizationResult] = {}
    table_rows: List[Dict[str, Any]] = []

    if df.empty or not quasi_identifiers:
        # Edge case: Empty dataset or no QIs
        empty_df = pd.DataFrame(columns=[
            "K", "Min Equivalence Class", "Achieved Status", "Re-identification Risk",
            "Information Loss", "Utility Score", "Retention Rate", "Balance Score", "Recommended"
        ])
        return KEvaluationSummary(
            recommended_k=k_min,
            recommendation_reason="Dataset is empty or no quasi-identifiers selected.",
            results_by_k={},
            anonymized_results_by_k={},
            comparison_df=empty_df,
            privacy_weight=privacy_weight,
            utility_weight=utility_weight,
            k_min=k_min,
            k_max=k_max
        )

    # Multi-K Evaluation Loop
    for k in range(k_min, k_max + 1):
        # 1. Apply K-Anonymity Transformation
        anon_res = apply_k_anonymity(
            df=df,
            quasi_identifiers=quasi_identifiers,
            direct_identifiers=direct_identifiers,
            k=k,
            generalization_level=generalization_level,
            suppress_outliers=suppress_outliers
        )
        anon_results_by_k[k] = anon_res

        # 2. Risk Metrics
        post_risk = calculate_risk_metrics(
            df=anon_res.anonymized_df,
            quasi_identifiers=quasi_identifiers,
            k=k
        )

        # 3. Privacy, Information Loss, and Utility Bundle
        metrics = compute_metrics_bundle(
            raw_df=df,
            anonymized_df=anon_res.anonymized_df,
            quasi_identifiers=quasi_identifiers,
            k_requested=k,
            k_achieved=anon_res.final_min_ec_size,
            is_k_satisfied=anon_res.is_k_satisfied,
            initial_records=anon_res.initial_records_count,
            retained_records=anon_res.final_records_count,
            suppressed_records=anon_res.num_suppressed_records,
            generalization_level=generalization_level,
            privacy_weight=privacy_weight,
            utility_weight=utility_weight,
            num_equivalence_classes=post_risk.num_equivalence_classes
        )
        results_by_k[k] = metrics

    # 4. Determine Recommended K
    # Filter candidates where strict K is satisfied and data is retained
    valid_ks = [
        k for k, m in results_by_k.items()
        if m.is_k_satisfied and m.retained_records > 0
    ]

    if not valid_ks:
        # If none strictly satisfied (e.g. suppression disabled or tiny dataset), consider all with records
        valid_ks = [k for k, m in results_by_k.items() if m.retained_records > 0]

    if valid_ks:
        # Select K with the highest Balance Score
        # In case of tie, select higher privacy (higher K) or lowest info loss
        recommended_k = max(
            valid_ks,
            key=lambda k: (results_by_k[k].balance_score, results_by_k[k].privacy_score, -results_by_k[k].total_information_loss)
        )
        best_m = results_by_k[recommended_k]
        reason = (
            f"K = {recommended_k} is recommended as it achieves the highest Balance Score ({best_m.balance_score:.3f}) "
            f"with Privacy Score of {best_m.privacy_score:.3f} (Max Re-ID Risk: {best_m.max_reid_risk * 100:.1f}%) "
            f"and Data Utility Score of {best_m.data_utility_score:.3f} ({best_m.retention_rate * 100:.1f}% records retained) "
            f"under Privacy Weight = {privacy_weight:.2f} and Utility Weight = {utility_weight:.2f}."
        )
    else:
        recommended_k = k_min
        reason = f"K = {recommended_k} chosen as baseline because dataset could not satisfy higher K constraints without full record suppression."

    # 5. Build Comparative DataFrame
    for k in range(k_min, k_max + 1):
        m = results_by_k[k]
        is_rec = (k == recommended_k)
        table_rows.append({
            "K": k,
            "Min Equivalence Class": m.k_achieved,
            "Strict K Satisfied": "✅ Yes" if m.is_k_satisfied else "❌ No",
            "Max Re-ID Risk": f"{m.max_reid_risk * 100:.1f}%",
            "Privacy Score": round(m.privacy_score, 3),
            "Information Loss": round(m.total_information_loss, 3),
            "Utility Score": round(m.data_utility_score, 3),
            "Retention Rate": f"{m.retention_rate * 100:.1f}%",
            "Balance Score": round(m.balance_score, 3),
            "Recommended": "⭐ RECOMMENDED" if is_rec else ""
        })

    comparison_df = pd.DataFrame(table_rows)

    return KEvaluationSummary(
        recommended_k=recommended_k,
        recommendation_reason=reason,
        results_by_k=results_by_k,
        anonymized_results_by_k=anon_results_by_k,
        comparison_df=comparison_df,
        privacy_weight=privacy_weight,
        utility_weight=utility_weight,
        k_min=k_min,
        k_max=k_max
    )
