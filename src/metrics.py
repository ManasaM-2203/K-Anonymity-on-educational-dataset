"""
Information Loss and Data Utility Metrics for Educational Data Anonymization.

Provides transparent, reproducible metrics for measuring the trade-off
between privacy protection and data utility in K-Anonymous datasets:
1. Suppression Loss (fraction of suppressed outlier records)
2. Generalization Loss (Normalized Certainty Penalty / hierarchy distortion)
3. Overall Normalized Information Loss (0.0 to 1.0)
4. Normalized Data Utility Score (0.0 to 1.0)
5. Normalized Privacy Score (0.0 to 1.0)
"""

from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple
import pandas as pd
import numpy as np


@dataclass
class UtilityMetrics:
    """Dataclass holding privacy, information loss, and utility measurements."""
    k_requested: int
    k_achieved: int
    is_k_satisfied: bool
    initial_records: int
    retained_records: int
    suppressed_records: int
    retention_rate: float          # 0.0 to 1.0 (retained / initial)
    suppression_loss: float        # 0.0 to 1.0 (suppressed / initial)
    generalization_loss: float     # 0.0 to 1.0 (attribute distortion on retained records)
    total_information_loss: float  # 0.0 to 1.0 (combined information loss)
    data_utility_score: float      # 0.0 to 1.0 (1 - total_information_loss)
    max_reid_risk: float           # 1 / k_achieved (or 1.0 if 0)
    avg_reid_risk: float           # num_ecs / retained_records
    privacy_score: float           # 0.0 to 1.0 (1 - max_reid_risk)
    balance_score: float = 0.0     # Weighted balance between privacy and utility

    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to a serializable dictionary."""
        return {
            "k_requested": self.k_requested,
            "k_achieved": self.k_achieved,
            "is_k_satisfied": self.is_k_satisfied,
            "initial_records": self.initial_records,
            "retained_records": self.retained_records,
            "suppressed_records": self.suppressed_records,
            "retention_rate": round(self.retention_rate, 4),
            "suppression_loss": round(self.suppression_loss, 4),
            "generalization_loss": round(self.generalization_loss, 4),
            "total_information_loss": round(self.total_information_loss, 4),
            "data_utility_score": round(self.data_utility_score, 4),
            "max_reid_risk": round(self.max_reid_risk, 4),
            "avg_reid_risk": round(self.avg_reid_risk, 4),
            "privacy_score": round(self.privacy_score, 4),
            "balance_score": round(self.balance_score, 4),
        }


def calculate_generalization_loss(
    raw_df: pd.DataFrame,
    anonymized_df: pd.DataFrame,
    quasi_identifiers: List[str],
    generalization_level: int = 1,
    max_hierarchy_depth: int = 2
) -> float:
    """
    Computes attribute generalization loss across retained records.

    For each quasi-identifier, measures the level of abstraction applied relative
    to the maximum available hierarchy depth.

    Formula:
        Loss(QI) = generalization_level / max_hierarchy_depth (if column modified)
        GenLoss = sum(Loss(QI) for QI in quasi_identifiers) / len(quasi_identifiers)

    Returns:
        float: Normalized generalization loss in [0.0, 1.0].
    """
    if anonymized_df.empty or not quasi_identifiers:
        return 0.0 if not anonymized_df.empty else 1.0

    valid_qis = [q for q in quasi_identifiers if q in anonymized_df.columns]
    if not valid_qis:
        return 0.0

    # Max depth across hierarchies is typically 2 (Level 1: standard interval/zone, Level 2: broad region/discipline)
    depth_factor = min(1.0, max(0.0, generalization_level / max(1, max_hierarchy_depth)))

    # Count how many QIs were actually modified between raw and anonymized
    modified_qis = 0
    for col in valid_qis:
        if col in raw_df.columns:
            # Check if values have been transformed
            # Compare sample of non-null values
            sample_size = min(len(anonymized_df), len(raw_df), 100)
            raw_sample = raw_df[col].dropna().iloc[:sample_size].astype(str)
            anon_sample = anonymized_df[col].dropna().iloc[:sample_size].astype(str)
            if not raw_sample.equals(anon_sample):
                modified_qis += 1
            else:
                if len(raw_sample) > 0 and len(anon_sample) > 0:
                    if raw_sample.iloc[0] != anon_sample.iloc[0]:
                        modified_qis += 1
        else:
            modified_qis += 1

    gen_loss = (modified_qis / len(valid_qis)) * depth_factor
    return float(np.clip(gen_loss, 0.0, 1.0))


def calculate_information_loss(
    initial_count: int,
    retained_count: int,
    suppressed_count: int,
    generalization_loss: float
) -> Tuple[float, float, float]:
    """
    Calculates the comprehensive information loss combining suppression and generalization.

    Formulas:
        Suppression Loss (S) = suppressed_count / initial_count
        Retention Rate (R)   = retained_count / initial_count
        Overall Information Loss (IL) = S + (R * Generalization Loss)

    Returns:
        Tuple[float, float, float]: (suppression_loss, retention_rate, total_information_loss)
    """
    if initial_count <= 0:
        return 0.0, 0.0, 1.0

    suppression_loss = suppressed_count / initial_count
    retention_rate = retained_count / initial_count

    # If all records are suppressed, total information loss is 1.0
    if retained_count == 0:
        return 1.0, 0.0, 1.0

    # Overall information loss: weighted by the records retained vs suppressed
    total_il = suppression_loss + (retention_rate * generalization_loss)
    total_il = float(np.clip(total_il, 0.0, 1.0))

    return float(suppression_loss), float(retention_rate), total_il


def calculate_utility_score(total_information_loss: float) -> float:
    """
    Calculates general-purpose dataset utility score.

    Formula:
        Utility Score (U) = 1.0 - Total Information Loss

    Returns:
        float: Utility score in [0.0, 1.0].
    """
    return float(np.clip(1.0 - total_information_loss, 0.0, 1.0))


def calculate_privacy_score(k_achieved: int, is_k_satisfied: bool) -> float:
    """
    Calculates the normalized privacy protection score based on minimum equivalence class size.

    Formula:
        If k_achieved <= 0: Privacy Score = 0.0
        Otherwise: Privacy Score = 1.0 - (1.0 / k_achieved)

    If K is not strictly satisfied (e.g. residual singletons exist), a penalty is applied.

    Returns:
        float: Privacy score in [0.0, 1.0].
    """
    if k_achieved <= 0:
        return 0.0

    # Theoretical maximum re-identification risk = 1 / k_achieved
    max_risk = 1.0 / k_achieved
    base_privacy = 1.0 - max_risk

    if not is_k_satisfied:
        base_privacy *= 0.5  # 50% penalty if K constraint was violated

    return float(np.clip(base_privacy, 0.0, 1.0))


def compute_metrics_bundle(
    raw_df: pd.DataFrame,
    anonymized_df: pd.DataFrame,
    quasi_identifiers: List[str],
    k_requested: int,
    k_achieved: int,
    is_k_satisfied: bool,
    initial_records: int,
    retained_records: int,
    suppressed_records: int,
    generalization_level: int = 1,
    privacy_weight: float = 0.5,
    utility_weight: float = 0.5,
    num_equivalence_classes: int = 0
) -> UtilityMetrics:
    """
    Aggregates all privacy, information loss, and utility calculations into a UtilityMetrics object.
    """
    # 1. Generalization Loss
    gen_loss = calculate_generalization_loss(
        raw_df=raw_df,
        anonymized_df=anonymized_df,
        quasi_identifiers=quasi_identifiers,
        generalization_level=generalization_level
    )

    # 2. Suppression & Total Information Loss
    supp_loss, ret_rate, total_il = calculate_information_loss(
        initial_count=initial_records,
        retained_count=retained_records,
        suppressed_count=suppressed_records,
        generalization_loss=gen_loss
    )

    # 3. Data Utility Score
    utility_score = calculate_utility_score(total_il)

    # 4. Privacy Score & Re-ID Risks
    privacy_score = calculate_privacy_score(k_achieved, is_k_satisfied)
    max_reid_risk = (1.0 / k_achieved) if k_achieved > 0 else 1.0
    avg_reid_risk = (num_equivalence_classes / retained_records) if retained_records > 0 else 1.0
    avg_reid_risk = float(np.clip(avg_reid_risk, 0.0, 1.0))

    # 5. Balance Score
    # Normalize weights to sum to 1.0
    total_weight = privacy_weight + utility_weight
    w_p = (privacy_weight / total_weight) if total_weight > 0 else 0.5
    w_u = (utility_weight / total_weight) if total_weight > 0 else 0.5

    balance_score = (w_p * privacy_score) + (w_u * utility_score)
    balance_score = float(np.clip(balance_score, 0.0, 1.0))

    return UtilityMetrics(
        k_requested=k_requested,
        k_achieved=k_achieved,
        is_k_satisfied=is_k_satisfied,
        initial_records=initial_records,
        retained_records=retained_records,
        suppressed_records=suppressed_records,
        retention_rate=ret_rate,
        suppression_loss=supp_loss,
        generalization_loss=gen_loss,
        total_information_loss=total_il,
        data_utility_score=utility_score,
        max_reid_risk=max_reid_risk,
        avg_reid_risk=avg_reid_risk,
        privacy_score=privacy_score,
        balance_score=balance_score
    )
