"""K-Anonymity Educational Data Package."""

from src.data_generator import generate_student_dataset, save_dataset_to_csv
from src.hierarchies import (
    generalize_age,
    generalize_cgpa,
    generalize_city,
    generalize_department,
    generalize_year
)
from src.risk_analysis import (
    RiskReport,
    get_equivalence_classes,
    calculate_risk_metrics,
    annotate_dataset_with_risk
)
from src.k_anonymity import (
    AnonymizationResult,
    apply_k_anonymity
)
from src.metrics import (
    UtilityMetrics,
    calculate_generalization_loss,
    calculate_information_loss,
    calculate_utility_score,
    calculate_privacy_score,
    compute_metrics_bundle
)
from src.k_evaluator import (
    KEvaluationSummary,
    evaluate_k_range
)
