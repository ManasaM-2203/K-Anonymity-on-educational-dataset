"""
K-Anonymity for Educational Data: Privacy Analyzer
A student micro-project for evaluating re-identification risk,
multi-K trade-offs, and recommending an optimal K value.
"""

import os
import streamlit as st
import pandas as pd
import plotly.express as px

from src.data_generator import generate_student_dataset, save_dataset_to_csv
from src.risk_analysis import calculate_risk_metrics
from src.k_evaluator import evaluate_k_range

# -----------------------------------------------------------------------------
# Configuration & Schema Definitions
# -----------------------------------------------------------------------------
DIRECT_IDENTIFIERS = ["Student_ID", "Name"]
QUASI_IDENTIFIERS = ["Age", "Gender", "Department", "Year", "City", "CGPA"]
SENSITIVE_ATTRIBUTES = ["Academic_Performance"]

PRIVACY_WEIGHT = 0.5
UTILITY_WEIGHT = 0.5

# -----------------------------------------------------------------------------
# Streamlit Page Setup
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="K-Anonymity for Educational Data",
    page_icon="🎓",
    layout="centered"
)

# -----------------------------------------------------------------------------
# Session State Initialization for K Evaluation Range
# -----------------------------------------------------------------------------
if "k_min" not in st.session_state:
    st.session_state["k_min"] = 2
if "k_max" not in st.session_state:
    st.session_state["k_max"] = 10

# -----------------------------------------------------------------------------
# Default Data Loader
# -----------------------------------------------------------------------------
@st.cache_data
def load_default_data() -> pd.DataFrame:
    """Loads default student dataset from CSV or generates a reproducible synthetic dataset."""
    data_path = "data/students_dataset.csv"
    if os.path.exists(data_path):
        return pd.read_csv(data_path)
    df = generate_student_dataset(n_records=750, random_seed=42)
    save_dataset_to_csv(df, data_path)
    return df


# -----------------------------------------------------------------------------
# Title and Subtitle
# -----------------------------------------------------------------------------
st.title("K-Anonymity for Educational Data")
st.caption("Analyzing re-identification risk and selecting a suitable K value")

# -----------------------------------------------------------------------------
# Main Navigation Tabs (3 Tabs Only)
# -----------------------------------------------------------------------------
tab1, tab2, tab3 = st.tabs([
    "Dataset & Baseline Risk",
    "K Analysis",
    "Anonymized Result"
])

# -----------------------------------------------------------------------------
# TAB 1: Dataset & Baseline Risk
# -----------------------------------------------------------------------------
with tab1:
    st.subheader("Dataset Source")
    uploaded_file = st.file_uploader("Upload Student Dataset (optional)", type=["csv"])

    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            if df.empty:
                st.warning("Uploaded file is empty. Using default student dataset instead.")
                df = load_default_data()
        except Exception as e:
            st.error(f"Failed to read uploaded CSV: {e}. Using default dataset.")
            df = load_default_data()
    else:
        df = load_default_data()

    # Active attribute classification from current dataframe
    active_direct = [col for col in DIRECT_IDENTIFIERS if col in df.columns]
    active_qis = [col for col in QUASI_IDENTIFIERS if col in df.columns]

    # Fallback if custom uploaded CSV has different columns
    if not active_qis:
        active_qis = [col for col in df.columns if col not in active_direct and col not in SENSITIVE_ATTRIBUTES]

    # Baseline risk calculation
    pre_risk = calculate_risk_metrics(df, active_qis, k=2)

    st.write("---")
    st.subheader("Dataset Summary")
    c1, c2 = st.columns(2)
    c1.metric("Number of Records", f"{len(df):,}")
    c2.metric("Number of Attributes", f"{len(df.columns)}")

    st.write("---")
    st.subheader("Attribute Classification")

    schema_data = []
    for col in df.columns:
        if col in active_direct:
            role = "Direct Identifier"
        elif col in active_qis:
            role = "Quasi-Identifier"
        elif col in SENSITIVE_ATTRIBUTES:
            role = "Sensitive Attribute"
        else:
            role = "Other"
        schema_data.append({"Attribute": col, "Role": role})

    st.dataframe(pd.DataFrame(schema_data), use_container_width=True, hide_index=True)

    st.write("---")
    st.subheader("Baseline Risk")
    st.write("Before anonymization, combinations of quasi-identifiers can uniquely identify students.")

    r1, r2, r3 = st.columns(3)
    r1.metric("Total Equivalence Classes", f"{pre_risk.num_equivalence_classes:,}")
    r2.metric("Singleton Records", f"{pre_risk.num_singletons:,}")
    r3.metric("Maximum Re-ID Risk", f"{pre_risk.max_reidentification_risk * 100:.1f}%")


# -----------------------------------------------------------------------------
# TAB 2: K Analysis
# -----------------------------------------------------------------------------
with tab2:
    st.subheader("Multi-K Evaluation")

    # Editable K Evaluation Range Controls
    col_k1, col_k2, col_btn = st.columns([1, 1, 1.2])
    with col_k1:
        input_k_min = st.number_input(
            "Minimum K",
            min_value=2,
            max_value=max(2, len(df)),
            value=int(st.session_state["k_min"]),
            step=1
        )
    with col_k2:
        input_k_max = st.number_input(
            "Maximum K",
            min_value=2,
            max_value=max(2, len(df)),
            value=int(st.session_state["k_max"]),
            step=1
        )
    with col_btn:
        st.write("")
        st.write("")
        if st.button("Evaluate K Values", use_container_width=True):
            if input_k_min < 2:
                st.error("Minimum K must be at least 2.")
            elif input_k_max < input_k_min:
                st.error("Maximum K must be greater than or equal to Minimum K.")
            elif input_k_max > len(df):
                st.error(f"Maximum K cannot exceed total records ({len(df)}).")
            else:
                st.session_state["k_min"] = int(input_k_min)
                st.session_state["k_max"] = int(input_k_max)
                st.rerun()

    # Active evaluated K range
    eval_k_min = int(st.session_state["k_min"])
    eval_k_max = int(st.session_state["k_max"])

    # Safe bounds clamping against active dataset size
    if eval_k_max > len(df) and len(df) >= 2:
        eval_k_max = len(df)
    if eval_k_min > eval_k_max:
        eval_k_min = min(2, eval_k_max)

    # Multi-K range evaluation
    eval_summary = evaluate_k_range(
        df=df,
        quasi_identifiers=active_qis,
        direct_identifiers=active_direct,
        k_min=eval_k_min,
        k_max=eval_k_max,
        privacy_weight=PRIVACY_WEIGHT,
        utility_weight=UTILITY_WEIGHT
    )

    rec_k = eval_summary.recommended_k
    rec_anon_result = eval_summary.anonymized_results_by_k[rec_k]
    post_risk = calculate_risk_metrics(rec_anon_result.anonymized_df, active_qis, k=rec_k)

    st.write(f"Evaluating candidate K values from **{eval_k_min}** to **{eval_k_max}** to measure privacy protection and data utility.")

    # K Comparison Table
    k_table_rows = []
    chart_records = []

    for k in range(eval_k_min, eval_k_max + 1):
        metrics = eval_summary.results_by_k[k]
        k_table_rows.append({
            "K": k,
            "Privacy Risk": f"{metrics.max_reid_risk * 100:.1f}%",
            "Information Loss": f"{metrics.total_information_loss:.3f}",
            "Utility": f"{metrics.data_utility_score:.3f}",
            "Balance Score": f"{metrics.balance_score:.3f}"
        })
        chart_records.append({
            "K": k,
            "Privacy Score": metrics.privacy_score,
            "Utility Score": metrics.data_utility_score,
            "Balance Score": metrics.balance_score
        })

    st.dataframe(pd.DataFrame(k_table_rows), use_container_width=True, hide_index=True)

    st.write("---")
    st.subheader(f"Recommended K: {rec_k}")
    st.write("Selected because it provides the highest balance between privacy protection and data utility for the current dataset.")

    st.write("---")
    # Single Plot: Privacy vs Utility across K
    chart_df = pd.DataFrame(chart_records)
    fig = px.line(
        chart_df,
        x="K",
        y=["Privacy Score", "Utility Score", "Balance Score"],
        markers=True,
        title="Privacy vs Utility across K",
        labels={"value": "Score", "variable": "Metric"}
    )
    fig.update_layout(
        xaxis=dict(dtick=1, title="K"),
        yaxis=dict(range=[0, 1.05], title="Score"),
        height=380,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    st.plotly_chart(fig, use_container_width=True)


# -----------------------------------------------------------------------------
# TAB 3: Anonymized Result
# -----------------------------------------------------------------------------
with tab3:
    st.subheader("Anonymized Dataset")
    st.write(f"Selected K = **{rec_k}**")

    # Before vs After Comparison Table
    before_after_rows = [
        {"Metric": "Direct Identifiers", "Before": str(len(active_direct)), "After": "0"},
        {"Metric": "Total Records", "Before": f"{len(df):,}", "After": f"{len(rec_anon_result.anonymized_df):,}"},
        {"Metric": "Singletons", "Before": str(pre_risk.num_singletons), "After": str(post_risk.num_singletons)},
        {"Metric": "Minimum Equivalence Class Size", "Before": str(pre_risk.min_equivalence_class_size), "After": str(rec_anon_result.final_min_ec_size)},
        {"Metric": "Maximum Re-identification Risk", "Before": f"{pre_risk.max_reidentification_risk * 100:.1f}%", "After": f"{post_risk.max_reidentification_risk * 100:.1f}%"}
    ]
    st.dataframe(pd.DataFrame(before_after_rows), use_container_width=True, hide_index=True)

    # K-Anonymity Validation Status
    if rec_anon_result.final_min_ec_size >= rec_k and len(rec_anon_result.anonymized_df) > 0:
        st.success("K-anonymity requirement satisfied")
    else:
        st.error("K-anonymity requirement not satisfied")

    st.write("---")
    st.subheader("Anonymized Data Preview")
    st.dataframe(rec_anon_result.anonymized_df.head(10), use_container_width=True, hide_index=True)

    # Download Button
    csv_bytes = rec_anon_result.anonymized_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download Anonymized CSV",
        data=csv_bytes,
        file_name=f"anonymized_students_k{rec_k}.csv",
        mime="text/csv",
        use_container_width=True
    )
