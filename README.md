# K-Anonymity for Educational Data: Privacy Analyzer

> **A College Micro-Project on Data Privacy & K-Anonymity in Educational Data Mining**

---

## 1. Project Title
**K-Anonymity for Educational Data: Multi-K Evaluation & Privacy-Utility Trade-Off Analysis**

---

## 2. Problem Statement
Academic institutions frequently share student datasets for educational research and administrative analysis. However, simply removing names and student identification numbers (de-identification) is insufficient to protect student privacy. 

An attacker with access to external voter or demographic records can link non-sensitive quasi-identifiers (such as Age, Gender, Department, Year, and City) to uniquely re-identify individual students and uncover confidential records (such as Academic Performance or probation status).

---

## 3. Objective
1. Identify and classify dataset attributes into Direct Identifiers, Quasi-Identifiers, and Sensitive Attributes.
2. Compute baseline re-identification risk metrics and singletons on the un-anonymized dataset.
3. Systematically evaluate candidate $K$ values ($K = 2$ to $K = 10$) using generalization hierarchies and outlier suppression.
4. Measure the trade-off between Privacy Risk and Information Loss / Data Utility.
5. Recommend a suitable $K$ value using a transparent Balance Score when the downstream machine-learning task is unknown.
6. Generate and export the final $K$-anonymous sanitized dataset.

---

## 4. Base Paper Reference
* **Title:** *"An Approach to Improve k-Anonymization Practices in Educational Data Mining"*
* **Authors:** Frank Stinar, Zihan Xiong, and Nigel Bosch
* **Journal:** *Journal of Educational Data Mining*, 2024
* **Core Topic:** The authors study $K$-anonymization practices in Educational Data Mining (EDM), analyzing how varying $K$ impacts predictive accuracy across different machine learning models and feature sets.

---

## 5. Research Gap Addressed
In the base paper, Stinar et al. (2024) observe that selecting an appropriate $K$ value remains challenging when the downstream analytical or machine-learning task is unknown or unspecified at the time of data release. 

Traditional tools often force custodians to pick a single $K$ arbitrarily without quantitative visibility into the trade-off between privacy protection and utility loss. Our micro-project addresses this gap by providing an automated multi-$K$ evaluation framework that balances privacy score and general data utility.

---

## 6. Proposed Solution
We implement a streamlined educational privacy analyzer that:
* Evaluates a continuous range of $K$ values ($K = 2$ to $K = 10$).
* Applies explainable domain generalization and outlier suppression.
* Formulates a normalized **Balance Score** ($50\%$ Privacy Score + $50\%$ Data Utility Score).
* Dynamically identifies and recommends the optimal $K$ with the highest balance score.
* Provides a clean 3-tab interactive dashboard to inspect baseline risks, trade-off curves, and download the anonymized dataset.

---

## 7. Methodology
The application executes the following 11-step pipeline:

1. **Load Educational Dataset:** Ingest synthetic student records containing demographic and academic attributes.
2. **Schema Role Classification:** Categorize columns into Direct Identifiers (redacted), Quasi-Identifiers (generalized), and Sensitive Attributes (preserved).
3. **Baseline Risk Calculation:** Compute pre-anonymization equivalence class sizes, singletons, and maximum re-identification risk.
4. **Multi-$K$ Range Iteration:** Iterate across candidate values $K \in [2, 10]$.
5. **Generalization:** Transform quasi-identifiers using interval binning and categorical clustering (e.g., Age intervals, Grade bands, Regional zones).
6. **Suppression:** Suppress outlier records belonging to equivalence classes smaller than $K$ to strictly satisfy $|EC| \ge K$.
7. **Privacy Metric Formulation:** Calculate maximum re-identification risk ($1/K$) and normalized Privacy Score ($1 - 1/K$).
8. **Information Loss & Utility Formulation:** Calculate suppression loss, attribute generalization loss, total information loss, and data utility ($1 - \text{Total Information Loss}$).
9. **Balance Score Computation:** Calculate $\text{Balance}(K) = 0.5 \times \text{Privacy}(K) + 0.5 \times \text{Utility}(K)$.
10. **K Recommendation & Verification:** Select $K^* = \arg\max_K \text{Balance}(K)$ and verify minimum equivalence class size $\ge K^*$.
11. **Sanitized Export:** Render before-vs-after metrics and provide a downloadable CSV.

---

## 8. Technologies Used
* **Language:** Python 3.10+
* **Data Processing:** Pandas, NumPy
* **Interactive UI:** Streamlit
* **Data Visualization:** Plotly Express
* **Testing:** Python `unittest` framework

---

## 9. How to Run

### Step 1: Clone or Open Project Directory
```bash
cd K-Anonymity-Educational-Data
```

### Step 2: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 3: Launch Streamlit Dashboard
```bash
streamlit run app.py
```
Open **`http://localhost:8501`** in your browser.

### Step 4: Run Unit Tests
```bash
python -m unittest discover tests
```

---

## 10. Results
* **Dynamic K Recommendation:** The application evaluates $K$ values from $2$ to $10$ and dynamically recommends the $K$ value with the highest balance score for the current dataset.
* **Risk Reduction:** Singletons ($100\%$ re-identification risk) are completely eliminated in the anonymized output.
* **Strict Compliance:** Verified that all retained equivalence classes strictly satisfy the target condition $|EC_{\min}| \ge K$.

---

## 11. Project Structure
```
K-Anonymity-Educational-Data/
├── app.py                     # Streamlit Web Application (3-Tab Interface)
├── requirements.txt           # Python dependencies
├── README.md                  # Project report & documentation
├── .gitignore                 # Git ignore rules
│
├── data/
│   └── students_dataset.csv   # Student dataset (750 records)
│
├── src/
│   ├── __init__.py            # Package initialization
│   ├── data_generator.py      # Synthetic student dataset generator
│   ├── hierarchies.py         # Domain generalization hierarchies
│   ├── k_anonymity.py         # K-Anonymity transformation engine
│   ├── risk_analysis.py       # Re-identification risk metrics engine
│   ├── metrics.py             # Information loss & utility calculations
│   └── k_evaluator.py         # Multi-K evaluation & recommendation
│
└── tests/
    ├── test_risk_analysis.py  # Tests for equivalence classes & risk
    ├── test_k_anonymity.py    # Tests for transformation & suppression
    ├── test_metrics.py        # Tests for information loss & utility
    └── test_k_evaluator.py    # Tests for multi-K evaluation & recommendation
```

---

## 12. Limitations
1. **Suppression Data Loss:** Higher values of $K$ lead to higher record suppression when outliers are present.
2. **Quasi-Identifier Scope:** Assumes the set of quasi-identifiers is known and pre-defined.
3. **Syntactic Privacy Scope:** $K$-anonymity protects against identity disclosure via record linkage, but does not prevent attribute disclosure if sensitive values within an equivalence class lack diversity ($l$-diversity / $t$-closeness).
