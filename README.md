# Type 2 Diabetes Risk Screening (ML)

Course project: binary classification of metabolic abnormality (prediabetes or diabetes vs. normal) using CDC BRFSS–derived health indicators, with statistical analysis and comparative ML models.

## Setup

```bash
python -m venv .venv
# activate venv, then:
pip install numpy pandas scipy matplotlib seaborn scikit-learn statsmodels imbalanced-learn xgboost lightgbm python-docx
```

Place `diabetes_012_health_indicators_BRFSS2015.csv` in the project root.

## Main scripts

| Script | Purpose |
|--------|---------|
| `binary_and_univariable_analysis.py` | ML pipeline + univariable tests |
| `multivariable_logistic_analysis.py` | Weighted logistic regression + VIF |

Paths resolve from each script’s location; run from the repository root.

