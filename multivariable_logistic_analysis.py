"""Multivariable Logistic Regression with OR, 95% CI, and VIF analysis."""
import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from statsmodels.stats.outliers_influence import variance_inflation_factor
import statsmodels.api as sm
from imblearn.under_sampling import RandomUnderSampler

BASE = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE, "diabetes_012_health_indicators_BRFSS2015.csv")
OUTPUT_DIR = os.path.join(BASE, "figures")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load data
print("Loading data...")
df = pd.read_csv(DATA_PATH)

# Create binary outcome
y = (df["Diabetes_012"] != 0).astype(int)
x = df.drop(columns=["Diabetes_012"])
feature_names = x.columns.tolist()

# Stratified train-test split
x_train_orig, x_test_orig, y_train_orig, y_test_orig = train_test_split(
    x, y, test_size=0.2, random_state=42, stratify=y
)

# Scale
scaler = StandardScaler()
x_train_sc = scaler.fit_transform(x_train_orig)
x_test_sc = scaler.transform(x_test_orig)

# For multivariable LR, use original test set (UNUNDERSAMPLED)
# This preserves true population proportions for unbiased OR estimation
print(f"\nTest set shape: {x_test_sc.shape}")
print(f"Test set abnormal prevalence: {y_test_orig.mean():.1%}")

# Compute class weights for balanced fitting
class_weights = compute_class_weight('balanced', classes=np.array([0, 1]), y=y_test_orig)
print(f"Class weights: {dict(zip([0, 1], class_weights))}")

# Create sample_weight array
sample_weight = np.array([class_weights[int(y)] for y in y_test_orig])

# Add constant for intercept
X_test_const = sm.add_constant(x_test_sc)

# Fit weighted logistic regression
print("\nFitting multivariable logistic regression...")
logit_model = sm.Logit(y_test_orig, X_test_const)
result = logit_model.fit(sample_weight=sample_weight, disp=0)

# Print summary
print(result.summary())

# Prepare output table
results_list = []

# Get coefficient names (first is 'const')
coef_names = result.params.index.tolist()

# Intercept
const_idx = 'const'
results_list.append({
    'Variable': 'Intercept',
    'Coefficient': result.params[const_idx],
    'Std_Error': result.bse[const_idx],
    'Adjusted_OR': np.exp(result.params[const_idx]),
    'OR_95CI_Lower': np.exp(result.params[const_idx] - 1.96 * result.bse[const_idx]),
    'OR_95CI_Upper': np.exp(result.params[const_idx] + 1.96 * result.bse[const_idx]),
    'z_value': result.tvalues[const_idx],
    'p_value': result.pvalues[const_idx],
    'VIF': np.nan,
    'Significant': 'Yes' if result.pvalues[const_idx] < 0.05 else 'No'
})

# Feature parameters
for i, feature in enumerate(feature_names):
    coef_name = f'x{i+1}'
    coef = result.params[coef_name]
    se = result.bse[coef_name]
    or_val = np.exp(coef)
    or_lower = np.exp(coef - 1.96 * se)
    or_upper = np.exp(coef + 1.96 * se)
    p_val = result.pvalues[coef_name]
    
    results_list.append({
        'Variable': feature,
        'Coefficient': coef,
        'Std_Error': se,
        'Adjusted_OR': or_val,
        'OR_95CI_Lower': or_lower,
        'OR_95CI_Upper': or_upper,
        'z_value': result.tvalues[coef_name],
        'p_value': p_val,
        'VIF': np.nan,
        'Significant': 'Yes' if p_val < 0.05 else 'No'
    })

results_df = pd.DataFrame(results_list)

# Calculate VIF for each feature
print("\nCalculating VIF for collinearity assessment...")
vif_values = []
for i in range(x_test_sc.shape[1]):
    vif = variance_inflation_factor(x_test_sc, i)
    vif_values.append(vif)

# Add VIF to results (for features only, not intercept)
results_df.loc[1:, 'VIF'] = vif_values

# Save results
output_path = os.path.join(OUTPUT_DIR, "multivariable_logistic_results.csv")
results_df.to_csv(output_path, index=False)
print(f"\nResults saved to: {output_path}")

# Display summary statistics
print("\n" + "="*100)
print("MULTIVARIABLE LOGISTIC REGRESSION SUMMARY (N=50,736, Abnormal=15.8%)")
print("="*100)
display_cols = ['Variable', 'Adjusted_OR', 'OR_95CI_Lower', 'OR_95CI_Upper', 'p_value', 'Significant', 'VIF']
print(results_df[display_cols].to_string(index=False))

# Top 5 risk factors
print("\n" + "="*100)
print("TOP 5 RISK FACTORS (Highest Adjusted OR, OR > 1)")
print("="*100)
top_risk = results_df[(results_df['Variable'] != 'Intercept') & (results_df['Adjusted_OR'] > 1)].nlargest(5, 'Adjusted_OR')[
    ['Variable', 'Adjusted_OR', 'OR_95CI_Lower', 'OR_95CI_Upper', 'p_value']
]
print(top_risk.to_string(index=False))

# Top 5 protective factors
print("\n" + "="*100)
print("TOP 5 PROTECTIVE FACTORS (Lowest Adjusted OR, OR < 1)")
print("="*100)
top_protective = results_df[(results_df['Variable'] != 'Intercept') & (results_df['Adjusted_OR'] < 1)].nsmallest(5, 'Adjusted_OR')[
    ['Variable', 'Adjusted_OR', 'OR_95CI_Lower', 'OR_95CI_Upper', 'p_value']
]
print(top_protective.to_string(index=False))

# VIF summary
print("\n" + "="*100)
print("MULTICOLLINEARITY ASSESSMENT (VIF Summary)")
print("="*100)
high_vif = results_df[(results_df['Variable'] != 'Intercept') & (results_df['VIF'] >= 5)]
if len(high_vif) > 0:
    print(f"Variables with VIF >= 5:")
    print(high_vif[['Variable', 'VIF']].to_string(index=False))
else:
    print("No variables with VIF >= 5. Good collinearity profile.")

very_high_vif = results_df[(results_df['Variable'] != 'Intercept') & (results_df['VIF'] >= 10)]
if len(very_high_vif) > 0:
    print(f"\nVariables with VIF >= 10 (concerning):")
    print(very_high_vif[['Variable', 'VIF']].to_string(index=False))
else:
    print("No variables with VIF >= 10. Multicollinearity is not a concern.")

print(f"\nMean VIF (excluding Intercept): {results_df[results_df['Variable'] != 'Intercept']['VIF'].mean():.2f}")
print(f"Median VIF (excluding Intercept): {results_df[results_df['Variable'] != 'Intercept']['VIF'].median():.2f}")

print("\n✓ Multivariable logistic regression analysis complete!")
