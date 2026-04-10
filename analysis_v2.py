"""
Progress Report Analysis Script (Optimized)
- EDA & Data Preprocessing
- Model Training (Logistic Regression, Random Forest, XGBoost, LightGBM)
- Figure Generation
"""
import os, warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    accuracy_score, precision_score, recall_score, f1_score,
    ConfusionMatrixDisplay
)
from imblearn.combine import SMOTETomek
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
import xgboost as xgb
import lightgbm as lgb

# ── Setup ──
_BASE = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(_BASE, "diabetes_012_health_indicators_BRFSS2015.csv")
FIG_DIR   = os.path.join(_BASE, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

plt.rcParams.update({"figure.dpi": 150, "savefig.bbox": "tight"})
sns.set_style("whitegrid")

df = pd.read_csv(DATA_PATH)
print(f"Dataset shape: {df.shape}")
print(f"Missing values total: {df.isnull().sum().sum()}")
print(f"Duplicates: {df.duplicated().sum()}")

# ============================================================
# 1. EDA FIGURES
# ============================================================
labels_short = ["No Diabetes", "Pre-diabetes", "Diabetes"]
colors3 = ["#2ecc71", "#f39c12", "#e74c3c"]

# Fig 1: Target distribution
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
counts = df["Diabetes_012"].value_counts().sort_index()
axes[0].bar(labels_short, counts.values, color=colors3, edgecolor="black", linewidth=0.5)
for i, v in enumerate(counts.values):
    axes[0].text(i, v + 2000, f"{v:,}\n({v/len(df)*100:.1f}%)", ha="center", fontweight="bold", fontsize=9)
axes[0].set_title("Distribution of Diabetes_012 (Count)", fontsize=13, fontweight="bold")
axes[0].set_ylabel("Number of Samples")
axes[1].pie(counts.values, labels=labels_short, autopct="%1.1f%%", colors=colors3,
            startangle=90, textprops={"fontsize": 11})
axes[1].set_title("Class Distribution (%)", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "fig1_target_distribution.png"))
plt.close()
print("✓ Fig 1 saved")

# Fig 2: BMI distribution
fig, ax = plt.subplots(figsize=(10, 5))
for val, lab, col in zip([0, 1, 2], labels_short, colors3):
    ax.hist(df.loc[df["Diabetes_012"] == val, "BMI"], bins=50, alpha=0.55, label=lab, color=col, density=True)
ax.set_title("BMI Distribution by Diabetes Status", fontsize=13, fontweight="bold")
ax.set_xlabel("BMI"); ax.set_ylabel("Density"); ax.legend()
plt.savefig(os.path.join(FIG_DIR, "fig2_bmi_distribution.png")); plt.close()
print("✓ Fig 2 saved")

# Fig 3: Correlation heatmap
fig, ax = plt.subplots(figsize=(14, 11))
corr = df.corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="RdBu_r",
            center=0, linewidths=0.5, ax=ax, annot_kws={"size": 7}, vmin=-1, vmax=1)
ax.set_title("Feature Correlation Heatmap", fontsize=14, fontweight="bold")
plt.savefig(os.path.join(FIG_DIR, "fig3_correlation_heatmap.png")); plt.close()
print("✓ Fig 3 saved")

# Fig 4: Key features by class
key_features = ["HighBP", "HighChol", "GenHlth", "Age", "Income", "PhysActivity"]
fig, axes = plt.subplots(2, 3, figsize=(16, 10))
for i, feat in enumerate(key_features):
    ax = axes[i // 3][i % 3]
    for val, lab, col in zip([0, 1, 2], labels_short, colors3):
        ax.hist(df.loc[df["Diabetes_012"] == val, feat], bins=20, alpha=0.55, label=lab, color=col, density=True)
    ax.set_title(feat, fontsize=12, fontweight="bold"); ax.legend(fontsize=7)
plt.suptitle("Key Feature Distributions by Diabetes Status", fontsize=14, fontweight="bold")
plt.tight_layout(); plt.savefig(os.path.join(FIG_DIR, "fig4_key_feature_distributions.png")); plt.close()
print("✓ Fig 4 saved")

# ============================================================
# 2. PREPROCESSING
# ============================================================
print("\n=== Preprocessing ===")
X = df.drop("Diabetes_012", axis=1)
y = df["Diabetes_012"].astype(int)
feature_names = X.columns.tolist()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)
print(f"Train: {X_train.shape}  Test: {X_test.shape}")
print(f"Train distribution:\n{y_train.value_counts().sort_index()}")

scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

# Resampling: first undersample majority to 50k, then SMOTE minorities up to 50k
print("\nResampling (UnderSample majority → SMOTE minorities)...")
resample_pipe = ImbPipeline([
    ("under", RandomUnderSampler(
        sampling_strategy={0: 50000}, random_state=42)),
    ("smote", SMOTE(random_state=42))
])
X_train_res, y_train_res = resample_pipe.fit_resample(X_train_sc, y_train)
print(f"After resampling: {pd.Series(y_train_res).value_counts().sort_index().to_dict()}")

# ============================================================
# 3. MODEL TRAINING
# ============================================================
print("\n=== Model Training ===")
results = {}

def evaluate(name, model, X_tr, y_tr, X_te, y_te):
    model.fit(X_tr, y_tr)
    y_pred = model.predict(X_te)
    y_proba = model.predict_proba(X_te) if hasattr(model, "predict_proba") else None
    auc = roc_auc_score(y_te, y_proba, multi_class="ovr", average="weighted") if y_proba is not None else None
    acc  = accuracy_score(y_te, y_pred)
    prec = precision_score(y_te, y_pred, average="weighted", zero_division=0)
    rec  = recall_score(y_te, y_pred, average="weighted", zero_division=0)
    f1   = f1_score(y_te, y_pred, average="weighted", zero_division=0)
    results[name] = {"Accuracy": acc, "Precision": prec, "Recall": rec,
                     "F1-Score": f1, "ROC-AUC": auc}
    print(f"\n--- {name} ---")
    auc_str = f"  AUC: {auc:.4f}" if auc else ""
    print(f"Acc: {acc:.4f}  Prec: {prec:.4f}  Rec: {rec:.4f}  F1: {f1:.4f}{auc_str}")
    print(classification_report(y_te, y_pred, target_names=labels_short, zero_division=0))
    return model, y_pred

lr_m, lr_p = evaluate("Logistic Regression",
    LogisticRegression(max_iter=1000, random_state=42),
    X_train_res, y_train_res, X_test_sc, y_test)

rf_m, rf_p = evaluate("Random Forest",
    RandomForestClassifier(n_estimators=200, max_depth=15, random_state=42, n_jobs=-1),
    X_train_res, y_train_res, X_test_sc, y_test)

xgb_m, xgb_p = evaluate("XGBoost",
    xgb.XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.1,
                       random_state=42, eval_metric="mlogloss", n_jobs=-1),
    X_train_res, y_train_res, X_test_sc, y_test)

lgb_m, lgb_p = evaluate("LightGBM",
    lgb.LGBMClassifier(n_estimators=200, max_depth=6, learning_rate=0.1,
                        random_state=42, verbose=-1, n_jobs=-1),
    X_train_res, y_train_res, X_test_sc, y_test)

# ============================================================
# 4. RESULTS FIGURES
# ============================================================
print("\n=== Results Summary ===")
res_df = pd.DataFrame(results).T.round(4)
print(res_df.to_string())
res_df.to_csv(os.path.join(FIG_DIR, "model_comparison.csv"))

# Fig 5: Model comparison
fig, ax = plt.subplots(figsize=(10, 6))
metrics = ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"]
x = np.arange(len(results)); w = 0.15
for i, m in enumerate(metrics):
    vals = [results[n].get(m) or 0 for n in results]
    ax.bar(x + i * w, vals, w, label=m)
ax.set_xticks(x + w * 2); ax.set_xticklabels(results.keys(), fontsize=10)
ax.set_ylabel("Score"); ax.set_ylim(0, 1.05)
ax.set_title("Model Performance Comparison", fontsize=13, fontweight="bold")
ax.legend(loc="lower right")
plt.savefig(os.path.join(FIG_DIR, "fig5_model_comparison.png")); plt.close()
print("✓ Fig 5 saved")

# Fig 6: Confusion matrices
fig, axes = plt.subplots(1, 4, figsize=(22, 5))
for ax, (nm, pred) in zip(axes, [("Logistic Regression", lr_p), ("Random Forest", rf_p),
                                   ("XGBoost", xgb_p), ("LightGBM", lgb_p)]):
    cm = confusion_matrix(y_test, pred)
    ConfusionMatrixDisplay(cm, display_labels=labels_short).plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title(nm, fontsize=11, fontweight="bold")
plt.suptitle("Confusion Matrices", fontsize=14, fontweight="bold")
plt.tight_layout(); plt.savefig(os.path.join(FIG_DIR, "fig6_confusion_matrices.png")); plt.close()
print("✓ Fig 6 saved")

# Fig 7: Feature importance
fig, axes = plt.subplots(1, 2, figsize=(16, 7))
pd.Series(rf_m.feature_importances_, index=feature_names).sort_values(ascending=True).plot(
    kind="barh", ax=axes[0], color="#3498db")
axes[0].set_title("Random Forest Feature Importance", fontsize=12, fontweight="bold")
pd.Series(xgb_m.feature_importances_, index=feature_names).sort_values(ascending=True).plot(
    kind="barh", ax=axes[1], color="#e67e22")
axes[1].set_title("XGBoost Feature Importance", fontsize=12, fontweight="bold")
plt.suptitle("Feature Importance Analysis", fontsize=14, fontweight="bold")
plt.tight_layout(); plt.savefig(os.path.join(FIG_DIR, "fig7_feature_importance.png")); plt.close()
print("✓ Fig 7 saved")

# ── 5-Fold CV on XGBoost ──
print("\n=== 5-Fold CV (XGBoost) ===")
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_f1 = cross_val_score(
    xgb.XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.1,
                       random_state=42, eval_metric="mlogloss", n_jobs=-1),
    X_train_res, y_train_res, cv=cv, scoring="f1_weighted")
print(f"Fold F1 scores: {cv_f1.round(4)}")
print(f"Mean: {cv_f1.mean():.4f} ± {cv_f1.std():.4f}")

print(f"\nAll figures saved to {FIG_DIR}")
print("DONE.")
