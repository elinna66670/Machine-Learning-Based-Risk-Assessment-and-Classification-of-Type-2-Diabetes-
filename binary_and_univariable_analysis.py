import os
import json
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.stats import randint as sp_randint
from scipy.stats import uniform as sp_uniform
from scipy.stats import loguniform

from scipy.stats import chi2_contingency, mannwhitneyu
from statsmodels.stats.multitest import multipletests

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
    fbeta_score,
    accuracy_score,
    ConfusionMatrixDisplay,
)

import xgboost as xgb
import lightgbm as lgb
from imblearn.under_sampling import RandomUnderSampler

BASE = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE, "diabetes_012_health_indicators_BRFSS2015.csv")
FIG_DIR = os.path.join(BASE, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

# ── Optimization goals (plan: screening → emphasize recall via F_beta, beta>1) ──
THRESHOLD_BETA = 2.0  # F_beta with beta=2 weights recall more than precision
VAL_SIZE = 0.25  # fraction of the 80% train used for threshold tuning
RANDOM_STATE = 42
UNDERSAMPLE_RATIO = 1.5  # majority size = minority_n * this (moderate undersampling; no SMOTE)
TUNE_N_ITER = 15
TUNE_CV = 3
TUNE_SCORING = "average_precision"

sns.set_style("whitegrid")
plt.rcParams.update({"figure.dpi": 150, "savefig.bbox": "tight"})


def undersample_majority(x_arr, y_arr, ratio=UNDERSAMPLE_RATIO, random_state=RANDOM_STATE):
    counts = pd.Series(y_arr).value_counts()
    minority_n = int(counts.loc[1])
    majority_target = int(minority_n * ratio)
    rus = RandomUnderSampler(
        sampling_strategy={0: majority_target, 1: minority_n},
        random_state=random_state,
    )
    return rus.fit_resample(x_arr, y_arr)


def best_threshold_fbeta(y_true, y_prob, beta=THRESHOLD_BETA):
    """Pick probability threshold on validation data (no test leakage)."""
    best_t, best_f = 0.5, -1.0
    for t in np.linspace(0.0, 1.0, 501):
        y_pred = (y_prob >= t).astype(int)
        f = fbeta_score(y_true, y_pred, beta=beta, pos_label=1, zero_division=0)
        if f > best_f:
            best_f = f
            best_t = t
    return best_t, best_f


def build_threshold_curve_df(y_true, y_prob, n_points=501):
    """Validation-only threshold sweep: recall, precision, FPR among normals, F1, F2."""
    thresholds = np.linspace(0.0, 1.0, n_points)
    rows = []
    y_true = np.asarray(y_true)
    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        denom = fp + tn
        fpr = fp / denom if denom > 0 else 0.0
        rows.append(
            {
                "threshold": float(t),
                "recall_abnormal": recall_score(y_true, y_pred, pos_label=1, zero_division=0),
                "precision_abnormal": precision_score(y_true, y_pred, pos_label=1, zero_division=0),
                "fpr_normal": fpr,
                "f1": f1_score(y_true, y_pred, pos_label=1, zero_division=0),
                "f2": fbeta_score(y_true, y_pred, beta=2.0, pos_label=1, zero_division=0),
            }
        )
    return pd.DataFrame(rows)


def operating_points_from_curve(curve_df, fpr_cap=0.25):
    """
    Three policies on the same validation curve:
    - F1-balanced: maximize F1
    - F2-screening: maximize F2 (beta=2)
    - Resource-constrained: among thresholds with FPR_normal <= fpr_cap, maximize recall;
      if none exist, choose threshold with minimum FPR (most conservative).
    """
    r_f1 = curve_df.loc[curve_df["f1"].idxmax()].copy()
    r_f2 = curve_df.loc[curve_df["f2"].idxmax()].copy()
    sub = curve_df[curve_df["fpr_normal"] <= fpr_cap]
    if len(sub) > 0:
        r_cap = sub.loc[sub["recall_abnormal"].idxmax()].copy()
        cap_label = f"Resource (FPR≤{int(fpr_cap * 100)}%)"
    else:
        r_cap = curve_df.loc[curve_df["fpr_normal"].idxmin()].copy()
        cap_label = "Resource (min FPR; no point ≤ cap)"
    return [
        ("F1-balanced", r_f1),
        ("F2-screening", r_f2),
        (cap_label, r_cap),
    ]


def evaluate_at_threshold(
    name,
    model,
    x_train_bal,
    y_train_bal,
    x_val,
    y_val,
    x_test,
    y_test,
    curve_file_slug,
):
    model.fit(x_train_bal, y_train_bal)

    if hasattr(model, "predict_proba"):
        y_val_prob = model.predict_proba(x_val)[:, 1]
        y_test_prob = model.predict_proba(x_test)[:, 1]
    else:
        y_val_prob = model.decision_function(x_val)
        y_val_prob = (y_val_prob - y_val_prob.min()) / (y_val_prob.max() - y_val_prob.min() + 1e-12)
        y_test_prob = model.decision_function(x_test)
        y_test_prob = (y_test_prob - y_test_prob.min()) / (y_test_prob.max() - y_test_prob.min() + 1e-12)

    # Validation-only threshold–performance curve (for reporting; no test leakage)
    curve_df = build_threshold_curve_df(y_val, y_val_prob)
    curve_df.to_csv(os.path.join(FIG_DIR, f"threshold_curve_val_{curve_file_slug}.csv"), index=False)

    threshold, val_fbeta = best_threshold_fbeta(y_val, y_val_prob, beta=THRESHOLD_BETA)
    y_pred = (y_test_prob >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    fpr_normal = fp / (fp + tn)

    metrics = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision_abnormal": precision_score(y_test, y_pred, pos_label=1, zero_division=0),
        "Recall_abnormal": recall_score(y_test, y_pred, pos_label=1, zero_division=0),
        "F1_abnormal": f1_score(y_test, y_pred, pos_label=1, zero_division=0),
        "F2_abnormal": fbeta_score(y_test, y_pred, beta=2.0, pos_label=1, zero_division=0),
        "ROC_AUC": roc_auc_score(y_test, y_test_prob),
        "PR_AUC": average_precision_score(y_test, y_test_prob),
        "FPR_normal": fpr_normal,
    }

    print(f"\n--- {name} ---")
    print(f"  Threshold (val F{THRESHOLD_BETA:.0f}): {threshold:.4f} | val F{THRESHOLD_BETA:.0f}={val_fbeta:.4f}")
    print(pd.Series(metrics).round(4).to_string())
    print(classification_report(y_test, y_pred, target_names=["Normal", "Abnormal"], zero_division=0))

    ops_records = []
    display_name = name.replace(" (Binary)", "")
    for policy_label, row in operating_points_from_curve(curve_df):
        ops_records.append(
            {
                "model": display_name,
                "policy": policy_label,
                "threshold": row["threshold"],
                "recall_abnormal_val": row["recall_abnormal"],
                "precision_abnormal_val": row["precision_abnormal"],
                "fpr_normal_val": row["fpr_normal"],
                "f1_val": row["f1"],
                "f2_val": row["f2"],
            }
        )

    return model, y_pred, metrics, float(threshold), float(val_fbeta), curve_df, ops_records


def random_search_tuned_model(name, base_estimator, param_distributions, x_bal, y_bal):
    search = RandomizedSearchCV(
        base_estimator,
        param_distributions=param_distributions,
        n_iter=TUNE_N_ITER,
        cv=TUNE_CV,
        scoring=TUNE_SCORING,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        refit=True,
    )
    search.fit(x_bal, y_bal)
    print(f"  [{name}] best {TUNE_SCORING}={search.best_score_:.4f} params={search.best_params_}")
    return search.best_estimator_, search.best_params_


def run_binary_classification(df):
    y = (df["Diabetes_012"] != 0).astype(int)
    x = df.drop(columns=["Diabetes_012"])

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    scaler = StandardScaler()
    x_train_sc = scaler.fit_transform(x_train)
    x_test_sc = scaler.transform(x_test)

    # Inner split: train_fit vs val (val keeps natural prevalence for threshold tuning)
    x_trf, x_val, y_trf, y_val = train_test_split(
        x_train_sc, y_train, test_size=VAL_SIZE, random_state=RANDOM_STATE, stratify=y_train
    )

    x_trf_bal, y_trf_bal = undersample_majority(x_trf, y_trf)
    print("\nBinary training distribution after undersampling (train_fit only):")
    print(pd.Series(y_trf_bal).value_counts().sort_index())

    # scale_pos_weight for XGBoost: match undersampled majority:minority ratio
    n_neg = int(np.sum(y_trf_bal == 0))
    n_pos = int(np.sum(y_trf_bal == 1))
    xgb_spw = n_neg / max(n_pos, 1)

    threshold_records = []
    best_params_all = {}
    multi_ops_all = []
    curves_for_plot = []

    # --- Logistic Regression (tune C) ---
    lr_base = LogisticRegression(
        max_iter=3000,
        random_state=RANDOM_STATE,
        class_weight="balanced",
        solver="lbfgs",
    )
    lr_model, lr_bp = random_search_tuned_model(
        "Logistic Regression",
        lr_base,
        {"C": loguniform(1e-3, 1e2)},
        x_trf_bal,
        y_trf_bal,
    )
    best_params_all["Logistic Regression"] = lr_bp
    lr_model, lr_pred, lr_metrics, lr_t, lr_vf, lr_curve, lr_ops = evaluate_at_threshold(
        "Logistic Regression (Binary)",
        lr_model,
        x_trf_bal,
        y_trf_bal,
        x_val,
        y_val,
        x_test_sc,
        y_test,
        "LR",
    )
    threshold_records.append({"model": "Logistic Regression", "threshold": lr_t, "val_fbeta": lr_vf})
    multi_ops_all.extend(lr_ops)
    curves_for_plot.append(("Logistic Regression", lr_curve))

    # --- Random Forest (tune structure) ---
    rf_base = RandomForestClassifier(
        random_state=RANDOM_STATE,
        n_jobs=-1,
        class_weight="balanced",
    )
    rf_model, rf_bp = random_search_tuned_model(
        "Random Forest",
        rf_base,
        {
            "n_estimators": sp_randint(150, 400),
            "max_depth": sp_randint(8, 24),
            "min_samples_leaf": sp_randint(1, 8),
        },
        x_trf_bal,
        y_trf_bal,
    )
    best_params_all["Random Forest"] = rf_bp
    rf_model, rf_pred, rf_metrics, rf_t, rf_vf, rf_curve, rf_ops = evaluate_at_threshold(
        "Random Forest (Binary)",
        rf_model,
        x_trf_bal,
        y_trf_bal,
        x_val,
        y_val,
        x_test_sc,
        y_test,
        "RF",
    )
    threshold_records.append({"model": "Random Forest", "threshold": rf_t, "val_fbeta": rf_vf})
    multi_ops_all.extend(rf_ops)
    curves_for_plot.append(("Random Forest", rf_curve))

    # --- XGBoost ---
    xgb_base = xgb.XGBClassifier(
        random_state=RANDOM_STATE,
        n_jobs=-1,
        eval_metric="logloss",
        scale_pos_weight=xgb_spw,
    )
    xgb_model, xgb_bp = random_search_tuned_model(
        "XGBoost",
        xgb_base,
        {
            "n_estimators": sp_randint(150, 350),
            "max_depth": sp_randint(4, 12),
            "learning_rate": sp_uniform(0.05, 0.15),
            "subsample": sp_uniform(0.65, 0.35),
            "colsample_bytree": sp_uniform(0.65, 0.35),
            "min_child_weight": sp_randint(1, 10),
        },
        x_trf_bal,
        y_trf_bal,
    )
    best_params_all["XGBoost"] = xgb_bp
    xgb_model, xgb_pred, xgb_metrics, xgb_t, xgb_vf, xgb_curve, xgb_ops = evaluate_at_threshold(
        "XGBoost (Binary)",
        xgb_model,
        x_trf_bal,
        y_trf_bal,
        x_val,
        y_val,
        x_test_sc,
        y_test,
        "XGB",
    )
    threshold_records.append({"model": "XGBoost", "threshold": xgb_t, "val_fbeta": xgb_vf})
    multi_ops_all.extend(xgb_ops)
    curves_for_plot.append(("XGBoost", xgb_curve))

    # --- LightGBM ---
    lgb_base = lgb.LGBMClassifier(
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=-1,
        class_weight="balanced",
    )
    lgb_model, lgb_bp = random_search_tuned_model(
        "LightGBM",
        lgb_base,
        {
            "n_estimators": sp_randint(150, 350),
            "max_depth": sp_randint(4, 12),
            "learning_rate": sp_uniform(0.05, 0.15),
            "subsample": sp_uniform(0.65, 0.35),
            "colsample_bytree": sp_uniform(0.65, 0.35),
            "min_child_samples": sp_randint(5, 40),
        },
        x_trf_bal,
        y_trf_bal,
    )
    best_params_all["LightGBM"] = lgb_bp
    lgb_model, lgb_pred, lgb_metrics, lgb_t, lgb_vf, lgb_curve, lgb_ops = evaluate_at_threshold(
        "LightGBM (Binary)",
        lgb_model,
        x_trf_bal,
        y_trf_bal,
        x_val,
        y_val,
        x_test_sc,
        y_test,
        "LGB",
    )
    threshold_records.append({"model": "LightGBM", "threshold": lgb_t, "val_fbeta": lgb_vf})
    multi_ops_all.extend(lgb_ops)
    curves_for_plot.append(("LightGBM", lgb_curve))

    results = {
        "Logistic Regression": lr_metrics,
        "Random Forest": rf_metrics,
        "XGBoost": xgb_metrics,
        "LightGBM": lgb_metrics,
    }

    res_df = pd.DataFrame(results).T.round(4)
    print("\n=== Binary Model Summary (test set; threshold from val F2) ===")
    print(res_df.to_string())
    res_df.to_csv(os.path.join(FIG_DIR, "binary_model_comparison.csv"), index_label="Model")

    thr_df = pd.DataFrame(threshold_records)
    thr_df.to_csv(os.path.join(FIG_DIR, "binary_model_thresholds.csv"), index=False)

    with open(os.path.join(FIG_DIR, "binary_model_best_params.json"), "w", encoding="utf-8") as f:
        json.dump(best_params_all, f, indent=2, default=str)

    pd.DataFrame(multi_ops_all).to_csv(
        os.path.join(FIG_DIR, "multi_operating_points_validation.csv"), index=False
    )

    long_curve_parts = []
    for name, cdf in curves_for_plot:
        c = cdf.copy()
        c.insert(0, "model", name)
        long_curve_parts.append(c)
    pd.concat(long_curve_parts, ignore_index=True).to_csv(
        os.path.join(FIG_DIR, "threshold_curves_validation_long.csv"), index=False
    )

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    for ax, (name, cdf) in zip(np.ravel(axes), curves_for_plot):
        ax.plot(cdf["threshold"], cdf["recall_abnormal"], label="Recall (abnormal)", linewidth=1.8)
        ax.plot(cdf["threshold"], cdf["precision_abnormal"], label="Precision (abnormal)", linewidth=1.8)
        ax.plot(cdf["threshold"], cdf["fpr_normal"], label="FPR (normal)", linewidth=1.8)
        ax.set_xlabel("Probability threshold")
        ax.set_ylabel("Metric value")
        ax.set_ylim(0, 1.05)
        ax.set_title(name, fontsize=11, fontweight="bold")
        ax.legend(loc="best", fontsize=7)
    plt.suptitle(
        "Validation-set threshold–performance curves (no test-set tuning)",
        fontsize=13,
        fontweight="bold",
        y=1.02,
    )
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "fig10_threshold_performance_curves_validation.png"))
    plt.close()

    metrics_plot = [
        "Accuracy",
        "Precision_abnormal",
        "Recall_abnormal",
        "F1_abnormal",
        "F2_abnormal",
        "ROC_AUC",
        "PR_AUC",
    ]
    fig, ax = plt.subplots(figsize=(12, 6))
    idx = np.arange(len(res_df.index))
    width = 0.1
    for i, metric in enumerate(metrics_plot):
        vals = res_df[metric].values
        ax.bar(idx + i * width, vals, width, label=metric)
    ax.set_xticks(idx + width * 3.0)
    ax.set_xticklabels(res_df.index, fontsize=10)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title(
        "Binary Model Performance (test; thresholds tuned on val for F2)",
        fontsize=13,
        fontweight="bold",
    )
    ax.legend(loc="lower right", fontsize=7)
    plt.savefig(os.path.join(FIG_DIR, "fig8_binary_model_comparison.png"))
    plt.close()

    fig, axes = plt.subplots(1, 4, figsize=(21, 5))
    for ax, (name, pred) in zip(
        axes,
        [
            ("Logistic Regression", lr_pred),
            ("Random Forest", rf_pred),
            ("XGBoost", xgb_pred),
            ("LightGBM", lgb_pred),
        ],
    ):
        cm = confusion_matrix(y_test, pred)
        ConfusionMatrixDisplay(cm, display_labels=["Normal", "Abnormal"]).plot(ax=ax, cmap="Blues", colorbar=False)
        ax.set_title(name, fontsize=10, fontweight="bold")
    plt.suptitle("Binary Confusion Matrices (test set)", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "fig9_binary_confusion_matrices.png"))
    plt.close()

    return res_df


def run_univariable_analysis(df):
    df_u = df.copy()
    df_u["abnormal"] = (df_u["Diabetes_012"] != 0).astype(int)

    binary_vars = [
        "HighBP",
        "HighChol",
        "CholCheck",
        "Smoker",
        "Stroke",
        "HeartDiseaseorAttack",
        "PhysActivity",
        "Fruits",
        "Veggies",
        "HvyAlcoholConsump",
        "AnyHealthcare",
        "NoDocbcCost",
        "DiffWalk",
        "Sex",
    ]
    cont_ord_vars = ["BMI", "GenHlth", "MentHlth", "PhysHlth", "Age", "Education", "Income"]

    rows = []

    for var in binary_vars:
        tab = pd.crosstab(df_u[var], df_u["abnormal"])
        for r in [0.0, 1.0]:
            if r not in tab.index:
                tab.loc[r] = [0, 0]
        for c in [0, 1]:
            if c not in tab.columns:
                tab[c] = 0
        tab = tab.sort_index().sort_index(axis=1)

        chi2, p, _, _ = chi2_contingency(tab.values)

        a = tab.loc[1.0, 1] + 0.5
        b = tab.loc[1.0, 0] + 0.5
        c = tab.loc[0.0, 1] + 0.5
        d = tab.loc[0.0, 0] + 0.5
        or_val = (a * d) / (b * c)

        p_norm = (df_u.loc[df_u["abnormal"] == 0, var] == 1).mean() * 100
        p_abn = (df_u.loc[df_u["abnormal"] == 1, var] == 1).mean() * 100

        rows.append(
            {
                "Variable": var,
                "Normal": f"{p_norm:.1f}%",
                "Prediabetes-Diabetes": f"{p_abn:.1f}%",
                "Test": "Chi-square",
                "p_value_raw": p,
                "Effect_Size": f"OR={or_val:.2f}",
            }
        )

    for var in cont_ord_vars:
        x_norm = df_u.loc[df_u["abnormal"] == 0, var].dropna()
        x_abn = df_u.loc[df_u["abnormal"] == 1, var].dropna()

        _, p = mannwhitneyu(x_norm, x_abn, alternative="two-sided")

        n_med, n_q1, n_q3 = x_norm.median(), x_norm.quantile(0.25), x_norm.quantile(0.75)
        a_med, a_q1, a_q3 = x_abn.median(), x_abn.quantile(0.25), x_abn.quantile(0.75)

        rows.append(
            {
                "Variable": var,
                "Normal": f"Median {n_med:.2f} ({n_q1:.2f}-{n_q3:.2f})",
                "Prediabetes-Diabetes": f"Median {a_med:.2f} ({a_q1:.2f}-{a_q3:.2f})",
                "Test": "Mann-Whitney U",
                "p_value_raw": p,
                "Effect_Size": f"Delta median={a_med - n_med:.2f}",
            }
        )

    uni = pd.DataFrame(rows)
    uni["p_adj_fdr"] = multipletests(uni["p_value_raw"], method="fdr_bh")[1]

    def fmt_p(x):
        return "<0.001" if x < 0.001 else f"{x:.3f}"

    uni["p_value"] = uni["p_value_raw"].map(fmt_p)
    uni["p_adj_fdr_fmt"] = uni["p_adj_fdr"].map(fmt_p)

    uni = uni.sort_values(["p_adj_fdr", "Variable"]).reset_index(drop=True)

    out_cols = ["Variable", "Normal", "Prediabetes-Diabetes", "Test", "p_value", "p_adj_fdr_fmt", "Effect_Size"]
    uni_out = uni[out_cols].rename(columns={"p_adj_fdr_fmt": "p_adj_fdr"})

    print("\n=== Univariable Analysis: Normal vs Prediabetes-Diabetes ===")
    print(uni_out.to_string(index=False))

    uni_out.to_csv(os.path.join(FIG_DIR, "univariable_normal_vs_abnormal.csv"), index=False)

    return uni_out


def main():
    df = pd.read_csv(DATA_PATH)
    print(f"Dataset shape: {df.shape}")

    binary_summary = run_binary_classification(df)
    uni_table = run_univariable_analysis(df)

    print("\nSaved outputs:")
    print(os.path.join(FIG_DIR, "binary_model_comparison.csv"))
    print(os.path.join(FIG_DIR, "binary_model_thresholds.csv"))
    print(os.path.join(FIG_DIR, "multi_operating_points_validation.csv"))
    print(os.path.join(FIG_DIR, "threshold_curves_validation_long.csv"))
    print(os.path.join(FIG_DIR, "threshold_curve_val_*.csv"))
    print(os.path.join(FIG_DIR, "binary_model_best_params.json"))
    print(os.path.join(FIG_DIR, "fig8_binary_model_comparison.png"))
    print(os.path.join(FIG_DIR, "fig9_binary_confusion_matrices.png"))
    print(os.path.join(FIG_DIR, "fig10_threshold_performance_curves_validation.png"))
    print(os.path.join(FIG_DIR, "univariable_normal_vs_abnormal.csv"))


if __name__ == "__main__":
    main()
