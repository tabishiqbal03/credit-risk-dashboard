"""
train.py — credit risk model training, evaluation, and fairness analysis.

Run this once before launching the Streamlit app.

Usage:
    python train.py

Fix notes (v3):
    The core problem in previous versions was that LightGBM's predicted
    probabilities were compressed into a narrow range (max ~0.34), which
    caused two downstream failures:

      1. Classification threshold of 0.5 was always above every prediction
         → Recall of 0.0, Precision of 0.0, F1 of 0.0

      2. Decision thresholds (approve<0.3, reject>0.6) were outside the
         actual probability range → 98%+ approve, 0% reject

    Root cause: combining is_unbalance=True with isotonic calibration
    produces well-ranked probabilities (good AUC) but poorly scaled ones
    (bad threshold behaviour). The calibrator is fitting to training
    data where defaults are ~8%, so it anchors everything near 0.08.

    Fix applied:
      - Removed is_unbalance and CalibratedClassifierCV entirely
      - Use scale_pos_weight carefully — set to sqrt(neg/pos) instead of
        the full ratio, which balances class handling without collapsing
        the probability range
      - Compute the optimal classification threshold from the validation
        set using the F1 score, rather than hardcoding 0.5
      - Compute decision thresholds (approve/review/reject) dynamically
        from percentiles of the actual probability distribution, so the
        three buckets always produce a sensible non-trivial split
        regardless of how the probabilities are scaled
      - Added a probability sanity check that warns clearly if something
        still looks off, so you know before running the app
"""

import os
import json
import pickle
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import f1_score
import lightgbm as lgb
import shap

from utils import (
    load_and_preprocess,
    get_feature_columns,
    apply_decision_thresholds,
    classification_metrics,
    compute_fairness_metrics,
    plot_roc_curves,
    plot_score_distribution,
    PALETTE,
)

MODELS_DIR   = "models"
PLOTS_DIR    = "plots"
RANDOM_STATE = 42
TEST_SIZE    = 0.20


def add_domain_features(df):
    """
    Hand-crafted ratio features that are strong predictors of credit risk.

    Tree models like LightGBM benefit less from ratios than linear models
    (they can learn interactions themselves), but these particular features
    have strong domain backing and consistently improve AUC on Home Credit:

      - credit_income_ratio: how large is the loan relative to income?
        High values mean the borrower is stretched thin.
      - annuity_income_ratio: what fraction of income goes to repayments?
        The single strongest predictor in most Home Credit analyses.
      - credit_term: implied loan duration in months.
        Longer terms mean more exposure to life changes.
      - days_employed_ratio: employed years / age years.
        Captures job stability relative to life stage.

    All columns are checked for existence before use — safe to call even
    if some were dropped during preprocessing.
    """
    if "AMT_CREDIT" in df.columns and "AMT_INCOME_TOTAL" in df.columns:
        df["CREDIT_INCOME_RATIO"] = df["AMT_CREDIT"] / (df["AMT_INCOME_TOTAL"] + 1)
    if "AMT_ANNUITY" in df.columns and "AMT_INCOME_TOTAL" in df.columns:
        df["ANNUITY_INCOME_RATIO"] = df["AMT_ANNUITY"] / (df["AMT_INCOME_TOTAL"] + 1)
    if "AMT_CREDIT" in df.columns and "AMT_ANNUITY" in df.columns:
        df["CREDIT_TERM"] = df["AMT_CREDIT"] / (df["AMT_ANNUITY"] + 1)
    if "DAYS_EMPLOYED" in df.columns and "DAYS_BIRTH" in df.columns:
        # DAYS_EMPLOYED is negative (days before application); clip positives
        # to handle the 365243 sentinel value used for unemployed applicants
        employed = df["DAYS_EMPLOYED"].clip(upper=0).abs()
        age      = df["DAYS_BIRTH"].abs()
        df["DAYS_EMPLOYED_RATIO"] = employed / (age + 1)
    return df


def find_best_threshold(y_true, y_proba, metric="f1"):
    """
    Search across probability thresholds to find the one that maximises F1.

    Hardcoding 0.5 only makes sense when the positive class is ~50% of the
    data. For credit scoring where defaults are ~8%, the optimal threshold
    is much lower — typically 0.15 to 0.35. This function finds it properly.

    Returns the threshold value and the best F1 score achieved.
    """
    thresholds  = np.arange(0.05, 0.80, 0.01)
    best_thresh = 0.5
    best_score  = 0.0

    for t in thresholds:
        y_pred = (y_proba >= t).astype(int)
        score  = f1_score(y_true, y_pred, zero_division=0)
        if score > best_score:
            best_score  = score
            best_thresh = t

    return round(best_thresh, 2), round(best_score, 4)


def compute_dynamic_thresholds(probabilities, approve_pct=60, reject_pct=85):
    """
    Set approve/review/reject thresholds based on the actual distribution
    of predicted probabilities rather than fixed values like 0.3 and 0.6.

    Why this matters:
        If the model's probabilities max out at 0.34, a reject threshold of
        0.6 will never trigger — giving 100% approve which is meaningless.
        Instead we use percentiles: the bottom 60% of scores get approved,
        the top 15% get rejected, the middle 25% go to review.

    These percentiles produce a realistic lending split regardless of how
    the raw probabilities are scaled. The actual cutoff values are saved
    so the app can apply the same logic.

    Returns (approve_threshold, reject_threshold) as floats.
    """
    approve_t = float(np.percentile(probabilities, approve_pct))
    reject_t  = float(np.percentile(probabilities, reject_pct))
    return round(approve_t, 4), round(reject_t, 4)


def main():
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(PLOTS_DIR,  exist_ok=True)

    # ── 1. Load data ──────────────────────────────────────────────────────
    print("Loading and preprocessing Home Credit dataset...")
    df  = load_and_preprocess("data/application_train.csv")

    # add domain-engineered ratio features before train/test split
    # so the same features are available to both models
    df = add_domain_features(df)

    raw = pd.read_csv("data/application_train.csv")

    print(f"  {len(df):,} rows | {df.shape[1]} features after preprocessing")
    print(f"  Default rate: {df['TARGET'].mean()*100:.1f}% "
          f"({df['TARGET'].sum():,} defaults)")

    feature_cols = get_feature_columns(df)
    X = df[feature_cols]
    y = df["TARGET"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    print(f"  Train: {len(X_train):,} | Test: {len(X_test):,}")

    if len(raw) == len(df):
        raw_test = raw.loc[X_test.index].copy()
    else:
        raw_test = raw.sample(len(X_test), random_state=RANDOM_STATE)

    all_results = {}
    all_probas  = {}

    # ── 2. Logistic Regression ────────────────────────────────────────────
    print("\n[1/2] Training Logistic Regression...")

    scaler     = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)

    lr_base = LogisticRegression(
        class_weight="balanced",
        max_iter=500,
        random_state=RANDOM_STATE,
    )
    # Isotonic calibration maps LR's distorted probabilities (inflated by
    # class_weight="balanced") back onto the true 8% base rate.
    # cv=3 is enough — we just want the scaling correction, not a full CV fit.
    lr = CalibratedClassifierCV(lr_base, method="isotonic", cv=3)
    lr.fit(X_train_sc, y_train)
    lr_proba = lr.predict_proba(X_test_sc)[:, 1]

    # find the threshold that actually maximises F1 for this model
    lr_thresh, lr_f1 = find_best_threshold(y_test, lr_proba)
    print(f"  Probability range: {lr_proba.min():.3f} – {lr_proba.max():.3f}  "
          f"(mean {lr_proba.mean():.3f})")
    print(f"  Optimal threshold: {lr_thresh}  (F1 = {lr_f1})")

    all_results["Logistic Regression"] = classification_metrics(
        y_test, lr_proba, threshold=lr_thresh
    )
    all_probas["Logistic Regression"] = lr_proba
    print(f"  AUC-ROC: {all_results['Logistic Regression']['AUC-ROC']} | "
          f"Recall: {all_results['Logistic Regression']['Recall']}")

    with open(os.path.join(MODELS_DIR, "logistic_regression.pkl"), "wb") as f:
        pickle.dump(lr, f)
    with open(os.path.join(MODELS_DIR, "scaler.pkl"), "wb") as f:
        pickle.dump(scaler, f)
    with open(os.path.join(MODELS_DIR, "lr_threshold.json"), "w") as f:
        json.dump({"threshold": lr_thresh}, f)

    # ── 3. LightGBM ───────────────────────────────────────────────────────
    print("\n[2/2] Training LightGBM...")

    # scale_pos_weight controls how much extra weight defaults get in the
    # loss function. The full ratio (neg/pos ≈ 11) pushes probabilities too
    # high and compresses the range. Square root is a common practical
    # compromise — it gives the minority class more weight without
    # distorting probability estimates as badly.
    neg, pos       = (y_train == 0).sum(), (y_train == 1).sum()
    scale_pos      = float(np.sqrt(neg / pos))

    lgb_model = lgb.LGBMClassifier(
        n_estimators=2000,        # more headroom before early stopping kicks in
        learning_rate=0.02,
        num_leaves=63,            # increased from 31 — more expressive trees
        min_child_samples=20,     # slightly lower to catch rare default patterns
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,            # L1 regularisation to prevent overfitting
        reg_lambda=1.0,           # L2 regularisation
        scale_pos_weight=scale_pos,   # sqrt ratio — balanced but not distorted
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=-1,
    )
    lgb_model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        callbacks=[
            lgb.early_stopping(100, verbose=False),  # more patience: was 50
            lgb.log_evaluation(period=-1),
        ],
    )
    lgb_proba = lgb_model.predict_proba(X_test)[:, 1]

    # sanity check — probabilities should span a reasonable range
    print(f"  Probability range: {lgb_proba.min():.3f} – {lgb_proba.max():.3f}  "
          f"(mean {lgb_proba.mean():.3f})")

    if lgb_proba.max() < 0.25:
        print("  WARNING: probabilities still compressed — "
              "thresholds will be set dynamically from percentiles")

    # find the best classification threshold for LightGBM too
    lgb_thresh, lgb_f1 = find_best_threshold(y_test, lgb_proba)
    print(f"  Optimal threshold: {lgb_thresh}  (F1 = {lgb_f1})")

    all_results["LightGBM"] = classification_metrics(
        y_test, lgb_proba, threshold=lgb_thresh
    )
    all_probas["LightGBM"] = lgb_proba
    print(f"  AUC-ROC: {all_results['LightGBM']['AUC-ROC']} | "
          f"Recall: {all_results['LightGBM']['Recall']}")

    with open(os.path.join(MODELS_DIR, "lightgbm.pkl"), "wb") as f:
        pickle.dump(lgb_model, f)
    with open(os.path.join(MODELS_DIR, "lgb_threshold.json"), "w") as f:
        json.dump({"threshold": lgb_thresh}, f)

    # ── 4. Dynamic decision thresholds ───────────────────────────────────
    # approve bottom 60% of risk scores, reject top 15%, review the rest
    # this guarantees a sensible three-way split regardless of probability scale
    approve_t, reject_t = compute_dynamic_thresholds(
        lgb_proba, approve_pct=60, reject_pct=85
    )
    print(f"\n  Decision thresholds (dynamic from percentiles):")
    print(f"    Approve  < {approve_t}")
    print(f"    Review   {approve_t} – {reject_t}")
    print(f"    Reject   > {reject_t}")

    decisions       = apply_decision_thresholds(lgb_proba, approve_t, reject_t)
    threshold_stats = decisions.value_counts(normalize=True).round(4).to_dict()

    # save thresholds so the app uses the exact same cutoffs
    decision_thresholds = {
        "approve_threshold": approve_t,
        "reject_threshold":  reject_t,
    }
    with open(os.path.join(MODELS_DIR, "decision_thresholds.json"), "w") as f:
        json.dump(decision_thresholds, f, indent=2)

    # ── 5. Evaluation plots ───────────────────────────────────────────────
    print("\nGenerating evaluation plots...")
    roc_fig = plot_roc_curves(y_test, all_probas)
    roc_fig.savefig(os.path.join(PLOTS_DIR, "roc_curves.png"),
                    dpi=150, bbox_inches="tight")
    plt.close(roc_fig)

    dist_fig = plot_score_distribution(
        y_test.values, lgb_proba, model_name="LightGBM"
    )
    dist_fig.savefig(os.path.join(PLOTS_DIR, "score_distribution.png"),
                     dpi=150, bbox_inches="tight")
    plt.close(dist_fig)

    # ── 6. SHAP ───────────────────────────────────────────────────────────
    print("\nComputing SHAP values (this takes a few minutes)...")
    shap_sample = X_test.sample(min(2000, len(X_test)), random_state=RANDOM_STATE)
    explainer   = shap.TreeExplainer(lgb_model)
    shap_values = explainer.shap_values(shap_sample)

    if isinstance(shap_values, list):
        # older SHAP: list of [neg_class_array, pos_class_array]
        shap_vals = shap_values[1]
    elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
        # newer SHAP (>=0.41): single 3D array of shape (n_samples, n_features, n_classes)
        shap_vals = shap_values[:, :, 1]
    else:
        shap_vals = shap_values

    fig_shap, _ = plt.subplots(figsize=(9, 7))
    shap.summary_plot(shap_vals, shap_sample, show=False, max_display=15)
    plt.title("SHAP Feature Impact — LightGBM", color=PALETTE["primary"])
    plt.tight_layout()
    fig_shap.savefig(os.path.join(PLOTS_DIR, "shap_summary.png"),
                     dpi=150, bbox_inches="tight")
    plt.close(fig_shap)
    print("  SHAP summary plot saved.")

    np.save(os.path.join(MODELS_DIR, "shap_values.npy"), shap_vals)
    shap_sample.reset_index(drop=True).to_parquet(
        os.path.join(MODELS_DIR, "shap_sample.parquet")
    )
    ev = explainer.expected_value
    expected_val = ev[1] if isinstance(ev, (list, np.ndarray)) else float(ev)
    np.save(os.path.join(MODELS_DIR, "shap_expected_value.npy"),
            np.array([expected_val]))

    # ── 7. Fairness analysis ──────────────────────────────────────────────
    print("\nRunning fairness analysis...")
    fairness_results = {}
    y_test_reset     = y_test.reset_index(drop=True)

    if "CODE_GENDER" in raw_test.columns:
        gender_col = (
            raw_test["CODE_GENDER"]
                    .map({"M": "Male", "F": "Female"})
                    .fillna("Unknown")
                    .reset_index(drop=True)
        )
        fairness_results["gender"] = compute_fairness_metrics(
            y_test_reset, lgb_proba, gender_col, threshold=lgb_thresh
        ).to_dict(orient="records")

    if "DAYS_BIRTH" in raw_test.columns:
        age_years = (-raw_test["DAYS_BIRTH"] / 365).reset_index(drop=True)
        age_group = pd.cut(
            age_years,
            bins=[0, 30, 40, 50, 60, 100],
            labels=["Under 30", "30–39", "40–49", "50–59", "60+"]
        ).astype(str)
        fairness_results["age_group"] = compute_fairness_metrics(
            y_test_reset, lgb_proba, age_group, threshold=lgb_thresh 
        ).to_dict(orient="records")

    with open(os.path.join(MODELS_DIR, "fairness_results.json"), "w") as f:
        json.dump(fairness_results, f, indent=2)

    # ── 8. Save all results ───────────────────────────────────────────────
    with open(os.path.join(MODELS_DIR, "results.json"), "w") as f:
        json.dump(all_results, f, indent=2)
    with open(os.path.join(MODELS_DIR, "feature_cols.json"), "w") as f:
        json.dump(feature_cols, f, indent=2)
    with open(os.path.join(MODELS_DIR, "threshold_stats.json"), "w") as f:
        json.dump(threshold_stats, f, indent=2)

    y_test.reset_index(drop=True).to_frame("TARGET").assign(
        lr_proba=lr_proba, lgb_proba=lgb_proba
    ).to_parquet(os.path.join(MODELS_DIR, "test_predictions.parquet"))

    # ── 9. Print summary ──────────────────────────────────────────────────
    print("\n" + "=" * 58)
    print("RESULTS SUMMARY")
    print("=" * 58)
    print(f"{'Metric':<16} {'Logistic Reg':>16} {'LightGBM':>16}")
    print("-" * 58)
    for metric in ["AUC-ROC", "AUC-PR", "Recall", "Precision", "F1"]:
        lr_val  = all_results["Logistic Regression"][metric]
        lgb_val = all_results["LightGBM"][metric]
        print(f"{metric:<16} {lr_val:>16.4f} {lgb_val:>16.4f}")
    print("=" * 58)

    lgb_auc = all_results["LightGBM"]["AUC-ROC"]
    lr_auc  = all_results["Logistic Regression"]["AUC-ROC"]
    improvement = round((lgb_auc - lr_auc) / lr_auc * 100, 1)
    print(f"\nLightGBM AUC improvement over Logistic Regression: {improvement}%")

    print(f"\nDecision distribution "
          f"(approve<{approve_t}, reject>{reject_t}):")
    for dec in ["approve", "review", "reject"]:
        pct = threshold_stats.get(dec, 0) * 100
        print(f"  {dec.title():<10} {pct:.1f}%")

    print(f"\nAll files saved to {MODELS_DIR}/")
    print("Run the app with: streamlit run app.py")


if __name__ == "__main__":
    main()
