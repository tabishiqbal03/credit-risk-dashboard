"""
utils.py — data preprocessing, evaluation metrics, fairness helpers, and plotting.

All shared logic between train.py and app.py lives here.
The fairness analysis functions are the most important part of this project —
they're what makes it stand out from a standard classification task.
"""

import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import warnings
warnings.filterwarnings("ignore")


# ── Colours ───────────────────────────────────────────────────────────────────

PALETTE = {
    "primary":   "#154360",
    "secondary": "#1a5276",
    "approve":   "#1e8449",
    "review":    "#d68910",
    "reject":    "#922b21",
    "light":     "#d6eaf8",
    "grey":      "#717d7e",
}


# ── Data loading and preprocessing ───────────────────────────────────────────

def clean_column_names(df):
    """
    LightGBM rejects column names that contain special JSON characters
    such as brackets, commas, quotes, or slashes. These sneak in after
    pd.get_dummies() because the original categorical values (e.g.
    'Laborers/cleaners', 'Self-employed (not registered)') become part
    of the column name.

    This function replaces every non-alphanumeric character (except
    underscores) with an underscore, then collapses any runs of multiple
    underscores into one. Safe to call on any DataFrame.
    """
    df.columns = [
        re.sub(r'[^A-Za-z0-9_]', '_', col).strip('_')
        for col in df.columns
    ]
    # collapse consecutive underscores — e.g. NAME__TYPE becomes NAME_TYPE
    df.columns = [re.sub(r'_+', '_', col) for col in df.columns]
    return df


def load_and_preprocess(data_path="data/application_train.csv", sample_size=None):
    """
    Load the Home Credit dataset and do the minimum cleaning needed
    to get it into a trainable state.

    Home Credit is a good real-world dataset because it's genuinely messy —
    lots of missing values, mixed types, and some heavily skewed features.
    Cleaning it properly is part of what we're demonstrating.

    Args:
        data_path:   path to application_train.csv
        sample_size: if set, randomly sample this many rows (useful for quick tests)
    """
    df = pd.read_csv(data_path)

    if sample_size:
        df = df.sample(n=sample_size, random_state=42).reset_index(drop=True)

    # ── Target ───────────────────────────────────────────────────────────
    # TARGET = 1 means the client had payment difficulties (default)
    # We keep the column name as-is since it's the official Kaggle name

    # ── Drop columns with too many missing values ─────────────────────
    # Anything missing in over 40% of rows is more likely to hurt than help
    threshold = 0.40
    missing_rate = df.isnull().mean()
    cols_to_drop = missing_rate[missing_rate > threshold].index.tolist()
    df = df.drop(columns=cols_to_drop)

    # ── Drop ID column (not a feature) ────────────────────────────────
    df = df.drop(columns=["SK_ID_CURR"], errors="ignore")

    # ── Encode binary categoricals ────────────────────────────────────
    # Some yes/no columns are stored as strings
    binary_map = {"Y": 1, "N": 0, "M": 1, "F": 0, "XNA": 0}
    for col in df.select_dtypes(include="object").columns:
        if df[col].nunique() <= 2:
            df[col] = df[col].map(binary_map).fillna(0)

    # ── One-hot encode remaining categoricals ─────────────────────────
    cat_cols = df.select_dtypes(include="object").columns.tolist()
    if cat_cols:
        df = pd.get_dummies(df, columns=cat_cols, drop_first=True, dtype=int)

    # ── Clean column names ────────────────────────────────────────────
    # Must happen after get_dummies — categorical values like
    # 'Laborers/cleaners' become column name suffixes and introduce
    # slashes, brackets, and commas that LightGBM cannot handle.
    df = clean_column_names(df)

    # ── Fill remaining missing values with median ──────────────────────
    # Median is more robust than mean for skewed financial data
    for col in df.columns:
        if df[col].isnull().any():
            df[col] = df[col].fillna(df[col].median())

    # ── Cap extreme outliers ──────────────────────────────────────────
    # Some income/credit columns have extreme values that can destabilise training
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols = [c for c in numeric_cols if c != "TARGET"]
    for col in numeric_cols:
        p99 = df[col].quantile(0.99)
        p01 = df[col].quantile(0.01)
        df[col] = df[col].clip(lower=p01, upper=p99)

    return df


def get_feature_columns(df):
    """Return all columns except the target."""
    return [c for c in df.columns if c != "TARGET"]


# ── Threshold system ──────────────────────────────────────────────────────────

def apply_decision_thresholds(probabilities, approve_threshold=0.30, reject_threshold=0.60):
    """
    Convert raw default probabilities into a three-tier decision.

    The logic:
      - probability < approve_threshold  → APPROVE  (low risk)
      - probability > reject_threshold   → REJECT   (high risk)
      - in between                       → REVIEW   (human review needed)

    In practice, financial institutions always have a manual review lane
    for borderline applications. Predicting binary approve/reject ignores
    this and leads to either too many bad loans or too many false rejections.

    Returns a Series of strings: 'approve', 'review', or 'reject'.
    """
    decisions = pd.Series("review", index=range(len(probabilities)), dtype=str)
    proba_arr = np.asarray(probabilities)  # ensure numpy for boolean indexing
    decisions[proba_arr < approve_threshold] = "approve"
    decisions[proba_arr > reject_threshold]  = "reject"
    return decisions


# ── Evaluation metrics ────────────────────────────────────────────────────────

def classification_metrics(y_true, y_pred_proba, threshold=0.5):
    """
    Compute the metrics most relevant to credit risk:
      - AUC-ROC   : overall discrimination ability
      - AUC-PR    : more meaningful than ROC for imbalanced datasets
      - Recall    : our primary optimisation target — catching actual defaults
      - Precision : how many flagged as default actually defaulted
      - F1        : harmonic mean of precision and recall
    """
    from sklearn.metrics import (
        roc_auc_score, average_precision_score,
        precision_score, recall_score, f1_score,
    )

    y_pred = (y_pred_proba >= threshold).astype(int)

    return {
        "AUC-ROC":   round(roc_auc_score(y_true, y_pred_proba), 4),
        "AUC-PR":    round(average_precision_score(y_true, y_pred_proba), 4),
        "Recall":    round(recall_score(y_true, y_pred), 4),
        "Precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
        "F1":        round(f1_score(y_true, y_pred, zero_division=0), 4),
    }


# ── Fairness analysis ─────────────────────────────────────────────────────────

def compute_fairness_metrics(y_true, y_pred_proba, sensitive_col, threshold=0.5):
    """
    Compute demographic parity and equalised odds across groups defined
    by a sensitive attribute (e.g. gender, age group).

    Demographic parity: do different groups get approved at the same rate?
    Equalised odds:     do different groups have the same TPR and FPR?

    These are the standard fairness criteria used in credit lending research
    and referenced in EU AI Act guidance for high-risk AI systems.

    Returns a DataFrame with one row per group.
    """
    y_pred = (y_pred_proba >= threshold).astype(int)
    groups = sensitive_col.unique()
    rows   = []

    for group in groups:
        mask = sensitive_col == group
        if mask.sum() < 10:
            continue

        yt    = y_true[mask]
        yp    = y_pred[mask]
        yprob = y_pred_proba[mask]

        # approval rate = predicted non-default rate
        approval_rate = (yp == 0).mean()

        # true positive rate (recall on the default class)
        tp  = ((yp == 1) & (yt == 1)).sum()
        fn  = ((yp == 0) & (yt == 1)).sum()
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0

        # false positive rate (non-defaults incorrectly flagged as default)
        fp  = ((yp == 1) & (yt == 0)).sum()
        tn  = ((yp == 0) & (yt == 0)).sum()
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0

        rows.append({
            "Group":          str(group),
            "N":              int(mask.sum()),
            "Default Rate":   round(float(yt.mean()), 4),
            "Approval Rate":  round(float(approval_rate), 4),
            "TPR (Recall)":   round(float(tpr), 4),
            "FPR":            round(float(fpr), 4),
            "Avg Risk Score": round(float(yprob.mean()), 4),
        })

    return pd.DataFrame(rows)


# ── Plotting ──────────────────────────────────────────────────────────────────

def plot_roc_curves(y_true, models_proba_dict):
    """
    ROC curves for multiple models on the same axes.
    AUC is shown in the legend for easy comparison.
    """
    from sklearn.metrics import roc_curve, roc_auc_score

    fig, ax = plt.subplots(figsize=(7, 5))
    colors  = [PALETTE["primary"], PALETTE["secondary"], PALETTE["approve"]]

    for i, (name, proba) in enumerate(models_proba_dict.items()):
        fpr, tpr, _ = roc_curve(y_true, proba)
        auc         = roc_auc_score(y_true, proba)
        ax.plot(fpr, tpr, color=colors[i % len(colors)],
                linewidth=1.8, label=f"{name}  (AUC = {auc:.4f})")

    ax.plot([0, 1], [0, 1], color=PALETTE["grey"],
            linestyle="--", linewidth=1, label="Random classifier")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves — Model Comparison", color=PALETTE["primary"])
    ax.legend(fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    return fig


def plot_precision_recall_curve(y_true, proba, model_name="Model"):
    """
    Precision-recall curve. More informative than ROC for imbalanced datasets
    like credit scoring where defaults are a small minority.
    """
    from sklearn.metrics import precision_recall_curve, average_precision_score

    precision, recall, _ = precision_recall_curve(y_true, proba)
    ap       = average_precision_score(y_true, proba)
    baseline = y_true.mean()

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(recall, precision, color=PALETTE["primary"],
            linewidth=1.8, label=f"{model_name}  (AP = {ap:.4f})")
    ax.axhline(baseline, color=PALETTE["grey"], linestyle="--",
               linewidth=1, label=f"Random baseline ({baseline:.3f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve", color=PALETTE["primary"])
    ax.legend(fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    return fig


def plot_score_distribution(y_true, proba, model_name="Model"):
    """
    Overlapping histogram of predicted default probabilities split by
    actual outcome. A well-separated model shows two distinct peaks.
    """
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(proba[y_true == 0], bins=50, alpha=0.6,
            color=PALETTE["approve"], label="Non-default (actual)", density=True)
    ax.hist(proba[y_true == 1], bins=50, alpha=0.6,
            color=PALETTE["reject"],  label="Default (actual)", density=True)
    ax.set_xlabel("Predicted Default Probability")
    ax.set_ylabel("Density")
    ax.set_title(f"{model_name} — Score Distribution by Actual Outcome",
                 color=PALETTE["primary"])
    ax.legend(fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    return fig


def plot_decision_distribution(decisions):
    """
    Donut chart showing the split of approve / review / reject decisions.
    """
    counts  = decisions.value_counts()
    colours = [PALETTE["approve"], PALETTE["review"], PALETTE["reject"]]
    labels  = ["Approve", "Review", "Reject"]
    vals    = [counts.get(k, 0) for k in ["approve", "review", "reject"]]

    fig, ax = plt.subplots(figsize=(5, 4))
    wedges, texts, autotexts = ax.pie(
        vals, labels=labels, colors=colours, autopct="%1.1f%%",
        startangle=90, pctdistance=0.75,
        wedgeprops=dict(width=0.5),
    )
    for t in autotexts:
        t.set_fontsize(10)
    ax.set_title("Decision Distribution", color=PALETTE["primary"])
    return fig


def plot_fairness_bars(fairness_df, metric="Approval Rate"):
    """
    Horizontal bar chart comparing a fairness metric across demographic groups.
    A vertical line marks the overall mean so disparities are easy to spot.
    """
    df  = fairness_df.sort_values(metric)
    fig, ax = plt.subplots(figsize=(8, max(3, len(df) * 0.5 + 1)))

    bars = ax.barh(df["Group"], df[metric],
                   color=PALETTE["secondary"], alpha=0.85)

    overall_mean = fairness_df[metric].mean()
    ax.axvline(overall_mean, color=PALETTE["reject"], linestyle="--",
               linewidth=1.2, label=f"Overall mean: {overall_mean:.3f}")

    for bar, val in zip(bars, df[metric]):
        ax.text(val + 0.005, bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}", va="center", fontsize=9)

    ax.set_xlabel(metric)
    ax.set_title(f"{metric} by Demographic Group", color=PALETTE["primary"])
    ax.legend(fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    return fig
