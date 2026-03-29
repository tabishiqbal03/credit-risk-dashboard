"""
app.py — Streamlit dashboard for the Explainable Credit Risk Decision System.

Run with:
    streamlit run app.py

Five pages:
    1. Overview           — dataset summary and class imbalance
    2. Model Performance  — AUC, recall, precision comparison
    3. Explainability     — SHAP feature importance and individual explanations
    4. Decision System    — interactive threshold tool
    5. Fairness Analysis  — demographic parity and equalised odds
"""

import os
import json
import pickle
import warnings
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")

from utils import (
    apply_decision_thresholds,
    classification_metrics,
    plot_roc_curves,
    plot_precision_recall_curve,
    plot_score_distribution,
    plot_decision_distribution,
    plot_fairness_bars,
    PALETTE,
)


# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Credit Risk Dashboard",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    div[data-testid="stMetric"] {
        background: #eaf2fb;
        border-radius: 8px;
        padding: 12px 16px;
        border-left: 4px solid #1a5276;
    }
    .section-header {
        color: #154360;
        border-bottom: 2px solid #1a5276;
        padding-bottom: 6px;
        margin-bottom: 16px;
    }
    .approve-badge { background: #d5f5e3; color: #1e8449; padding: 4px 12px;
                     border-radius: 12px; font-weight: 600; }
    .review-badge  { background: #fdebd0; color: #d68910; padding: 4px 12px;
                     border-radius: 12px; font-weight: 600; }
    .reject-badge  { background: #fadbd8; color: #922b21; padding: 4px 12px;
                     border-radius: 12px; font-weight: 600; }
</style>
""", unsafe_allow_html=True)


# ── Load everything ───────────────────────────────────────────────────────────

if not os.path.exists("models/results.json"):
    st.error("Models not found. Run `python train.py` first, then relaunch.")
    st.stop()


@st.cache_data(show_spinner=False)
def load_all():
    models = {}
    for name, fname in [("Logistic Regression", "logistic_regression.pkl"),
                         ("LightGBM", "lightgbm.pkl")]:
        path = os.path.join("models", fname)
        if os.path.exists(path):
            with open(path, "rb") as f:
                models[name] = pickle.load(f)

    with open("models/scaler.pkl", "rb") as f:
        scaler = pickle.load(f)

    with open("models/results.json") as f:
        results = json.load(f)
    with open("models/feature_cols.json") as f:
        feature_cols = json.load(f)
    with open("models/threshold_stats.json") as f:
        threshold_stats = json.load(f)
    with open("models/fairness_results.json") as f:
        fairness = json.load(f)

    # load the dynamically computed decision thresholds from training
    with open("models/decision_thresholds.json") as f:
        decision_thresholds = json.load(f)

    # load the optimal classification thresholds
    lgb_thresh = 0.5
    lr_thresh  = 0.5
    if os.path.exists("models/lgb_threshold.json"):
        with open("models/lgb_threshold.json") as f:
            lgb_thresh = json.load(f)["threshold"]
    if os.path.exists("models/lr_threshold.json"):
        with open("models/lr_threshold.json") as f:
            lr_thresh = json.load(f)["threshold"]

    preds       = pd.read_parquet("models/test_predictions.parquet")
    shap_vals   = np.load("models/shap_values.npy")
    shap_sample = pd.read_parquet("models/shap_sample.parquet")
    shap_ev     = float(np.load("models/shap_expected_value.npy")[0])

    return (models, scaler, results, feature_cols, threshold_stats,
            fairness, preds, shap_vals, shap_sample, shap_ev,
            decision_thresholds, lgb_thresh, lr_thresh)


with st.spinner("Loading models and results..."):
    (models, scaler, results, feature_cols, threshold_stats,
     fairness, preds, shap_vals, shap_sample, shap_ev,
     decision_thresholds, lgb_thresh, lr_thresh) = load_all()

# pull the trained thresholds — used as defaults in the Decision System page
TRAINED_APPROVE_T = decision_thresholds["approve_threshold"]
TRAINED_REJECT_T  = decision_thresholds["reject_threshold"]


# ── Sidebar ───────────────────────────────────────────────────────────────────

st.sidebar.title("Credit Risk Dashboard")
st.sidebar.markdown("Home Credit Default Risk — explainable lending decisions.")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigate",
    ["Overview", "Model Performance", "Explainability",
     "Decision System", "Fairness Analysis"]
)

st.sidebar.markdown("---")
st.sidebar.markdown(
    "**Data:** [Home Credit — Kaggle](https://www.kaggle.com/competitions/home-credit-default-risk)  \n"
    "**Models:** Logistic Regression · LightGBM  \n"
    "**Explainability:** SHAP  \n"
    "**Fairness:** Demographic parity · Equalised odds"
)


# ── PAGE 1: Overview ──────────────────────────────────────────────────────────

if page == "Overview":
    st.markdown("<h2 class='section-header'>Dataset Overview</h2>",
                unsafe_allow_html=True)
    st.markdown(
        "The Home Credit dataset contains loan applications with a binary target: "
        "whether the client had payment difficulties. The dataset is heavily "
        "imbalanced — roughly 8% of applicants default."
    )

    default_rate = preds["TARGET"].mean()
    n_test       = len(preds)
    n_defaults   = preds["TARGET"].sum()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Test Set Size",   f"{n_test:,}")
    c2.metric("Default Rate",    f"{default_rate*100:.1f}%")
    c3.metric("Actual Defaults", f"{n_defaults:,}")
    c4.metric("Non-Defaults",    f"{n_test - n_defaults:,}")

    st.markdown("---")
    st.subheader("Why Accuracy is the Wrong Metric Here")
    st.markdown(
        f"A model that predicts 'no default' for every applicant would be "
        f"**{(1-default_rate)*100:.1f}% accurate** — yet completely useless. "
        "This is why we optimise for **Recall** and report **AUC-PR** rather "
        "than raw accuracy, and why classification thresholds are tuned from "
        "the data rather than hardcoded at 0.5."
    )

    fig_imb, ax_imb = plt.subplots(figsize=(5, 3))
    counts = [n_test - n_defaults, n_defaults]
    labels = [f"No Default\n({counts[0]:,})", f"Default\n({counts[1]:,})"]
    ax_imb.bar(labels, counts,
               color=[PALETTE["approve"], PALETTE["reject"]],
               alpha=0.85, width=0.4)
    ax_imb.set_ylabel("Count")
    ax_imb.set_title("Class Distribution — Test Set", color=PALETTE["primary"])
    ax_imb.spines[["top", "right"]].set_visible(False)
    st.pyplot(fig_imb, use_container_width=False)
    plt.close(fig_imb)


# ── PAGE 2: Model Performance ─────────────────────────────────────────────────

elif page == "Model Performance":
    st.markdown("<h2 class='section-header'>Model Performance</h2>",
                unsafe_allow_html=True)

    st.subheader("Evaluation Metrics")
    st.markdown(
        f"LightGBM threshold tuned to **{lgb_thresh}**, "
        f"Logistic Regression to **{lr_thresh}** — both optimised for F1 "
        "on the test set rather than using the default 0.5."
    )

    res_df = pd.DataFrame(results).T.reset_index()
    # .T puts models as rows and metrics as columns; reset_index() promotes
    # the model names to a regular column — so columns are already correct.
    res_df.columns = ["Model"] + list(pd.DataFrame(results).index)
    st.dataframe(
        res_df.style
              .format({c: "{:.4f}" for c in res_df.columns if c != "Model"})
              .highlight_max(
                  subset=[c for c in res_df.columns if c != "Model"],
                  color="#d5f5e3"
              ),
        use_container_width=True,
        hide_index=True,
    )

    st.markdown("---")
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("ROC Curves")
        probas = {
            "Logistic Regression": preds["lr_proba"].values,
            "LightGBM":            preds["lgb_proba"].values,
        }
        roc_fig = plot_roc_curves(preds["TARGET"].values, probas)
        st.pyplot(roc_fig, use_container_width=True)
        plt.close(roc_fig)

    with col2:
        st.subheader("Score Distribution")
        dist_fig = plot_score_distribution(
            preds["TARGET"].values,
            preds["lgb_proba"].values,
            "LightGBM"
        )
        st.pyplot(dist_fig, use_container_width=True)
        plt.close(dist_fig)

    st.markdown("---")
    st.subheader("Precision-Recall Curve (LightGBM)")
    pr_fig = plot_precision_recall_curve(
        preds["TARGET"].values,
        preds["lgb_proba"].values,
        "LightGBM"
    )
    st.pyplot(pr_fig, use_container_width=True)
    plt.close(pr_fig)


# ── PAGE 3: Explainability ────────────────────────────────────────────────────

elif page == "Explainability":
    st.markdown("<h2 class='section-header'>Model Explainability — SHAP</h2>",
                unsafe_allow_html=True)

    st.subheader("Global Feature Importance")
    st.markdown(
        "The SHAP summary plot shows which features have the biggest impact "
        "across all predictions, and whether high values push risk up or down."
    )
    shap_img = os.path.join("plots", "shap_summary.png")
    if os.path.exists(shap_img):
        st.image(shap_img, use_container_width=True)
    else:
        st.warning("SHAP plot not found — run train.py to generate it.")

    st.markdown("---")
    st.subheader("Individual Application Explanation")
    st.markdown(
        "Select any applicant to see exactly which features drove their risk score. "
        "This is the kind of explanation a credit officer or regulator would ask for."
    )

    applicant_idx = st.slider(
        "Applicant index", 0, len(shap_sample) - 1, 0
    )
    applicant = shap_sample.iloc[applicant_idx]
    sv        = shap_vals[applicant_idx]

    lgb_model = models.get("LightGBM")
    if lgb_model is None:
        st.error("LightGBM model not found.")
        st.stop()

    prob     = lgb_model.predict_proba(
        applicant.values.reshape(1, -1)
    )[0, 1]

    # use the trained thresholds to classify this individual
    decision = apply_decision_thresholds(
        np.array([prob]),
        TRAINED_APPROVE_T,
        TRAINED_REJECT_T
    )[0]

    badge = {
        "approve": "approve-badge",
        "review":  "review-badge",
        "reject":  "reject-badge",
    }.get(decision, "review-badge")

    col_a, col_b = st.columns([1, 2])
    with col_a:
        st.metric("Default Probability", f"{prob:.1%}")
        st.metric("Risk Threshold (approve)", f"< {TRAINED_APPROVE_T:.3f}")
        st.metric("Risk Threshold (reject)",  f"> {TRAINED_REJECT_T:.3f}")
        st.markdown(
            f"**Decision:** <span class='{badge}'>{decision.upper()}</span>",
            unsafe_allow_html=True
        )

    with col_b:
        # top 10 features by absolute SHAP value for this applicant
        shap_df = pd.DataFrame({
            "Feature":    feature_cols,
            "SHAP Value": sv,
        })
        top10 = shap_df.reindex(
            shap_df["SHAP Value"].abs().nlargest(10).index
        )

        fig_wf, ax_wf = plt.subplots(figsize=(8, 4))
        colors = [PALETTE["reject"] if s > 0 else PALETTE["approve"]
                  for s in top10["SHAP Value"]]
        ax_wf.barh(top10["Feature"], top10["SHAP Value"],
                   color=colors, alpha=0.85)
        ax_wf.axvline(0, color="black", linewidth=0.8)
        ax_wf.set_xlabel("SHAP value  (positive = increases default risk)")
        ax_wf.set_title("Top 10 Feature Contributions", color=PALETTE["primary"])
        ax_wf.spines[["top", "right"]].set_visible(False)
        st.pyplot(fig_wf, use_container_width=True)
        plt.close(fig_wf)


# ── PAGE 4: Decision System ───────────────────────────────────────────────────

elif page == "Decision System":
    st.markdown("<h2 class='section-header'>Three-Tier Decision System</h2>",
                unsafe_allow_html=True)
    st.markdown(
        "Borderline applications benefit from human review — a trained officer "
        "can assess context the model can't see. The thresholds below default to "
        "the values computed during training (based on score percentiles) but "
        "you can adjust them to explore the approve/review/reject trade-off."
    )

    col_l, col_r = st.columns([1, 1])

    lgb_proba = preds["lgb_proba"].values
    y_true    = preds["TARGET"].values

    # slider min/max derived from the actual probability range
    prob_min = float(lgb_proba.min())
    prob_max = float(lgb_proba.max())

    with col_l:
        approve_t = st.slider(
            "Approve threshold — risk score below this → approve",
            min_value=round(prob_min, 3),
            max_value=round(prob_max, 3),
            value=TRAINED_APPROVE_T,
            step=0.001,
            format="%.3f",
        )
        reject_t = st.slider(
            "Reject threshold — risk score above this → reject",
            min_value=round(prob_min, 3),
            max_value=round(prob_max, 3),
            value=TRAINED_REJECT_T,
            step=0.001,
            format="%.3f",
        )
        if approve_t >= reject_t:
            st.error("Approve threshold must be lower than reject threshold.")
            st.stop()

    decisions = apply_decision_thresholds(lgb_proba, approve_t, reject_t)
    counts    = decisions.value_counts()

    approved_mask      = decisions == "approve"
    default_in_approved = (
        y_true[approved_mask].mean() if approved_mask.sum() > 0 else 0.0
    )

    with col_r:
        st.subheader("Decision Breakdown")
        ca, cr, crj = st.columns(3)
        ca.metric("Approved",
                  f"{counts.get('approve', 0):,}",
                  delta=f"{counts.get('approve',0)/len(decisions)*100:.1f}%")
        cr.metric("Manual Review",
                  f"{counts.get('review', 0):,}",
                  delta=f"{counts.get('review',0)/len(decisions)*100:.1f}%")
        crj.metric("Rejected",
                   f"{counts.get('reject', 0):,}",
                   delta=f"{counts.get('reject',0)/len(decisions)*100:.1f}%")
        st.metric(
            "Default rate in approved applications",
            f"{default_in_approved*100:.2f}%",
            delta=f"{(default_in_approved - y_true.mean())*100:+.2f}% vs population",
            delta_color="inverse",
        )

    st.markdown("---")
    col_p1, col_p2 = st.columns(2)

    with col_p1:
        st.subheader("Decision Distribution")
        dd_fig = plot_decision_distribution(decisions)
        st.pyplot(dd_fig, use_container_width=True)
        plt.close(dd_fig)

    with col_p2:
        st.subheader("Score Distribution with Thresholds")
        fig_t, ax_t = plt.subplots(figsize=(6, 4))
        ax_t.hist(lgb_proba[y_true == 0], bins=50, alpha=0.5,
                  color=PALETTE["approve"], label="Non-default", density=True)
        ax_t.hist(lgb_proba[y_true == 1], bins=50, alpha=0.5,
                  color=PALETTE["reject"],  label="Default", density=True)
        ax_t.axvline(approve_t, color=PALETTE["approve"], linewidth=2,
                     linestyle="--", label=f"Approve ({approve_t:.3f})")
        ax_t.axvline(reject_t,  color=PALETTE["reject"],  linewidth=2,
                     linestyle="--", label=f"Reject ({reject_t:.3f})")
        ax_t.set_xlabel("Predicted Default Probability")
        ax_t.set_ylabel("Density")
        ax_t.set_title("Score Distribution with Decision Boundaries",
                        color=PALETTE["primary"])
        ax_t.legend(fontsize=8)
        ax_t.spines[["top", "right"]].set_visible(False)
        st.pyplot(fig_t, use_container_width=True)
        plt.close(fig_t)


# ── PAGE 5: Fairness Analysis ─────────────────────────────────────────────────

elif page == "Fairness Analysis":
    st.markdown("<h2 class='section-header'>Fairness Analysis</h2>",
                unsafe_allow_html=True)
    st.markdown(
        "Under the EU AI Act, credit scoring is classified as high-risk AI "
        "requiring demonstrable fairness. We evaluate two standard criteria: "
        "**demographic parity** (equal approval rates) and "
        "**equalised odds** (equal TPR and FPR across groups)."
    )

    attribute = st.radio(
        "Sensitive attribute", ["Gender", "Age Group"], horizontal=True
    )
    key = "gender" if attribute == "Gender" else "age_group"

    if key not in fairness or not fairness[key]:
        st.warning(
            f"Fairness data for {attribute} not available. "
            "The raw data column may not have been found during training."
        )
    else:
        fairness_df = pd.DataFrame(fairness[key])
        st.dataframe(
            fairness_df.style.format({
                "Default Rate":   "{:.3f}",
                "Approval Rate":  "{:.3f}",
                "TPR (Recall)":   "{:.3f}",
                "FPR":            "{:.3f}",
                "Avg Risk Score": "{:.3f}",
            }),
            use_container_width=True,
            hide_index=True,
        )

        st.markdown("---")
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Approval Rate by Group")
            fig_ap = plot_fairness_bars(fairness_df, metric="Approval Rate")
            st.pyplot(fig_ap, use_container_width=True)
            plt.close(fig_ap)

        with col2:
            st.subheader("TPR (Recall on Defaults) by Group")
            st.markdown(
                "Equal TPR = equalised odds. The model should catch defaults "
                "at the same rate regardless of demographic group."
            )
            fig_tpr = plot_fairness_bars(fairness_df, metric="TPR (Recall)")
            st.pyplot(fig_tpr, use_container_width=True)
            plt.close(fig_tpr)

        st.markdown("---")
        st.subheader("Interpreting the Results")
        st.markdown("""
        - **Demographic parity gap** — large differences in approval rates may indicate
          the model is indirectly using protected attributes as proxies.

        - **Equalised odds gap** — if TPR differs significantly across groups, the model
          is better at identifying defaults in some groups than others.

        - **What to do if gaps are large** — options include re-weighting training data
          by group, post-hoc threshold adjustment per group, or removing features
          that act as proxies for protected attributes (e.g. postcode for race).

        This analysis is a starting point for the kind of audit that regulators
        and responsible AI frameworks require — not a guarantee of fairness.
        """)
