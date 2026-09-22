"""Streamlit interface for the AI Underwriting & Credit Risk Copilot."""
from __future__ import annotations

import json
import os
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from src.audit.store import AuditStore
from src.cases import case_document_dir, load_cases
from src.config import settings
from src.credit.model_service import CreditModelService
from src.documents.extraction import extract_facts
from src.documents.loader import load_case_documents
from src.pipeline import analyse_case, build_demo_store
from src.rag.ollama import OllamaClient
from src.tools.registry import ToolRegistry
from utils import (
    PALETTE,
    plot_fairness_bars,
    plot_precision_recall_curve,
    plot_roc_curves,
    plot_score_distribution,
)


st.set_page_config(
    page_title="AI Underwriting & Credit Risk Copilot",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
div[data-testid="stMetric"] {
    background:#171b22;
    border:1px solid #30363d;
    border-radius:8px;
    padding:10px 14px;
    border-left:4px solid #1a5276;
}
div[data-testid="stMetric"] * {
    color:#f0f3f6 !important;
}
.small-note {color:#5d6d7e;font-size:0.9rem;}
.source-box {border:1px solid #d5d8dc;border-radius:8px;padding:10px;margin-bottom:8px;background:#fbfcfc;}
</style>
""",
    unsafe_allow_html=True,
)

ROOT = Path(__file__).resolve().parent


@st.cache_resource(show_spinner=False)
def get_store():
    return build_demo_store(settings.rag_backend)


@st.cache_resource(show_spinner=False)
def get_audit():
    return AuditStore(settings.audit_db)


@st.cache_resource(show_spinner=False)
def get_tools():
    return ToolRegistry(get_audit(), settings.outputs_dir / "case_exports")


def load_json(path: Path, default=None):
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def load_legacy_artifacts():
    models_dir = ROOT / "models"
    if not (models_dir / "results.json").exists():
        return None
    payload = {
        "results": load_json(models_dir / "results.json", {}),
        "fairness": load_json(models_dir / "fairness_results.json", {}),
        "feature_cols": load_json(models_dir / "feature_cols.json", []),
        "thresholds": load_json(models_dir / "decision_thresholds.json", {}),
    }
    try:
        payload["preds"] = pd.read_parquet(models_dir / "test_predictions.parquet")
        payload["shap_vals"] = np.load(models_dir / "shap_values.npy")
        payload["shap_sample"] = pd.read_parquet(models_dir / "shap_sample.parquet")
        with (models_dir / "lightgbm.pkl").open("rb") as f:
            payload["lgb_model"] = pickle.load(f)
    except Exception:
        payload["preds"] = None
        payload["shap_vals"] = None
        payload["shap_sample"] = None
        payload["lgb_model"] = None
    return payload


cases = load_cases()
case_labels = {f"{c['case_id']} · {c['application']['applicant_name']}": c["case_id"] for c in cases}
model_service = CreditModelService(ROOT / "models")

st.sidebar.title("Underwriting Copilot")
selected_label = st.sidebar.selectbox("Synthetic case", list(case_labels))
case_id = case_labels[selected_label]
use_fixture = st.sidebar.checkbox(
    "Use synthetic evaluation risk band when ML artefacts are missing",
    value=not model_service.available,
    help="This is a routing simulation fixture, not a model prediction.",
)
use_ollama = st.sidebar.checkbox("Use local Ollama when available", value=False)
page = st.sidebar.radio(
    "Navigate",
    [
        "Application & Risk Assessment",
        "Model Performance",
        "Model Explanation",
        "Supporting Evidence",
        "Underwriting Copilot",
        "Human Review",
        "Audit Trail",
        "Fairness Analysis",
        "Responsible AI & Evaluation",
    ],
)

client = OllamaClient(settings.ollama_url, settings.ollama_model) if use_ollama else None
analysis = analyse_case(
    case_id,
    get_store(),
    model_service=model_service,
    audit=None,
    ollama_client=client,
    use_fixture_band=use_fixture,
)
case = next(c for c in cases if c["case_id"] == case_id)

st.title("AI Underwriting & Credit Risk Copilot")
st.caption("Enterprise-style prototype combining credit-risk ML, evidence retrieval, controlled automation and human oversight.")

if not model_service.available:
    missing = ", ".join(model_service.missing_artifacts())
    st.info(
        "Credit-model artefacts are not present in this repository, by design. "
        f"Run `python train.py` after placing Home Credit data locally. Missing now: {missing}."
    )
if analysis.risk.get("fixture_band_used"):
    st.warning("Synthetic evaluation band is active. The displayed band is a test fixture and is not a trained-model output.")


if page == "Application & Risk Assessment":
    st.header("Application & Risk Assessment")
    app = analysis.application
    left, right = st.columns([1.3, 1])
    with left:
        st.subheader("Structured application")
        rows = [
            ("Applicant", app["applicant_name"]),
            ("Declared annual income", f"€{app['declared_annual_income']:,.0f}"),
            ("Employment status", app["employment_status"]),
            ("Employer", app["employer"]),
            ("Employment duration", f"{app['employment_duration_months']} months"),
            ("Requested credit", f"€{app['requested_credit']:,.0f}"),
            ("Annual annuity", f"€{app['annuity']:,.0f}"),
            ("Age", app["age_years"]),
        ]
        st.dataframe(pd.DataFrame(rows, columns=["Field", "Value"]), hide_index=True, use_container_width=True)
    with right:
        st.subheader("Risk context")
        risk = analysis.risk
        if risk["predicted_default_probability"] is not None:
            st.metric("Predicted default probability", f"{risk['predicted_default_probability']:.1%}")
        else:
            st.metric("Predicted default probability", "Not available")
        st.metric("Decision band", risk["decision_band"].upper())
        st.caption(risk["note"])
        if risk.get("key_drivers"):
            st.markdown("**Top model contributions**")
            for driver in risk["key_drivers"]:
                st.write(f"• {driver}")

    st.divider()
    st.subheader("Current deterministic checks")
    if analysis.inconsistencies:
        st.dataframe(pd.DataFrame(analysis.inconsistencies), hide_index=True, use_container_width=True)
    else:
        st.success("No deterministic evidence inconsistencies were detected in this synthetic case.")


elif page == "Model Performance":
    st.header("Model Performance")
    artefacts = load_legacy_artifacts()
    if not artefacts:
        st.warning("Model evaluation artefacts are missing. Run `python train.py` first.")
        st.markdown(
            "Historical verified metrics from the existing project: Logistic Regression ROC-AUC **0.7444**, PR-AUC **0.2256**; "
            "LightGBM ROC-AUC **0.7407**, PR-AUC **0.2277**. These are historical results, not a fresh run."
        )
    else:
        results = artefacts["results"]
        table = pd.DataFrame(results).T
        st.dataframe(table, use_container_width=True)
        preds = artefacts.get("preds")
        if preds is not None:
            probas = {"Logistic Regression": preds["lr_proba"].values, "LightGBM": preds["lgb_proba"].values}
            y_true = preds["TARGET"].values
            c1, c2 = st.columns(2)
            with c1:
                fig = plot_roc_curves(y_true, probas)
                st.pyplot(fig, use_container_width=True)
                plt.close(fig)
            with c2:
                fig = plot_precision_recall_curve(y_true, probas["LightGBM"], "LightGBM")
                st.pyplot(fig, use_container_width=True)
                plt.close(fig)
            fig = plot_score_distribution(y_true, probas["LightGBM"], "LightGBM")
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)


elif page == "Model Explanation":
    st.header("Model Explanation")
    shap_img = ROOT / "plots" / "shap_summary.png"
    if shap_img.exists():
        st.subheader("Global SHAP summary")
        st.image(str(shap_img), use_container_width=True)
    else:
        st.warning("Global SHAP plot is not present. Run `python train.py` to generate it.")

    if analysis.risk.get("key_drivers"):
        st.subheader("Selected synthetic case · model contributions")
        st.caption("Computed from the trained LightGBM model using a median-baseline synthetic feature vector.")
        st.dataframe(
            pd.DataFrame({"Contribution": analysis.risk["key_drivers"]}),
            hide_index=True,
            use_container_width=True,
        )
    else:
        st.info("Individual model contributions become available after training artefacts are generated.")

    artefacts = load_legacy_artifacts()
    if artefacts and artefacts.get("shap_sample") is not None and artefacts.get("shap_vals") is not None:
        st.subheader("Legacy holdout applicant explanation")
        sample = artefacts["shap_sample"]
        shap_vals = artefacts["shap_vals"]
        idx = st.slider("Holdout applicant index", 0, len(sample) - 1, 0)
        top = pd.DataFrame({"Feature": sample.columns, "SHAP Value": shap_vals[idx]})
        top = top.reindex(top["SHAP Value"].abs().nlargest(10).index).sort_values("SHAP Value")
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.barh(top["Feature"], top["SHAP Value"])
        ax.axvline(0, linewidth=0.8)
        ax.set_xlabel("SHAP value")
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)


elif page == "Supporting Evidence":
    st.header("Supporting Evidence")
    pages = load_case_documents(case_document_dir(case_id), case_id)
    tabs = st.tabs([p.source for p in pages])
    for tab, doc in zip(tabs, pages):
        with tab:
            st.caption(f"{doc.document_type.upper()} · page {doc.page}")
            st.code(doc.text, language=None)

    st.subheader("Extracted facts")
    facts_df = pd.DataFrame(analysis.extracted_facts)
    if not facts_df.empty:
        st.dataframe(facts_df, hide_index=True, use_container_width=True)
    else:
        st.info("No structured facts were extracted.")

    st.subheader("Deterministic consistency checks")
    if analysis.inconsistencies:
        st.dataframe(pd.DataFrame(analysis.inconsistencies), hide_index=True, use_container_width=True)
    else:
        st.success("No inconsistencies detected.")

    st.subheader("Retrieved evidence")
    for item in analysis.retrieved_evidence:
        st.markdown(f"**{item['source']} · page {item['page']} · score {item['score']:.3f}**")
        st.write(item["text"])


elif page == "Underwriting Copilot":
    st.header("Underwriting Copilot")
    summary = analysis.copilot_summary
    c1, c2, c3 = st.columns(3)
    c1.metric("Generation mode", summary["generation_mode"])
    c2.metric("Abstained", "Yes" if summary["abstained"] else "No")
    c3.metric("Ollama model", summary.get("model") or "Not used")
    if summary.get("error"):
        st.warning(summary["error"])
    st.subheader("Grounded case summary")
    st.write(summary["case_summary"])
    st.subheader("Evidence assessment")
    st.write(summary["evidence_assessment"])
    if summary["inconsistencies"]:
        st.markdown("**Inconsistencies**")
        for item in summary["inconsistencies"]:
            st.write(f"• {item}")
    if summary["missing_evidence"]:
        st.markdown("**Missing evidence**")
        for item in summary["missing_evidence"]:
            st.write(f"• {item}")
    st.markdown("**Source provenance**")
    if summary["citations"]:
        for citation in summary["citations"]:
            st.code(citation, language=None)
    else:
        st.write("No document citations available.")
    st.info("The copilot can explain, retrieve, draft and route. It cannot make the final lending decision.")


elif page == "Human Review":
    st.header("Human Review & Controlled Actions")
    plan = analysis.workflow
    st.metric("Proposed route", plan["route"])
    for reason in plan["rationale"]:
        st.write(f"• {reason}")

    st.subheader("Proposed actions")
    registry = get_tools()
    for i, action in enumerate(plan["actions"]):
        with st.expander(f"{action['tool']} · {'approval required' if action['approval_required'] else 'allow-listed'}"):
            st.write(action["reason"])
            st.json(action["arguments"])
            label = "Approve & execute" if action["approval_required"] else "Execute"
            if st.button(label, key=f"action-{case_id}-{i}"):
                args = dict(action["arguments"])
                if action["tool"] == "generate_underwriter_case_notes":
                    args["notes"] = analysis.copilot_summary["case_summary"]
                result = registry.execute(
                    action["tool"],
                    args,
                    approved=action["approval_required"],
                    actor="streamlit_underwriter",
                )
                if result.ok:
                    st.success("Action executed and written to the audit trail.")
                    st.json(result.result)
                else:
                    st.error(result.error)

    st.subheader("Record final human decision")
    final_decision = st.selectbox("Final decision", ["refer", "approve", "decline", "withdraw"])
    override = st.checkbox("This overrides the workflow/model recommendation")
    override_reason = st.text_area("Override reason", disabled=not override)
    reviewer = st.text_input("Reviewer", value="human_reviewer")
    if st.button("Record human decision"):
        result = registry.execute(
            "record_human_decision",
            {
                "case_id": case_id,
                "final_decision": final_decision,
                "override": override,
                "override_reason": override_reason or None,
                "reviewer": reviewer,
            },
            approved=True,
            actor=reviewer,
        )
        if result.ok:
            st.success("Human decision recorded in the audit trail.")
        else:
            st.error(result.error)

    if st.button("Record current analysis snapshot in audit trail"):
        analyse_case(
            case_id,
            get_store(),
            model_service=model_service,
            audit=get_audit(),
            ollama_client=client,
            use_fixture_band=use_fixture,
        )
        st.success("Analysis snapshot recorded.")


elif page == "Audit Trail":
    st.header("Audit Trail")
    events = get_audit().events(case_id)
    if not events:
        st.info("No audit events have been recorded for this case yet.")
    else:
        audit_rows = [
            {
                "id": event["id"],
                "timestamp": event["timestamp"],
                "event_type": event["event_type"],
                "actor": event["actor"],
            }
            for event in events
        ]
        st.dataframe(pd.DataFrame(audit_rows), hide_index=True, use_container_width=True)

        st.subheader("Event details")
        for event in reversed(events):
            with st.expander(
                f'#{event["id"]} ? {event["event_type"]} ? {event["actor"]}'
            ):
                st.json(event["payload"])

        st.download_button(
            "Download case audit JSON",
            data=json.dumps(events, indent=2, default=str),
            file_name=f"{case_id}_audit.json",
            mime="application/json",
        )


elif page == "Fairness Analysis":
    st.header("Fairness Analysis")
    fairness = load_json(ROOT / "models" / "fairness_results.json", {})
    if not fairness:
        st.warning("Fairness artefacts are missing. Run `python train.py` first.")
    else:
        attribute = st.radio("Sensitive attribute", ["Gender", "Age Group"], horizontal=True)
        key = "gender" if attribute == "Gender" else "age_group"
        rows = fairness.get(key, [])
        if not rows:
            st.info(f"No {attribute.lower()} fairness data were produced by training.")
        else:
            df = pd.DataFrame(rows)
            st.dataframe(df, hide_index=True, use_container_width=True)
            c1, c2 = st.columns(2)
            with c1:
                fig = plot_fairness_bars(df, "Approval Rate")
                st.pyplot(fig, use_container_width=True)
                plt.close(fig)
            with c2:
                fig = plot_fairness_bars(df, "TPR (Recall)")
                st.pyplot(fig, use_container_width=True)
                plt.close(fig)
            st.caption("These metrics support monitoring and investigation; they do not by themselves establish that a system is fair.")


elif page == "Responsible AI & Evaluation":
    st.header("Responsible AI & Evaluation")
    st.markdown(
        """
- **Final credit decision stays human-controlled.** No workflow plan contains an autonomous approval/decline tool call.
- **Retrieval before generation.** Copilot summaries receive structured application data, model context, deterministic checks and retrieved passages.
- **Source provenance.** Retrieved passages carry filename/page metadata and generated citations are validated against supplied sources.
- **Deterministic rules for deterministic tasks.** Income, employment, duration and missing-evidence checks are explicit and testable.
- **Tool allow-listing and schema validation.** Arbitrary tool names, shell commands and Python execution are not supported.
- **Sensitive actions require approval.** Evidence requests, review-case creation and final-decision recording are approval-gated.
- **Graceful failure.** Missing ML artefacts and unavailable Ollama are surfaced rather than hidden.
- **Synthetic documents only.** The committed demo evidence contains fictional identities and no real personal financial documents.
"""
    )
    metrics_path = ROOT / "outputs" / "evaluation" / "deterministic_metrics.json"
    if metrics_path.exists():
        metrics = load_json(metrics_path, {})
        st.subheader("Observed deterministic evaluation")
        st.json(metrics["observed_deterministic_metrics"])
        st.caption(
            "These results are from the deliberately labelled synthetic evaluation set and should not be interpreted as real-world underwriting performance."
        )
        st.subheader("Pending local-LLM evaluation")
        st.json(metrics["pending_ollama_metrics"])
    else:
        st.info("Run `python scripts/run_evaluation.py` to generate deterministic evaluation metrics.")
