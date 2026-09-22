from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory

from src.audit.store import AuditStore, audit_completeness
from src.cases import case_document_dir, load_cases
from src.credit.model_service import CreditModelService
from src.documents.consistency import check_consistency
from src.documents.extraction import extract_facts, facts_by_field
from src.documents.loader import load_case_documents
from src.pipeline import analyse_case, build_demo_store
from src.tools.registry import ToolRegistry
from src.workflows.router import plan_workflow


def _values_equal(expected, actual) -> bool:
    if isinstance(expected, (int, float)) and isinstance(actual, (int, float)):
        return abs(float(expected) - float(actual)) <= 0.05
    return str(expected).strip().lower() == str(actual).strip().lower()


def _safe_div(a: int | float, b: int | float) -> float:
    return float(a / b) if b else 0.0


def run_deterministic_evaluation(top_k: int = 3) -> dict:
    cases = load_cases()
    store = build_demo_store("tfidf")

    expected_fields = matched_fields = 0
    retrieval_total = retrieval_hits = 0
    issue_tp = issue_fp = issue_fn = 0
    route_total = route_hits = 0
    escalation_total = escalation_hits = 0
    unsafe_action_count = 0
    valid_tool_calls = successful_tool_calls = 0
    citation_total = citation_valid = 0

    with TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        audit = AuditStore(Path(tmp) / "eval_audit.db")
        tools = ToolRegistry(audit, Path(tmp) / "exports")
        model = CreditModelService(Path(tmp) / "no_models")

        for case in cases:
            case_id = case["case_id"]
            pages = load_case_documents(case_document_dir(case_id), case_id)
            facts = extract_facts(pages)
            grouped = facts_by_field(facts)

            for doc in case.get("document_manifest", []):
                source = doc["filename"]
                for field, expected in doc.get("expected_facts", {}).items():
                    expected_fields += 1
                    candidates = [f for f in grouped.get(field, []) if f.source == source]
                    if any(_values_equal(expected, f.value) for f in candidates):
                        matched_fields += 1

            issues = check_consistency(
                case["application"],
                facts,
                case.get("required_evidence", []),
            )
            actual_codes = {i.code for i in issues}
            expected_codes = set(case.get("expected_issue_codes", []))
            issue_tp += len(actual_codes & expected_codes)
            issue_fp += len(actual_codes - expected_codes)
            issue_fn += len(expected_codes - actual_codes)

            plan = plan_workflow(case_id, case["evaluation_model_band"], issues)
            route_total += 1
            route_hits += int(plan.route == case["expected_route"])
            escalation_total += 1
            expected_escalation = case["expected_route"] != "standard_underwriting"
            escalation_hits += int(plan.requires_human_review == expected_escalation)
            unsafe_action_count += sum(a.tool == "record_human_decision" for a in plan.actions)

            for query_case in case.get("retrieval_eval", []):
                retrieval_total += 1
                results = store.search(query_case["query"], top_k=top_k, case_id=case_id)
                retrieval_hits += int(any(r.source == query_case["expected_source"] for r in results))

            analysis = analyse_case(
                case_id,
                store,
                model_service=model,
                audit=audit,
                ollama_client=None,
                use_fixture_band=True,
            )
            allowed_sources = {
                f"DOC:{item['source']}:p{item['page']}"
                for item in analysis.retrieved_evidence
            }
            for citation in analysis.copilot_summary.get("citations", []):
                citation_total += 1
                citation_valid += int(citation in allowed_sources)

            for action in plan.actions:
                valid_tool_calls += 1
                args = dict(action.arguments)
                if action.tool == "generate_underwriter_case_notes":
                    args["notes"] = "Evaluation note generated from structured evidence."
                result = tools.execute(
                    action.tool,
                    args,
                    approved=action.approval_required,
                    actor="evaluation",
                )
                successful_tool_calls += int(result.ok)

        all_events = audit.events()
        audit_rate = audit_completeness(all_events)

    precision = _safe_div(issue_tp, issue_tp + issue_fp)
    recall = _safe_div(issue_tp, issue_tp + issue_fn)
    f1 = _safe_div(2 * precision * recall, precision + recall)

    return {
        "evaluation_scope": {
            "synthetic_cases": len(cases),
            "retrieval_queries": retrieval_total,
            "retrieval_top_k": top_k,
            "rag_backend": "tfidf",
            "llm_used": False,
        },
        "observed_deterministic_metrics": {
            "document_field_extraction_accuracy": round(_safe_div(matched_fields, expected_fields), 4),
            "document_fields_expected": expected_fields,
            "document_fields_matched": matched_fields,
            "retrieval_hit_rate_at_k": round(_safe_div(retrieval_hits, retrieval_total), 4),
            "retrieval_hits": retrieval_hits,
            "inconsistency_precision": round(precision, 4),
            "inconsistency_recall": round(recall, 4),
            "inconsistency_f1": round(f1, 4),
            "workflow_routing_accuracy": round(_safe_div(route_hits, route_total), 4),
            "escalation_accuracy": round(_safe_div(escalation_hits, escalation_total), 4),
            "unsafe_autonomous_credit_decision_count": int(unsafe_action_count),
            "tool_call_success_rate": round(_safe_div(successful_tool_calls, valid_tool_calls), 4),
            "audit_completeness_rate": round(audit_rate, 4),
            "deterministic_citation_validity_rate": round(_safe_div(citation_valid, citation_total), 4) if citation_total else None,
        },
        "pending_ollama_metrics": {
            "groundedness": "pending local Ollama run",
            "unsupported_claim_rate": "pending local Ollama run",
            "llm_citation_accuracy": "pending local Ollama run",
            "structured_output_success_rate": "pending local Ollama run",
        },
    }


def write_evaluation(path: str | Path, top_k: int = 3) -> dict:
    metrics = run_deterministic_evaluation(top_k=top_k)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics
