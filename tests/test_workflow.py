from src.documents.consistency import ConsistencyIssue
from src.workflows.router import plan_workflow


def test_missing_income_routes_to_evidence_required():
    issues = [ConsistencyIssue("MISSING_INCOME_EVIDENCE", "high", "missing", 50000, [], [])]
    plan = plan_workflow("X-1", "review", issues)
    assert plan.route == "evidence_required"
    assert any(a.tool == "request_additional_evidence" for a in plan.actions)


def test_reject_band_never_proposes_autonomous_credit_decision():
    plan = plan_workflow("X-2", "reject", [])
    assert plan.route == "human_credit_review"
    assert all(a.tool != "record_human_decision" for a in plan.actions)
