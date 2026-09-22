from src.audit.store import AuditStore, audit_completeness
from src.tools.registry import ToolRegistry


def test_tool_allowlist_and_approval(tmp_path):
    audit = AuditStore(tmp_path / "audit.db")
    registry = ToolRegistry(audit, tmp_path / "exports")

    denied = registry.execute("run_shell", {"case_id": "X"}, approved=True)
    assert not denied.ok
    assert "allow-list" in denied.error

    pending = registry.execute(
        "create_review_case", {"case_id": "X", "reason": "review"}, approved=False
    )
    assert not pending.ok
    assert pending.approval_required

    allowed = registry.execute(
        "create_review_case", {"case_id": "X", "reason": "review"}, approved=True
    )
    assert allowed.ok


def test_audit_events_are_complete(tmp_path):
    audit = AuditStore(tmp_path / "audit.db")
    audit.log("X", "risk_assessment", {"probability": 0.2})
    events = audit.events("X")
    assert len(events) == 1
    assert audit_completeness(events) == 1.0
