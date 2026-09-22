from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Iterable

from src.documents.consistency import ConsistencyIssue


@dataclass(frozen=True)
class ProposedAction:
    tool: str
    reason: str
    approval_required: bool
    arguments: dict

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class WorkflowPlan:
    route: str
    rationale: list[str]
    actions: list[ProposedAction]
    requires_human_review: bool

    def to_dict(self) -> dict:
        return {
            "route": self.route,
            "rationale": self.rationale,
            "actions": [a.to_dict() for a in self.actions],
            "requires_human_review": self.requires_human_review,
        }


def plan_workflow(
    case_id: str,
    model_band: str,
    issues: Iterable[ConsistencyIssue],
) -> WorkflowPlan:
    model_band = (model_band or "unavailable").lower()
    issue_list = list(issues)
    codes = {i.code for i in issue_list}
    rationale: list[str] = []
    actions: list[ProposedAction] = []

    if model_band == "reject":
        route = "human_credit_review"
        rationale.append("The model score is in the reject band; a human must make the final lending decision.")
    elif model_band == "review":
        route = "manual_review"
        rationale.append("The model score is in the manual-review band.")
    elif model_band == "approve":
        route = "standard_underwriting"
        rationale.append("The model score is in the approve band, subject to evidence checks and human policy controls.")
    else:
        route = "model_unavailable_review"
        rationale.append("A trained model score is unavailable, so the case cannot use model-based routing.")

    missing_codes = sorted(c for c in codes if c.startswith("MISSING_"))
    mismatch_codes = sorted(c for c in codes if "MISMATCH" in c or c.startswith("CONFLICTING_"))

    if missing_codes:
        route = "evidence_required"
        rationale.append("Required evidence is missing.")
        actions.append(
            ProposedAction(
                "request_additional_evidence",
                "Request the missing evidence before the case progresses.",
                True,
                {"case_id": case_id, "missing_items": missing_codes},
            )
        )
    if mismatch_codes:
        route = "manual_review"
        rationale.append("One or more deterministic consistency checks found conflicting evidence.")

    if route in {"manual_review", "human_credit_review", "model_unavailable_review", "evidence_required"}:
        actions.append(
            ProposedAction(
                "create_review_case",
                "Create a controlled review item for an underwriter.",
                True,
                {"case_id": case_id, "reason": route},
            )
        )

    actions.append(
        ProposedAction(
            "generate_underwriter_case_notes",
            "Prepare traceable case notes from the approved evidence bundle.",
            False,
            {"case_id": case_id},
        )
    )

    return WorkflowPlan(
        route=route,
        rationale=rationale,
        actions=actions,
        requires_human_review=route != "standard_underwriting" or model_band in {"review", "reject", "unavailable"},
    )
