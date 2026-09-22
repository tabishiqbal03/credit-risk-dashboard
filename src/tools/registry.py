from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Callable

from pydantic import BaseModel, Field, ValidationError

from src.audit.store import AuditStore


class CreateReviewCaseInput(BaseModel):
    case_id: str
    reason: str


class RequestEvidenceInput(BaseModel):
    case_id: str
    missing_items: list[str] = Field(min_length=1)


class GenerateNotesInput(BaseModel):
    case_id: str
    notes: str | None = None


class RouteCaseInput(BaseModel):
    case_id: str
    route: str


class RecordDecisionInput(BaseModel):
    case_id: str
    final_decision: str
    override: bool = False
    override_reason: str | None = None
    reviewer: str = "human_reviewer"


class ExportCaseInput(BaseModel):
    case_id: str
    summary: dict[str, Any]


@dataclass(frozen=True)
class ToolSpec:
    name: str
    schema: type[BaseModel]
    approval_required: bool
    handler: Callable[[BaseModel], dict[str, Any]]


@dataclass(frozen=True)
class ToolExecution:
    ok: bool
    tool: str
    approval_required: bool
    approved: bool
    result: dict[str, Any] | None = None
    error: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)


class ToolRegistry:
    def __init__(self, audit: AuditStore, output_dir: str | Path) -> None:
        self.audit = audit
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._tools: dict[str, ToolSpec] = {
            "create_review_case": ToolSpec("create_review_case", CreateReviewCaseInput, True, self._create_review_case),
            "request_additional_evidence": ToolSpec("request_additional_evidence", RequestEvidenceInput, True, self._request_evidence),
            "generate_underwriter_case_notes": ToolSpec("generate_underwriter_case_notes", GenerateNotesInput, False, self._generate_notes),
            "route_case": ToolSpec("route_case", RouteCaseInput, False, self._route_case),
            "record_human_decision": ToolSpec("record_human_decision", RecordDecisionInput, True, self._record_decision),
            "export_case_summary": ToolSpec("export_case_summary", ExportCaseInput, False, self._export_case),
        }

    @property
    def allowed_tools(self) -> list[str]:
        return sorted(self._tools)

    def spec(self, name: str) -> ToolSpec | None:
        return self._tools.get(name)

    def execute(
        self,
        name: str,
        arguments: dict[str, Any],
        approved: bool = False,
        actor: str = "system",
    ) -> ToolExecution:
        spec = self._tools.get(name)
        case_id = str(arguments.get("case_id", "unknown"))
        if spec is None:
            execution = ToolExecution(False, name, False, approved, error="Tool is not on the allow-list")
            self.audit.log(case_id, "tool_rejected", execution.to_dict(), actor)
            return execution
        try:
            validated = spec.schema.model_validate(arguments)
        except ValidationError as exc:
            execution = ToolExecution(False, name, spec.approval_required, approved, error=f"Invalid tool arguments: {exc}")
            self.audit.log(case_id, "tool_validation_failed", execution.to_dict(), actor)
            return execution
        if spec.approval_required and not approved:
            execution = ToolExecution(False, name, True, False, error="Human approval is required before this tool can run")
            self.audit.log(case_id, "tool_approval_required", {"tool": name, "arguments": validated.model_dump()}, actor)
            return execution

        try:
            result = spec.handler(validated)
            execution = ToolExecution(True, name, spec.approval_required, approved, result=result)
            event_type = (
                "human_decision_recorded"
                if name == "record_human_decision"
                else "tool_executed"
            )
            self.audit.log(
                case_id,
                event_type,
                {"tool": name, "arguments": validated.model_dump(), "result": result},
                actor,
            )
            return execution
        except Exception as exc:
            execution = ToolExecution(False, name, spec.approval_required, approved, error=str(exc))
            self.audit.log(case_id, "tool_failed", execution.to_dict(), actor)
            return execution

    def _create_review_case(self, payload: CreateReviewCaseInput) -> dict[str, Any]:
        return {"review_case_id": f"REV-{payload.case_id}", "status": "queued", "reason": payload.reason}

    def _request_evidence(self, payload: RequestEvidenceInput) -> dict[str, Any]:
        return {
            "request_id": f"EVID-{payload.case_id}",
            "status": "drafted_not_sent",
            "missing_items": payload.missing_items,
        }

    def _generate_notes(self, payload: GenerateNotesInput) -> dict[str, Any]:
        return {"case_id": payload.case_id, "notes": payload.notes or "Case notes prepared from the current evidence bundle."}

    def _route_case(self, payload: RouteCaseInput) -> dict[str, Any]:
        return {"case_id": payload.case_id, "route": payload.route, "status": "routed"}

    def _record_decision(self, payload: RecordDecisionInput) -> dict[str, Any]:
        allowed = {"approve", "decline", "refer", "withdraw"}
        decision = payload.final_decision.lower()
        if decision not in allowed:
            raise ValueError(f"final_decision must be one of {sorted(allowed)}")
        if payload.override and not payload.override_reason:
            raise ValueError("override_reason is required when override=true")
        return {
            "case_id": payload.case_id,
            "human_final_decision": decision,
            "override": payload.override,
            "override_reason": payload.override_reason,
            "reviewer": payload.reviewer,
        }

    def _export_case(self, payload: ExportCaseInput) -> dict[str, Any]:
        path = self.output_dir / f"{payload.case_id}_case_summary.json"
        path.write_text(json.dumps(payload.summary, indent=2, default=str), encoding="utf-8")
        return {"path": str(path)}
