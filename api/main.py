from __future__ import annotations

from functools import lru_cache

from fastapi import FastAPI, HTTPException

from api.schemas import (
    EvidenceRequest,
    HumanDecisionRequest,
    PredictRequest,
    ReviewCaseRequest,
    SearchRequest,
)
from src.audit.store import AuditStore
from src.cases import get_case, load_cases
from src.config import settings
from src.credit.model_service import CreditModelService
from src.pipeline import analyse_case, build_demo_store
from src.tools.registry import ToolRegistry


app = FastAPI(
    title="AI Underwriting & Credit Risk Copilot API",
    version="1.0.0",
    description="Local enterprise-style prototype. The API never delegates the final credit decision to an LLM.",
)

_audit = AuditStore(settings.audit_db)
_tools = ToolRegistry(_audit, settings.outputs_dir / "case_exports")


@lru_cache(maxsize=1)
def demo_store():
    return build_demo_store(settings.rag_backend)


def model_service():
    return CreditModelService(settings.models_dir)


@app.get("/health")
def health() -> dict:
    model = model_service()
    return {
        "status": "ok",
        "credit_model_available": model.available,
        "missing_model_artifacts": model.missing_artifacts(),
        "rag_backend": demo_store().backend,
        "final_credit_decision_by_llm": False,
    }


@app.get("/cases")
def cases() -> list[dict]:
    return [
        {
            "case_id": c["case_id"],
            "applicant_name": c["application"]["applicant_name"],
            "required_evidence": c["required_evidence"],
        }
        for c in load_cases()
    ]


@app.get("/cases/{case_id}")
def case_detail(case_id: str, use_fixture_band: bool = False) -> dict:
    try:
        return analyse_case(
            case_id,
            demo_store(),
            model_service=model_service(),
            audit=_audit,
            use_fixture_band=use_fixture_band,
        ).to_dict()
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.post("/predict")
def predict(request: PredictRequest) -> dict:
    return model_service().predict_application(request.application).to_dict()


@app.post("/documents/search")
def document_search(request: SearchRequest) -> list[dict]:
    if request.case_id:
        try:
            get_case(request.case_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
    return [
        r.to_dict()
        for r in demo_store().search(request.query, request.top_k, request.case_id)
    ]


@app.post("/tools/request-evidence")
def request_evidence(request: EvidenceRequest) -> dict:
    return _tools.execute(
        "request_additional_evidence",
        {"case_id": request.case_id, "missing_items": request.missing_items},
        approved=request.approved,
        actor="api_user",
    ).to_dict()


@app.post("/tools/create-review-case")
def create_review_case(request: ReviewCaseRequest) -> dict:
    return _tools.execute(
        "create_review_case",
        {"case_id": request.case_id, "reason": request.reason},
        approved=request.approved,
        actor="api_user",
    ).to_dict()


@app.post("/decisions")
def record_decision(request: HumanDecisionRequest) -> dict:
    return _tools.execute(
        "record_human_decision",
        {
            "case_id": request.case_id,
            "final_decision": request.final_decision,
            "override": request.override,
            "override_reason": request.override_reason,
            "reviewer": request.reviewer,
        },
        approved=request.approved,
        actor=request.reviewer,
    ).to_dict()


@app.get("/audit/{case_id}")
def audit(case_id: str) -> list[dict]:
    return _audit.events(case_id)
