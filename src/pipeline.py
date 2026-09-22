from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path

from src.audit.store import AuditStore
from src.cases import case_document_dir, get_case, load_cases
from src.config import settings
from src.credit.model_service import CreditModelService
from src.documents.consistency import check_consistency
from src.documents.extraction import extract_facts
from src.documents.loader import chunk_pages, load_case_documents
from src.rag.copilot import generate_grounded_summary
from src.rag.ollama import OllamaClient
from src.rag.vector_store import LocalVectorStore
from src.workflows.router import plan_workflow


@dataclass(frozen=True)
class CaseAnalysis:
    case_id: str
    application: dict
    risk: dict
    extracted_facts: list[dict]
    inconsistencies: list[dict]
    retrieved_evidence: list[dict]
    copilot_summary: dict
    workflow: dict

    def to_dict(self) -> dict:
        return asdict(self)


def build_demo_store(backend: str = "tfidf") -> LocalVectorStore:
    chunks = []
    for case in load_cases():
        pages = load_case_documents(case_document_dir(case["case_id"]), case["case_id"])
        chunks.extend(chunk_pages(pages))
    return LocalVectorStore(backend=backend, embedding_model=settings.embedding_model).build(chunks)


def analyse_case(
    case_id: str,
    store: LocalVectorStore,
    model_service: CreditModelService | None = None,
    audit: AuditStore | None = None,
    ollama_client: OllamaClient | None = None,
    use_fixture_band: bool = False,
) -> CaseAnalysis:
    case = get_case(case_id)
    application = case["application"]
    model_service = model_service or CreditModelService(settings.models_dir)
    prediction = model_service.predict_application(application)
    risk = prediction.to_dict()
    if use_fixture_band and not prediction.available:
        risk["decision_band"] = case["evaluation_model_band"]
        risk["note"] = (
            "Evaluation fixture band is being used because trained model artefacts are unavailable. "
            "This is a synthetic routing fixture, not a model prediction."
        )
        risk["fixture_band_used"] = True
    else:
        risk["fixture_band_used"] = False

    pages = load_case_documents(case_document_dir(case_id), case_id)
    facts = extract_facts(pages)
    issues = check_consistency(
        application,
        facts,
        required_evidence=case.get("required_evidence", ["income", "employment"]),
        income_tolerance_pct=settings.income_tolerance_pct,
        duration_tolerance_months=settings.duration_tolerance_months,
    )
    query = (
        "Find evidence about annual income, employment status, employment duration, "
        "bank salary credits, missing documents, and any facts relevant to underwriting review."
    )
    retrieved = store.search(query, top_k=6, case_id=case_id)
    summary = generate_grounded_summary(
        case_id,
        application,
        risk,
        facts,
        issues,
        retrieved,
        client=ollama_client,
    )
    workflow = plan_workflow(case_id, risk["decision_band"], issues)

    if audit:
        audit.log(case_id, "risk_assessment", risk)
        audit.log(case_id, "facts_extracted", {"facts": [f.to_dict() for f in facts]})
        audit.log(case_id, "consistency_checked", {"issues": [i.to_dict() for i in issues]})
        audit.log(case_id, "evidence_retrieved", {"sources": [r.to_dict() for r in retrieved]})
        audit.log(case_id, "copilot_summary_generated", summary.to_dict())
        audit.log(case_id, "workflow_planned", workflow.to_dict())

    return CaseAnalysis(
        case_id=case_id,
        application=application,
        risk=risk,
        extracted_facts=[f.to_dict() for f in facts],
        inconsistencies=[i.to_dict() for i in issues],
        retrieved_evidence=[r.to_dict() for r in retrieved],
        copilot_summary=summary.to_dict(),
        workflow=workflow.to_dict(),
    )
