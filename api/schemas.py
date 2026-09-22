from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    application: dict[str, Any]


class SearchRequest(BaseModel):
    query: str = Field(min_length=2)
    case_id: str | None = None
    top_k: int = Field(default=4, ge=1, le=20)


class EvidenceRequest(BaseModel):
    case_id: str
    missing_items: list[str] = Field(min_length=1)
    approved: bool = False


class ReviewCaseRequest(BaseModel):
    case_id: str
    reason: str
    approved: bool = False


class HumanDecisionRequest(BaseModel):
    case_id: str
    final_decision: str
    override: bool = False
    override_reason: str | None = None
    reviewer: str = "human_reviewer"
    approved: bool = True
