from __future__ import annotations

import re
from dataclasses import dataclass, asdict
from typing import Iterable

from src.documents.loader import DocumentPage


@dataclass(frozen=True)
class ExtractedFact:
    field: str
    value: str | float | int
    source: str
    page: int
    raw_text: str
    confidence: float = 1.0

    def to_dict(self) -> dict:
        return asdict(self)


MONEY_RE = r"(?:EUR|€)?\s*([0-9][0-9,]*(?:\.\d{1,2})?)"


def _money(value: str) -> float:
    return float(value.replace(",", ""))


def _match(pattern: str, text: str, flags: int = re.IGNORECASE) -> re.Match | None:
    return re.search(pattern, text, flags)


def extract_facts_from_page(page: DocumentPage) -> list[ExtractedFact]:
    text = page.text
    facts: list[ExtractedFact] = []

    patterns = [
        ("case_id", r"Case\s*ID\s*:\s*([A-Z]{2,5}-\d{3,})", str),
        ("applicant_name", r"Applicant\s*:\s*([^\n]+)", lambda x: x.strip()),
        ("employer", r"Employer\s*:\s*([^\n]+)", lambda x: x.strip()),
        ("employment_status", r"Employment\s*Status\s*:\s*([^\n]+)", lambda x: x.strip().lower()),
        ("employment_duration_months", r"Employment\s*Duration\s*:\s*(\d+)\s*months?", int),
        ("annual_income", rf"Annual\s*(?:Gross\s*)?Income\s*:\s*{MONEY_RE}", _money),
        ("verified_annual_income", rf"Verified\s*Annual\s*Income\s*:\s*{MONEY_RE}", _money),
        ("monthly_salary_credit", rf"Monthly\s*Salary\s*Credit\s*:\s*{MONEY_RE}", _money),
    ]

    for field, pattern, converter in patterns:
        match = _match(pattern, text)
        if not match:
            continue
        raw = match.group(0).strip()
        value = converter(match.group(1))
        facts.append(ExtractedFact(field, value, page.source, page.page, raw))

    # A monthly salary credit can provide a deterministic annualised income signal.
    monthly = next((f for f in facts if f.field == "monthly_salary_credit"), None)
    if monthly:
        facts.append(
            ExtractedFact(
                "annualised_salary_credit",
                round(float(monthly.value) * 12, 2),
                page.source,
                page.page,
                monthly.raw_text,
            )
        )

    for evidence_name, label in [
        ("income_evidence_present", "Income Evidence"),
        ("employment_evidence_present", "Employment Evidence"),
        ("bank_evidence_present", "Bank Evidence"),
    ]:
        match = _match(rf"{label}\s*:\s*(present|missing)", text)
        if match:
            facts.append(
                ExtractedFact(
                    evidence_name,
                    match.group(1).lower(),
                    page.source,
                    page.page,
                    match.group(0).strip(),
                )
            )
    return facts


def extract_facts(pages: Iterable[DocumentPage]) -> list[ExtractedFact]:
    facts: list[ExtractedFact] = []
    for page in pages:
        facts.extend(extract_facts_from_page(page))
    return facts


def facts_by_field(facts: Iterable[ExtractedFact]) -> dict[str, list[ExtractedFact]]:
    grouped: dict[str, list[ExtractedFact]] = {}
    for fact in facts:
        grouped.setdefault(fact.field, []).append(fact)
    return grouped
