from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Iterable

from src.documents.extraction import ExtractedFact, facts_by_field


@dataclass(frozen=True)
class ConsistencyIssue:
    code: str
    severity: str
    message: str
    application_value: str | float | int | None
    evidence_values: list[str | float | int]
    sources: list[str]

    def to_dict(self) -> dict:
        return asdict(self)


def _normalise_status(value: str) -> str:
    value = value.strip().lower().replace("-", " ")
    aliases = {
        "permanent employee": "permanent",
        "permanent employment": "permanent",
        "full time permanent": "permanent",
        "full-time permanent": "permanent",
        "contract employee": "contract",
        "fixed term": "contract",
        "self employed": "self employed",
        "self-employed": "self employed",
        "unemployed": "unemployed",
    }
    return aliases.get(value, value)


def _relative_gap(a: float, b: float) -> float:
    denom = max(abs(a), 1.0)
    return abs(a - b) / denom


def check_consistency(
    application: dict,
    facts: Iterable[ExtractedFact],
    required_evidence: Iterable[str] = ("income", "employment"),
    income_tolerance_pct: float = 0.05,
    duration_tolerance_months: int = 2,
) -> list[ConsistencyIssue]:
    grouped = facts_by_field(facts)
    issues: list[ConsistencyIssue] = []

    income_facts = (
        grouped.get("verified_annual_income", [])
        + grouped.get("annual_income", [])
        + grouped.get("annualised_salary_credit", [])
    )
    employment_facts = (
        grouped.get("employment_status", [])
        + grouped.get("employment_duration_months", [])
        + grouped.get("employer", [])
    )
    bank_facts = grouped.get("monthly_salary_credit", [])

    required = set(required_evidence)
    if "income" in required and not income_facts:
        issues.append(
            ConsistencyIssue(
                "MISSING_INCOME_EVIDENCE",
                "high",
                "No supporting income evidence was found.",
                application.get("declared_annual_income"),
                [],
                [],
            )
        )
    if "employment" in required and not employment_facts:
        issues.append(
            ConsistencyIssue(
                "MISSING_EMPLOYMENT_EVIDENCE",
                "high",
                "No supporting employment evidence was found.",
                application.get("employment_status"),
                [],
                [],
            )
        )
    if "bank" in required and not bank_facts:
        issues.append(
            ConsistencyIssue(
                "MISSING_BANK_EVIDENCE",
                "medium",
                "No bank evidence was found.",
                None,
                [],
                [],
            )
        )

    declared_income = application.get("declared_annual_income")
    if declared_income is not None and income_facts:
        mismatches = [
            fact for fact in income_facts
            if _relative_gap(float(declared_income), float(fact.value)) > income_tolerance_pct
        ]
        if mismatches:
            issues.append(
                ConsistencyIssue(
                    "INCOME_MISMATCH",
                    "high",
                    f"Declared annual income differs from supporting evidence by more than {income_tolerance_pct:.0%}.",
                    float(declared_income),
                    [float(f.value) for f in mismatches],
                    sorted({f.source for f in mismatches}),
                )
            )

        values = [float(f.value) for f in income_facts]
        if len(values) >= 2:
            low, high = min(values), max(values)
            if _relative_gap(low, high) > income_tolerance_pct:
                issues.append(
                    ConsistencyIssue(
                        "CONFLICTING_INCOME_EVIDENCE",
                        "high",
                        "Income values conflict across supporting documents.",
                        float(declared_income),
                        values,
                        sorted({f.source for f in income_facts}),
                    )
                )

    declared_status = application.get("employment_status")
    status_facts = grouped.get("employment_status", [])
    if declared_status and status_facts:
        mismatches = [
            f for f in status_facts
            if _normalise_status(str(f.value)) != _normalise_status(str(declared_status))
        ]
        if mismatches:
            issues.append(
                ConsistencyIssue(
                    "EMPLOYMENT_STATUS_MISMATCH",
                    "high",
                    "Declared employment status conflicts with supporting evidence.",
                    str(declared_status),
                    [str(f.value) for f in mismatches],
                    sorted({f.source for f in mismatches}),
                )
            )

    declared_duration = application.get("employment_duration_months")
    duration_facts = grouped.get("employment_duration_months", [])
    if declared_duration is not None and duration_facts:
        mismatches = [
            f for f in duration_facts
            if abs(int(declared_duration) - int(f.value)) > duration_tolerance_months
        ]
        if mismatches:
            issues.append(
                ConsistencyIssue(
                    "EMPLOYMENT_DURATION_MISMATCH",
                    "medium",
                    f"Declared employment duration differs from evidence by more than {duration_tolerance_months} months.",
                    int(declared_duration),
                    [int(f.value) for f in mismatches],
                    sorted({f.source for f in mismatches}),
                )
            )

    return _deduplicate(issues)


def _deduplicate(issues: list[ConsistencyIssue]) -> list[ConsistencyIssue]:
    seen: set[str] = set()
    result: list[ConsistencyIssue] = []
    for issue in issues:
        if issue.code not in seen:
            seen.add(issue.code)
            result.append(issue)
    return result
