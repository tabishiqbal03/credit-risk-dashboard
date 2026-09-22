from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from docx import Document
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas


DATA_DIR = ROOT / "data" / "synthetic_demo"
DOCS_DIR = DATA_DIR / "documents"


CASES = [
    {
        "case_id": "CR-001",
        "application": {"applicant_name": "Ava Murphy", "declared_annual_income": 54000, "employment_status": "permanent", "employer": "Northstar Analytics Ltd", "employment_duration_months": 36, "requested_credit": 180000, "annuity": 13200, "age_years": 34},
        "required_evidence": ["income", "employment", "bank"],
        "evaluation_model_band": "approve",
        "expected_issue_codes": [],
        "expected_route": "standard_underwriting",
        "documents": [
            {"filename": "employment_confirmation.txt", "text": "Employment Confirmation\nCase ID: CR-001\nApplicant: Ava Murphy\nEmployer: Northstar Analytics Ltd\nEmployment Status: permanent\nEmployment Duration: 36 months\nEmployment Evidence: present\n", "expected_facts": {"employment_status": "permanent", "employment_duration_months": 36, "employer": "Northstar Analytics Ltd"}},
            {"filename": "income_confirmation.pdf", "text": "Income Verification\nCase ID: CR-001\nApplicant: Ava Murphy\nVerified Annual Income: EUR 54,000\nIncome Evidence: present\n", "expected_facts": {"verified_annual_income": 54000.0}},
            {"filename": "bank_statement_summary.docx", "text": "Synthetic Bank Statement Summary\nCase ID: CR-001\nApplicant: Ava Murphy\nMonthly Salary Credit: EUR 4,500\nBank Evidence: present\n", "expected_facts": {"monthly_salary_credit": 4500.0, "annualised_salary_credit": 54000.0}},
        ],
        "retrieval_eval": [
            {"query": "What evidence supports the applicant's annual income?", "expected_source": "income_confirmation.pdf"},
            {"query": "How long has Ava Murphy worked for Northstar Analytics?", "expected_source": "employment_confirmation.txt"},
        ],
    },
    {
        "case_id": "CR-002",
        "application": {"applicant_name": "Liam Byrne", "declared_annual_income": 46000, "employment_status": "permanent", "employer": "Harbour Tech Services", "employment_duration_months": 22, "requested_credit": 145000, "annuity": 10800, "age_years": 29},
        "required_evidence": ["income", "employment"],
        "evaluation_model_band": "review",
        "expected_issue_codes": ["MISSING_INCOME_EVIDENCE"],
        "expected_route": "evidence_required",
        "documents": [
            {"filename": "employment_confirmation.docx", "text": "Employment Confirmation\nCase ID: CR-002\nApplicant: Liam Byrne\nEmployer: Harbour Tech Services\nEmployment Status: permanent\nEmployment Duration: 22 months\nEmployment Evidence: present\n", "expected_facts": {"employment_status": "permanent", "employment_duration_months": 22, "employer": "Harbour Tech Services"}},
            {"filename": "applicant_note.txt", "text": "Applicant Note\nCase ID: CR-002\nApplicant: Liam Byrne\nThe applicant has not yet supplied a salary certificate or bank salary summary.\n", "expected_facts": {}},
        ],
        "retrieval_eval": [{"query": "What employment evidence is available for Liam Byrne?", "expected_source": "employment_confirmation.docx"}],
    },
    {
        "case_id": "CR-003",
        "application": {"applicant_name": "Sofia Khan", "declared_annual_income": 72000, "employment_status": "permanent", "employer": "Greenline Pharma", "employment_duration_months": 48, "requested_credit": 220000, "annuity": 15600, "age_years": 38},
        "required_evidence": ["income", "employment", "bank"],
        "evaluation_model_band": "review",
        "expected_issue_codes": ["INCOME_MISMATCH"],
        "expected_route": "manual_review",
        "documents": [
            {"filename": "employment_confirmation.txt", "text": "Employment Confirmation\nCase ID: CR-003\nApplicant: Sofia Khan\nEmployer: Greenline Pharma\nEmployment Status: permanent\nEmployment Duration: 48 months\nEmployment Evidence: present\n", "expected_facts": {"employment_status": "permanent", "employment_duration_months": 48}},
            {"filename": "income_confirmation.pdf", "text": "Income Verification\nCase ID: CR-003\nApplicant: Sofia Khan\nVerified Annual Income: EUR 60,000\nIncome Evidence: present\n", "expected_facts": {"verified_annual_income": 60000.0}},
            {"filename": "bank_statement_summary.docx", "text": "Synthetic Bank Statement Summary\nCase ID: CR-003\nApplicant: Sofia Khan\nMonthly Salary Credit: EUR 5,000\nBank Evidence: present\n", "expected_facts": {"monthly_salary_credit": 5000.0, "annualised_salary_credit": 60000.0}},
        ],
        "retrieval_eval": [{"query": "What verified income is shown for Sofia Khan?", "expected_source": "income_confirmation.pdf"}],
    },
    {
        "case_id": "CR-004",
        "application": {"applicant_name": "Noah Silva", "declared_annual_income": 48000, "employment_status": "permanent", "employer": "Orion Logistics", "employment_duration_months": 18, "requested_credit": 130000, "annuity": 9600, "age_years": 31},
        "required_evidence": ["income", "employment"],
        "evaluation_model_band": "approve",
        "expected_issue_codes": ["EMPLOYMENT_STATUS_MISMATCH"],
        "expected_route": "manual_review",
        "documents": [
            {"filename": "employment_confirmation.pdf", "text": "Employment Confirmation\nCase ID: CR-004\nApplicant: Noah Silva\nEmployer: Orion Logistics\nEmployment Status: contract\nEmployment Duration: 18 months\nEmployment Evidence: present\n", "expected_facts": {"employment_status": "contract", "employment_duration_months": 18}},
            {"filename": "income_confirmation.txt", "text": "Income Verification\nCase ID: CR-004\nApplicant: Noah Silva\nVerified Annual Income: EUR 48,000\nIncome Evidence: present\n", "expected_facts": {"verified_annual_income": 48000.0}},
        ],
        "retrieval_eval": [{"query": "What employment status is confirmed for Noah Silva?", "expected_source": "employment_confirmation.pdf"}],
    },
    {
        "case_id": "CR-005",
        "application": {"applicant_name": "Mia O'Connor", "declared_annual_income": 51000, "employment_status": "permanent", "employer": "Cedar Health Systems", "employment_duration_months": 30, "requested_credit": 155000, "annuity": 11200, "age_years": 33},
        "required_evidence": ["income", "employment"],
        "evaluation_model_band": "review",
        "expected_issue_codes": ["EMPLOYMENT_DURATION_MISMATCH"],
        "expected_route": "manual_review",
        "documents": [
            {"filename": "employment_confirmation.docx", "text": "Employment Confirmation\nCase ID: CR-005\nApplicant: Mia O'Connor\nEmployer: Cedar Health Systems\nEmployment Status: permanent\nEmployment Duration: 8 months\nEmployment Evidence: present\n", "expected_facts": {"employment_status": "permanent", "employment_duration_months": 8}},
            {"filename": "income_confirmation.pdf", "text": "Income Verification\nCase ID: CR-005\nApplicant: Mia O'Connor\nVerified Annual Income: EUR 51,000\nIncome Evidence: present\n", "expected_facts": {"verified_annual_income": 51000.0}},
        ],
        "retrieval_eval": [{"query": "How many months of employment are documented for Mia O'Connor?", "expected_source": "employment_confirmation.docx"}],
    },
    {
        "case_id": "CR-006",
        "application": {"applicant_name": "Ethan Walsh", "declared_annual_income": 66000, "employment_status": "permanent", "employer": "Atlas Energy Consulting", "employment_duration_months": 41, "requested_credit": 200000, "annuity": 14400, "age_years": 40},
        "required_evidence": ["income", "employment", "bank"],
        "evaluation_model_band": "review",
        "expected_issue_codes": ["INCOME_MISMATCH", "CONFLICTING_INCOME_EVIDENCE"],
        "expected_route": "manual_review",
        "documents": [
            {"filename": "employment_confirmation.txt", "text": "Employment Confirmation\nCase ID: CR-006\nApplicant: Ethan Walsh\nEmployer: Atlas Energy Consulting\nEmployment Status: permanent\nEmployment Duration: 41 months\nEmployment Evidence: present\n", "expected_facts": {"employment_status": "permanent", "employment_duration_months": 41}},
            {"filename": "income_confirmation.docx", "text": "Income Verification\nCase ID: CR-006\nApplicant: Ethan Walsh\nVerified Annual Income: EUR 66,000\nIncome Evidence: present\n", "expected_facts": {"verified_annual_income": 66000.0}},
            {"filename": "bank_statement_summary.pdf", "text": "Synthetic Bank Statement Summary\nCase ID: CR-006\nApplicant: Ethan Walsh\nMonthly Salary Credit: EUR 4,800\nBank Evidence: present\n", "expected_facts": {"monthly_salary_credit": 4800.0, "annualised_salary_credit": 57600.0}},
        ],
        "retrieval_eval": [{"query": "What salary credit appears in Ethan Walsh's bank evidence?", "expected_source": "bank_statement_summary.pdf"}],
    },
    {
        "case_id": "CR-007",
        "application": {"applicant_name": "Zara Patel", "declared_annual_income": 39000, "employment_status": "contract", "employer": "Bluebridge Media", "employment_duration_months": 11, "requested_credit": 118000, "annuity": 9000, "age_years": 27},
        "required_evidence": ["income", "employment"],
        "evaluation_model_band": "approve",
        "expected_issue_codes": ["MISSING_INCOME_EVIDENCE", "MISSING_EMPLOYMENT_EVIDENCE"],
        "expected_route": "evidence_required",
        "documents": [
            {"filename": "applicant_note.txt", "text": "Applicant Note\nCase ID: CR-007\nApplicant: Zara Patel\nSupporting income and employment documents are still outstanding.\n", "expected_facts": {}},
        ],
        "retrieval_eval": [{"query": "Which documents are still outstanding for Zara Patel?", "expected_source": "applicant_note.txt"}],
    },
    {
        "case_id": "CR-008",
        "application": {"applicant_name": "Daniel Chen", "declared_annual_income": 83000, "employment_status": "permanent", "employer": "Summit Engineering Ireland", "employment_duration_months": 67, "requested_credit": 280000, "annuity": 19200, "age_years": 45},
        "required_evidence": ["income", "employment", "bank"],
        "evaluation_model_band": "reject",
        "expected_issue_codes": [],
        "expected_route": "human_credit_review",
        "documents": [
            {"filename": "employment_confirmation.pdf", "text": "Employment Confirmation\nCase ID: CR-008\nApplicant: Daniel Chen\nEmployer: Summit Engineering Ireland\nEmployment Status: permanent\nEmployment Duration: 67 months\nEmployment Evidence: present\n", "expected_facts": {"employment_status": "permanent", "employment_duration_months": 67}},
            {"filename": "income_confirmation.docx", "text": "Income Verification\nCase ID: CR-008\nApplicant: Daniel Chen\nVerified Annual Income: EUR 83,000\nIncome Evidence: present\n", "expected_facts": {"verified_annual_income": 83000.0}},
            {"filename": "bank_statement_summary.txt", "text": "Synthetic Bank Statement Summary\nCase ID: CR-008\nApplicant: Daniel Chen\nMonthly Salary Credit: EUR 6,916.67\nBank Evidence: present\n", "expected_facts": {"monthly_salary_credit": 6916.67, "annualised_salary_credit": 83000.04}},
        ],
        "retrieval_eval": [{"query": "What income evidence exists for Daniel Chen?", "expected_source": "income_confirmation.docx"}],
    },
]


def write_txt(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")


def write_docx(path: Path, text: str) -> None:
    doc = Document()
    for line in text.splitlines():
        doc.add_paragraph(line)
    doc.save(path)


def write_pdf(path: Path, text: str) -> None:
    c = canvas.Canvas(str(path), pagesize=A4)
    width, height = A4
    y = height - 72
    for line in text.splitlines():
        c.drawString(72, y, line)
        y -= 18
        if y < 72:
            c.showPage()
            y = height - 72
    c.save()


def main() -> None:
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    public_cases = []
    for case in CASES:
        case_dir = DOCS_DIR / case["case_id"]
        case_dir.mkdir(parents=True, exist_ok=True)
        document_manifest = []
        for document in case["documents"]:
            path = case_dir / document["filename"]
            suffix = path.suffix.lower()
            if suffix == ".txt":
                write_txt(path, document["text"])
            elif suffix == ".docx":
                write_docx(path, document["text"])
            elif suffix == ".pdf":
                write_pdf(path, document["text"])
            else:
                raise ValueError(suffix)
            document_manifest.append({"filename": document["filename"], "expected_facts": document["expected_facts"]})

        public_case = {k: v for k, v in case.items() if k not in {"documents"}}
        public_case["document_manifest"] = document_manifest
        public_cases.append(public_case)

    (DATA_DIR / "cases.json").write_text(
        json.dumps({"description": "Fictional synthetic underwriting cases for demo and evaluation. No real PII.", "cases": public_cases}, indent=2),
        encoding="utf-8",
    )
    print(f"Generated {len(public_cases)} synthetic cases in {DATA_DIR}")


if __name__ == "__main__":
    main()
