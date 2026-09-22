from src.cases import case_document_dir, get_case
from src.documents.extraction import extract_facts, facts_by_field
from src.documents.loader import load_case_documents


def test_loads_txt_pdf_and_docx_and_extracts_key_fields():
    pages = load_case_documents(case_document_dir("CR-001"), "CR-001")
    assert {p.document_type for p in pages} == {"txt", "pdf", "docx"}
    grouped = facts_by_field(extract_facts(pages))
    assert grouped["verified_annual_income"][0].value == 54000.0
    assert grouped["employment_duration_months"][0].value == 36
    assert grouped["monthly_salary_credit"][0].value == 4500.0


def test_synthetic_case_is_fictional_demo_metadata():
    case = get_case("CR-001")
    assert case["case_id"] == "CR-001"
    assert case["application"]["applicant_name"] == "Ava Murphy"
