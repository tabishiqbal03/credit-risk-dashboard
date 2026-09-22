from src.cases import case_document_dir, get_case
from src.documents.consistency import check_consistency
from src.documents.extraction import extract_facts
from src.documents.loader import load_case_documents


def issue_codes(case_id):
    case = get_case(case_id)
    facts = extract_facts(load_case_documents(case_document_dir(case_id), case_id))
    return {
        i.code for i in check_consistency(
            case["application"], facts, case["required_evidence"]
        )
    }


def test_income_mismatch_is_deterministic():
    assert issue_codes("CR-003") == {"INCOME_MISMATCH"}


def test_missing_evidence_is_detected():
    assert issue_codes("CR-007") == {
        "MISSING_INCOME_EVIDENCE",
        "MISSING_EMPLOYMENT_EVIDENCE",
    }


def test_conflicting_income_is_detected():
    assert issue_codes("CR-006") == {
        "INCOME_MISMATCH",
        "CONFLICTING_INCOME_EVIDENCE",
    }
