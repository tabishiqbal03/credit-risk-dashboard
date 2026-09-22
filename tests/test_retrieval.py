from src.pipeline import build_demo_store


def test_case_filtered_retrieval_finds_income_source():
    store = build_demo_store("tfidf")
    results = store.search("verified annual income evidence", top_k=3, case_id="CR-003")
    assert any(r.source == "income_confirmation.pdf" for r in results)
    assert all(r.case_id == "CR-003" for r in results)
