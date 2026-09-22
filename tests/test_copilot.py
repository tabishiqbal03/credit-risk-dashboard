from src.documents.extraction import ExtractedFact
from src.rag.copilot import generate_grounded_summary
from src.rag.vector_store import SearchResult


def test_copilot_falls_back_without_ollama_and_keeps_provenance():
    fact = ExtractedFact("verified_annual_income", 50000.0, "income.pdf", 1, "Verified Annual Income: EUR 50,000")
    retrieved = [SearchResult("c1", "income.pdf", 1, "Verified Annual Income: EUR 50,000", 0.9, "X", "pdf")]
    summary = generate_grounded_summary(
        "X",
        {"applicant_name": "Demo Person"},
        {"decision_band": "review", "predicted_default_probability": 0.2, "key_drivers": []},
        [fact],
        [],
        retrieved,
        client=None,
    )
    assert summary.generation_mode == "deterministic_fallback"
    assert "DOC:income.pdf:p1" in summary.citations
