from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.cases import load_cases
from src.config import settings
from src.credit.model_service import CreditModelService
from src.pipeline import analyse_case, build_demo_store
from src.rag.ollama import OllamaClient


def main() -> None:
    parser = argparse.ArgumentParser(description="Run local Ollama structured-output/citation checks.")
    parser.add_argument("--limit", type=int, default=8)
    parser.add_argument("--output", default="outputs/evaluation/ollama_metrics.json")
    args = parser.parse_args()

    client = OllamaClient(settings.ollama_url, settings.ollama_model)
    if not client.health():
        raise SystemExit(
            f"Ollama is not reachable at {settings.ollama_url}. Start Ollama and pull {settings.ollama_model} first."
        )

    store = build_demo_store(settings.rag_backend)
    model = CreditModelService(settings.models_dir)
    cases = load_cases()[: max(1, args.limit)]
    structured_ok = 0
    citation_total = 0
    citation_valid = 0
    abstentions = 0
    errors = []

    for case in cases:
        result = analyse_case(
            case["case_id"],
            store,
            model_service=model,
            ollama_client=client,
            use_fixture_band=not model.available,
        )
        summary = result.copilot_summary
        if summary["generation_mode"] == "ollama" and not summary.get("error"):
            structured_ok += 1
        if summary.get("error"):
            errors.append({"case_id": case["case_id"], "error": summary["error"]})
        abstentions += int(bool(summary["abstained"]))
        allowed = {f"DOC:{r['source']}:p{r['page']}" for r in result.retrieved_evidence}
        for citation in summary.get("citations", []):
            citation_total += 1
            citation_valid += int(citation in allowed)

    metrics = {
        "model": settings.ollama_model,
        "cases": len(cases),
        "structured_output_success_rate": structured_ok / len(cases),
        "citation_validity_rate": citation_valid / citation_total if citation_total else None,
        "abstention_rate": abstentions / len(cases),
        "semantic_groundedness": "not automatically scored; inspect generated outputs or add an independent local judge",
        "unsupported_claim_rate": "not automatically scored; citation validity is measured separately",
        "errors": errors,
    }
    path = ROOT / args.output
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
