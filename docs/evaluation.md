# Evaluation

The evaluation suite uses eight fictional underwriting cases with labelled document facts, expected inconsistency codes, expected workflow routes and retrieval targets. It is intentionally small and deterministic so it can run in CI without Home Credit data, Ollama or model downloads.

Run:

```powershell
python scripts/run_evaluation.py
```

The script writes `outputs/evaluation/deterministic_metrics.json`.

## Observed deterministic results in this packaged build

These values were produced by actually running the evaluation script in the build environment. They measure only the labelled synthetic test set and must not be interpreted as real-world underwriting accuracy.

| Metric | Observed |
|---|---:|
| Synthetic cases | 8 |
| Labelled retrieval queries | 9 |
| Document field extraction accuracy | 1.0000 (30/30 labelled fields) |
| Retrieval hit rate@3 | 1.0000 (9/9 queries) |
| Inconsistency precision | 1.0000 |
| Inconsistency recall | 1.0000 |
| Inconsistency F1 | 1.0000 |
| Workflow routing accuracy | 1.0000 |
| Escalation accuracy | 1.0000 |
| Unsafe autonomous credit-decision count | 0 |
| Tool-call success rate for expected valid calls | 1.0000 |
| Audit event completeness rate | 1.0000 |
| Deterministic citation validity rate | 1.0000 |

Perfect scores here are expected because the rules and fixtures are deliberately controlled. They demonstrate that the evaluation harness works; they do not prove generalisation.

## Local Ollama evaluation

After Ollama is running:

```powershell
python scripts/run_ollama_evaluation.py
```

This measures structured-output success and citation validity for the configured local model. Semantic groundedness and unsupported-claim rate are left unscored rather than fabricated because an independent judge or manual rubric would be required for a credible value.

## What is not measured here

- Real-world underwriting performance of the synthetic workflow layer.
- Business savings, productivity gains or analyst time saved.
- Human user satisfaction.
- Real document OCR quality.
- Production latency, throughput or reliability.
- Any new ML uplift beyond the historical verified model results.
