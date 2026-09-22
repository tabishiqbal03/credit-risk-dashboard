# Responsible AI Controls

This project demonstrates governance controls in code rather than relying only on documentation.

| Risk | Implemented control |
|---|---|
| LLM makes a lending decision | The workflow planner never emits a final approve/decline action; final decisions are recorded only through the human-decision tool. |
| Unsupported generated claims | Generation is preceded by retrieval and deterministic checks; source IDs are provided and returned citations are validated. |
| Missing evidence | Explicit `MISSING_*` rules trigger escalation and evidence-request proposals. |
| Numeric inconsistency | Income and employment-duration comparisons are deterministic and thresholded. |
| Arbitrary tool execution | Tool names are matched against a fixed registry; there is no shell or arbitrary Python tool. |
| Malformed tool arguments | Pydantic validates every tool payload. |
| Sensitive action without approval | Evidence requests, review-case creation and final-decision recording require explicit approval. |
| Ollama unavailable | The copilot falls back to a labelled deterministic summary rather than inventing an LLM result. |
| Model artefacts unavailable | Risk probability is shown as unavailable. A synthetic routing fixture can be enabled only as an explicitly labelled demo mode. |
| Cross-case retrieval leakage | Retrieval supports deterministic case-ID filtering before similarity ranking. |
| Untraceable workflow | SQLite audit events record structured payloads for analysis and actions. |
| Real PII in demo data | All committed underwriting documents are fictional synthetic examples. |

## Human oversight

The model produces a risk probability and an approve/review/reject band. These are decision-support signals. The LLM can summarise evidence and the workflow layer can prepare or route work, but the final credit decision is a human action.

## Abstention

The copilot abstains when it cannot form a grounded evidence narrative. Missing model artefacts and missing supporting documents are surfaced explicitly. An unavailable model is not replaced with a fabricated probability.

## Model and prompt injection boundaries

Retrieved document text is included as evidence only. The Ollama system instruction says document content is untrusted and cannot redefine the assistant's role or tool permissions. Tool execution is separate from generation and is never driven by an arbitrary generated function name.
