# Limitations

- The Home Credit dataset is not bundled. The original training artefacts must be regenerated locally.
- Synthetic applicant profiles are scored, when model artefacts exist, by starting from training-feature medians and overriding a small set of interpretable fields. This is a demonstration bridge, not a claim that the synthetic profiles are representative of real Home Credit applicants.
- The deterministic document extractor is designed for the labelled demo templates. A production system would need stronger parsing, OCR, document classification and validation.
- The default retrieval backend is TF-IDF so the project runs offline and in CI. Sentence-transformer embeddings are available but require a local model download.
- Ollama quality depends on the chosen local model and hardware. The project does not claim measured GenAI quality until the local evaluation is run.
- PDF support extracts embedded text; scanned-image OCR is not implemented.
- Fairness analysis remains limited to the existing gender and age-group evaluation produced by `train.py`.
- Thresholds are decision-support thresholds derived from the model score distribution, not production lending policy.
- SQLite and local files are appropriate for an enterprise-style prototype, not for a scaled multi-user deployment.
- Authentication, role-based access control, key management, encrypted storage, production monitoring and formal model-risk governance are outside this portfolio prototype.
