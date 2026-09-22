from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Settings:
    root_dir: Path = ROOT_DIR
    synthetic_dir: Path = ROOT_DIR / "data" / "synthetic_demo"
    models_dir: Path = ROOT_DIR / "models"
    plots_dir: Path = ROOT_DIR / "plots"
    outputs_dir: Path = ROOT_DIR / "outputs"
    audit_db: Path = ROOT_DIR / "audit" / "audit.db"
    vector_store_dir: Path = ROOT_DIR / "outputs" / "vector_store"
    ollama_url: str = os.getenv("OLLAMA_URL", "http://localhost:11434")
    ollama_model: str = os.getenv("OLLAMA_MODEL", "qwen2.5:3b")
    rag_backend: str = os.getenv("RAG_BACKEND", "tfidf")
    embedding_model: str = os.getenv(
        "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
    )
    income_tolerance_pct: float = float(os.getenv("INCOME_TOLERANCE_PCT", "0.05"))
    duration_tolerance_months: int = int(os.getenv("DURATION_TOLERANCE_MONTHS", "2"))


settings = Settings()
