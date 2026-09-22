from __future__ import annotations

import json
from pathlib import Path

from src.config import settings


def cases_path() -> Path:
    return settings.synthetic_dir / "cases.json"


def load_cases(path: str | Path | None = None) -> list[dict]:
    path = Path(path) if path else cases_path()
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload["cases"] if isinstance(payload, dict) else payload


def get_case(case_id: str, path: str | Path | None = None) -> dict:
    for case in load_cases(path):
        if case["case_id"] == case_id:
            return case
    raise KeyError(f"Unknown case_id: {case_id}")


def case_document_dir(case_id: str) -> Path:
    return settings.synthetic_dir / "documents" / case_id
