from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import requests


@dataclass(frozen=True)
class OllamaResult:
    ok: bool
    content: str
    model: str
    error: str | None = None


class OllamaClient:
    def __init__(self, base_url: str, model: str, timeout: int = 180) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout

    def health(self) -> bool:
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=3)
            return response.ok
        except requests.RequestException:
            return False

    def chat_json(self, system: str, user: str, schema: dict[str, Any] | None = None) -> OllamaResult:
        payload = {
            "model": self.model,
            "stream": False,
            "format": schema or "json",
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "options": {"temperature": 0, "num_predict": 700},
        }
        try:
            response = requests.post(
                f"{self.base_url}/api/chat",
                json=payload,
                timeout=self.timeout,
            )
            response.raise_for_status()
            data: dict[str, Any] = response.json()
            content = data.get("message", {}).get("content", "")
            if not content:
                return OllamaResult(False, "", self.model, "Ollama returned no content")
            # Parse once here so malformed JSON is detected close to the boundary.
            json.loads(content)
            return OllamaResult(True, content, self.model)
        except (requests.RequestException, ValueError, json.JSONDecodeError) as exc:
            return OllamaResult(False, "", self.model, str(exc))
