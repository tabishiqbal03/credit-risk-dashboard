from __future__ import annotations

import json
import pickle
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable

import numpy as np

from src.documents.loader import TextChunk


@dataclass(frozen=True)
class SearchResult:
    chunk_id: str
    source: str
    page: int
    text: str
    score: float
    case_id: str | None = None
    document_type: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)


class LocalVectorStore:
    """Small local retrieval store with semantic embeddings or an offline TF-IDF fallback."""

    def __init__(
        self,
        backend: str = "tfidf",
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
    ) -> None:
        if backend not in {"tfidf", "sentence-transformers", "auto"}:
            raise ValueError("backend must be tfidf, sentence-transformers, or auto")
        self.requested_backend = backend
        self.backend = backend
        self.embedding_model = embedding_model
        self.chunks: list[TextChunk] = []
        self._matrix = None
        self._vectorizer = None
        self._encoder = None

    def _load_encoder(self):
        if self._encoder is None:
            from sentence_transformers import SentenceTransformer

            self._encoder = SentenceTransformer(self.embedding_model)
        return self._encoder

    def build(self, chunks: Iterable[TextChunk]) -> "LocalVectorStore":
        self.chunks = list(chunks)
        if not self.chunks:
            raise ValueError("Cannot build a vector store with no chunks")

        backend = self.requested_backend
        if backend == "auto":
            try:
                self._load_encoder()
                backend = "sentence-transformers"
            except Exception:
                backend = "tfidf"
        self.backend = backend

        texts = [c.text for c in self.chunks]
        if backend == "sentence-transformers":
            encoder = self._load_encoder()
            self._matrix = np.asarray(
                encoder.encode(texts, normalize_embeddings=True, show_progress_bar=False),
                dtype=np.float32,
            )
            self._vectorizer = None
        else:
            from sklearn.feature_extraction.text import TfidfVectorizer

            self._vectorizer = TfidfVectorizer(
                lowercase=True,
                ngram_range=(1, 2),
                min_df=1,
                stop_words="english",
            )
            self._matrix = self._vectorizer.fit_transform(texts)
        return self

    def search(
        self,
        query: str,
        top_k: int = 4,
        case_id: str | None = None,
    ) -> list[SearchResult]:
        if self._matrix is None or not self.chunks:
            raise RuntimeError("Vector store has not been built or loaded")
        if top_k < 1:
            raise ValueError("top_k must be at least 1")

        candidate_idx = [
            i for i, chunk in enumerate(self.chunks)
            if case_id is None or chunk.case_id == case_id
        ]
        if not candidate_idx:
            return []

        if self.backend == "sentence-transformers":
            encoder = self._load_encoder()
            q = np.asarray(
                encoder.encode([query], normalize_embeddings=True, show_progress_bar=False)[0],
                dtype=np.float32,
            )
            scores = self._matrix[candidate_idx] @ q
        else:
            q = self._vectorizer.transform([query])
            scores = (self._matrix[candidate_idx] @ q.T).toarray().ravel()

        order = np.argsort(-scores)[:top_k]
        results: list[SearchResult] = []
        for rank_idx in order:
            idx = candidate_idx[int(rank_idx)]
            chunk = self.chunks[idx]
            results.append(
                SearchResult(
                    chunk_id=chunk.chunk_id,
                    source=chunk.source,
                    page=chunk.page,
                    text=chunk.text,
                    score=round(float(scores[int(rank_idx)]), 6),
                    case_id=chunk.case_id,
                    document_type=chunk.document_type,
                )
            )
        return results

    def save(self, directory: str | Path) -> None:
        if self._matrix is None:
            raise RuntimeError("Build the store before saving it")
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        metadata = {
            "backend": self.backend,
            "embedding_model": self.embedding_model,
            "chunks": [c.to_dict() for c in self.chunks],
        }
        (directory / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        if self.backend == "sentence-transformers":
            np.save(directory / "embeddings.npy", self._matrix)
        else:
            with (directory / "tfidf.pkl").open("wb") as f:
                pickle.dump({"vectorizer": self._vectorizer, "matrix": self._matrix}, f)

    @classmethod
    def load(cls, directory: str | Path) -> "LocalVectorStore":
        directory = Path(directory)
        metadata = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
        store = cls(metadata["backend"], metadata.get("embedding_model", "sentence-transformers/all-MiniLM-L6-v2"))
        store.backend = metadata["backend"]
        store.chunks = [TextChunk(**item) for item in metadata["chunks"]]
        if store.backend == "sentence-transformers":
            store._matrix = np.load(directory / "embeddings.npy")
        else:
            with (directory / "tfidf.pkl").open("rb") as f:
                payload = pickle.load(f)
            store._vectorizer = payload["vectorizer"]
            store._matrix = payload["matrix"]
        return store
