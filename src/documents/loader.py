from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class DocumentPage:
    source: str
    page: int
    text: str
    document_type: str | None = None
    case_id: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class TextChunk:
    chunk_id: str
    source: str
    page: int
    text: str
    document_type: str | None = None
    case_id: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)


def load_document(path: str | Path, case_id: str | None = None) -> list[DocumentPage]:
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".txt":
        return [DocumentPage(path.name, 1, path.read_text(encoding="utf-8"), suffix[1:], case_id)]
    if suffix == ".pdf":
        from pypdf import PdfReader

        reader = PdfReader(str(path))
        return [
            DocumentPage(path.name, i + 1, page.extract_text() or "", "pdf", case_id)
            for i, page in enumerate(reader.pages)
        ]
    if suffix == ".docx":
        from docx import Document

        doc = Document(str(path))
        text = "\n".join(p.text for p in doc.paragraphs if p.text.strip())
        return [DocumentPage(path.name, 1, text, "docx", case_id)]
    raise ValueError(f"Unsupported document format: {suffix}")


def load_case_documents(case_dir: str | Path, case_id: str) -> list[DocumentPage]:
    case_dir = Path(case_dir)
    pages: list[DocumentPage] = []
    for path in sorted(case_dir.iterdir()):
        if path.is_file() and path.suffix.lower() in {".txt", ".pdf", ".docx"}:
            pages.extend(load_document(path, case_id=case_id))
    return pages


def chunk_pages(
    pages: Iterable[DocumentPage],
    chunk_size: int = 850,
    overlap: int = 120,
) -> list[TextChunk]:
    if chunk_size <= overlap:
        raise ValueError("chunk_size must be greater than overlap")

    chunks: list[TextChunk] = []
    for page in pages:
        text = " ".join(page.text.split())
        if not text:
            continue
        start = 0
        n = 0
        while start < len(text):
            end = min(len(text), start + chunk_size)
            chunk_text = text[start:end]
            chunk_id = f"{page.case_id or 'case'}::{page.source}::p{page.page}::c{n}"
            chunks.append(
                TextChunk(
                    chunk_id=chunk_id,
                    source=page.source,
                    page=page.page,
                    text=chunk_text,
                    document_type=page.document_type,
                    case_id=page.case_id,
                )
            )
            if end == len(text):
                break
            start = end - overlap
            n += 1
    return chunks
