from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import settings
from src.pipeline import build_demo_store


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=["tfidf", "sentence-transformers", "auto"], default=settings.rag_backend)
    parser.add_argument("--output", default="outputs/vector_store")
    args = parser.parse_args()
    store = build_demo_store(args.backend)
    store.save(ROOT / args.output)
    print(f"Saved {len(store.chunks)} chunks using backend={store.backend} to {ROOT / args.output}")


if __name__ == "__main__":
    main()
