from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evaluation.evaluator import write_evaluation


def main() -> None:
    parser = argparse.ArgumentParser(description="Run measurable synthetic underwriting evaluation.")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--output", default="outputs/evaluation/deterministic_metrics.json")
    args = parser.parse_args()
    metrics = write_evaluation(ROOT / args.output, top_k=args.top_k)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
