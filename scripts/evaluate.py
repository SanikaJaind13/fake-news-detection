"""Evaluate the pipeline on a labelled claim set (needs real API keys).

Usage:  python scripts/evaluate.py [--limit N] [--file data/sample_claims.json]
Writes results to data/evaluation_results.csv and prints accuracy, a confusion
matrix and per-class precision/recall/F1 (useful for the project report).
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.config import load_settings  # noqa: E402
from src.errors import FakeNewsError  # noqa: E402
from src.models import VERDICTS  # noqa: E402
from src.pipeline import analyze  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", default=str(ROOT / "data" / "sample_claims.json"))
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    settings = load_settings(max_claims=1)
    if settings.missing_keys():
        sys.exit("Missing API keys: " + ", ".join(settings.missing_keys()))
    data = json.loads(Path(args.file).read_text(encoding="utf-8"))
    if args.limit:
        data = data[: args.limit]

    rows, matrix = [], {t: {p: 0 for p in VERDICTS} for t in VERDICTS}
    for i, item in enumerate(data, 1):
        try:
            res = analyze(item["claim"], settings, save=False)
            pred, conf = res.overall_verdict, res.overall_confidence
        except FakeNewsError as exc:
            print(f"[{i}] ERROR: {exc}")
            pred, conf = "Insufficient Evidence", 0.0
        matrix[item["label"]][pred] += 1
        rows.append({"claim": item["claim"], "gold": item["label"], "predicted": pred, "confidence": conf})
        print(f"[{i}/{len(data)}] gold={item['label']:<22} pred={pred:<22} conf={conf:.2f}")
        time.sleep(1)  # be gentle with free-tier rate limits

    correct = sum(matrix[v][v] for v in VERDICTS)
    print(f"\nAccuracy: {correct}/{len(rows)} = {correct / max(len(rows), 1):.1%}\n")
    print(f"{'gold \\ pred':<24}" + "".join(f"{v[:12]:>14}" for v in VERDICTS))
    for t in VERDICTS:
        print(f"{t:<24}" + "".join(f"{matrix[t][p]:>14}" for p in VERDICTS))
    print("\nPer-class metrics:")
    for v in VERDICTS:
        tp = matrix[v][v]
        fp = sum(matrix[t][v] for t in VERDICTS if t != v)
        fn = sum(matrix[v][p] for p in VERDICTS if p != v)
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        print(f"  {v:<22} P={prec:.2f} R={rec:.2f} F1={f1:.2f}")

    out = ROOT / "data" / "evaluation_results.csv"
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["claim", "gold", "predicted", "confidence"])
        w.writeheader()
        w.writerows(rows)
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
