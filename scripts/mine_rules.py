"""Measure whether each Layer 1 rule earns its place.

A rule that flags functions at the dataset's own vulnerable rate tells you
nothing: it may as well fire at random. The number that matters is **lift** --
the vulnerable rate among the functions a rule flags, divided by the base rate.

    lift = P(vul | rule fires) / P(vul)

Lift near 1.0 means the rule carries no signal, and every hit it produces is a
false positive a reviewer has to read. Such a rule should be dropped, however
plausible it sounds.

This reads TRAIN ONLY. Choosing rules by looking at val or test would tune the
scanner on the data it is later judged by, which is the same leakage the split
was rebuilt to remove -- only this time introduced by hand.

Usage:
    python scripts/mine_rules.py
    python scripts/mine_rules.py --split val    # after the rules are frozen
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.pipeline.regex_layer import RULES, scan

LABEL_COL = "vul"
CODE_COL = "func_before"


def measure(df: pd.DataFrame) -> pd.DataFrame:
    """One row per rule: how often it fires, and how much signal it carries."""
    findings = [scan(code) for code in df[CODE_COL]]
    hit_names = [{f.rule.name for f in per_row} for per_row in findings]

    labels = df[LABEL_COL].to_numpy()
    base = labels.mean()
    n_vul = int(labels.sum())

    rows = []
    for rule in RULES:
        mask = pd.Series([rule.name in hits for hits in hit_names])
        matched = int(mask.sum())
        if not matched:
            rows.append(
                {
                    "rule": rule.name,
                    "cwe": rule.cwe,
                    "severity": rule.severity,
                    "matched": 0,
                    "vul_rate": 0.0,
                    "lift": 0.0,
                    "vul_covered": 0,
                    "recall": 0.0,
                }
            )
            continue

        hit_labels = labels[mask.to_numpy()]
        vul_rate = float(hit_labels.mean())
        rows.append(
            {
                "rule": rule.name,
                "cwe": rule.cwe,
                "severity": rule.severity,
                "matched": matched,
                "vul_rate": vul_rate,
                "lift": vul_rate / base,
                "vul_covered": int(hit_labels.sum()),
                "recall": float(hit_labels.sum()) / n_vul,
            }
        )

    return pd.DataFrame(rows).sort_values("lift", ascending=False)


def report_layer_alone(df: pd.DataFrame) -> None:
    """Treat "any rule fires" as a classifier, to see the layer's own ceiling."""
    predicted = pd.Series([bool(scan(code)) for code in df[CODE_COL]])
    actual = df[LABEL_COL].astype(bool).reset_index(drop=True)

    tp = int((predicted & actual).sum())
    fp = int((predicted & ~actual).sum())
    fn = int((~predicted & actual).sum())

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0

    print("\nLayer 1 alone, as a classifier:")
    print(f"  flags      {int(predicted.sum()):>7,} / {len(df):,} functions")
    print(f"  precision  {precision:.4f}   (of the functions it flags, this share are vulnerable)")
    print(f"  recall     {recall:.4f}   (of the vulnerable functions, this share get flagged)")
    print(f"  F1         {f1:.4f}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", default="train", choices=["train", "val", "test"])
    parser.add_argument("--data-dir", default="data/processed")
    args = parser.parse_args()

    if args.split != "train":
        print(f"!! Reading '{args.split}'. Do not choose rules from this.\n")

    path = ROOT / args.data_dir / f"{args.split}.parquet"
    df = pd.read_parquet(path)
    base = df[LABEL_COL].mean()
    print(f"{path.name}: {len(df):,} functions, base vulnerable rate {base:.2%}")

    table = measure(df)

    print(f"\n{'rule':<30}{'cwe':<10}{'matched':>9}{'vul rate':>10}{'lift':>7}{'recall':>9}")
    print("-" * 75)
    for row in table.itertuples():
        print(
            f"{row.rule:<30}{row.cwe:<10}{row.matched:>9,}"
            f"{row.vul_rate:>9.1%}{row.lift:>7.2f}{row.recall:>9.1%}"
        )

    print("\nlift < 1.0 means the rule is worse than guessing at the base rate.")
    weak = table[table["lift"] < 1.2]
    if len(weak):
        print("Carrying little or no signal:")
        for row in weak.itertuples():
            print(f"  {row.rule}  (lift {row.lift:.2f}, {row.matched:,} hits)")

    report_layer_alone(df)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
