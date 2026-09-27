"""Build the train/val/test splits from Big-Vul, without data leakage.

Big-Vul emits one row per function touched by a CVE fix commit, so its 217,007
rows come from roughly 4,000 commits -- a median of 24 functions each, up to
1,692. Those functions share a repo, an author, a patch and often a body.

Splitting per row therefore puts the same commit on both sides. Measured on the
split published on the Hub: 99.8% of its test commits also appear in train, and
76.5% of test function bodies have a token-exact structural twin there -- worse
for the vulnerable class (83.6%) than the safe one (76.1%). See
notebooks/analysis_data_leakage.ipynb.

So the published split is discarded and rebuilt here, grouped on commit_id.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils import code_fingerprint

HF_DATASET = "benjis/bigvul"
GROUP_COL = "commit_id"
LABEL_COL = "vul"
CODE_COL = "func_before"
LANG_MAP = {"C": "C", "CPP": "C++", "C++": "C++"}
KEEP_COLS = ["project", "commit_id", "CWE ID", "lang", "func_before", "func_after", "vul"]

#: 10 folds -> one is test, one is val, the remaining eight are train.
N_FOLDS = 10


def load_raw() -> pd.DataFrame:
    """Concatenate the Hub's three splits, because they are rebuilt anyway."""
    # Imported here so the tests never need the datasets package.
    from datasets import concatenate_datasets, load_dataset

    print(f"Loading '{HF_DATASET}' ...")
    dsd = load_dataset(HF_DATASET)
    print("  Hub splits:", {k: len(v) for k, v in dsd.items()})
    df = concatenate_datasets([dsd["train"], dsd["validation"], dsd["test"]]).to_pandas()
    print(f"  Combined: {len(df):,} rows")
    return df


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize the language column, then drop rows the pipeline cannot use.

    Every filter reports how many rows it removed. On Big-Vul the last three
    remove nothing at all -- but that is only knowable by measuring, and a
    different dataset, or a newer revision of this one, will not be so tidy.
    """
    df = df.copy()
    df["lang"] = df["lang"].map(LANG_MAP)
    df[CODE_COL] = df[CODE_COL].astype(str).str.strip()

    conditions = {
        "language outside C / C++": df["lang"].isin(["C", "C++"]),
        "empty function body": df[CODE_COL].str.len() > 0,
        f"missing {GROUP_COL}": (
            df[GROUP_COL].notna() & (df[GROUP_COL].astype(str).str.strip() != "")
        ),
    }

    keep = pd.Series(True, index=df.index)
    for reason, condition in conditions.items():
        print(f"  dropped {int((~condition).sum()):>7,}  {reason}")
        keep &= condition

    out = df[keep].reset_index(drop=True)
    print(f"Cleaning: {len(df):,} -> {len(out):,} rows")
    return out


def deduplicate(df: pd.DataFrame) -> pd.DataFrame:
    """Drop functions that are identical once whitespace is normalized.

    This must happen *before* the split. Deduplicating afterwards is useless:
    by then the copies already sit on opposite sides.
    """
    df = df.copy()
    df["_fingerprint"] = [code_fingerprint(c) for c in df[CODE_COL]]

    before = len(df)
    out = df.drop_duplicates(subset=["_fingerprint"], keep="first").reset_index(drop=True)
    dropped = before - len(out)
    share = dropped / before if before else 0.0
    print(f"Deduplication: dropped {dropped:,} ({share:.1%}), {len(out):,} remain")
    return out


def split(df: pd.DataFrame, seed: int = 42, group_col: str = GROUP_COL) -> dict[str, pd.DataFrame]:
    """80/10/10, grouped so no group spans two splits, stratified on the label.

    `groups=` is what removes the leakage: every row sharing a commit_id lands
    wholly inside one fold. `y=` keeps the label rate even across folds, so an
    F1 measured on test is comparable to one measured on val.

    n_splits=10 gives 10% per fold, which is exactly the 80/10/10 wanted -- no
    second split call, no nested proportions to get wrong.
    """
    sgkf = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
    folds = [test_idx for _, test_idx in sgkf.split(df, df[LABEL_COL], groups=df[group_col])]

    test_idx, val_idx = folds[0], folds[1]
    train_idx = np.setdiff1d(np.arange(len(df)), np.concatenate([test_idx, val_idx]))

    return {
        "train": df.iloc[train_idx].reset_index(drop=True),
        "val": df.iloc[val_idx].reset_index(drop=True),
        "test": df.iloc[test_idx].reset_index(drop=True),
    }


def verify(splits: dict[str, pd.DataFrame], group_col: str = GROUP_COL) -> dict[str, int]:
    """Fail loudly on any leakage. This is the check the first version lacked.

    Two families of check, because either alone is insufficient. The group check
    misses a body that was copied under a different commit id; the code check
    misses two different functions from the same patch.

    It raises rather than prints: a warning in a 200-line log gets scrolled past,
    and the cost of missing it is a headline number that is quietly false.
    """
    pairs = {
        "train_test": (splits["train"], splits["test"]),
        "train_val": (splits["train"], splits["val"]),
        "val_test": (splits["val"], splits["test"]),
    }

    checks: dict[str, int] = {}
    for name, (a, b) in pairs.items():
        checks[f"{group_col}_overlap_{name}"] = len(set(a[group_col]) & set(b[group_col]))
        checks[f"code_overlap_{name}"] = len(set(a["_fingerprint"]) & set(b["_fingerprint"]))

    print("\nLeakage checks (all must be 0):")
    for name, value in checks.items():
        print(f"  {'OK  ' if value == 0 else 'FAIL'} {name}: {value}")

    failed = {name: value for name, value in checks.items() if value}
    if failed:
        raise AssertionError(f"Leakage detected: {failed}")
    return checks


def class_weights(train: pd.DataFrame) -> dict[str, float]:
    """Inverse-frequency weights: w(c) = N / (2 * n_c).

    The factor 2 is the number of classes, which makes the weights average to 1
    and keeps the loss on the same scale as an unweighted run. The property that
    matters is n_c * w(c) = N / 2 for both classes -- each class contributes the
    same total mass, whatever the imbalance.
    """
    counts = train[LABEL_COL].value_counts().to_dict()
    return {str(label): len(train) / (2 * n) for label, n in counts.items()}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", default="data/processed", help="where to write the parquet files")
    parser.add_argument("--seed", type=int, default=42, help="split seed")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    df = deduplicate(clean(load_raw()))
    print(
        f"\nGrouping on '{GROUP_COL}': {df[GROUP_COL].nunique():,} distinct groups "
        f"for {len(df):,} functions "
        f"(median {df.groupby(GROUP_COL).size().median():.0f} per group)"
    )

    splits = split(df, args.seed)

    print("\nSplit sizes:")
    summary = {}
    for name, part in splits.items():
        n_vul = int(part[LABEL_COL].sum())
        summary[name] = {
            "rows": len(part),
            "share": round(len(part) / len(df), 4),
            "vulnerable": n_vul,
            "vulnerable_rate": round(float(part[LABEL_COL].mean()), 4),
            "groups": int(part[GROUP_COL].nunique()),
        }
        print(
            f"  {name:5s}: {len(part):7,} rows ({len(part) / len(df):5.1%})  "
            f"vul=1: {n_vul:5,} ({part[LABEL_COL].mean():.2%})  "
            f"{GROUP_COL}s: {part[GROUP_COL].nunique():,}"
        )

    # Nothing is written before this line, so a leak leaves no usable artefacts.
    checks = verify(splits)

    weights = class_weights(splits["train"])
    (out_dir / "class_weights.json").write_text(json.dumps(weights, indent=2), encoding="utf-8")
    print(f"\nClass weights: {weights}")

    for name, part in splits.items():
        path = out_dir / f"{name}.parquet"
        part[KEEP_COLS].to_parquet(path, index=False)
        print(f"  Saved {path.name}: {part[KEEP_COLS].shape}")

    report = {
        "dataset": HF_DATASET,
        "grouped_on": GROUP_COL,
        "seed": args.seed,
        "n_folds": N_FOLDS,
        "total_rows_after_cleaning": len(df),
        "distinct_groups": int(df[GROUP_COL].nunique()),
        "splits": summary,
        "leakage_checks": checks,
        "class_weights": weights,
    }
    (out_dir / "split_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("  Saved split_report.json")

    print(f"\nDone. Splits are {GROUP_COL}-disjoint and duplicate-free.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
