# VulnerableCheck

Detecting vulnerabilities in C/C++ functions. Built from scratch as a learning
project, one phase at a time.

**Status:** Phase 3 of 9 — the data pipeline and the regex layer are in, with 91 tests.

## Why the published Big-Vul split is discarded

Four tests, from cheapest to most expensive, each harder to fool than the last:

| Test | Question | Result |
|---|---|---|
| exact | Any function byte-identical across train and test? | 75.7% of test functions |
| normalized | ... ignoring indentation and line breaks? | 76.1% |
| **group** | Is the unit the data was *collected in* split apart? | **99.8% of test commits are also in train** |
| structural | ... ignoring variable names and literals? | **76.5% of test bodies have a twin in train** |

Broken down by class, the structural test is worse for the positive class —
83.6% of vulnerable test functions have a token-exact structural twin in train,
against 76.1% of safe ones. Precision, recall and F1 are all computed on that
class, so the leakage inflates exactly the numbers a report would quote.

The cause: Big-Vul emits one row per function touched by a CVE fix commit.
217,007 rows come from roughly 4,000 commits — a median of 24 functions per
commit, up to 1,692. The published split is random *per row*, so a 24-function
commit survives intact with probability 0.2^24 ≈ 1.7e-17. Effectively every
commit is torn in half.

See [`notebooks/analysis_data_leakage.ipynb`](notebooks/analysis_data_leakage.ipynb).

## The rebuilt split

`python scripts/build_dataset.py` turns 217,007 raw rows into 163,636 after
whitespace-normalized deduplication, then splits them on `commit_id`:

| | rows | share | vulnerable | commits |
|---|---|---|---|---|
| train | 130,910 | 80.0% | 6,962 (5.32%) | 3,208 |
| val | 16,363 | 10.0% | 871 (5.32%) | 389 |
| test | 16,363 | 10.0% | 870 (5.32%) | 394 |

`verify()` asserts six overlap checks and raises rather than warns, and it runs
before anything is written, so a leak leaves no usable parquet behind.

## Layer 1: regex rules

Thirteen rules, each measured on train with `scripts/mine_rules.py`, which
reports lift — the vulnerable rate among the functions a rule flags, over the
5.32% base rate. All eleven that fire sit between 2.0x and 5.5x. Together they
flag 6,113 of 130,910 functions at precision 0.120 and recall 0.105.

Three candidates measured well and were rejected anyway: `sizeof(*p)` (lift
2.35, but that is the recommended idiom and CWE-467 is `sizeof(ptr)`),
`goto err` (lift 1.83, a proxy for complex error handling rather than a defect),
and adding `syslog` to the format-string rule — which lowered lift, since its
first argument is a priority constant. Two rules that fire on nothing are kept,
because `gets()` and `mktemp()` have no safe use and cost no false positives.

Layer 1 alone misses 89.5% of vulnerabilities. That is the case for Layer 3.

## Layout

| Path | Contents |
|---|---|
| `notebooks/` | analysis, narrated step by step |
| `scripts/` | runnable pipeline stages |
| `src/` | shared library code |
| `tests/` | pytest suite |
| `samples/` | small C/C++ files used as fixtures |
| `data/` | not tracked — rebuilt by the pipeline |
| `models/` | not tracked — checkpoints |

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Roadmap

| Phase | Deliverable |
|---|---|
| 0 | environment, repository skeleton — done |
| 1 | EDA and the leakage audit — done |
| 2 | leakage-free train/val/test split, with tests — done |
| 3 | Layer 1 — regex rules chosen by measured lift — done |
| 4 | Layer 2 — Tree-sitter AST validation |
| 5 | Layer 3 — fine-tuned GraphCodeBERT |
| 6 | verdict logic combining the three layers |
| 7 | evaluation, threshold tuning on validation only |
| 8 | ONNX export and INT8 quantization |
| 9 | CI workflow and documentation |

## License

MIT — see [LICENSE](LICENSE).
