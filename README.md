# VulnerableCheck

Detecting vulnerabilities in C/C++ functions. Built from scratch as a learning
project, one phase at a time.

**Status:** Phase 1 of 9 — exploratory analysis and a leakage audit of Big-Vul.

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
| 0 | environment, repository skeleton |
| 1 | EDA and the leakage audit |
| 2 | leakage-free train/val/test split, with tests |
| 3 | Layer 1 — regex rules chosen by measured lift |
| 4 | Layer 2 — Tree-sitter AST validation |
| 5 | Layer 3 — fine-tuned GraphCodeBERT |
| 6 | verdict logic combining the three layers |
| 7 | evaluation, threshold tuning on validation only |
| 8 | ONNX export and INT8 quantization |
| 9 | CI workflow and documentation |

## License

MIT — see [LICENSE](LICENSE).
