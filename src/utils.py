"""Shared helpers used by the notebooks, the scripts and the tests."""

from __future__ import annotations

import hashlib
import re

_WHITESPACE = re.compile(r"\s+")


def normalize_code(code: str) -> str:
    """Collapse every run of whitespace into a single space.

    Two functions that differ only in indentation or line breaks normalize to
    the same string:

        int add(int a,   int b) {    return a + b;     }
        int add(int a, int b) { return a + b; }

    Non-string input is coerced, because real dataframes contain NaN.
    """
    return _WHITESPACE.sub(" ", str(code)).strip()


def code_fingerprint(code: str) -> str:
    """SHA-256 of the normalized source, used to detect duplicates.

    Reformatted copies collide; renamed copies do not. Catching renamed copies
    needs a structural skeleton, which is a measurement tool rather than part
    of the pipeline -- see notebooks/analysis_data_leakage.ipynb.
    """
    return hashlib.sha256(normalize_code(code).encode("utf-8")).hexdigest()
