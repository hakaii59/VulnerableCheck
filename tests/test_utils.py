"""Specification for the code-normalization helpers.

These tests were written before src/utils.py existed. They describe what the
functions must do; the implementation then has to satisfy them.
"""
import pytest

from src.utils import code_fingerprint, normalize_code

class TestNormalizeCode:
    def test_collapse_repeated_spaces(self):
        assert normalize_code("int  a") == "int a"

    def test_collapses_tabs_and_newlines(self):
        assert normalize_code("int a;\n\tint b;") == "int a; int b;"

    def test_strips_leading_and_trailing_whitespace(self):
        assert normalize_code("\n  int a;  \n") == "int a;"

    def test_leaves_already_normalized_code_untouched(self):
        code = "int add(int a, int b) { return a + b; }"
        assert normalize_code(code) == code

    def test_is_idempotent(self):
        """Normalizing twice must equal normalizing once."""
        messy = "void  f( int   x )\n{\n\treturn x;\n}"
        once = normalize_code(messy)
        assert normalize_code(once) == once

    @pytest.mark.parametrize("value", [None, 42, float("nan")])
    def test_survives_non_string_input(self, value):
        """Real dataframes contain NaN. This must not raise."""
        assert isinstance(normalize_code(value), str)

class TestCodeFingerprint:
    def test_returns_64_hex_characters(self):
        fp = code_fingerprint("int a;")
        assert len(fp) == 64
        assert all(c in "0123456789abcdef" for c in fp)

    def test_is_deterministic(self):
        assert code_fingerprint("int a;") == code_fingerprint("int a;")

    def test_identical_code_collides(self):
        code = "int add(int a, int b) { return a + b; }"
        assert code_fingerprint(code) == code_fingerprint(code)

    def test_reformatted_code_collides(self):
        """The whole point: indentation must not create a new fingerprint."""
        tight = "int add(int a, int b) { return a + b; }"
        loose = "int add(int a,   int b) {\n    return a + b;\n}"
        assert code_fingerprint(tight) == code_fingerprint(loose)

    def test_different_code_does_not_collide(self):
        assert code_fingerprint("int a;") != code_fingerprint("int b;")

    def test_renaming_a_variable_does_NOT_collide(self):
        """Documents a deliberate limit.

        This helper only normalizes whitespace, so a renamed copy survives it.
        Catching those needs the structural skeleton from the leakage audit,
        which is a measurement tool, not part of the pipeline.
        """
        a = "int add(int a, int b) { return a + b; }"
        b = "int add(int x, int y) { return x + y; }"
        assert code_fingerprint(a) != code_fingerprint(b)