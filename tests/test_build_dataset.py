"""Specification for the dataset-building stages.

Every test runs on a tiny hand-written DataFrame, so it is obvious what the
right answer is. That is the point: a fixture small enough to reason about
catches logic errors that 217,007 real rows would hide.
"""

import pandas as pd
import pytest

from scripts.build_dataset import class_weights, clean, deduplicate, split, verify


@pytest.fixture
def raw():
    """Seven rows, each one exercising a different filter.

    row 0  clean C function
    row 1  lang "CPP", and the body is padded with whitespace
    row 2  same function as row 0, only reindented  -> a near-duplicate
    row 3  lang "Java"          -> dropped, not C/C++
    row 4  commit_id ""         -> dropped, no group to split on
    row 5  commit_id None       -> dropped, no group to split on
    row 6  body is whitespace   -> dropped, nothing to learn from
    """
    return pd.DataFrame(
        {
            "project": ["chrome"] * 7,
            "commit_id": ["c1", "c1", "c2", "c2", "", None, "c3"],
            "CWE ID": ["CWE-119"] * 7,
            "lang": ["C", "CPP", "C++", "Java", "C", "C", "C"],
            "func_before": [
                "int a() { return 1; }",
                "  int b() { return 2; }  ",
                "int a() {\n    return 1;\n}",
                "int d() { return 4; }",
                "int e() { return 5; }",
                "int f() { return 6; }",
                "   ",
            ],
            "func_after": ["x"] * 7,
            "vul": [1, 0, 0, 1, 0, 1, 0],
        }
    )


@pytest.fixture
def grouped():
    """60 commits x 5 unique functions, 12 of the commits vulnerable.

    StratifiedGroupKFold needs comfortably more groups than folds, and enough
    groups of each class to stratify on. Every body is unique, so deduplicate()
    keeps all 300 rows and only adds the _fingerprint column that verify()
    needs.
    """
    rows = [
        {
            "project": f"p{group % 7}",
            "commit_id": f"c{group:03d}",
            "CWE ID": "CWE-119",
            "lang": "C",
            "func_before": f"int f{group}_{fn}() {{ return {group * 10 + fn}; }}",
            "func_after": "x",
            "vul": 1 if group % 5 == 0 else 0,
        }
        for group in range(60)
        for fn in range(5)
    ]
    return deduplicate(pd.DataFrame(rows))


class TestClean:
    def test_keeps_only_the_three_valid_rows(self, raw):
        assert len(clean(raw)) == 3

    def test_normalizes_cpp_to_plus_plus(self, raw):
        assert set(clean(raw)["lang"]) == {"C", "C++"}

    def test_drops_languages_outside_the_map(self, raw):
        assert "Java" not in set(clean(raw)["lang"])

    def test_strips_the_function_body(self, raw):
        out = clean(raw)
        assert "int b() { return 2; }" in set(out["func_before"])

    def test_drops_rows_with_a_blank_group(self, raw):
        """A row with no commit_id cannot be placed in a grouped split."""
        out = clean(raw)
        assert out["commit_id"].notna().all()
        assert (out["commit_id"].astype(str).str.strip() != "").all()

    def test_returns_a_fresh_index(self, raw):
        assert list(clean(raw).index) == [0, 1, 2]

    def test_does_not_mutate_its_input(self, raw):
        """Silent mutation of a caller's frame is a debugging nightmare."""
        clean(raw)
        assert len(raw) == 7
        assert raw.loc[1, "lang"] == "CPP"


class TestDeduplicate:
    def test_collapses_a_reindented_copy(self, raw):
        """rows 0 and 2 are the same function, differently formatted."""
        assert len(deduplicate(clean(raw))) == 2

    def test_keeps_the_first_occurrence(self, raw):
        out = deduplicate(clean(raw))
        kept = out[out["func_before"].str.contains("return 1")]
        assert len(kept) == 1
        assert kept.iloc[0]["commit_id"] == "c1"

    def test_adds_a_fingerprint_column(self, raw):
        assert "_fingerprint" in deduplicate(clean(raw)).columns

    def test_keeps_every_distinct_function(self):
        df = pd.DataFrame(
            {
                "commit_id": ["c1", "c2", "c3"],
                "func_before": ["int a();", "int b();", "int c();"],
                "vul": [0, 1, 0],
            }
        )
        assert len(deduplicate(df)) == 3

    def test_does_not_mutate_its_input(self, raw):
        cleaned = clean(raw)
        deduplicate(cleaned)
        assert "_fingerprint" not in cleaned.columns


class TestSplit:
    """The one guarantee that matters: a group may not span two splits."""

    def test_no_group_spans_two_splits(self, grouped):
        splits = split(grouped, seed=42)
        for group in grouped["commit_id"].unique():
            hits = sum(group in set(part["commit_id"]) for part in splits.values())
            assert hits == 1, f"{group} appears in {hits} splits"

    def test_proportions_are_roughly_80_10_10(self, grouped):
        splits = split(grouped, seed=42)
        n = len(grouped)
        assert 0.75 <= len(splits["train"]) / n <= 0.85
        assert 0.05 <= len(splits["val"]) / n <= 0.15
        assert 0.05 <= len(splits["test"]) / n <= 0.15

    def test_every_row_lands_in_exactly_one_split(self, grouped):
        """No row may be lost, and none may be duplicated."""
        splits = split(grouped, seed=42)
        assert sum(len(p) for p in splits.values()) == len(grouped)
        seen: set[str] = set()
        for part in splits.values():
            fingerprints = set(part["_fingerprint"])
            assert not (seen & fingerprints)
            seen |= fingerprints

    def test_label_rate_is_similar_across_splits(self, grouped):
        """Stratification: otherwise F1 is not comparable between runs."""
        splits = split(grouped, seed=42)
        rates = [part["vul"].mean() for part in splits.values()]
        assert max(rates) - min(rates) < 0.10

    def test_is_deterministic_for_a_given_seed(self, grouped):
        a = split(grouped, seed=42)
        b = split(grouped, seed=42)
        for name in a:
            assert list(a[name]["_fingerprint"]) == list(b[name]["_fingerprint"])

    def test_a_different_seed_gives_a_different_split(self, grouped):
        a = split(grouped, seed=42)
        b = split(grouped, seed=7)
        assert set(a["test"]["commit_id"]) != set(b["test"]["commit_id"])


class TestVerify:
    def test_returns_all_zeros_for_a_clean_split(self, grouped):
        checks = verify(split(grouped, seed=42))
        assert checks
        assert all(value == 0 for value in checks.values())

    def test_raises_when_a_group_spans_train_and_test(self, grouped):
        """The check the first version of this project did not have."""
        splits = split(grouped, seed=42)
        leaked = splits["test"]["commit_id"].iloc[0]
        splits["train"] = pd.concat(
            [splits["train"], splits["test"][splits["test"]["commit_id"] == leaked]],
            ignore_index=True,
        )
        with pytest.raises(AssertionError, match="commit_id_overlap_train_test"):
            verify(splits)

    def test_raises_when_the_same_code_appears_on_both_sides(self, grouped):
        """Same body, different commit id -- the group check alone misses this."""
        splits = split(grouped, seed=42)
        smuggled = splits["test"].iloc[[0]].copy()
        smuggled["commit_id"] = "brand-new-commit"
        splits["train"] = pd.concat([splits["train"], smuggled], ignore_index=True)
        with pytest.raises(AssertionError, match="code_overlap_train_test"):
            verify(splits)

    def test_raises_rather_than_warns(self, grouped):
        """A printed warning gets ignored; a raise stops the pipeline."""
        splits = split(grouped, seed=42)
        splits["val"] = pd.concat([splits["val"], splits["test"]], ignore_index=True)
        with pytest.raises(AssertionError):
            verify(splits)


class TestClassWeights:
    def test_balances_the_two_classes(self, grouped):
        """n_c * w_c must be equal for both classes -- that is the whole point."""
        train = split(grouped, seed=42)["train"]
        weights = class_weights(train)
        counts = train["vul"].value_counts()
        mass = [counts[label] * weights[str(label)] for label in counts.index]
        assert abs(mass[0] - mass[1]) < 1e-6

    def test_follows_the_inverse_frequency_formula(self, grouped):
        train = split(grouped, seed=42)["train"]
        weights = class_weights(train)
        for label, n in train["vul"].value_counts().items():
            assert weights[str(label)] == pytest.approx(len(train) / (2 * n))

    def test_the_rare_class_gets_the_larger_weight(self, grouped):
        train = split(grouped, seed=42)["train"]
        weights = class_weights(train)
        assert weights["1"] > weights["0"]
