"""Specification for Layer 1.

Two things are being pinned down. First, that each rule fires on the pattern it
claims to describe. Second -- and this is where a regex scanner usually goes
wrong -- that it does *not* fire on text that only looks like code: comments,
string literals, and identifiers that merely contain a dangerous name.
"""


import pytest

from src.pipeline.regex_layer import HIGH, RULES, mask_noise, scan


def names(code):
    return {finding.rule.name for finding in scan(code)}


class TestRulesAreWellFormed:
    def test_every_name_is_unique(self):
        assert len({rule.name for rule in RULES}) == len(RULES)

    @pytest.mark.parametrize("rule", RULES, ids=lambda r: r.name)
    def test_rule_carries_the_metadata_later_layers_need(self, rule):
        assert rule.cwe.startswith("CWE-")
        assert rule.severity in {"high", "medium"}
        assert rule.func_names
        assert rule.message


class TestScan:
    def test_safe_code_produces_nothing(self):
        assert scan("int add(int a, int b) { return a + b; }") == []

    def test_finds_strcpy(self):
        assert "unbounded-string-copy" in names("void f(char *s) { strcpy(buf, s); }")

    def test_reports_the_cwe(self):
        finding = scan("void f(char *s) { strcpy(buf, s); }")[0]
        assert finding.rule.cwe == "CWE-120"
        assert finding.rule.severity == HIGH

    def test_reports_the_line_number(self):
        code = "void f(char *s)\n{\n    char buf[8];\n    strcpy(buf, s);\n}"
        assert scan(code)[0].line == 4

    def test_reports_the_source_line_as_a_snippet(self):
        code = "void f(char *s)\n{\n    strcpy(buf, s);\n}"
        assert scan(code)[0].snippet == "strcpy(buf, s);"

    def test_findings_come_back_in_source_order(self):
        code = "void f()\n{\n    gets(a);\n    strcpy(b, c);\n}"
        assert [f.line for f in scan(code)] == [3, 4]

    def test_a_longer_identifier_is_not_a_match(self):
        """my_strcpy_checked() is somebody's safe wrapper, not CWE-120."""
        assert scan("void f() { my_strcpy_checked(a, b); }") == []


class TestIgnoresNoise:
    def test_ignores_a_line_comment(self):
        assert scan("void f() { // TODO: replace strcpy here\n}") == []

    def test_ignores_a_block_comment(self):
        assert scan("void f() { /* strcpy(a, b) is banned */ }") == []

    def test_ignores_a_string_literal(self):
        assert scan('void f() { log("do not use strcpy("); }') == []

    def test_line_numbers_survive_a_multiline_comment(self):
        """Masking must keep newlines, or every later line number shifts."""
        code = "void f()\n{\n    /* a\n       long\n       comment */\n    gets(buf);\n}"
        assert scan(code)[0].line == 6

    def test_a_real_call_after_a_comment_is_still_found(self):
        code = "void f() { /* strcpy */ gets(buf); }"
        assert names(code) == {"inherently-unsafe-gets"}


class TestMaskNoise:
    def test_preserves_length(self):
        code = 'int a; // note\nchar *s = "hi";\n'
        assert len(mask_noise(code)) == len(code)

    def test_preserves_line_breaks(self):
        code = "a\n/* x\ny */\nb\n"
        assert mask_noise(code).count("\n") == code.count("\n")

    def test_leaves_real_code_alone(self):
        assert mask_noise("gets(buf);") == "gets(buf);"


class TestIndividualRules:
    @pytest.mark.parametrize(
        ("code", "expected"),
        [
            ("void f() { strcat(a, b); }", "unbounded-string-copy"),
            ("void f() { sprintf(buf, \"%d\", n); }", "unbounded-sprintf"),
            ("void f() { gets(buf); }", "inherently-unsafe-gets"),
            ("void f() { system(cmd); }", "command-injection-sink"),
            ("void f() { popen(cmd, \"r\"); }", "command-injection-sink"),
            ("void f() { sscanf(in, \"%s\", buf); }", "scanf-unbounded-string"),
            ("void f() { char *p = alloca(n); }", "alloca-on-stack"),
            ("void f() { strncpy(d, s, n); }", "strncpy-may-not-terminate"),
            ("void f() { p = malloc(n * size); }", "allocation-size-arithmetic"),
            ("void f() { p = realloc(p, n + 1); }", "allocation-size-arithmetic"),
            ("void f() { p = malloc(strlen(s)); }", "strlen-allocation-off-by-one"),
            ("void f(char *fmt) { printf(fmt); }", "non-literal-format-string"),
            ("void f() { char *p = tmpnam(NULL); }", "insecure-temporary-file"),
            ("void f() { memcpy(dst, src, n); }", "raw-memory-copy"),
            ("void f() { int n = atoi(s); }", "atoi-cannot-report-errors"),
        ],
    )
    def test_rule_fires(self, code, expected):
        assert expected in names(code)

    @pytest.mark.parametrize(
        "code",
        [
            'void f() { snprintf(buf, sizeof buf, "%d", n); }',
            'void f() { sscanf(in, "%31s", buf); }',
            "void f() { p = malloc(n); }",
            "void f() { p = calloc(n, size); }",
            'void f() { printf("%d", n); }',
            'void f() { printf(_("%d"), n); }',
            "void f() { n = strtol(s, &end, 10); }",
        ],
    )
    def test_safe_idiom_does_not_fire(self, code):
        assert scan(code) == []

    def test_the_correct_strlen_idiom_is_not_an_off_by_one(self):
        """malloc(strlen(s) + 1) is right, and must not be reported as CWE-131.

        allocation-size-arithmetic does still fire here, because the size
        contains a "+". That is a deliberate false positive: Layer 1 is tuned
        for recall, and Layers 2 and 3 exist to discard hits like this one.
        """
        assert names("void f() { p = malloc(strlen(s) + 1); }") == {
            "allocation-size-arithmetic"
        }
