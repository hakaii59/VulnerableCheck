"""Layer 1: find suspicious calls with regular expressions.

This is the cheap first pass. It needs no model, no GPU and no training, it runs
in microseconds, and when it fires it can point at a line number -- which a
classifier cannot.

What it cannot do is judge context. `strcpy(dst, src)` right after a length
check is fine; the same call with attacker-controlled `src` is CWE-120. Regex
sees the call, not the guard. That is what Layer 2 (AST) and Layer 3 (the
classifier) are for, so the rules here are deliberately generous: their job is
recall, not precision.

Each rule carries `func_names` so Layer 2 can look for those calls in the parse
tree rather than re-deriving them from the pattern.

How this rule set was chosen
----------------------------
Every rule was measured on the train split with `scripts/mine_rules.py`, which
reports lift -- the vulnerable rate among the functions a rule flags, over the
5.32% base rate. All thirteen sit between 2.0x and 5.5x. Together they flag
6,113 of 130,910 functions at precision 0.120 and recall 0.105.

Two rules fire on nothing in Big-Vul and are kept anyway, because the pattern
*is* the defect and the cost of keeping them is zero: `gets()` cannot be used
safely at all, and `tmpnam`/`mktemp` hand back a name before creating the file.
Big-Vul is mined from Chrome, Linux and FFmpeg, which abandoned both long ago --
that says nothing about the code this scanner will be pointed at.

Three candidates measured well and were still rejected, because lift alone is
not evidence:

- `sizeof(*p)` -- lift 2.35, recall 4.9%. But `malloc(sizeof(*p))` is the
  *recommended* idiom, and CWE-467 is `sizeof(ptr)`, not `sizeof(*ptr)`. The
  pattern does not describe the defect it would be named after. Its lift comes
  from marking low-level memory code, which is merely where CVEs live.
- `goto err/fail/out` -- lift 1.83, recall 6.0%. A proxy for complex error
  handling in C, not a defect. It would flag correct code by the thousand.
- `syslog(priority, fmt)` added to the format-string rule -- it *lowered* lift
  from 2.94 to 2.13, because the first argument is a priority constant, so the
  rule fired on every correct call.

A pattern that correlates with buggy code is not the same as a pattern that is
a bug. Only the second kind belongs here.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

#: Severities that let a confirmed finding decide the verdict on its own.
HIGH = "high"
MEDIUM = "medium"


#: How much source after a call site `confirm` may look at.
CONFIRM_WINDOW = 200


@dataclass(frozen=True)
class Rule:
    cwe: str
    name: str
    severity: str
    pattern: re.Pattern[str]
    func_names: tuple[str, ...]
    message: str
    #: Optional second pattern, matched against the *unmasked* statement.
    #: `pattern` runs on masked source, so a rule can never be triggered by a
    #: comment or a literal. But some rules need to read a literal: an
    #: unbounded scanf is identified by the "%s" in its format string. Those
    #: rules find the call with `pattern` and inspect the argument with this.
    confirm: re.Pattern[str] | None = None


@dataclass(frozen=True)
class RegexFinding:
    rule: Rule
    line: int
    snippet: str


RULES: tuple[Rule, ...] = (
    Rule(
        cwe="CWE-120",
        name="unbounded-string-copy",
        severity=HIGH,
        pattern=re.compile(r"\b(?:strcpy|strcat)\s*\("),
        func_names=("strcpy", "strcat"),
        message="copies until a NUL byte, with no bound on the destination",
    ),
    Rule(
        cwe="CWE-120",
        name="unbounded-sprintf",
        severity=HIGH,
        pattern=re.compile(r"\b(?:sprintf|vsprintf)\s*\("),
        func_names=("sprintf", "vsprintf"),
        message="writes a formatted string with no size limit; use snprintf",
    ),
    Rule(
        cwe="CWE-242",
        name="inherently-unsafe-gets",
        severity=HIGH,
        pattern=re.compile(r"\bgets\s*\("),
        func_names=("gets",),
        message="gets() cannot be used safely; it was removed in C11",
    ),
    Rule(
        cwe="CWE-78",
        name="command-injection-sink",
        severity=HIGH,
        pattern=re.compile(r"\b(?:system|popen|execlp|execvp)\s*\("),
        func_names=("system", "popen", "execlp", "execvp"),
        message="passes a string to a shell or resolves a program via PATH",
    ),
    Rule(
        cwe="CWE-1240",
        name="scanf-unbounded-string",
        severity=HIGH,
        pattern=re.compile(r"\b[fs]?scanf\s*\("),
        func_names=("scanf", "fscanf", "sscanf"),
        message="%s without a field width reads an unbounded token",
        # "%s" is a plain conversion; "%31s" and "%*s" both carry a bound, and
        # neither contains the literal "%s".
        confirm=re.compile(r"%s"),
    ),
    Rule(
        cwe="CWE-676",
        name="alloca-on-stack",
        severity=MEDIUM,
        pattern=re.compile(r"\balloca\s*\("),
        func_names=("alloca",),
        message="a large or attacker-influenced size overflows the stack",
    ),
    Rule(
        cwe="CWE-170",
        name="strncpy-may-not-terminate",
        severity=MEDIUM,
        pattern=re.compile(r"\bstrncpy\s*\("),
        func_names=("strncpy",),
        message="strncpy leaves the destination unterminated when it is full",
    ),
    Rule(
        cwe="CWE-190",
        name="allocation-size-arithmetic",
        severity=MEDIUM,
        pattern=re.compile(r"\b(?:malloc|realloc)\s*\([^;{}]{0,200}?[*+]"),
        func_names=("malloc", "realloc"),
        message="arithmetic in an allocation size can wrap around",
    ),
    Rule(
        cwe="CWE-131",
        name="strlen-allocation-off-by-one",
        severity=HIGH,
        pattern=re.compile(r"\bmalloc\s*\(\s*strlen\s*\([^;{}]{0,80}?\)\s*\)"),
        func_names=("malloc",),
        message="strlen() excludes the NUL terminator, so this needs + 1",
    ),
    Rule(
        cwe="CWE-134",
        name="non-literal-format-string",
        severity=HIGH,
        # `pattern` runs on masked source, where a string literal has been
        # blanked to spaces. So a format argument that survives as an
        # identifier was never a literal, and an attacker may control it.
        # `_( )` is the gettext idiom wrapping a literal, so it is excluded.
        pattern=re.compile(r"\bprintf\s*\(\s*(?!_\()[A-Za-z_]"),
        func_names=("printf",),
        message="the format string is a variable, not a literal",
    ),
    Rule(
        cwe="CWE-377",
        name="insecure-temporary-file",
        severity=HIGH,
        pattern=re.compile(r"\b(?:tmpnam|mktemp|tempnam)\s*\("),
        func_names=("tmpnam", "mktemp", "tempnam"),
        message="returns a name before creating the file, so it can be raced",
    ),
    Rule(
        cwe="CWE-787",
        name="raw-memory-copy",
        severity=MEDIUM,
        pattern=re.compile(r"\b(?:memcpy|memmove|bcopy)\s*\("),
        func_names=("memcpy", "memmove", "bcopy"),
        message="the length must be bounded by the destination size",
    ),
    Rule(
        cwe="CWE-252",
        name="atoi-cannot-report-errors",
        severity=MEDIUM,
        pattern=re.compile(r"\b(?:atoi|atol|atof)\s*\("),
        func_names=("atoi", "atol", "atof"),
        message="returns 0 for invalid input; use strtol and check errno",
    ),
)

_COMMENT_OR_STRING = re.compile(
    r"""
      //[^\n]*                  # line comment
    | /\*.*?\*/                 # block comment
    | "(?:[^"\\\n]|\\.)*"       # string literal
    | '(?:[^'\\\n]|\\.)*'       # char literal
    """,
    re.VERBOSE | re.DOTALL,
)


def mask_noise(code: str) -> str:
    """Blank out comments and literals, preserving offsets and line breaks.

    A rule must not fire on `// remember to replace strcpy` or on the string
    "strcpy". Deleting those regions would shift every later offset, so each one
    is replaced by spaces of the same length instead -- newlines are kept, so a
    match's line number is still correct.
    """

    def blank(match: re.Match[str]) -> str:
        return "".join("\n" if ch == "\n" else " " for ch in match.group(0))

    return _COMMENT_OR_STRING.sub(blank, code)


def _statement_at(text: str, start: int) -> str:
    """The rest of the statement beginning at `start`, capped in length."""
    window = text[start : start + CONFIRM_WINDOW]
    end = window.find(";")
    return window if end == -1 else window[:end]


def scan(code: str) -> list[RegexFinding]:
    """Return every rule match in `code`, in source order."""
    text = str(code)
    masked = mask_noise(text)
    lines = text.splitlines()

    findings: list[RegexFinding] = []
    for rule in RULES:
        for match in rule.pattern.finditer(masked):
            if rule.confirm is not None and not rule.confirm.search(
                _statement_at(text, match.start())
            ):
                continue
            line = masked.count("\n", 0, match.start()) + 1
            snippet = lines[line - 1].strip() if line - 1 < len(lines) else ""
            findings.append(RegexFinding(rule=rule, line=line, snippet=snippet))

    findings.sort(key=lambda f: (f.line, f.rule.name))
    return findings
