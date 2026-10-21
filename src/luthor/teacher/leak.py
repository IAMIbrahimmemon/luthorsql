"""The prompt gate: nothing goes to an external API if it matches a private term or a secret.

Same sources as scripts/vast/leakcheck.sh: private terms live outside the repo in ~/.config/luthor/leak_patterns
(override with $LEAK_PATTERNS), one ERE per line as WORK=<regex> / CODE_WORK=<regex> / PERSON=<regex>, matched
case-insensitively as whole words. Prompts are data (public schemas and questions), so like leakcheck.sh on data
paths they are checked for WORK terms and secrets; CODE_WORK and PERSON are code-only checks there, because those
terms are ordinary words in public datasets. The patterns themselves are never printed or logged: a hit reports
only which class matched.
"""
from __future__ import annotations

import os
import re
from pathlib import Path

# Copied from leakcheck.sh SECRET (ERE -> Python: [[:space:]] -> \s).
SECRET = (r"BEGIN [A-Z ]*PRIVATE KEY-----(\\\\n|\\n|\s)*[A-Za-z0-9+/]{40}|\bhf_[A-Za-z0-9]{30,}"
          r"|\bsk-(ant-|proj-)?[A-Za-z0-9_-]{30,}|tskey-[a-z]+-[A-Za-z0-9]{10,}|\bAKIA[0-9A-Z]{16}\b"
          r"|\bghp_[A-Za-z0-9]{30,}|\bxox[abp]-[0-9A-Za-z-]{10,}")
AWS_EXAMPLE = re.compile(r"AKIA[0-9A-Z]{9}EXAMPLE", re.I)  # documented public placeholder keys
_POSIX = {"[[:space:]]": r"\s", "[[:alpha:]]": "[A-Za-z]", "[[:digit:]]": r"\d", "[[:alnum:]]": "[A-Za-z0-9]",
          "[[:upper:]]": "[A-Z]", "[[:lower:]]": "[a-z]", r"\<": r"\b", r"\>": r"\b"}


def patterns_path() -> Path:
    return Path(os.environ.get("LEAK_PATTERNS") or Path.home() / ".config/luthor/leak_patterns")


def _ere_to_py(p: str) -> str:
    for a, b in _POSIX.items():
        p = p.replace(a, b)
    return p


def load_patterns(path: Path | None = None, classes=("WORK",)) -> dict[str, re.Pattern]:
    """{class: compiled regex}. SECRET always; the private classes only when the file is readable."""
    out = {"SECRET": re.compile(SECRET, re.I)}
    path = path or patterns_path()
    try:
        lines = path.read_text().splitlines()
    except OSError:
        return out
    for cls in classes:
        for ln in lines:  # first line of each class, like leakcheck.sh's `sed -n ... | head -1`
            if ln.startswith(cls + "="):
                pat = ln[len(cls) + 1:].strip()
                if pat:
                    out[cls] = re.compile(_ere_to_py(pat), re.I)
                break
    return out


class LeakGate:
    """gate.hit(text) -> None when clean, else the matching class name (never the matched text or pattern)."""

    def __init__(self, path: Path | None = None, classes=("WORK",), require_private: bool = True):
        self.pats = load_patterns(path, classes)
        self.private = [c for c in self.pats if c != "SECRET"]
        if require_private and not self.private:
            raise RuntimeError("leak gate: no private patterns loaded (expected WORK= in the leak_patterns file); "
                               "refusing to send anything. Pass require_private=False only in tests.")

    def hit(self, text: str) -> str | None:
        for cls, rx in self.pats.items():
            for m in rx.finditer(text):
                if cls == "SECRET" and AWS_EXAMPLE.fullmatch(m.group(0)):
                    continue
                return cls
        return None
