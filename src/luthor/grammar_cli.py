"""`--grammar {v3,v4}` for the System-1 scripts (train_sq2, rerank_sq2, eval_sq2, dump_heads, scripts/rl/rl_train).

luthor.sq.plan reads LUTHOR_GRAMMAR ONCE, at import, so the grammar has to be in the environment before the first
luthor.sq import; worker processes (multiprocessing spawn / fork) then inherit it. A script calls `preset()` at its
top, under `if __name__ == "__main__":`, BEFORE importing anything from luthor.sq / luthor.sq2:

    if __name__ == "__main__":
        from luthor.grammar_cli import preset   # this module imports nothing from luthor.sq
        preset()

The grammar is chosen in this order: --grammar on the command line, else LUTHOR_GRAMMAR, else (scripts that load a
trained run) the "grammar" of that run's args.json (missing -> v3, every run before the flag), else v3. Then, after
parsing, `resolve()` (training: the grammar recorded in args.json) or `check_run()` (loading a run: the process's
grammar must be the run's; a conflicting --grammar is an error).

This module must not import luthor.sq at module level.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

CHOICES = ("v3", "v4")
HELP = ("grammar version (sets LUTHOR_GRAMMAR before luthor is imported; inherited by worker processes). Default: "
        "LUTHOR_GRAMMAR, else the loaded run's args.json grammar (missing = v3), else v3. v4 runs train on "
        "data/plans_v4/... with --vocab data/plans_v4/templates_union.json")


def _flag(argv: list[str], name: str) -> str | None:
    for i, x in enumerate(argv):
        if x == name and i + 1 < len(argv):
            return argv[i + 1]
        if x.startswith(name + "="):
            return x.split("=", 1)[1]
    return None


def _norm(g: str | None, where: str) -> str:
    g = (g or "v3").strip().lower() or "v3"
    if g not in CHOICES:
        raise SystemExit(f"{where}: grammar {g!r}, expected one of {CHOICES}")
    return g


def run_grammar(run: Path | str) -> str:
    """The grammar a run was trained with: args.json "grammar"; v3 when absent (runs before the flag) or unreadable."""
    try:
        g = json.loads((Path(run) / "args.json").read_text()).get("grammar")
    except (OSError, ValueError):
        g = None
    if g in (None, "", "None"):
        return "v3"
    return _norm(g, f"{run}/args.json")


def dump_grammar(dump: Path | str) -> str | None:
    """The grammar a head dump was written with (manifest heads.grammar), None if unreadable."""
    try:
        return json.loads((Path(dump) / "manifest.json").read_text())["heads"]["grammar"]
    except (OSError, ValueError, KeyError, TypeError):
        return None


def preset(argv: list[str] | None = None, run_flags: tuple[str, ...] = ("--run",)) -> str:
    """Choose the grammar from argv / env / the run's args.json and export LUTHOR_GRAMMAR. Call before luthor.sq."""
    argv = sys.argv[1:] if argv is None else list(argv)
    g = _flag(argv, "--grammar")
    src = "--grammar"
    if g is None and os.environ.get("LUTHOR_GRAMMAR", "").strip():
        g, src = os.environ["LUTHOR_GRAMMAR"], "LUTHOR_GRAMMAR"
    if g is None:
        for f in run_flags:
            r = _flag(argv, f)
            if r is None:
                continue
            g, src = (dump_grammar(r) if f == "--dump" else run_grammar(r)), f"{f} {r}"
            if g not in CHOICES:  # a dump without a readable v3/v4 grammar: fall through to the default
                g = None
            if g is not None:
                break
    g = _norm(g, src)
    loaded = sys.modules.get("luthor.sq.plan")
    if loaded is not None and loaded.GRAMMAR != g:
        raise SystemExit(f"grammar {g} ({src}) but luthor.sq.plan was already imported with {loaded.GRAMMAR}: "
                         "set it before any luthor.sq import")
    os.environ["LUTHOR_GRAMMAR"] = g
    return g


def add_argument(ap) -> None:
    ap.add_argument("--grammar", choices=CHOICES, default=None, help=HELP)


def resolve(requested: str | None) -> str:
    """The grammar this process runs (luthor.sq.plan.GRAMMAR); an explicit --grammar must agree with it."""
    from luthor.sq.plan import GRAMMAR
    if requested is not None and requested != GRAMMAR:
        raise SystemExit(f"--grammar {requested}, but luthor.sq.plan was imported with grammar {GRAMMAR} "
                         "(LUTHOR_GRAMMAR); run the script directly or set LUTHOR_GRAMMAR to match")
    return GRAMMAR


def check_run(run: Path | str, requested: str | None = None) -> str:
    """Loading a trained run: its grammar (args.json, missing -> v3) must be this process's grammar, and an explicit
    --grammar must agree with the run. Returns the grammar."""
    from luthor.sq.plan import GRAMMAR
    want = run_grammar(run)
    if requested is not None and requested != want:
        raise SystemExit(f"--grammar {requested}, but {run} was trained with grammar {want} (its args.json)")
    if GRAMMAR != want:
        raise SystemExit(f"{run} was trained with grammar {want}, but this process runs grammar {GRAMMAR} "
                         f"(LUTHOR_GRAMMAR={os.environ.get('LUTHOR_GRAMMAR')!r}); pass --grammar {want} or unset it")
    return want
