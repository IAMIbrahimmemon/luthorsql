"""Replay logged sql.py commands from the 2026-10-10/12 Opus runs through the rebuilt tool and compare outputs.
Each logged shell command is run as written inside a sandbox whose tools/sql.py is the rebuilt tool (the old
`cd <old scratchpad>` is replaced). Commands that touch other files (cat, redirects, python -c, chunks/answers) or
other databases than BIRD dev are skipped.

  python3 scripts/system2/replay_check.py <calls.json> <dev_db_root> [--show N]
"""
import json, re, subprocess, sys, tempfile, difflib
from pathlib import Path

TOOL = Path(__file__).resolve().parent / "sql_tool.py"
calls = json.load(open(sys.argv[1]))
root = Path(sys.argv[2]).resolve()
show = int(sys.argv[sys.argv.index("--show") + 1]) if "--show" in sys.argv else 3
dev = {p.name for p in root.iterdir() if p.is_dir()}
box = Path(tempfile.mkdtemp(prefix="sqlreplay_"))
(box / "tools").mkdir()
(box / "tools/sql.py").symlink_to(TOOL)
same = diff = skip = 0
examples = []
for c in calls:
    cmd = c["cmd"]
    dbs = set(re.findall(r"sql\.py\s+\"?(\w+)", cmd)) | set(re.findall(r'sql\.py "?\$?\{?\w*\}?\s*', ""))
    used = set(re.findall(r"\b(" + "|".join(sorted(dev | {"x" * 40})) + r")\b", cmd))
    if (re.search(r"\bcat\b|>>|> *[\w/.]|python3 -c|chunks|answers|\bjq\b|head -c|wc |python3 (?!tools/sql\.py)\S+\.py", cmd) or not used
            or re.search(r"^Exit code|\(eval\):|persisted-output", c["out"] or "", re.M) or re.search(r"grep -n", cmd)
            or re.search(r"tools/sql\.py\s+(?!\"?\$|" + "|".join(dev) + r")\w+", cmd)):
        skip += 1
        continue
    cmd = re.sub(r"cd /private/tmp/[^;&]*?/scratchpad/[\w/]+", f"cd {box}", cmd)
    want = "\n".join(l for l in (c["out"] or "").splitlines() if not re.fullmatch(r"Welcome \w+, zsh", l)).strip()  # login-shell greeting
    r = subprocess.run(["bash", "-c", cmd], capture_output=True, text=True, timeout=300,
                       env={"LUTHOR_SQL_DB_ROOT": str(root), "PATH": "/usr/bin:/bin:/opt/homebrew/bin", "LANG": "en_US.UTF-8", "LC_ALL": "en_US.UTF-8"}, cwd=box)
    got = (r.stdout + r.stderr).strip()
    if re.search(r"No such file or directory|can't open file", r.stderr) and not re.search(r"No such file", want):
        skip += 1  # the logged command needs a file from the deleted 2026-10-10/12 scratchpad: not a tool comparison
        continue
    if got == want:
        same += 1
    else:
        diff += 1
        if len(examples) < show:
            d = "\n".join(list(difflib.unified_diff(want.splitlines(), got.splitlines(), "logged", "rebuilt", lineterm="", n=0))[:12])
            examples.append((cmd[-150:], d))
print(f"replayed {same + diff}: identical {same}, different {diff}; skipped {skip}")
for e in examples:
    print("=== cmd tail:", e[0]); print(e[1])
