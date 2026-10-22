"""Scan every subagent transcript of one sandbox for access outside its own chunk (scripts/system2/leak_check.py).

  python3 experiments/cost_confirmation/leak_scan.py --box <sandbox> --tasks <dir> --out <jsonl>
"""
import argparse
import glob
import json
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ap = argparse.ArgumentParser()
ap.add_argument("--box", required=True)
ap.add_argument("--tasks", required=True)
ap.add_argument("--out", required=True)
a = ap.parse_args()
pat = re.compile(re.escape(a.box) + r"/chunks/(B0|B1|B2|C)/([A-Za-z0-9_]+_\d+)\.jsonl")
fs, mp = [], {}
for f in glob.glob(a.tasks + "/*.output"):
    m = pat.search(open(f).read(20000))
    if m:
        fs.append(f)
        mp[f.split("/")[-1]] = m.group(1) + "/" + m.group(2)
dup = [k for k, c in Counter(mp.values()).items() if c > 1]
print(len(fs), "transcripts", dict(Counter(v.split("/")[0] for v in mp.values())), "duplicates:", dup)
out = subprocess.run([sys.executable, str(ROOT / "scripts/system2/leak_check.py"), a.box, *fs], capture_output=True, text=True).stdout
rows = [json.loads(l) for l in out.splitlines()]
for r in rows:
    r["chunk"] = mp.get(r["transcript"])
Path(a.out).write_text("".join(json.dumps(r) + "\n" for r in rows))
print("clean", sum(r["clean"] for r in rows), "of", len(rows))
for r in rows:
    if not r["clean"]:
        print(" ", r["chunk"], json.dumps(r["flags"])[:400])
