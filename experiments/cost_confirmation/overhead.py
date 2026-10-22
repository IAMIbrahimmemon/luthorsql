"""Fixed per-chunk overhead of the Claude Code harness vs per-question cost (exploratory, not pre-registered).

Each subagent pays a fixed cost (harness system prompt, tool schema, first schema read) plus a cost per question.
Fit chunk_cost = a + b * n_questions per arm by least squares, over every scored chunk (RESERVED runs 1-3 and the
external test). With a = 0 (an API deployment that amortises the fixed prompt over many questions), the saving vs B1 is
recomputed from the per-question slopes.
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "external_cascade"))
import score  # noqa: E402
import score_ext  # noqa: E402

S = Path(sys.argv[1])  # session dir holding scratchpad/ and tasks/
pts = defaultdict(list)
boxes = [("res", S / "scratchpad/s2res"), ("res", S / "scratchpad/s2res_r2"), ("res", S / "scratchpad/s2res_r3")]
for tag, box in boxes:
    U = score.usage(S / "tasks", box)
    for arm, d in U.items():
        for ch, t in d.items():
            n = sum(1 for _ in open(box / f"chunks/{arm}/{ch}.jsonl"))
            pts[(tag, arm)].append((n, score.dollars(t)))
box = S / "scratchpad/s2ext"
U = score_ext.usage(S / "tasks", box)
for arm, d in U.items():
    for ch, t in d.items():
        n = sum(1 for _ in open(box / f"chunks/{arm}/{ch}.jsonl"))
        pts[("ext", arm)].append((n, score.dollars(t)))
out = {}
for k, v in sorted(pts.items()):
    n, c = np.array(v).T
    A = np.vstack([np.ones_like(n), n]).T
    (a, b), *_ = np.linalg.lstsq(A, c, rcond=None)
    out[f"{k[0]}_{k[1]}"] = {"chunks": len(v), "fixed_per_chunk": round(a, 4), "per_question": round(b, 5),
                             "fixed_share_of_cost": round(a * len(v) / c.sum(), 3)}
print(json.dumps(out, indent=1))
Path(__file__).with_name("overhead.json").write_text(json.dumps(out, indent=1))
