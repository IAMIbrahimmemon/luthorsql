"""Cost of every agent arm with the Claude Code harness's own context removed (exploratory, not pre-registered).

Each Claude Code subagent session carries context that a plain-API agent would not: the harness system prompt and
tool schemas, the session's CLAUDE.md and connector listings, and reminders injected between turns. That context is
written to the prompt cache once per session and re-read on every later turn, so arms that run more sessions (B0, B1)
pay more of it than the cascade. This script reprices every session from its transcript with that context removed:

  turn 1: everything in the context except the task prompt is harness (H1); a plain-API agent would instead carry
          its own short system prompt and one tool schema, for which we keep ALLOWANCE tokens (cache write).
  turn k>1: text injected by the harness before turn k (records of type api-request-blob that are not the task
          prompt) is converted to tokens at CHARS_PER_TOKEN and removed from that turn's cache write; the harness
          context accumulated so far is removed from that turn's cache read.
  output tokens are unchanged (max of logged and characters / 4, as in score.py).

Savings use the same formulas as the registered scorers: System 1's GPU cost is added to B1 and C on every question.

  .venv/bin/python experiments/cost_confirmation/harness_free.py <session dir> [chars_per_token]
"""
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import score  # noqa: E402

S = Path(sys.argv[1])
CPT = float(sys.argv[2]) if len(sys.argv) > 2 else 4.0
ALLOWANCE = 1500
P = score.PRICE
TESTS = {  # test: (boxes, arms, System 1 $ per question, questions per box)
    "reserved_run1": (["s2res"], 0.000849),
    "reserved_run2": (["s2res_r2"], 0.000849),
    "reserved_run3": (["s2res_r3"], 0.000849),
    "external": (["s2ext"], None),
    "fresh": (["s2fresh"], 0.001046),
}


def session(f):
    recs = []
    for ln in open(f):
        try:
            recs.append(json.loads(ln))
        except json.JSONDecodeError:
            pass
    prompt = next((r["message"]["content"] for r in recs if r.get("type") == "user"
                   and isinstance((r.get("message") or {}).get("content"), str)), "")
    turns, seen, pending, chars = [], set(), 0, 0
    started = False
    for r in recs:
        t, m = r.get("type"), r.get("message") or {}
        if t == "api-request-blob" and started:
            for b in m.get("content") or [] if isinstance(m, dict) else []:
                if isinstance(b, dict) and b.get("type") == "text" and b.get("text", "") != prompt:
                    pending += len(b["text"])
        if t != "assistant" or not isinstance(m, dict):
            continue
        for b in m.get("content") or []:
            if isinstance(b, dict):
                chars += len(b.get("text", "")) + len(b.get("thinking", ""))
                if b.get("type") == "tool_use":
                    chars += len(json.dumps(b.get("input")))
        if m.get("id") in seen:
            continue
        seen.add(m.get("id"))
        started = True
        u = m.get("usage") or {}
        turns.append(({k: u.get(k, 0) or 0 for k in list(P) + ["output_tokens"]}, pending / CPT))
        pending = 0
    if not turns:
        return None
    out_tok = max(sum(u["output_tokens"] for u, _ in turns), chars / 4)
    full = sum(sum(u[k] * v for k, v in P.items()) for u, _ in turns) / 1e6 + out_tok * score.PRICE_OUT / 1e6
    u1 = turns[0][0]
    ctx1 = u1["input_tokens"] + u1["cache_creation_input_tokens"] + u1["cache_read_input_tokens"]
    task = len(prompt) / CPT
    H = max(0.0, ctx1 - task)
    free = (task + ALLOWANCE) * P["cache_creation_input_tokens"]
    for u, added in turns[1:]:
        cw = max(0.0, u["cache_creation_input_tokens"] - added)
        cr = max(0.0, u["cache_read_input_tokens"] - H) + ALLOWANCE
        free += u["input_tokens"] * P["input_tokens"] + cw * P["cache_creation_input_tokens"] + cr * P["cache_read_input_tokens"]
        H += added
    free = free / 1e6 + out_tok * score.PRICE_OUT / 1e6
    return full, free, ctx1 - task


def main():
    pat = re.compile(r"scratchpad/(s2\w+)/chunks/(B0|B1|B2|C)/([A-Za-z0-9_]+_\d+)\.jsonl")
    agg = defaultdict(lambda: [0.0, 0.0, 0, 0.0])  # (box, arm) -> full $, free $, sessions, harness tokens
    for f in (S / "tasks").glob("*.output"):
        m = pat.search(open(f).read(20000))
        if not m:
            continue
        r = session(f)
        if r:
            a = agg[(m.group(1), m.group(2))]
            a[0] += r[0]; a[1] += r[1]; a[2] += 1; a[3] += r[2]
    nq = {"s2res": 348, "s2res_r2": 348, "s2res_r3": 348, "s2ext": 693, "s2fresh": 227}
    out = {"chars_per_token": CPT, "allowance_tokens": ALLOWANCE, "tests": {}}
    for test, (boxes, s1q) in TESTS.items():
        b = boxes[0]
        n = nq[b]
        arms = {arm: agg[(b, arm)] for arm in ("B0", "B1", "C") if (b, arm) in agg}
        if s1q is None:  # external: recover System 1's $ per question from the registered result
            ext = json.load(open(Path(__file__).parents[1] / "external_cascade/result.json"))["pooled"]["cost_per_q"]
            s1q = ext["C"] - arms["C"][0] / n
        row = {"sessions": {a: v[2] for a, v in arms.items()},
               "harness_tokens_per_session": {a: round(v[3] / v[2]) for a, v in arms.items()},
               "per_q_full": {}, "per_q_harness_free": {}}
        for a, v in arms.items():
            add = s1q * n if a in ("B1", "C") else 0.0
            row["per_q_full"][a] = round((v[0] + add) / n, 5)
            row["per_q_harness_free"][a] = round((v[1] + add) / n, 5)
        for base in ("B1", "B0"):
            if base in arms:
                row[f"saving_vs_{base}"] = {k: round(1 - row[k]["C"] / row[k][base], 4) for k in ("per_q_full", "per_q_harness_free")}
        out["tests"][test] = row
    print(json.dumps(out, indent=1))
    if CPT == 4.0:
        Path(__file__).with_name("harness_free.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
