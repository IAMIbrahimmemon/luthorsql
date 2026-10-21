"""Question variants for test-time ensembling over rewordings (held-out training DBs only; eval-only, never training).

  python scripts/analysis/qvariants.py build [--workers 3]     # -> data/sys1/qvar/<variant>/{plans.jsonl,cards/}

Rows: data/sys1/bird_train/plans.jsonl rows whose db_id is a held-out training DB (data/sys1/dev_dbs_bird.json);
only the "question" field changes. Variants:
  orig      unchanged (regenerated with the rest, same software)
  p1 p2 p3  DeepSeek rephrasings (one call per question, three outputs). A rephrasing that drops any number, quoted
            string or capitalised token of the original is replaced by the original (counted as a fallback).
  f         filler dropped: articles and politeness/lead-in phrases ("please", "can you tell me", "list out" -> "list")
  s         slight scramble: one adjacent pair of plain lower-case words swapped (never numbers, quoted text,
            capitalised words or negations), seeded by the question id
DeepSeek prompts carry public BIRD-train questions/evidence only and pass the leak gate (Provider.chat).
"""
import argparse
import hashlib
import json
import random
import re
import shutil
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
SRC = ROOT / "data/sys1/bird_train"
OUT = ROOT / "data/sys1/qvar"
HELD = set(json.loads((ROOT / "data/sys1/dev_dbs_bird.json").read_text()))
CACHE = ROOT / "runs/qvar/paraphrases.jsonl"
# default: DeepSeek V4.1 Flash via OpenRouter (key from .env.txt openrouter_api_key -> $OPENROUTER_API_KEY);
# QVAR_MODEL=deepseek-v4.1-flash:cloud uses Ollama Cloud instead (max 3 concurrent DeepSeek calls there)
import os  # noqa: E402
MODEL = os.environ.get("QVAR_MODEL", "openai:deepseek/deepseek-v4.1-flash@https://openrouter.ai/api/v1#OPENROUTER_API_KEY")

PROMPT = """Rewrite the question below in 3 different ways that keep exactly the same meaning, for a text-to-SQL system.
Rules: keep every name, number, date, quoted value and domain term exactly as written (same spelling and case);
do not add or drop any condition, column or qualifier; do not answer it. Vary wording and sentence structure only.
Evidence (context, do not rewrite): {ev}
Question: {q}
Return only JSON: {{"rewrites": ["...", "...", "..."]}}"""

LEAD = [r"\bcan you (please )?(tell|show|give|list) (me|us)\b", r"\bcould you (please )?(tell|show|give|list) (me|us)\b",
        r"\bplease\b", r"\bkindly\b", r"\bi (want|would like) to know\b", r"\btell me\b", r"\bshow me\b", r"\bgive me\b",
        r"\bfind out\b"]
ARTICLES = r"\b(the|a|an)\b"
NEG = {"not", "no", "never", "without", "except", "nor", "none", "only", "least", "most", "than", "between", "and", "or"}


def keep_tokens(q):
    """Tokens a rewrite must keep: numbers, quoted strings, capitalised words (not sentence-initial)."""
    toks = set(re.findall(r"\d[\d.,:/-]*", q))
    toks |= {m.strip("'\"") for m in re.findall(r"'[^']+'|\"[^\"]+\"", q)}
    words = q.split()
    toks |= {w.strip(".,?;:!()'\"") for i, w in enumerate(words) if i > 0 and w[:1].isupper()}
    return {t for t in toks if t}


def filler(q):
    s = q
    for p in LEAD:
        s = re.sub(p, "", s, flags=re.I)
    s = re.sub(r"\blist out\b", "list", s, flags=re.I)
    s = re.sub(ARTICLES, "", s, flags=re.I)
    s = re.sub(r"\s+", " ", s).strip(" ,")
    s = re.sub(r"\s+([?.,])", r"\1", s)
    return (s[:1].upper() + s[1:]) if s else q


def scramble(q, qid):
    w = q.split()
    ok = lambda t: t.isalpha() and t.islower() and t not in NEG  # noqa: E731
    pairs = [i for i in range(1, len(w) - 1) if ok(w[i]) and ok(w[i + 1])]
    if not pairs:
        return q
    i = random.Random(int(hashlib.md5(qid.encode()).hexdigest()[:8], 16)).choice(pairs)
    w[i], w[i + 1] = w[i + 1], w[i]
    return " ".join(w)


def paraphrase_all(rows, workers):
    from luthor.teacher.leak import LeakGate
    from luthor.teacher.ledger import Ledger
    from luthor.teacher.providers import make
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    have = {json.loads(l)["id"]: json.loads(l) for l in open(CACHE)} if CACHE.exists() else {}
    prov = make(MODEL, LeakGate(), ledger=Ledger(ROOT / "data/teacher/usage.jsonl"), job="qvar")
    lock = threading.Lock()
    todo = [r for r in rows if r["id"] not in have]
    print(f"paraphrase: {len(todo)} to do ({len(have)} cached)", flush=True)

    def one(r):
        rec = {"id": r["id"], "rewrites": []}
        try:
            rep = prov.chat([{"role": "user", "content": PROMPT.format(ev=r.get("context") or "(none)", q=r["question"])}],
                            temperature=0.7, max_tokens=2500, think=False)
            m = re.search(r"\{.*\}", rep.text, re.S)
            rec["rewrites"] = [str(x).strip() for x in json.loads(m.group(0))["rewrites"]][:3] if m else []
        except Exception as e:  # noqa: BLE001
            rec["error"] = f"{type(e).__name__}: {str(e)[:150]}"
        with lock, open(CACHE, "a") as f:
            f.write(json.dumps(rec) + "\n")
        return rec

    with ThreadPoolExecutor(workers) as ex:
        for n, rec in enumerate(ex.map(one, todo), 1):
            have[rec["id"]] = rec
            if n % 200 == 0:
                print(f"  {n}/{len(todo)}", flush=True)
    return have


def build(a):
    if MODEL.startswith("openai:") and not os.environ.get("OPENROUTER_API_KEY"):
        sys.path.insert(0, str(ROOT / "scripts/analysis"))
        from jev_judge import key
        os.environ["OPENROUTER_API_KEY"] = key("openrouter_api_key")
    rows = [json.loads(l) for l in open(SRC / "plans.jsonl")]
    rows = [r for r in rows if r["db_id"] in HELD]
    print(f"{len(rows)} held-out plan rows on {len({r['db_id'] for r in rows})} DBs")
    para = paraphrase_all(rows, a.workers)
    stats = {}
    for v in ("orig", "p1", "p2", "p3", "f", "s"):
        d = OUT / v
        (d / "cards").mkdir(parents=True, exist_ok=True)
        for db in {r["db_id"] for r in rows}:
            shutil.copy(SRC / "cards" / f"{db}.json", d / "cards" / f"{db}.json")
        changed = fallback = 0
        with open(d / "plans.jsonl", "w") as f:
            for r in rows:
                q = r["question"]
                if v == "f":
                    nq = filler(q)
                elif v == "s":
                    nq = scramble(q, r["id"])
                elif v.startswith("p"):
                    rw = para.get(r["id"], {}).get("rewrites") or []
                    k = int(v[1]) - 1
                    nq = rw[k] if k < len(rw) else q
                    if nq != q and not all(t in nq for t in keep_tokens(q)):
                        nq, fallback = q, fallback + 1
                    if k >= len(rw):
                        fallback += 1
                else:
                    nq = q
                changed += nq != q
                f.write(json.dumps(dict(r, question=nq)) + "\n")
        stats[v] = {"rows": len(rows), "changed": changed, "fallback_to_orig": fallback}
    (OUT / "stats.json").write_text(json.dumps(stats, indent=1))
    print(json.dumps(stats, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["build"])
    ap.add_argument("--workers", type=int, default=3)
    build(ap.parse_args())
