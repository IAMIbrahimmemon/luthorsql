"""Round-6 step 1: sample SQL from a model with several prompt styles on a question list, execute every
candidate, give failing ones one fix retry, and score greedy / vote / oracle per style and pooled.

  python scripts/gen6.py --model seeklhy/OmniSQL-32B --questions dev6c_bird.jsonl --cards bird_dev/cards \
      --db-roots dev_databases --shots prompts.jsonl --styles think:8 sql:8 fewshot:8 --out runs/r6/s1_omni.jsonl

Each style also gets one greedy answer (src "<style>_g"). Correctness: the candidate's result signature equals
the question's gold_sig (BIRD-style set match). --dry builds prompts and prints their sizes, no model.
"""
from __future__ import annotations

import argparse
import collections
import json
import sqlite3
import sys
import time
from multiprocessing import Pool
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from luthor.sq.candidates import ValueIndex  # noqa: E402
from luthor.sq.schema_card import SchemaCard  # noqa: E402
from luthor.sqlw.evalkit import db_paths, paired, pick_vote  # noqa: E402
from luthor.sqlw.execute import connect, signature  # noqa: E402
from luthor.sqlw.prompt import value_hints  # noqa: E402
from luthor.sqlw.prompt6 import (ShotIndex, arctic_messages, extract_sql, fewshot_messages, sql_messages,  # noqa: E402
                                 think_messages, unresolved_identifiers)

BUILD = {"think": think_messages, "sql": sql_messages, "fewshot": fewshot_messages, "arctic": arctic_messages}
MAXTOK = {"think": 2048, "sql": 512, "fewshot": 512, "arctic": 6144}


def run_task(task):
    """(db path, [(qid, [sql])], seconds) -> [(qid, [dict(runs, empty, sig, error)])]."""
    path, items, seconds = task
    run = connect(path, seconds)
    out = []
    for qid, sqls in items:
        res, seen = [], {}
        for s in sqls:
            if s not in seen:
                if not s:
                    seen[s] = {"runs": False, "empty": True, "sig": "", "error": "no SQL"}
                else:
                    try:
                        rows = run(s)
                        seen[s] = {"runs": True, "empty": not any(v is not None for r in rows for v in r),
                                   "sig": signature(rows), "error": ""}
                    except Exception as e:
                        seen[s] = {"runs": False, "empty": True, "sig": "", "error": str(e)[:300]}
            res.append(seen[s])
        out.append((qid, res))
    return out


def execute(recs, paths, seconds, workers, key="cands"):
    by_db = collections.defaultdict(list)
    for r in recs:
        by_db[r["db_id"]].append((r["id"], [c["sql"] for c in r[key]]))
    idx = {r["id"]: r for r in recs}
    tasks = sorted(((paths[db], items, seconds) for db, items in by_db.items()), key=lambda t: -len(t[1]))
    # split big databases so the pool stays busy
    split = []
    for p, items, s in tasks:
        for i in range(0, len(items), 25):
            split.append((p, items[i:i + 25], s))
    with Pool(workers) as pool:
        for part in pool.imap_unordered(run_task, split):
            for qid, res in part:
                for c, x in zip(idx[qid][key], res):
                    c.update(x)


def score(recs, gold, srcs, card_of, reject_bad_ids=False, use_fix=False):
    """Per question 0/1 for greedy (first candidate of the first source), vote and oracle over `srcs`."""
    g, v, o = {}, {}, {}
    for r in recs:
        cs = [c for c in r["cands"] if c["src"].split("_")[0] in srcs]
        if use_fix:
            cs = [c.get("fixed", c) for c in cs]
        live = [dict(c) for c in cs]
        if reject_bad_ids:
            for c in live:
                if c["runs"] and unresolved_identifiers(card_of(r["db_id"]), c["sql"]):
                    c["runs"] = False
        ok = lambda c: bool(c and c["runs"] and c["sig"] == gold[r["id"]])
        greedy = next((c for c in live if c["src"].endswith("_g")), live[0] if live else None)
        g[r["id"]], v[r["id"]], o[r["id"]] = int(ok(greedy)), int(ok(pick_vote(live) if live else None)), int(any(map(ok, live)))
    return g, v, o


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="seeklhy/OmniSQL-32B")
    ap.add_argument("--lora", type=Path)
    ap.add_argument("--questions", type=Path)
    ap.add_argument("--cards", type=Path, help="directory of <db_id>.json schema cards")
    ap.add_argument("--db-roots", nargs="+")
    ap.add_argument("--shots", type=Path, help="training questions for few-shot examples (data/rl6/prompts.jsonl)")
    ap.add_argument("--styles", nargs="+", default=["think:8", "sql:8", "fewshot:8"])
    ap.add_argument("--hints", action="store_true")
    ap.add_argument("--fix", action="store_true", help="one retry with the SQLite error for failing candidates")
    ap.add_argument("--temp", type=float, default=0.8)
    ap.add_argument("--max-model-len", type=int, default=16384)
    ap.add_argument("--eager", action="store_true", help="no torch.compile / CUDA graphs (fp8-on-A100 compile bug)")
    ap.add_argument("--quant", default=None, help="e.g. fp8: 8-bit weights (on A100: weight-only Marlin), frees KV cache")
    ap.add_argument("--gpu-mem", type=float, default=0.92)
    ap.add_argument("--seconds", type=float, default=30.0)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--dry", action="store_true")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--build-prompts", action="store_true", help="Mac: write <out>_prompts.jsonl (needs DBs), no model")
    ap.add_argument("--from-prompts", type=Path, help="GPU: read prompts built on the Mac; no cards or DBs needed")
    ap.add_argument("--no-exec", action="store_true", help="GPU: write candidates only; execute later on the Mac")
    ap.add_argument("--keep-text", action="store_true", help="store each candidate's full answer text")
    ap.add_argument("--exec-only", action="store_true", help="Mac: execute and score the candidates in --out")
    a = ap.parse_args()
    if a.from_prompts:
        rows = [json.loads(l) for l in open(a.from_prompts)]
        styles = [(s.split(":")[0], int(s.split(":")[1])) for s in a.styles]
        recs = [{k: r[k] for k in ("id", "db_id", "gold_sig") if k in r} for r in rows]
        prompts = {st: [r["prompts"][st] for r in rows] for st, _ in styles}
        return generate(a, recs, prompts, styles)
    recs = [json.loads(l) for l in open(a.questions)]
    if a.limit:
        recs = recs[:a.limit]
    paths = db_paths(a.db_roots)
    if a.exec_only:
        recs = [json.loads(l) for l in open(a.out)]
        styles = [(s.split(":")[0], int(s.split(":")[1])) for s in a.styles]
        cards = {}
        return finish(a, recs, paths, lambda db: cards.setdefault(db, SchemaCard.load(a.cards / f"{db}.json")),
                      styles, time.time())
    cards = {}
    card_of = lambda db: cards.setdefault(db, SchemaCard.load(a.cards / f"{db}.json"))
    shots = ShotIndex([json.loads(l) for l in open(a.shots) if '"bird_train"' in l]) if a.shots else None
    styles = [(s.split(":")[0], int(s.split(":")[1])) for s in a.styles]
    vidx = {}
    prompts = {st: [] for st, _ in styles}
    for r in recs:
        hints = None
        if a.hints:
            if r["db_id"] not in vidx:
                con = sqlite3.connect(f"file:{paths[r['db_id']]}?mode=ro", uri=True)
                con.text_factory = lambda b: b.decode("utf-8", "replace")
                vidx[r["db_id"]] = ValueIndex.build(con, card_of(r["db_id"]), skip_card_values=False)
                con.close()
            hints = value_hints(vidx[r["db_id"]], r["question"], r.get("context", ""))
        for st, _ in styles:
            kw = {"shots": shots.get(r["question"], r.get("context", ""), r["db_id"])} if st == "fewshot" and shots else {}
            prompts[st].append(BUILD[st](card_of(r["db_id"]), r["question"], r.get("context", ""), hints, **kw))
    if a.build_prompts:
        out = a.out.parent / (a.out.stem + "_prompts.jsonl")
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text("".join(json.dumps({"id": r["id"], "db_id": r["db_id"], "gold_sig": r.get("gold_sig", ""),
                                           "prompts": {st: prompts[st][i] for st, _ in styles}}) + "\n"
                               for i, r in enumerate(recs)))
        print("BUILT", out, len(recs), flush=True)
        return
    if a.dry:
        for st, ps in prompts.items():
            lens = sorted(sum(len(m["content"]) for m in p) for p in ps)
            print(st, "prompts", len(ps), "chars median", lens[len(lens) // 2], "max", lens[-1])
        print(json.dumps(prompts[styles[0][0]][0], indent=1)[:3000])
        return
    t0 = generate(a, recs, prompts, styles)
    if not a.no_exec:
        finish(a, recs, paths, card_of, styles, t0)


def generate(a, recs, prompts, styles):
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest
    t0 = time.time()
    llm = LLM(a.model, enable_lora=a.lora is not None, max_lora_rank=64, max_model_len=a.max_model_len,
              gpu_memory_utilization=a.gpu_mem, enable_prefix_caching=True, seed=0,
              enforce_eager=a.eager, **({"quantization": a.quant} if a.quant else {}))
    lr = LoRARequest("w", 1, str(a.lora)) if a.lora else None
    a._llm, a._lr = llm, lr
    for r in recs:
        r["cands"] = []
    for st, n in styles:
        # a pre-filled assistant turn (arctic) is continued, not answered
        cont = {"add_generation_prompt": False, "continue_final_message": True} if prompts[st][0][-1]["role"] == "assistant" else {}
        top_p = 1.0 if st == "arctic" else 0.95  # Arctic's own eval: temperature 0.8, no nucleus cut
        g = llm.chat(prompts[st], SamplingParams(temperature=0.0, max_tokens=MAXTOK[st]), lora_request=lr, **cont)
        s = llm.chat(prompts[st], SamplingParams(temperature=a.temp, top_p=top_p, n=n, max_tokens=MAXTOK[st], seed=0),
                     lora_request=lr, **cont) if n else [None] * len(recs)
        for r, gi, si in zip(recs, g, s):
            for src, o in [(f"{st}_g", gi.outputs[0])] + [(f"{st}_s", o) for o in (si.outputs if si else [])]:
                c = {"src": src, "sql": extract_sql(o.text), "ntok": len(o.token_ids)}
                if a.keep_text:  # the full answer (reasoning + SQL), for self-sampled SFT rows
                    c["text"] = o.text
                r["cands"].append(c)
        print(f"STYLE {st} generated, {(time.time() - t0) / 60:.1f} min", flush=True)
    if a.no_exec or a.from_prompts:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text("".join(json.dumps(r) + "\n" for r in recs))
        print("GENERATED", a.out, flush=True)
    return t0


def finish(a, recs, paths, card_of, styles, t0):
    execute(recs, paths, a.seconds, a.workers)
    print(f"executed, {(time.time() - t0) / 60:.1f} min", flush=True)
    if a.fix and getattr(a, "_llm", None):
        from vllm import SamplingParams
        llm, lr = a._llm, a._lr
        todo = [(r, c) for r in recs for c in r["cands"] if not c["runs"] and c["sql"]]
        msgs = []
        for r, c in todo:
            st = c["src"].split("_")[0]
            base = BUILD["sql" if st == "fewshot" else st](card_of(r["db_id"]), r["question"], r.get("context", ""), None)
            msgs.append(base + [{"role": "assistant", "content": f"```sql\n{c['sql']}\n```"},
                                {"role": "user", "content": f"That query failed: {c['error']}\nWrite a corrected query. "
                                                            "Put the SQL in a ```sql code block."}])
        outs = llm.chat(msgs, SamplingParams(temperature=0.0, max_tokens=1024), lora_request=lr) if msgs else []
        tmp = collections.defaultdict(list)
        for (r, c), o in zip(todo, outs):
            tmp[r["id"]].append((c, {"src": c["src"], "sql": extract_sql(o.outputs[0].text)}))
        fx = [{"id": qid, "db_id": next(r["db_id"] for r in recs if r["id"] == qid), "fx": [f for _, f in pairs]}
              for qid, pairs in tmp.items()]
        execute(fx, paths, a.seconds, a.workers, key="fx")
        for f in fx:
            for (c, _), new in zip(tmp[f["id"]], f["fx"]):
                c["fixed"] = new
        print(f"FIX retried {len(todo)}, now run {sum(f2['runs'] for f in fx for f2 in f['fx'])}", flush=True)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text("".join(json.dumps(r) + "\n" for r in recs))
    gold = {r["id"]: r["gold_sig"] for r in recs}
    db_of = {r["id"]: r["db_id"] for r in recs}
    rep, sysm = {"n": len(recs), "minutes": round((time.time() - t0) / 60, 1)}, {}
    names = [st for st, _ in styles]
    for label, srcs in [(st, {st}) for st in names] + [("pooled", set(names))]:
        for variant, kw in (("", {}), ("+ids", {"reject_bad_ids": True}), ("+fix+ids", {"reject_bad_ids": True, "use_fix": a.fix})):
            if variant == "+fix+ids" and not a.fix:
                continue
            g, v, o = score(recs, gold, srcs, card_of, **kw)
            sysm[f"{label}{variant}"] = v
            rep[f"{label}{variant}"] = {"greedy": round(100 * sum(g.values()) / len(g), 1), "vote": round(100 * sum(v.values()) / len(v), 1),
                                        "oracle": round(100 * sum(o.values()) / len(o), 1)}
    for st in names:
        if st != names[0]:
            rep[f"vote {st} - {names[0]}"] = paired(sysm[st], sysm[names[0]], db_of)
        rep[f"vote pooled - {st}"] = paired(sysm["pooled"], sysm[st], db_of)
    rep["mean_tokens"] = {st: round(sum(c["ntok"] for r in recs for c in r["cands"] if c["src"].startswith(st)) /
                                    max(1, sum(1 for r in recs for c in r["cands"] if c["src"].startswith(st))), 1) for st in names}
    (a.out.parent / (a.out.stem + "_metrics.json")).write_text(json.dumps(rep, indent=1))
    print("GEN6", json.dumps(rep), flush=True)


if __name__ == "__main__":
    main()
