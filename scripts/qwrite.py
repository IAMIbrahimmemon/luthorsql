"""Rare-skeleton training data, step 2 (GPU, vLLM): write BIRD-style questions for selected plan mutants
(data/rare1/selected.jsonl from scripts/build_rare_mix.py), keep only round-trip-checked ones, and emit
training rows in the exact format of data/sys1/bird_train (what train_sq2.py --plans loads).

  write  public instruct model (default Qwen/Qwen2.5-14B-Instruct, Apache-2.0): for each selected SQL, n=2
         (question, evidence) pairs at temperature 0.7. Prompt: touched-table schema snippet + SQL + sample
         result rows + 3 real BIRD-TRAIN (question, evidence, SQL) style shots (the mutant's own seed question
         and two from other databases, same skeleton when one exists). Never BIRD dev material.
  check  round trip: seeklhy/OmniSQL-7B, greedy, OmniSQL's own prompt (prompt6.think_messages, full schema)
         writes SQL from (question, evidence); a pair is kept only if that SQL's result signature
         (sqlw/execute.signature) equals the mutant's. At most one pair per plan.
  build  (CPU) re-grounds every literal of the plan on the question's candidates exactly as
         prepare_plans.py does (db_values, value_labels, limit_label), re-executes the compiled plan, proves
         each row loads with sq2.train.make_example, and writes <out>/plans.jsonl + <out>/cards/.

  python scripts/qwrite.py write [--shard 0/2] [--dry]            # Mac defaults: data/rare1/selected.jsonl, data/sys1/...
  python scripts/qwrite.py check [--shard 0/2] [--dry | --no-exec | --exec-only]
  python scripts/qwrite.py build                                   # CPU; writes data/rare1/plans.jsonl + cards/
  On the Vast box add --data /root/d (push6.sh layout: /root/d/sys1/..., /root/d/rare1/...).

Sharding: a plan belongs to shard md5(id) % n, the same in every stage, so GPU i can run write then check on
shard i/n with no barrier (CUDA_VISIBLE_DEVICES=i picks the card; vLLM sees one GPU). build merges all shards.
--dry builds the prompts and prints counts and sizes; it imports no model code (runs on the Mac).
"""
from __future__ import annotations

import argparse
import collections
import glob
import hashlib
import json
import os
import random
import re
import shutil
import sqlite3
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))
from luthor.sq.schema_card import SchemaCard  # noqa: E402
from luthor.sqlw.execute import connect, signature  # noqa: E402
from luthor.sqlw.prompt6 import extract_sql, think_messages  # noqa: E402

D = REPO / "data"
SHOT_PLANS = D / "sys1/bird_train/plans.jsonl"   # real BIRD-TRAIN questions (style shots)
HELD_DBS = D / "sys1/dev_dbs_bird.json"
BIRD_DEV_11 = {"california_schools", "card_games", "codebase_community", "debit_card_specializing", "european_football_2",
               "financial", "formula_1", "student_club", "superhero", "thrombosis_prediction", "toxicology"}
WRITE_MODEL = "Qwen/Qwen2.5-14B-Instruct"
CHECK_MODEL = "seeklhy/OmniSQL-7B"
CHARS_PER_TOKEN = 3.6  # rough, for --dry token estimates (Qwen2.5 tokenizer on schema + English)

WRITE_SYSTEM = ("You write questions for a text-to-SQL benchmark in the style of BIRD. You are given part of a "
                "database schema, a SQL query and a sample of its result. You write the natural-language question a "
                "data analyst would ask that this exact query answers, plus BIRD's short 'evidence' note.")

WRITE_RULES = """Write ONE question that this SQL answers exactly, and its evidence note.
Rules:
- Ask for exactly what the SQL returns: the same number of output values with the same meaning, and every filter, aggregate, grouping, ordering and limit the SQL applies. Nothing extra.
- Plain English, as a person would type it. Avoid SQL keywords and raw table/column identifiers when an everyday word works.
- Every literal value in the SQL (strings, numbers, dates) must appear in the question or the evidence exactly as written in the SQL (same spelling and letter case).
- Evidence: at most one or two short clauses in BIRD style that map phrases to columns or values, or state a formula, e.g. "east is a direction; train no.1 refers to train_id = 1" or "percentage = DIVIDE(COUNT(x), COUNT(y)) * 100". Use an empty string if nothing needs explaining.
- Do not mention or describe the result values.
Reply with JSON only: {"question": "...", "evidence": "..."}"""


# ---- shared helpers ---------------------------------------------------------------------------------------
def shard_of(qid: str, n: int) -> int:
    return int(hashlib.md5(qid.encode()).hexdigest()[:8], 16) % n


def parse_shard(s: str) -> tuple[int, int]:
    i, n = (int(x) for x in s.split("/"))
    assert 0 <= i < n, f"bad --shard {s}"
    return i, n


def in_shard(rows: list[dict], shard: tuple[int, int]) -> list[dict]:
    i, n = shard
    return [r for r in rows if shard_of(r["id"], n) == i]


def tag(shard: tuple[int, int]) -> str:
    return f"s{shard[0]}of{shard[1]}"


def read_jsonl(path) -> list[dict]:
    return [json.loads(l) for l in open(path) if l.strip()]


def read_stage(work: Path, stage: str) -> list[dict]:
    """All shard files of a stage (written.*.jsonl / checked.*.jsonl), one record per plan id."""
    out = {}
    for p in sorted(glob.glob(str(work / f"{stage}.*.jsonl"))):
        for r in read_jsonl(p):
            out[r["id"]] = r
    return list(out.values())


def shape(p: dict) -> str:
    """Coarse plan shape (support.py level 0), for picking style shots near a rare skeleton."""
    np_, nf = len(p["projections"]), len(p["filters"])
    tabs = {x["col"]["table"] for x in p["projections"]} | {f["col"]["table"] for f in p["filters"]}
    agg = any(x["agg"] != "none" for x in p["projections"])
    return (f"P{min(np_, 3)}F{min(nf, 3)}T{min(len(tabs), 3)}{'A' if agg else ''}{'G' if p.get('group_by') else ''}"
            f"{'O' if p.get('order_dir', 'none') != 'none' else ''}")


class Shots:
    """Real BIRD-train (question, evidence, SQL) rows for style shots: never a BIRD dev or held-out db."""

    def __init__(self, path: Path | None = None, held: set | None = None):
        from build_rare_mix import skeleton
        path = path or SHOT_PLANS
        held = held if held is not None else set(json.loads(HELD_DBS.read_text()))
        self.by_id, self.by_skel, self.by_shape = {}, collections.defaultdict(list), collections.defaultdict(list)
        for r in read_jsonl(path):
            if r.get("source") != "bird" or r.get("split") != "train" or r.get("twin_of") or not r.get("gold_sql"):
                continue
            assert r["db_id"] not in BIRD_DEV_11
            self.by_id[r["id"]] = r
            if r["db_id"] in held or r.get("outcome") != "plan":
                continue
            self.by_skel[skeleton(r["plan"])].append(r)
            self.by_shape[shape(r["plan"])].append(r)
        self.all = [r for lst in self.by_shape.values() for r in lst]

    def get(self, row: dict, k: int = 3) -> list[dict]:
        rnd = random.Random(int(hashlib.md5(row["id"].encode()).hexdigest()[:8], 16))
        out = []
        seed = self.by_id.get(row.get("seed_id"))
        if seed is not None:
            out.append(seed)
        dbs = {row["db_id"]}
        for pool in (self.by_skel.get(row.get("skel"), []), self.by_shape.get(shape(row["plan"]), []), self.all):
            cand = [r for r in pool if r["db_id"] not in dbs]
            rnd.shuffle(cand)
            for r in cand:
                if len(out) >= k:
                    break
                if r["db_id"] in dbs:
                    continue
                out.append(r)
                dbs.add(r["db_id"])
            if len(out) >= k:
                break
        return out[:k]


def write_messages(row: dict, shots: list[dict]) -> list[dict]:
    ex = "\n\n".join(f"Question: {s['question'].strip()}\nEvidence: {(s.get('context') or '').strip()}\n"
                     f"SQL: {' '.join(s['gold_sql'].split())}" for s in shots)
    rows = "\n".join(" | ".join(map(str, r)) for r in row["sample_rows"])
    user = (f"Real BIRD examples (style only; most are from other databases):\n\n{ex}\n\n---\n\n"
            f"Database: {row['db_id']}\nSchema (tables the query uses):\n{row['schema']}\n\n"
            f"SQL:\n{row['compiled_sql']}\n\n"
            f"Result: {row['n_rows']} row(s); first {len(row['sample_rows'])}:\n{rows}\n\n{WRITE_RULES}")
    return [{"role": "system", "content": WRITE_SYSTEM}, {"role": "user", "content": user}]


def parse_written(text: str) -> dict | None:
    """{"question", "evidence"} from the writer's reply (JSON, else 'Question:' / 'Evidence:' lines)."""
    text = re.sub(r"```(?:json)?", "", text).strip()
    m = re.search(r"\{.*\}", text, re.S)
    q = ev = None
    if m:
        try:
            d = json.loads(m.group(0))
            q, ev = d.get("question"), d.get("evidence", "")
        except Exception:
            pass
    if not q:
        mq = re.search(r"question\"?\s*:\s*\"?(.+?)\"?\s*(?:,?\s*\n|$)", text, re.I)
        me = re.search(r"evidence\"?\s*:\s*\"?(.*?)\"?\s*(?:\}|\n|$)", text, re.I)
        q, ev = (mq.group(1) if mq else None), (me.group(1) if me else "")
    if not isinstance(q, str) or len(q.strip()) < 8:
        return None
    ev = ev if isinstance(ev, str) else ""
    q, ev = " ".join(q.split()), " ".join(ev.split())
    if ev.lower() in ("none", "n/a", "null", "-"):
        ev = ""
    return {"question": q, "evidence": ev}


def sizes(prompts: list[list[dict]]) -> dict:
    lens = sorted(sum(len(m["content"]) for m in p) for p in prompts) or [0]
    return {"n": len(prompts), "chars_median": lens[len(lens) // 2], "chars_max": lens[-1],
            "chars_mean": round(sum(lens) / len(lens)), "tokens_mean_est": round(sum(lens) / len(lens) / CHARS_PER_TOKEN),
            "tokens_total_est": round(sum(lens) / CHARS_PER_TOKEN)}


# ---- model backend (vLLM, imported only off --dry) -----------------------------------------------------
class VLLMChat:
    def __init__(self, model, max_model_len, gpu_mem, quant=None, eager=False):
        from vllm import LLM
        print(f"vLLM {model} on CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '(all)')}", flush=True)
        self.llm = LLM(model, max_model_len=max_model_len, gpu_memory_utilization=gpu_mem, enable_prefix_caching=True,
                       seed=0, enforce_eager=eager, **({"quantization": quant} if quant else {}))

    def __call__(self, prompts, n=1, temperature=0.0, max_tokens=512, top_p=1.0):
        from vllm import SamplingParams
        sp = SamplingParams(n=n, temperature=temperature, top_p=top_p if temperature else 1.0, max_tokens=max_tokens, seed=0)
        outs = self.llm.chat(prompts, sp)
        return [[(o.text, len(o.token_ids)) for o in r.outputs] for r in outs]


# ---- stages -------------------------------------------------------------------------------------------
def load_selected(a) -> list[dict]:
    rows = read_jsonl(a.selected)
    for r in rows:
        assert r["db_id"] not in BIRD_DEV_11, r["db_id"]
    rows = in_shard(rows, parse_shard(a.shard))
    return rows[:a.limit] if a.limit else rows


def stage_write(a, gen=None) -> list[dict]:
    rows = load_selected(a)
    shots = Shots()
    prompts = [write_messages(r, shots.get(r)) for r in rows]
    if a.dry:
        print("WRITE dry", json.dumps(sizes(prompts)), flush=True)
        if prompts:
            print(prompts[0][1]["content"][:6000])
        return []
    gen = gen or VLLMChat(a.write_model, 8192, a.gpu_mem, a.quant, a.eager)
    t0 = time.time()
    outs = gen(prompts, n=a.n, temperature=a.temp, max_tokens=256, top_p=0.95)
    recs, ntok, bad = [], 0, 0
    for r, o in zip(rows, outs):
        cands, seen = [], set()
        for text, nt in o:
            ntok += nt
            w = parse_written(text)
            if w is None:
                bad += 1
                continue
            key = (w["question"].lower(), w["evidence"].lower())
            if key not in seen:
                seen.add(key)
                cands.append(w)
        recs.append({"id": r["id"], "db_id": r["db_id"], "cands": cands})
    work = Path(a.work)
    work.mkdir(parents=True, exist_ok=True)
    out = work / f"written.{tag(parse_shard(a.shard))}.jsonl"
    out.write_text("".join(json.dumps(r) + "\n" for r in recs))
    print(f"WRITTEN {out} plans={len(recs)} pairs={sum(len(r['cands']) for r in recs)} unparsed={bad} "
          f"out_tokens={ntok} min={(time.time() - t0) / 60:.1f}", flush=True)
    return recs


def _exec_chunk(task):
    path, lst, seconds = task
    run = connect(path, seconds)
    out = []
    for k, sql in lst:
        if not sql:
            out.append((k, (False, "")))
            continue
        try:
            out.append((k, (True, signature(run(sql)))))
        except Exception:
            out.append((k, (False, "")))
    return out


def run_exec(items, paths, seconds=5.0, workers=1):
    """[(key, db_id, sql)] -> {key: (runs, signature)} on read-only SQLite, `seconds` per query."""
    by_db = collections.defaultdict(list)
    for k, db, sql in items:
        by_db[db].append((k, sql))
    res, tasks = {}, []
    for db, lst in by_db.items():
        if db not in paths:
            res.update({k: (False, "no-db") for k, _ in lst})
            continue
        tasks += [(paths[db], lst[i:i + 50], seconds) for i in range(0, len(lst), 50)]
    if workers > 1 and len(tasks) > 1:
        from multiprocessing import Pool
        with Pool(workers) as p:
            for part in p.imap_unordered(_exec_chunk, tasks):
                res.update(dict(part))
    else:
        for t in tasks:
            res.update(dict(_exec_chunk(t)))
    return res


def accept(sel: dict, cands: list[dict]) -> dict | None:
    """First (question, evidence) whose round-trip SQL ran and returned the mutant's result signature."""
    for c in cands:
        if c.get("runs") and c.get("sig") == sel["sig"]:
            return c
    return None


def stage_check(a, gen=None) -> list[dict]:
    sel = {r["id"]: r for r in load_selected(a)}
    work = Path(a.work)
    shard = parse_shard(a.shard)
    out = work / f"checked.{tag(shard)}.jsonl"
    cards = {}
    card_of = lambda db: cards.setdefault(db, SchemaCard.load(Path(a.cards) / f"{db}.json"))
    if a.exec_only:  # Mac: execute candidates generated on the GPU with --no-exec
        recs = read_jsonl(out)
    else:
        written = [w for w in read_stage(work, "written") if w["id"] in sel]
        if not written and a.dry:  # nothing written yet: seed questions stand in, only to size the prompts
            shots = Shots()
            written = [{"id": i, "db_id": r["db_id"], "cands": [{"question": shots.by_id[r["seed_id"]]["question"],
                                                                 "evidence": shots.by_id[r["seed_id"]].get("context", "")}]}
                       for i, r in sel.items() if r.get("seed_id") in shots.by_id]
        todo = [(w, c) for w in written for c in w["cands"]]
        prompts = [think_messages(card_of(w["db_id"]), c["question"], c["evidence"]) for w, c in todo]
        if a.dry:
            print("CHECK dry", json.dumps(sizes(prompts)), flush=True)
            if prompts:
                print(prompts[0][0]["content"][:4000])
            return []
        gen = gen or VLLMChat(a.check_model, 16384, a.gpu_mem, a.quant, a.eager)
        t0 = time.time()
        outs = gen(prompts, n=1, temperature=0.0, max_tokens=2048)
        ntok = 0
        for (w, c), o in zip(todo, outs):
            text, nt = o[0]
            ntok += nt
            c["sql"] = extract_sql(text)
            c["ntok"] = nt
        recs = [{"id": w["id"], "db_id": w["db_id"], "cands": w["cands"]} for w in written]
        print(f"CHECK generated {len(todo)} out_tokens={ntok} min={(time.time() - t0) / 60:.1f}", flush=True)
    if not a.no_exec:
        from build_rare_mix import db_paths
        paths = db_paths(a.db_roots)
        res = run_exec([((i, j), r["db_id"], c.get("sql", "")) for i, r in enumerate(recs) for j, c in enumerate(r["cands"])],
                       paths, a.seconds, a.workers)
        for i, r in enumerate(recs):
            for j, c in enumerate(r["cands"]):
                c["runs"], c["sig"] = res[(i, j)]
            r["accepted"] = accept(sel[r["id"]], r["cands"])
        n_ok = sum(r["accepted"] is not None for r in recs)
        print(f"CHECK accepted {n_ok}/{len(recs)} plans ({100 * n_ok / max(1, len(recs)):.1f}%), "
              f"pairs ok {sum(c.get('sig') == sel[r['id']]['sig'] for r in recs for c in r['cands'])}"
              f"/{sum(len(r['cands']) for r in recs)}", flush=True)
    work.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(json.dumps(r) + "\n" for r in recs))
    print("CHECKED", out, flush=True)
    return recs


def build_row(sel: dict, q: str, ev: str, card: SchemaCard, graph, con, vindex) -> tuple[dict | None, str]:
    """One training row exactly as prepare_plans._one writes a BIRD-train row, or (None, reason)."""
    from prepare_plans import _ground_plan, norm_rows, run
    from luthor.sq.candidates import find_candidates, ground
    from luthor.sq.plan import Plan, compile_plan
    text = q + (f"\nContext: {ev}" if ev else "")
    db_values = vindex.match(text) if vindex is not None else []
    cands = find_candidates(text, card, extra=db_values)
    plan = Plan.from_dict(sel["plan"])
    value_labels: list = []
    if not _ground_plan(plan, cands, value_labels, top=True):
        return None, "ungrounded_value"
    limit_label = None
    if plan.limit is not None and plan.limit != 1:
        limit_label = ground(plan.limit, "", cands)
        if limit_label is None:
            return None, "ungrounded_limit"
    try:
        sql = compile_plan(plan, graph)
        got, ref = run(con, sql, 5.0), run(con, sel["compiled_sql"], 5.0)
    except Exception:
        return None, "exec_error"
    if norm_rows(got) != norm_rows(ref) or not got:
        return None, "exec_mismatch"
    return {"id": f"rare1_{sel['id']}", "source": "bird_mut", "split": "train", "db_id": sel["db_id"],
            "question": q, "context": ev, "gold_sql": sel["compiled_sql"], "outcome": "plan", "plan": plan.to_dict(),
            "compiled_sql": sql, "value_labels": value_labels, "limit_label": limit_label, "empty_result": False,
            "db_values": db_values, "seed_id": sel["seed_id"], "mut_op": sel["op"], "skel": sel["skel"]}, "ok"


def stage_build(a) -> list[dict]:
    from build_rare_mix import db_paths
    from luthor.sq.candidates import ValueIndex
    sel = {r["id"]: r for r in read_jsonl(a.selected)}
    checked = read_stage(Path(a.work), "checked")
    paths = db_paths(a.db_roots)
    vocab = None
    try:  # prove the trainer accepts every row (grammar v3 vocabulary, as train_sq2.py --vocab sets it)
        from luthor.sq2.targets import set_vocab
        from luthor.sq2.train import make_example
        vt = Path(a.vocab)
        set_vocab(json.loads(vt.read_text()), json.loads(vt.with_name(vt.stem + "_nums.json").read_text()))
        vocab = make_example
    except Exception as e:  # torch missing: rows are still built, unverified
        print("WARNING: make_example check skipped:", e, flush=True)
    st = collections.Counter()
    rows, dbs = [], {}
    by_db = collections.defaultdict(list)
    for r in checked:
        if r.get("accepted") and r["id"] in sel:
            by_db[r["db_id"]].append(r)
        else:
            st["not_accepted"] += 1
    for db, lst in sorted(by_db.items()):
        assert db not in BIRD_DEV_11
        if db not in paths:
            st["no_db"] += len(lst)
            continue
        card = SchemaCard.load(Path(a.cards) / f"{db}.json")
        graph = card.graph()
        con = sqlite3.connect(f"file:{paths[db]}?mode=ro", uri=True)
        con.text_factory = lambda b: b.decode("utf-8", "replace")
        try:
            vindex = ValueIndex.build(con, card)
        except Exception:
            vindex = None
        rnd = random.Random(0)
        for r in lst:
            c = r["accepted"]
            row, why = build_row(sel[r["id"]], c["question"], c["evidence"], card, graph, con, vindex)
            if row is not None and vocab is not None and vocab(row, card, rnd) is None:
                row, why = None, "trainer_rejects"
            st[why] += 1
            if row is not None:
                rows.append(row)
                dbs[db] = card
        con.close()
    out = Path(a.out)
    (out / "cards").mkdir(parents=True, exist_ok=True)
    for db in dbs:
        shutil.copyfile(Path(a.cards) / f"{db}.json", out / "cards" / f"{db}.json")
    rows.sort(key=lambda r: r["id"])
    (out / "plans.jsonl").write_text("".join(json.dumps(r, default=str) + "\n" for r in rows))
    stats = {"rows": len(rows), "dbs": len(dbs), "skeletons": len({r["skel"] for r in rows}), "counts": dict(st)}
    (out / "build_stats.json").write_text(json.dumps(stats, indent=1))
    print("BUILT", out / "plans.jsonl", json.dumps(stats), flush=True)
    return rows


def main(argv=None, gen=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=["write", "check", "build"])
    ap.add_argument("--data", default=str(D), help="data root laid out like data/ (on the Vast box: /root/d); "
                    "every path below defaults to a place under it")
    ap.add_argument("--selected", help="default <data>/rare1/selected.jsonl")
    ap.add_argument("--work", help="stage files written.sIofN.jsonl / checked.sIofN.jsonl; default <data>/rare1/work")
    ap.add_argument("--out", help="build: plans.jsonl + cards/ go here; default <data>/rare1")
    ap.add_argument("--cards", help="default <data>/sys1/bird_train/cards (the cards sys1 trained with)")
    ap.add_argument("--db-roots", nargs="+", help="default <data>/sys1/rrdbs (48 BIRD-train dbs)")
    ap.add_argument("--vocab", help="default <data>/sys1/templates.json")
    ap.add_argument("--shot-plans", help="BIRD-train rows for style shots; default <data>/sys1/bird_train/plans.jsonl")
    ap.add_argument("--held-dbs", help="held-out training dbs (no shots from them); default <data>/sys1/dev_dbs_bird.json")
    ap.add_argument("--shard", default="0/1", help="i/n: this process handles plans with md5(id) %% n == i")
    ap.add_argument("--write-model", default=WRITE_MODEL)
    ap.add_argument("--check-model", default=CHECK_MODEL)
    ap.add_argument("--n", type=int, default=2, help="write: (question, evidence) samples per plan")
    ap.add_argument("--temp", type=float, default=0.7)
    ap.add_argument("--gpu-mem", type=float, default=0.90)
    ap.add_argument("--quant", default=None)
    ap.add_argument("--eager", action="store_true")
    ap.add_argument("--seconds", type=float, default=5.0)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2), help="SQLite execution processes")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--dry", action="store_true", help="build prompts, print counts and sizes; no model")
    ap.add_argument("--no-exec", action="store_true", help="check: generate only (no DBs on the box)")
    ap.add_argument("--exec-only", action="store_true", help="check: execute an existing checked file (Mac)")
    a = ap.parse_args(argv)
    parse_shard(a.shard)
    d = Path(a.data)
    a.selected = a.selected or str(d / "rare1/selected.jsonl")
    a.work = a.work or str(d / "rare1/work")
    a.out = a.out or str(d / "rare1")
    a.cards = a.cards or str(d / "sys1/bird_train/cards")
    a.db_roots = a.db_roots or [str(d / "sys1/rrdbs")]
    a.vocab = a.vocab or str(d / "sys1/templates.json")
    global SHOT_PLANS, HELD_DBS
    SHOT_PLANS = Path(a.shot_plans or d / "sys1/bird_train/plans.jsonl")
    HELD_DBS = Path(a.held_dbs or d / "sys1/dev_dbs_bird.json")
    if a.stage == "build":
        return stage_build(a)
    return {"write": stage_write, "check": stage_check}[a.stage](a, gen)


if __name__ == "__main__":
    main()
