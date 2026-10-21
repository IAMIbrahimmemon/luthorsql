"""Free-form SQL candidates: n raw SQL per question from an open text-to-SQL model (default seeklhy/OmniSQL-7B) with
OmniSQL's official prompt, via vLLM. GPU box only for generation; --prompt-only runs on CPU (no vLLM) and writes the
prompts, for checking prompt lengths and format on the Mac.

  # DECISION (796 held-out BIRD-train questions, uncovered ones included)
  python scripts/freeform/gen.py --questions data/raw/bird_train/train.json --id-prefix bird_train_train_ \
      --ids data/sys1/heldout_v2/decision.json --db-roots data/raw/bird_train/train_databases \
      --model seeklhy/OmniSQL-7B --n 16 --temp 0.8 --greedy --out runs/ff/omni7b/decision_ff.jsonl

  # questions taken from an existing pool (same ids / question / context / gold_sql)
  python scripts/freeform/gen.py --pool runs/phase1/p1w/p1w/r1/cands_train.jsonl --db-roots ... --out ...

Output: one JSON line per question in the pool schema (heldout_union_by_r1.jsonl): id, db_id, question, context,
gold_sql, in_slots, covered, cands. in_slots / covered are copied from --pool when given, else null. Each candidate:
  {"rank": i, "sql": str | null, "src": "ff", "gen": <model tag>, "temp": T, "logp": mean token logprob,
   "ntok": output tokens, "finish": "stop" | "length"}
rank 0 is the greedy answer when --greedy (temperature 0), then the n samples in vLLM order. Candidates are NOT
executed here: scripts/freeform/merge.py labels them exactly as at grammar gen time and merges them into a pool.

Guards: --ids is required for BIRD-train held-out files; the RESERVED manifest is refused unless --confirm (generation
is not scoring, but RESERVED stays untouched until its one confirmation). Files named like BIRD dev are refused
unless --dev-look (each BIRD-dev run is one of the owner's <= 3 looks only once it is scored, but generate only with
the owner's go).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import prompt as P  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
RESERVED = ROOT / "data/sys1/heldout_v2/reserved.json"


def db_paths(roots: list[str]) -> dict[str, str]:
    out = {}
    for r in roots:
        for p in sorted(Path(r).rglob("*.sqlite")):
            out.setdefault(p.stem, str(p))
    return out


def load_questions(a) -> list[dict]:
    """-> [{id, db_id, question, context, gold_sql, in_slots, covered}] in file order, filtered by --ids."""
    ids = None
    if a.ids:
        man = json.loads(Path(a.ids).read_text())
        ids = man["ids"] if isinstance(man, dict) else man
        if Path(a.ids).resolve() == RESERVED.resolve() and not a.confirm:
            sys.exit("refusing the RESERVED manifest without --confirm")
    if a.pool:
        rows = [json.loads(l) for l in open(a.pool)]
        qs = [{k: r.get(k) for k in ("id", "db_id", "question", "context", "gold_sql", "in_slots", "covered")}
              for r in rows]
    else:
        raw = json.loads(Path(a.questions).read_text())
        qs = []
        for i, r in enumerate(raw):
            qid = r.get("question_id", i) if a.id_field == "question_id" else i
            qs.append({"id": f"{a.id_prefix}{qid}", "db_id": r["db_id"], "question": r["question"],
                       "context": r.get("evidence", ""), "gold_sql": r.get("SQL"), "in_slots": None,
                       "covered": None})
    if ids is not None:
        keep = set(ids)
        qs = [q for q in qs if q["id"] in keep]
    if a.max_q:
        qs = qs[:a.max_q]
    return qs


def build_prompts(qs: list[dict], paths: dict, a, tok=None) -> list[tuple[dict, str | None]]:
    """Prompt per question; shrinks the schema (fewer example values, then no descriptions) when over budget."""
    budget = a.max_model_len - a.max_tokens - 64
    cache, out = {}, []
    for q in qs:
        db = paths.get(q["db_id"])
        if db is None:
            out.append((q, None))
            continue
        text = None
        ctx = "" if getattr(a, "no_evidence", False) else q["context"]  # P3 probe: evidence removed from the prompt only
        for ex, desc in ((a.examples, True), (1, True), (0, True), (0, False)):
            key = (q["db_id"], ex, desc)
            # the schema part depends on the question only through matched literals; cache when there are none
            lits = P._literals(P.question_text(q["question"], ctx))
            if lits or key not in cache:
                body = P.db_details(db, P.question_text(q["question"], ctx), examples=ex, descriptions=desc)
                if not lits:
                    cache[key] = body
            else:
                body = cache[key]
            text = P.TEMPLATE.format(db_details=body, question=P.question_text(q["question"], ctx))
            n = len(tok.encode(text)) if tok is not None else len(text) // 3
            if n <= budget:
                break
            text = None
        out.append((q, text))
    return out


def main():
    ap = argparse.ArgumentParser()
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--questions", type=Path, help="BIRD-format json (list of {db_id, question, evidence, SQL})")
    src.add_argument("--pool", type=Path, help="existing candidates jsonl: reuse its questions")
    ap.add_argument("--id-prefix", default="bird_train_train_", help="--questions: id = prefix + index")
    ap.add_argument("--id-field", choices=["index", "question_id"], default="index")
    ap.add_argument("--ids", type=Path, help="manifest ({ids: [...]} or a list) restricting the questions")
    ap.add_argument("--confirm", action="store_true", help="allow the RESERVED manifest")
    ap.add_argument("--dev-look", action="store_true", help="allow a BIRD-dev question file (owner's go needed)")
    ap.add_argument("--db-roots", nargs="+", required=True)
    ap.add_argument("--model", default="seeklhy/OmniSQL-7B")
    ap.add_argument("--tag", default="", help="model tag stored in each candidate (default: last part of --model)")
    ap.add_argument("--n", type=int, default=16)
    ap.add_argument("--temp", type=float, default=0.8, help="OmniSQL paper: 0.8 for its 8-sample majority vote")
    ap.add_argument("--top-p", type=float, default=1.0)
    ap.add_argument("--greedy", action="store_true", help="also one temperature-0 answer, stored as rank 0")
    ap.add_argument("--max-tokens", type=int, default=2048, help="model card: 2048")
    ap.add_argument("--max-model-len", type=int, default=16384)
    ap.add_argument("--examples", type=int, default=2, help="example values per column (OmniSQL paper: two)")
    ap.add_argument("--tp", type=int, default=1, help="tensor parallel size (14B: 1 x 80 GB; 32B: 2)")
    ap.add_argument("--gpu-mem", type=float, default=0.90)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-q", type=int, default=0)
    ap.add_argument("--no-evidence", action="store_true",
                    help="P3 probe: drop the BIRD evidence from the prompt (records keep their context field)")
    ap.add_argument("--prompt-only", action="store_true", help="CPU: write {id, prompt, chars} and stop (no vLLM)")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    name = str(a.questions or a.pool)
    if "bird_dev" in name.lower().replace("-", "_") or "dev.json" in name.lower():
        if not a.dev_look:
            sys.exit("refusing a BIRD-dev file without --dev-look (owner's go)")
    if a.questions and "bird_train" in str(a.questions) and not a.ids:
        sys.exit("--ids is required with a BIRD-train question file (DECISION or a training-DB list)")
    qs = load_questions(a)
    paths = db_paths(a.db_roots)
    tag = a.tag or a.model.rstrip("/").split("/")[-1]
    a.out.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    if a.prompt_only:
        pr = build_prompts(qs, paths, a)
        with open(a.out, "w") as f:
            for q, text in pr:
                f.write(json.dumps({"id": q["id"], "db_id": q["db_id"], "chars": len(text or ""), "prompt": text}) + "\n")
        n = [len(t) for _, t in pr if t]
        print(f"PROMPTS {len(pr)} questions, {len(pr) - len(n)} without a prompt, chars p50 "
              f"{sorted(n)[len(n) // 2] if n else 0} max {max(n, default=0)} in {time.time() - t0:.0f}s", flush=True)
        return

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    tok = AutoTokenizer.from_pretrained(a.model)
    pr = build_prompts(qs, paths, a, tok)
    chats = [tok.apply_chat_template([{"role": "user", "content": t}], tokenize=False, add_generation_prompt=True)
             for _, t in pr if t]
    llm = LLM(model=a.model, tensor_parallel_size=a.tp, dtype="bfloat16", max_model_len=a.max_model_len,
              gpu_memory_utilization=a.gpu_mem, enable_prefix_caching=True, seed=a.seed)
    print(f"gen {len(chats)} prompts ({len(pr) - len(chats)} skipped: no DB or over length), model {a.model}", flush=True)
    outs_s = (llm.generate(chats, SamplingParams(n=a.n, temperature=a.temp, top_p=a.top_p, max_tokens=a.max_tokens,
                                                 seed=a.seed)) if a.n else [None] * len(chats))
    outs_g = llm.generate(chats, SamplingParams(n=1, temperature=0.0, max_tokens=a.max_tokens)) if a.greedy \
        else [None] * len(chats)

    def cand(o, rank, temp):
        ntok = len(o.token_ids)
        lp = (o.cumulative_logprob / ntok) if (o.cumulative_logprob is not None and ntok) else None
        return {"rank": rank, "sql": P.extract_sql(o.text), "src": "ff", "gen": tag, "temp": temp,
                "logp": None if lp is None else round(lp, 4), "ntok": ntok, "finish": o.finish_reason}

    it = iter(zip(outs_g, outs_s))
    n_none = n_c = 0
    with open(a.out, "w") as f:
        for q, text in pr:
            cl = []
            if text:
                g, s = next(it)
                if g is not None:
                    cl.append(cand(g.outputs[0], 0, 0.0))
                if s is not None:
                    cl += [cand(o, len(cl), a.temp) for o in s.outputs]
            n_c += len(cl)
            n_none += sum(c["sql"] is None for c in cl)
            f.write(json.dumps({**q, "cands": cl, "ff_model": a.model, "ff_skipped": text is None,
                                "ff_no_evidence": a.no_evidence}) + "\n")
    print(f"GEN {len(pr)} questions, {n_c} candidates ({n_none} without a SQL block), "
          f"{(time.time() - t0) / 60:.1f} min -> {a.out}", flush=True)


if __name__ == "__main__":
    main()
