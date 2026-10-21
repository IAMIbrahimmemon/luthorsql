"""Dump every slot head's scores per question (GPU, once), then re-decode on the Mac without the model.

  dump    one GPU pass over a question set, exactly as rerank_sq2.py gen selects and encodes it (same questions,
          same serve-time literal candidates, same chunks of 8): per question the exact choice / pointer scores,
          every vector the decoders asked for, the projected states for recomputing any other pair / value score,
          and the decoder's own greedy plan, flip pool (gen --k 8) and lazy k-best lists (src/luthor/sq2/headdump.py).
            python scripts/dump_heads.py dump --run runs/r1r2/r1 --plans D/sys1/bird_train --db-roots D/sys1/dbs \
                --dev-dbs D/sys1/dev_dbs_bird.json --split dev --out dumps/r1/heldout \
                --check-cands runs/r1r2/r1/rr/cands_dev.jsonl
  verify  CPU: replay the dump through the unchanged decoders and check greedy, flips and k-best come back
          identical (plans, order, costs, SQL). --no-recorded instead re-decodes from dense scores + recomputed
          states only and reports how often the lists still agree (the fp16 recompute path).
  info    the manifest in short.

Grammar: --grammar, else LUTHOR_GRAMMAR, else the run's args.json grammar (missing = v3; R1/R2 are v3), and it must
match the run (verify: the dump's manifest). Only
held-out training databases and BIRD dev are read; nothing is tuned here.
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import random
import sqlite3
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))
if __name__ == "__main__":  # --grammar: LUTHOR_GRAMMAR before the first luthor.sq import (workers inherit it)
    from luthor.grammar_cli import preset
    preset(run_flags=("--run", "--dump"))

from luthor import grammar_cli  # noqa: E402
from luthor.sq.plan import compile_plan  # noqa: E402
from luthor.sq2 import headdump as H  # noqa: E402


def gold_sql(r: dict) -> str | None:
    """eval_sq2.gold_sql (the verified compiled plan, else the original SQL)."""
    return r.get("compiled_sql") or r.get("gold_sql")


# ---- question selection: rerank_sq2.gen's, line for line ---------------------------------------------------------

def select_questions(a):
    from train_sq2 import load_sets
    cards, exs, gold = load_sets(a.plans, strict=a.split == "train", all_rows=a.all_questions and a.split == "dev",
                                 questions=a.questions)
    exs = [e for e in exs if not gold[e.id].get("twin_of")]
    dev = set(json.loads(a.dev_dbs.read_text()))
    paths = {}
    for root in a.db_roots:
        for p in Path(root).rglob("*.sqlite"):
            paths.setdefault(p.stem, str(p))
    exs = [e for e in exs if (e.db_id in dev) == (a.split == "dev") and e.db_id in paths]
    if a.split == "train":  # spread over databases; drop exact question repeats
        rnd = random.Random(0)
        rnd.shuffle(exs)
        seenq, keep = set(), []
        for e in exs:
            k = (e.db_id, e.question.lower())
            if k not in seenq:
                seenq.add(k)
                keep.append(e)
        exs = keep
    if a.max_q:
        exs = exs[:a.max_q]
    by_db = collections.defaultdict(list)
    for e in exs:
        by_db[e.db_id].append(e)
    return cards, by_db, gold, paths


def live_cands(card, group, path: str, cand_order: str, gold: dict) -> None:
    """Serve-time literal candidates (card values + live database lookup), as gen builds them."""
    from luthor.sq.candidates import ValueIndex, find_candidates
    from luthor.sq2.train import question_body, select_cands
    try:
        con = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
        con.text_factory = lambda b: b.decode("utf-8", "replace")
        vindex = ValueIndex.build(con, card)
        for e in group:
            body = question_body(gold[e.id])
            e.cands, _ = select_cands(find_candidates(body, card, extra=vindex.match(body)), order=cand_order, body=body)
        con.close()
    except Exception as ex:
        print(f"value index failed on {card.db_id}: {ex}", flush=True)


@torch.no_grad()
def dump_group(m, card, exs, writer: H.DumpWriter, gold: dict, flips_k: int = 8, kbest_ks=(8,), bs: int = 8) -> int:
    """Encode one database's questions in gen's chunks and write each question's dump. Returns questions written."""
    from luthor.sq2.train import encode_db, serve_chunks
    g = card.graph()
    compile_fn = lambda p: compile_plan(p, g)  # noqa: E731
    writer.add_card(card)
    n = 0
    for tabs, chunk in serve_chunks(card, exs, bs):
        st, oix, enc = encode_db(m, card, chunk, tabs)
        writer.begin_group(card.db_id, tabs, st.option_ids)
        for e, x in zip(chunk, enc):
            arr, meta = H.capture(m, x, st.option_ids, e.cands, compile_fn, flips_k, kbest_ks)
            r = gold.get(e.id, {})
            meta.update(id=e.id, db_id=card.db_id, question=e.question, context=e.context, gold_sql=gold_sql(r),
                        in_slots=e.t is not None, covered=r.get("outcome") == "plan", targets=H.targets_dict(e.t))
            writer.add(arr, meta)
            n += 1
    return n


def check_cands(writer_out: Path, cands_file: Path) -> dict:
    """The dump's flip pool against a rerank_sq2 gen candidates file (same model, same questions): the share of
    questions whose candidate SQL lists are identical. Differences mean nondeterminism or a code change."""
    ref = {}
    for line in open(cands_file):
        r = json.loads(line)
        ref[r["id"]] = [c["sql"] for c in r["cands"]]
    d = H.HeadDump(writer_out)
    same = n = missing = 0
    for qd in d:
        if qd.id not in ref:
            missing += 1
            continue
        n += 1
        same += [c["sql"] for c in qd.lists.get("flips", [])] == ref[qd.id]
    return {"file": str(cands_file), "compared": n, "identical": same, "share": round(same / max(n, 1), 4),
            "not_in_file": missing, "file_only": len(set(ref) - set(d.ids))}


def cmd_dump(a):
    from eval_sq2 import load_model
    from rerank_sq2 import use_run_settings
    use_run_settings(a)
    cards, by_db, gold, paths = select_questions(a)
    print(f"dump {a.split}: {sum(map(len, by_db.values()))} questions on {len(by_db)} databases", flush=True)
    t0 = time.time()
    m = load_model(a.run, a.base, a.subfolder, a.device)
    t_load = time.time() - t0
    manifest = {"checkpoint": H.checkpoint_info(a.run, m), "git": H.git_rev(ROOT), "heads": H.head_spec(m),
                "source": {"plans": [str(p) for p in a.plans], "db_roots": a.db_roots, "dev_dbs": str(a.dev_dbs),
                           "split": a.split, "all_questions": a.all_questions, "questions": str(a.questions),
                           "max_q": a.max_q, "cand_order": a.cand_order, "name": a.name},
                "lists": {"flips_k": a.flips_k, "kbest_ks": a.kbest_k}, "torch": torch.__version__,
                "device": a.device, "luthor_grammar_env": os.environ.get("LUTHOR_GRAMMAR")}
    w = H.DumpWriter(a.out, manifest, H.head_weights(m), a.state_dtype, a.shard_size)
    t1, n = time.time(), 0
    for i, (db, group) in enumerate(by_db.items()):
        live_cands(cards[db], group, paths[db], a.cand_order, gold)
        n += dump_group(m, cards[db], group, w, gold, a.flips_k, a.kbest_k)
        print(f"  dumped {n} questions, {i + 1}/{len(by_db)} dbs, {time.time() - t1:.0f}s", flush=True)
    t_dump = time.time() - t1
    if a.device.startswith("cuda"):
        peak = torch.cuda.max_memory_allocated() / 2**30
    else:
        peak = None
    man = w.close({"timing": {"load_s": round(t_load, 1), "dump_s": round(t_dump, 1),
                              "per_question_s": round(t_dump / max(n, 1), 4), "peak_gb": peak}})
    print(f"DUMP {a.out}: {n} questions, {man['bytes'] / 2**20:.1f} MiB, {t_dump / 60:.1f} min "
          f"(load {t_load / 60:.1f} min)", flush=True)
    if a.check_cands is not None and a.check_cands.exists():
        c = check_cands(a.out, a.check_cands)
        print(f"CHECK_CANDS {json.dumps(c)}", flush=True)
        man["check_cands"] = c
        (a.out / "manifest.json").write_text(json.dumps(man, indent=1, default=str))


def cmd_verify(a) -> dict:
    d = H.HeadDump(a.dump)
    tally, n, t0 = collections.Counter(), 0, time.time()
    src = collections.Counter()
    bad = []
    for qd in d:
        if a.limit and n >= a.limit:
            break
        n += 1
        lists, ds = H.redecode(qd, recorded=not a.no_recorded, strict=not a.no_recorded)
        src.update(hits=ds.hits, dense=ds.dense, recomputed=ds.recomputed)
        for name, ok in H.same_lists(lists, qd.lists, 1e-3 if a.no_recorded else 0.0).items():
            tally[name] += ok
            if not ok and len(bad) < 20:
                bad.append((qd.id, name))
    out = {"dump": str(a.dump), "questions": n, "mode": "recompute" if a.no_recorded else "replay",
           "identical": {k: round(v / max(n, 1), 4) for k, v in sorted(tally.items())}, "sources": dict(src),
           "ms_per_question": round(1000 * (time.time() - t0) / max(n, 1), 1), "first_differences": bad}
    print(f"VERIFY {json.dumps(out)}", flush=True)
    if not a.no_recorded and any(v < n for v in tally.values()):
        sys.exit(1)
    return out


def cmd_info(a):
    m = json.loads((a.dump / "manifest.json").read_text())
    keep = {k: m.get(k) for k in ("format", "created", "n_questions", "n_groups", "n_shards", "bytes", "state_dtype",
                                  "lists", "git", "timing", "check_cands", "device")}
    keep["grammar"] = m["heads"]["grammar"]
    keep["checkpoint"] = {k: v for k, v in m["checkpoint"].items() if k != "args"}
    keep["source"] = m.get("source")
    print(json.dumps(keep, indent=1))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("dump")
    d.add_argument("--run", type=Path, required=True, help="run dir with best_lora/, best_heads.pt, args.json, vocab.json")
    d.add_argument("--plans", nargs="+", type=Path, required=True)
    d.add_argument("--db-roots", nargs="+", required=True)
    d.add_argument("--dev-dbs", type=Path, required=True)
    d.add_argument("--split", choices=["train", "dev"], default="dev")
    d.add_argument("--all-questions", action="store_true", help="as rerank_sq2 gen: uncovered questions too")
    d.add_argument("--questions", type=Path, help="with --all-questions: BIRD-format dev.json")
    d.add_argument("--max-q", type=int, default=0)
    d.add_argument("--out", type=Path, required=True)
    d.add_argument("--name", default="", help="free label stored in the manifest (e.g. heldout1000, bird_dev)")
    d.add_argument("--flips-k", type=int, default=8, help="flip pool size (gen --k); 0 = none")
    d.add_argument("--kbest-k", type=int, nargs="*", default=[8], help="lazy k-best list sizes to record")
    d.add_argument("--state-dtype", choices=["float16", "float32"], default="float16")
    d.add_argument("--shard-size", type=int, default=256)
    d.add_argument("--check-cands", type=Path, help="a gen candidates file of the same run and questions")
    d.add_argument("--device", default="cuda")
    d.add_argument("--base", default="AlexWortega/openjev")
    d.add_argument("--subfolder", default="qwen3.5-4b-nli-v5")
    grammar_cli.add_argument(d)
    v = sub.add_parser("verify")
    v.add_argument("--dump", type=Path, required=True)
    v.add_argument("--limit", type=int, default=0)
    v.add_argument("--no-recorded", action="store_true", help="ignore recorded vectors: dense + recomputed states")
    grammar_cli.add_argument(v)
    i = sub.add_parser("info")
    i.add_argument("--dump", type=Path, required=True)
    a = ap.parse_args(argv)
    if a.cmd == "verify":  # the dump's grammar (manifest) must be this process's; HeadDump checks the slots too
        a.grammar = grammar_cli.resolve(a.grammar)
        g = grammar_cli.dump_grammar(a.dump)
        if g in grammar_cli.CHOICES and g != a.grammar:
            raise SystemExit(f"{a.dump} was dumped with grammar {g}, this process runs {a.grammar} (--grammar {g})")
    if a.cmd == "dump":
        torch.backends.cuda.enable_cudnn_sdp(False)  # as rerank_sq2 / train_sq2
        cmd_dump(a)
    elif a.cmd == "verify":
        return cmd_verify(a)
    else:
        cmd_info(a)


if __name__ == "__main__":
    main()
