"""Keep reward-learning prompts the SQL writer sometimes gets right and sometimes wrong (round 4).
A question it always or never answers gives GRPO no signal (every sample in the group gets the same reward).

  python scripts/sqlw_rlfilter.py --model /workspace/m/sqlw_sft --data rl.jsonl --db-roots ... --out rl_hard.jsonl
"""
from __future__ import annotations

import argparse
import collections
import json
import sys
from multiprocessing import Pool
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from luthor.sqlw.execute import check_db  # noqa: E402
from luthor.sqlw.prompt import clean_sql  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--db-roots", nargs="+", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--n", type=int, default=8)
    a = ap.parse_args()
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    paths = {}
    for root in a.db_roots:
        for p in Path(root).rglob("*.sqlite"):
            paths.setdefault(p.stem, str(p))
    rows = [json.loads(l) for l in open(a.data)]
    rows = [r for r in rows if r["db_id"] in paths]
    tok = AutoTokenizer.from_pretrained(a.model)
    n_all = len(rows)  # GRPO's vLLM holds 4352 tokens: prompt <= 4096 plus 256 for the answer
    def n_tok(r):
        text = tok.apply_chat_template(r["messages"], tokenize=False, add_generation_prompt=True)
        return len(tok(text, add_special_tokens=False)["input_ids"])
    rows = [r for r in rows if n_tok(r) <= 4096]
    print(f"prompts within 4096 tokens: {len(rows)} of {n_all}", flush=True)
    llm = LLM(a.model, max_model_len=4352, gpu_memory_utilization=0.85, seed=0)
    outs = llm.chat([r["messages"] for r in rows], SamplingParams(temperature=1.0, n=a.n, max_tokens=256, seed=0))
    by_db = collections.defaultdict(list)
    for i, r in enumerate(rows):
        by_db[r["db_id"]].append(i)
    right = [0] * len(rows)
    tasks = [(paths[db], [(rows[i]["gold_sql"], [clean_sql(o.text) for o in outs[i].outputs]) for i in idx], 3.0)
             for db, idx in by_db.items()]
    with Pool(16) as pool:
        for idx, res in zip(by_db.values(), pool.map(check_db, tasks)):
            for i, rr in zip(idx, res):
                right[i] = -1 if rr is None else sum(x[2] for x in rr)
    hist = collections.Counter(right)
    keep = [r for r, k in zip(rows, right) if 0 < k < a.n]
    a.out.write_text("".join(json.dumps(r) + "\n" for r in keep))
    # every question's right-count (round 5: the 0/8 questions are a training target of their own)
    a.out.with_name(a.out.stem + "_counts.jsonl").write_text(
        "".join(json.dumps({"id": r.get("id"), "db_id": r["db_id"], "right": k}) + "\n" for r, k in zip(rows, right)))
    print(f"RLFILTER right-count histogram {dict(sorted(hist.items()))}; kept {len(keep)} of {len(rows)}; "
          f"sampled accuracy {sum(max(k, 0) for k in right) / (a.n * len(rows)):.3f}", flush=True)


if __name__ == "__main__":
    main()
