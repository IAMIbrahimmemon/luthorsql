import json, re, sys, collections
from multiprocessing import Pool
sys.path.insert(0, "src")
from luthor.sqlw.execute import connect, signature
R = "data/raw/bird_train/train_databases"
V = json.load(open('data/raw/bird_verified/bird-verified-train.json')) + json.load(open('data/raw/bird_verified/bird-verified-val.json'))
T = json.load(open('data/raw/bird_train/train.json'))
for v in V:
    t = T[int(v['question_id'])]
    assert t['db_id'] == v['db_id'], (v['question_id'], t['db_id'], v['db_id'])
    v['original_SQL'] = v.get('original_SQL') or t['SQL']
    v['original_question'] = v.get('original_question') or t['question']
def sig(db, s):
    try:
        return signature(connect(f"{R}/{db}/{db}.sqlite", 20)(s))
    except Exception as e:
        return "ERR"
def work(v):
    return sig(v['db_id'], v['SQL']), sig(v['db_id'], v['original_SQL'])
if __name__ == "__main__":
    with Pool(10) as p:
        res = p.map(work, V, chunksize=8)
    n = lambda s: re.sub(r'\W+', ' ', (s or '').lower()).strip()
    filt = {(json.loads(l)['db_id'], n(json.loads(l)['question'])) for l in open('data/sys1/bird_filtered.jsonl')}
    plans = {(json.loads(l)['db_id'], n(json.loads(l)['question'])) for l in open('data/sys1/bird_train/plans.jsonl')}
    diff = [(v, a, b) for v, (a, b) in zip(V, res) if a != b]
    print(f"verified {len(V)}: result changed {len(diff)} ({len(diff)/len(V):.0%}); new gold errors {sum(a=='ERR' for a,_ in res)}, old gold errors {sum(b=='ERR' for _,b in res)}")
    k = lambda v: (v['db_id'], n(v['original_question']))
    same = [v for v, (a, b) in zip(V, res) if a == b]
    print(f"official filter keeps {sum(k(v) in filt for v,_,_ in diff)}/{len(diff)} wrong-result items, {sum(k(v) in filt for v in same)}/{len(same)} same-result items")
    print(f"in our plan set: wrong-result {sum(k(v) in plans for v,_,_ in diff)}, same-result {sum(k(v) in plans for v in same)}")
    out = {f"{v['db_id']}|{n(v['original_question'])}": {"changed": a != b, "sql": v['SQL'], "question": v['question'], "evidence": v['evidence'], "split": "val" if i >= 2064 else "train"} for i, (v, (a, b)) in enumerate(zip(V, res))}
    json.dump(out, open("data/raw/bird_verified/cmp.json", "w"))
