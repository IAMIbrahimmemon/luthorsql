import json, sys, random
def load(p): return {json.loads(l)["id"]: json.loads(l) for l in open(p)}
B0, B1 = load(sys.argv[1]), load(sys.argv[2])            # scored cands
S0, S1 = load(sys.argv[3]), load(sys.argv[4])            # stack rows
ids = sorted(B0); assert set(ids) == set(B1) == set(S0) == set(S1)
def first(r): return bool(r["cands"]) and bool(r["cands"][0].get("ok"))
def orc(r): return any(c.get("ok") for c in r["cands"])
M = {"first pick": (lambda i: first(B0[i]), lambda i: first(B1[i])),
     "validator": (lambda i: S0[i]["res"]["V:-"][0], lambda i: S1[i]["res"]["V:-"][0]),
     "oracle@8": (lambda i: orc(B0[i]), lambda i: orc(B1[i])),
     "VAL+DROP (D)": (lambda i: S0[i]["res"]["V:VAL+DROP"][0], lambda i: S1[i]["res"]["V:VAL+DROP"][0])}
dbs = {}
for i in ids: dbs.setdefault(B0[i]["db_id"], []).append(i)
keys = sorted(dbs); random.seed(0)
for name, (f0, f1) in M.items():
    a = sum(map(f0, ids)); b = sum(map(f1, ids)); n = len(ids)
    d = {i: int(bool(f1(i))) - int(bool(f0(i))) for i in ids}
    boots = []
    for _ in range(2000):
        s = [i for k in random.choices(keys, k=len(keys)) for i in dbs[k]]
        boots.append(100 * sum(d[i] for i in s) / len(s))
    boots.sort()
    win = sum(v == 1 for v in d.values()); loss = sum(v == -1 for v in d.values())
    print(f"{name:13s} {100*a/n:6.2f} -> {100*b/n:6.2f}  delta {100*(b-a)/n:+5.2f}  95% DB-CI [{boots[50]:+.2f}, {boots[1949]:+.2f}]  wins {win} losses {loss}")
