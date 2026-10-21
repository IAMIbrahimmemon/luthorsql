"""Head-score dumps: every slot head's outputs for a question, written once on the GPU box, re-decoded on the Mac.

Alternative decoders (constrained MAP, per-head temperatures, set models, Steiner coherence, provenance masks,
evidence constraints, repair search) only need the heads' outputs, not the 7B backbone. One GPU pass writes them
(scripts/dump_heads.py); everything after runs on CPU from this format.

Per question the dump holds
  exact (fp32, as the box computed them)
    choice    every choice head's logits (counts, distinct, order, limit kind, compose, ...), all blocks
    ptr       (n_slots, n_options) pointer scores of every slot over every option
    rec       every score vector the box's own decoders asked for (greedy, the flip pool, lazy k-best):
              choice / ptr / pair (slot, option, head) / vals (slot, option) / limit (slot), keyed as in topk._Steer
  states (fp16 by default) from which any other pair / value / limit score is recomputed on CPU with the dumped
  head weights (heads.npz): q (normalised W_q S), vq (W_v S), k (normalised option keys), cv (normalised literal
  candidate vectors), mentioned, cand_col, om
  the decoder's decisions: greedy plan, the flip pool (rerank_sq2.py gen --k 8, the R1/R2 candidate files) and
  lazy k-best lists, each plan with its cost and compiled SQL; option ids, literal candidates, gold targets.

Replaying the recorded vectors through the unchanged decoders (targets.decode, topk.candidates, topk.kbest)
reproduces the box's greedy plan and lists exactly (DumpScores, strict=True). A query the box never made is
answered from the exact dense arrays (choice, ptr) or recomputed from the states (pair, vals, limit), which
matches the model to fp16 rounding of the states (to fp32 rounding with --state-dtype float32).

Layout of a dump directory
  manifest.json   format, checkpoint hashes, grammar / vocabulary, git revision, head spec, question ids
  heads.npz       fp32 weights of the pair heads, the value head and the per-slot pointer scale/bias
  groups.json     schema groups (questions encoded against one schema prefix): db, tables, option ids, shard
  cards/<db>.json the schema cards (plans compile on the Mac without the data directory)
  shard_NNNNN.npz arrays: g<gid>.k (group option keys), q<n>.<field> per question
  shard_NNNNN.jsonl one JSON line per question: ids, text, candidates, targets, lists, recorded keys
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from ..sq.candidates import Candidate
from ..sq.plan import Plan, compile_plan
from ..sq.schema_card import SchemaCard
from . import targets as T
from .serialize import SLOTS

FORMAT = "luthor-headdump/1"
SLOT_IX = {s: i for i, s in enumerate(SLOTS)}
STATE_FIELDS = ("q", "vq", "cv", "mentioned", "cand_col", "om")


class MissingScore(BaseException):
    """strict replay asked for a vector the box never computed. A BaseException on purpose: topk.candidates and
    topk.kbest swallow Exception from a failing re-decode, which would silently shorten a list."""


# ---- head spec ---------------------------------------------------------------------------------------------------

def choice_keys(model) -> list[str]:
    """Choice head names ("m.n_proj", ...) in the model's own order (ModuleDict keys are "m_n_proj")."""
    return [k[0] + "." + k[2:] for k in model.choice.keys()]


def grammar_name() -> str:
    from . import serialize
    if getattr(serialize, "GRAMMAR_V4", False):
        return "v4"
    return "v3" if "fn" in T.PAIRS else "v2"


def head_spec(model) -> dict:
    ck = choice_keys(model)
    return {"slots": list(SLOTS), "d_proj": model.d_proj,
            "choice": [[k, model.choice[k.replace(".", "_")].out_features] for k in ck],
            "pairs": {h: model.pair[h][-1].out_features for h in model.pair.keys()},
            "col_mentions": bool(model.col_mentions), "value_side": bool(model.value_side),
            "joint_layers": int(model.joint_layers), "grammar": grammar_name(),
            "vocab": {"templates": len(T.VOCAB["templates"]), "nums": len(T.VOCAB["nums"])}}


HEAD_KEYS = ("pair.", "value_head.", "value_role", "limit_col", "log_scale", "bias", "om_bias")


def head_weights(model) -> dict[str, np.ndarray]:
    """The weights CPU recomputation needs (pair MLPs, value MLP, roles, LIMIT stand-in, pointer scale / bias)."""
    out = {}
    for k, v in model.state_dict().items():
        if k.startswith("backbone."):
            continue
        if k.startswith(HEAD_KEYS[:2]) or k in HEAD_KEYS[2:]:
            out[k] = v.detach().float().cpu().numpy()
    return out


def _sha(path: Path, limit: int | None = None) -> str | None:
    if not path.exists():
        return None
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()[:limit] if limit else h.hexdigest()


def tensor_hash(model) -> str:
    """sha256 over the non-backbone weights (identifies a checkpoint even without its files)."""
    h = hashlib.sha256()
    for k, v in sorted(model.head_state().items()):
        h.update(k.encode())
        h.update(v.detach().float().cpu().numpy().tobytes())
    return h.hexdigest()


def checkpoint_info(run: Path | None, model) -> dict:
    info = {"heads_tensor_sha256": tensor_hash(model)}
    if run is not None:
        info.update(run=str(run), best_heads_sha256=_sha(run / "best_heads.pt"),
                    lora_sha256=_sha(run / "best_lora" / "adapter_model.safetensors"),
                    vocab_sha256=_sha(run / "vocab.json"))
        if (run / "args.json").exists():
            info["args"] = json.loads((run / "args.json").read_text())
    return info


def git_rev(repo: Path) -> dict:
    def run(*a):
        try:
            return subprocess.run(["git", *a], cwd=repo, capture_output=True, text=True, timeout=20).stdout.strip()
        except Exception:
            return ""
    return {"rev": run("rev-parse", "HEAD") or None, "dirty": bool(run("status", "--porcelain", "--untracked-files=no"))}


# ---- capture (box side) ------------------------------------------------------------------------------------------

class RecordingScores:
    """targets.Scores over another Scores that keeps every vector asked for (keys as in topk._Steer). A vector is
    computed once: later asks (another decoder's cache) get the recorded copy."""

    def __init__(self, base):
        self.base, self.rec = base, {}
        self.option_ids, self.cands = base.option_ids, base.cands

    def _get(self, key, fn):
        if key not in self.rec:
            self.rec[key] = list(fn())
        return list(self.rec[key])

    def choice(self, k):
        return self._get(("c", k), lambda: self.base.choice(k))

    def ptr(self, s):
        return self._get(("p", s), lambda: self.base.ptr(s))

    def pair(self, s, o, h):
        return self._get(("q", s, int(o), h), lambda: self.base.pair(s, o, h))

    def vals(self, s, o):
        return self._get(("v", s, int(o)), lambda: self.base.vals(s, o))

    def limit(self, s):
        return self._get(("l", s), lambda: self.base.limit(s))


def plan_entry(p: Plan, cost: float, compile_fn) -> dict:
    sql = None
    if compile_fn is not None:
        try:
            sql = compile_fn(p)
        except Exception:
            sql = None
    return {"plan": p.to_dict(), "repr": repr(p), "cost": cost, "sql": sql}


def run_lists(sc, compile_fn=None, flips_k: int = 8, kbest_ks=(8,)) -> dict:
    """The decoder's own outputs over Scores `sc`: greedy, the flip pool exactly as rerank_sq2.py gen builds it
    (topk.candidates with its defaults, k=flips_k), and lazy k-best (topk.kbest with compile_fn) for each k."""
    from .topk import candidates, kbest
    out: dict = {"errors": {}}
    try:
        out["greedy"] = plan_entry(T.decode(sc), 0.0, compile_fn)
    except Exception as ex:
        out["greedy"], out["errors"]["greedy"] = None, repr(ex)
    if flips_k:
        try:
            out["flips"] = [plan_entry(p, c, compile_fn)
                            for p, c in candidates(sc, k=flips_k, with_cost=True, compile_fn=compile_fn)]
        except Exception as ex:  # gen keeps an empty pool
            out["flips"], out["errors"]["flips"] = [], repr(ex)
    for k in kbest_ks:
        try:
            out[f"kbest{k}"] = [plan_entry(p, c, compile_fn) for p, c in kbest(sc, k, compile_fn, with_cost=True)]
        except Exception as ex:
            out[f"kbest{k}"], out["errors"][f"kbest{k}"] = [], repr(ex)
    return out


@torch.no_grad()
def capture(model, e, option_ids: list[str], cands: list[Candidate], compile_fn=None, flips_k: int = 8,
            kbest_ks=(8,)) -> tuple[dict, dict]:
    """One projected question (model.project output) -> (arrays, meta). Arrays are fp32 / int; the writer casts
    the states. Runs the decoders over the model's own scores (TorchScores), recording what they ask for."""
    from .model import TorchScores
    t0 = time.time()
    rs = RecordingScores(TorchScores(model, e, option_ids, cands))
    lists = run_lists(rs, compile_fn, flips_k, kbest_ks)
    t_dec = time.time() - t0
    ck = choice_keys(model)
    choice = [model.choice_logits(e, k).float().cpu() for k in ck]
    arr = {"choice": torch.cat(choice).numpy(),
           "ptr": torch.stack([model.ptr_scores(e, s).float() for s in SLOTS]).cpu().numpy(),
           "q": e.q.float().cpu().numpy(), "vq": model.W_v(e.S).float().cpu().numpy(),
           "k": e.k.float().cpu().numpy(), "cv": e.cv.float().cpu().numpy(),
           "mentioned": e.mentioned.float().cpu().numpy(), "cand_col": e.cand_col.cpu().numpy().astype(np.int32)}
    if e.om is not None:
        arr["om"] = e.om.float().cpu().numpy()
    keys = list(rs.rec)
    flat = [x for k in keys for x in rs.rec[k]]
    rec = np.asarray(flat, dtype=np.float64)
    r32 = rec.astype(np.float32)
    arr["rec"] = r32 if np.array_equal(r32.astype(np.float64), rec) else rec  # fp32 heads: lossless as fp32
    arr["rec_off"] = np.cumsum([0] + [len(rs.rec[k]) for k in keys]).astype(np.int64)
    meta = {"rec_keys": [list(k) for k in keys], "lists": lists,
            "cands": [{"kind": c.kind, "value": c.value, "col": c.col, "span": list(c.span) if c.span else None}
                      for c in cands],
            "n_options": len(option_ids), "decode_s": round(t_dec, 4)}
    return arr, meta


def targets_dict(t) -> dict | None:
    return None if t is None else dataclasses.asdict(t)


# ---- writer ------------------------------------------------------------------------------------------------------

class DumpWriter:
    """Writes shards of at most `shard_size` questions (a schema group never spans two shards)."""

    def __init__(self, out: Path, manifest: dict, heads: dict[str, np.ndarray], state_dtype: str = "float16",
                 shard_size: int = 256, compress: bool = True):
        self.out = Path(out)
        self.out.mkdir(parents=True, exist_ok=True)
        (self.out / "cards").mkdir(exist_ok=True)
        np.savez(self.out / "heads.npz", **heads)
        self.manifest = dict(manifest, format=FORMAT, state_dtype=state_dtype, created=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
        self.sdt = np.dtype(state_dtype)
        self.shard_size, self.compress = shard_size, compress
        self.groups: list[dict] = []
        self.index: list[list] = []  # [id, shard, n]
        self.cards: set[str] = set()
        self._arr, self._lines, self._shard = {}, [], 0
        self._k = None

    def add_card(self, card: SchemaCard) -> None:
        if card.db_id not in self.cards:
            card.save(self.out / "cards" / f"{card.db_id}.json")
            self.cards.add(card.db_id)

    def begin_group(self, db_id: str, tables: list[str] | None, option_ids: list[str]) -> int:
        if len(self._lines) >= self.shard_size:
            self._flush()
        gid = len(self.groups)
        self.groups.append({"gid": gid, "db_id": db_id, "tables": tables, "option_ids": list(option_ids),
                            "shard": self._shard})
        self._k, self._gid = None, gid
        return gid

    def add(self, arr: dict, meta: dict) -> None:
        n = len(self.index)
        p = f"q{n}."
        k = arr["k"]
        if self._k is None:  # the group's keys; later questions store only the rows that differ (mentions)
            self._k = k
            self._arr[f"g{self._gid}.k"] = k.astype(self.sdt)
        diff = np.nonzero((k != self._k).any(1))[0].astype(np.int32)
        self._arr[p + "k_idx"] = diff
        self._arr[p + "k_rows"] = k[diff].astype(self.sdt)
        for f in ("choice", "ptr", "rec", "rec_off", "cand_col"):
            self._arr[p + f] = arr[f]
        for f in ("q", "vq", "cv", "mentioned", "om"):
            if f in arr:
                self._arr[p + f] = arr[f].astype(self.sdt) if f in ("q", "vq", "cv") else arr[f].astype(np.float32)
        self._lines.append(json.dumps({**meta, "n": n, "gid": self._gid}, default=str))
        self.index.append([meta["id"], self._shard, n])

    def _flush(self) -> None:
        if not self._lines:
            return
        name = self.out / f"shard_{self._shard:05d}"
        (np.savez_compressed if self.compress else np.savez)(str(name) + ".npz", **self._arr)
        Path(str(name) + ".jsonl").write_text("\n".join(self._lines) + "\n")
        self._arr, self._lines = {}, []
        self._shard += 1

    def close(self, extra: dict | None = None) -> dict:
        self._flush()
        (self.out / "groups.json").write_text(json.dumps(self.groups))
        m = dict(self.manifest, **(extra or {}), n_questions=len(self.index), n_groups=len(self.groups),
                 n_shards=self._shard, ids=[i for i, _, _ in self.index], index=self.index)
        size = sum(p.stat().st_size for p in self.out.rglob("*") if p.is_file())
        m["bytes"] = size
        (self.out / "manifest.json").write_text(json.dumps(m, indent=1, default=str))
        return m


# ---- loader + CPU scores ------------------------------------------------------------------------------------------

class Heads:
    """CPU (torch, fp32) copies of the pair / value heads for recomputing scores from dumped states."""

    def __init__(self, w: dict[str, np.ndarray]):
        self.w = {k: torch.from_numpy(np.asarray(v, dtype=np.float32)) for k, v in w.items()}
        self.pairs = sorted({k.split(".")[1] for k in w if k.startswith("pair.")})

    def _mlp(self, prefix: str, x: torch.Tensor) -> torch.Tensor:
        w = self.w
        h = F.gelu(F.linear(x, w[prefix + "0.weight"], w[prefix + "0.bias"]))
        return F.linear(h, w[prefix + "2.weight"], w[prefix + "2.bias"])

    def pair(self, q: torch.Tensor, k: torch.Tensor, head: str) -> torch.Tensor:
        qq = q.expand_as(k)
        return self._mlp(f"pair.{head}.", torch.cat([qq, k, qq * k], -1))

    def value(self, vq: torch.Tensor, role: int, col: torch.Tensor | None, cv: torch.Tensor, cand_col: torch.Tensor,
              mentioned: torch.Tensor, opt: int | None) -> torch.Tensor:
        if cv.shape[0] == 0:
            return cv.new_zeros(0)
        qv = F.normalize(vq + self.w["value_role"][role], dim=-1).expand_as(cv)
        c = (col if col is not None else F.normalize(self.w["limit_col"], dim=-1)).expand_as(cv)
        same = ((cand_col == opt).float() if opt is not None else torch.zeros_like(mentioned))[:, None]
        x = torch.cat([qv, c, cv, c * cv, same, mentioned[:, None]], -1)
        return self._mlp("value_head.", x).squeeze(-1)


class _KeyedList(list):
    """A score vector that knows its decision key (targets._TRACE records decisions on such lists)."""
    key = None


class QDump:
    """One dumped question. `arrays` are this question's npz entries (prefix stripped); states come back as fp32."""

    def __init__(self, dump: "HeadDump", meta: dict, arrays: dict, group: dict, k_base: np.ndarray):
        self.dump, self.meta, self.a, self.group = dump, meta, arrays, group
        k = k_base.astype(np.float32).copy()
        if len(arrays["k_idx"]):
            k[arrays["k_idx"]] = arrays["k_rows"].astype(np.float32)
        self.k = k
        self.id, self.db_id = meta["id"], meta["db_id"]
        self.option_ids = group["option_ids"]
        self.cands = [Candidate(c["kind"], c["value"], c["col"], tuple(c["span"]) if c["span"] else None)
                      for c in meta["cands"]]
        offs = arrays["rec_off"]
        rec = arrays["rec"]
        self.recorded = {_key(k): rec[offs[i]:offs[i + 1]] for i, k in enumerate(meta["rec_keys"])}
        ck = dump.manifest["heads"]["choice"]
        o = np.cumsum([0] + [n for _, n in ck])
        self.choice = {name: arrays["choice"][o[i]:o[i + 1]] for i, (name, _) in enumerate(ck)}
        self.ptr = arrays["ptr"]  # (n_slots, n_options), fp32 exact

    def state(self, name: str) -> np.ndarray | None:
        x = self.a.get(name)
        return None if x is None else x.astype(np.float32)

    @property
    def lists(self) -> dict:
        return self.meta["lists"]

    def plans(self, name: str) -> list[tuple[Plan, float]]:
        v = self.meta["lists"].get(name)
        if v is None:
            return []
        v = [v] if isinstance(v, dict) else v
        return [(Plan.from_dict(x["plan"]), x["cost"]) for x in v]

    def scores(self, recorded: bool = True, strict: bool = False, keyed: bool = False) -> "DumpScores":
        return DumpScores(self, recorded=recorded, strict=strict, keyed=keyed)

    def compile_fn(self):
        g = self.dump.graph(self.db_id)
        return lambda p: compile_plan(p, g)


def _key(k: list) -> tuple:
    return tuple(int(x) if i == 2 and k[0] in ("q", "v") else x for i, x in enumerate(k))


class DumpScores:
    """targets.Scores over a dumped question.

    recorded=True: vectors the box computed are served exactly (replay). Otherwise / on a miss: choice and ptr from
    the exact dense arrays, pair / vals / limit recomputed from the dumped states (fp16-rounded by default).
    strict=True raises MissingScore on any vector the box did not record (proves an exact replay).
    keyed=True returns lists carrying .key, for targets._TRACE (see `decisions`)."""

    def __init__(self, qd: QDump, recorded: bool = True, strict: bool = False, keyed: bool = False):
        self.qd, self.use_rec, self.strict, self.keyed = qd, recorded, strict, keyed
        self.option_ids, self.cands = qd.option_ids, qd.cands
        self.hits = self.dense = self.recomputed = 0
        self._t: dict = {}

    def _out(self, key, xs):
        xs = [float(x) for x in xs]
        if self.keyed:
            xs = _KeyedList(xs)
            xs.key = key
        return xs

    def _get(self, key, dense):
        if self.use_rec and key in self.qd.recorded:
            self.hits += 1
            return self._out(key, self.qd.recorded[key])
        if self.strict:
            raise MissingScore(key)
        xs = dense()
        return self._out(key, xs)

    def _tens(self, name):
        if name not in self._t:
            x = self.qd.k if name == "k" else self.qd.state(name)
            self._t[name] = None if x is None else torch.from_numpy(x)
        return self._t[name]

    def choice(self, k):
        def d():
            self.dense += 1
            return self.qd.choice[k]
        return self._get(("c", k), d)

    def ptr(self, s):
        def d():
            self.dense += 1
            return self.qd.ptr[SLOT_IX[s]]
        return self._get(("p", s), d)

    def pair(self, s, o, h):
        def d():
            self.recomputed += 1
            with torch.no_grad():
                return self.qd.dump.heads.pair(self._tens("q")[SLOT_IX[s]], self._tens("k")[int(o)], h).tolist()
        return self._get(("q", s, int(o), h), d)

    def _value(self, s, o):
        slot, _, role = s.partition("|t")
        with torch.no_grad():
            return self.qd.dump.heads.value(
                self._tens("vq")[SLOT_IX[slot]], int(role) + 1 if role else 0,
                self._tens("k")[int(o)] if o is not None else None, self._tens("cv"),
                torch.from_numpy(self.qd.a["cand_col"].astype(np.int64)), self._tens("mentioned"),
                int(o) if o is not None else None).tolist()

    def vals(self, s, o):
        def d():
            self.recomputed += 1
            return self._value(s, o)
        return self._get(("v", s, int(o)), d)

    def limit(self, s):
        def d():
            self.recomputed += 1
            return self._value(s, None)
        return self._get(("l", s), d)


class HeadDump:
    """A dump directory. Iterating loads one shard at a time."""

    def __init__(self, path: Path | str, check_grammar: bool = True):
        self.path = Path(path)
        self.manifest = json.loads((self.path / "manifest.json").read_text())
        if self.manifest.get("format") != FORMAT:
            raise ValueError(f"{path}: format {self.manifest.get('format')}, expected {FORMAT}")
        if check_grammar and self.manifest["heads"]["slots"] != list(SLOTS):
            raise ValueError(f"{path}: dumped with grammar {self.manifest['heads']['grammar']} slots; this process "
                             f"has {len(SLOTS)} different slots (set LUTHOR_GRAMMAR=v3 for R1/R2 dumps)")
        self.groups = json.loads((self.path / "groups.json").read_text())
        self.heads = Heads(dict(np.load(self.path / "heads.npz")))
        self.where = {i: (s, n) for i, s, n in self.manifest["index"]}
        self._shard_no, self._npz, self._meta = None, None, None
        self._cards: dict = {}

    def __len__(self):
        return len(self.manifest["ids"])

    @property
    def ids(self) -> list[str]:
        return self.manifest["ids"]

    def _load(self, s: int):
        if self._shard_no != s:
            base = self.path / f"shard_{s:05d}"
            self._npz = dict(np.load(str(base) + ".npz"))
            self._meta = {m["n"]: m for m in map(json.loads, open(str(base) + ".jsonl"))}
            self._shard_no = s

    def question(self, qid: str) -> QDump:
        s, n = self.where[qid]
        self._load(s)
        m = self._meta[n]
        p = f"q{n}."
        arrays = {k[len(p):]: v for k, v in self._npz.items() if k.startswith(p)}
        g = self.groups[m["gid"]]
        return QDump(self, m, arrays, g, self._npz[f"g{m['gid']}.k"])

    def __iter__(self):
        for i in sorted(self.ids, key=lambda i: self.where[i]):  # shard order: each shard loads once
            yield self.question(i)

    def card(self, db: str) -> SchemaCard:
        if db not in self._cards:
            self._cards[db] = SchemaCard.load(self.path / "cards" / f"{db}.json")
        return self._cards[db]

    def graph(self, db: str):
        key = ("g", db)
        if key not in self._cards:
            self._cards[key] = self.card(db).graph()
        return self._cards[key]


# ---- CPU re-decoding ---------------------------------------------------------------------------------------------

def redecode(qd: QDump, recorded: bool = True, strict: bool = False, flips_k: int | None = None,
             kbest_ks=None, compile: bool = True) -> tuple[dict, DumpScores]:
    """Greedy, flip pool and k-best for a dumped question, with the same decoder code the box ran. Defaults: the
    lists the dump was written with."""
    m = qd.dump.manifest.get("lists", {})
    flips_k = m.get("flips_k", 8) if flips_k is None else flips_k
    kbest_ks = m.get("kbest_ks", [8]) if kbest_ks is None else kbest_ks
    ds = qd.scores(recorded=recorded, strict=strict)
    return run_lists(ds, qd.compile_fn() if compile else None, flips_k, kbest_ks), ds


def same_lists(a: dict, b: dict, cost_tol: float = 0.0) -> dict[str, bool]:
    """Per list name: same plans (repr), same order, same SQL, costs equal (within cost_tol: recomputed scores)."""
    out = {}
    for name in b:
        if name == "errors":
            continue
        x, y = a.get(name), b.get(name)
        x = [x] if isinstance(x, dict) else (x or [])
        y = [y] if isinstance(y, dict) else (y or [])
        out[name] = len(x) == len(y) and all(p["repr"] == q["repr"] and p["sql"] == q["sql"] and
                                             abs(p["cost"] - q["cost"]) <= cost_tol
                                             for p, q in zip(x, y))
    return out


def decisions(qd: QDump, recorded: bool = True) -> list[tuple]:
    """The greedy decoder's decisions: (key, chosen indices in order, allowed indices or None, approximate), as
    targets._TRACE records them (RL's trace format)."""
    ds = qd.scores(recorded=recorded, keyed=True)
    old = T._TRACE
    T._TRACE = []
    try:
        T.decode(ds)
        return list(T._TRACE)
    finally:
        T._TRACE = old
