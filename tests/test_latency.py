"""System-1 latency (scripts/analysis/latency/sys1_latency.py, scripts/chains/{latency_box,run_latency_box}.sh): the
seeded DECISION sample, label stripping in the staged rows, DRY runs of both shell scripts, and a CPU smoke run of the
whole in-process measurement with tiny random models (CUDA calls stubbed). No label, BIRD dev, RESERVED or sealed file
is read (the sample reads DECISION ids and BIRD-train question text only)."""
from __future__ import annotations

import importlib.util
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
DEC_DBS = ["car_retails", "cs_semester", "hockey", "law_episode", "movie", "student_loan", "synthea", "university"]
FORBIDDEN = ("bird_dev", "dev_databases", "sealed", "reserved", "gate", "dev.json", "spider_dk", "kaggledbqa",
             "realistic", "heldout_pool", "k32_pool", "cands_bird_dev")
ROW_KEYS = {"id", "db_id", "question", "context", "outcome"}


def _mod(name, path):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


SL = _mod("sys1_latency", "scripts/analysis/latency/sys1_latency.py")
have_data = (ROOT / "data/sys1/heldout_v2/decision.json").exists() and (ROOT / "data/sys1/bird_train/plans.jsonl").exists()
need_data = pytest.mark.skipif(not have_data, reason="DECISION ids / BIRD-train plan rows not on this machine")
have_dbs = all((ROOT / f"data/sys1/dbs/{d}/{d}.sqlite").exists() for d in DEC_DBS)


# ---- sample -----------------------------------------------------------------------------------------------------------
def test_sample_seeded_and_order_free():
    ids = [f"q{i}" for i in range(500)]
    w1, m1 = SL.sample_ids(ids, set(ids))
    w2, m2 = SL.sample_ids(list(reversed(ids)), set(ids))  # file order does not matter
    assert (w1, m1) == (w2, m2) and len(w1) == 3 and len(m1) == 200 and not set(w1) & set(m1)
    assert SL.sample_ids(ids, set(ids), seed="other") != (w1, m1)
    w3, m3 = SL.sample_ids(ids, set(ids[:300]))  # only ids with a question row are eligible
    assert set(w3 + m3) <= set(ids[:300])
    with pytest.raises(SystemExit):
        SL.sample_ids(ids[:100], set(ids))


@need_data
def test_sample_is_decision_only_and_fixed():
    dec = SL.decision()
    rows = SL.plan_rows()
    w, m = SL.sample_ids(dec["ids"], set(rows))
    assert (w, m) == SL.sample_ids(dec["ids"], set(rows))
    assert set(w + m) <= set(dec["ids"]) and len(set(w + m)) == 203
    assert {rows[i]["db_id"] for i in w + m} <= set(DEC_DBS) == set(dec["database_counts"])
    assert all(set(rows[i]) == set(SL.Q_KEYS) for i in w + m)  # plan_rows keeps question fields only


# ---- stage ------------------------------------------------------------------------------------------------------------
@pytest.fixture
def fake_rr(tmp_path):
    d = tmp_path / "rr"
    (d / "rerank_lora/rerank").mkdir(parents=True)
    for f in ("rerank_head.pt", "rerank_lora/rerank/adapter_model.safetensors"):
        (d / f).write_bytes(b"\0weights")
    (d / "rerank_lora/rerank/adapter_config.json").write_text("{}")
    (d / "settings.json").write_text(json.dumps({"result_preview": False}))
    return d


def _stage(out, rr):
    a = SimpleNamespace(out=out, plans=SL.PLANS, cards=SL.CARDS, rr=rr, seed=SL.SEED, n=SL.N_MEASURE, n_warm=SL.N_WARM)
    assert SL.stage(a) == 0
    return out


@need_data
def test_stage_strips_labels(tmp_path, fake_rr):
    out = _stage(tmp_path / "latency", fake_rr)
    txt = (out / "qs/plans.jsonl").read_text()
    rows = [json.loads(l) for l in txt.splitlines()]
    assert len(rows) == 203 and all(set(r) == ROW_KEYS and r["outcome"] == "missing" for r in rows)
    for k in ('"gold_sql"', '"plan"', '"compiled_sql"', '"ok"', '"covered"', '"in_slots"', '"value_labels"', '"db_values"',
              '"empty_result"', "SELECT "):
        assert k not in txt, k
    s = json.loads((out / "sample.json").read_text())
    assert [r["id"] for r in rows] == s["warmup"] + s["measured"] and s["split"] == "decision"
    assert set(s["warmup"] + s["measured"]) <= set(SL.decision()["ids"])
    assert sorted(p.stem for p in (out / "qs/cards").glob("*.json")) == DEC_DBS
    names = sorted(l.split("  ", 1)[1] for l in (out / "MANIFEST.sha256").read_text().splitlines())
    assert names == sorted(str(p.relative_to(out)) for p in out.rglob("*") if p.is_file() and p.name != "MANIFEST.sha256")
    assert "rr_valhard/rerank_lora/rerank/adapter_model.safetensors" in names
    _stage(tmp_path / "again", fake_rr)  # deterministic
    assert (tmp_path / "again/qs/plans.jsonl").read_text() == txt
    assert (tmp_path / "again/MANIFEST.sha256").read_text() == (out / "MANIFEST.sha256").read_text()
    for p in out.rglob("*"):
        assert not any(f in str(p.relative_to(out)).lower() for f in FORBIDDEN), p


def test_audit_refuses_labels():
    with pytest.raises(SystemExit):
        SL.audit_rows([{"id": "a", "db_id": "b", "question": "q", "gold_sql": "SELECT 1"}])
    SL.audit_rows([{"id": "a", "db_id": "b", "question": "q", "context": "", "outcome": "missing"}])


# ---- box script (DRY) -------------------------------------------------------------------------------------------------
@pytest.fixture
def box_layout(tmp_path, fake_rr):
    if not (have_data and have_dbs):
        pytest.skip("DECISION data / sqlite files not on this machine")
    d = tmp_path / "d"
    _stage(d / "latency", fake_rr)
    for db in DEC_DBS:
        (d / "sys1/dbs" / db).mkdir(parents=True)
        (d / "sys1/dbs" / db / f"{db}.sqlite").symlink_to(ROOT / f"data/sys1/dbs/{db}/{db}.sqlite")
    return tmp_path


def _box(tmp_path, extra=None):
    env = {**os.environ, "DRY": "1", "REMOTE_ROOT": str(tmp_path / "box"), "REMOTE_D": str(tmp_path / "d"),
           "REPO": str(ROOT), "PY": sys.executable, **(extra or {})}
    return subprocess.run(["bash", str(ROOT / "scripts/chains/latency_box.sh")], env=env, capture_output=True, text=True)


def test_box_dry(box_layout):
    t = box_layout
    r = _box(t, {"PRICE_H": "2.50"})
    out = r.stdout
    assert r.returncode == 0 and "LATENCY_DONE" in out and "LATENCY_FAIL" not in out, out + r.stderr
    assert "staged: 203 label-free DECISION rows on 8 DBs; R1 grammar v3" in out  # the system-python preflight ran
    cmds = [l for l in out.splitlines() if "DRY: " in l]
    assert len(cmds) == 1
    c = cmds[0]
    for flag in ("scripts/analysis/latency/sys1_latency.py run", "--run runs/r1r2/r1", f"--data {t}/d/latency",
                 f"--db-roots {t}/d/sys1/dbs", f"--out {t}/box/latency", "--price-h 2.50", "--sq-max-min 35", "--workers "):
        assert flag in c, flag
    assert "--n " not in c
    assert "--n 5" in [l for l in _box(t, {"N": "5"}).stdout.splitlines() if "DRY: " in l][0]
    for f in FORBIDDEN:
        assert f not in c.lower(), (f, c)


def test_box_refuses(box_layout):
    t = box_layout
    assert "bad PRICE_H" in _box(t, {"PRICE_H": "x"}).stdout
    (t / "d/reserved.json").write_text("{}")  # a RESERVED file on the box
    r = _box(t)
    assert "LATENCY_FAIL" in r.stdout and "LATENCY_DONE" not in r.stdout
    (t / "d/reserved.json").unlink()
    p = t / "d/latency/qs/plans.jsonl"
    rows = p.read_text().splitlines()
    rows[0] = json.dumps({**json.loads(rows[0]), "gold_sql": "SELECT 1"})
    p.write_text("\n".join(rows) + "\n")  # a label (and a stale manifest)
    r = _box(t)
    assert "LATENCY_FAIL" in r.stdout and "LATENCY_DONE" not in r.stdout


# ---- launcher (DRY): stage + leakcheck + push plan --------------------------------------------------------------------
@pytest.mark.skipif(not (ROOT / "runs/r1r2/r1/best_heads.pt").exists() or not have_dbs or
                    not (ROOT / "runs/phase1/valhard/valhard/rr/rerank_head.pt").exists(), reason="R1 / VALHARD not here")
def test_launcher_dry_push_plan():
    r = subprocess.run(["bash", str(ROOT / "scripts/chains/run_latency_box.sh"), "-"], env={**os.environ, "DRY": "1"},
                       capture_output=True, text=True, cwd=ROOT)
    out = r.stdout
    assert r.returncode == 0 and "leakcheck ok" in out, out + r.stderr
    push = {l.split("would push -> ")[1].split(":")[0]: l.split(": ", 1)[1].rsplit(" (", 1)[0].split()
            for l in out.splitlines() if "would push -> " in l}
    assert push["/root/d"] == ["data/latency"]
    assert push["/root/d/sys1/dbs"] == [f"data/sys1/dbs/{d}" for d in DEC_DBS]
    assert push["/workspace/luthor/runs/r1r2/r1"] == ["runs/r1r2/r1/best_lora", "runs/r1r2/r1/best_heads.pt",
                                                     "runs/r1r2/r1/args.json", "runs/r1r2/r1/vocab.json"]
    for paths in push.values():
        for p in paths:
            assert not any(f in p.lower() for f in FORBIDDEN), p
    assert "PRICE_H=2.50" in out and "latency_box.sh" in out and "LATENCY_DONE /workspace/latency" in out
    assert "FAIL_REGEX=LATENCY_FAIL CAP=6" in out
    rows = [json.loads(l) for l in open(ROOT / "data/latency/qs/plans.jsonl")]
    assert all(set(r) == ROW_KEYS for r in rows)  # what is pushed is label-free


def test_leakcheck_allows_data_latency_and_scripts_are_clean():
    t = (ROOT / "scripts/vast/leakcheck.sh").read_text()
    assert "|data/latency|" in t and "# data/latency:" in t
    for p in ("scripts/chains/latency_box.sh", "scripts/chains/run_latency_box.sh", "scripts/analysis/latency/sys1_latency.py"):
        s = (ROOT / p).read_text()
        for f in ("cands_bird_dev", "bird_dev_scored", "data/eval/sealed", "data/sys1/sealed", "leak_patterns",
                  "heldout_v2/reserved", "heldout_pool", "data/bench", "dev_20240627", "sf.py create <"):
            if f == "sf.py create <" and p.endswith("run_latency_box.sh"):
                continue  # the usage comment tells the operator how to create a box by hand; the script never does
            assert f not in s, (p, f)
    assert "sf.py create" not in "\n".join(l for l in (ROOT / "scripts/chains/run_latency_box.sh").read_text().splitlines()
                                          if not l.lstrip().startswith("#"))


def test_rerank_sq2_score_cli_unchanged():
    t = (ROOT / "scripts/rerank_sq2.py").read_text()
    assert "def load_validator" in t and "def validator_settings" in t
    s = t.split("def score(a):")[1].split("\ndef ")[0]
    assert "validator_settings(a.rr, a.joint_results, a.joint_orders)" in s and "load_validator(a.rr, a.base" in s
    assert s.index("load_cards") < s.index("load_validator(")  # same order as before: cards + rows, then the model


# ---- CPU smoke of the whole measurement (tiny random models, CUDA stubbed) ---------------------------------------------
def _tok_dir():
    import glob
    for p in glob.glob(str(Path.home() / ".cache/huggingface/hub/models--seeklhy--OmniSQL-7B/snapshots/*")) + \
            [str(ROOT / "runs/sq4b_v1/merged")]:
        if Path(p, "tokenizer.json").exists():
            return p
    pytest.skip("no local tokenizer")


@pytest.fixture
def restore_globals(monkeypatch):
    """run() sets the R1 run's vocabulary / candidate order process-wide (rerank_sq2.use_run_settings): put them back."""
    import copy
    pytest.importorskip("torch")
    for p in (ROOT / "scripts/devlook", ROOT / "scripts/repair", ROOT / "scripts", ROOT / "src"):
        monkeypatch.syspath_prepend(str(p))
    monkeypatch.setenv("LUTHOR_GRAMMAR", "v3")
    import luthor.sq2.targets as tg
    import luthor.sq2.train as tr
    import train_sq2
    dicts = [d for d in (getattr(tg, n, None) for n in ("VOCAB", "PAIRS", "MAIN_CHOICES", "CHOICES")) if isinstance(d, dict)]
    saved = [copy.deepcopy(d) for d in dicts]
    attrs = [(train_sq2, "VOCAB_ARGS"), (train_sq2, "CAND_ORDER"), (tr, "CAND_ORDER")]
    vals = [getattr(m, n) for m, n in attrs]
    yield
    for d, s in zip(dicts, saved):
        d.clear()
        d.update(s)
    for (m, n), v in zip(attrs, vals):
        setattr(m, n, v)


@pytest.mark.skipif(not (have_data and have_dbs and (ROOT / "runs/r1r2/r1/args.json").exists()), reason="data not here")
def test_tiny_run_end_to_end(tmp_path, fake_rr, monkeypatch, restore_globals):
    torch = pytest.importorskip("torch")
    from transformers import AutoTokenizer, Qwen2Config, Qwen2Model
    import rerank_sq2 as R
    from eval_sq2 import build_model, run_args
    from luthor.sq.model import Backbone
    from luthor.sq2.rerank import Reranker
    tok = AutoTokenizer.from_pretrained(_tok_dir())
    cfg = Qwen2Config(vocab_size=len(tok), hidden_size=64, intermediate_size=128, num_hidden_layers=2,
                      num_attention_heads=4, num_key_value_heads=2)

    def tiny_gen(run, base, sub):
        torch.manual_seed(0)
        return build_model(Backbone(Qwen2Model(cfg), tok), run_args(run)).eval()

    def tiny_rr(rr_dir, base, sub, lora_r):
        from peft import LoraConfig, get_peft_model
        torch.manual_seed(1)
        body = get_peft_model(Qwen2Model(cfg), LoraConfig(r=4, lora_alpha=8, lora_dropout=0.0, target_modules="all-linear"))
        return Reranker(Backbone(body, tok), lora_r=4)

    monkeypatch.setattr(R, "load_slot_model", tiny_gen)
    monkeypatch.setattr(R, "load_validator", tiny_rr)
    for name, fn in (("synchronize", lambda: None), ("reset_peak_memory_stats", lambda: None),
                     ("max_memory_allocated", lambda: 0), ("max_memory_reserved", lambda: 0),
                     ("memory_allocated", lambda: 0), ("get_device_name", lambda i=0: "cpu-stub"),
                     ("device_count", lambda: 0),
                     ("get_device_properties", lambda i=0: SimpleNamespace(total_memory=0))):
        monkeypatch.setattr(torch.cuda, name, fn)
    # a small staged dir: 1 warm-up + 2 measured questions on the two smallest DECISION DBs
    full = _stage(tmp_path / "full", fake_rr)
    rows = [json.loads(l) for l in open(full / "qs/plans.jsonl")]
    small = [r for r in rows if r["db_id"] in ("cs_semester", "movie")][:3]
    d = tmp_path / "small"
    (d / "qs/cards").mkdir(parents=True)
    (d / "qs/plans.jsonl").write_text("".join(json.dumps(r) + "\n" for r in small))
    for db in {r["db_id"] for r in small}:
        shutil.copy2(full / f"qs/cards/{db}.json", d / f"qs/cards/{db}.json")
    shutil.copytree(full / "rr_valhard", d / "rr_valhard")
    (d / "sample.json").write_text(json.dumps({"seed": "t", "warmup": [small[0]["id"]],
                                               "measured": [r["id"] for r in small[1:]]}))
    dbs = tmp_path / "dbs"
    for db in {r["db_id"] for r in small}:
        (dbs / db).mkdir(parents=True)
        (dbs / db / f"{db}.sqlite").symlink_to(ROOT / f"data/sys1/dbs/{db}/{db}.sqlite")
    out = tmp_path / "out"
    rc = SL.main(["run", "--run", str(ROOT / "runs/r1r2/r1"), "--data", str(d), "--db-roots", str(dbs), "--out", str(out),
                  "--price-h", "2.5", "--workers", "2"])
    assert rc == 0
    res = json.loads((out / "latency.json").read_text())
    sq = res["single_query"]
    assert sq["n_questions"] == 2
    for mode in ("greedy", "k32", "k64r"):
        assert sq[mode]["end_to_end_s"]["n"] == 2 and sq[mode]["end_to_end_s"]["p50"] > 0
    assert set(sq["k64r"]["stages_s"]) == {"lookup", "generate", "compile", "execute", "repair", "repair_execute",
                                           "validate", "pick"}
    for mode in ("k32", "k64r"):
        tp = res["throughput"][mode]
        assert tp["n_questions"] == 2 and tp["questions_per_s"] > 0 and tp["usd_per_1000"] > 0
    txt = (out / "latency.txt").read_text()
    assert "SINGLE QUERY" in txt and "THROUGHPUT" in txt
    def keys(o):
        if isinstance(o, dict):
            for k, v in o.items():
                yield k
                yield from keys(v)
        elif isinstance(o, list):
            for v in o:
                yield from keys(v)
    for k in keys(res):  # latency only: no accuracy / EX / label field anywhere
        assert k.lower() not in ("ok", "ex", "em", "rr") and not any(b in k.lower() for b in ("accuracy", "gold", "correct", "oracle")), k
    assert '"ok"' not in (out / "rows.jsonl").read_text() and "gold" not in txt
    assert len((out / "rows.jsonl").read_text().splitlines()) == 2
