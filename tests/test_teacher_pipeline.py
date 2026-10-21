"""Teacher-data pipeline (src/luthor/teacher, scripts/teacher/teacher.py) without any network call: leak gate,
throttle / back-off, token ledger and caps, guards, execution signatures and votes, plan rows, and the
label -> verify -> export flow end to end with a fake teacher on the public BIRD-train 'trains' database.
   pytest tests/test_teacher_pipeline.py"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "scripts/teacher"))

from luthor.teacher import guards, verify  # noqa: E402
from luthor.teacher.leak import LeakGate  # noqa: E402
from luthor.teacher.ledger import CapReached, Ledger  # noqa: E402
from luthor.teacher import providers as P  # noqa: E402

TRAINS_DB = REPO / "data/sys1/rrdbs/trains/trains.sqlite"
CARDS = REPO / "data/sys1/bird_train/cards"
needs_data = pytest.mark.skipif(not TRAINS_DB.exists() or not (CARDS / "trains.json").exists(),
                                reason="needs data/sys1 (BIRD-train trains db and card)")


@pytest.fixture()
def patterns(tmp_path, monkeypatch):
    p = tmp_path / "leak_patterns"
    p.write_text("WORK=\\bzzprivatecorp\\b\nCODE_WORK=\\bzzcode\\b\nPERSON=\\bzzperson\\b\n")
    monkeypatch.setenv("LEAK_PATTERNS", str(p))
    return p


# ---- leak gate ----------------------------------------------------------------------------------------------
def test_leak_gate(patterns):
    g = LeakGate()
    assert g.hit("How many trains run east?") is None
    assert g.hit("revenue for ZZPrivateCorp users") == "WORK"
    assert g.hit("zzperson and zzcode are public-dataset words") is None  # code-only classes, as leakcheck.sh
    assert g.hit("key sk-" + "a" * 40) == "SECRET"
    assert g.hit("AKIAIOSFODNN7EXAMPLE") is None  # AWS documentation placeholder


def test_leak_gate_requires_private_patterns(tmp_path, monkeypatch):
    monkeypatch.setenv("LEAK_PATTERNS", str(tmp_path / "missing"))
    with pytest.raises(RuntimeError):
        LeakGate()
    assert LeakGate(require_private=False).hit("hf_" + "x" * 34) == "SECRET"


# ---- throttle / providers / ledger --------------------------------------------------------------------------
class Clock:
    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        return self.t

    def sleep(self, s):
        self.t += s


def test_throttle_backoff_and_give_up():
    c = Clock()
    th = P.Throttle(max_wait=1000, sleep=c.sleep, clock=c, log=lambda *a, **k: None)
    th.limited(None, "429")
    assert th.until == c.t + P.BACKOFF[0]
    th.limited(None, "429")  # already paused: no extra step
    assert th.step == 1
    th.wait()
    assert c.t >= th.until
    th.limited(30, "429 retry-after")
    assert th.until == c.t + 30
    c.t += 2000
    with pytest.raises(P.GaveUp):
        th.limited(None, "still limited")


class Fake(P.Provider):
    name = "fake"

    def __init__(self, replies, **kw):
        super().__init__("m", **kw)
        self.replies, self.calls = list(replies), 0

    def _call(self, messages, temperature, max_tokens, seed, think):
        self.calls += 1
        r = self.replies.pop(0)
        if isinstance(r, Exception):
            raise r
        return P.Reply(r, 10, 5, 0.01)


def test_provider_waits_out_rate_limit_and_records_tokens(patterns, tmp_path):
    c = Clock()
    th = P.Throttle(sleep=c.sleep, clock=c, log=lambda *a, **k: None)
    led = Ledger(tmp_path / "usage.jsonl")
    pr = Fake([P.RateLimited("HTTP 429 session limit"), RuntimeError("you have reached your usage limit"), "SELECT 1"],
              gate=LeakGate(), ledger=led, throttle=th)
    assert pr.chat([{"role": "user", "content": "q"}]).text == "SELECT 1"
    assert pr.calls == 3 and c.t > 1000 + P.BACKOFF[0]
    assert led.used("fake", "m") == (15, 15)
    assert Ledger(tmp_path / "usage.jsonl").used("fake", "m") == (15, 15)  # persisted


def test_provider_leak_refused_never_calls(patterns):
    pr = Fake(["x"], gate=LeakGate())
    with pytest.raises(P.LeakRefused):
        pr.chat([{"role": "user", "content": "schema for zzprivatecorp"}])
    assert pr.calls == 0


def test_ledger_caps_and_weeks(tmp_path):
    led = Ledger(tmp_path / "u.jsonl", day_cap=100, week_cap=150)
    t = 1_760_000_000.0  # a fixed instant
    led.record("ollama", "m", 60, 30, ts=t)
    assert led.used("ollama", "m", ts=t) == (90, 90)
    led.record("ollama", "m", 5, 5, ts=t)
    s = led.summary()["ollama/m"]
    assert sum(s["days"].values()) == 100 and sum(s["weeks"].values()) == 100
    led2 = Ledger(tmp_path / "u2.jsonl", day_cap=10)
    led2.record("ollama", "m", 10, 0)
    with pytest.raises(CapReached):
        led2.check("ollama", "m")
    led2.check("ollama", "other")


def test_openai_spec_needs_key(patterns, monkeypatch):
    monkeypatch.delenv("ZZ_TEST_KEY", raising=False)
    with pytest.raises(RuntimeError):
        P.make("openai:grok-x@https://api.example.com/v1#ZZ_TEST_KEY", LeakGate())
    monkeypatch.setenv("ZZ_TEST_KEY", "secret")
    pr = P.make("openai:grok-x@https://api.example.com/v1#ZZ_TEST_KEY", LeakGate())
    assert pr.url == "https://api.example.com/v1/chat/completions" and pr.model == "grok-x"
    assert isinstance(P.make("deepseek-v4.1-flash:cloud", LeakGate()), P.Ollama)


# ---- guards -------------------------------------------------------------------------------------------------
def test_guards():
    bd = guards.bird_dev_dbs()
    assert "financial" in bd
    with pytest.raises(guards.Refused):
        guards.check_prompt_row({"id": "x", "db_id": "financial", "source": "bird"}, bd)
    with pytest.raises(guards.Refused):
        guards.check_prompt_row({"id": "x", "db_id": "trains", "source": "spider_dk"}, bd)
    guards.check_prompt_row({"id": "x", "db_id": "trains", "source": "bird_train"}, bd)
    for p in ("data/sys1/sealed/x.jsonl", "data/eval/sealed_bird_dev.jsonl", "runs/r1r2/r2/rr/bird_dev_scored.jsonl"):
        with pytest.raises(guards.Refused):
            guards.check_input_path(p)
    guards.check_input_path("data/sys1/bird_train/plans.jsonl")
    held = guards.held_out_dbs()
    assert {"movie", "hockey", "financial"} <= held and "trains" not in held


# ---- verification -------------------------------------------------------------------------------------------
def test_signature_and_vote():
    assert verify.result_sig([(1,), (2.0,)]) == verify.result_sig([(2,), (1.0,)])
    assert verify.result_sig([(1,)]) != verify.result_sig([("1",)])
    assert verify.vote(["a", "a", "b", ""]) == ("a", 0.5, 2)
    assert verify.vote(["", ""]) == ("", 0.0, 0)


@needs_data
def test_execute_and_plan_row():
    paths = {"trains": str(TRAINS_DB)}
    res = verify.execute([(1, "trains", "SELECT COUNT(id) FROM trains WHERE direction = 'east'"),
                          (2, "trains", "SELECT count(*) FROM trains WHERE direction='east'"),
                          (3, "trains", "SELECT nope FROM trains"), (4, "nodb", "SELECT 1")], paths)
    assert res[1][0] == "ok" and res[1][1] == res[2][1]
    assert res[3][0].startswith("error") and res[4][0] == "no-db"
    verify.set_grammar("v3", REPO / "data/sys1/templates.json")
    task = (str(TRAINS_DB), str(CARDS / "trains.json"),
            [("q1", "How many trains are there that run in the east direction?", "east is a direction",
              ["SELECT nope", "SELECT COUNT(id) FROM trains WHERE direction = 'east'"])], "v3", None)
    [(key, row, why)] = verify.plan_rows_db(task)
    assert key == "q1" and row is not None and row["outcome"] == "plan" and row["sql_index"] == 1
    assert why[0] != "ok" and why[1] == "ok"


# ---- end to end with a fake teacher -------------------------------------------------------------------------
@needs_data
def test_label_verify_export_end_to_end(patterns, tmp_path, monkeypatch):
    import teacher as T
    rows = [{"id": "t1", "db_id": "trains", "question": "How many trains are there that run in the east direction?",
             "context": "east is a direction", "gold_sql": "SELECT COUNT(id) FROM trains WHERE direction = 'east'",
             "source": "bird_train"},
            {"id": "t2", "db_id": "trains", "question": "How many cars are there on train no.1?",
             "context": "train no.1 refers to train_id = 1", "source": "bird_train_nogold"},
            {"id": "t3", "db_id": "financial", "question": "q", "context": "", "source": "bird_train"}]  # BIRD dev db
    inp = tmp_path / "in.jsonl"
    inp.write_text("".join(json.dumps(r) + "\n" for r in rows))
    answers = {"How many trains": "```sql\nSELECT COUNT(*) FROM trains WHERE direction = 'east'\n```",
               "How many cars": "```sql\nSELECT COUNT(id) FROM cars WHERE train_id = 1\n```"}
    prompted = []

    class Teacher(P.Provider):
        name = "fake"

        def _call(self, messages, temperature, max_tokens, seed, think):
            text = messages[-1]["content"]
            prompted.append(text)
            for k, v in answers.items():
                if k in text:
                    return P.Reply(v, 100, 20, 0.01)
            raise AssertionError("unexpected prompt")

    monkeypatch.setattr(T, "make", lambda spec, gate, **kw: Teacher("fake-model", gate,
                                                                      **{k: v for k, v in kw.items() if k != "ollama_url"}))
    run = tmp_path / "runs/r1"
    common = ["--cards", str(CARDS), "--db-roots", str(REPO / "data/sys1/rrdbs"), "--ledger", str(tmp_path / "u.jsonl"),
              "--workers", "1"]
    T.main(["label", "--input", str(inp), "--run", str(run), "--n", "2", "--temp", "0.7", *common])
    assert len(prompted) == 4 and not any("financial" in p.split("Question:")[-1] for p in prompted)
    assert all("SELECT COUNT(id) FROM trains" not in p for p in prompted)  # gold never in a prompt
    T.main(["label", "--input", str(inp), "--run", str(run), "--n", "2", "--temp", "0.7", *common])  # resume: no calls
    assert len(prompted) == 4
    T.main(["verify", "--run", str(run), *common])
    ver = {r["id"]: r for r in T.read_jsonl(run / "verified.jsonl")}
    assert ver["t1"]["verification"] == "gold_match" and ver["t1"]["plan_row"] is not None
    assert ver["t2"]["verification"] == "vote" and ver["t2"]["agreement"] == 1.0
    if guards.contamination_checker() is None:  # module not there yet: export refuses unless explicitly allowed
        with pytest.raises(SystemExit):
            T.main(["export", "--runs", str(run), "--out", str(tmp_path / "exp0"), *common])
    T.main(["export", "--runs", str(run), "--out", str(tmp_path / "exp"), "--allow-unchecked-contamination", *common])
    man = json.loads((tmp_path / "exp/manifest.json").read_text())
    out = T.read_jsonl(tmp_path / "exp" / man["shards"][0]["file"])
    assert man["rows"] == len(out) >= 1 and (tmp_path / "exp/cards/trains.json").exists()
    r = out[0]
    for k in ("plan", "compiled_sql", "value_labels", "db_values", "outcome"):
        assert k in r
    for k in ("teacher_model", "prompt_version", "temperature", "sample_idx", "source_dataset", "generated_at",
              "verification", "result_sig", "provider"):
        assert k in r["teacher"]
    assert r["teacher"]["teacher_model"] == T.DEFAULT_MODEL and r["teacher"]["provider"] == "fake"


@needs_data
def test_export_refuses_eval_runs_and_held_out(patterns, tmp_path):
    import teacher as T
    run = tmp_path / "r"
    run.mkdir()
    (run / "run.json").write_text(json.dumps({"job": "label", "model": "m", "eval_only": True}))
    with pytest.raises(SystemExit):
        T.main(["export", "--runs", str(run), "--out", str(tmp_path / "e"), "--allow-unchecked-contamination"])
    bk = tmp_path / "bakeoff/x"
    bk.mkdir(parents=True)
    (bk / "run.json").write_text(json.dumps({"job": "label", "model": "m"}))
    with pytest.raises(SystemExit):
        T.main(["export", "--runs", str(bk), "--out", str(tmp_path / "e"), "--allow-unchecked-contamination"])
    held = tmp_path / "h"
    held.mkdir()
    (held / "run.json").write_text(json.dumps({"job": "label", "model": "m", "prompt_version": "p", "temperature": 0}))
    (held / "verified.jsonl").write_text(json.dumps({"id": "x", "db_id": "movie", "plan_row": {"outcome": "plan"},
                                                     "verification": "gold_match"}) + "\n")
    T.main(["export", "--runs", str(held), "--out", str(tmp_path / "e2"), "--allow-unchecked-contamination"])
    man = json.loads((tmp_path / "e2/manifest.json").read_text())
    assert man["rows"] == 0 and man["counts"]["refused_held_out_db"] == 1


def test_glm_refused(monkeypatch):
    import teacher as T
    monkeypatch.delenv("LUTHOR_ALLOW_GLM", raising=False)
    with pytest.raises(SystemExit):
        T.check_model("glm-5.3:cloud")
    T.check_model("deepseek-v4.1-flash:cloud")


@needs_data
def test_write_job_emits_label_inputs(patterns, tmp_path, monkeypatch):
    """write on a plan row (schema / sample rows filled in) -> verify -> questions.jsonl with the source SQL as gold."""
    import teacher as T
    src = {"id": "m1", "db_id": "trains", "compiled_sql": "SELECT COUNT(\"id\") FROM \"trains\" WHERE \"direction\" = 'west'",
           "plan": json.loads(next(l for l in open(REPO / "data/sys1/bird_train/plans.jsonl")
                                   if '"bird_train_train_690"' in l))["plan"], "seed_id": "bird_train_train_690"}
    inp = tmp_path / "w.jsonl"
    inp.write_text(json.dumps(src) + "\n")
    seen = []

    class Writer(P.Provider):
        name = "fake"

        def _call(self, messages, temperature, max_tokens, seed, think):
            seen.append(messages[-1]["content"])
            return P.Reply('{"question": "How many trains run west?", "evidence": "west is a direction"}', 50, 10, 0.01)

    monkeypatch.setattr(T, "make", lambda spec, gate, **kw: Writer("w", gate,
                                                                     **{k: v for k, v in kw.items() if k != "ollama_url"}))
    run = tmp_path / "wr"
    common = ["--cards", str(CARDS), "--db-roots", str(REPO / "data/sys1/rrdbs"), "--ledger", str(tmp_path / "u.jsonl"),
              "--workers", "1"]
    T.main(["write", "--input", str(inp), "--run", str(run), "--n", "2", *common])
    assert len(seen) == 2 and "CREATE TABLE" in seen[0] and "Result:" in seen[0]
    T.main(["verify", "--run", str(run), *common])
    qs = T.read_jsonl(run / "questions.jsonl")
    assert len(qs) == 1  # the duplicate (question, evidence) pair is dropped
    q = qs[0]
    assert q["gold_sql"] == src["compiled_sql"] and q["source"].startswith("teacher_write")
    assert q["writer"]["prompt_version"] == T.WRITE_PROMPT


def test_near_dev_schema():
    assert guards.near_dev_schema("financial", []) == "financial"
    assert guards.near_dev_schema("trains", ["trains", "cars"]) is None
    try:
        import luthor.contamination  # noqa: F401
    except Exception:
        pytest.skip("luthor.contamination not available")
    from luthor.contamination import dev_table_sets
    f1 = sorted(dev_table_sets()["formula_1"])
    assert guards.near_dev_schema("formula_1_spider_copy", f1) == "formula_1"
