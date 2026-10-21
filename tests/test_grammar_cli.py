"""--grammar plumbing (luthor.grammar_cli) and train_sq2 --drop-dbs. CPU, no model."""
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from luthor import grammar_cli as gc  # noqa: E402
from luthor.sq.plan import GRAMMAR  # noqa: E402


def _run(tmp_path, grammar=None):
    d = tmp_path / f"run_{grammar}"
    d.mkdir()
    (d / "args.json").write_text(json.dumps({"lora_r": "32", **({"grammar": grammar} if grammar else {})}))
    return d


def test_run_grammar_missing_is_v3(tmp_path):
    assert gc.run_grammar(_run(tmp_path)) == "v3"
    assert gc.run_grammar(_run(tmp_path, "v4")) == "v4"
    assert gc.run_grammar(tmp_path / "nowhere") == "v3"


@pytest.fixture
def env_restored():
    """preset() writes os.environ["LUTHOR_GRAMMAR"] itself (and monkeypatch's own env undo would then restore a value
    preset wrote): put the exact original back, since later tests spawn workers that read it."""
    before = os.environ.get("LUTHOR_GRAMMAR")
    yield
    if before is None:
        os.environ.pop("LUTHOR_GRAMMAR", None)
    else:
        os.environ["LUTHOR_GRAMMAR"] = before


def test_preset_order(tmp_path, monkeypatch, env_restored):
    os.environ.pop("LUTHOR_GRAMMAR", None)
    monkeypatch.delitem(sys.modules, "luthor.sq.plan")  # preset refuses to contradict an imported grammar
    r4 = _run(tmp_path, "v4")
    assert gc.preset(["--run", str(r4)]) == "v4" and os.environ["LUTHOR_GRAMMAR"] == "v4"
    os.environ.pop("LUTHOR_GRAMMAR")
    assert gc.preset(["--run", str(r4), "--grammar", "v3"]) == "v3"  # the flag wins
    os.environ["LUTHOR_GRAMMAR"] = "v3"
    assert gc.preset(["--run", str(r4)]) == "v3"  # then the environment (check_run will refuse the mismatch)
    os.environ.pop("LUTHOR_GRAMMAR")
    assert gc.preset([]) == "v3" and gc.preset(["--grammar=v4"]) == "v4"
    with pytest.raises(SystemExit):
        gc.preset(["--grammar", "v5"])


def test_preset_refuses_after_import(env_restored):
    other = "v4" if GRAMMAR == "v3" else "v3"
    os.environ.pop("LUTHOR_GRAMMAR", None)
    with pytest.raises(SystemExit, match="already imported"):
        gc.preset(["--grammar", other])


def test_check_run_and_resolve(tmp_path):
    other = "v4" if GRAMMAR == "v3" else "v3"
    same, diff = _run(tmp_path, GRAMMAR), _run(tmp_path, other)
    assert gc.check_run(same) == GRAMMAR and gc.check_run(same, GRAMMAR) == GRAMMAR
    with pytest.raises(SystemExit, match="trained with grammar"):
        gc.check_run(diff)
    with pytest.raises(SystemExit, match="--grammar"):
        gc.check_run(same, other)
    assert gc.resolve(None) == GRAMMAR
    with pytest.raises(SystemExit):
        gc.resolve(other)


@pytest.mark.parametrize("g", ["v3", "v4"])
def test_flag_reaches_luthor_and_workers(g):
    """The script pattern: preset() under __main__ before the first luthor.sq import; a spawned worker inherits it."""
    code = ("import sys; sys.path.insert(0, 'src')\n"
            "from luthor.grammar_cli import preset; preset()\n"
            "import multiprocessing as mp\n"
            "from luthor.sq.plan import GRAMMAR\n"
            "def w(_):\n    from luthor.sq.plan import GRAMMAR as G; return G\n"
            "if __name__ == '__main__':\n"
            "    with mp.get_context('spawn').Pool(1) as p: print(GRAMMAR, p.map(w, [0])[0])\n")
    env = {k: v for k, v in os.environ.items() if k != "LUTHOR_GRAMMAR"}
    f = ROOT / ".pytest_cache" / "grammar_flag_probe.py"
    f.parent.mkdir(exist_ok=True)
    f.write_text(code)
    r = subprocess.run([sys.executable, str(f), "--grammar", g], cwd=ROOT, env=env, capture_output=True, text=True,
                       timeout=120)
    assert r.returncode == 0, r.stderr
    assert r.stdout.split() == [g, g]


def test_train_sq2_records_grammar_and_resume_keys():
    import train_sq2
    assert "grammar" in train_sq2.RESUME_KEYS and "drop_dbs" in train_sq2.RESUME_KEYS
    a = train_sq2.parser().parse_args(["--plans", "x", "--out", "o", "--dev-dbs", "d.json"])
    assert a.grammar is None and a.drop_dbs is None  # main() resolves grammar to the process's before args.json


def test_drop_dbs(tmp_path):
    import train_sq2
    assert train_sq2.load_drop_dbs(None) == set() and train_sq2.load_drop_dbs("") == set()
    assert train_sq2.load_drop_dbs("formula_1, soccer_1") == {"formula_1", "soccer_1"}
    f = tmp_path / "l.json"
    f.write_text(json.dumps(["a", "b"]))
    assert train_sq2.load_drop_dbs(str(f)) == {"a", "b"}
    assert train_sq2.load_drop_dbs(str(ROOT / "data/sys1/contaminated_dbs.json")) == {"formula_1", "soccer_1"} \
        or not (ROOT / "data/sys1/contaminated_dbs.json").exists()
    E = lambda i, db: SimpleNamespace(id=i, db_id=db)  # noqa: E731
    rows = [E("s1", "formula_1"), E("s1", "formula_1"), E("s2", "soccer_1"), E("t1", "formula_1"), E("b1", "trains")]
    gold = {"s1": {"source": "spider"}, "s2": {"source": "spider"}, "t1": {"source": "teacher", "teacher": "omni"},
            "b1": {"source": "bird"}}
    kept, st = train_sq2.drop_rows(rows, {"formula_1", "soccer_1"}, gold)
    assert [e.id for e in kept] == ["b1"]
    assert st["removed"] == 4 and st["by_db"] == {"formula_1": 3, "soccer_1": 1}
    assert st["by_source"] == {"spider": 3, "teacher/omni": 1}
