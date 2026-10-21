"""train_sq2.main end to end on CPU (tiny Qwen2 + LoRA saved to disk, CUDA calls stubbed): a run stopped by
--job-hours and continued by --auto-resume ends with exactly the weights of the same run done in one go; --seed and
--extra-train reach the run (args.json, train_mix.json). Single process."""
import json
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import train_sq2 as ts  # noqa: E402
from test_sq2_r1_levers import _card, _rows, _tok_dir  # noqa: E402


@pytest.fixture
def cpu(monkeypatch):
    monkeypatch.setattr(torch.nn.Module, "cuda", lambda self, *a, **k: self)
    for f in ("max_memory_allocated", "memory_allocated", "memory_reserved"):
        monkeypatch.setattr(torch.cuda, f, lambda *a, **k: 0)
    for v in ("WORLD_SIZE", "RANK", "LOCAL_RANK"):
        monkeypatch.delenv(v, raising=False)


def _data(tmp: Path):
    from transformers import AutoTokenizer, Qwen2Config, Qwen2Model
    tok = AutoTokenizer.from_pretrained(_tok_dir())
    base = tmp / "base"
    torch.manual_seed(0)
    Qwen2Model(Qwen2Config(vocab_size=len(tok), hidden_size=64, intermediate_size=128, num_hidden_layers=2,
                           num_attention_heads=4, num_key_value_heads=2)).save_pretrained(base)
    tok.save_pretrained(base)
    d = tmp / "plans"
    (d / "cards").mkdir(parents=True)
    rows = []
    for db in ("a", "b", "c", "dev"):
        card = _card()
        card.db_id = db
        card.save(d / "cards" / f"{db}.json")
        rows += [{**r, "id": f"{db}{r['id']}", "db_id": db, "source": "bird", "compiled_sql": "SELECT 1"} for r in _rows()]
    (d / "plans.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    (tmp / "dev.json").write_text(json.dumps(["dev"]))
    teacher = tmp / "teacher.jsonl"
    teacher.write_text("".join(json.dumps({**r, "id": f"t{r['id']}", "db_id": "a", "teacher": "big"}) + "\n"
                               for r in _rows()))
    return base, d, teacher


def _run(monkeypatch, base, d, out, *extra):
    argv = ["train_sq2.py", "--plans", str(d), "--dev-dbs", str(d.parent / "dev.json"), "--out", str(out),
            "--base", str(base), "--subfolder", "", "--lora-r", "4", "--dbs-per-step", "1", "--q-per-db", "2",
            "--max-steps", "8", "--eval-every", "4", "--budget-chars", "100000", "--schema-aug", "0.3", *extra]
    monkeypatch.setattr(sys, "argv", argv)
    ts.main()


def _heads(out: Path):
    return torch.load(out / "last_heads.pt")


def test_job_stop_and_auto_resume_match_one_run(tmp_path, cpu, monkeypatch, capsys):
    base, d, teacher = _data(tmp_path)
    flags = ["--seed", "1", "--extra-train", f"{teacher}:2", "--auto-resume"]
    _run(monkeypatch, base, d, tmp_path / "one", *flags)
    _run(monkeypatch, base, d, tmp_path / "two", *flags, "--job-hours", "1e-9")  # stops after its first step
    stop = json.loads((tmp_path / "two" / "job_stopped").read_text())
    assert stop == {"step": 1, "total": 8} and not (tmp_path / "two" / "oom.json").exists()
    _run(monkeypatch, base, d, tmp_path / "two", *flags, "--save-every", "3")
    assert not (tmp_path / "two" / "job_stopped").exists()
    assert "resumed" in capsys.readouterr().out
    one, two = _heads(tmp_path / "one"), _heads(tmp_path / "two")
    for k in one:
        assert torch.equal(one[k], two[k]), k
    from safetensors.torch import load_file
    la = load_file(str(tmp_path / "one" / "last_lora" / "adapter_model.safetensors"))
    lb = load_file(str(tmp_path / "two" / "last_lora" / "adapter_model.safetensors"))
    assert all(torch.equal(la[k], lb[k]) for k in la)
    args = json.loads((tmp_path / "one" / "args.json").read_text())
    assert args["seed"] == "1"
    mix = json.loads((tmp_path / "one" / "train_mix.json").read_text())
    assert mix["extra"][0]["train_rows"] == 6 and mix["train_by_source"]["?/big"] == 6
    # another seed: another run
    _run(monkeypatch, base, d, tmp_path / "s2", "--seed", "2", "--extra-train", f"{teacher}:2")
    assert any(not torch.equal(one[k], v) for k, v in _heads(tmp_path / "s2").items())
    with pytest.raises(SystemExit, match="seed"):  # resuming with other settings is refused
        _run(monkeypatch, base, d, tmp_path / "one", "--seed", "3", "--extra-train", f"{teacher}:2", "--auto-resume")
