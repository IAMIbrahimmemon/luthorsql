"""BIRD train without the disk: the 8.9 GB archive (a zip holding a zipped database folder)
is unzipped as it downloads, and each database is converted to plans (prepare_plans.process_db)
as soon as its files are complete, then its SQLite file is deleted. Keeps only plans + cards.

  python scripts/fetch_bird_train.py --out data/plans [--keep-dbs N]
"""
import argparse
import collections
import json
import shutil
import sys
import urllib.request
from pathlib import Path

from remotezip import RemoteZip
from stream_unzip import stream_unzip

sys.path.insert(0, str(Path(__file__).resolve().parent))
from prepare_plans import process_db  # noqa: E402

URL = "https://bird-bench.oss-cn-beijing.aliyuncs.com/train.zip"
RAW = Path("data/raw/bird_train")


def http_chunks(url, size=1 << 20, retries=50):
    """One continuous byte stream; on a dropped connection, resume with a Range request."""
    import time

    pos, total, tries = 0, None, 0
    while True:
        req = urllib.request.Request(url, headers={"Range": f"bytes={pos}-"} if pos else {})
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                if total is None:
                    total = int(r.headers["Content-Length"])
                while b := r.read(size):
                    pos += len(b)
                    yield b
            if pos >= total:
                return
        except Exception as e:
            print(f"connection dropped at {pos}: {type(e).__name__}; resuming", flush=True)
        tries += 1
        if tries > retries:
            raise RuntimeError("too many retries")
        time.sleep(3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("data/plans"))
    ap.add_argument("--keep-dbs", type=int, default=0, help="keep the first N SQLite files (e.g. for dev evaluation)")
    a = ap.parse_args()
    RAW.mkdir(parents=True, exist_ok=True)
    with RemoteZip(URL) as z:  # the small json files first, by range request
        for n in ("train/train.json", "train/train_tables.json"):
            (RAW / Path(n).name).write_bytes(z.read(n))
    qs = json.loads((RAW / "train.json").read_text())
    tables = {e["db_id"]: e for e in json.loads((RAW / "train_tables.json").read_text())}
    by_db = collections.defaultdict(list)
    for i, q in enumerate(qs):
        by_db[q["db_id"]].append((i, {"db_id": q["db_id"], "question": q["question"], "sql": q["SQL"],
                                      "context": q.get("evidence", ""), "question_id": i}))
    outdir = a.out / "bird_train"
    (outdir / "cards").mkdir(parents=True, exist_ok=True)
    done = {p.stem for p in (outdir / "cards").glob("*.json")}  # converted on an earlier run
    fh = open(outdir / "plans.jsonl", "a")
    stats = collections.Counter(json.loads((outdir / "stats_partial.json").read_text())) \
        if (outdir / "stats_partial.json").exists() else collections.Counter()
    kept = 0

    def finish(db):
        nonlocal kept
        if db in done or db not in tables:
            return
        rows, s, card = process_db(("bird", RAW, "train", db, tables[db], by_db[db]))
        for r in rows:
            fh.write(json.dumps(r, default=str) + "\n")
        fh.flush()
        stats.update(s)
        if card is not None:
            card.save(outdir / "cards" / f"{db}.json")
        done.add(db)
        (outdir / "stats_partial.json").write_text(json.dumps(stats))
        ok = s["plan_ok"] + s["plan_empty"]
        print(f"{db}: {len(by_db[db])} questions -> {ok} plans", flush=True)
        if kept < a.keep_dbs:
            kept += 1
        else:
            shutil.rmtree(RAW / "train_databases" / db, ignore_errors=True)

    current = None
    for name, _, chunks in stream_unzip(http_chunks(URL)):
        name = name.decode()
        if not name.endswith("train_databases.zip") or "__MACOSX" in name:
            for _ in chunks:
                pass
            continue
        for inner, _, ic in stream_unzip(chunks):
            rel = inner.decode().split("train_databases/", 1)[-1]
            parts = rel.split("/")
            if not rel or "__MACOSX" in rel or parts[-1].startswith("._") or rel.endswith("/"):
                for _ in ic:
                    pass
                continue
            db = parts[0]
            if current and db != current:
                finish(current)
            current = db
            if db in done:
                for _ in ic:
                    pass
                continue
            p = RAW / "train_databases" / rel
            p.parent.mkdir(parents=True, exist_ok=True)
            with open(p, "wb") as f:
                for c in ic:
                    f.write(c)
    if current:
        finish(current)
    fh.close()
    total = sum(len(v) for v in by_db.values())
    ok = stats["plan_ok"] + stats["plan_empty"]
    summary = {"questions": total, "plans_verified": ok, "coverage": round(ok / max(1, total), 4),
               "twins": stats["twin_value"] + stats["twin_order"], "stats": dict(stats.most_common())}
    (outdir / "stats.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps({k: v for k, v in summary.items() if k != "stats"}))


if __name__ == "__main__":
    main()
