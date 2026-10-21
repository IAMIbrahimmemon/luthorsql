"""VAL3 box helper: the large BIRD-train SQLite files (not in data/sys1/rrdbs, ~21 GB, too big to push from the Mac),
taken from the PUBLIC BIRD train archive (the URL scripts/fetch_bird_train.py uses). Standard library only, so it runs
with the system python3 before (and while) the venv is set up.

  python3 scripts/val3/fetch_big_dbs.py --want /root/d/val3/big_dbs.json --out /root/d/val3/bigdbs --work /workspace/bird_dl

1 download train.zip (8.9 GB) with N parallel HTTP range requests (resumable per part, retried);
2 decompress its member train/train_databases.zip to disk, delete train.zip;
3 extract only train_databases/<db>/<db>.sqlite for each db in --want ({db: bytes}), check each size against --want,
  delete the inner zip. Writes <out>/<db>/<db>.sqlite and <out>/FETCH_OK (or FETCH_FAIL <reason>; exit 1).
Needs ~45 GB free under --work at the peak. Progress lines every ~60 s (the watcher's stall check sees them).
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import threading
import time
import urllib.request
import zipfile
from pathlib import Path

URL = "https://bird-bench.oss-cn-beijing.aliyuncs.com/train.zip"
INNER = "train/train_databases.zip"


def log(msg: str) -> None:
    print(f"{time.strftime('%m-%d %H:%M:%S', time.gmtime())} [fetch_big] {msg}", flush=True)


def size_of(url: str) -> int:
    with urllib.request.urlopen(urllib.request.Request(url, method="HEAD"), timeout=60) as r:
        return int(r.headers["Content-Length"])


def parts(total: int, n: int) -> list[tuple[int, int]]:
    """n contiguous [start, end] byte ranges covering 0..total-1."""
    step = -(-total // n)
    return [(s, min(s + step, total) - 1) for s in range(0, total, step)]


def fetch_part(url: str, path: Path, start: int, end: int, done: list, k: int, retries: int = 200) -> None:
    """Write bytes start..end at their offset in path; resume after a dropped connection."""
    pos, tries = start, 0
    while pos <= end:
        try:
            req = urllib.request.Request(url, headers={"Range": f"bytes={pos}-{end}"})
            with urllib.request.urlopen(req, timeout=60) as r, open(path, "r+b") as f:
                f.seek(pos)
                while pos <= end:
                    b = r.read(1 << 20)
                    if not b:
                        break
                    f.write(b)
                    pos += len(b)
                    done[k] = pos - start
        except Exception as e:  # noqa: BLE001
            tries += 1
            if tries > retries:
                raise RuntimeError(f"part {k}: too many retries ({type(e).__name__})") from e
            time.sleep(min(30, 2 * tries))


def download(url: str, path: Path, n: int) -> None:
    total = size_of(url)
    with open(path, "wb") as f:
        f.truncate(total)
    rs = parts(total, n)
    done = [0] * len(rs)
    errs: list[str] = []

    def run(k, s, e):
        try:
            fetch_part(url, path, s, e, done, k)
        except Exception as ex:  # noqa: BLE001
            errs.append(str(ex))
    ts = [threading.Thread(target=run, args=(k, s, e), daemon=True) for k, (s, e) in enumerate(rs)]
    for t in ts:
        t.start()
    t0, last = time.time(), 0.0
    while any(t.is_alive() for t in ts):
        time.sleep(5)
        if time.time() - last >= 60:
            last = time.time()
            got = sum(done)
            log(f"download {got / 1e9:.2f} / {total / 1e9:.2f} GB, {got / 1e6 / max(1, time.time() - t0):.1f} MB/s")
    if errs or sum(done) != total:
        raise RuntimeError(f"download incomplete: {sum(done)} of {total} bytes; {errs[:2]}")
    log(f"downloaded {total / 1e9:.2f} GB in {(time.time() - t0) / 60:.1f} min")


def copy_member(z: zipfile.ZipFile, name: str, dst: Path) -> None:
    t0, last, n = time.time(), time.time(), 0
    with z.open(name) as src, open(dst, "wb") as out:
        while True:
            b = src.read(8 << 20)
            if not b:
                break
            out.write(b)
            n += len(b)
            if time.time() - last >= 60:
                last = time.time()
                log(f"  {name}: {n / 1e9:.2f} GB written, {(time.time() - t0) / 60:.1f} min")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--want", type=Path, required=True, help="JSON {db_id: bytes}")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--work", type=Path, required=True)
    ap.add_argument("--conns", type=int, default=8)
    ap.add_argument("--min-free-gb", type=float, default=45.0)
    ap.add_argument("--url", default=URL)
    a = ap.parse_args()
    want = json.loads(a.want.read_text())
    a.out.mkdir(parents=True, exist_ok=True)
    for f in ("FETCH_OK", "FETCH_FAIL"):
        (a.out / f).unlink(missing_ok=True)
    try:
        have = {db for db, b in want.items() if (a.out / db / f"{db}.sqlite").exists()
                and (a.out / db / f"{db}.sqlite").stat().st_size == b}
        if have == set(want):
            log(f"all {len(want)} databases already present")
        else:
            a.work.mkdir(parents=True, exist_ok=True)
            free = shutil.disk_usage(a.work).free / 1e9
            if free < a.min_free_gb:
                raise RuntimeError(f"only {free:.0f} GB free under {a.work} (need {a.min_free_gb:.0f})")
            outer, inner = a.work / "train.zip", a.work / "train_databases.zip"
            if not inner.exists():
                download(a.url, outer, a.conns)
                with zipfile.ZipFile(outer) as z:
                    log(f"decompressing {INNER}")
                    copy_member(z, INNER, inner)
                outer.unlink()
            with zipfile.ZipFile(inner) as z:
                names = set(z.namelist())
                for db in sorted(set(want) - have):
                    m = next((n for n in (f"train_databases/{db}/{db}.sqlite", f"{db}/{db}.sqlite") if n in names), None)
                    if m is None:
                        raise RuntimeError(f"{db}.sqlite not in the archive")
                    (a.out / db).mkdir(parents=True, exist_ok=True)
                    dst = a.out / db / f"{db}.sqlite"
                    copy_member(z, m, dst)
                    if dst.stat().st_size != want[db]:
                        raise RuntimeError(f"{db}: {dst.stat().st_size} bytes, expected {want[db]}")
                    log(f"extracted {db} ({want[db] / 1e9:.2f} GB)")
            inner.unlink()
        (a.out / "FETCH_OK").write_text(json.dumps(sorted(want)) + "\n")
        log(f"FETCH_OK {len(want)} databases")
    except Exception as e:  # noqa: BLE001
        (a.out / "FETCH_FAIL").write_text(f"{type(e).__name__}: {e}\n")
        log(f"FETCH_FAIL {type(e).__name__}: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
