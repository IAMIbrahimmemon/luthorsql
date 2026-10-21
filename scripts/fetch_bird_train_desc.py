"""BIRD-train column descriptions (database_description/*.csv) from the official train.zip, nothing else.

The official archive is a zip holding a deflated train_databases.zip, so single members cannot be range-read: the
archive (8.9 GB) is streamed once and unzipped on the fly, and only the `<db>/database_description/<table>.csv`
entries are written (about 1 MB); the SQLite files are discarded as they stream. BIRD dev is never fetched.

Downloads are untrusted data: the CSVs land in a new, empty directory and are only parsed (luthor.sq.bird_desc),
never executed. Run with an isolated interpreter from outside the target directory:

  .venv/bin/python -I scripts/fetch_bird_train_desc.py --out data/raw/bird_train_desc
"""
import argparse
import re
import time
import urllib.request
from pathlib import Path

from stream_unzip import stream_unzip

URL = "https://bird-bench.oss-cn-beijing.aliyuncs.com/train.zip"
SAFE = re.compile(r"^[\w\-. ()&']+$")


def http_chunks(url, size=1 << 20, retries=50):
    """One continuous byte stream; on a dropped connection, resume with a Range request."""
    pos, total, tries = 0, None, 0
    while True:
        req = urllib.request.Request(url, headers={"Range": f"bytes={pos}-"} if pos else {})
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                if total is None:
                    total = int(r.headers["Content-Length"])
                while b := r.read(size):
                    pos += len(b)
                    if pos // (1 << 30) != (pos - len(b)) // (1 << 30):
                        print(f"{pos / (1 << 30):.0f} GiB of {total / (1 << 30):.1f}", flush=True)
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
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    out = a.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    n = 0
    for name, _, chunks in stream_unzip(http_chunks(URL)):
        name = name.decode("utf-8", "replace")
        if not name.endswith("train_databases.zip") or "__MACOSX" in name or "dev" in name.lower():
            for _ in chunks:
                pass
            continue
        for inner, _, ic in stream_unzip(chunks):
            rel = inner.decode("utf-8", "replace").split("train_databases/", 1)[-1]
            parts = rel.split("/")
            keep = (len(parts) == 3 and parts[1] == "database_description" and parts[2].lower().endswith(".csv")
                    and not parts[2].startswith("._") and "__MACOSX" not in rel
                    and all(SAFE.match(p) and p not in (".", "..") for p in parts))
            if not keep:
                for _ in ic:
                    pass
                continue
            p = (out / parts[0] / parts[2]).resolve()
            assert out in p.parents, p
            p.parent.mkdir(parents=True, exist_ok=True)
            with open(p, "wb") as f:
                for c in ic:
                    f.write(c)
            n += 1
    print(f"wrote {n} csv files under {out}")


if __name__ == "__main__":
    main()
