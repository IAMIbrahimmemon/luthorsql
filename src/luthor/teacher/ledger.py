"""Tokens used per provider/model per day and per ISO week, append-only (one json line per call), safe across
threads and processes (each record is one short O_APPEND write). Optional caps stop a job cleanly before it
would go over: the next run resumes where it stopped.
"""
from __future__ import annotations

import collections
import datetime as dt
import json
import os
import threading
from pathlib import Path


def _day(ts: float) -> str:
    return dt.datetime.fromtimestamp(ts, dt.timezone.utc).strftime("%Y-%m-%d")


def _week(ts: float) -> str:
    y, w, _ = dt.datetime.fromtimestamp(ts, dt.timezone.utc).isocalendar()
    return f"{y}-W{w:02d}"


class CapReached(Exception):
    pass


class Ledger:
    def __init__(self, path: Path, day_cap: int = 0, week_cap: int = 0):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.day_cap, self.week_cap = day_cap, week_cap
        self.lock = threading.Lock()
        self.tot = collections.Counter()  # (provider, model, "d:"+day | "w:"+week) -> tokens
        if self.path.exists():
            for ln in open(self.path):
                try:
                    self._add(json.loads(ln))
                except (json.JSONDecodeError, KeyError):
                    continue

    def _add(self, r: dict) -> None:
        n = int(r.get("in", 0)) + int(r.get("out", 0))
        self.tot[(r["provider"], r["model"], "d:" + _day(r["ts"]))] += n
        self.tot[(r["provider"], r["model"], "w:" + _week(r["ts"]))] += n

    def record(self, provider: str, model: str, n_in: int, n_out: int, job: str = "", ts: float | None = None) -> None:
        import time
        r = {"ts": ts if ts is not None else time.time(), "provider": provider, "model": model,
             "in": int(n_in), "out": int(n_out), "job": job}
        line = (json.dumps(r) + "\n").encode()
        with self.lock:
            fd = os.open(self.path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o644)
            try:
                os.write(fd, line)
            finally:
                os.close(fd)
            self._add(r)

    def used(self, provider: str, model: str, ts: float | None = None) -> tuple[int, int]:
        import time
        ts = ts if ts is not None else time.time()
        return (self.tot[(provider, model, "d:" + _day(ts))], self.tot[(provider, model, "w:" + _week(ts))])

    def check(self, provider: str, model: str) -> None:
        d, w = self.used(provider, model)
        if self.day_cap and d >= self.day_cap:
            raise CapReached(f"{provider}/{model}: {d:,} tokens today >= day cap {self.day_cap:,}")
        if self.week_cap and w >= self.week_cap:
            raise CapReached(f"{provider}/{model}: {w:,} tokens this week >= week cap {self.week_cap:,}")

    def summary(self) -> dict:
        """{provider/model: {"days": {day: tokens}, "weeks": {week: tokens}}}"""
        out: dict = {}
        for (p, m, k), n in sorted(self.tot.items()):
            e = out.setdefault(f"{p}/{m}", {"days": {}, "weeks": {}})
            (e["days"] if k.startswith("d:") else e["weeks"])[k[2:]] = n
        return out
