"""Teacher model adapters. Both take chat messages and return text plus token counts.

  ollama   the LOCAL ollama daemon (default http://localhost:11434), so any model `ollama list` shows works,
           including Ollama Cloud models (deepseek-v4.1-flash:cloud, glm-5.3:cloud) on the owner's subscription;
           or the ollama.com API directly (--ollama-url https://ollama.com, model without ":cloud") with the
           owner's API key from ~/.config/ollama/api_key (or $OLLAMA_KEY_FILE).
  openai   any OpenAI-compatible /v1/chat/completions endpoint (xAI and others); the key is read from an
           environment variable named on the command line and is never printed or logged.

Every prompt passes the leak gate first (leak.LeakGate); a hit raises LeakRefused and nothing is sent.

Rate limits: a 429, or an error that says usage/session/rate limit or quota, pauses EVERY worker that shares the
Throttle (the subscription's limit is account-wide), with growing waits (1, 2, 5, 10, 30, 30... minutes, or the
server's Retry-After) until the 5-hour session window rolls over. A job gives up only after max_wait seconds in
one continuous pause; since outputs are appended as they arrive, re-running the same command resumes.
"""
from __future__ import annotations

import json
import os
import re
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from .leak import LeakGate
from .ledger import Ledger

LIMIT_WORDS = re.compile(r"\b(429|rate.?limit|usage limit|session limit|quota|too many requests|limit reached|"
                         r"weekly limit|limit exceeded)\b", re.I)
BACKOFF = [60, 120, 300, 600, 1800]  # then 1800 s repeated until max_wait


class LeakRefused(Exception):
    pass


class RateLimited(Exception):
    def __init__(self, msg: str, retry_after: float | None = None):
        super().__init__(msg)
        self.retry_after = retry_after


class GaveUp(Exception):
    pass


@dataclass
class Reply:
    text: str
    n_in: int
    n_out: int
    seconds: float
    thinking: str = ""


class Throttle:
    """Shared pause: when one worker hits a limit every worker waits; the step grows while limits continue."""

    def __init__(self, max_wait: float = 6 * 3600, sleep=time.sleep, clock=time.time, log=print):
        self.max_wait, self.sleep, self.clock, self.log = max_wait, sleep, clock, log
        self.lock = threading.Lock()
        self.until = 0.0
        self.step = 0
        self.pause_start = None

    def wait(self) -> None:
        while True:
            with self.lock:
                left = self.until - self.clock()
            if left <= 0:
                return
            self.sleep(min(left, 30))

    def limited(self, retry_after: float | None, why: str) -> None:
        with self.lock:
            now = self.clock()
            if now < self.until:  # another worker already paused us; no extra step
                return
            if self.pause_start is None:
                self.pause_start = now
            if now - self.pause_start > self.max_wait:
                raise GaveUp(f"rate limited for {now - self.pause_start:.0f}s > max_wait; re-run to resume ({why[:120]})")
            d = retry_after if retry_after else BACKOFF[min(self.step, len(BACKOFF) - 1)]
            self.step += 1
            self.until = now + d
            self.log(f"[throttle] limit hit ({why[:120]}); all workers pause {d:.0f}s", flush=True)

    def ok(self) -> None:
        with self.lock:
            if self.clock() >= self.until:
                self.step, self.pause_start = 0, None


def _retry_after(headers) -> float | None:
    try:
        v = headers.get("Retry-After") if headers is not None else None
        return float(v) if v else None
    except (TypeError, ValueError):
        return None


class Provider:
    name = "base"

    def __init__(self, model: str, gate: LeakGate, ledger: Ledger | None = None, throttle: Throttle | None = None,
                 timeout: float = 600, retries: int = 4, job: str = ""):
        self.model, self.gate, self.ledger = model, gate, ledger
        self.throttle = throttle or Throttle()
        self.timeout, self.retries, self.job = timeout, retries, job

    def _post(self, url: str, body: dict, headers: dict) -> dict:
        req = urllib.request.Request(url, data=json.dumps(body).encode(), headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                return json.load(r)
        except urllib.error.HTTPError as e:
            txt = ""
            try:
                txt = e.read().decode("utf-8", "replace")[:500]
            except Exception:
                pass
            if e.code == 429 or (e.code in (402, 403, 503) and LIMIT_WORDS.search(txt)):
                raise RateLimited(f"HTTP {e.code} {txt}", _retry_after(e.headers)) from None
            raise RuntimeError(f"HTTP {e.code} {txt}") from None

    def _call(self, messages, temperature, max_tokens, seed, think) -> Reply:
        raise NotImplementedError

    def chat(self, messages: list[dict], temperature: float = 0.0, max_tokens: int = 4096, seed: int | None = None,
             think: bool | None = None) -> Reply:
        text = "\n".join(m["content"] for m in messages)
        cls = self.gate.hit(text)
        if cls is not None:
            raise LeakRefused(f"prompt matched {cls} pattern; not sent")
        if self.ledger is not None:
            self.ledger.check(self.name, self.model)
        err = None
        k = 0
        while k < self.retries:
            self.throttle.wait()
            try:
                rep = self._call(messages, temperature, max_tokens, seed, think)
            except RateLimited as e:  # does not use up a retry: waiting out the window is the point
                self.throttle.limited(e.retry_after, str(e))
                continue
            except Exception as e:
                msg = str(e)
                if LIMIT_WORDS.search(msg) and not isinstance(e, TimeoutError):
                    self.throttle.limited(None, msg)
                    continue
                err = e
                k += 1
                time.sleep(min(60, 3 * 2 ** k))
                continue
            self.throttle.ok()
            if self.ledger is not None:
                self.ledger.record(self.name, self.model, rep.n_in, rep.n_out, self.job)
            return rep
        raise RuntimeError(f"{self.name}/{self.model} failed after {self.retries} tries: {err}")


class Ollama(Provider):
    name = "ollama"

    def __init__(self, model: str, gate: LeakGate, url: str = "http://localhost:11434", **kw):
        super().__init__(model, gate, **kw)
        self.url = url.rstrip("/") + "/api/chat"
        # direct ollama.com API (not the local daemon): key from a private file, never printed or logged
        self.headers = {"Content-Type": "application/json"}
        if not re.match(r"https?://(localhost|127\.0\.0\.1)[:/]", self.url):
            kf = Path(os.environ.get("OLLAMA_KEY_FILE", "~/.config/ollama/api_key")).expanduser()
            if not kf.exists():
                raise RuntimeError(f"remote ollama url needs a key file ({kf})")
            self.headers["Authorization"] = "Bearer " + kf.read_text().strip()

    def _call(self, messages, temperature, max_tokens, seed, think) -> Reply:
        opts = {"temperature": temperature, "num_predict": max_tokens}
        if seed is not None:
            opts["seed"] = seed
        body = {"model": self.model, "stream": False, "options": opts, "messages": messages}
        if think is not None:
            body["think"] = think
        t0 = time.time()
        d = self._post(self.url, body, self.headers)
        if d.get("error"):
            e = str(d["error"])
            if LIMIT_WORDS.search(e):
                raise RateLimited(e)
            raise RuntimeError(e)
        m = d.get("message") or {}
        return Reply(m.get("content") or "", int(d.get("prompt_eval_count") or 0), int(d.get("eval_count") or 0),
                     time.time() - t0, m.get("thinking") or "")


class OpenAICompat(Provider):
    """POST {base_url}/chat/completions with Authorization: Bearer $<key_env>. Not needed for the bake-off."""
    name = "openai"

    def __init__(self, model: str, gate: LeakGate, base_url: str, key_env: str, **kw):
        super().__init__(model, gate, **kw)
        self.url = base_url.rstrip("/") + "/chat/completions"
        self.key_env = key_env
        self.name = f"openai:{re.sub(r'^https?://', '', base_url).split('/')[0]}"
        if not os.environ.get(key_env):
            raise RuntimeError(f"environment variable {key_env} is not set")

    def _call(self, messages, temperature, max_tokens, seed, think) -> Reply:
        body = {"model": self.model, "messages": messages, "temperature": temperature, "max_tokens": max_tokens}
        if seed is not None:
            body["seed"] = seed
        t0 = time.time()
        d = self._post(self.url, body, {"Content-Type": "application/json",
                                        "Authorization": "Bearer " + os.environ[self.key_env]})
        ch = (d.get("choices") or [{}])[0].get("message") or {}
        u = d.get("usage") or {}
        return Reply(ch.get("content") or "", int(u.get("prompt_tokens") or 0), int(u.get("completion_tokens") or 0),
                     time.time() - t0, ch.get("reasoning_content") or "")


def make(spec: str, gate: LeakGate, **kw) -> Provider:
    """'ollama:<model>' (or a bare model name) or 'openai:<model>@<base_url>#<KEY_ENV>'."""
    url = kw.pop("ollama_url", None) or "http://localhost:11434"
    if spec.startswith("openai:"):
        m = re.fullmatch(r"openai:(.+?)@(https?://[^#]+)#([A-Z_][A-Z0-9_]*)", spec)
        if not m:
            raise ValueError("openai spec: openai:<model>@<base_url>#<KEY_ENV>")
        return OpenAICompat(m.group(1), gate, m.group(2), m.group(3), **kw)
    return Ollama(spec.removeprefix("ollama:"), gate, url=url, **kw)
