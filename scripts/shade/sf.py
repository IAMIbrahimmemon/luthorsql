"""Shadeform GPU client (https://docs.shadeform.ai). On-demand only, delete (never stop).

  python scripts/shade/sf.py offers [H100] [--num-gpus 2] [--n 10]   # cheapest available on-demand offers (free)
  python scripts/shade/sf.py addkey                       # upload ~/.ssh/runpod_luthor.pub once
  python scripts/shade/sf.py create <cloud> <region> <shade_instance_type> --max-spend DOLLARS \
         [--num-gpus N] [--os IMAGE] [--name NAME]        # prints the id; --max-spend is required (auto-delete cap)
  python scripts/shade/sf.py info <id>                    # status, ip, ssh user/port, price, cost so far
  python scripts/shade/sf.py status <id>                  # one word: the status, or "gone" if the API no longer knows it
  python scripts/shade/sf.py hostline <id>                # "HOST PORT USER" once active (for ssh.sh)
  python scripts/shade/sf.py delete <id>
  python scripts/shade/sf.py list

create checks the type against the live catalogue first: it must exist in that cloud and region with exactly
--num-gpus GPUs (default 1) and be available on demand, and --os must be one of its os_options.
The instance name is --name, else $SF_NAME, else luthor-r6 (Shadeform refuses two live instances with one name).

The API key is read from ~/.config/shadeform/api_key (chmod 600) or SHADEFORM_API_KEY. It is never printed,
logged or sent anywhere but api.shadeform.ai.

Exit codes: 0 ok, 1 API refused (HTTP error) or bad arguments, 3 network error (DNS, timeout) after retries.
GET requests are retried on network errors; POSTs are not (a retried create could start a second VM).
"""
from __future__ import annotations

import argparse
import json
import os
import socket
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

API = "https://api.shadeform.ai/v1"
KEY_FILE = Path.home() / ".config" / "shadeform" / "api_key"
PUB = Path(os.environ.get("VAST_KEY", str(Path.home() / ".ssh" / "runpod_luthor")) + ".pub")
NAME = os.environ.get("SF_NAME", "luthor-r6")
NET_EXIT = 3


class Gone(Exception):
    """The API answered 404 for an instance: it no longer exists."""


def _key() -> str:
    k = os.environ.get("SHADEFORM_API_KEY") or (KEY_FILE.read_text().strip() if KEY_FILE.exists() else "")
    if not k:
        sys.exit(f"no Shadeform API key: put it in {KEY_FILE} (chmod 600)")
    return k


def call(method: str, path: str, body: dict | None = None, tries: int | None = None, missing_ok: bool = False) -> dict:
    tries = tries or (4 if method == "GET" else 1)
    for t in range(tries):
        req = urllib.request.Request(API + path, method=method,
                                     headers={"X-API-KEY": _key(), "Content-Type": "application/json"},
                                     data=json.dumps(body).encode() if body is not None else None)
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                txt = r.read().decode()
            return json.loads(txt) if txt.strip() else {}
        except urllib.error.HTTPError as e:
            if missing_ok and e.code == 404:
                raise Gone(path) from None
            if e.code >= 500 and t + 1 < tries:
                time.sleep(5 * (t + 1))
                continue
            sys.exit(f"Shadeform {method} {path}: HTTP {e.code} {e.read().decode()[:300]}")
        except (urllib.error.URLError, socket.timeout, TimeoutError, ConnectionError) as e:
            if t + 1 < tries:
                time.sleep(5 * (t + 1))
                continue
            print(f"Shadeform {method} {path}: network error: {getattr(e, 'reason', e)}", file=sys.stderr)
            sys.exit(NET_EXIT)
    raise AssertionError("unreachable")


def types(gpu: str, num_gpus: int) -> list[dict]:
    return call("GET", f"/instances/types?num_gpus={num_gpus}&available=true&sort=price&gpu_type={gpu}").get("instance_types", [])


def offers(gpu: str, n: int, num_gpus: int) -> None:
    rows = []
    for it in types(gpu, num_gpus):
        c = it.get("configuration", {})
        for av in it.get("availability", []):
            if av.get("available") and av.get("rental_type", "on_demand") == "on_demand":
                rows.append((it["hourly_price"] / 100, it["cloud"], av["region"], it["shade_instance_type"], c.get("gpu_type"),
                             c.get("num_gpus"), c.get("vram_per_gpu_in_gb"), c.get("storage_in_gb"), c.get("vcpus"),
                             it.get("deployment_type"), ",".join(c.get("os_options") or [])[:80]))
    rows.sort()
    print(f"{'$/h':>6}  {'cloud':14s} {'region':18s} {'type':16s} {'gpu':10s} ngpu vram disk vcpu deploy  os")
    for r in rows[:n]:
        print(f"{r[0]:6.2f}  {r[1]:14s} {r[2]:18s} {r[3]:16s} {str(r[4]):10s} {r[5]!s:>4} {r[6]!s:>4} {r[7]!s:>4} {r[8]!s:>4} "
              f"{r[9]!s:6s} {r[10]}")
    if not rows:
        print(f"no available on-demand {num_gpus}x{gpu} offers")


def addkey() -> None:
    have = call("GET", "/sshkeys").get("ssh_keys", [])
    pub = PUB.read_text().strip()
    for k in have:
        if k.get("public_key", "").strip() == pub:
            print(f"key already added: {k['id']}")
            return
    print(call("POST", "/sshkeys/add", {"name": "luthor", "public_key": pub})["id"])


def key_id() -> str:
    pub = PUB.read_text().strip()
    for k in call("GET", "/sshkeys").get("ssh_keys", []):
        if k.get("public_key", "").strip() == pub:
            return k["id"]
    sys.exit("SSH key not on the account: run sf.py addkey")


def check_type(cloud: str, region: str, typ: str, num_gpus: int, os_: str | None) -> dict:
    """The catalogue entry for (cloud, type) with num_gpus GPUs, available on demand in region; exits otherwise."""
    for it in call("GET", f"/instances/types?num_gpus={num_gpus}&available=true").get("instance_types", []):
        if it.get("cloud") != cloud or it.get("shade_instance_type") != typ:
            continue
        c = it.get("configuration", {})
        if int(c.get("num_gpus") or it.get("num_gpus") or 0) != num_gpus:
            continue
        if not any(av.get("region") == region and av.get("available") and av.get("rental_type", "on_demand") == "on_demand"
                   for av in it.get("availability", [])):
            sys.exit(f"{cloud} {typ} is not available on demand in {region} right now (run: sf.py offers --num-gpus {num_gpus})")
        if os_ and os_ not in (c.get("os_options") or []):
            sys.exit(f"--os {os_} not offered for {cloud} {typ}; options: {c.get('os_options')}")
        return it
    sys.exit(f"no {num_gpus}-GPU type {typ!r} in {cloud}'s available catalogue (run: sf.py offers --num-gpus {num_gpus})")


def create(cloud: str, region: str, typ: str, os_: str | None, max_spend: float, num_gpus: int, name: str) -> None:
    if not max_spend or max_spend <= 0:
        sys.exit("create needs an explicit --max-spend DOLLARS (Shadeform auto-deletes the VM once that much is spent)")
    it = check_type(cloud, region, typ, num_gpus, os_)
    price = (it.get("hourly_price") or 0) / 100
    print(f"creating {name}: {cloud} {region} {typ} ({num_gpus} GPU) ${price:.2f}/h, os={os_ or 'default'}, "
          f"auto-delete at ${max_spend:.2f} (about {max_spend / price:.1f} h)" if price else "", file=sys.stderr)
    body = {"cloud": cloud, "region": region, "shade_instance_type": typ, "shade_cloud": True, "name": name,
            "rental_type": "on_demand", "ssh_key_id": key_id(),
            "auto_delete": {"spend_threshold": f"{max_spend:.2f}"}}  # hard cost cap: Shadeform deletes the VM past it
    if os_:
        body["os"] = os_
    print(call("POST", "/instances/create", body)["id"])


def info(i: str, missing_ok: bool = False) -> dict:
    return call("GET", f"/instances/{i}/info", missing_ok=missing_ok)


def status(i: str) -> str:
    """The instance's status, or "gone" when the API returns 404 for it and the instance list does not show it."""
    try:
        return str(info(i, missing_ok=True).get("status") or "unknown")
    except Gone:
        live = [d for d in call("GET", "/instances").get("instances", []) if d.get("id") == i]
        return str(live[0].get("status")) if live else "gone"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["offers", "addkey", "create", "info", "status", "hostline", "delete", "list", "set-cap"])
    ap.add_argument("args", nargs="*")
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--num-gpus", type=int, default=None, help="offers / create: GPUs per instance (default 1)")
    ap.add_argument("--os", help="create: OS image, one of the type's os_options (e.g. ubuntu24.04_cuda13.0_shade_os)")
    ap.add_argument("--name", help="create: instance name (default $SF_NAME or luthor-r6)")
    ap.add_argument("--max-spend", type=float, default=None, help="create (required): auto-delete once this many dollars are spent")
    a = ap.parse_args()
    ngpu = a.num_gpus or 1
    if a.cmd == "offers":
        offers(a.args[0] if a.args else "H100", a.n, ngpu)
    elif a.cmd == "addkey":
        addkey()
    elif a.cmd == "create":
        if len(a.args) != 3:
            sys.exit("usage: sf.py create <cloud> <region> <shade_instance_type> --max-spend DOLLARS [--num-gpus N] [--os OS] [--name NAME]")
        if a.max_spend is None:
            sys.exit("create needs an explicit --max-spend DOLLARS (no default)")
        create(*a.args, a.os, a.max_spend, ngpu, a.name or NAME)
    elif a.cmd == "set-cap":   # sf.py set-cap <id> --max-spend DOLLARS: change a running box's auto-delete cap
        if len(a.args) != 1 or not a.max_spend or a.max_spend <= 0:
            sys.exit("usage: sf.py set-cap <id> --max-spend DOLLARS")
        call("POST", f"/instances/{a.args[0]}/update", {"auto_delete": {"spend_threshold": f"{a.max_spend:.2f}"}})
        print(f"auto-delete cap of {a.args[0]} set to ${a.max_spend:.2f}")
    elif a.cmd == "info":
        d = info(a.args[0])
        print(json.dumps({k: d.get(k) for k in ("id", "name", "status", "cloud", "region", "shade_instance_type", "ip",
                                                 "ssh_user", "ssh_port", "hourly_price", "cost_estimate", "active_at")}, indent=1))
    elif a.cmd == "status":
        print(status(a.args[0]))
    elif a.cmd == "hostline":
        d = info(a.args[0])
        if d.get("status") != "active" or not d.get("ip"):
            sys.exit(f"not active yet: {d.get('status')}")
        print(d["ip"], d.get("ssh_port") or 22, d.get("ssh_user") or "root")
    elif a.cmd == "delete":
        d = info(a.args[0])
        print(f"instance {d.get('id')} {d.get('shade_instance_type')} {d.get('cloud')} status={d.get('status')} "
              f"${(d.get('hourly_price') or 0) / 100:.2f}/h cost so far ${d.get('cost_estimate')}")
        call("POST", f"/instances/{a.args[0]}/delete")
        print(f"DELETE_REQUESTED {a.args[0]}")
    elif a.cmd == "list":
        for d in call("GET", "/instances").get("instances", []):
            print(d.get("id"), d.get("name"), d.get("status"), d.get("shade_instance_type"), d.get("cloud"),
                  f"${(d.get('hourly_price') or 0) / 100:.2f}/h")


if __name__ == "__main__":
    main()
