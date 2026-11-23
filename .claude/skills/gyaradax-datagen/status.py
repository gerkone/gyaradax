"""Table of a datagen campaign: per config its state, host and GPU, simulated time, rate and time left.

usage: status.py CAMPAIGN [--json]
"""

import json
import os
import sys
import time

import numpy as np
import yaml

STATES = {"0": "done", "2": "diverged", "6": "banded", "75": "interrupted"}


def read(path):
    try:
        with open(path) as fh:
            return fh.read().strip()
    except OSError:
        return None


def row(c, name):
    run, claim = f"{c}/runs/{name}", f"{c}/claims/{name}"
    code, who, failed = read(f"{run}/datagen_exit"), read(f"{claim}/who"), read(f"{claim}/failed")
    cfg = yaml.safe_load(open(f"{c}/configs/{name}.yaml"))
    t_end = (cfg.get("solver") or {}).get("t_end")
    r = {
        "name": name,
        "state": "queued",
        "where": " ".join(who.split()[:2]) if who else "",
        "t": None,
        "t_end": t_end,
        "rate": None,
        "left_h": None,
        "exit": code,
    }
    if failed:
        r["state"] = f"failed ({failed})"
    elif code in STATES and not (code == "75" and who):
        r["state"] = STATES[code]
    elif who:
        r["state"] = "running"
    if os.path.exists(f"{run}/fluxes.npz"):
        t = np.ravel(np.load(f"{run}/fluxes.npz")["time"])
        r["t"] = float(t[-1]) if len(t) else 0.0
        info = json.loads(open(f"{run}/run_info.jsonl").readlines()[-1])
        start = time.mktime(time.strptime(info["started"][:19], "%Y-%m-%dT%H:%M:%S"))
        t0 = 0.0
        if info.get("start_step"):
            steps = np.ravel(np.load(f"{run}/fluxes.npz")["step"])
            t0 = float(t[min(np.searchsorted(steps, info["start_step"]), len(t) - 1)])
        hours = (os.path.getmtime(f"{run}/fluxes.npz") - start) / 3600
        if hours > 0 and r["t"] > t0:
            r["rate"] = (r["t"] - t0) / hours
            if r["state"] == "running" and t_end:
                r["left_h"] = max(t_end - r["t"], 0) / r["rate"]
    return r


def fmt(v, f):
    return "" if v is None else format(v, f)


def main():
    c = os.path.abspath(sys.argv[1])
    names = [n.strip() for n in open(f"{c}/order.txt") if n.strip() and not n.startswith("#")]
    rows = [row(c, n) for n in names]
    if "--json" in sys.argv:
        print(json.dumps(rows, indent=1))
        return
    print(
        f"{'name':16s} {'state':14s} {'host gpu':14s} {'t':>7s} {'t_end':>6s} {'t/h':>6s} {'left h':>6s}"
    )
    for r in rows:
        print(
            f"{r['name']:16s} {r['state']:14s} {r['where']:14s} {fmt(r['t'], '7.1f')} {fmt(r['t_end'], '6.0f')} "
            f"{fmt(r['rate'], '6.1f')} {fmt(r['left_h'], '6.1f')}"
        )
    counts = {}
    for r in rows:
        counts[r["state"]] = counts.get(r["state"], 0) + 1
    print(", ".join(f"{k} {v}" for k, v in sorted(counts.items())))


if __name__ == "__main__":
    main()
