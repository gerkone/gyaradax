"""Campaign page: parameter distributions against an existing dataset, progress, and a head-cut cutter (or flux traces).

Reads CAMPAIGN/configs/NAME.yaml (order from CAMPAIGN/order.txt when present), CAMPAIGN/runs/NAME (run directories of
`gyaradax datagen`), the trajectories NAME_ifft_realpotens under each --dataset directory, and the configs of the
trajectories matching --existing. Writes OUT.html from template.html next to this file. Publish it as an artifact with
the `db` capability; cut decisions land in its collection `cuts` ({status: accept | keep, cut: t}).

usage: build_page.py CAMPAIGN OUT.html [--title T] [--dataset DIR ...] [--existing GLOB] [--axes eps,rlt_i] [--t-end 500] [--eval]
"""

import argparse
import glob
import json
import os
import pickle
import time

import numpy as np
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
PARAMS = ("rlt_i", "rlt_e", "rln", "te_ti", "q", "shat", "eps")
FINAL = {"0": "done", "2": "diverged", "6": "banded", "75": "interrupted"}


def as_list(v):
    return list(v) if isinstance(v, (list, tuple)) else [v]


def finite(d):
    return {k: (None if isinstance(v, float) and not np.isfinite(v) else v) for k, v in d.items()}


def params(cfg_path):
    cfg = yaml.safe_load(open(cfg_path))
    ph, ge = cfg.get("physics", {}), cfg.get("geometry", {})
    rlt, rln, tmp = (
        as_list(ph.get("rlt", np.nan)),
        as_list(ph.get("rln", np.nan)),
        as_list(ph.get("tmp", 1.0)),
    )
    return {
        "rlt_i": float(rlt[0]),
        "rlt_e": float(rlt[-1]) if len(rlt) > 1 else float("nan"),
        "rln": float(rln[0]),
        "te_ti": float(tmp[-1]) / float(tmp[0]) if len(tmp) > 1 else 1.0,
        "q": float(ge.get("q", np.nan)),
        "shat": float(ge.get("shat", np.nan)),
        "eps": float(ge.get("eps", np.nan)),
    }


def flat(d, p=""):
    for k, v in d.items():
        if isinstance(v, dict):
            yield from flat(v, p + k + "/")
        else:
            yield p + k, v


def traj_meta(name, datasets):
    for d in datasets:
        path = os.path.join(d, f"{name}_ifft_realpotens", "metadata_light.pkl")
        if os.path.exists(path):
            m = dict(flat(pickle.load(open(path, "rb"))))
            return {k.split("/")[-1]: v for k, v in m.items()}
    return {}


def thin(t, cols, n=400):
    step = max(1, len(t) // n)
    return [round(float(x), 3) for x in t[::step]], [
        [float(f"{y:.4g}") for y in c[::step]] for c in cols
    ]


def proposal(t, qi):
    # first frame where the 5-frame mean of qi reaches its mean over the last 30 %
    sat = float(qi[t >= 0.7 * t[-1]].mean())
    sm = np.convolve(qi, np.ones(5) / 5, mode="same")
    hit = np.flatnonzero(sm >= sat) if sat > 0 else []
    return (float(t[hit[0]]) if len(hit) and hit[0] > 0 else None), sat


def run_row(c, name, datasets, t_end):
    rd = os.path.join(c, "runs", name)
    row = {
        "name": name,
        "params": params(os.path.join(c, "configs", f"{name}.yaml")),
        "state": "planned",
        "where": "",
        "t": [],
        "qi": [],
        "qe": None,
        "t_now": None,
        "rate": None,
        "frames": 0,
        "frame_t": [],
        "proposal": None,
        "sat_qi": None,
        "qi_late": None,
        "qe_late": None,
        "stable": False,
    }
    who = os.path.join(c, "claims", name, "who")
    if os.path.exists(who):
        row["where"] = " ".join(open(who).read().split()[:2])
    code = (
        open(os.path.join(rd, "datagen_exit")).read().strip()
        if os.path.exists(os.path.join(rd, "datagen_exit"))
        else None
    )
    if not os.path.exists(os.path.join(rd, "fluxes.npz")):
        return row
    z = np.load(os.path.join(rd, "fluxes.npz"))
    t, f = np.ravel(z["time"]).astype(float), np.asarray(z["fluxes"], float)
    f = f.reshape(len(t), -1, 3)
    fresh = time.time() - os.path.getmtime(os.path.join(rd, "fluxes.npz")) < 1800
    row["state"] = (
        "running"
        if fresh and code in (None, "75")
        else FINAL.get(code, f"exit {code}" if code else "interrupted")
    )
    info = json.loads(open(os.path.join(rd, "run_info.jsonl")).readlines()[-1])
    start = time.mktime(time.strptime(info["started"][:19], "%Y-%m-%dT%H:%M:%S"))
    t0 = float(
        t[min(np.searchsorted(np.ravel(z["step"]), info.get("start_step") or 0), len(t) - 1)]
    )
    hours = (os.path.getmtime(os.path.join(rd, "fluxes.npz")) - start) / 3600
    row["t_now"], row["rate"] = float(t[-1]), (float(t[-1]) - t0) / hours if hours > 0 else None
    kinetic = f.shape[1] > 1
    row["t"], cols = thin(t, [f[:, 0, 1], f[:, -1, 1]] if kinetic else [f[:, 0, 1]])
    row["qi"], row["qe"] = cols[0], cols[1] if kinetic else None
    late = t >= 0.7 * t[-1]
    row["qi_late"], row["qe_late"] = (
        float(f[late, 0, 1].mean()),
        float(f[late, -1, 1].mean()) if kinetic else None,
    )
    m = traj_meta(name, datasets)
    row["stable"] = bool(m.get("stable", False))
    if "timesteps" in m:
        row["frame_t"] = [round(float(x), 3) for x in np.asarray(m["timesteps"])]
        row["frames"] = len(row["frame_t"]) if not m.get("frames_dropped") else 0
    if row["state"] == "done" and not row["stable"]:
        row["proposal"], row["sat_qi"] = proposal(t, f[:, 0, 1])
    return row


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("campaign")
    p.add_argument("out")
    p.add_argument("--title", default=None)
    p.add_argument("--dataset", nargs="*", default=[])
    p.add_argument(
        "--existing", default=None, help="glob of existing trajectories (each with a config.yaml)"
    )
    p.add_argument("--axes", default="eps,rlt_i")
    p.add_argument("--t-end", type=float, default=500.0)
    p.add_argument(
        "--eval", action="store_true", help="evaluation campaign: flux traces instead of the cutter"
    )
    a = p.parse_args()

    c = os.path.abspath(a.campaign)
    order = os.path.join(c, "order.txt")
    names = (
        [n.strip() for n in open(order) if n.strip() and not n.startswith("#")]
        if os.path.exists(order)
        else sorted(
            os.path.basename(f)[:-5] for f in glob.glob(os.path.join(c, "configs", "*.yaml"))
        )
    )
    runs = [
        run_row(c, n, a.dataset, a.t_end)
        for n in names
        if os.path.exists(os.path.join(c, "configs", f"{n}.yaml"))
    ]
    existing = [
        params(os.path.join(tr, "config.yaml"))
        for tr in sorted(glob.glob(a.existing or ""))
        if os.path.exists(os.path.join(tr, "config.yaml"))
    ]
    payload = {
        "built": time.strftime("%Y-%m-%d %H:%M"),
        "t_end": a.t_end,
        "eval": a.eval,
        "axes": a.axes.split(","),
        "params": list(PARAMS),
        "existing": [finite(e) for e in existing],
        "existing_from": a.existing,
        "runs": [{**r, "params": finite(r["params"])} for r in runs],
    }
    title = a.title or os.path.basename(c)
    html = open(os.path.join(HERE, "template.html")).read()
    with open(a.out, "w") as fh:
        fh.write(html.replace("__TITLE__", title).replace("/*DATA*/null", json.dumps(payload)))
    print(a.out, payload["built"], len(runs), "runs,", len(existing), "existing")


if __name__ == "__main__":
    main()
