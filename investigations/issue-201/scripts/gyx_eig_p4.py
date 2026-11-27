"""Acceptance variant of gyx_eig.py (env NKX, NS, DISP_PAR, NALPHA, NEND, PX0, KX0, NOBPAR). Leading linear eigenpairs of one gapcheck gyaradax config (fp64 JAX Krylov-Schur on the RK4 step operator of the ky row).

Works with gyaradax master or gyaradax-cgyroconv (whichever is on PYTHONPATH); the config is a translated YAML with a
``<name>.units.json`` sidecar (CGYRO unit factors). Appends one JSON line per solve to RESULTS.

usage: gyx_eig.py CONFIG RESULTS TAG
env: K, KEEP, NCV, TWIN, TOL, MAXR, DT_SCALE, SEED, GMIN
"""

import json
import math
import os
import subprocess
import sys
import time
from dataclasses import replace

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from gyaradax.core.jax_config import enable_x64

enable_x64()
import jax
import jax.numpy as jnp
import numpy as np

import gyaradax
import ks
from gyaradax.core.params import gkparams_from_config, load_config
from gyaradax.geometry.geom import compute_geometry_from_config
from gyaradax.operators.fields import _compute_fields
from gyaradax.operators.precompute import linear_precompute
from gyaradax.solver.cfl import estimate_linear_timestep

cfgp, resp, tag = sys.argv[1:4]
env = os.environ.get
units = json.load(open(cfgp.replace(".yaml", ".units.json")))
rate = units["rate"]
cfg = load_config(cfgp)
# acceptance overrides: grid, dissipation, the local projection, kx shift
for _k, _sec, _t in (("NKX", "grid", int), ("NS", "grid", int), ("DISP_PAR", "solver", float)):
    if env(_k):
        cfg[_sec][{"NKX": "nkx", "NS": "ns", "DISP_PAR": "disp_par"}[_k]] = _t(env(_k))
cfg["solver"]["backend"] = "jax"
if env("NALPHA") is not None:
    cfg["solver"].update(disp_par_neutral=True, disp_par_neutral_end=int(env("NEND", 4)), disp_par_neutral_alpha=float(env("NALPHA")))
if env("NVARIANT"):
    cfg["solver"]["disp_par_neutral_variant"] = env("NVARIANT")
if env("KYIDX"):
    # one ky of a nonlinear deck: nky 2 with krhomax at the deck's ky of that index
    cfg["grid"]["krhomax"] = float(cfg["grid"]["krhomax"]) * int(env("KYIDX")) / (int(cfg["grid"]["nky"]) - 1)
    cfg["grid"]["nky"] = 2
    cfg["solver"].update(non_linear=False, adaptive_dt=False)
if env("NOBPAR"):
    cfg["solver"]["nlbpar"] = False
    cfg["physics"]["betaprime"] = 0.0
code_dir = os.path.dirname(os.path.dirname(gyaradax.__file__))
if "gyaradax-cgyroconv" not in code_dir:
    # master reads no miller pressure gradient from yaml, so a nonzero beta_* would be dropped silently
    assert abs(float(cfg["physics"].get("betaprime", 0.0))) == 0.0, "beta_* needs gyaradax-cgyroconv"
    cfg["solver"].pop("disp_par_conserve", None)
commit = subprocess.run(["git", "-C", code_dir, "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
params = gkparams_from_config(cfg, backend="jax", mixed_precision=False, fused_linear=False, linear_arith="fp64")
params = replace(params, non_linear=False, n_gpus_sp=1, n_gpus_vp=1, n_gpus_mu=1)
geom = compute_geometry_from_config(cfg)
_kx = np.asarray(geom["kxrh"])
if env("PX0"):
    geom["kxrh"] = geom["kxrh"] + float(env("PX0")) * float(_kx[1] - _kx[0])
if env("KX0"):
    geom["kxrh"] = geom["kxrh"] + float(env("KX0"))
pre = linear_precompute(geom, params)
dt_est = float(estimate_linear_timestep(pre, params=params))
iky = 1
g, pre_s, ops, rhs, shape = ks.block(geom, params, iky)
rho = ks.spectral_radius(rhs, shape)
dt = min(dt_est, 2.5 / rho) * float(env("DT_SCALE", 1.0))
K, KEEP, NCV = int(env("K", 8)), int(env("KEEP", 12)), int(env("NCV", 32))
twin, tol, maxr, gmin = float(env("TWIN", 2.5)), float(env("TOL", 1e-6)), int(env("MAXR", 60)), float(env("GMIN", 0.0))
print(f"{tag}: shape {shape} spectral radius {rho:.4e} dt_est {dt_est:.4e} dt {dt:.4e} code {code_dir} {commit}", flush=True)
t0 = time.time()
for _ in range(5):
    nst = max(1, int(round(twin / dt)))
    try:
        vals, vecs, res, bound, conv, nmv, hist = ks.solve(g, params, pre_s, ops, rhs, shape, dt, nst, k=K, keep=KEEP, ncv=NCV,
                                                           tol=tol, gmin=gmin, maxr=maxr, seed=int(env("SEED", 42)),
                                                           log=lambda m: print(m, flush=True))
    except np.linalg.LinAlgError:
        dt *= 0.5
        continue
    # a step-operator mode beyond the rk4 stability edge dominates
    if abs(vals[0]) * dt < 1.0:
        break
    dt *= 0.5
wall = time.time() - t0
geom, pre, iky = g, pre_s, 0
ns = int(cfg["grid"]["ns"])
ixp, ixm = np.asarray(geom["ixplus"])[:, iky], np.asarray(geom["ixminus"])[:, iky]
chain = [int(np.where(ixm < 0)[0][0])]
while ixp[chain[-1]] >= 0:
    chain.append(int(ixp[chain[-1]]))


def gs(x, axis):
    # second-difference norm over 4 * norm: 1 for a sawtooth, ~0 for smooth
    x = np.moveaxis(x, axis, -1)
    d2 = x[..., 2:] - 2 * x[..., 1:-1] + x[..., :-2]
    return float(np.sqrt(np.sum(np.abs(d2) ** 2) / max(np.sum(np.abs(x[..., 1:-1]) ** 2), 1e-300)) / 4.0)


def kcent(x):
    p = np.abs(np.fft.fft(x, axis=-1)) ** 2
    k = np.abs(np.fft.fftfreq(x.shape[-1])) * 2.0
    return float(np.sum(p * k) / max(np.sum(p), 1e-300))


def even(v):
    e, o = np.linalg.norm(v + v[::-1]), np.linalg.norm(v - v[::-1])
    return float(e / max(e + o, 1e-300))


def diag(v):
    phi, apar, bpar = _compute_fields(jnp.asarray(v), geom, params, pre)
    seg = np.asarray(phi)[:, :, iky].T
    ball = np.concatenate([seg[ix] for ix in chain])
    a2 = np.abs(ball) ** 2
    th = 2 * np.pi * (np.arange(ball.size) - (ball.size - 1) / 2) / ns
    out = {"gs_phi": gs(seg, 1), "kc_phi": kcent(seg), "ev_phi": even(ball), "peak_frac": float(a2.max() / a2.sum()),
           "theta_rms": float(np.sqrt(np.sum(a2 * th**2) / a2.sum())),
           "end_frac": float((a2[:ns].sum() + a2[-ns:].sum()) / a2.sum()) if len(chain) > 2 else None}
    if apar is not None:
        ab = np.concatenate([np.asarray(apar)[:, :, iky].T[ix] for ix in chain])
        out.update(ev_apar=even(ab), apar_phi_cgyro=float(np.linalg.norm(ab) / np.linalg.norm(ball) * math.sqrt(2.0)))
    vv = np.asarray(v)[..., iky]
    f2 = np.sum(np.abs(vv) ** 2, axis=tuple(range(1, vv.ndim)))
    out["frac_sp"] = [float(x) for x in f2 / f2.sum()]
    num = out["gs_phi"] > 0.35 or (out["gs_phi"] > 0.3 and out["kc_phi"] > 0.4) or out["peak_frac"] > 0.3
    out["kind"] = "num" if num else "phys"
    return out


modes = []
for v, r, b, vec in zip(vals, res, bound, vecs):
    modes.append(dict(gamma=float(v.real), omega=float(v.imag), gamma_c=float(v.real) / rate, omega_c=-float(v.imag) / rate,
                      res=float(r), ritz_bound=float(b), **diag(vec)))
phys = [m for m in modes if m["kind"] == "phys"]
gr = cfg["grid"]
sol = cfg["solver"]
out = dict(code="gyx", tag=tag, config=cfgp, nvariant=env("NVARIANT"), kyidx=env("KYIDX"), nalpha=env("NALPHA"), nend=env("NEND", "4") if env("NALPHA") is not None else None, px0=env("PX0"), kx0=env("KX0"), code_dir=code_dir, commit=commit, ky_cgyro=units["ky_cgyro"], kthrho=units["ky"],
           ns=gr["ns"], nkx=gr["nkx"], nvpar=gr["nvpar"], nmu=gr["nmu"], vpar_max=gr["vpar_max"],
           disp_par=sol.get("disp_par"), disp_vp=sol.get("disp_vp"), idisp=sol.get("idisp"),
           disp_par_conserve=sol.get("disp_par_conserve", 0), beta=float(params.beta), betaprime=float(params.betaprime),
           nlapar=bool(params.nlapar), nlbpar=bool(params.nlbpar), coll=bool(cfg.get("collisions", {}).get("enabled", False)),
           rate=rate, dt=dt, dt_est=dt_est, spectral_radius=rho, twin=dt * nst, K=K, NCV=NCV, tol=tol, converged=bool(conv),
           matvecs=nmv, chain_len=len(chain), top_phys=phys[0] if phys else None, modes=modes, wall=wall,
           device=str(jax.devices()[0]), host=os.uname().nodename)
print(json.dumps({k: out[k] for k in ("tag", "top_phys", "converged", "matvecs", "wall")}), flush=True)
with open(resp, "a") as f:
    f.write(json.dumps(out) + "\n")
