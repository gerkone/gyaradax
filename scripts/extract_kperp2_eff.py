"""Eigenmode-weighted <k_perp^2>(ky) for every GKW linear run.

canonical and sat0 need <k_perp^2>, which needs the s-resolved |phi|^2 and
the metric. Both are already on disk: phi from parallel.dat, the metric from
gyaradax's own analytic geometry. No reruns, and no dependence on the
kx-resolved flux that GKW never writes -- W(ky) is already a column.
"""
import os
import sys

import numpy as np
import pandas as pd

D = "/restricteddata/ukaea/gyrokinetics/preprocessed/summary/"


def kperp2_eff_for(lin_path, q, shat, eps, nmod, nx, ns, geom_cache):
    par = os.path.join(lin_path, "parallel.dat")
    if not os.path.isfile(par):
        return None, "no parallel.dat"
    a = pd.read_csv(par, sep=r"\s+", header=None, usecols=[1, 2],
                    dtype=np.float64).to_numpy()
    if ns is None or ns <= 0:
        if a.shape[0] % (nmod * nx):
            return None, "bad length"
        ns = a.shape[0] // (nmod * nx)
    if a.shape[0] != nmod * nx * ns:
        return None, f"length {a.shape[0]} != {nmod}*{nx}*{ns}"

    krho = np.loadtxt(os.path.join(lin_path, "krho"), ndmin=2)
    kxrh = np.loadtxt(os.path.join(lin_path, "kxrh"), ndmin=2)
    ky = krho[:, 0]                       # (nmod,), constant along x
    kx = kxrh[0, :]                       # (nx,), identical for every ky row

    # (nmod, nx, ns) -> (ns, nkx, nky), the order k_perp_squared expects
    phi2 = (np.abs(a[:, 0] + 1j * a[:, 1]) ** 2).reshape(nmod, nx, ns)
    phi2 = np.transpose(phi2, (2, 1, 0))

    key = (round(q, 6), round(shat, 6), round(eps, 6), nx, nmod, ns)
    if key not in geom_cache:
        from gyaradax.geometry import build_topology, compute_continuous_geometry
        topo = build_topology(nx, nmod, 5, ns)
        g = compute_continuous_geometry(q=q, shat=shat, eps=eps, ns=ns, nkx=nx,
                                        nky=nmod, nvpar=32, nmu=8, ikxspace=5,
                                        krhomax=float(np.nanmax(ky)), topology=topo)
        lg = np.asarray(g["little_g"])
        geom_cache[key] = (lg if lg.shape[0] == 3 else lg.T)   # want (3, ns)
    lg = geom_cache[key]

    # k_perp^2(s,kx,ky) = ky^2 g_zz + 2 ky kx g_ez + kx^2 g_ee
    kp2 = (ky[None, None, :] ** 2 * lg[0][:, None, None]
           + 2.0 * ky[None, None, :] * kx[None, :, None] * lg[1][:, None, None]
           + kx[None, :, None] ** 2 * lg[2][:, None, None])
    norm = phi2.sum(axis=(0, 1))
    eff = np.where(norm > 0, (kp2 * phi2).sum(axis=(0, 1)) / np.maximum(norm, 1e-300), np.nan)
    return eff, "ok"


def main(out):
    cache, res = {}, {}
    for nm in ("adiabatic", "new_runs_adiabatic"):
        df = pd.read_parquet(D + nm + ".parquet")
        got = 0
        for _, r in df.iterrows():
            lp = r.get("lin_path") or ""
            ns = None if pd.isna(r.n_s) else int(r.n_s)
            if not lp:
                continue
            eff, why = kperp2_eff_for(lp, float(r.q), float(r.shat), float(r.eps),
                                      int(r.n_ky), int(r.n_kx), ns, cache)
            if eff is not None:
                res[r["name"]] = eff.tolist()
                got += 1
        print(f"{nm}: {got}/{len(df)}", flush=True)
    import json
    json.dump(res, open(out, "w"))
    print("wrote", out, len(res))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "kperp2_eff.json")
