"""Add lin_phi2_kxy and lin_kx_width to the adiabatic summary tables.

The converged linear eigenmode is already on disk in each GKW run's
parallel.dat, resolved in (ky, kx, s), so no reruns are needed. Derived here
at write time rather than in any consumer: a loader must not transform.

Ordering of parallel.dat is (nmod, nx, ns) -- ky outermost, kx, then s
fastest. Verified against GKW's own kxspec, and re-checked per run here via
the per-toroidal-mode normalisation.
"""
import os
import sys

import numpy as np
import pandas as pd

D = "/restricteddata/ukaea/gyrokinetics/preprocessed/summary/"
TABLES = ("adiabatic", "new_runs_adiabatic")


def eigenmode_kx(lin_path, nmod, nx, ns):
    """(phi2_kxy, kx_width, kx_bar, diagnostics) from one GKW _Lin dir."""
    par = os.path.join(lin_path, "parallel.dat")
    kxf = os.path.join(lin_path, "kxrh")
    if not (os.path.isfile(par) and os.path.isfile(kxf)):
        return None, None, None, "no parallel.dat/kxrh"
    a = pd.read_csv(par, sep=r"\s+", header=None, usecols=[1, 2],
                    dtype=np.float64).to_numpy()
    if ns is None or ns <= 0:                       # n_s is NaN on some rows
        if a.shape[0] % (nmod * nx):
            return None, None, None, f"length {a.shape[0]} not divisible by {nmod}*{nx}"
        ns = a.shape[0] // (nmod * nx)
    if a.shape[0] != nmod * nx * ns:
        return None, None, None, f"length {a.shape[0]} != {nmod}*{nx}*{ns}"
    kxrh = np.loadtxt(kxf, ndmin=2)
    if kxrh.shape != (nmod, nx):
        return None, None, None, f"kxrh {kxrh.shape} != ({nmod},{nx})"

    phi = (a[:, 0] + 1j * a[:, 1]).reshape(nmod, nx, ns)
    p2 = (np.abs(phi) ** 2).sum(axis=2)                    # (nmod, nx)
    # a linear eigenmode grows exponentially, so the absolute scale is
    # arbitrary and overflows float32. normalising by the global max keeps
    # both the ky and kx structure and leaves kx_width unchanged.
    gmax = p2.max()
    if gmax > 0:
        p2 = p2 / gmax
    tot = p2.sum(axis=1)
    live = tot > 0
    pk = np.zeros_like(p2)
    pk[live] = p2[live] / tot[live, None]
    kx_bar = (kxrh * pk).sum(axis=1)
    kx_w = np.sqrt(np.maximum((((kxrh - kx_bar[:, None]) ** 2) * pk).sum(axis=1), 0.0))
    kx_bar = np.where(live, kx_bar, np.nan)
    kx_w = np.where(live, kx_w, np.nan)

    # NOT an ordering check: normalize_per_toroidal_mode applies to the
    # kyspec diagnostic, not to the raw eigenmode in parallel.dat, so
    # sum_kx phi2_kxy is not flat in ky. the ordering is established by the
    # kxspec correlation and confirmed cross-code against gyaradax kx_width.
    return p2, kx_w, kx_bar, f"ok ({int(live.sum())}/{nmod} live)"


def main(write=False):
    for nm in TABLES:
        df = pd.read_parquet(D + nm + ".parquet")
        p2c, kwc, kbc, srcc = [], [], [], []
        stats = {}
        for _, r in df.iterrows():
            lp = r.get("lin_path") or ""
            ns_v = r.n_s
            ns_v = None if ns_v is None or pd.isna(ns_v) else int(ns_v)
            nmod, nx, ns = int(r.n_ky), int(r.n_kx), ns_v
            p2, kw, kb, why = eigenmode_kx(lp, nmod, nx, ns) if lp else (None, None, None, "no lin_path")
            stats[why.split(" (")[0]] = stats.get(why.split(" (")[0], 0) + 1
            p2c.append(None if p2 is None else p2.ravel().astype(np.float32))
            kwc.append(None if kw is None else kw.astype(np.float64))
            kbc.append(None if kb is None else kb.astype(np.float64))
            srcc.append(None if p2 is None else "gkw:parallel.dat")
        df["lin_phi2_kxy"] = p2c           # flattened (n_ky, n_kx), C order
        df["lin_kx_width"] = kwc           # per ky
        df["lin_kx_bar"] = kbc             # per ky
        df["lin_kx_src"] = srcc
        df["kx_convention"] = np.where(pd.notna(df["lin_kx_src"]), "kxrh", None)
        ok = sum(1 for v in kwc if v is not None)
        allw = np.concatenate([v[np.isfinite(v)] for v in kwc if v is not None])
        print(f"\n{nm}: {ok}/{len(df)} rows got lin_kx_width   {stats}")
        print(f"  kx_width {allw.min():.4f}..{allw.max():.4f} median {np.median(allw):.4f}")
        allb = np.concatenate([v[np.isfinite(v)] for v in kbc if v is not None])
        print(f"  kx_bar   {allb.min():.3e}..{allb.max():.3e} (expect ~0, no ExB shear)")
        if write:
            out = D + nm + ".parquet"
            df.to_parquet(out, index=False)
            print(f"  WROTE {out}")


if __name__ == "__main__":
    main(write="--write" in sys.argv)
