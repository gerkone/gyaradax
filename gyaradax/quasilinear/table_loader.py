"""Rebuild saturation-rule inputs from the raw summary-table columns.

The tables store what the linear solve produced and nothing derived, so the
deriving happens here. Two conventions matter and both have bitten us:

  krho = lin_ky / kthnorm. little_g carries kthnorm^2 in g_zz, so pairing it
  with the stored k_theta*rho axis double-counts by g_zz (0.5 to 38).

  <k_perp^2> is evaluated on the kx = 0 chain. Every theta0 chain is
  unit-normalised, so an all-chain average weights the large-kx chains
  equally with the physical theta0 = 0 mode; the result then scales with
  ikxspace (adiabatic 1.39 at ikxspace 5-8 against 2.03 at 9, kinetic 4.9 to
  12.6). W(ky) is a ratio and largely cancels that, so it does not need the
  mask -- an average does.
"""

import numpy as np


def _arr(row, key):
    v = row.get(key)
    if v is None or np.size(v) == 0:
        return None
    a = np.asarray(v, dtype=np.float64).ravel()
    shape = row.get(f"{key}_shape")
    return a.reshape([int(x) for x in np.asarray(shape).ravel()]) if shape is not None else a


def chain0_mask(n_kx, n_ky, ikxspace, ixzero):
    """(n_kx, n_ky) mask of the theta0 = 0 chain, per the table README."""
    i = np.arange(n_kx)[:, None]
    j = np.arange(n_ky)[None, :]
    step = np.maximum(ikxspace * j, 1)
    m = ((i - ixzero) % step) == 0
    m[:, 0] = False                      # the zonal mode is not a chain
    return m


def kperp2_eff(row, chain0=True):
    """<k_perp^2>(ky), eigenmode-weighted, on the kx = 0 chain by default."""
    phi = _arr(row, "lin_phi")           # (2, ns, nkx, nky)
    lg = _arr(row, "lin_little_g")       # (3, ns)
    kx = _arr(row, "lin_kxrh")
    ky = np.asarray(row["lin_ky"], float)
    if phi is None or lg is None or kx is None:
        return None
    kth = float(row["kthnorm"])
    krho = ky / kth                      # the pairing little_g expects
    phi2 = phi[0] ** 2 + phi[1] ** 2     # (ns, nkx, nky)
    ints = _arr(row, "lin_ints")
    w_s = (ints / ints.sum())[:, None, None] if ints is not None else 1.0 / phi2.shape[0]

    kp2 = (krho[None, None, :] ** 2 * lg[0][:, None, None]
           + 2.0 * krho[None, None, :] * kx[None, :, None] * lg[1][:, None, None]
           + kx[None, :, None] ** 2 * lg[2][:, None, None])

    sel = 1.0
    if chain0:
        ik = row.get("ikxspace")
        if ik is None or not np.isfinite(ik):
            return None
        sel = chain0_mask(phi2.shape[1], phi2.shape[2],
                          int(ik), int(np.argmin(np.abs(kx))))[None, :, :]
    wphi = phi2 * w_s * sel
    norm = wphi.sum(axis=(0, 1))
    return np.where(norm > 0, (kp2 * wphi).sum(axis=(0, 1)) / np.maximum(norm, 1e-300), np.nan)


def chain0_degenerate_from(row):
    """First ky index where the chain0 mask holds only kx = 0 (no kx info)."""
    ik = row.get("ikxspace")
    if ik is None or not np.isfinite(ik):
        return None
    return int((int(row["n_kx"]) // 2) / int(ik)) + 1


def rule_inputs(row, channel="Qi", chain0_kperp2=True, chain0_weight=False):
    """(ky, gamma, W, <k_perp^2>) for one row, or None if unusable."""
    ky = np.asarray(row["lin_ky"], float)
    g = np.asarray(row["lin_gamma"], float)
    wkey = f"lin_weight_{channel}" + ("_chain0" if chain0_weight else "")
    w = row.get(wkey)
    if w is None or np.size(w) == 0:
        return None
    w = np.asarray(w, float)
    kp = kperp2_eff(row, chain0=chain0_kperp2)
    if kp is None:
        return None
    n = min(len(ky), len(g), len(w), len(kp))
    ky, g, w, kp = ky[:n], g[:n], w[:n], kp[:n]
    m = (np.isfinite(ky) & (ky > 0) & np.isfinite(g) & np.isfinite(w)
         & np.isfinite(kp) & (kp > 0))
    if m.sum() < 4 or not (g[m] > 0).any():
        return None
    o = np.argsort(ky[m])
    return ky[m][o], np.maximum(g[m], 0.0)[o], w[m][o], kp[m][o]
