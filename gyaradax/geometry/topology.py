"""Mode connectivity and parallel-boundary topology helpers.

These helpers are model-independent: analytic and loaded/reference geometry
paths both use them to build spectral mode connectivity and parallel stencil
shift maps.
"""

from __future__ import annotations

import numpy as np


def _build_mode_connectivity(mode_label, kxrh, krho):
    """Build spectral parallel-boundary connectivity from mode labels.

    Returns (mode_label_kxky, ixplus, ixminus, ixzero, iyzero, iyzero_bc).
    `ixplus`/`ixminus` use -1 to mark open boundaries. `iyzero_bc` is -1
    when ky=0 is absent so the periodic zonal treatment never applies to
    a non-zonal mode.
    """
    mode_label = np.atleast_1d(np.asarray(mode_label, dtype=np.int32))
    kxrh_np = np.atleast_1d(np.asarray(kxrh))
    krho_np = np.atleast_1d(np.asarray(krho))
    nkx = int(kxrh_np.shape[0])
    nky = int(krho_np.shape[0])

    if mode_label.shape == (nkx, nky):
        mode_label_kxky = mode_label
    elif mode_label.shape == (nky, nkx):
        mode_label_kxky = mode_label.T
    elif mode_label.size == nkx * nky:
        mode_label_kxky = mode_label.reshape(nkx, nky)
    else:
        raise ValueError(
            f"mode_label shape {mode_label.shape} incompatible with nkx/nky=({nkx},{nky})"
        )

    ixzero = int(np.argmin(np.abs(kxrh_np)))
    iyzero = int(np.argmin(np.abs(krho_np)))
    ky_is_truly_zonal = np.abs(krho_np[iyzero]) < 1e-10

    ixplus = -np.ones((nkx, nky), dtype=np.int32)
    ixminus = -np.ones((nkx, nky), dtype=np.int32)

    for iy in range(nky):
        if iy == iyzero and ky_is_truly_zonal:
            ix = np.arange(nkx, dtype=np.int32)
            ixplus[:, iy] = ix
            ixminus[:, iy] = ix
            continue

        labels = mode_label_kxky[:, iy]
        for lbl in np.unique(labels):
            chain = np.where(labels == lbl)[0].astype(np.int32)
            if chain.size <= 1:
                continue
            chain = np.sort(chain)
            ixplus[chain[:-1], iy] = chain[1:]
            ixminus[chain[1:], iy] = chain[:-1]

    iyzero_bc = iyzero if ky_is_truly_zonal else -1
    return mode_label_kxky, ixplus, ixminus, ixzero, iyzero, iyzero_bc


def _build_pos_par_grid_classes(ixplus, ixminus, ns):
    """Position class (-2,-1,0,1,2) for open parallel boundary handling.

    Shape: [ns, nkx, nky].
    """
    pos = np.zeros((ns,) + ixplus.shape, dtype=np.int8)
    left_open = ixminus < 0
    right_open = ixplus < 0

    if ns >= 1:
        pos[0, left_open] = -2
        pos[ns - 1, right_open] = 2
    if ns >= 2:
        pos[1, left_open] = -1
        pos[ns - 2, right_open] = 1

    return pos


def _build_parallel_shift_maps(ixplus, ixminus, iyzero, ns, max_shift=4):
    """Precompute parallel shift connectivity maps for s-stencil application.

    Returns arrays with shape [2*max_shift+1, ns, nkx, nky]: s_shift, kx_shift,
    valid. `valid` is False on out-of-grid shifts (open boundary).
    """
    nkx, nky = ixplus.shape
    nshifts = 2 * max_shift + 1

    s_shift = np.zeros((nshifts, ns, nkx, nky), dtype=np.int32)
    kx_shift = np.zeros((nshifts, ns, nkx, nky), dtype=np.int32)
    valid = np.zeros((nshifts, ns, nkx, nky), dtype=np.bool_)

    for shift_idx, delta_s in enumerate(range(-max_shift, max_shift + 1)):
        for s in range(ns):
            for kx in range(nkx):
                for ky in range(nky):
                    tgt_s = s + delta_s
                    tgt_kx = kx
                    ok = True

                    if tgt_s < 0:
                        if ky == iyzero:
                            tgt_s += ns
                        else:
                            kx_conn = ixminus[kx, ky]
                            if kx_conn >= 0:
                                tgt_kx = kx_conn
                                tgt_s += ns
                            else:
                                ok = False
                    elif tgt_s >= ns:
                        if ky == iyzero:
                            tgt_s -= ns
                        else:
                            kx_conn = ixplus[kx, ky]
                            if kx_conn >= 0:
                                tgt_kx = kx_conn
                                tgt_s -= ns
                            else:
                                ok = False

                    if ok and 0 <= tgt_s < ns:
                        s_shift[shift_idx, s, kx, ky] = tgt_s
                        kx_shift[shift_idx, s, kx, ky] = tgt_kx
                        valid[shift_idx, s, kx, ky] = True

    return s_shift, kx_shift, valid
