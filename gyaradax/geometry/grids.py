"""Model-independent geometry grid and mode-label helpers.

These helpers build the parallel, velocity, wavevector, and mode-label grids
shared by analytic geometry construction. They intentionally do not contain
continuous geometry formulas, tensor assembly, or parallel-boundary topology.
"""

from __future__ import annotations

from typing import Any

import jax.numpy as jnp
import numpy as np


def _parallel_grid(ns, nperiod):
    """Cell-centered uniform parallel grid on [-sgrmax, sgrmax]."""
    sgrmax = nperiod - 0.5
    return jnp.array([-sgrmax + 2 * sgrmax * (i + 0.5) / ns for i in range(ns)])


def _parallel_weights(sgrid):
    """Uniform integration weights (cell width)."""
    if len(sgrid) < 2:
        return jnp.ones(1)
    return jnp.full(len(sgrid), sgrid[1] - sgrid[0])


def _build_velocity_grids(nvpar, nmu, vpar_max):
    """Uniform v_par grid and uniform-in-v_perp mu grid (GKW convention)."""
    dvp = 2 * vpar_max / nvpar
    vpgr = jnp.linspace(-vpar_max + dvp / 2, vpar_max - dvp / 2, nvpar)
    dvperp = vpar_max / nmu
    vperp = jnp.linspace(dvperp / 2, vpar_max - dvperp / 2, nmu)
    return vpgr, vperp**2 / 2, jnp.full(nvpar, dvp), 2 * jnp.pi * vperp * dvperp


def _build_wavevector_grids(
    nkx, nky, kxmax, krhomax, q=1.0, shat=0.0, eps=0.1, ikxspace=5, kthnorm: Any = 1.0
):
    """Centered kx grid and uniform ky grid.

    For nky=1 the single mode sits at krhomax. For nky>1 the kx spacing
    follows the shear connectivity: kxspace = |q*shat*krho[1]/(eps*ikxspace)|
    (GKW mode.f90:698).
    """
    if nky == 1:
        half = (nkx - 1) // 2
        dkx = kxmax / half if half > 0 else 0.0
        return jnp.arange(-half, half + 1) * dkx, jnp.array([krhomax])

    dky = krhomax / (nky - 1)
    krho_norm = jnp.arange(nky) * dky / kthnorm

    half = (nkx - 1) // 2
    # branch on static shape only; the (q, shat, eps)-dependent choice is traced
    # via jnp.where so the function stays jit/AD-safe when those are tracers
    # (the torax-plugin path jits over q, shat, eps).
    if half == 0:
        kxspace = jnp.asarray(0.0)
    else:
        eps_safe = jnp.where(jnp.abs(eps) > 1e-30, eps, 1.0)
        shat_safe = jnp.where(jnp.abs(shat) > 1e-30, shat, 1.0)
        kxspace_shear = jnp.abs(q * shat_safe * krho_norm[1] / (eps_safe * ikxspace))
        use_shear = (jnp.abs(shat) > 1e-10) & (eps > 1e-10)
        kxspace = jnp.where(use_shear, kxspace_shear, kxmax / half)

    kxrh = jnp.arange(-half, half + 1) * kxspace
    return kxrh, jnp.arange(nky) * dky


def _build_mode_label(nkx, nky, ikxspace):
    """Mode-label array for open parallel boundary connectivity.

    ky=0: each kx is its own mode (periodic). ky>0: modes grouped into
    chains spaced ikxspace apart in kx-index.
    """
    ml = np.zeros((nkx, nky), dtype=np.int32)
    label = 1
    for ix in range(nkx):
        ml[ix, 0] = label
        label += 1
    for iy in range(1, nky):
        for offset in range(ikxspace):
            lbl = label
            label += 1
            for ix in range(offset, nkx, ikxspace):
                ml[ix, iy] = lbl
    return ml
