"""Linear timestep cap from the largest-|lambda| eigenvalues of the linear operator (``dt_eig_cfl``).

The per-term estimate in :func:`gyaradax.solver.cfl.estimate_linear_timestep` follows GKW's
``get_estimated_timestep`` and misses fast modes it has no term for, such as the zonal
electrostatic Alfven oscillation. This measures the operator's spectral radius on the ky blocks
where that mode sits and caps dt at ``params.dt_eig_cfl / max|lambda|``.
"""

from __future__ import annotations

import dataclasses
import time
from typing import Any, Dict, Sequence

import numpy as np

# zonal block and the first non-zonal one
EIG_KY_BLOCKS = (0, 1)


def eig_dt_cap(geometry: Dict[str, Any], params: Any, ky_blocks: Sequence[int] = EIG_KY_BLOCKS,
               tol: float = 1e-3) -> Dict[str, Any]:
    """Largest |lambda| per ky block and the resulting cap ``dt_eig_cfl / max|lambda|``."""
    from gyaradax.operators.precompute import linear_precompute
    from gyaradax.solver.eigenvalue import eigensolve_linear

    lin = dataclasses.replace(params, non_linear=False, n_gpus_sp=1, n_gpus_vp=1, n_gpus_mu=1)
    pre = linear_precompute(geometry, lin)
    nky = int(np.shape(geometry["krho"])[0])
    t0 = time.time()
    blocks = {}
    for iky in ky_blocks:
        if iky >= nky:
            continue
        lam, _ = eigensolve_linear(geometry, lin, pre=pre, k=1, which="LM", mode="rhs", tol=tol,
                                   ky_select=int(iky), refine=False)
        top = complex(np.asarray(lam)[np.argmax(np.abs(lam))])
        blocks[int(iky)] = top
    lam_max = max(abs(v) for v in blocks.values())
    return {"dt_cap": float(params.dt_eig_cfl) / lam_max, "lam_max": lam_max,
            "blocks": {k: [v.real, v.imag] for k, v in blocks.items()}, "seconds": time.time() - t0}
