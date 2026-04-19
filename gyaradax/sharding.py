"""Multi-GPU grid parallelism via JAX GSPMD.

All sharding-specific logic lives here. The rest of the codebase
(solver, backends, integrals, collisions, simulate) is unaware of the
mesh — operations on `jax.Array`s with `NamedSharding` are partitioned
automatically by XLA's GSPMD under `jit`.

Public API
----------
- ``build_mesh(params)`` — mesh from GKParams, or None on single-device.
- ``shard_df(df, mesh, grid_shape)`` — place the distribution function.
- ``shard_pre(pre, mesh, grid_shape)`` — place the precompute pytree.
- ``is_active(mesh)`` — helper for single-device early return.

Single-device path: all helpers are idempotent no-ops when
``n_gpus_sp * n_gpus_vp * n_gpus_mu == 1``.

Partition convention (velocity-space sharding on the adiabatic path):

    (nvpar,  nmu,  ns,  nkx,  nky)            5D: ("vp", "mu", None, None, None)
    (nsp,   nvpar, nmu, ns,  nkx,  nky)      6D: ("sp", "vp", "mu", None, None, None)
    (ns,    nkx,   nky)                       field: replicated
    (9,     nvpar, nmu, ns)                   coll 5D: (None, "vp", "mu", None)
    (nsp,   9,     nvpar, nmu, ns)            coll 6D: (None, None, "vp", "mu", None)

Arrays whose shape doesn't match any of the above stay replicated.
"""

from typing import Any, Dict, NamedTuple, Optional

import jax
from jax.sharding import Mesh, NamedSharding, PartitionSpec


class GridShape(NamedTuple):
    """Grid dimensions used to classify arrays by shape match."""

    nsp: int
    nvpar: int
    nmu: int
    ns: int
    nkx: int
    nky: int


_AXIS_SP = "sp"
_AXIS_VP = "vp"
_AXIS_MU = "mu"


def build_mesh(params) -> Optional[Mesh]:
    """Build a 3-axis device mesh from ``params.n_gpus_{sp,vp,mu}``.

    Returns None when all three are 1 (single-device path). Errors out
    when the requested mesh size exceeds the visible JAX devices.
    """
    p_sp = int(getattr(params, "n_gpus_sp", 1))
    p_vp = int(getattr(params, "n_gpus_vp", 1))
    p_mu = int(getattr(params, "n_gpus_mu", 1))
    total = p_sp * p_vp * p_mu
    if total == 1:
        return None
    devices = jax.devices()
    if total > len(devices):
        raise RuntimeError(
            f"requested {total} devices ({p_sp}×{p_vp}×{p_mu}) but "
            f"only {len(devices)} are visible to JAX"
        )
    import numpy as np
    dev_np = np.asarray(devices[:total], dtype=object).reshape((p_sp, p_vp, p_mu))
    return Mesh(dev_np, (_AXIS_SP, _AXIS_VP, _AXIS_MU))


def is_active(mesh: Optional[Mesh]) -> bool:
    """True when a non-trivial (multi-device) mesh is configured."""
    return mesh is not None


def _spec_for_shape(shape, grid: GridShape) -> PartitionSpec:
    """Classify an array by shape and return its partition spec.

    Unmatched shapes fall back to replicated. The check is strictly
    prefix-based: an array's leading dims must match one of the known
    grid axis signatures.
    """
    s = tuple(shape)
    if s == (grid.nvpar, grid.nmu, grid.ns, grid.nkx, grid.nky):
        return PartitionSpec(_AXIS_VP, _AXIS_MU, None, None, None)
    if s == (grid.nsp, grid.nvpar, grid.nmu, grid.ns, grid.nkx, grid.nky):
        return PartitionSpec(_AXIS_SP, _AXIS_VP, _AXIS_MU, None, None, None)
    if len(s) >= 3 and s[:3] == (grid.ns, grid.nkx, grid.nky):
        return PartitionSpec()
    if len(s) == 4 and s == (9, grid.nvpar, grid.nmu, grid.ns):
        return PartitionSpec(None, _AXIS_VP, _AXIS_MU, None)
    if len(s) == 5 and s == (grid.nsp, 9, grid.nvpar, grid.nmu, grid.ns):
        return PartitionSpec(None, None, _AXIS_VP, _AXIS_MU, None)
    # velocity-broadcast pre arrays produced by _compute_species_coeffs often
    # have the full 5D shape; handled above. Handle collapsed velocity shapes
    # that retain nvpar/nmu as leading dims with broadcast singletons for s,
    # kx, ky (shape (nvpar, nmu, 1, 1, 1), etc.) — these are reshapes of the
    # per-species arrays, same vp/mu sharding.
    if (
        len(s) == 5
        and s[0] == grid.nvpar
        and s[1] == grid.nmu
    ):
        return PartitionSpec(_AXIS_VP, _AXIS_MU, None, None, None)
    if (
        len(s) == 6
        and s[0] == grid.nsp
        and s[1] == grid.nvpar
        and s[2] == grid.nmu
    ):
        return PartitionSpec(_AXIS_SP, _AXIS_VP, _AXIS_MU, None, None, None)
    return PartitionSpec()


def _place(array, spec: PartitionSpec, mesh: Mesh):
    """Put an array onto devices with the given spec. Source buffer is
    released by Python GC once the caller stops referencing it; explicit
    ``.delete()`` is unsafe because some ``pre`` leaves alias geometry
    arrays still in use downstream."""
    return jax.device_put(array, NamedSharding(mesh, spec))


def shard_df(df, mesh: Optional[Mesh], grid: GridShape):
    """Place the distribution function onto the device mesh.

    Dispatches on ``df.ndim``: 5D (adiabatic) or 6D (kinetic).
    """
    if mesh is None:
        return df
    spec = _spec_for_shape(df.shape, grid)
    return _place(df, spec, mesh)


def shard_pre(pre: Dict[str, Any], mesh: Optional[Mesh], grid: GridShape) -> Dict[str, Any]:
    """Place each ``pre`` leaf per shape (build-then-shard fallback path).

    Prefer :func:`precompute_sharded` for large grids — it partitions the
    construction itself and never materialises the full array anywhere.
    """
    if mesh is None:
        return pre

    def _maybe_shard(leaf):
        if not hasattr(leaf, "shape") or len(leaf.shape) == 0:
            return leaf
        spec = _spec_for_shape(leaf.shape, grid)
        return _place(leaf, spec, mesh)

    return jax.tree_util.tree_map(_maybe_shard, pre)


# NOTE: option C (jit-wrapped `linear_precompute` with ``out_shardings``) is
# the scalable path for grids that exceed single-GPU memory — XLA's GSPMD
# partitions the entire construction so no full-size array materialises on
# any device. A prototype was attempted here; it needs additional work
# because ``linear_precompute`` returns a mixed pytree with Python ints
# (``nl_mrad``, ``nl_mphi``, ``nl_mphiw3``, ``nsp``) alongside JAX arrays,
# and ``jax.eval_shape`` returns those ints unchanged while
# ``jit(..., out_shardings=tree)`` expects the sharding pytree to match the
# output tree structure exactly — including non-array leaves.
#
# Two viable paths to finish it:
#   (1) wrap the return of ``linear_precompute`` to convert all ints into
#       0-D JAX scalars (so ``eval_shape`` yields a uniform pytree);
#   (2) partition the output dict into "array leaves" vs "aux" sub-trees and
#       only pass the array sub-tree through ``jit(out_shardings=…)``.
# Neither is implemented here. For grids that fit on a single GPU, the
# build-then-shard path in ``shard_pre`` (with ``.delete()`` of the source
# buffer) is sufficient and already correct.


def grid_shape_from(params, geometry) -> GridShape:
    """Infer grid dimensions for the classifier."""
    import jax.numpy as jnp

    nvpar = int(jnp.asarray(geometry["vpgr"]).shape[0])
    nmu = int(jnp.asarray(geometry["mugr"]).shape[0])
    ns = int(jnp.asarray(geometry["ints"]).shape[0])
    nkx = int(jnp.asarray(geometry["kxrh"]).shape[0])
    nky = int(jnp.asarray(geometry["krho"]).shape[0])
    if getattr(params, "adiabatic_electrons", True):
        nsp = 1
    else:
        mas = getattr(params, "mas", 1.0)
        nsp = int(jnp.atleast_1d(jnp.asarray(mas)).shape[0])
    return GridShape(nsp=nsp, nvpar=nvpar, nmu=nmu, ns=ns, nkx=nkx, nky=nky)
