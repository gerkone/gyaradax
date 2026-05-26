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


def get_device_count() -> int:
    """Return the number of available GPU devices."""
    return len(jax.devices("gpu"))


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
    # fused-stencil arrays from _fuse_stencils: 6D adiabatic (9, vp, mu, s, kx, ky)
    # and 7D kinetic (9, sp, vp, mu, s, kx, ky), with broadcast singletons allowed
    # on mu/kx/ky (mu becomes 1 after jnp.sign(upar)).
    if len(s) == 6 and s[0] == 9 and s[1] == grid.nvpar:
        return PartitionSpec(
            None, _AXIS_VP, _AXIS_MU if s[2] == grid.nmu else None, None, None, None
        )
    if len(s) == 7 and s[0] == 9 and s[1] == grid.nsp and s[2] == grid.nvpar:
        return PartitionSpec(
            None,
            _AXIS_SP,
            _AXIS_VP,
            _AXIS_MU if s[3] == grid.nmu else None,
            None,
            None,
            None,
        )
    # velocity-broadcast pre arrays produced by _compute_species_coeffs often
    # have the full 5D shape; handled above. Handle collapsed velocity shapes
    # that retain nvpar/nmu as leading dims with broadcast singletons for s,
    # kx, ky (shape (nvpar, nmu, 1, 1, 1), etc.) — these are reshapes of the
    # per-species arrays, same vp/mu sharding.
    if len(s) == 5 and s[0] == grid.nvpar and s[1] == grid.nmu:
        return PartitionSpec(_AXIS_VP, _AXIS_MU, None, None, None)
    if len(s) == 6 and s[0] == grid.nsp and s[1] == grid.nvpar and s[2] == grid.nmu:
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


def precompute_sharded(geometry, params, mesh: Optional[Mesh], grid: GridShape):
    """Build the ``pre`` pytree directly sharded across the mesh.

    Works because ``GKPre`` cleanly separates JAX-array leaves from Python
    aux (``nl_m*``, ``nsp``, ``ixzero``, ``iyzero``), so ``jax.eval_shape``
    returns a pytree whose leaves are all ``ShapeDtypeStruct`` —
    ``out_shardings`` only needs to map those leaves. XLA's GSPMD then
    partitions the entire construction; full-size intermediates never
    materialise on a single device.

    On single-device (mesh=None) this falls through to the normal
    ``linear_precompute`` path.
    """
    from gyaradax.solver import linear_precompute
    import jax.numpy as jnp

    if mesh is None:
        return linear_precompute(geometry, params)

    replicated = NamedSharding(mesh, PartitionSpec())

    def _replicate(x):
        if isinstance(x, jnp.ndarray) and x.ndim > 0:
            return jax.device_put(x, replicated)
        return x

    # Extract 0-D int scalars from geometry (e.g. ixzero, iyzero). These get
    # closed over so ``linear_precompute`` sees them as literal Python ints
    # at trace time (it calls .item() on them). Passing them as jit args
    # would turn them into tracers and break the .item() call.
    int_scalars = {}
    geom_rep = {}
    for k, v in geometry.items():
        if isinstance(v, jax.Array) and v.ndim == 0 and jnp.issubdtype(v.dtype, jnp.integer):
            int_scalars[k] = int(v)
        else:
            geom_rep[k] = _replicate(v)
    params_rep = jax.tree_util.tree_map(_replicate, params)

    # Return pre._items (plain dict) so the jit output pytree is a clean
    # dict/array structure — GKPre's custom flatten routes non-array leaves
    # into aux, which trips up tree_map(_leaf_sharding, ...).
    # Use _linear_precompute_core to avoid auto-sharding recursion.
    def _wrapped(geom, p):
        from gyaradax.solver import _linear_precompute_core

        pre = _linear_precompute_core({**geom, **int_scalars}, p)
        return pre._items

    shapes = jax.eval_shape(_wrapped, geom_rep, params_rep)

    def _leaf_sharding(leaf):
        if hasattr(leaf, "shape") and hasattr(leaf, "dtype") and len(leaf.shape) > 0:
            return NamedSharding(mesh, _spec_for_shape(leaf.shape, grid))
        return None

    out_shardings = jax.tree_util.tree_map(_leaf_sharding, shapes)
    result_dict = jax.jit(_wrapped, out_shardings=out_shardings)(geom_rep, params_rep)

    from gyaradax.state import GKPre

    return GKPre(result_dict)


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


def init_f_sharded(
    geometry,
    params=None,
    mesh: Mesh = None,
    grid: GridShape = None,
    finit: str = "cosine2",
    amp_init_real: float = 1.0e-4,
    amp_init_imag: float = 0.0,
    n_species: int = 1,
    seed: int = 42,
):
    """Initialize distribution function directly sharded across the mesh.

    DEPRECATED: This is now a thin wrapper around solver.init_f.
    Use solver.init_f with params for automatic sharding detection.

    Returns sharded df with the same values as init_f would produce.
    """
    from jax.sharding import NamedSharding, PartitionSpec
    from gyaradax.solver import init_f

    # Build output sharding from mesh and grid
    if n_species > 1:
        spec = PartitionSpec(_AXIS_SP, _AXIS_VP, _AXIS_MU, None, None, None)
    else:
        spec = PartitionSpec(_AXIS_VP, _AXIS_MU, None, None, None)
    out_sharding = NamedSharding(mesh, spec)

    # Delegate to unified init_f
    return init_f(
        geometry,
        finit=finit,
        amp_init_real=amp_init_real,
        amp_init_imag=amp_init_imag,
        n_species=n_species,
        seed=seed,
        out_sharding=out_sharding,
    )
