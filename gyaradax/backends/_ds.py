"""Double-single tables for the linear_rhs_ds CUDA kernel (``linear_arith="ds"``).

Every real coefficient table of the electrostatic linear RHS is stored as
interleaved float32 (hi, lo) pairs with hi + lo equal to the float64 value to
about 48 bits. The vpar stencil and drive coefficients are folded on the host
into ``ds_wvp``, ``ds_drive_a`` and ``ds_drive_b``.
"""

import jax
import jax.numpy as jnp

from gyaradax.operators import stencils

DS_KEYS = (
    "ds_bessel",
    "ds_s_upar_tab",
    "ds_s_t7_tab",
    "ds_wvp",
    "ds_drift_x",
    "ds_drift_y",
    "ds_drive_a",
    "ds_drive_b",
    "ds_hyper",
    "ds_kx_vals",
    "ds_ky_vals",
    "ds_coll_stencil",
)


def ds_pairs(x) -> jnp.ndarray:
    """Split float64 values into (..., 2) float32 (hi, lo) pairs."""
    x = jnp.asarray(x, dtype=jnp.float64)
    bits = jax.lax.bitcast_convert_type(x, jnp.uint64)
    # round to 24 significant bits on the integer bits
    hi64 = jax.lax.bitcast_convert_type(
        (bits + jnp.uint64(1 << 28)) & jnp.uint64(0xFFFFFFFFE0000000), jnp.float64
    )
    return jnp.stack([hi64.astype(jnp.float32), (x - hi64).astype(jnp.float32)], axis=-1)


def ds_tables(pre, disp_vp: float, drive_scale: float) -> dict:
    """Build the double-single coefficient tables from linear_precompute output."""
    kinetic = pre["s_t7_tab"].ndim == 6

    def sp6(x):
        x = jnp.asarray(x, dtype=jnp.float64)
        return x if kinetic else x[None]

    s_t7 = sp6(pre["s_t7_tab"])
    nsp_t, nv, nmu, ns, n_class, _ = s_t7.shape
    bessel = sp6(pre["bessel"])
    nsp = max(nsp_t, bessel.shape[0])
    nkx, nky = bessel.shape[-2:]

    def table(key, shape, out_shape):
        return jnp.broadcast_to(sp6(pre[key]), shape).reshape(out_shape)

    def per_species(key):
        return jnp.broadcast_to(jnp.asarray(pre[key], dtype=jnp.float64).reshape(-1), (nsp,))

    inv_dvp = 1.0 / float(pre["dvp"])
    d1 = jnp.asarray(stencils.VPAR_D1, dtype=jnp.float64)
    d4 = jnp.asarray(stencils.VPAR_D4, dtype=jnp.float64)
    mu_s = (nsp, 1, nmu, ns, 1, 1)
    v_mu_s = (nsp, nv, nmu, ns, 1, 1)
    utrap = table("utrap", mu_s, (nsp, nmu, ns, 1))
    abs_vp = table("abs_dum2_vp", mu_s, (nsp, nmu, ns, 1))
    wvp = utrap * inv_dvp * d1 + disp_vp * abs_vp * inv_dvp * d4

    fmaxwl = table("fmaxwl", v_mu_s, (nsp, nv, nmu, ns))
    signz0 = per_species("signz0")[:, None, None, None]
    tmp0 = per_species("tmp0")[:, None, None, None]
    dmaxwel = table("dmaxwel_fm_ek", (nsp, nv, nmu, ns, 1, nky), (nsp, nv, nmu, ns, nky))

    s_upar = sp6(pre["s_upar_tab"])
    if s_upar.shape[2] != 1:
        raise ValueError(f"linear_arith='ds' expects mu-independent s_upar_tab, got {s_upar.shape}")
    tables = {
        "ds_bessel": table("bessel", (nsp, 1, nmu, ns, nkx, nky), (nsp, nmu, ns, nkx, nky)),
        "ds_s_upar_tab": jnp.broadcast_to(s_upar, (nsp, nv, 1, ns, n_class, 9)).reshape(
            nsp, nv, ns, n_class, 9
        ),
        "ds_s_t7_tab": jnp.broadcast_to(s_t7, (nsp, nv, nmu, ns, n_class, 9)),
        "ds_wvp": wvp,
        "ds_drift_x": table("drift_x", v_mu_s, (nsp, nv, nmu, ns)),
        "ds_drift_y": table("drift_y", v_mu_s, (nsp, nv, nmu, ns)),
        "ds_drive_a": drive_scale * dmaxwel,
        "ds_drive_b": drive_scale * signz0 * fmaxwl * (1.0 / jnp.maximum(tmp0, 1e-15)),
        "ds_hyper": jnp.broadcast_to(pre["hyper"], (1, 1, 1, nkx, nky)).reshape(nkx, nky),
        "ds_kx_vals": jnp.asarray(pre["kx_b"]).reshape(-1)[:nkx],
        "ds_ky_vals": jnp.asarray(pre["ky_b"]).reshape(-1)[:nky],
    }
    if "coll_stencil" in pre:
        coll = sp6(pre["coll_stencil"])
        tables["ds_coll_stencil"] = jnp.broadcast_to(coll, (nsp, 9, nv, nmu, ns))
    return {k: ds_pairs(v) for k, v in tables.items()}
