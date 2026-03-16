"""
Gyrokinetic Vlasov-Poisson solver for the local flux-tube limit.

Supports both adiabatic-electron (single species) and kinetic-electron
(multi-species) configurations.

Implemented Equations:
The solver evolves the perturbed distribution function `f` in phase space.
Adiabatic: (vpar, mu, s, kx, ky).  Kinetic: (nsp, vpar, mu, s, kx, ky).

Active RHS Terms from the GKW formulation:
1. Term I   — Parallel Advection: v_par nabla_par f
2. Term II  — Drift Advection: v_d . nabla_perp f
3. Term III — Nonlinear ExB Advection: v_E . nabla_perp f (pseudospectral)
4. Term IV  — Trapping/Mirror: parallel velocity space advection
5. Term V   — Equilibrium Drive: v_E . nabla F_M
6. Term VII — Parallel Field Drive: v_par nabla_par phi coupling
7. Term VIII— Drift Field Drive: v_d . nabla phi coupling

Dissipation: parallel (4th order), velocity space, perpendicular hyper-diffusion.

Time Integration: Explicit RK4 with optional per-ky normalization (linear mode).
"""

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_compilation_cache_dir", "/tmp/jax_cache")

import math
import functools
from typing import Dict, Tuple, Optional, Any
from dataclasses import dataclass

import jax.random
from gyaradax.integrals import (
    get_integrals,
    j0,
    geom_tensors,
    calculate_phi,
    calculate_phi_kinetic,
)
from gyaradax import stencils
from gyaradax.params import GKParams
from einops import rearrange


@jax.tree_util.register_pytree_node_class
class GKPre:
    """precomputed terms container. separates dynamic arrays (leaves) from
    static metadata (auxiliary) so FFT sizes stay concrete under JIT."""

    def __init__(self, items: Dict[str, Any]):
        self._items = items

    def tree_flatten(self):
        leaves = []
        leaf_keys = []
        aux = {}
        for k, v in self._items.items():
            if k.startswith("nl_m") or k in ("ixzero", "iyzero", "nsp"):
                aux[k] = v
            elif isinstance(v, (jnp.ndarray, float, int, bool)):
                leaves.append(v)
                leaf_keys.append(k)
            else:
                aux[k] = v
        return tuple(leaves), {"leaf_keys": tuple(leaf_keys), "aux": aux}

    @classmethod
    def tree_unflatten(cls, metadata, leaves):
        items = dict(zip(metadata["leaf_keys"], leaves))
        items.update(metadata["aux"])
        return cls(items)

    def __getitem__(self, key):
        return self._items[key]

    def get(self, key, default=None):
        return self._items.get(key, default)

    def items(self):
        return self._items.items()

    def keys(self):
        return self._items.keys()


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class GKState:
    """Diagnostic state for large-step growth tracking and normalization."""

    time: jnp.ndarray
    step: jnp.ndarray
    accumulated_norm_factor: jnp.ndarray
    window_start_amp: jnp.ndarray
    last_growth_rate: jnp.ndarray

    def tree_flatten(self):
        return tuple(vars(self).values()), None

    @classmethod
    def tree_unflatten(cls, aux_data, leaves):
        return cls(*leaves)


def default_state(nky: int = 1) -> GKState:
    return GKState(
        time=jnp.array(0.0, dtype=jnp.float64),
        step=jnp.array(0, dtype=jnp.int32),
        accumulated_norm_factor=jnp.ones(nky, dtype=jnp.float64),
        window_start_amp=jnp.ones(nky, dtype=jnp.float64),
        last_growth_rate=jnp.zeros(nky, dtype=jnp.float64),
    )


def kx_ky_grids(geometry: Dict[str, jnp.ndarray]) -> Tuple[jnp.ndarray, jnp.ndarray]:
    kx = jnp.asarray(geometry["kxrh"], dtype=jnp.float64)
    ky = jnp.asarray(geometry["krho"], dtype=jnp.float64)
    if kx.ndim == 2:
        kx = kx[0]
    if ky.ndim == 2:
        ky = ky[:, 0]
    return kx, ky


def mode_amplitude(
    phi: jnp.ndarray, geometry: Dict[str, jnp.ndarray], eps: float
) -> jnp.ndarray:
    ints = jnp.asarray(geometry["ints"], dtype=jnp.float64)
    ds = ints[0]
    amp2 = ds * jnp.sum(jnp.abs(phi) ** 2, axis=(0, 1))
    return jnp.sqrt(jnp.maximum(amp2, eps))


def normalize_per_ky(
    df: jnp.ndarray, geometry: Dict[str, jnp.ndarray], params: GKParams
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    phi = calculate_phi(geom_tensors(geometry, params=params), df)
    amp_per_ky = mode_amplitude(phi, geometry, params.norm_eps)
    safe_amp = jnp.where(amp_per_ky < params.norm_eps, 1.0, amp_per_ky)
    inv = 1.0 / safe_amp
    inv_shape = (1,) * (df.ndim - 1) + (-1,)
    return df * jnp.reshape(inv, inv_shape), inv, safe_amp


def prime_factors_smallereq_than(number: int, max_prime: int) -> bool:
    i = 2
    n = int(number)
    while True:
        if n % i == 0:
            n //= i
        elif i == max_prime:
            return n == 1
        else:
            i += 1


def extended_firstdim_fft_size(nmod: int) -> Tuple[int, int]:
    posspace_size = 3 * nmod - 2
    if posspace_size % 2 != 0:
        posspace_size += 1
    while not prime_factors_smallereq_than(posspace_size, 7):
        posspace_size += 2
    for i in range(1, 9):
        cand = posspace_size + 2 * i
        if prime_factors_smallereq_than(cand, 2):
            posspace_size = cand
            break
    return posspace_size, int(math.floor(posspace_size / 2.0) + 1)


def extended_seconddim_fft_size(nx: int) -> int:
    dum = int(math.ceil(1.5 * float(nx + 1)) + 1)
    while not prime_factors_smallereq_than(dum, 7):
        dum += 1
    for i in range(1, 9):
        cand = dum + i
        if prime_factors_smallereq_than(cand, 2):
            dum = cand
            break
    return dum


def build_jind(nkx: int, mrad: int, ixzero: int) -> jnp.ndarray:
    ix = jnp.arange(nkx, dtype=jnp.int32)
    return jnp.where(ix >= ixzero, ix - ixzero, mrad + ix - ixzero)


def pack_half_spectrum(
    spec_kxky: jnp.ndarray, jind: jnp.ndarray, mrad: int, mphiw3: int
) -> jnp.ndarray:
    out_shape = spec_kxky.shape[:-2] + (mrad, mphiw3)
    out = jnp.zeros(out_shape, dtype=jnp.complex128)
    nky = spec_kxky.shape[-1]
    return out.at[..., jind, :nky].set(spec_kxky)


def unpack_half_spectrum(
    spec_half: jnp.ndarray, jind: jnp.ndarray, nky: int
) -> jnp.ndarray:
    return spec_half[..., jind, :nky]


def nonlinear_term_iii(
    df: jnp.ndarray,
    phi: jnp.ndarray,
    geometry: Dict[str, jnp.ndarray],
    pre: Dict[str, jnp.ndarray],
    *,
    efun_sign: float = 1.0,
    fft_prefactor: complex = 1.0 + 0.0j,
    exclude_zero_mode: bool = True,
    mixed_precision: bool = True,
) -> jnp.ndarray:
    """Nonlinear ExB advection via pseudospectral method. df is 5D."""
    mrad, mphi, mphiw3 = pre["nl_mrad"], pre["nl_mphi"], pre["nl_mphiw3"]
    fft_scale, jind = pre["nl_fft_scale"], pre["nl_jind"]
    kx2d, ky2d, bessel = pre["nl_kx2d"], pre["nl_ky2d"], pre["bessel"]
    dum_s, ixzero, iyzero = pre["nl_dum_s"], pre["ixzero"], pre["iyzero"]
    nky = df.shape[-1]

    df_by_s = jnp.moveaxis(df, 2, 0)
    bessel_by_s = jnp.moveaxis(bessel, 2, 0)

    def _per_s(df_s, phi_s, bessel_s, dum):
        gyro_phi = bessel_s * phi_s[None, None, :, :]
        grad_phi_y_k = 1j * ky2d[None, None, :, :] * gyro_phi
        grad_phi_x_k = 1j * kx2d[None, None, :, :] * gyro_phi
        grad_f_x_k = 1j * kx2d[None, None, :, :] * df_s
        grad_f_y_k = 1j * ky2d[None, None, :, :] * df_s

        fft_dtype = jnp.complex64 if mixed_precision else jnp.complex128
        real_dtype = jnp.float32 if mixed_precision else jnp.float64

        def _to_real(spec):
            packed = pack_half_spectrum(spec, jind, mrad, mphiw3).astype(fft_dtype)
            return jnp.fft.irfft2(
                packed, s=(mrad, mphi), axes=(-2, -1), norm="backward"
            )

        nl_real = (efun_sign * dum).astype(real_dtype) * (
            _to_real(grad_phi_y_k) * _to_real(grad_f_x_k)
            - _to_real(grad_phi_x_k) * _to_real(grad_f_y_k)
        )
        nl_half = (
            jnp.asarray(fft_prefactor, dtype=jnp.complex128)
            * jnp.asarray(fft_scale, dtype=jnp.complex128)
            * jnp.fft.rfft2(
                nl_real.astype(jnp.float64),
                s=(mrad, mphi),
                axes=(-2, -1),
                norm="backward",
            )
        )
        return unpack_half_spectrum(nl_half, jind, nky)

    nl = jnp.moveaxis(jax.vmap(_per_s)(df_by_s, phi, bessel_by_s, dum_s), 0, 2)
    return nl.at[:, :, :, ixzero, iyzero].set(0.0) if exclude_zero_mode else nl


def estimate_nl_timestep(
    phi: jnp.ndarray,
    pre: Dict[str, jnp.ndarray],
    bessel: jnp.ndarray,
    dt_input: float,
    safety_factor: float = 0.95,
) -> jnp.ndarray:
    """CFL-adaptive timestep estimate from the nonlinear ExB velocity.

    Computes max|grad phi| in real space (dealiased grid) and returns
    dt_est = safety_factor * 2 / max_value, clamped to dt_input.

    Matches GKW's spectral CFL: non_linear_terms.F90 lines 1530-1777.

    Args:
        phi: Electrostatic potential (ns, nkx, nky).
        pre: Precomputed dict with FFT metadata and Bessel functions.
        bessel: Bessel J0 array for gyro-averaging phi. For adiabatic this
            is pre["bessel"] (nv, nmu, ns, nkx, nky). For kinetic, pass
            the ion Bessel (first species) since it has the largest FLR.
        dt_input: Maximum allowed timestep.
        safety_factor: CFL safety factor (default 0.95).

    Returns:
        Scalar dt estimate.
    """
    mrad, mphi, mphiw3 = pre["nl_mrad"], pre["nl_mphi"], pre["nl_mphiw3"]
    jind = pre["nl_jind"]
    kx2d, ky2d = pre["nl_kx2d"], pre["nl_ky2d"]

    bessel_s0 = bessel[0, 0, :, :, :]  # (ns, nkx, nky)

    def _max_grad_per_s(phi_s, bes_s):
        gyro_phi = bes_s * phi_s
        grad_y_k = 1j * ky2d * gyro_phi
        grad_x_k = 1j * kx2d * gyro_phi

        def _to_real(spec):
            return jnp.fft.irfft2(
                pack_half_spectrum(spec[None, None, :, :], jind, mrad, mphiw3),
                s=(mrad, mphi),
                axes=(-2, -1),
                norm="backward",
            )

        max_y = jnp.max(jnp.abs(_to_real(grad_y_k))) * mrad
        max_x = jnp.max(jnp.abs(_to_real(grad_x_k))) * mphi
        return jnp.maximum(max_y, max_x)

    max_vals = jax.vmap(_max_grad_per_s)(phi, bessel_s0)
    max_value = jnp.max(max_vals)

    dt_est = jnp.where(
        max_value > 1e-30,
        jnp.asarray(safety_factor, dtype=jnp.float64) * 2.0 / max_value,
        jnp.asarray(dt_input, dtype=jnp.float64),
    )
    return jnp.minimum(dt_est, jnp.asarray(dt_input, dtype=jnp.float64))


def _precompute_shared(
    geometry, params, kx, ky, ns, nkx, nky, vpgr, mugr, bn, ffun, gfun, dfun, efun
):
    """Species-independent precomputed quantities shared by both paths."""
    pos_par = jnp.asarray(geometry["pos_par_grid_class"], dtype=jnp.int32)
    ixzero = jnp.asarray(
        geometry.get("ixzero", jnp.argmin(jnp.abs(kx))), dtype=jnp.int32
    )
    iyzero = jnp.asarray(
        geometry.get("iyzero", jnp.argmin(jnp.abs(ky))), dtype=jnp.int32
    )
    mphi, mphiw3 = extended_firstdim_fft_size(nky)
    mrad = extended_seconddim_fft_size(nkx)

    def _parallel_coefficients(pos_par_class, table):
        idx = jnp.clip(jnp.asarray(pos_par_class, dtype=jnp.int32) + 2, 0, 4)
        return jnp.moveaxis(table[idx] / 12.0, -1, 0)

    kx_b = jnp.reshape(kx, (1, 1, 1, -1, 1))
    ky_b = jnp.reshape(ky, (1, 1, 1, 1, -1))

    hyper = -(
        jnp.abs(params.disp_y)
        * (ky_b / jnp.maximum(params.kymax, 1e-15))
        ** jnp.where(params.disp_y < 0.0, 2.0, 4.0)
        + jnp.abs(params.disp_x)
        * (kx_b / jnp.maximum(params.kxmax, 1e-15))
        ** jnp.where(params.disp_x < 0.0, 2.0, 4.0)
    )

    return {
        "kx_b": kx_b,
        "ky_b": ky_b,
        "hyper": hyper,
        "s_d1_ipos": _parallel_coefficients(pos_par, stencils.D1_IPW_POS),
        "s_d1_ineg": _parallel_coefficients(pos_par, stencils.D1_IPW_NEG),
        "s_d4_ipos": _parallel_coefficients(pos_par, stencils.D4_IPW_POS),
        "s_d4_ineg": _parallel_coefficients(pos_par, stencils.D4_IPW_NEG),
        "dvp": params.dvp,
        "sgr_dist": params.sgr_dist,
        "ixzero": ixzero,
        "iyzero": iyzero,
        "nl_mphi": mphi,
        "nl_mphiw3": mphiw3,
        "nl_mrad": mrad,
        "nl_fft_scale": jnp.asarray(float(mrad * mphi), dtype=jnp.float64),
        "nl_jind": build_jind(nkx, mrad, ixzero),
        "nl_kx2d": jnp.broadcast_to(jnp.reshape(kx, (nkx, 1)), (nkx, nky)),
        "nl_ky2d": jnp.broadcast_to(jnp.reshape(ky, (1, nky)), (nkx, nky)),
        "nl_dum_s": -jnp.asarray(efun, dtype=jnp.float64),
        "s_shift": jnp.asarray(geometry["s_shift"], dtype=jnp.int32),
        "kx_shift": jnp.asarray(geometry["kx_shift"], dtype=jnp.int32),
        "valid_shift": jnp.asarray(geometry["valid_shift"], dtype=jnp.bool_),
    }


def _fuse_stencils(
    upar,
    abs_par,
    term7_fac,
    disp_par,
    sgr_dist,
    s_d1_ipos,
    s_d1_ineg,
    s_d4_ipos,
    s_d4_ineg,
    stencil_ndim,
):
    """Compute fused streaming + dissipation stencils.

    Works for both 5D (adiabatic) and 6D (kinetic) coefficient arrays.
    stencil_ndim is the number of dimensions for the stencil coefficient
    rearrange pattern: 5 for adiabatic, 6 for kinetic.
    """
    if stencil_ndim == 5:
        # Adiabatic: arrays are (nv, nmu, ns, nkx, nky)
        pat_coeff = "v m s x y -> 1 v m s x y"
        pat_stencil = "i s x y -> i 1 1 s x y"
    else:
        # Kinetic: arrays are (nsp, nv, nmu, ns, nkx, nky)
        pat_coeff = "sp v m s x y -> 1 sp v m s x y"
        pat_stencil = "i s x y -> i 1 1 1 s x y"

    s_d1p = rearrange(s_d1_ipos, pat_stencil)
    s_d1n = rearrange(s_d1_ineg, pat_stencil)
    s_d4p = rearrange(s_d4_ipos, pat_stencil)
    s_d4n = rearrange(s_d4_ineg, pat_stencil)

    upar_sign = rearrange(jnp.sign(upar), pat_coeff)
    s_d1_upar = jnp.where(upar_sign > 0, s_d1p, s_d1n)
    s_d4_upar = jnp.where(upar_sign > 0, s_d4p, s_d4n)

    t7_sign = rearrange(jnp.sign(term7_fac), pat_coeff)
    s_d1_t7 = jnp.where(t7_sign < 0, s_d1p, s_d1n)

    s_total_upar = (
        rearrange(upar, pat_coeff) * s_d1_upar
        + jnp.asarray(disp_par, dtype=jnp.float64)
        * rearrange(abs_par, pat_coeff)
        * s_d4_upar
    ) / jnp.asarray(sgr_dist, dtype=jnp.float64)

    s_total_t7 = (rearrange(term7_fac, pat_coeff) * s_d1_t7) / jnp.asarray(
        sgr_dist, dtype=jnp.float64
    )

    return s_total_upar, s_total_t7


def _compute_species_coeffs(
    mas,
    signz,
    vthrat,
    tmp,
    de,
    rln,
    rlt,
    vpgr,
    mugr,
    bn,
    ffun,
    gfun,
    efun,
    dfun,
    kx,
    ky,
    little_g,
    params,
    ndim,
):
    """Compute species-dependent RHS coefficients.

    When ndim=5, species params are scalars and output arrays are 5D.
    When ndim=6, species params have shape (nsp,) and output arrays are 6D.
    """
    if ndim == 5:
        # Adiabatic: scalar species params, 5D grid arrays
        vp2 = jnp.reshape(vpgr**2, (-1, 1, 1, 1, 1))
        vp = jnp.reshape(vpgr, (-1, 1, 1, 1, 1))
        mu = jnp.reshape(mugr, (1, -1, 1, 1, 1))
        bn_b = jnp.reshape(bn, (1, 1, -1, 1, 1))
        ffun_b = jnp.reshape(ffun, (1, 1, -1, 1, 1))
        gfun_b = jnp.reshape(gfun, (1, 1, -1, 1, 1))
        efun_b = jnp.reshape(efun, (1, 1, -1, 1, 1))
        kx_b = jnp.reshape(kx, (1, 1, 1, -1, 1))
        ky_b = jnp.reshape(ky, (1, 1, 1, 1, -1))

        def g_shape(arr):
            return jnp.reshape(arr, (1, 1, -1, 1, 1))

        def d_shape(arr):
            return jnp.reshape(arr, (1, 1, -1, 1, 1))

        sz = jnp.where(jnp.abs(signz) < 1e-15, 1.0, signz)
    else:
        # Kinetic: per-species params, 6D arrays
        nsp = mas.shape[0]

        def r6(arr):
            return arr.reshape(nsp, 1, 1, 1, 1, 1)

        mas, signz, vthrat, tmp, de, rln, rlt = (
            r6(mas),
            r6(signz),
            r6(vthrat),
            r6(tmp),
            r6(de),
            r6(rln),
            r6(rlt),
        )
        vp2 = jnp.reshape(vpgr**2, (1, -1, 1, 1, 1, 1))
        vp = jnp.reshape(vpgr, (1, -1, 1, 1, 1, 1))
        mu = jnp.reshape(mugr, (1, 1, -1, 1, 1, 1))
        bn_b = jnp.reshape(bn, (1, 1, 1, -1, 1, 1))
        ffun_b = jnp.reshape(ffun, (1, 1, 1, -1, 1, 1))
        gfun_b = jnp.reshape(gfun, (1, 1, 1, -1, 1, 1))
        efun_b = jnp.reshape(efun, (1, 1, 1, -1, 1, 1))
        kx_b = jnp.reshape(kx, (1, 1, 1, 1, -1, 1))
        ky_b = jnp.reshape(ky, (1, 1, 1, 1, 1, -1))

        def g_shape(arr):
            return jnp.reshape(arr, (1, 1, 1, -1, 1, 1))

        def d_shape(arr):
            return jnp.reshape(arr, (1, 1, 1, -1, 1, 1))

        sz = jnp.where(jnp.abs(signz) < 1e-15, 1.0, signz)

    # krloc
    krloc_sq = (
        ky_b**2 * g_shape(little_g[:, 0])
        + 2.0 * ky_b * kx_b * g_shape(little_g[:, 1])
        + kx_b**2 * g_shape(little_g[:, 2])
    )
    krloc = jnp.sqrt(jnp.maximum(krloc_sq, 0.0))

    # Bessel J0
    b_arg = (
        mas
        * vthrat
        * krloc
        * jnp.sqrt(jnp.maximum(2.0 * mu / jnp.maximum(bn_b, 1e-15), 0.0))
        / sz
    )
    bessel = j0(b_arg)

    # Maxwellian
    t_rat = tmp / jnp.asarray(params.tgrid, dtype=jnp.float64)
    fmax = (
        (de / jnp.asarray(params.dgrid, dtype=jnp.float64))
        * jnp.exp(-(vp2 + 2.0 * bn_b * mu) / t_rat)
        / (jnp.sqrt(t_rat * jnp.pi) ** 3)
    )
    et = (vp2 + 2.0 * bn_b * mu) / t_rat - 1.5
    dmax_ek = (rln + rlt * et) * fmax * (efun_b * ky_b)

    # Drifts
    ed = vp2 + bn_b * mu
    drift_x = ed * d_shape(dfun[:, 0]) / sz
    drift_y = ed * d_shape(dfun[:, 1]) / sz

    # Characteristic speeds
    upar = -ffun_b * vthrat * vp
    utrap = vthrat * mu * bn_b * gfun_b

    # Dissipation speeds
    vp_rms = jnp.asarray(params.dvp, dtype=jnp.float64)
    mu_rms = jnp.asarray(1.0, dtype=jnp.float64)
    idisp = jnp.asarray(params.idisp, dtype=jnp.int32)
    use_abs = jnp.logical_or(jnp.equal(idisp, 1), jnp.equal(idisp, -1))
    abs_par = jnp.where(use_abs, jnp.abs(upar), jnp.abs(ffun_b * vthrat * vp_rms))
    abs_vp = jnp.where(
        use_abs, jnp.abs(utrap), jnp.abs(vthrat * bn_b * gfun_b * mu_rms)
    )

    term7_fac = -signz * ffun_b * vthrat * vp * fmax / tmp

    return {
        "bessel": bessel,
        "fmaxwl": fmax,
        "dmaxwel_fm_ek": dmax_ek,
        "drift_x": drift_x,
        "drift_y": drift_y,
        "upar": upar,
        "utrap": utrap,
        "abs_dum2_par": abs_par,
        "abs_dum2_vp": abs_vp,
        "term7_fac": term7_fac,
        "tmp0": jnp.asarray(tmp if ndim == 5 else tmp.squeeze(), dtype=jnp.float64),
        "signz0": jnp.asarray(
            signz if ndim == 5 else signz.squeeze(), dtype=jnp.float64
        ),
    }


def linear_precompute(
    geometry: Dict[str, jnp.ndarray], params: GKParams
) -> Dict[str, jnp.ndarray]:
    """Precompute static geometry-dependent coefficients and Bessel terms."""
    kx, ky = kx_ky_grids(geometry)
    ns, nkx, nky = len(geometry["ints"]), int(kx.shape[0]), int(ky.shape[0])

    vpgr = jnp.asarray(geometry["vpgr"], dtype=jnp.float64)
    mugr = jnp.asarray(geometry["mugr"], dtype=jnp.float64)
    bn = jnp.asarray(geometry["bn"], dtype=jnp.float64)
    ffun = jnp.asarray(geometry["ffun"], dtype=jnp.float64)
    gfun = jnp.asarray(geometry.get("gfun", jnp.zeros_like(bn)), dtype=jnp.float64)
    dfun = jnp.asarray(
        geometry.get("dfun", jnp.zeros((ns, 3), dtype=jnp.float64)), dtype=jnp.float64
    )
    efun = jnp.asarray(geometry.get("efun", jnp.ones_like(bn)), dtype=jnp.float64)
    little_g = jnp.asarray(geometry["little_g"], dtype=jnp.float64)

    # Species-independent shared quantities
    out = _precompute_shared(
        geometry, params, kx, ky, ns, nkx, nky, vpgr, mugr, bn, ffun, gfun, dfun, efun
    )

    if not params.adiabatic_electrons:
        # kinetic: per-species arrays with leading nsp dimension
        mas_arr = jnp.asarray(params.mas, dtype=jnp.float64)
        nsp = int(mas_arr.shape[0])
        sp = _compute_species_coeffs(
            mas_arr,
            jnp.asarray(params.signz, dtype=jnp.float64),
            jnp.asarray(params.vthrat, dtype=jnp.float64),
            jnp.asarray(params.tmp, dtype=jnp.float64),
            jnp.asarray(params.de, dtype=jnp.float64),
            jnp.asarray(params.rln, dtype=jnp.float64),
            jnp.asarray(params.rlt, dtype=jnp.float64),
            vpgr,
            mugr,
            bn,
            ffun,
            gfun,
            efun,
            dfun,
            kx,
            ky,
            little_g,
            params,
            ndim=6,
        )
        # Override vpgr_rms/mugr_rms if available
        if "vpgr_rms" in geometry:
            vp_rms = jnp.asarray(geometry["vpgr_rms"], dtype=jnp.float64)
            mu_rms = jnp.asarray(geometry.get("mugr_rms", 1.0), dtype=jnp.float64)
            vthrat_6 = jnp.asarray(params.vthrat, dtype=jnp.float64).reshape(
                nsp, 1, 1, 1, 1, 1
            )
            ffun_6 = jnp.reshape(ffun, (1, 1, 1, -1, 1, 1))
            bn_6 = jnp.reshape(bn, (1, 1, 1, -1, 1, 1))
            gfun_6 = jnp.reshape(gfun, (1, 1, 1, -1, 1, 1))
            idisp = jnp.asarray(params.idisp, dtype=jnp.int32)
            use_abs = jnp.logical_or(jnp.equal(idisp, 1), jnp.equal(idisp, -1))
            sp["abs_dum2_par"] = jnp.where(
                use_abs, jnp.abs(sp["upar"]), jnp.abs(ffun_6 * vthrat_6 * vp_rms)
            )
            sp["abs_dum2_vp"] = jnp.where(
                use_abs,
                jnp.abs(sp["utrap"]),
                jnp.abs(vthrat_6 * bn_6 * gfun_6 * mu_rms),
            )

        sp["s_total_upar"], sp["s_total_t7"] = _fuse_stencils(
            sp["upar"],
            sp["abs_dum2_par"],
            sp["term7_fac"],
            params.disp_par,
            params.sgr_dist,
            out["s_d1_ipos"],
            out["s_d1_ineg"],
            out["s_d4_ipos"],
            out["s_d4_ineg"],
            stencil_ndim=6,
        )
        out.update(sp)
        out["geom_tensors"] = None
        out["nsp"] = nsp

        # precompute kinetic phi solve arrays (avoids recomputing bessel/gamma per RHS call)
        from gyaradax.integrals import precompute_phi_kinetic

        phi_w, phi_d = precompute_phi_kinetic(geometry)
        out["phi_weight"] = phi_w
        out["phi_diag"] = phi_d
    else:
        # Adiabatic: scalar species params, 5D arrays
        sp = _compute_species_coeffs(
            params.mas,
            params.signz,
            params.vthrat,
            params.tmp,
            params.de,
            params.rln,
            params.rlt,
            vpgr,
            mugr,
            bn,
            ffun,
            gfun,
            efun,
            dfun,
            kx,
            ky,
            little_g,
            params,
            ndim=5,
        )
        # Override vpgr_rms/mugr_rms if available
        if "vpgr_rms" in geometry:
            vp_rms = jnp.asarray(geometry["vpgr_rms"], dtype=jnp.float64)
            mu_rms = jnp.asarray(geometry.get("mugr_rms", 1.0), dtype=jnp.float64)
            ffun_b = jnp.reshape(ffun, (1, 1, -1, 1, 1))
            bn_b = jnp.reshape(bn, (1, 1, -1, 1, 1))
            gfun_b = jnp.reshape(gfun, (1, 1, -1, 1, 1))
            idisp = jnp.asarray(params.idisp, dtype=jnp.int32)
            use_abs = jnp.logical_or(jnp.equal(idisp, 1), jnp.equal(idisp, -1))
            sp["abs_dum2_par"] = jnp.where(
                use_abs, jnp.abs(sp["upar"]), jnp.abs(ffun_b * params.vthrat * vp_rms)
            )
            sp["abs_dum2_vp"] = jnp.where(
                use_abs,
                jnp.abs(sp["utrap"]),
                jnp.abs(params.vthrat * bn_b * gfun_b * mu_rms),
            )

        sp["s_total_upar"], sp["s_total_t7"] = _fuse_stencils(
            sp["upar"],
            sp["abs_dum2_par"],
            sp["term7_fac"],
            params.disp_par,
            params.sgr_dist,
            out["s_d1_ipos"],
            out["s_d1_ineg"],
            out["s_d4_ipos"],
            out["s_d4_ineg"],
            stencil_ndim=5,
        )
        out.update(sp)
        out["geom_tensors"] = geom_tensors(geometry, params=params)

    return GKPre(out)


def _linear_rhs_core(
    df: jnp.ndarray,
    phi_b: jnp.ndarray,
    pre: Dict[str, jnp.ndarray],
    params_dvp: float,
    params_disp_vp: float,
    params_drive_scale: float,
) -> jnp.ndarray:
    """Core linear RHS for a single species slice (5D arrays).

    All arrays in *pre* must be 5D (nv, nmu, ns, nkx, nky) — species
    dimension has been removed by vmap or was never present.
    """

    def _apply_parallel(field, coeffs):
        out = jnp.zeros_like(field)
        nky = field.shape[-1]
        ky_idx = jnp.reshape(jnp.arange(nky, dtype=jnp.int32), (1, 1, -1))
        for i in range(9):
            s_map = pre["s_shift"][i]
            kx_map = pre["kx_shift"][i]
            valid = pre["valid_shift"][i]
            shifted = jnp.where(
                valid[None, None, :, :, :], field[:, :, s_map, kx_map, ky_idx], 0.0
            )
            out = out + coeffs[i] * shifted
        return out

    def _apply_vpar(field, coeffs):
        nv = field.shape[0]
        out = jnp.zeros_like(field)
        for c, s in zip(coeffs, (-2, -1, 0, 1, 2)):
            idx = jnp.clip(jnp.arange(nv, dtype=jnp.int32) + s, 0, nv - 1)
            valid = jnp.logical_and(jnp.arange(nv) + s >= 0, jnp.arange(nv) + s < nv)
            shifted = jnp.take(field, idx, axis=0)
            out = out + c * jnp.where(valid[:, None, None, None, None], shifted, 0.0)
        return out

    term_par = _apply_parallel(df, pre["s_total_upar"])
    term_iv = pre["utrap"] * _apply_vpar(df, stencils.VPAR_D1) / params_dvp
    term_vp_diss = (
        jnp.asarray(params_disp_vp, dtype=jnp.float64)
        * pre["abs_dum2_vp"]
        * _apply_vpar(df, stencils.VPAR_D4)
        / params_dvp
    )
    kdotvd = pre["drift_x"] * pre["kx_b"] + pre["drift_y"] * pre["ky_b"]
    gyro_phi = pre["bessel"] * phi_b
    term_vii = _apply_parallel(gyro_phi, pre["s_total_t7"])

    return (
        term_par
        + term_iv
        + term_vp_diss
        - 1j * kdotvd * df
        + pre["hyper"] * df
        + 1j
        * jnp.asarray(params_drive_scale, dtype=jnp.float64)
        * (
            pre["dmaxwel_fm_ek"]
            - pre["signz0"] * kdotvd * (pre["fmaxwl"] / jnp.maximum(pre["tmp0"], 1e-15))
        )
        * gyro_phi
        + term_vii
    )


def linear_rhs(
    df: jnp.ndarray,
    geometry: Dict[str, jnp.ndarray],
    params: GKParams,
    pre: Dict[str, jnp.ndarray],
    phi: Optional[jnp.ndarray] = None,
) -> jnp.ndarray:
    """Adiabatic linear RHS (5D df). Delegates to _linear_rhs_core."""
    if phi is None:
        phi = calculate_phi(pre["geom_tensors"], df)
    phi_b = jnp.reshape(phi, (1, 1, phi.shape[0], phi.shape[1], phi.shape[2]))
    return _linear_rhs_core(
        df, phi_b, pre, params.dvp, params.disp_vp, params.drive_scale
    )


def init_f(
    geometry: Dict[str, jnp.ndarray],
    finit: str = "cosine2",
    amp_init_real: float = 1.0e-4,
    amp_init_imag: float = 0.0,
    normalize_per_toroidal_mode: bool = False,
    norm_eps: float = 1.0e-14,
    n_species: int = 1,
    seed: int = 42,
) -> jnp.ndarray:
    """Initialize the distribution function.

    Supported finit modes (matching GKW):
        cosine2 (default): amp * (cos(2*pi*s) + 1), flat in velocity space
        cosine:  amp * cos(2*pi*s), flat in velocity space
        cosine3: like cosine2 but weighted by exp(-E) in velocity space
        sine:    amp * de(is) * (sin(2*pi*s) + 1), density-weighted
        noise:   uniform random on [-1, 1] (real + imag)
        gnoise:  gaussian random (Box-Muller transform)
    """
    nv, nmu, ns, nkx, nky = (
        len(geometry["intvp"]),
        len(geometry["intmu"]),
        len(geometry["ints"]),
        len(geometry["kxrh"]),
        len(geometry["krho"]),
    )
    sgrid = jnp.asarray(
        geometry.get("sgrid", jnp.linspace(-0.5, 0.5, ns)), dtype=jnp.float64
    )
    vpgr = jnp.asarray(geometry["vpgr"], dtype=jnp.float64)
    mugr = jnp.asarray(geometry["mugr"], dtype=jnp.float64)
    bn = jnp.asarray(geometry["bn"], dtype=jnp.float64)

    amp = jnp.asarray(amp_init_real, dtype=jnp.float64) + 1j * jnp.asarray(
        amp_init_imag, dtype=jnp.float64
    )

    shape_5d = (nv, nmu, ns, nkx, nky)
    shape_6d = (n_species, nv, nmu, ns, nkx, nky)
    full_shape = shape_6d if n_species > 1 else shape_5d

    # velocity-space Maxwellian envelope: exp(-(vpar^2 + 2*mu*B))
    vp2 = vpgr**2  # (nv,)
    energy = vp2[:, None, None] + 2.0 * mugr[None, :, None] * bn[None, None, :]
    maxwellian_env = jnp.exp(-energy)  # (nv, nmu, ns)

    if finit in ("noise", "gnoise"):
        key = jax.random.PRNGKey(seed)
        k1, k2 = jax.random.split(key)
        if finit == "gnoise":
            noise_real = jax.random.normal(k1, full_shape)
            noise_imag = jax.random.normal(k2, full_shape)
        else:
            noise_real = jax.random.uniform(k1, full_shape, minval=-1.0, maxval=1.0)
            noise_imag = jax.random.uniform(k2, full_shape, minval=-1.0, maxval=1.0)
        df = amp * (noise_real + 1j * noise_imag)

    elif finit == "cosine2":
        # amp * (cos(2*pi*s) + 1), uniform in velocity
        prof_s = amp * (jnp.cos(2.0 * jnp.pi * sgrid) + 1.0)  # (ns,)
        df = _broadcast_profile(prof_s, None, n_species, nv, nmu, ns, nkx, nky)

    elif finit == "cosine":
        prof_s = amp * jnp.cos(2.0 * jnp.pi * sgrid)
        df = _broadcast_profile(prof_s, None, n_species, nv, nmu, ns, nkx, nky)

    elif finit == "cosine3":
        # amp * (cos(2*pi*s) + 1) * exp(-(vpar^2 + 2*mu*B))
        prof_s = amp * (jnp.cos(2.0 * jnp.pi * sgrid) + 1.0)
        df = _broadcast_profile(
            prof_s, maxwellian_env, n_species, nv, nmu, ns, nkx, nky
        )

    elif finit == "sine":
        # amp * de(is) * (sin(2*pi*s) + 1)
        de = jnp.asarray(
            geometry.get("de", jnp.ones(max(n_species, 1))), dtype=jnp.float64
        )
        prof_s = amp * (jnp.sin(2.0 * jnp.pi * sgrid) + 1.0)
        if n_species > 1 and de.ndim >= 1 and de.shape[0] > 1:
            # (nsp, ns)
            prof_2d = prof_s[None, :] * de[:, None]
            df = jnp.broadcast_to(prof_2d[:, None, None, :, None, None], shape_6d)
        else:
            de_val = float(de) if de.ndim == 0 else float(de[0])
            prof_s = prof_s * de_val
            df = _broadcast_profile(prof_s, None, n_species, nv, nmu, ns, nkx, nky)

    else:
        raise ValueError(f"unknown finit: {finit}")

    df = df.astype(jnp.complex128)

    # zero out the zonal mode (ky=0)
    if nky > 1:
        iy0 = int(
            jnp.asarray(
                geometry.get(
                    "iyzero", jnp.argmin(jnp.abs(jnp.asarray(geometry["krho"])))
                )
            ).item()
        )
        df = df.at[..., iy0].set(0.0)

    if normalize_per_toroidal_mode:
        df, _, _ = normalize_per_ky(df, geometry, GKParams(norm_eps=norm_eps))
    return df


def _broadcast_profile(prof_s, vel_env, n_species, nv, nmu, ns, nkx, nky):
    """broadcast a parallel profile (and optional velocity envelope) to full shape."""
    if vel_env is not None:
        # vel_env: (nv, nmu, ns), prof_s: (ns,)
        base = vel_env * prof_s[None, None, :]  # (nv, nmu, ns)
        if n_species > 1:
            return jnp.broadcast_to(
                base[None, :, :, :, None, None], (n_species, nv, nmu, ns, nkx, nky)
            )
        else:
            return jnp.broadcast_to(base[:, :, :, None, None], (nv, nmu, ns, nkx, nky))
    else:
        # flat in velocity
        if n_species > 1:
            prof = jnp.broadcast_to(prof_s[None, :], (n_species, ns))
            return jnp.broadcast_to(
                prof[:, None, None, :, None, None], (n_species, nv, nmu, ns, nkx, nky)
            )
        else:
            return jnp.broadcast_to(
                prof_s[None, None, :, None, None], (nv, nmu, ns, nkx, nky)
            )


def advance_state(
    state: GKState,
    params: GKParams,
    is_window_end: jnp.ndarray,
    per_mode_amp: jnp.ndarray,
    per_mode_norm_fac: jnp.ndarray,
    dt_used: Optional[jnp.ndarray] = None,
) -> GKState:
    dt = dt_used if dt_used is not None else jnp.array(params.dt, dtype=jnp.float64)
    new_step = state.step + jnp.array(1, dtype=jnp.int32)
    new_time = state.time + dt
    valid_growth = jnp.logical_and(
        state.window_start_amp > params.norm_eps, per_mode_amp > params.norm_eps
    )
    steps_in_window = jnp.mod(new_step - 1, params.naverage) + 1
    growth_dt = jnp.array(params.dt * steps_in_window, dtype=jnp.float64)
    growth_rate = jnp.where(
        valid_growth,
        jnp.log(per_mode_amp / state.window_start_amp) / growth_dt,
        state.last_growth_rate,
    )
    new_window_start_amp = jnp.where(
        is_window_end, jnp.ones_like(state.window_start_amp), state.window_start_amp
    )
    return GKState(
        time=new_time,
        step=new_step,
        accumulated_norm_factor=state.accumulated_norm_factor * per_mode_norm_fac,
        window_start_amp=new_window_start_amp,
        last_growth_rate=growth_rate,
    )


def _compute_phi(df, geometry, params, pre):
    """compute phi via the appropriate solver."""
    if params.adiabatic_electrons:
        return calculate_phi(pre["geom_tensors"], df)
    else:
        return calculate_phi_kinetic(
            geometry,
            df,
            phi_weight=pre.get("phi_weight"),
            phi_diag=pre.get("phi_diag"),
        )


def _compute_linear_rhs(df, phi, geometry, params, pre):
    """Compute linear RHS for adiabatic (5D) or kinetic (6D) df."""
    if params.adiabatic_electrons:
        return linear_rhs(df, geometry, params, pre, phi=phi)
    else:
        phi_b = phi[None, None, :, :, :]

        def _per_species(
            df_sp,
            bessel_sp,
            fmaxwl_sp,
            dmaxwel_sp,
            drift_x_sp,
            drift_y_sp,
            upar_sp,
            utrap_sp,
            abs_par_sp,
            abs_vp_sp,
            term7_fac_sp,
            tmp0_sp,
            signz0_sp,
            s_total_upar_sp,
            s_total_t7_sp,
        ):
            sp_pre = {
                "bessel": bessel_sp,
                "fmaxwl": fmaxwl_sp,
                "dmaxwel_fm_ek": dmaxwel_sp,
                "drift_x": drift_x_sp,
                "drift_y": drift_y_sp,
                "upar": upar_sp,
                "utrap": utrap_sp,
                "abs_dum2_par": abs_par_sp,
                "abs_dum2_vp": abs_vp_sp,
                "term7_fac": term7_fac_sp,
                "tmp0": tmp0_sp,
                "signz0": signz0_sp,
                "s_total_upar": s_total_upar_sp,
                "s_total_t7": s_total_t7_sp,
                "kx_b": pre["kx_b"],
                "ky_b": pre["ky_b"],
                "hyper": pre["hyper"],
                "s_shift": pre["s_shift"],
                "kx_shift": pre["kx_shift"],
                "valid_shift": pre["valid_shift"],
            }
            return _linear_rhs_core(
                df_sp, phi_b, sp_pre, pre["dvp"], params.disp_vp, params.drive_scale
            )

        return jax.vmap(
            _per_species,
            in_axes=(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1),
        )(
            df,
            pre["bessel"],
            pre["fmaxwl"],
            pre["dmaxwel_fm_ek"],
            pre["drift_x"],
            pre["drift_y"],
            pre["upar"],
            pre["utrap"],
            pre["abs_dum2_par"],
            pre["abs_dum2_vp"],
            pre["term7_fac"],
            pre["tmp0"],
            pre["signz0"],
            pre["s_total_upar"],
            pre["s_total_t7"],
        )


def _compute_nonlinear_rhs(df, phi, geometry, params, pre):
    """Compute nonlinear Term III for adiabatic (5D) or kinetic (6D) df."""
    mp = params.mixed_precision
    if params.adiabatic_electrons:
        return nonlinear_term_iii(df, phi, geometry, pre, mixed_precision=mp)
    else:

        def _nl_sp(df_sp, bessel_sp):
            pre_sp = {**pre, "bessel": bessel_sp}
            return nonlinear_term_iii(df_sp, phi, geometry, pre_sp, mixed_precision=mp)

        return jax.vmap(_nl_sp)(df, pre["bessel"])


def gkstep_single(
    prev_df: jnp.ndarray,
    geometry: Dict[str, jnp.ndarray],
    params: GKParams,
    state: GKState,
    pre: GKPre,
    dt_override: Optional[jnp.ndarray] = None,
) -> Tuple[
    jnp.ndarray,
    Tuple[jnp.ndarray, Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]],
    GKState,
]:
    """Single small-step RK4 time integration.

    Args:
        dt_override: If provided, use this dt instead of params.dt.
            Used by the adaptive CFL path where dt varies per step.
    """
    dt = (
        dt_override
        if dt_override is not None
        else jnp.array(params.dt, dtype=jnp.float64)
    )

    def _rhs(df):
        phi_local = _compute_phi(df, geometry, params, pre)
        rhs = _compute_linear_rhs(df, phi_local, geometry, params, pre)
        if params.non_linear:
            rhs = rhs + _compute_nonlinear_rhs(df, phi_local, geometry, params, pre)
        return rhs

    # RK4
    k1 = _rhs(prev_df)
    k2 = _rhs(prev_df + 0.5 * dt * k1)
    k3 = _rhs(prev_df + 0.5 * dt * k2)
    k4 = _rhs(prev_df + dt * k3)
    next_df_raw = prev_df + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

    # Post-step: normalization and amplitude tracking
    new_step = state.step + jnp.array(1, dtype=jnp.int32)
    is_window_end = jnp.equal(jnp.mod(new_step, params.naverage), 0)

    if params.non_linear:
        phi = _compute_phi(next_df_raw, geometry, params, pre)
        current_amp = mode_amplitude(phi, geometry, params.norm_eps)
        next_df = next_df_raw
        norm_factor = jnp.ones_like(state.accumulated_norm_factor)
    else:

        def _apply_norm(_):
            return normalize_per_ky(next_df_raw, geometry, params)

        def _skip_norm(_):
            phi_curr = _compute_phi(next_df_raw, geometry, params, pre)
            amp_curr = mode_amplitude(phi_curr, geometry, params.norm_eps)
            return (next_df_raw, jnp.ones_like(state.accumulated_norm_factor), amp_curr)

        next_df, norm_factor, current_amp = jax.lax.cond(
            is_window_end, _apply_norm, _skip_norm, operand=None
        )
        phi = _compute_phi(next_df, geometry, params, pre)

    z = jnp.array(0.0, dtype=jnp.float64)
    next_state = advance_state(
        state, params, is_window_end, current_amp, norm_factor, dt_used=dt
    )
    return next_df, (phi, (z, z, z)), next_state


@functools.partial(jax.jit, static_argnames=("n_steps",))
def gksolve(
    df: jnp.ndarray,
    geometry: Dict[str, jnp.ndarray],
    params: GKParams,
    state: GKState,
    n_steps: int = 1,
    pre: Optional[GKPre] = None,
) -> Tuple[
    jnp.ndarray,
    Tuple[jnp.ndarray, Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]],
    GKState,
]:
    """Gyrokinetics solver forward.

    Executes multiple time steps via jax.lax.scan.
    When params.adaptive_dt is True, uses CFL-adaptive timestep with
    one-step lag (current step uses CFL estimate from previous step's phi).
    Returns (final_df, (final_phi, final_fluxes), final_state).
    """
    if pre is None:
        pre = linear_precompute(geometry, params)

    if params.adaptive_dt and params.non_linear:
        # Adaptive CFL path: carry dt as part of scan state
        dt_input = jnp.array(params.dt, dtype=jnp.float64)

        def _scan_body(carry, _):
            curr_df, curr_state, curr_dt = carry
            next_df, out, next_state = gkstep_single(
                curr_df, geometry, params, curr_state, pre, dt_override=curr_dt
            )
            # Estimate next dt from current phi (one-step lag)
            phi_for_cfl = out[0]  # phi from gkstep_single
            bessel_for_cfl = pre["bessel"]
            if not params.adiabatic_electrons:
                bessel_for_cfl = bessel_for_cfl[0:1]  # use ion Bessel (largest FLR)
            next_dt = estimate_nl_timestep(
                phi_for_cfl,
                pre,
                bessel_for_cfl,
                dt_input=float(params.dt),
                safety_factor=float(params.cfl_safety),
            )
            return (next_df, next_state, next_dt), None

        init_dt = dt_input
        (final_df, final_state, _), _ = jax.lax.scan(
            _scan_body, (df, state, init_dt), None, length=n_steps
        )
    else:
        # Fixed dt path
        def _scan_body(carry, _):
            curr_df, curr_state = carry
            next_df, out, next_state = gkstep_single(
                curr_df, geometry, params, curr_state, pre
            )
            return (next_df, next_state), None

        (final_df, final_state), _ = jax.lax.scan(
            _scan_body, (df, state), None, length=n_steps
        )

    phi, fluxes = get_integrals(
        final_df,
        geometry,
        params=params,
        geom=pre.get("geom_tensors"),
        adiabatic_electrons=params.adiabatic_electrons,
    )
    return final_df, (phi, fluxes), final_state
