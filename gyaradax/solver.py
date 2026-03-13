import jax
import jax.numpy as jnp
import numpy as np
import math
import os
from dataclasses import dataclass
from typing import Dict, Tuple, Any, Optional

from gyaradax.integrals import get_integrals, j0
from gyaradax import stencils


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class GKParams:
    """
    Runtime controls and physical parameters for the electrostatic solver.

    This dataclass mirrors the GKW 'control', 'gridsize', and 'species' namelists,
    handling numerical hyperparameters and physical constants required for the
    gyrokinetic Vlasov-Poisson system.
    """

    # runtime controls
    dt: float = 0.01
    naverage: int = 40
    disp_par: float = 1.0
    disp_vp: float = 0.2
    disp_x: float = 0.1
    disp_y: float = 0.1
    idisp: int = 2
    drive_scale: float = 1.0
    norm_eps: float = 1.0e-14
    non_linear: bool = False
    enable_term_iii: bool = True

    # physical parameters (typically from the kinetic species)
    rlt: float = 1.0
    rln: float = 1.0
    mas: float = 1.0
    tmp: float = 1.0
    de: float = 1.0
    signz: float = 1.0
    vthrat: float = 1.0

    # geometry scalars
    shat: float = 0.0
    q: float = 1.0
    eps: float = 0.0
    kthnorm: float = 1.0
    Rref: float = 1.0
    d2X: float = 1.0
    signB: float = 1.0

    # grid metadata and scaling
    dvp: float = 1.0
    sgr_dist: float = 1.0
    kxmax: float = 1.0
    kymax: float = 1.0
    dgrid: float = 1.0
    tgrid: float = 1.0

    def tree_flatten(self):
        return tuple(vars(self).values()), None

    @classmethod
    def tree_unflatten(cls, aux_data, leaves):
        return cls(*leaves)


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class GKState:
    """
    Explicit diagnostic state used for large-step growth tracking and normalization.

    This state tracks metadata across 'naverage' intervals to calculate growth rates
    and maintain normalization history. It is separate from the physical distribution
    function to keep the gksolve interface functional.

    Attributes:
        time: Current simulation time.
        step: Cumulative step count.
        accumulated_norm_factor: Product of all normalization rescalings applied.
        window_start_amp: Mode amplitude at the beginning of the current naverage window.
        last_growth_rate: Calculated exponential growth rate from the previous window.
    """

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


def default_state() -> GKState:
    """
    Construct a default diagnostic state initialized at simulation startup.

    Returns:
        GKState object with time and steps zeroed, and unit normalization factors.
    """
    return GKState(
        time=jnp.array(0.0, dtype=jnp.float64),
        step=jnp.array(0, dtype=jnp.int32),
        accumulated_norm_factor=jnp.array(1.0, dtype=jnp.float64),
        window_start_amp=jnp.array(1.0, dtype=jnp.float64),
        last_growth_rate=jnp.array(0.0, dtype=jnp.float64),
    )


def gkparams_from_runtime(runtime: Dict[str, Any], **overrides) -> GKParams:
    """
    Build GKParams from a GKW-compatible runtime-controls dictionary.

    Args:
        runtime: Dictionary of parameters (dtim, naverage, etc.) typically parsed from input.dat.
        overrides: Keyword arguments to override specific params manually.

    Returns:
        Configured GKParams instance.
    """
    # this helper is legacy and might not fill all physical params
    # users should prefer gkparams_from_config or filling manually
    params_dict = {
        "dt": float(runtime.get("dtim", 0.01)),
        "naverage": int(runtime.get("naverage", 40)),
        "disp_par": float(runtime.get("disp_par", 1.0)),
        "disp_vp": float(runtime.get("disp_vp", 0.2)),
        "disp_x": float(runtime.get("disp_x", 0.1)),
        "disp_y": float(runtime.get("disp_y", 0.1)),
        "non_linear": bool(runtime.get("non_linear", False)),
    }
    # try to fill physical params if available in runtime dict
    for k in [
        "rlt",
        "rln",
        "mas",
        "tmp",
        "de",
        "signz",
        "vthrat",
        "shat",
        "q",
        "eps",
        "kthnorm",
        "Rref",
        "d2X",
        "signB",
        "dvp",
        "sgr_dist",
        "kxmax",
        "kymax",
    ]:
        if k in runtime:
            params_dict[k] = float(runtime[k])

    if overrides:
        params_dict.update(overrides)
    return GKParams(**params_dict)


def gkparams_from_input_dat(input_dat_path: str, **overrides) -> GKParams:
    """
    Load all runtime, physics, and geometry scalars from a GKW run directory.

    Args:
        input_dat_path: Path to the GKW input.dat file.
        overrides: Manual parameter overrides.

    Returns:
        Configured GKParams instance.
    """
    from gyaradax.geometry import load_scalars

    # get the directory containing input.dat to load geom.dat as well
    directory = os.path.dirname(input_dat_path)
    scalars = load_scalars(directory)
    return gkparams_from_runtime(scalars, **overrides)


def load_config(config_path: str) -> Any:
    """
    Load a structured YAML configuration using OmegaConf.

    Args:
        config_path: Path to the .yaml configuration file.

    Returns:
        OmegaConf DictConfig object containing solver and grid settings.
    """
    from omegaconf import OmegaConf

    return OmegaConf.load(config_path)


def gkparams_from_config(config: Any, **overrides) -> GKParams:
    """
    Build GKParams from an OmegaConf configuration object.

    Args:
        config: Configuration object with 'solver' and optionally 'physics' sections.
        overrides: Manual parameter overrides.

    Returns:
        Configured GKParams instance.
    """
    solver_cfg = config.solver
    physics_cfg = getattr(config, "physics", {})
    geometry_cfg = getattr(config, "geometry", {})

    params_dict = {
        "dt": float(getattr(solver_cfg, "dt", 0.01)),
        "naverage": int(getattr(solver_cfg, "naverage", 40)),
        "disp_par": float(getattr(solver_cfg, "disp_par", 1.0)),
        "disp_vp": float(getattr(solver_cfg, "disp_vp", 0.2)),
        "disp_x": float(getattr(solver_cfg, "disp_x", 0.1)),
        "disp_y": float(getattr(solver_cfg, "disp_y", 0.1)),
        "non_linear": bool(getattr(solver_cfg, "non_linear", False)),
        "enable_term_iii": bool(getattr(solver_cfg, "enable_term_iii", True)),
    }

    # fill physics scalars
    for k in ["rlt", "rln", "mas", "tmp", "de", "signz", "vthrat", "dgrid", "tgrid"]:
        if hasattr(physics_cfg, k):
            params_dict[k] = float(getattr(physics_cfg, k))

    # fill geometry scalars
    for k in ["shat", "q", "eps", "kthnorm", "Rref", "d2X", "signB"]:
        if hasattr(geometry_cfg, k):
            params_dict[k] = float(getattr(geometry_cfg, k))

    # fill scaling/grid scalars
    for k in ["dvp", "sgr_dist", "kxmax", "kymax"]:
        if hasattr(geometry_cfg, k):
            params_dict[k] = float(getattr(geometry_cfg, k))

    if overrides:
        params_dict.update(overrides)
    return GKParams(**params_dict)


def kx_ky_grids(geometry: Dict[str, jnp.ndarray]) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Extract and normalize the spectral wavevector grids from geometry metadata.

    Args:
        geometry: Dictionary containing kxrh and krho grid metadata.

    Returns:
        Tuple of (kx, ky) grids as 1D JAX arrays.
    """
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
    """
    Calculate the L2 mode amplitude of the electrostatic potential for each ky.

    The amplitude is defined as the square root of the flux-surface integrated potential:
    amp = sqrt( ds * sum_{s,kx} |phi(s, kx, ky)|^2 ).

    Args:
        phi: Complex electrostatic potential [ns, nkx, nky].
        geometry: Geometry dictionary for integration weights (ints).
        eps: Numerical floor to prevent zero amplitudes.

    Returns:
        Array of amplitudes for each ky mode.
    """
    ints = jnp.asarray(geometry["ints"], dtype=jnp.float64)
    ds = ints[0]
    amp2 = ds * jnp.sum(jnp.abs(phi) ** 2, axis=(0, 1))
    return jnp.sqrt(jnp.maximum(amp2, eps))


def normalize_per_ky(
    df: jnp.ndarray, geometry: Dict[str, jnp.ndarray], params: GKParams
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Rescale the distribution function such that each ky mode has unit potential amplitude.

    This is the standard GKW normalization for linear simulations, preventing
    exponential overflow and allowing consistent growth rate diagnostics.

    Args:
        df: 5D distribution function [vpar, mu, s, kx, ky].
        geometry: Geometry dictionary for potential calculation.
        params: Parameters for the normalization floor.

    Returns:
        Tuple of (normalized_df, average_inv_factor, max_amplitude).
    """
    phi, _ = get_integrals(df, geometry, params=params, include_fluxes=False)
    amp_per_ky = mode_amplitude(phi, geometry, params.norm_eps)
    # prevent division by zero for stable or zero modes
    safe_amp = jnp.where(amp_per_ky < params.norm_eps, 1.0, amp_per_ky)
    inv = 1.0 / safe_amp
    # apply normalization factor across velocity and space dimensions
    normalized_df = df * jnp.reshape(inv, (1, 1, 1, 1, inv.shape[0]))
    dominant_amp = jnp.max(safe_amp)
    return normalized_df, jnp.mean(inv), dominant_amp


def prime_factors_smallereq_than(number: int, max_prime: int) -> bool:
    """Check if all prime factors of a number are less than or equal to max_prime."""
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
    """
    Calculate the dealiased FFT size for the binormal (ky) dimension.

    Implements the 3/2 rule for pseudospectral dealiasing, ensuring the grid
    size is numerically efficient for FFTW-like algorithms.

    Args:
        nmod: Number of physical binormal modes.

    Returns:
        Tuple of (mphi, mphiw3) representing real-space and spectral storage sizes.
    """
    posspace_size = 3 * nmod - 2
    if posspace_size % 2 != 0:
        posspace_size += 1
    # find next size with small prime factors for efficiency
    while not prime_factors_smallereq_than(posspace_size, 7):
        posspace_size += 2
    # prefer powers of two if within reasonable range
    for i in range(1, 9):
        cand = posspace_size + 2 * i
        if prime_factors_smallereq_than(cand, 2):
            posspace_size = cand
            break
    kgrid_size = int(math.floor(posspace_size / 2.0) + 1)
    return posspace_size, kgrid_size


def extended_seconddim_fft_size(nx: int) -> int:
    """Calculate the dealiased FFT size for the radial (kx) dimension."""
    dum = int(math.ceil(1.5 * float(nx + 1)) + 1)
    while not prime_factors_smallereq_than(dum, 7):
        dum += 1
    # optimize for power-of-two FFTs
    for i in range(1, 9):
        cand = dum + i
        if prime_factors_smallereq_than(cand, 2):
            dum = cand
            break
    return dum


def build_jind(nkx: int, mrad: int, ixzero: int) -> jnp.ndarray:
    """
    Map physical kx modes to the Fortran-style FFT storage indexing.

    This handles the split between positive and negative radial wavevectors
    required for the 2D Real-to-Complex FFT layout.
    """
    ix = jnp.arange(nkx, dtype=jnp.int32)
    return jnp.where(ix >= ixzero, ix - ixzero, mrad + ix - ixzero)


def pack_half_spectrum(
    spec_kxky: jnp.ndarray, jind: jnp.ndarray, mrad: int, mphiw3: int
) -> jnp.ndarray:
    """Pack physical spectral modes into a zero-padded dealiased FFT buffer."""
    out_shape = spec_kxky.shape[:-2] + (mrad, mphiw3)
    out = jnp.zeros(out_shape, dtype=jnp.complex128)
    nky = spec_kxky.shape[-1]
    return out.at[..., jind, :nky].set(spec_kxky)


def unpack_half_spectrum(
    spec_half: jnp.ndarray, jind: jnp.ndarray, nky: int
) -> jnp.ndarray:
    """Extract physical spectral modes from a dealiased FFT storage buffer."""
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
) -> jnp.ndarray:
    """
    Calculate Nonlinear Term III (ExB Advection) using the pseudospectral method.
    """
    mrad, mphi, mphiw3 = pre["nl_mrad"], pre["nl_mphi"], pre["nl_mphiw3"]
    fft_scale, jind = pre["nl_fft_scale"], pre["nl_jind"]
    kx2d, ky2d, bessel = pre["nl_kx2d"], pre["nl_ky2d"], pre["bessel"]
    dum_s, ixzero, iyzero = pre["nl_dum_s"], pre["ixzero"], pre["iyzero"]
    nky = df.shape[-1]

    # Vectorize over parallel grid
    df_by_s = jnp.moveaxis(df, 2, 0)
    bessel_by_s = jnp.moveaxis(bessel, 2, 0)

    def _per_s(
        df_s: jnp.ndarray, phi_s: jnp.ndarray, bessel_s: jnp.ndarray, dum: jnp.ndarray
    ) -> jnp.ndarray:
        gyro_phi = bessel_s * phi_s[None, None, :, :]
        grad_phi_y_k = 1j * ky2d[None, None, :, :] * gyro_phi
        grad_phi_x_k = 1j * kx2d[None, None, :, :] * gyro_phi
        grad_f_x_k = 1j * kx2d[None, None, :, :] * df_s
        grad_f_y_k = 1j * ky2d[None, None, :, :] * df_s

        def _to_real(spec):
            return jnp.fft.irfft2(
                pack_half_spectrum(spec, jind, mrad, mphiw3),
                s=(mrad, mphi),
                axes=(-2, -1),
                norm="backward",
            )

        nl_real = (efun_sign * dum) * (
            _to_real(grad_phi_y_k) * _to_real(grad_f_x_k)
            - _to_real(grad_phi_x_k) * _to_real(grad_f_y_k)
        )

        nl_half = (
            jnp.asarray(fft_prefactor, dtype=jnp.complex128)
            * jnp.asarray(fft_scale, dtype=jnp.complex128)
            * jnp.fft.rfft2(nl_real, s=(mrad, mphi), axes=(-2, -1), norm="backward")
        )
        return unpack_half_spectrum(nl_half, jind, nky)

    nl = jnp.moveaxis(jax.vmap(_per_s)(df_by_s, phi, bessel_by_s, dum_s), 0, 2)
    return nl.at[:, :, :, ixzero, iyzero].set(0.0) if exclude_zero_mode else nl


def linear_precompute(
    geometry: Dict[str, jnp.ndarray], params: GKParams
) -> Dict[str, jnp.ndarray]:
    """Precompute static geometry-dependent coefficients and Bessel terms."""

    def _parallel_coefficients(
        pos_par_class: jnp.ndarray, table: jnp.ndarray
    ) -> jnp.ndarray:
        idx = jnp.clip(jnp.asarray(pos_par_class, dtype=jnp.int32) + 2, 0, 4)
        return jnp.moveaxis(table[idx] / 12.0, -1, 0)

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

    # broadcasting into 5D [vpar, mu, s, kx, ky]
    vp2 = jnp.reshape(vpgr**2, (-1, 1, 1, 1, 1))
    vp = jnp.reshape(vpgr, (-1, 1, 1, 1, 1))
    mu = jnp.reshape(mugr, (1, -1, 1, 1, 1))
    bn_b = jnp.reshape(bn, (1, 1, -1, 1, 1))
    ffun_b = jnp.reshape(ffun, (1, 1, -1, 1, 1))
    gfun_b = jnp.reshape(gfun, (1, 1, -1, 1, 1))
    efun_b = jnp.reshape(efun, (1, 1, -1, 1, 1))
    kx_b = jnp.reshape(kx, (1, 1, 1, -1, 1))
    ky_b = jnp.reshape(ky, (1, 1, 1, 1, -1))

    # Bessel J0 evaluation
    little_g = jnp.asarray(geometry["little_g"], dtype=jnp.float64)
    krloc_sq = (
        ky_b**2 * jnp.reshape(little_g[:, 0], (1, 1, -1, 1, 1))
        + 2.0 * ky_b * kx_b * jnp.reshape(little_g[:, 1], (1, 1, -1, 1, 1))
        + kx_b**2 * jnp.reshape(little_g[:, 2], (1, 1, -1, 1, 1))
    )
    krloc = jnp.sqrt(jnp.maximum(krloc_sq, 0.0))
    sz = jnp.where(jnp.abs(params.signz) < 1e-15, 1.0, params.signz)
    b_arg = (
        params.mas
        * params.vthrat
        * krloc
        * jnp.sqrt(jnp.maximum(2.0 * mu / jnp.maximum(bn_b, 1e-15), 0.0))
        / sz
    )
    bessel = j0(b_arg)

    # Maxwellian and Linear drive
    t_rat = params.tmp / params.tgrid
    fmax = (
        (params.de / params.dgrid)
        * jnp.exp(-(vp2 + 2.0 * bn_b * mu) / t_rat)
        / (jnp.sqrt(t_rat * jnp.pi) ** 3)
    )
    et = (vp2 + 2.0 * bn_b * mu) / t_rat - 1.5
    dmax_ek = (params.rln + params.rlt * et) * fmax * (efun_b * ky_b)

    # Advection and Dissipation
    ed = vp2 + bn_b * mu
    drift_x = ed * jnp.reshape(dfun[:, 0], (1, 1, -1, 1, 1)) / sz
    drift_y = ed * jnp.reshape(dfun[:, 1], (1, 1, -1, 1, 1)) / sz

    # characteristic advection speeds
    upar = -ffun_b * params.vthrat * vp
    utrap = params.vthrat * mu * bn_b * gfun_b

    vp_rms = jnp.asarray(geometry.get("vpgr_rms", params.dvp), dtype=jnp.float64)
    mu_rms = jnp.asarray(geometry.get("mugr_rms", 1.0), dtype=jnp.float64)
    idisp = jnp.asarray(params.idisp, dtype=jnp.int32)
    use_abs = jnp.logical_or(jnp.equal(idisp, 1), jnp.equal(idisp, -1))

    abs_par = jnp.where(
        use_abs, jnp.abs(upar), jnp.abs(ffun_b * params.vthrat * vp_rms)
    )
    abs_vp = jnp.where(
        use_abs, jnp.abs(utrap), jnp.abs(params.vthrat * bn_b * gfun_b * mu_rms)
    )

    # Stencils
    pos_par = jnp.asarray(geometry["pos_par_grid_class"], dtype=jnp.int32)

    # FFT Metadata
    ixzero = jnp.asarray(
        geometry.get("ixzero", jnp.argmin(jnp.abs(kx))), dtype=jnp.int32
    )
    iyzero = jnp.asarray(
        geometry.get("iyzero", jnp.argmin(jnp.abs(ky))), dtype=jnp.int32
    )
    mphi, mphiw3 = extended_firstdim_fft_size(nky)
    mrad = extended_seconddim_fft_size(nkx)

    return {
        "kx_b": kx_b,
        "ky_b": ky_b,
        "bessel": bessel,
        "fmaxwl": fmax,
        "tmp0": jnp.asarray(params.tmp, dtype=jnp.float64),
        "signz0": jnp.asarray(params.signz, dtype=jnp.float64),
        "drift_x": drift_x,
        "drift_y": drift_y,
        "dmaxwel_fm_ek": dmax_ek,
        "upar": upar,
        "utrap": utrap,
        "abs_dum2_par": abs_par,
        "abs_dum2_vp": abs_vp,
        "term7_fac": -params.signz * ffun_b * params.vthrat * vp * fmax / params.tmp,
        "hyper": -(
            jnp.abs(params.disp_y)
            * (ky_b / jnp.maximum(params.kymax, 1e-15))
            ** jnp.where(params.disp_y < 0.0, 2.0, 4.0)
            + jnp.abs(params.disp_x)
            * (kx_b / jnp.maximum(params.kxmax, 1e-15))
            ** jnp.where(params.disp_x < 0.0, 2.0, 4.0)
        ),
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
    }


def linear_rhs(
    df: jnp.ndarray,
    geometry: Dict[str, jnp.ndarray],
    params: GKParams,
    pre: Dict[str, jnp.ndarray],
    phi: Optional[jnp.ndarray] = None,
) -> jnp.ndarray:
    """Assemble the linear RHS contribution."""
    if phi is None:
        phi, _ = get_integrals(df, geometry, params=params, include_fluxes=False)
    phi_b = jnp.reshape(phi, (1, 1, phi.shape[0], phi.shape[1], phi.shape[2]))

    def _apply_parallel(field, coeffs):
        out = jnp.zeros_like(field)
        for i in range(9):
            s_map = jnp.asarray(geometry["s_shift"], dtype=jnp.int32)[i]
            kx_map = jnp.asarray(geometry["kx_shift"], dtype=jnp.int32)[i]
            valid = jnp.asarray(geometry["valid_shift"], dtype=jnp.bool_)[i]
            ky_idx = jnp.reshape(
                jnp.arange(field.shape[-1], dtype=jnp.int32), (1, 1, -1)
            )
            shifted = jnp.where(
                valid[None, None, :, :, :], field[:, :, s_map, kx_map, ky_idx], 0.0
            )
            out = out + coeffs[i][None, None, :, :, :] * shifted
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

    # Streaming and parallel dissipation
    term_i = (
        pre["upar"]
        * jnp.where(
            pre["upar"] > 0.0,
            _apply_parallel(df, pre["s_d1_ipos"]),
            _apply_parallel(df, pre["s_d1_ineg"]),
        )
        / pre["sgr_dist"]
    )

    term_par_diss = (
        jnp.asarray(params.disp_par, dtype=jnp.float64)
        * pre["abs_dum2_par"]
        * jnp.where(
            pre["upar"] > 0.0,
            _apply_parallel(df, pre["s_d4_ipos"]),
            _apply_parallel(df, pre["s_d4_ineg"]),
        )
    ) / pre["sgr_dist"]

    # Trapping and velocity dissipation
    term_iv = pre["utrap"] * _apply_vpar(df, stencils.VPAR_D1) / pre["dvp"]
    term_vp_diss = (
        jnp.asarray(params.disp_vp, dtype=jnp.float64)
        * pre["abs_dum2_vp"]
        * _apply_vpar(df, stencils.VPAR_D4)
        / pre["dvp"]
    )

    # Magnetic drift
    kdotvd = pre["drift_x"] * pre["kx_b"] + pre["drift_y"] * pre["ky_b"]

    # Potential drives
    gyro_phi = pre["bessel"] * phi_b
    term_vii = (
        pre["term7_fac"]
        * jnp.where(
            pre["term7_fac"] < 0.0,
            _apply_parallel(gyro_phi, pre["s_d1_ipos"]),
            _apply_parallel(gyro_phi, pre["s_d1_ineg"]),
        )
        / pre["sgr_dist"]
    )

    return (
        term_i
        + term_par_diss
        + term_iv
        + term_vp_diss
        - 1j * kdotvd * df
        + pre["hyper"] * df
        + 1j
        * jnp.asarray(params.drive_scale, dtype=jnp.float64)
        * (
            pre["dmaxwel_fm_ek"]
            - pre["signz0"] * kdotvd * (pre["fmaxwl"] / jnp.maximum(pre["tmp0"], 1e-15))
        )
        * gyro_phi
        + term_vii
    )


def init_f(
    geometry: Dict[str, jnp.ndarray],
    amp_init_real: float = 1.0e-4,
    amp_init_imag: float = 0.0,
    normalize_per_toroidal_mode: bool = True,
    norm_eps: float = 1.0e-14,
) -> jnp.ndarray:
    """
    Initialize the distribution function with a parallel cosine^2 profile.

    DISCLAIMER: This initialization implementation is currently experimental
    and may not correctly reproduce GKW seed parity. Use with caution.
    """
    nv, nmu, ns, nkx, nky = (
        len(geometry["intvp"]),
        len(geometry["intmu"]),
        len(geometry["ints"]),
        len(geometry["kxrh"]),
        len(geometry["krho"]),
    )
    sgrid = jnp.asarray(
        geometry.get("sgrid", jnp.linspace(-0.5, 0.5, ns, dtype=jnp.float64)),
        dtype=jnp.float64,
    )

    amp = jnp.asarray(amp_init_real, dtype=jnp.float64) + 1j * jnp.asarray(
        amp_init_imag, dtype=jnp.float64
    )
    prof = amp * (jnp.cos(2.0 * jnp.pi * sgrid) + 1.0)
    df = jnp.broadcast_to(
        jnp.reshape(prof, (1, 1, ns, 1, 1)), (nv, nmu, ns, nkx, nky)
    ).astype(jnp.complex128)

    if nky > 1:
        iy0 = int(
            jnp.asarray(
                geometry.get(
                    "iyzero",
                    jnp.argmin(
                        jnp.abs(jnp.asarray(geometry["krho"], dtype=jnp.float64))
                    ),
                )
            ).item()
        )
        df = df.at[..., iy0].set(0.0)

    if normalize_per_toroidal_mode:
        df, _, _ = normalize_per_ky(df, geometry, GKParams(norm_eps=norm_eps))
    return df


def advance_state(
    state: GKState,
    params: GKParams,
    is_window_end: jnp.ndarray,
    dominant_amp: jnp.ndarray,
    norm_fac: jnp.ndarray,
) -> GKState:
    """
    Internal metadata update for simulation diagnostics.

    Calculates exponential growth rates (gamma = log(A2/A1)/dt) and tracks the
    accumulated normalization factor across integration windows.
    """
    new_step = state.step + jnp.array(1, dtype=jnp.int32)
    new_time = state.time + jnp.array(params.dt, dtype=jnp.float64)

    # growth rate calculation at normalization boundaries
    valid_growth = jnp.logical_and(
        state.window_start_amp > params.norm_eps,
        dominant_amp > params.norm_eps,
    )
    growth_dt = jnp.array(params.dt * params.naverage, dtype=jnp.float64)

    growth_rate = jnp.where(
        jnp.logical_and(is_window_end, valid_growth),
        jnp.log(dominant_amp / state.window_start_amp) / growth_dt,
        state.last_growth_rate,
    )
    # reset baseline for the next diagnostic window
    new_window_start_amp = jnp.where(
        is_window_end,
        jnp.array(1.0, dtype=jnp.float64),
        state.window_start_amp,
    )

    return GKState(
        time=new_time,
        step=new_step,
        accumulated_norm_factor=state.accumulated_norm_factor * norm_fac,
        window_start_amp=new_window_start_amp,
        last_growth_rate=growth_rate,
    )


def gkstep_single(
    prev_df: jnp.ndarray,
    geometry: Dict[str, jnp.ndarray],
    params: GKParams,
    state: GKState,
) -> Tuple[
    jnp.ndarray,
    Tuple[jnp.ndarray, Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]],
    GKState,
]:
    """
    Perform a single small-step (dt) time integration using an explicit RK4 scheme.
    """
    dt = jnp.array(params.dt, dtype=jnp.float64)
    pre = linear_precompute(geometry, params)

    def _rhs(df: jnp.ndarray) -> jnp.ndarray:
        # electrostatic Poisson solve
        phi_local, _ = get_integrals(df, geometry, params=params, include_fluxes=False)
        rhs_linear = linear_rhs(df, geometry, params, pre, phi=phi_local)

        def _with_nl(_: None) -> jnp.ndarray:
            # add nonlinear Term III advection
            rhs_nl = nonlinear_term_iii(df, phi_local, geometry, pre)
            return rhs_linear + rhs_nl

        def _without_nl(_: None) -> jnp.ndarray:
            return rhs_linear

        # conditional inclusion of Term III
        term_iii_on = jnp.logical_and(
            jnp.asarray(params.non_linear, dtype=jnp.bool_),
            jnp.asarray(params.enable_term_iii, dtype=jnp.bool_),
        )
        return jax.lax.cond(term_iii_on, _with_nl, _without_nl, operand=None)

    # explicit Runge-Kutta 4th order integration
    k1 = _rhs(prev_df)
    k2 = _rhs(prev_df + 0.5 * dt * k1)
    k3 = _rhs(prev_df + 0.5 * dt * k2)
    k4 = _rhs(prev_df + dt * k3)

    next_df_raw = prev_df + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

    # determine if this step marks a large-step normalization boundary
    new_step = state.step + jnp.array(1, dtype=jnp.int32)
    is_window_end = jnp.equal(jnp.mod(new_step, params.naverage), 0)

    # normalization is usually only applied in linear regimes
    do_normalize = jnp.logical_and(
        is_window_end,
        jnp.logical_not(jnp.asarray(params.non_linear, dtype=jnp.bool_)),
    )

    def _apply_norm(_: None):
        return normalize_per_ky(next_df_raw, geometry, params)

    def _skip_norm(_: None):
        return (
            next_df_raw,
            jnp.array(1.0, dtype=jnp.float64),
            state.window_start_amp,
        )

    # conditional mode normalization
    next_df, norm_factor, dominant_amp = jax.lax.cond(
        do_normalize,
        _apply_norm,
        _skip_norm,
        operand=None,
    )

    # final field calculation for output
    phi, fluxes = get_integrals(next_df, geometry, params=params)
    next_state = advance_state(state, params, do_normalize, dominant_amp, norm_factor)
    return next_df, (phi, fluxes), next_state


def gksolve(
    df: jnp.ndarray,
    geometry: Dict[str, jnp.ndarray],
    params: GKParams,
    state: GKState,
    n_steps: int = 1,
) -> Tuple[
    jnp.ndarray, Tuple[jnp.ndarray, Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]], GKState
]:
    """
    Gyrokinetics solver forward.

    Executes multiple time steps using jax.lax.scan for efficient compilation.
    Returns the final distribution function, the final fields/fluxes, and final state.

    Args:
        df: Initial distribution function.
        geometry: Geometry metadata.
        params: Solver parameters.
        state: Diagnostic metadata state.
        n_steps: Number of small steps to execute.

    Returns:
        Tuple of (final_df, (final_phi, final_fluxes), final_state).
    """

    def _scan_body(carry, _):
        curr_df, curr_state = carry
        next_df, out, next_state = gkstep_single(curr_df, geometry, params, curr_state)
        return (next_df, next_state), None

    (final_df, final_state), _ = jax.lax.scan(
        _scan_body, (df, state), None, length=n_steps
    )

    # Calculate final diagnostics only at the end of the block
    phi, fluxes = get_integrals(final_df, geometry, params=params)

    return final_df, (phi, fluxes), final_state


def simulate(
    config_path: str,
    output_dir: str = "outputs",
    checkpoint_interval: Optional[int] = None,
    resume_from: Optional[str] = None,
    resume_k_file: Optional[str] = None,
    verbose: bool = True,
    **params_overrides,
) -> Tuple[jnp.ndarray, GKState]:
    """
    High-level entry point to run a simulation from a YAML config.
    Supports resuming from .npz checkpoints or specific GKW K* files.
    """
    from gyaradax.geometry import load_geometry
    from gyaradax.utils import (
        save_checkpoint,
        load_checkpoint,
        load_gkw_k_dump,
        read_gkw_dump_time,
    )

    # 1. Load config and params
    cfg = load_config(config_path)

    # Extract simulation-level overrides before building GKParams
    total_steps = int(
        params_overrides.pop("n_steps", getattr(cfg.solver, "n_steps", 400))
    )
    interval = checkpoint_interval or int(
        params_overrides.pop(
            "checkpoint_interval", getattr(cfg.solver, "dump_interval", 40)
        )
    )

    params = gkparams_from_config(cfg, **params_overrides)
    data_dir = cfg.run.data_dir
    geometry = load_geometry(data_dir)

    # Use naverage as default interval if not specified
    interval = interval or params.naverage

    # 2. Determine initial condition
    state = default_state()
    if resume_from:
        if verbose:
            print(f"Resuming from checkpoint: {resume_from}")
        ckpt = load_checkpoint(resume_from)
        df = ckpt["df"]
        state = GKState(
            time=ckpt["time"],
            step=ckpt["step"],
            accumulated_norm_factor=ckpt["accumulated_norm_factor"],
            window_start_amp=ckpt["window_start_amp"],
            last_growth_rate=ckpt["last_growth_rate"],
        )
    elif resume_k_file:
        if verbose:
            print(f"Resuming from K-file: {resume_k_file}")
        res = (
            len(geometry["intvp"]),
            len(geometry["intmu"]),
            len(geometry["ints"]),
            len(geometry["kxrh"]),
            len(geometry["krho"]),
        )
        df = load_gkw_k_dump(resume_k_file, res)
        # Attempt to load time from .dat file
        dat_path = resume_k_file + ".dat"
        if os.path.exists(dat_path):
            t_start = read_gkw_dump_time(dat_path)
            state = GKState(
                time=jnp.array(t_start, dtype=jnp.float64),
                step=jnp.array(0, dtype=jnp.int32),
                accumulated_norm_factor=jnp.array(1.0, dtype=jnp.float64),
                window_start_amp=jnp.array(1.0, dtype=jnp.float64),
                last_growth_rate=jnp.array(0.0, dtype=jnp.float64),
            )
            if verbose:
                print(f"Loaded start time from {dat_path}: {t_start:.4f}")
    else:
        if verbose:
            print("WARNING: Initializing new simulation from experimental init_f.")
            print("         This profile may not correctly reproduce GKW seed parity.")
        df = init_f(geometry, norm_eps=params.norm_eps)
    os.makedirs(output_dir, exist_ok=True)

    # Initial field calculation
    phi, fluxes = get_integrals(df, geometry, params=params)
    curr_df, curr_state = df, state

    # History of diagnostics
    all_diagnostics = []

    # 4. Simulation loop
    if verbose:
        print(f"Starting simulation: total_steps={total_steps}, interval={interval}")
        print(
            f"Initial state: step={int(curr_state.step)}, time={float(curr_state.time):.4f}"
        )

    from gyaradax.diag import get_diagnostics

    start_step = int(curr_state.step)
    for _ in range(start_step, total_steps, interval):
        # Save snapshot checkpoint
        ckpt_name = f"step_{int(curr_state.step):06d}.npz"
        save_checkpoint(
            os.path.join(output_dir, ckpt_name),
            curr_df,
            phi,
            fluxes,
            curr_state,
            geometry,
        )

        # Run chunk
        steps_to_run = min(interval, total_steps - int(curr_state.step))
        if steps_to_run <= 0:
            break

        curr_df, (phi, fluxes), curr_state = gksolve(
            curr_df, geometry, params, curr_state, n_steps=steps_to_run
        )

        # Compute and accumulate diagnostics only at the block boundary
        d = get_diagnostics(phi, fluxes, curr_state)
        all_diagnostics.append(d)

        if verbose:
            eflux = float(fluxes[1])
            growth = float(curr_state.last_growth_rate)
            print(
                f"Step {int(curr_state.step):06d} | Time {float(curr_state.time):.4f} | Heat Flux {eflux:.4e} | Growth {growth:.4e}"
            )

    # Final save snapshot
    ckpt_name = f"step_{int(curr_state.step):06d}.npz"
    save_checkpoint(
        os.path.join(output_dir, ckpt_name), curr_df, phi, fluxes, curr_state, geometry
    )

    # Save consolidated history
    if all_diagnostics:
        history_merged = {}
        for key in all_diagnostics[0].keys():
            history_merged[key] = jnp.stack([d[key] for d in all_diagnostics])

        np.savez(
            os.path.join(output_dir, "history.npz"),
            **{k: np.array(v) for k, v in history_merged.items()},
        )
        if verbose:
            print(
                f"Saved simulation history to {os.path.join(output_dir, 'history.npz')}"
            )

    if verbose:
        print("Simulation complete.")

    return curr_df, curr_state
