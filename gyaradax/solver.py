import jax
import jax.numpy as jnp
import math
from dataclasses import dataclass
from typing import Dict, Tuple, Any

from gyaradax.integrals import get_integrals, j0
from gyaradax.geometry import load_runtime_params

# ensure fp64 everywhere for scientific precision
jax.config.update("jax_enable_x64", True)


Array = jnp.ndarray


def _center_5pt(stencil5):
    """
    Center a 5-point finite difference stencil into a 9-point zero-padded array.
    
    Args:
        stencil5: Sequence of 5 coefficients representing the central stencil.
        
    Returns:
        List of 9 coefficients with zero-padding on both ends.
    """
    out = [0.0] * 9
    out[2:7] = stencil5
    return out


# differential stencils from linear_terms.f90::differential_scheme, order='fourth_order'.
# these correspond to the fortran implementation of upwinded fourth-order finite differences.
_D1_IPW_POS = jnp.asarray(
    [
        _center_5pt([0.0, 0.0, -18.0, 24.0, -6.0]),
        [0.0, 0.0, 0.0, -4.0, -6.0, 12.0, -2.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, -8.0, 0.0, 8.0, -1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, -8.0, 0.0, 8.0, 0.0, 0.0, 0.0],
        _center_5pt([0.0, -6.0, 0.0, 0.0, 0.0]),
    ],
    dtype=jnp.float64,
)
_D1_IPW_NEG = jnp.asarray(
    [
        _center_5pt([0.0, 0.0, 0.0, 6.0, 0.0]),
        [0.0, 0.0, 0.0, -8.0, 0.0, 8.0, -1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, -8.0, 0.0, 8.0, -1.0, 0.0, 0.0],
        [0.0, 0.0, 2.0, -12.0, 6.0, 4.0, 0.0, 0.0, 0.0],
        _center_5pt([6.0, -24.0, 18.0, 0.0, 0.0]),
    ],
    dtype=jnp.float64,
)

_D4_IPW_POS = jnp.asarray(
    [
        [0.0] * 9,
        [0.0] * 9,
        _center_5pt([-1.0, 4.0, -6.0, 4.0, -1.0]),
        [0.0, 0.0, -1.0, 4.0, -6.0, 4.0, 0.0, 0.0, 0.0],
        _center_5pt([0.0, 12.0, -24.0, 0.0, 0.0]),
    ],
    dtype=jnp.float64,
)
_D4_IPW_NEG = jnp.asarray(
    [
        _center_5pt([0.0, 0.0, -24.0, 12.0, 0.0]),
        [0.0, 0.0, 0.0, 4.0, -6.0, 4.0, -1.0, 0.0, 0.0],
        _center_5pt([-1.0, 4.0, -6.0, 4.0, -1.0]),
        [0.0] * 9,
        [0.0] * 9,
    ],
    dtype=jnp.float64,
)

_VPAR_D1 = jnp.asarray([1.0, -8.0, 0.0, 8.0, -1.0], dtype=jnp.float64) / 12.0
_VPAR_D4 = jnp.asarray([-1.0, 4.0, -6.0, 4.0, -1.0], dtype=jnp.float64) / 12.0


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class GKParams:
    """
    Runtime controls and numerical hyperparameters for the electrostatic solver.

    This dataclass mirrors the GKW 'control' namelist and manages switches for 
    time-stepping, dissipation coefficients (Term IV/VIII), and nonlinear activation.
    
    Attributes:
        dt: Small time step for the RK4 integrator.
        naverage: Number of small steps before diagnostic output and normalization.
        disp_par: Coefficient for parallel dissipation (stabilizes Term I).
        disp_vp: Coefficient for velocity space dissipation (Term IV smoothing).
        disp_x: Radial hyper-dissipation coefficient.
        disp_y: Binormal hyper-dissipation coefficient.
        idisp: Dissipation scheme identifier (e.g., idisp=2 for speed-based).
        drive_scale: Multiplier for the electrostatic drive (Term V/VIII).
        norm_eps: Numerical floor for amplitude-based normalization.
        non_linear: Enable nonlinear ExB advection (Term III).
        enable_term_iii: Switch for the pseudospectral Term III implementation.
    """

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

    def tree_flatten(self):
        leaves = (
            self.dt,
            self.naverage,
            self.disp_par,
            self.disp_vp,
            self.disp_x,
            self.disp_y,
            self.idisp,
            self.drive_scale,
            self.norm_eps,
            self.non_linear,
            self.enable_term_iii,
        )
        return leaves, None

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

    time: Array
    step: Array
    accumulated_norm_factor: Array
    window_start_amp: Array
    last_growth_rate: Array

    def tree_flatten(self):
        leaves = (
            self.time,
            self.step,
            self.accumulated_norm_factor,
            self.window_start_amp,
            self.last_growth_rate,
        )
        return leaves, None

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
    params = GKParams(
        dt=float(runtime.get("dtim", 0.01)),
        naverage=int(runtime.get("naverage", 40)),
        disp_par=float(runtime.get("disp_par", 1.0)),
        disp_vp=float(runtime.get("disp_vp", 0.2)),
        disp_x=float(runtime.get("disp_x", 0.1)),
        disp_y=float(runtime.get("disp_y", 0.1)),
        non_linear=bool(runtime.get("non_linear", False)),
    )
    if overrides:
        return GKParams(**{**params.__dict__, **overrides})
    return params


def gkparams_from_input_dat(input_dat_path: str, **overrides) -> GKParams:
    """
    Load runtime controls directly from a GKW 'input.dat' file.
    
    Args:
        input_dat_path: Path to the GKW input.dat configuration.
        overrides: Manual parameter overrides.
        
    Returns:
        Configured GKParams instance.
    """
    runtime = load_runtime_params(input_dat_path)
    return gkparams_from_runtime(runtime, **overrides)


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
        config: Configuration object with a 'solver' section (dt, naverage, etc.).
        overrides: Manual parameter overrides.
        
    Returns:
        Configured GKParams instance.
    """
    solver_cfg = config.solver
    params = GKParams(
        dt=float(getattr(solver_cfg, "dt", 0.01)),
        naverage=int(getattr(solver_cfg, "naverage", 40)),
        disp_par=float(getattr(solver_cfg, "disp_par", 1.0)),
        disp_vp=float(getattr(solver_cfg, "disp_vp", 0.2)),
        disp_x=float(getattr(solver_cfg, "disp_x", 0.1)),
        disp_y=float(getattr(solver_cfg, "disp_y", 0.1)),
        non_linear=bool(getattr(solver_cfg, "non_linear", False)),
        enable_term_iii=bool(getattr(solver_cfg, "enable_term_iii", True)),
    )
    if overrides:
        return GKParams(**{**params.__dict__, **overrides})
    return params


def _kx_ky_grids(geometry: Dict[str, Array]) -> Tuple[Array, Array]:
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


def _mode_amplitude(phi: Array, geometry: Dict[str, Array], eps: float) -> Array:
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


def _normalize_per_ky(
    df: Array, geometry: Dict[str, Array], params: GKParams
) -> Tuple[Array, Array, Array]:
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
    phi, _ = get_integrals(df, geometry)
    amp_per_ky = _mode_amplitude(phi, geometry, params.norm_eps)
    # prevent division by zero for stable or zero modes
    safe_amp = jnp.where(amp_per_ky < params.norm_eps, 1.0, amp_per_ky)
    inv = 1.0 / safe_amp
    # apply normalization factor across velocity and space dimensions
    normalized_df = df * jnp.reshape(inv, (1, 1, 1, 1, inv.shape[0]))
    dominant_amp = jnp.max(safe_amp)
    return normalized_df, jnp.mean(inv), dominant_amp


def _parallel_coefficients(pos_par_class: Array, table: Array) -> Array:
    """
    Select appropriate parallel finite-difference coefficients based on boundary class.
    
    GKW uses different stencils at the parallel boundaries (open/periodic) to 
    maintain high-order accuracy and upwinding.
    
    Args:
        pos_par_class: Grid of boundary markers (-2 to 2).
        table: 5x9 table of differential coefficients.
        
    Returns:
        Mapped coefficients in [9, s, kx, ky] format for stencil application.
    """
    idx = jnp.asarray(pos_par_class, dtype=jnp.int32) + 2
    idx = jnp.clip(idx, 0, 4)
    coeff = table[idx] / 12.0
    return jnp.moveaxis(coeff, -1, 0)


def _shift_parallel(field: Array, geometry: Dict[str, Array], shift_idx: int) -> Array:
    """
    Execute parallel coordinate shift accounting for ballooning boundary connectivity.
    
    This function implements the complex kx-chain remapping required when shifting 
    across parallel boundaries in spectral gyrokinetics.
    
    Args:
        field: Input phase-space field to be shifted.
        geometry: Geometry dictionary containing precomputed connectivity maps.
        shift_idx: Index into the 9-point parallel stencil.
        
    Returns:
        Shifted field with correct kx-remapping and zero-padding for open boundaries.
    """
    s_map = jnp.asarray(geometry["s_shift"], dtype=jnp.int32)[shift_idx]
    kx_map = jnp.asarray(geometry["kx_shift"], dtype=jnp.int32)[shift_idx]
    valid = jnp.asarray(geometry["valid_shift"], dtype=jnp.bool_)[shift_idx]

    nky = field.shape[-1]
    ky_idx = jnp.arange(nky, dtype=jnp.int32)
    ky_idx = jnp.reshape(ky_idx, (1, 1, nky))

    # apply precomputed indices for ballooning connectivity
    shifted = field[:, :, s_map, kx_map, ky_idx]
    # zero out connections that fall outside open boundaries
    return jnp.where(valid[None, None, :, :, :], shifted, 0.0)


def _apply_parallel_stencil(
    field: Array, coeffs: Array, geometry: Dict[str, Array]
) -> Array:
    """
    Apply a 9-point finite difference stencil in the field-line coordinate s.
    
    This supports upwinded differentials for the streaming term (Term I) and 
    gyro-averaged field gradients (Term VII).
    
    Args:
        field: Phase-space field [..., ns, nkx, nky].
        coeffs: Parallel coefficients [9, ns, nkx, nky].
        geometry: Geometry dictionary for boundary shifts.
        
    Returns:
        Approximated parallel derivative or dissipation contribution.
    """
    out = jnp.zeros_like(field)
    for shift_idx in range(9):
        shifted = _shift_parallel(field, geometry, shift_idx)
        out = out + coeffs[shift_idx][None, None, :, :, :] * shifted
    return out


def _apply_vpar_stencil(field: Array, coeffs: Array) -> Array:
    """
    Apply a centered 5-point stencil in the parallel velocity coordinate vpar.
    
    This is primarily used for trapping effects (Term IV) and velocity-space dissipation.
    
    Args:
        field: Phase-space field [nvpar, ...].
        coeffs: Finite difference coefficients.
        
    Returns:
        Approximated vpar derivative with zero-padding at velocity boundaries.
    """
    nvpar = field.shape[0]
    base = jnp.arange(nvpar, dtype=jnp.int32)
    out = jnp.zeros_like(field)
    for c, shift in zip(coeffs, (-2, -1, 0, 1, 2)):
        idx = base + shift
        valid = jnp.logical_and(idx >= 0, idx < nvpar)
        idx_clip = jnp.clip(idx, 0, nvpar - 1)
        shifted = jnp.take(field, idx_clip, axis=0)
        # enforce zero-distribution boundary condition in velocity space
        out = out + c * jnp.where(valid[:, None, None, None, None], shifted, 0.0)
    return out


def _prime_factors_smallereq_than(number: int, max_prime: int) -> bool:
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


def _extended_firstdim_fft_size(nmod: int) -> Tuple[int, int]:
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
    while not _prime_factors_smallereq_than(posspace_size, 7):
        posspace_size += 2
    # prefer powers of two if within reasonable range
    for i in range(1, 9):
        cand = posspace_size + 2 * i
        if _prime_factors_smallereq_than(cand, 2):
            posspace_size = cand
            break
    kgrid_size = int(math.floor(posspace_size / 2.0) + 1)
    return posspace_size, kgrid_size


def _extended_seconddim_fft_size(nx: int) -> int:
    """Calculate the dealiased FFT size for the radial (kx) dimension."""
    dum = int(math.ceil(1.5 * float(nx + 1)) + 1)
    while not _prime_factors_smallereq_than(dum, 7):
        dum += 1
    # optimize for power-of-two FFTs
    for i in range(1, 9):
        cand = dum + i
        if _prime_factors_smallereq_than(cand, 2):
            dum = cand
            break
    return dum


def _build_jind(nkx: int, mrad: int, ixzero: int) -> Array:
    """
    Map physical kx modes to the Fortran-style FFT storage indexing.
    
    This handles the split between positive and negative radial wavevectors 
    required for the 2D Real-to-Complex FFT layout.
    """
    ix = jnp.arange(nkx, dtype=jnp.int32)
    return jnp.where(ix >= ixzero, ix - ixzero, mrad + ix - ixzero)


def _pack_half_spectrum(spec_kxky: Array, jind: Array, mrad: int, mphiw3: int) -> Array:
    """Pack physical spectral modes into a zero-padded dealiased FFT buffer."""
    out_shape = spec_kxky.shape[:-2] + (mrad, mphiw3)
    out = jnp.zeros(out_shape, dtype=jnp.complex128)
    nky = spec_kxky.shape[-1]
    return out.at[..., jind, :nky].set(spec_kxky)


def _unpack_half_spectrum(spec_half: Array, jind: Array, nky: int) -> Array:
    """Extract physical spectral modes from a dealiased FFT storage buffer."""
    return spec_half[..., jind, :nky]


def _nonlinear_term_iii(
    df: Array,
    phi: Array,
    geometry: Dict[str, Array],
    pre: Dict[str, Array],
    *,
    efun_sign: float = 1.0,
    fft_prefactor: complex = 1.0 + 0.0j,
    exclude_zero_mode: bool = True,
) -> Array:
    """
    Calculate Nonlinear Term III (ExB Advection) using the pseudospectral method.
    
    This term represents the advection of the distribution function by the 
    fluctuating ExB velocity: v_E . grad(f). It uses transforms to dealiased 
    real-space grids, evaluates the Poisson bracket, and transforms back to spectral space.
    
    Term III = sum_{k'+k''=k} (k' x k'')_s phi(k') f(k'')
    
    Args:
        df: Complex distribution function [vpar, mu, s, kx, ky].
        phi: Complex electrostatic potential [ns, nkx, nky].
        geometry: Geometry metadata.
        pre: Precomputed coefficients and FFT metadata.
        efun_sign: Directional sign for the ExB drift.
        fft_prefactor: Additional complex scaling for the result.
        exclude_zero_mode: Ensure the (0,0) zonal mode remains zero.
        
    Returns:
        Nonlinear RHS contribution in spectral space.
    """
    mrad = pre["nl_mrad"]
    mphi = pre["nl_mphi"]
    mphiw3 = pre["nl_mphiw3"]
    fft_scale = pre["nl_fft_scale"]
    jind = pre["nl_jind"]
    kx2d = pre["nl_kx2d"]
    ky2d = pre["nl_ky2d"]
    bessel = pre["bessel"]
    dum_s = pre["nl_dum_s"]
    ixzero = pre["ixzero"]
    iyzero = pre["iyzero"]
    nky = df.shape[-1]

    # vectorize over parallel grid to manage memory bandwidth
    df_by_s = jnp.moveaxis(df, 2, 0)
    bessel_by_s = jnp.moveaxis(bessel, 2, 0)

    def _per_s(df_s: Array, phi_s: Array, bessel_s: Array, dum: Array) -> Array:
        # compute gradients in spectral space
        gyro_phi = bessel_s * phi_s[None, None, :, :]
        grad_phi_y_k = 1j * ky2d[None, None, :, :] * gyro_phi
        grad_phi_x_k = 1j * kx2d[None, None, :, :] * gyro_phi

        grad_f_x_k = 1j * kx2d[None, None, :, :] * df_s
        grad_f_y_k = 1j * ky2d[None, None, :, :] * df_s

        # transform all gradients to real space with dealiasing
        ar = jnp.fft.irfft2(
            _pack_half_spectrum(grad_phi_y_k, jind, mrad, mphiw3),
            s=(mrad, mphi),
            axes=(-2, -1),
            norm="backward",
        )
        br = jnp.fft.irfft2(
            _pack_half_spectrum(grad_phi_x_k, jind, mrad, mphiw3),
            s=(mrad, mphi),
            axes=(-2, -1),
            norm="backward",
        )
        cr = jnp.fft.irfft2(
            _pack_half_spectrum(grad_f_x_k, jind, mrad, mphiw3),
            s=(mrad, mphi),
            axes=(-2, -1),
            norm="backward",
        )
        dr = jnp.fft.irfft2(
            _pack_half_spectrum(grad_f_y_k, jind, mrad, mphiw3),
            s=(mrad, mphi),
            axes=(-2, -1),
            norm="backward",
        )

        # evaluate the bracket: V_E dot grad(f) = (dphi/dy * df/dx - dphi/dx * df/dy)
        nl_real = (efun_sign * dum) * (ar * cr - br * dr)

        # transform back to spectral space with explicit normalization
        nl_half = (
            jnp.asarray(fft_prefactor, dtype=jnp.complex128)
            * jnp.asarray(fft_scale, dtype=jnp.complex128)
            * jnp.fft.rfft2(
                nl_real,
                s=(mrad, mphi),
                axes=(-2, -1),
                norm="backward",
            )
        )
        return _unpack_half_spectrum(nl_half, jind, nky)

    # vmap the parallel-slice calculation
    nl_by_s = jax.vmap(_per_s, in_axes=(0, 0, 0, 0))(df_by_s, phi, bessel_by_s, dum_s)
    nl = jnp.moveaxis(nl_by_s, 0, 2)
    # enforce spectral convention for the zonal zero-mode
    if exclude_zero_mode:
        return nl.at[:, :, :, ixzero, iyzero].set(0.0 + 0.0j)
    return nl


def term_iii_rhs(
    df: Array,
    geometry: Dict[str, Array],
    params: GKParams | None = None,
    *,
    efun_sign: float = 1.0,
    fft_prefactor: complex = 1.0 + 0.0j,
    exclude_zero_mode: bool = True,
) -> Array:
    """
    Public diagnostic interface for the Nonlinear Term III contribution.
    
    Args:
        df: distribution function.
        geometry: geometry metadata.
        params: optional solver parameters.
        
    Returns:
        Nonlinear RHS contribution array.
    """
    if params is None:
        params = GKParams()
    pre = _linear_precompute(geometry, params)
    phi, _ = get_integrals(df, geometry)
    return _nonlinear_term_iii(
        df,
        phi,
        geometry,
        pre,
        efun_sign=efun_sign,
        fft_prefactor=fft_prefactor,
        exclude_zero_mode=exclude_zero_mode,
    )


def term_iii_fft_pack_roundtrip(spec_kxky: Array, geometry: Dict[str, Array]) -> Array:
    """
    Verify the dealiased packing and FFT roundtrip for spectral modes.
    
    This utility checks if information is preserved through the Real-to-Complex 
    FFT pipeline used by the nonlinear solver.
    """
    nkx = spec_kxky.shape[-2]
    nky = spec_kxky.shape[-1]
    kx = jnp.asarray(geometry["kxrh"], dtype=jnp.float64)
    if kx.ndim > 1:
        kx = kx[0]
    ixzero = int(jnp.asarray(geometry.get("ixzero", jnp.argmin(jnp.abs(kx)))).item())
    mphi, mphiw3 = _extended_firstdim_fft_size(nky)
    mrad = _extended_seconddim_fft_size(nkx)
    jind = _build_jind(nkx, mrad, ixzero)
    packed = _pack_half_spectrum(spec_kxky, jind, mrad, mphiw3)
    real = jnp.fft.irfft2(packed, s=(mrad, mphi), axes=(-2, -1), norm="backward")
    repacked = jnp.fft.rfft2(real, s=(mrad, mphi), axes=(-2, -1), norm="backward")
    return _unpack_half_spectrum(repacked, jind, nky)


def _linear_precompute(
    geometry: Dict[str, Array], params: GKParams
) -> Dict[str, Array]:
    """
    Precompute static geometry-dependent coefficients and gyro-averaging Bessel terms.
    
    This function handles the complex broadcasting of geometry tensors into the 
    5D phase space and evaluates the J0 Bessel function used for gyro-averaging 
    the electrostatic potential (phi).
    
    Args:
        geometry: Loaded geometry dictionary containing metric tensors and species info.
        params: Solver parameters for dissipation settings.
        
    Returns:
        Dictionary of precomputed arrays optimized for the small-step integration.
    """
    kx, ky = _kx_ky_grids(geometry)
    ns = len(geometry["ints"])
    nkx = int(kx.shape[0])
    nky = int(ky.shape[0])

    # primary geometry tensors
    vpgr = jnp.asarray(geometry["vpgr"], dtype=jnp.float64)
    mugr = jnp.asarray(geometry["mugr"], dtype=jnp.float64)
    bn = jnp.asarray(geometry["bn"], dtype=jnp.float64)
    ffun = jnp.asarray(geometry["ffun"], dtype=jnp.float64)
    gfun = jnp.asarray(geometry.get("gfun", jnp.zeros_like(bn)), dtype=jnp.float64)
    dfun = jnp.asarray(
        geometry.get("dfun", jnp.zeros((ns, 3), dtype=jnp.float64)),
        dtype=jnp.float64,
    )
    efun = jnp.asarray(geometry.get("efun", jnp.ones_like(bn)), dtype=jnp.float64)

    # species constants for the kinetic species
    mas = jnp.asarray(geometry["mas"], dtype=jnp.float64)
    tmp = jnp.asarray(geometry["tmp"], dtype=jnp.float64)
    de = jnp.asarray(geometry["de"], dtype=jnp.float64)
    signz = jnp.asarray(geometry["signz"], dtype=jnp.float64)
    vthrat = jnp.asarray(geometry["vthrat"], dtype=jnp.float64)
    rln = jnp.asarray(geometry["rln"], dtype=jnp.float64)
    rlt = jnp.asarray(geometry["rlt"], dtype=jnp.float64)

    # single species extraction (adiabatic electron setup)
    mas0 = mas[0] if mas.ndim > 0 else mas
    tmp0 = tmp[0] if tmp.ndim > 0 else tmp
    de0 = de[0] if de.ndim > 0 else de
    signz0 = signz[0] if signz.ndim > 0 else signz
    vthrat0 = vthrat[0] if vthrat.ndim > 0 else vthrat
    rln0 = rln[0] if rln.ndim > 0 else rln
    rlt0 = rlt[0] if rlt.ndim > 0 else rlt

    # scaling factors
    dgrid0 = jnp.array(1.0, dtype=jnp.float64)
    if "dgrid" in geometry:
        dgrid = jnp.asarray(geometry["dgrid"], dtype=jnp.float64)
        dgrid0 = dgrid[0] if dgrid.ndim > 0 else dgrid

    tgrid0 = jnp.array(1.0, dtype=jnp.float64)
    if "tgrid" in geometry:
        tgrid = jnp.asarray(geometry["tgrid"], dtype=jnp.float64)
        tgrid0 = tgrid[0] if tgrid.ndim > 0 else tgrid

    # broadcasting into 5D [vpar, mu, s, kx, ky]
    vp2 = jnp.reshape(vpgr**2, (vpgr.shape[0], 1, 1, 1, 1))
    vp = jnp.reshape(vpgr, (vpgr.shape[0], 1, 1, 1, 1))
    mu = jnp.reshape(mugr, (1, mugr.shape[0], 1, 1, 1))
    bn_b = jnp.reshape(bn, (1, 1, bn.shape[0], 1, 1))
    ffun_b = jnp.reshape(ffun, (1, 1, ffun.shape[0], 1, 1))
    gfun_b = jnp.reshape(gfun, (1, 1, gfun.shape[0], 1, 1))
    efun_b = jnp.reshape(efun, (1, 1, efun.shape[0], 1, 1))

    kx_b = jnp.reshape(kx, (1, 1, 1, kx.shape[0], 1))
    ky_b = jnp.reshape(ky, (1, 1, 1, 1, ky.shape[0]))

    # local perpendicular wavevector magnitude
    little_g = jnp.asarray(geometry["little_g"], dtype=jnp.float64)
    gzz = jnp.reshape(little_g[:, 0], (1, 1, ns, 1, 1))
    gez = jnp.reshape(little_g[:, 1], (1, 1, ns, 1, 1))
    gee = jnp.reshape(little_g[:, 2], (1, 1, ns, 1, 1))
    krloc_sq = ky_b**2 * gzz + 2.0 * ky_b * kx_b * gez + kx_b**2 * gee
    krloc_sq = jnp.where(krloc_sq < 0.0, 0.0, krloc_sq)
    krloc = jnp.sqrt(krloc_sq)

    # gyro-averaging kernel (J0 Bessel)
    signz_safe = jnp.where(jnp.abs(signz0) < 1.0e-15, 1.0, signz0)
    bessel_arg = (
        mas0
        * vthrat0
        * krloc
        * jnp.sqrt(jnp.maximum(2.0 * mu / jnp.maximum(bn_b, 1.0e-15), 0.0))
        / signz_safe
    )
    bessel = j0(bessel_arg)

    # maxwellian background distribution
    temp_ratio = tmp0 / jnp.maximum(tgrid0, 1.0e-15)
    fmaxwl = (
        de0
        / jnp.maximum(dgrid0, 1.0e-15)
        * jnp.exp(-(vp2 + 2.0 * bn_b * mu) / jnp.maximum(temp_ratio, 1.0e-15))
        / (jnp.sqrt(jnp.maximum(temp_ratio, 1.0e-15) * jnp.pi) ** 3)
    )

    # drift advection (Term II and VIII)
    ed = vp2 + bn_b * mu
    drift_x = ed * jnp.reshape(dfun[:, 0], (1, 1, ns, 1, 1)) / signz_safe
    drift_y = ed * jnp.reshape(dfun[:, 1], (1, 1, ns, 1, 1)) / signz_safe

    # linear drive coefficients (Term V)
    et = (vp2 + 2.0 * bn_b * mu) / jnp.maximum(temp_ratio, 1.0e-15) - 1.5
    dmaxwel = rln0 + rlt0 * et
    ekapka = efun_b * ky_b
    dmaxwel_fm_ek = dmaxwel * fmaxwl * ekapka

    # characteristic advection speeds
    upar = -ffun_b * vthrat0 * vp
    utrap = vthrat0 * mu * bn_b * gfun_b

    # speed-dependent dissipation magnitudes
    vpgr_rms = jnp.asarray(
        geometry.get("vpgr_rms", jnp.sqrt(jnp.mean(vpgr**2))), dtype=jnp.float64
    )
    mugr_rms = jnp.asarray(
        geometry.get("mugr_rms", jnp.sqrt(jnp.mean(mugr**2))), dtype=jnp.float64
    )
    idisp = jnp.asarray(params.idisp, dtype=jnp.int32)
    use_abs_vel = jnp.logical_or(jnp.equal(idisp, 1), jnp.equal(idisp, -1))

    abs_dum2_par = jnp.where(
        use_abs_vel,
        jnp.abs(upar),
        jnp.abs(ffun_b * vthrat0 * vpgr_rms),
    )
    abs_dum2_vp = jnp.where(
        use_abs_vel,
        jnp.abs(utrap),
        jnp.abs(vthrat0 * bn_b * gfun_b * mugr_rms),
    )

    # Term VII coupling factor
    term7_fac = -signz0 * ffun_b * vthrat0 * vp * fmaxwl / jnp.maximum(tmp0, 1.0e-15)

    # spectral perpendicular hyper-dissipation
    kxmax = jnp.asarray(geometry["kxmax"], dtype=jnp.float64)
    kymax = jnp.asarray(geometry["kymax"], dtype=jnp.float64)
    kxmax = jnp.where(jnp.abs(kxmax) < 1.0e-15, 1.0, kxmax)
    kymax = jnp.where(jnp.abs(kymax) < 1.0e-15, 1.0, kymax)

    dspx = jnp.abs(jnp.asarray(params.disp_x, dtype=jnp.float64))
    dspy = jnp.abs(jnp.asarray(params.disp_y, dtype=jnp.float64))
    kpowx = jnp.where(jnp.asarray(params.disp_x) < 0.0, 2.0, 4.0)
    kpowy = jnp.where(jnp.asarray(params.disp_y) < 0.0, 2.0, 4.0)
    hyper = -(dspy * (ky_b / kymax) ** kpowy + dspx * (kx_b / kxmax) ** kpowx)

    # parallel finite difference stencils
    pos_par = jnp.asarray(geometry["pos_par_grid_class"], dtype=jnp.int32)
    s_d1_ipos = _parallel_coefficients(pos_par, _D1_IPW_POS)
    s_d1_ineg = _parallel_coefficients(pos_par, _D1_IPW_NEG)
    s_d4_ipos = _parallel_coefficients(pos_par, _D4_IPW_POS)
    s_d4_ineg = _parallel_coefficients(pos_par, _D4_IPW_NEG)

    dvp = jnp.asarray(geometry.get("dvp", jnp.mean(jnp.diff(vpgr))), dtype=jnp.float64)
    dvp = jnp.where(jnp.abs(dvp) < 1.0e-15, 1.0, dvp)
    sgr_dist = jnp.asarray(geometry.get("sgr_dist", 1.0), dtype=jnp.float64)
    sgr_dist = jnp.where(jnp.abs(sgr_dist) < 1.0e-15, 1.0, sgr_dist)

    ixzero = jnp.asarray(
        geometry.get("ixzero", jnp.argmin(jnp.abs(jnp.asarray(kx, dtype=jnp.float64)))),
        dtype=jnp.int32,
    )
    iyzero = jnp.asarray(
        geometry.get("iyzero", jnp.argmin(jnp.abs(jnp.asarray(ky, dtype=jnp.float64)))),
        dtype=jnp.int32,
    )

    # nonlinear FFT grid metadata
    mphi, mphiw3 = _extended_firstdim_fft_size(nky)
    mrad = _extended_seconddim_fft_size(nkx)
    jind = _build_jind(nkx, mrad, ixzero)
    kx2d = jnp.broadcast_to(jnp.reshape(kx, (nkx, 1)), (nkx, nky))
    ky2d = jnp.broadcast_to(jnp.reshape(ky, (1, nky)), (nkx, nky))
    nl_dum_s = -jnp.asarray(efun, dtype=jnp.float64)

    return {
        "kx_b": kx_b,
        "ky_b": ky_b,
        "bessel": bessel,
        "fmaxwl": fmaxwl,
        "tmp0": jnp.asarray(tmp0, dtype=jnp.float64),
        "signz0": jnp.asarray(signz0, dtype=jnp.float64),
        "drift_x": drift_x,
        "drift_y": drift_y,
        "dmaxwel_fm_ek": dmaxwel_fm_ek,
        "upar": upar,
        "utrap": utrap,
        "abs_dum2_par": abs_dum2_par,
        "abs_dum2_vp": abs_dum2_vp,
        "term7_fac": term7_fac,
        "hyper": hyper,
        "s_d1_ipos": s_d1_ipos,
        "s_d1_ineg": s_d1_ineg,
        "s_d4_ipos": s_d4_ipos,
        "s_d4_ineg": s_d4_ineg,
        "dvp": dvp,
        "sgr_dist": sgr_dist,
        "ixzero": ixzero,
        "iyzero": iyzero,
        "nl_mphi": mphi,
        "nl_mphiw3": mphiw3,
        "nl_mrad": mrad,
        "nl_fft_scale": jnp.asarray(float(mrad * mphi), dtype=jnp.float64),
        "nl_jind": jind,
        "nl_kx2d": kx2d,
        "nl_ky2d": ky2d,
        "nl_dum_s": nl_dum_s,
    }


def _linear_rhs(
    df: Array,
    geometry: Dict[str, Array],
    params: GKParams,
    pre: Dict[str, Array],
    phi: Array | None = None,
) -> Array:
    """
    Assemble the linear Right-Hand Side (RHS) contribution for adiabatic electrons.
    
    This function implements the primary electrostatic gyrokinetic terms:
    - Term I: Parallel advection (streaming term).
    - Term II: Drift advection (curvature and grad-B drifts).
    - Term IV: Trapping effects (mirror force).
    - Term V: Equilibrium drive (density and temperature gradients).
    - Term VII: Parallel field drive.
    - Term VIII: Drift field drive.
    
    Includes parallel, velocity-space, and perpendicular hyper-dissipation branches.
    
    Args:
        df: Current distribution function.
        geometry: Geometry metadata.
        params: Solver parameters.
        pre: Precomputed coefficients and tensors.
        phi: Electrostatic potential [ns, nkx, nky]. If None, it is recalculated.
        
    Returns:
        Linear RHS contribution array.
    """
    if phi is None:
        phi, _ = get_integrals(df, geometry)
    phi_b = jnp.reshape(phi, (1, 1, phi.shape[0], phi.shape[1], phi.shape[2]))

    # Term I: Parallel advection with upwinded stencils
    ddf_ds_pos = (
        _apply_parallel_stencil(df, pre["s_d1_ipos"], geometry) / pre["sgr_dist"]
    )
    ddf_ds_neg = (
        _apply_parallel_stencil(df, pre["s_d1_ineg"], geometry) / pre["sgr_dist"]
    )
    term_i = pre["upar"] * jnp.where(pre["upar"] > 0.0, ddf_ds_pos, ddf_ds_neg)

    # parallel dissipation for stability
    d4f_ds_pos = (
        _apply_parallel_stencil(df, pre["s_d4_ipos"], geometry) / pre["sgr_dist"]
    )
    d4f_ds_neg = (
        _apply_parallel_stencil(df, pre["s_d4_ineg"], geometry) / pre["sgr_dist"]
    )
    term_par_diss = (
        jnp.asarray(params.disp_par, dtype=jnp.float64)
        * pre["abs_dum2_par"]
        * jnp.where(pre["upar"] > 0.0, d4f_ds_pos, d4f_ds_neg)
    )

    # Term IV: Mirror force and velocity dissipation
    ddf_dvp = _apply_vpar_stencil(df, _VPAR_D1) / pre["dvp"]
    d4f_dvp = _apply_vpar_stencil(df, _VPAR_D4) / pre["dvp"]
    term_iv = pre["utrap"] * ddf_dvp
    term_vp_diss = (
        jnp.asarray(params.disp_vp, dtype=jnp.float64) * pre["abs_dum2_vp"] * d4f_dvp
    )

    # Term II: Magnetic drift advection
    kdotvd = pre["drift_x"] * pre["kx_b"] + pre["drift_y"] * pre["ky_b"]
    term_ii = -1j * kdotvd * df

    # perpendicular damping
    term_hyper = pre["hyper"] * df

    # Term V: Electrostatic potential drive
    term_v = (
        1j
        * jnp.asarray(params.drive_scale, dtype=jnp.float64)
        * pre["dmaxwel_fm_ek"]
        * pre["bessel"]
        * phi_b
    )

    # Term VIII: Drift coupling with phi
    term_viii = (
        -1j
        * jnp.asarray(params.drive_scale, dtype=jnp.float64)
        * pre["signz0"]
        * kdotvd
        * (pre["fmaxwl"] / jnp.maximum(pre["tmp0"], 1.0e-15))
        * pre["bessel"]
        * phi_b
    )

    # Term VII: Parallel phi gradient drive
    gyro_phi = pre["bessel"] * phi_b
    dgyro_ds_pos = (
        _apply_parallel_stencil(gyro_phi, pre["s_d1_ipos"], geometry) / pre["sgr_dist"]
    )
    dgyro_ds_neg = (
        _apply_parallel_stencil(gyro_phi, pre["s_d1_ineg"], geometry) / pre["sgr_dist"]
    )
    term_vii = pre["term7_fac"] * jnp.where(
        pre["term7_fac"] < 0.0, dgyro_ds_pos, dgyro_ds_neg
    )

    return (
        term_i
        + term_par_diss
        + term_iv
        + term_vp_diss
        + term_ii
        + term_hyper
        + term_v
        + term_viii
        + term_vii
    )


def init_df_cosine2(
    geometry: Dict[str, Array],
    amp_init_real: float = 1.0e-4,
    amp_init_imag: float = 0.0,
    normalize_per_toroidal_mode: bool = True,
    norm_eps: float = 1.0e-14,
) -> Array:
    """
    Initialize the distribution function with a parallel cosine^2 profile.
    
    This mirrors the GKW 'cosine2' branch, creating a seeded perturbation 
    localized along the field line. In spectral mode-box runs, the zonal 
    (ky=0) mode is suppressed by default.
    
    Args:
        geometry: Geometry dictionary.
        amp_init_real: Real seed amplitude.
        amp_init_imag: Imaginary seed amplitude.
        normalize_per_toroidal_mode: Rescale so each non-zonal ky mode has unit phi amplitude.
        norm_eps: Normalization floor.
        
    Returns:
        Complex complex128 distribution function [vpar, mu, s, kx, ky].
    """
    nvpar = len(geometry["intvp"])
    nmu = len(geometry["intmu"])
    ns = len(geometry["ints"])
    nkx = len(geometry["kxrh"])
    nky = len(geometry["krho"])

    if "sgrid" in geometry:
        sgrid = jnp.asarray(geometry["sgrid"], dtype=jnp.float64)
    else:
        idx = jnp.arange(ns, dtype=jnp.float64)
        sgrid = (idx + 0.5) / ns - 0.5

    amp_ini = jnp.asarray(amp_init_real, dtype=jnp.float64) + 1j * jnp.asarray(
        amp_init_imag, dtype=jnp.float64
    )
    s_profile = amp_ini * (jnp.cos(2.0 * jnp.pi * sgrid) + 1.0)

    # broadcast seed across the full phase space
    df = jnp.broadcast_to(
        jnp.reshape(s_profile, (1, 1, ns, 1, 1)),
        (nvpar, nmu, ns, nkx, nky),
    ).astype(jnp.complex128)

    # suppress seed in the zonal flow mode to focus on drift-wave instability
    if nky > 1:
        if "iyzero" in geometry:
            iyzero = int(jnp.asarray(geometry["iyzero"]).item())
        else:
            iyzero = int(
                jnp.argmin(
                    jnp.abs(jnp.asarray(geometry["krho"], dtype=jnp.float64))
                ).item()
            )
        df = df.at[..., iyzero].set(0.0 + 0.0j)

    if normalize_per_toroidal_mode:
        norm_params = GKParams(norm_eps=norm_eps)
        df, _, _ = _normalize_per_ky(df, geometry, norm_params)

    return df


def _advance_state(
    state: GKState,
    params: GKParams,
    is_window_end: Array,
    dominant_amp: Array,
    norm_fac: Array,
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


def gksolve_with_state(
    prev_df: Array,
    geometry: Dict[str, Array],
    params: GKParams,
    state: GKState,
) -> Tuple[Array, Tuple[Array, Tuple[Array, Array, Array]], GKState]:
    """
    Perform a single small-step (dt) time integration using an explicit RK4 scheme.
    
    This is the primary stateful stepping function. It computes the total RHS 
    (Linear + Nonlinear Term III), integrates the distribution function, and 
    conditionally applies mode normalization at large-step boundaries.
    
    Args:
        prev_df: Initial complex distribution function at time t.
        geometry: Geometry dictionary.
        params: Solver hyperparameters and switches.
        state: Current diagnostic metadata state.
        
    Returns:
        Tuple of (next_df, (phi, fluxes), next_state).
    """
    dt = jnp.array(params.dt, dtype=jnp.float64)
    pre = _linear_precompute(geometry, params)

    def _rhs(df: Array) -> Array:
        # electrostatic Poisson solve
        phi_local, _ = get_integrals(df, geometry)
        rhs_linear = _linear_rhs(df, geometry, params, pre, phi=phi_local)

        def _with_nl(_: None) -> Array:
            # add nonlinear Term III advection
            rhs_nl = _nonlinear_term_iii(df, phi_local, geometry, pre)
            return rhs_linear + rhs_nl

        def _without_nl(_: None) -> Array:
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
        return _normalize_per_ky(next_df_raw, geometry, params)

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
    phi, fluxes = get_integrals(next_df, geometry)
    next_state = _advance_state(state, params, do_normalize, dominant_amp, norm_factor)
    return next_df, (phi, fluxes), next_state


def gksolve(
    prev_df: Array,
    geometry: Dict[str, Array],
    params: GKParams,
    state: GKState,
) -> Tuple[Array, Tuple[Array, Tuple[Array, Array, Array]]]:
    """
    Stateless core interface for a single small-step gyrokinetic integration.
    
    Args:
        prev_df: Current distribution function.
        geometry: Geometry metadata.
        params: Solver parameters.
        state: Integration state.
        
    Returns:
        Tuple of (next_df, (phi, fluxes)).
    """
    next_df, out, _ = gksolve_with_state(prev_df, geometry, params, state)
    return next_df, out


def kx0_mode_columns(mode_label: Array, kxrh: Array) -> Tuple[int, Array]:
    """
    Identify the wavevector columns corresponding to the kx=0 baseline.
    
    Used to map global spectral mode labels to the 1D growth/frequency 
    diagnostics that traditionally focus on the kx=0 slice.
    """
    mode_label = jnp.asarray(mode_label)
    kxrh = jnp.asarray(kxrh)
    kx_line = kxrh[0] if kxrh.ndim == 2 else kxrh
    ixzero = int(jnp.argmin(jnp.abs(kx_line)).item())
    # correct for fortran 1-based indexing in mode_label files
    cols = mode_label[ixzero].astype(jnp.int32) - 1
    return ixzero, cols


def project_all_modes_to_kx0(all_modes: Array, mode_label: Array, kxrh: Array) -> Array:
    """Project a flattened diagnostic array back to the kx=0 wavevector slice."""
    _, cols = kx0_mode_columns(mode_label, kxrh)
    return jnp.asarray(all_modes)[:, cols]
