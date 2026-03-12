import jax
import jax.numpy as jnp
import math
from dataclasses import dataclass
from typing import Dict, Tuple, Any

from jax_integrals import get_integrals, j0
from jax_geometry import load_runtime_params

# Ensure fp64 everywhere.
jax.config.update("jax_enable_x64", True)


Array = jnp.ndarray


def _center_5pt(stencil5):
    out = [0.0] * 9
    out[2:7] = stencil5
    return out


# Differential stencils from linear_terms.f90::differential_scheme, order='fourth_order'.
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
    """Runtime controls for the electrostatic solver."""

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
    Explicit diagnostic state used for large-step growth tracking.

    This state is intentionally separate from `gksolve` return values so the
    mandatory core interface remains:
      next_df, (phi, fluxes) = gksolve(prev_df, ...)
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
    """Construct a default diagnostic state."""
    return GKState(
        time=jnp.array(0.0, dtype=jnp.float64),
        step=jnp.array(0, dtype=jnp.int32),
        accumulated_norm_factor=jnp.array(1.0, dtype=jnp.float64),
        window_start_amp=jnp.array(1.0, dtype=jnp.float64),
        last_growth_rate=jnp.array(0.0, dtype=jnp.float64),
    )


def gkparams_from_runtime(runtime: Dict[str, Any], **overrides) -> GKParams:
    """Build GKParams from a runtime-controls dictionary with optional overrides."""
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
    """Load runtime controls from `input.dat` and convert them to GKParams."""
    runtime = load_runtime_params(input_dat_path)
    return gkparams_from_runtime(runtime, **overrides)


def _kx_ky_grids(geometry: Dict[str, Array]) -> Tuple[Array, Array]:
    kx = jnp.asarray(geometry["kxrh"], dtype=jnp.float64)
    ky = jnp.asarray(geometry["krho"], dtype=jnp.float64)
    if kx.ndim == 2:
        kx = kx[0]
    if ky.ndim == 2:
        ky = ky[:, 0]
    return kx, ky


def _mode_amplitude(phi: Array, geometry: Dict[str, Array], eps: float) -> Array:
    """Per-ky amplitude used by normalization (similar to mode-wise normalization)."""
    ints = jnp.asarray(geometry["ints"], dtype=jnp.float64)
    ds = ints[0]
    amp2 = ds * jnp.sum(jnp.abs(phi) ** 2, axis=(0, 1))
    return jnp.sqrt(jnp.maximum(amp2, eps))


def _normalize_per_ky(
    df: Array, geometry: Dict[str, Array], params: GKParams
) -> Tuple[Array, Array, Array]:
    phi, _ = get_integrals(df, geometry)
    amp_per_ky = _mode_amplitude(phi, geometry, params.norm_eps)
    safe_amp = jnp.where(amp_per_ky < params.norm_eps, 1.0, amp_per_ky)
    inv = 1.0 / safe_amp
    normalized_df = df * jnp.reshape(inv, (1, 1, 1, 1, inv.shape[0]))
    dominant_amp = jnp.max(safe_amp)
    return normalized_df, jnp.mean(inv), dominant_amp


def _parallel_coefficients(pos_par_class: Array, table: Array) -> Array:
    """Return stencil coefficients with shape [9, s, kx, ky]."""
    idx = jnp.asarray(pos_par_class, dtype=jnp.int32) + 2
    idx = jnp.clip(idx, 0, 4)
    coeff = table[idx] / 12.0
    return jnp.moveaxis(coeff, -1, 0)


def _shift_parallel(field: Array, geometry: Dict[str, Array], shift_idx: int) -> Array:
    """Shift field in s with open-boundary kx remapping from precomputed maps."""
    s_map = jnp.asarray(geometry["s_shift"], dtype=jnp.int32)[shift_idx]
    kx_map = jnp.asarray(geometry["kx_shift"], dtype=jnp.int32)[shift_idx]
    valid = jnp.asarray(geometry["valid_shift"], dtype=jnp.bool_)[shift_idx]

    nky = field.shape[-1]
    ky_idx = jnp.arange(nky, dtype=jnp.int32)
    ky_idx = jnp.reshape(ky_idx, (1, 1, nky))

    shifted = field[:, :, s_map, kx_map, ky_idx]
    return jnp.where(valid[None, None, :, :, :], shifted, 0.0)


def _apply_parallel_stencil(field: Array, coeffs: Array, geometry: Dict[str, Array]) -> Array:
    """Apply a 9-point-in-s stencil with open-boundary connectivity."""
    out = jnp.zeros_like(field)
    for shift_idx in range(9):
        shifted = _shift_parallel(field, geometry, shift_idx)
        out = out + coeffs[shift_idx][None, None, :, :, :] * shifted
    return out


def _apply_vpar_stencil(field: Array, coeffs: Array) -> Array:
    """Apply a centered 5-point stencil in vpar with zero-outside-grid boundaries."""
    nvpar = field.shape[0]
    base = jnp.arange(nvpar, dtype=jnp.int32)
    out = jnp.zeros_like(field)
    for c, shift in zip(coeffs, (-2, -1, 0, 1, 2)):
        idx = base + shift
        valid = jnp.logical_and(idx >= 0, idx < nvpar)
        idx_clip = jnp.clip(idx, 0, nvpar - 1)
        shifted = jnp.take(field, idx_clip, axis=0)
        out = out + c * jnp.where(valid[:, None, None, None, None], shifted, 0.0)
    return out


def _prime_factors_smallereq_than(number: int, max_prime: int) -> bool:
    """Mirror GKW helper used to choose FFT sizes with small prime factors."""
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
    Port of non_linear_terms::get_extended_firstdim_fft_size.

    Returns:
      mphi   : real-space binormal grid size.
      mphiw3 : reduced k-space size for real FFT storage (mphi/2 + 1).
    """
    posspace_size = 3 * nmod - 2
    if posspace_size % 2 != 0:
        posspace_size += 1
    while not _prime_factors_smallereq_than(posspace_size, 7):
        posspace_size += 2
    for i in range(1, 9):
        cand = posspace_size + 2 * i
        if _prime_factors_smallereq_than(cand, 2):
            posspace_size = cand
            break
    kgrid_size = int(math.floor(posspace_size / 2.0) + 1)
    return posspace_size, kgrid_size


def _extended_seconddim_fft_size(nx: int) -> int:
    """Port of non_linear_terms::get_extended_seconddim_fft_size."""
    dum = int(math.ceil(1.5 * float(nx + 1)) + 1)
    while not _prime_factors_smallereq_than(dum, 7):
        dum += 1
    for i in range(1, 9):
        cand = dum + i
        if _prime_factors_smallereq_than(cand, 2):
            dum = cand
            break
    return dum


def _build_jind(nkx: int, mrad: int, ixzero: int) -> Array:
    """
    Build Fortran-equivalent kx-to-FFT index mapping (0-based).

    Fortran logic:
      if ix>=ixzero: jind = ix-ixzero+1
      else         : jind = mrad + ix-ixzero+1
    """
    ix = jnp.arange(nkx, dtype=jnp.int32)
    return jnp.where(ix >= ixzero, ix - ixzero, mrad + ix - ixzero)


def _pack_half_spectrum(spec_kxky: Array, jind: Array, mrad: int, mphiw3: int) -> Array:
    """
    Pack physical spectral modes [kx, ky_nonneg] into dealiased FFT storage.
    """
    out_shape = spec_kxky.shape[:-2] + (mrad, mphiw3)
    out = jnp.zeros(out_shape, dtype=jnp.complex128)
    nky = spec_kxky.shape[-1]
    return out.at[..., jind, :nky].set(spec_kxky)


def _unpack_half_spectrum(spec_half: Array, jind: Array, nky: int) -> Array:
    """Unpack dealiased FFT storage back to physical [kx, ky_nonneg] modes."""
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
    ES spectral Term III for adiabatic-electron runs.

    Implemented from `non_linear_terms.F90::add_non_linear_terms_spectral` with
    electromagnetic/shear branches disabled for this scope.
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

    # Vectorize over parallel grid index s to control peak memory.
    df_by_s = jnp.moveaxis(df, 2, 0)
    bessel_by_s = jnp.moveaxis(bessel, 2, 0)

    def _per_s(df_s: Array, phi_s: Array, bessel_s: Array, dum: Array) -> Array:
        # Gyro-averaged potential gradients in k-space.
        gyro_phi = bessel_s * phi_s[None, None, :, :]
        grad_phi_y_k = 1j * ky2d[None, None, :, :] * gyro_phi
        grad_phi_x_k = 1j * kx2d[None, None, :, :] * gyro_phi

        # Distribution gradients in k-space.
        grad_f_x_k = 1j * kx2d[None, None, :, :] * df_s
        grad_f_y_k = 1j * ky2d[None, None, :, :] * df_s

        # Transform to dealiased real space.
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

        # Real-space Poisson bracket: V_E dot grad(f).
        nl_real = (efun_sign * dum) * (ar * cr - br * dr)

        # Back to spectral space. GKW's FFTW path uses unnormalized inverse and
        # then explicit `1/(mrad*mphi)` scaling after the forward transform.
        # With JAX default normalization, this is equivalent to multiplying by
        # `(mrad*mphi)` at this point.
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

    nl_by_s = jax.vmap(_per_s, in_axes=(0, 0, 0, 0))(df_by_s, phi, bessel_by_s, dum_s)
    nl = jnp.moveaxis(nl_by_s, 0, 2)
    # Match spectral copy-back behavior: suppress explicit (0,0) mode forcing.
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
    Public helper for isolated ES Term III diagnostics and ablation tests.
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
    Diagnostic helper: pack -> irfft2 -> rfft2 -> unpack on Term-III grids.
    """
    nkx = spec_kxky.shape[-2]
    nky = spec_kxky.shape[-1]
    kx = jnp.asarray(geometry["kxrh"], dtype=jnp.float64)
    if kx.ndim > 1:
        kx = kx[0]
    ixzero = int(
        jnp.asarray(
            geometry.get("ixzero", jnp.argmin(jnp.abs(kx)))
        ).item()
    )
    mphi, mphiw3 = _extended_firstdim_fft_size(nky)
    mrad = _extended_seconddim_fft_size(nkx)
    jind = _build_jind(nkx, mrad, ixzero)
    packed = _pack_half_spectrum(spec_kxky, jind, mrad, mphiw3)
    real = jnp.fft.irfft2(packed, s=(mrad, mphi), axes=(-2, -1), norm="backward")
    repacked = jnp.fft.rfft2(real, s=(mrad, mphi), axes=(-2, -1), norm="backward")
    return _unpack_half_spectrum(repacked, jind, nky)


def _linear_precompute(geometry: Dict[str, Array], params: GKParams) -> Dict[str, Array]:
    """Precompute geometry-only factors for active linear electrostatic terms."""
    kx, ky = _kx_ky_grids(geometry)
    ns = len(geometry["ints"])
    nkx = int(kx.shape[0])
    nky = int(ky.shape[0])

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

    mas = jnp.asarray(geometry["mas"], dtype=jnp.float64)
    tmp = jnp.asarray(geometry["tmp"], dtype=jnp.float64)
    de = jnp.asarray(geometry["de"], dtype=jnp.float64)
    signz = jnp.asarray(geometry["signz"], dtype=jnp.float64)
    vthrat = jnp.asarray(geometry["vthrat"], dtype=jnp.float64)
    rln = jnp.asarray(geometry["rln"], dtype=jnp.float64)
    rlt = jnp.asarray(geometry["rlt"], dtype=jnp.float64)

    mas0 = mas[0] if mas.ndim > 0 else mas
    tmp0 = tmp[0] if tmp.ndim > 0 else tmp
    de0 = de[0] if de.ndim > 0 else de
    signz0 = signz[0] if signz.ndim > 0 else signz
    vthrat0 = vthrat[0] if vthrat.ndim > 0 else vthrat
    rln0 = rln[0] if rln.ndim > 0 else rln
    rlt0 = rlt[0] if rlt.ndim > 0 else rlt

    dgrid0 = jnp.array(1.0, dtype=jnp.float64)
    if "dgrid" in geometry:
        dgrid = jnp.asarray(geometry["dgrid"], dtype=jnp.float64)
        dgrid0 = dgrid[0] if dgrid.ndim > 0 else dgrid

    tgrid0 = jnp.array(1.0, dtype=jnp.float64)
    if "tgrid" in geometry:
        tgrid = jnp.asarray(geometry["tgrid"], dtype=jnp.float64)
        tgrid0 = tgrid[0] if tgrid.ndim > 0 else tgrid

    vp2 = jnp.reshape(vpgr**2, (vpgr.shape[0], 1, 1, 1, 1))
    vp = jnp.reshape(vpgr, (vpgr.shape[0], 1, 1, 1, 1))
    mu = jnp.reshape(mugr, (1, mugr.shape[0], 1, 1, 1))
    bn_b = jnp.reshape(bn, (1, 1, bn.shape[0], 1, 1))
    ffun_b = jnp.reshape(ffun, (1, 1, ffun.shape[0], 1, 1))
    gfun_b = jnp.reshape(gfun, (1, 1, gfun.shape[0], 1, 1))
    efun_b = jnp.reshape(efun, (1, 1, efun.shape[0], 1, 1))

    kx_b = jnp.reshape(kx, (1, 1, 1, kx.shape[0], 1))
    ky_b = jnp.reshape(ky, (1, 1, 1, 1, ky.shape[0]))

    # Bessel factor used by gyro-averaged electrostatic terms.
    little_g = jnp.asarray(geometry["little_g"], dtype=jnp.float64)
    gzz = jnp.reshape(little_g[:, 0], (1, 1, ns, 1, 1))
    gez = jnp.reshape(little_g[:, 1], (1, 1, ns, 1, 1))
    gee = jnp.reshape(little_g[:, 2], (1, 1, ns, 1, 1))
    krloc_sq = ky_b**2 * gzz + 2.0 * ky_b * kx_b * gez + kx_b**2 * gee
    krloc_sq = jnp.where(krloc_sq < 0.0, 0.0, krloc_sq)
    krloc = jnp.sqrt(krloc_sq)

    signz_safe = jnp.where(jnp.abs(signz0) < 1.0e-15, 1.0, signz0)
    bessel_arg = mas0 * vthrat0 * krloc * jnp.sqrt(
        jnp.maximum(2.0 * mu / jnp.maximum(bn_b, 1.0e-15), 0.0)
    ) / signz_safe
    bessel = j0(bessel_arg)

    temp_ratio = tmp0 / jnp.maximum(tgrid0, 1.0e-15)
    fmaxwl = (
        de0
        / jnp.maximum(dgrid0, 1.0e-15)
        * jnp.exp(-(vp2 + 2.0 * bn_b * mu) / jnp.maximum(temp_ratio, 1.0e-15))
        / (jnp.sqrt(jnp.maximum(temp_ratio, 1.0e-15) * jnp.pi) ** 3)
    )

    # Drift components for term II and term VIII (coriolis/cf/rho* branches disabled).
    ed = vp2 + bn_b * mu
    drift_x = ed * jnp.reshape(dfun[:, 0], (1, 1, ns, 1, 1)) / signz_safe
    drift_y = ed * jnp.reshape(dfun[:, 1], (1, 1, ns, 1, 1)) / signz_safe

    # Term-V dmaxwel simplification for adiabatic-electron, linear ES setup.
    et = (vp2 + 2.0 * bn_b * mu) / jnp.maximum(temp_ratio, 1.0e-15) - 1.5
    dmaxwel = rln0 + rlt0 * et
    ekapka = efun_b * ky_b
    dmaxwel_fm_ek = dmaxwel * fmaxwl * ekapka

    # Term-I and term-IV advection coefficients.
    upar = -ffun_b * vthrat0 * vp
    utrap = vthrat0 * mu * bn_b * gfun_b

    # Dissipation speed magnitudes (idisp branch).
    vpgr_rms = jnp.asarray(geometry.get("vpgr_rms", jnp.sqrt(jnp.mean(vpgr**2))), dtype=jnp.float64)
    mugr_rms = jnp.asarray(geometry.get("mugr_rms", jnp.sqrt(jnp.mean(mugr**2))), dtype=jnp.float64)
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

    # Term-VII common factor.
    term7_fac = (
        -signz0
        * ffun_b
        * vthrat0
        * vp
        * fmaxwl
        / jnp.maximum(tmp0, 1.0e-15)
    )

    # Spectral perpendicular (hyper)dissipation.
    kxmax = jnp.asarray(geometry["kxmax"], dtype=jnp.float64)
    kymax = jnp.asarray(geometry["kymax"], dtype=jnp.float64)
    kxmax = jnp.where(jnp.abs(kxmax) < 1.0e-15, 1.0, kxmax)
    kymax = jnp.where(jnp.abs(kymax) < 1.0e-15, 1.0, kymax)

    dspx = jnp.abs(jnp.asarray(params.disp_x, dtype=jnp.float64))
    dspy = jnp.abs(jnp.asarray(params.disp_y, dtype=jnp.float64))
    kpowx = jnp.where(jnp.asarray(params.disp_x) < 0.0, 2.0, 4.0)
    kpowy = jnp.where(jnp.asarray(params.disp_y) < 0.0, 2.0, 4.0)
    hyper = -(dspy * (ky_b / kymax) ** kpowy + dspx * (kx_b / kxmax) ** kpowx)

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

    # Nonlinear spectral/dealias geometry (Term III).
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
    Active linear electrostatic spectral terms (adiabatic-electron setup):
    I, II, IV, V, VII, VIII + parallel/vpar/perp dissipation.
    """
    if phi is None:
        phi, _ = get_integrals(df, geometry)
    phi_b = jnp.reshape(phi, (1, 1, phi.shape[0], phi.shape[1], phi.shape[2]))

    # Term I: vpar_grd_df with open parallel boundaries and kx-chain connectivity.
    ddf_ds_pos = _apply_parallel_stencil(df, pre["s_d1_ipos"], geometry) / pre["sgr_dist"]
    ddf_ds_neg = _apply_parallel_stencil(df, pre["s_d1_ineg"], geometry) / pre["sgr_dist"]
    term_i = pre["upar"] * jnp.where(pre["upar"] > 0.0, ddf_ds_pos, ddf_ds_neg)

    # Parallel dissipation branch (idisp handling in precomputed abs_dum2_par).
    d4f_ds_pos = _apply_parallel_stencil(df, pre["s_d4_ipos"], geometry) / pre["sgr_dist"]
    d4f_ds_neg = _apply_parallel_stencil(df, pre["s_d4_ineg"], geometry) / pre["sgr_dist"]
    term_par_diss = (
        jnp.asarray(params.disp_par, dtype=jnp.float64)
        * pre["abs_dum2_par"]
        * jnp.where(pre["upar"] > 0.0, d4f_ds_pos, d4f_ds_neg)
    )

    # Term IV: dfdvp_trap + dfdvp_dissipation.
    ddf_dvp = _apply_vpar_stencil(df, _VPAR_D1) / pre["dvp"]
    d4f_dvp = _apply_vpar_stencil(df, _VPAR_D4) / pre["dvp"]
    term_iv = pre["utrap"] * ddf_dvp
    term_vp_diss = jnp.asarray(params.disp_vp, dtype=jnp.float64) * pre["abs_dum2_vp"] * d4f_dvp

    # Term II: vdgradf.
    kdotvd = pre["drift_x"] * pre["kx_b"] + pre["drift_y"] * pre["ky_b"]
    term_ii = -1j * kdotvd * df

    # Perpendicular hyper-dissipation.
    term_hyper = pre["hyper"] * df

    # Term V: ve_grad_fm electrostatic branch.
    term_v = (
        1j
        * jnp.asarray(params.drive_scale, dtype=jnp.float64)
        * pre["dmaxwel_fm_ek"]
        * pre["bessel"]
        * phi_b
    )

    # Term VIII: vd_grad_phi_fm electrostatic branch.
    term_viii = (
        -1j
        * jnp.asarray(params.drive_scale, dtype=jnp.float64)
        * pre["signz0"]
        * kdotvd
        * (pre["fmaxwl"] / jnp.maximum(pre["tmp0"], 1.0e-15))
        * pre["bessel"]
        * phi_b
    )

    # Term VII: vpar_grd_phi electrostatic branch.
    gyro_phi = pre["bessel"] * phi_b
    dgyro_ds_pos = _apply_parallel_stencil(gyro_phi, pre["s_d1_ipos"], geometry) / pre["sgr_dist"]
    dgyro_ds_neg = _apply_parallel_stencil(gyro_phi, pre["s_d1_ineg"], geometry) / pre["sgr_dist"]
    term_vii = pre["term7_fac"] * jnp.where(pre["term7_fac"] < 0.0, dgyro_ds_pos, dgyro_ds_neg)

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
    JAX port of `init_fdis` branch for `finit='cosine2'`.

    For spectral mode-box runs with nmod>1, ky=0 is initialized to zero.
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

    df = jnp.broadcast_to(
        jnp.reshape(s_profile, (1, 1, ns, 1, 1)),
        (nvpar, nmu, ns, nkx, nky),
    ).astype(jnp.complex128)

    if nky > 1:
        if "iyzero" in geometry:
            iyzero = int(jnp.asarray(geometry["iyzero"]).item())
        else:
            iyzero = int(
                jnp.argmin(jnp.abs(jnp.asarray(geometry["krho"], dtype=jnp.float64))).item()
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
    new_step = state.step + jnp.array(1, dtype=jnp.int32)
    new_time = state.time + jnp.array(params.dt, dtype=jnp.float64)

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
    # After large-step normalization, each new window starts with unit mode amplitude.
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
    One small-step (dt) electrostatic update with explicit RK4.
    """
    dt = jnp.array(params.dt, dtype=jnp.float64)
    pre = _linear_precompute(geometry, params)

    def _rhs(df: Array) -> Array:
        phi_local, _ = get_integrals(df, geometry)
        rhs_linear = _linear_rhs(df, geometry, params, pre, phi=phi_local)

        def _with_nl(_: None) -> Array:
            rhs_nl = _nonlinear_term_iii(df, phi_local, geometry, pre)
            return rhs_linear + rhs_nl

        def _without_nl(_: None) -> Array:
            return rhs_linear

        term_iii_on = jnp.logical_and(
            jnp.asarray(params.non_linear, dtype=jnp.bool_),
            jnp.asarray(params.enable_term_iii, dtype=jnp.bool_),
        )
        return jax.lax.cond(term_iii_on, _with_nl, _without_nl, operand=None)

    k1 = _rhs(prev_df)
    k2 = _rhs(prev_df + 0.5 * dt * k1)
    k3 = _rhs(prev_df + 0.5 * dt * k2)
    k4 = _rhs(prev_df + dt * k3)

    next_df_raw = prev_df + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

    new_step = state.step + jnp.array(1, dtype=jnp.int32)
    is_window_end = jnp.equal(jnp.mod(new_step, params.naverage), 0)
    # In GKW nonlinear runs normalization is disabled by default.
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

    next_df, norm_factor, dominant_amp = jax.lax.cond(
        do_normalize,
        _apply_norm,
        _skip_norm,
        operand=None,
    )

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
    Required core interface:
      next_df, (phi, fluxes) = gksolve(prev_df, ...)
    """
    next_df, out, _ = gksolve_with_state(prev_df, geometry, params, state)
    return next_df, out


def kx0_mode_columns(mode_label: Array, kxrh: Array) -> Tuple[int, Array]:
    """
    Return the kx=0 row index and corresponding flattened-mode columns.
    """
    mode_label = jnp.asarray(mode_label)
    kxrh = jnp.asarray(kxrh)
    kx_line = kxrh[0] if kxrh.ndim == 2 else kxrh
    ixzero = int(jnp.argmin(jnp.abs(kx_line)).item())
    cols = mode_label[ixzero].astype(jnp.int32) - 1
    return ixzero, cols


def project_all_modes_to_kx0(all_modes: Array, mode_label: Array, kxrh: Array) -> Array:
    """
    Select the all-mode diagnostic columns corresponding to kx=0, all ky.
    """
    _, cols = kx0_mode_columns(mode_label, kxrh)
    return jnp.asarray(all_modes)[:, cols]
