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

from gyaradax.jax_config import enable_x64

enable_x64()

import functools
from typing import Any, Dict, Tuple, Optional

from gyaradax.constants import EPS
from gyaradax.integrals import get_integrals, calculate_phi
from gyaradax.backends import create_ops
from gyaradax.params import GKParams
from gyaradax.state import GKPre, GKState, Precompute
from gyaradax.backends.ops import SolverOps
from gyaradax.cfl import (  # noqa: F401
    estimate_linear_timestep,
    estimate_nl_timestep,
    estimate_timestep,
)
from gyaradax.fields import _compute_fields, _compute_phi, f_to_g, g_to_f  # noqa: F401
from gyaradax.precompute import (  # noqa: F401
    _compute_species_coeffs,
    _fuse_stencils,
    _linear_precompute_core,
    _precompute_shared,
    build_jind,
    extended_firstdim_fft_size,
    extended_seconddim_fft_size,
    kx_ky_grids,
    linear_precompute,
    prime_factors_smallereq_than,
)
from gyaradax.utils import pack_half_spectrum, unpack_half_spectrum  # noqa: F401


def default_state(nky: int = 1) -> GKState:
    return GKState(
        time=jnp.array(0.0, dtype=jnp.float64),
        step=jnp.array(0, dtype=jnp.int32),
        accumulated_norm_factor=jnp.ones(nky, dtype=jnp.float64),
        window_start_amp=jnp.ones(nky, dtype=jnp.float64),
        last_growth_rate=jnp.zeros(nky, dtype=jnp.float64),
    )


def mode_amplitude(phi: jnp.ndarray, geometry: Dict[str, jnp.ndarray], eps: float) -> jnp.ndarray:
    """
    Per-ky mode amplitude over the connected kx chain containing kx=0.

    Matches GKW convention (diagnos_growth_freq.f90): only kx modes sharing
    the same mode_label as kx=0 contribute to the amplitude for each ky.
    """
    ds = jnp.asarray(geometry["ints"], dtype=jnp.float64)[0]
    mode_label = jnp.asarray(geometry["mode_label"], dtype=jnp.int32)
    ixzero = jnp.asarray(geometry["ixzero"], dtype=jnp.int32)
    chain_mask = mode_label == mode_label[ixzero, :]
    amp2 = ds * jnp.sum(jnp.abs(phi) ** 2 * chain_mask[None, :, :], axis=(0, 1))
    return jnp.sqrt(jnp.maximum(amp2, eps))


def normalize_per_ky(
    df: jnp.ndarray,
    geometry: Dict[str, jnp.ndarray],
    params: GKParams,
    pre: Optional[Precompute] = None,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    phi = calculate_phi(geometry, df, params=params, pre=pre)
    amp_per_ky = mode_amplitude(phi, geometry, params.norm_eps)
    # only normalize modes with meaningful amplitude
    active = amp_per_ky > jnp.sqrt(params.norm_eps)
    inv = jnp.where(active, 1.0 / amp_per_ky, 1.0)
    inv_shape = (1,) * (df.ndim - 1) + (-1,)
    return df * jnp.reshape(inv, inv_shape), inv, amp_per_ky


def nonlinear_term_iii(
    df: jnp.ndarray,
    phi: jnp.ndarray,
    geometry: Dict[str, jnp.ndarray],
    pre: GKPre,
    efun_sign: float = 1.0,
    fft_prefactor: complex = 1.0 + 0.0j,
    exclude_zero_mode: bool = True,
    mixed_precision: bool = True,
    ops: Optional[SolverOps] = None,
    backend: str = "jax",
    use_z2z: bool = False,
) -> jnp.ndarray:
    """Nonlinear ExB advection via pseudospectral method. df is 5D."""
    if ops is None:
        ops = create_ops(pre, backend=backend, use_z2z=use_z2z, mixed_precision=mixed_precision)

    return ops.nonlinear_term_iii(
        df,
        phi,
        geometry,
        efun_sign=efun_sign,
        fft_prefactor=fft_prefactor,
        exclude_zero_mode=exclude_zero_mode,
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
    *,
    params=None,
    out_sharding=None,
) -> jnp.ndarray:
    """Initialize the distribution function.

    Supported finit modes (matching GKW):
        cosine2 (default): amp * (cos(2*pi*s) + 1), flat in velocity space
        cosine:  amp * cos(2*pi*s), flat in velocity space
        cosine3: like cosine2 but weighted by exp(-E) in velocity space
        sine:    amp * de(is) * (sin(2*pi*s) + 1), density-weighted
        noise:   uniform random on [-1, 1] (real + imag)
        gnoise:  gaussian random (Box-Muller transform)
        zonal:   Rosenbluth-Hinton test — only ky=0, kx=±1 with Maxwellian weight

    Args:
        geometry: Geometry dictionary
        finit: Initialization mode
        amp_init_real: Initial amplitude (real part)
        amp_init_imag: Initial amplitude (imaginary part)
        normalize_per_toroidal_mode: Whether to normalize per toroidal mode
        norm_eps: Normalization epsilon
        n_species: Number of species
        seed: Random seed for noise modes
        params: Optional params object. If provided with n_gpus_* > 1, uses sharded init.
        out_sharding: Optional JAX sharding to apply to output. If None and params
            indicates multi-GPU, auto-detects and applies sharding.
    """
    # auto-detect multi-GPU sharding when params indicates it (don't override explicit)
    if out_sharding is None and params is not None:
        n_gpus_sp = int(getattr(params, "n_gpus_sp", 1))
        n_gpus_vp = int(getattr(params, "n_gpus_vp", 1))
        n_gpus_mu = int(getattr(params, "n_gpus_mu", 1))
        if n_gpus_sp * n_gpus_vp * n_gpus_mu > 1:
            import gyaradax.sharding as sharding
            from jax.sharding import NamedSharding, PartitionSpec

            mesh = sharding.build_mesh(params)
            if mesh is not None:
                if n_species > 1:
                    spec = PartitionSpec(
                        sharding._AXIS_SP, sharding._AXIS_VP, sharding._AXIS_MU, None, None, None
                    )
                else:
                    spec = PartitionSpec(sharding._AXIS_VP, sharding._AXIS_MU, None, None, None)
                out_sharding = NamedSharding(mesh, spec)

    nv, nmu, ns, nkx, nky = (
        len(geometry["intvp"]),
        len(geometry["intmu"]),
        len(geometry["ints"]),
        len(geometry["kxrh"]),
        len(geometry["krho"]),
    )
    sgrid = jnp.asarray(geometry.get("sgrid", jnp.linspace(-0.5, 0.5, ns)), dtype=jnp.float64)
    vpgr = jnp.asarray(geometry["vpgr"], dtype=jnp.float64)
    mugr = jnp.asarray(geometry["mugr"], dtype=jnp.float64)
    bn = jnp.asarray(geometry["bn"], dtype=jnp.float64)

    amp = jnp.asarray(amp_init_real, dtype=jnp.float64) + 1j * jnp.asarray(
        amp_init_imag, dtype=jnp.float64
    )

    shape_5d = (nv, nmu, ns, nkx, nky)
    shape_6d = (n_species, nv, nmu, ns, nkx, nky)
    full_shape = shape_6d if n_species > 1 else shape_5d

    # velocity-space Maxwellian (GKW components.f90, dens=dref=tref=1):
    # (n/n_grid) * exp(-(vpar^2 + 2*mu*B)/T) / (sqrt(T*pi))^3
    vp2 = vpgr**2
    tmp_val = jnp.asarray(geometry.get("tmp", jnp.ones(1)), dtype=jnp.float64)
    if tmp_val.ndim > 0:
        tmp_val = tmp_val[0]
    tgrid_val = jnp.asarray(geometry.get("tgrid", jnp.ones(1)), dtype=jnp.float64)
    if tgrid_val.ndim > 0:
        tgrid_val = tgrid_val[0]
    t_rat = tmp_val / tgrid_val
    energy = vp2[:, None, None] + 2.0 * mugr[None, :, None] * bn[None, None, :]
    maxwellian_env = jnp.exp(-energy / t_rat) / (jnp.sqrt(t_rat * jnp.pi) ** 3)

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
        prof_s = amp * (jnp.cos(2.0 * jnp.pi * sgrid) + 1.0)
        df = _broadcast_profile(prof_s, None, n_species, nv, nmu, ns, nkx, nky)

    elif finit == "cosine":
        prof_s = amp * jnp.cos(2.0 * jnp.pi * sgrid)
        df = _broadcast_profile(prof_s, None, n_species, nv, nmu, ns, nkx, nky)

    elif finit == "cosine3":
        prof_s = amp * (jnp.cos(2.0 * jnp.pi * sgrid) + 1.0)
        df = _broadcast_profile(prof_s, maxwellian_env, n_species, nv, nmu, ns, nkx, nky)

    elif finit == "sine":
        de = jnp.asarray(geometry.get("de", jnp.ones(max(n_species, 1))), dtype=jnp.float64)
        prof_s = amp * (jnp.sin(2.0 * jnp.pi * sgrid) + 1.0)
        if n_species > 1 and de.ndim >= 1 and de.shape[0] > 1:
            prof_2d = prof_s[None, :] * de[:, None]
            df = jnp.broadcast_to(prof_2d[:, None, None, :, None, None], shape_6d)
        else:
            de_val = float(de) if de.ndim == 0 else float(de[0])
            prof_s = prof_s * de_val
            df = _broadcast_profile(prof_s, None, n_species, nv, nmu, ns, nkx, nky)

    elif finit == "zonal":
        # Rosenbluth-Hinton: spectral kx = ±1 around kx=0 with ±i*amp*fmaxwl/2,
        # ions only (signz > 0). GKW init.f90:1471-1514.
        kxrh = jnp.asarray(geometry["kxrh"], dtype=jnp.float64)
        ixzero = int(jnp.argmin(jnp.abs(kxrh)).item())
        iy0 = int(
            jnp.asarray(
                geometry.get("iyzero", jnp.argmin(jnp.abs(jnp.asarray(geometry["krho"]))))
            ).item()
        )

        df = jnp.zeros(full_shape, dtype=jnp.complex128)

        if n_species > 1:
            signz = jnp.asarray(geometry.get("signz", jnp.ones(n_species)), dtype=jnp.float64)
            for isp in range(n_species):
                if float(signz[isp]) > 0:
                    if ixzero > 0:
                        df = df.at[isp, :, :, :, ixzero - 1, iy0].set(
                            -1j * amp * maxwellian_env / 2.0
                        )
                    if ixzero < nkx - 1:
                        df = df.at[isp, :, :, :, ixzero + 1, iy0].set(
                            1j * amp * maxwellian_env / 2.0
                        )
        else:
            if ixzero > 0:
                df = df.at[:, :, :, ixzero - 1, iy0].set(-1j * amp * maxwellian_env / 2.0)
            if ixzero < nkx - 1:
                df = df.at[:, :, :, ixzero + 1, iy0].set(1j * amp * maxwellian_env / 2.0)
        return df.astype(jnp.complex128)

    else:
        raise ValueError(f"unknown finit: {finit}")

    df = df.astype(jnp.complex128)

    # zero out the zonal mode (ky=0) — not for zonal init which IS the zonal mode
    if nky > 1:
        iy0 = int(
            jnp.asarray(
                geometry.get("iyzero", jnp.argmin(jnp.abs(jnp.asarray(geometry["krho"]))))
            ).item()
        )
        df = df.at[..., iy0].set(0.0)

    if normalize_per_toroidal_mode:
        df, _, _ = normalize_per_ky(df, geometry, GKParams(norm_eps=norm_eps))

    if out_sharding is not None:
        df = jax.device_put(df, out_sharding)
    return df


def _broadcast_profile(prof_s, vel_env, n_species, nv, nmu, ns, nkx, nky):
    """Broadcast a parallel profile (and optional velocity envelope) to full shape."""
    if vel_env is not None:
        base = vel_env * prof_s[None, None, :]
        if n_species > 1:
            return jnp.broadcast_to(
                base[None, :, :, :, None, None], (n_species, nv, nmu, ns, nkx, nky)
            )
        return jnp.broadcast_to(base[:, :, :, None, None], (nv, nmu, ns, nkx, nky))
    if n_species > 1:
        prof = jnp.broadcast_to(prof_s[None, :], (n_species, ns))
        return jnp.broadcast_to(
            prof[:, None, None, :, None, None], (n_species, nv, nmu, ns, nkx, nky)
        )
    return jnp.broadcast_to(prof_s[None, None, :, None, None], (nv, nmu, ns, nkx, nky))


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
    # post-normalization amplitude: linear → amp*(1/amp)=1, nonlinear → amp*1=amp
    new_window_start_amp = jnp.where(
        is_window_end, per_mode_amp * per_mode_norm_fac, state.window_start_amp
    )
    return GKState(
        time=new_time,
        step=new_step,
        accumulated_norm_factor=state.accumulated_norm_factor * per_mode_norm_fac,
        window_start_amp=new_window_start_amp,
        last_growth_rate=growth_rate,
    )


def gkstep_single(
    prev_df: jnp.ndarray,
    geometry: Dict[str, jnp.ndarray],
    params: GKParams,
    state: GKState,
    pre: GKPre,
    ops: Optional[SolverOps] = None,
    dt_override: Optional[jnp.ndarray] = None,
) -> Tuple[
    jnp.ndarray,
    Tuple[jnp.ndarray, Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]],
    GKState,
]:
    """Single small-step RK4 time integration with backend dispatch."""
    if ops is None:
        ops = create_ops(
            pre,
            backend=params.backend,
            use_z2z=params.use_z2z,
            mixed_precision=params.mixed_precision,
        )

    dt = dt_override if dt_override is not None else jnp.array(params.dt, dtype=jnp.float64)

    def _rhs(dg):
        phi_local, apar_local, bpar_local = _compute_fields(dg, geometry, params, pre)
        # linear terms act on f, not g (GKW exp_integration.F90:802-814 fdis_tmp = f)
        df_for_rhs = g_to_f(dg, apar_local, params, pre) if apar_local is not None else dg
        rhs = ops.linear_rhs(
            df_for_rhs, phi_local, geometry, params, pre, apar=apar_local, bpar=bpar_local
        )
        if params.non_linear:
            chi_corr = None
            if apar_local is not None and "apar_chi_factor" in pre:
                apar_b = apar_local[jnp.newaxis, jnp.newaxis, :, :, :]
                if dg.ndim == 6:
                    apar_b = apar_b[jnp.newaxis]
                chi_corr = pre["apar_chi_factor"] * apar_b
            if bpar_local is not None and "bpar_chi_factor" in pre:
                bpar_b = bpar_local[jnp.newaxis, jnp.newaxis, :, :, :]
                if dg.ndim == 6:
                    bpar_b = bpar_b[jnp.newaxis]
                bpar_chi = pre["bpar_chi_factor"] * bpar_b
                chi_corr = bpar_chi if chi_corr is None else chi_corr + bpar_chi
            rhs = rhs + ops.nonlinear_term_iii(dg, phi_local, geometry, chi_correction=chi_corr)
        return rhs, phi_local, apar_local

    # RK4 with inline CFL tracking across substages
    k1, phi1, apar1 = _rhs(prev_df)
    k2, phi2, apar2 = _rhs(prev_df + 0.5 * dt * k1)
    k3, phi3, apar3 = _rhs(prev_df + 0.5 * dt * k2)
    k4, phi4, apar4 = _rhs(prev_df + dt * k3)
    dt6 = dt / 6.0
    dt3 = dt / 3.0
    next_df_raw = prev_df + dt6 * k1 + dt3 * k2 + dt3 * k3 + dt6 * k4

    # inline NL CFL: max grad across all RK4 substages (GKW non_linear_terms.F90:1538)
    if params.non_linear:
        _ycorr = pre["nl_mrad"] * pre["nl_mrad"] * pre["nl_mphi"] * pre["nl_lxinv"]
        _xcorr = pre["nl_mrad"] * pre["nl_mphi"] * pre["nl_mphi"] * pre["nl_lyinv"]

        def _max_grad_inline(p):
            def _per_s(ps):
                gy = jnp.fft.irfft2(
                    pack_half_spectrum(
                        (1j * pre["nl_ky2d"] * ps)[None, None],
                        pre["nl_jind"],
                        pre["nl_mrad"],
                        pre["nl_mphiw3"],
                    ),
                    s=(pre["nl_mrad"], pre["nl_mphi"]),
                    axes=(-2, -1),
                    norm="backward",
                )
                gx = jnp.fft.irfft2(
                    pack_half_spectrum(
                        (1j * pre["nl_kx2d"] * ps)[None, None],
                        pre["nl_jind"],
                        pre["nl_mrad"],
                        pre["nl_mphiw3"],
                    ),
                    s=(pre["nl_mrad"], pre["nl_mphi"]),
                    axes=(-2, -1),
                    norm="backward",
                )
                return jnp.maximum(
                    jnp.max(jnp.abs(gy)) * _ycorr,
                    jnp.max(jnp.abs(gx)) * _xcorr,
                )

            return jnp.max(jax.vmap(_per_s)(p))

        mg_phi = jnp.maximum(
            jnp.maximum(_max_grad_inline(phi1), _max_grad_inline(phi2)),
            jnp.maximum(_max_grad_inline(phi3), _max_grad_inline(phi4)),
        )
        # em: 2*vthrat_max * max grad(apar) * vpmax (non_linear_terms.F90:1241,1790)
        mg_apar = jnp.array(0.0, dtype=jnp.float64)
        if params.nlapar:
            vpmax = pre["vpmax"]
            vthrat_max = pre.get("vthrat_max", jnp.asarray(1.0, dtype=jnp.float64))
            apar_fac = 2.0 * vthrat_max * vpmax
            _apar_grads = [
                _max_grad_inline(a) * apar_fac
                for a in [apar1, apar2, apar3, apar4]
                if a is not None
            ]
            if _apar_grads:
                mg_apar = jnp.maximum(mg_apar, jnp.stack(_apar_grads).max())
        mg_total = jnp.maximum(mg_phi, mg_apar)
        substage_dt_est = jnp.where(
            mg_total > EPS, 2.0 / mg_total, jnp.array(1e10, dtype=jnp.float64)
        )
    else:
        substage_dt_est = jnp.array(1e10, dtype=jnp.float64)

    new_step = state.step + jnp.array(1, dtype=jnp.int32)
    is_window_end = jnp.equal(jnp.mod(new_step, params.naverage), 0)

    if params.non_linear:
        phi, _, _ = _compute_fields(next_df_raw, geometry, params, pre)
        current_amp = mode_amplitude(phi, geometry, params.norm_eps)
        next_df = next_df_raw
        norm_factor = jnp.ones_like(state.accumulated_norm_factor)
    else:

        def _apply_norm(_):
            return normalize_per_ky(next_df_raw, geometry, params, pre=pre)

        def _skip_norm(_):
            phi_curr, _, _ = _compute_fields(next_df_raw, geometry, params, pre)
            amp_curr = mode_amplitude(phi_curr, geometry, params.norm_eps)
            return (next_df_raw, jnp.ones_like(state.accumulated_norm_factor), amp_curr)

        next_df, norm_factor, current_amp = jax.lax.cond(
            is_window_end, _apply_norm, _skip_norm, operand=None
        )
        phi, _, _ = _compute_fields(next_df, geometry, params, pre)

    z = jnp.array(0.0, dtype=jnp.float64)
    next_state = advance_state(state, params, is_window_end, current_amp, norm_factor, dt_used=dt)
    return next_df, (phi, (z, z, substage_dt_est)), next_state


@functools.partial(jax.jit, static_argnames=("n_steps", "return_dt_info"))
def gksolve(
    df: jnp.ndarray,
    geometry: Dict[str, jnp.ndarray],
    params: GKParams,
    state: GKState,
    n_steps: int = 1,
    pre: Optional[GKPre] = None,
    return_dt_info: bool = False,
) -> (
    Tuple[jnp.ndarray, Tuple[jnp.ndarray, Any], GKState]
    | Tuple[jnp.ndarray, Tuple[jnp.ndarray, Any], GKState, Dict[str, Any]]
):
    """Gyrokinetics solver forward.

    Executes multiple time steps via jax.lax.scan.
    When params.adaptive_dt is True, uses CFL-adaptive timestep with
    one-step lag (current step uses CFL estimate from previous step's phi).
    Returns (final_df, (final_phi, final_fluxes), final_state) by default.
    When ``return_dt_info`` is True, returns an extra trailing dict with
    per-step arrays ``dt_used``/``dt_nl``/``dt_lin`` (shape (n_steps,)) and
    scalar ``dt_input``. In the fixed-dt path ``dt_used`` is filled with
    ``params.dt`` and the CFL estimates are zero.
    """
    if pre is None:
        pre = linear_precompute(geometry, params)

    # ensure multi-species arrays are present for downstream flux calculations
    if not params.adiabatic_electrons:
        geometry = dict(geometry)
        for k in ("mas", "signz", "tmp", "de", "vthrat"):
            v = getattr(params, k, None)
            if v is not None:
                geometry[k] = jnp.atleast_1d(jnp.asarray(v, dtype=jnp.float64))

    ops = create_ops(
        pre, backend=params.backend, use_z2z=params.use_z2z, mixed_precision=params.mixed_precision
    )

    dt_input_scalar = jnp.array(params.dt, dtype=jnp.float64)

    if params.adaptive_dt and params.non_linear:
        # adaptive CFL path: carry dt as part of scan state
        dt_input = dt_input_scalar
        cfl_safety = jnp.array(params.cfl_safety, dtype=jnp.float64)

        def _scan_body(carry, _):
            curr_df, curr_state, curr_dt = carry
            next_df, out, next_state = gkstep_single(
                curr_df, geometry, params, curr_state, pre, ops, dt_override=curr_dt
            )
            # inline substage CFL from gkstep_single + linear CFL
            substage_dt = out[1][2]
            dt_lin = estimate_linear_timestep(pre, params=params)
            dt_nl = jnp.minimum(cfl_safety * substage_dt, dt_input)
            dt_cfl = jnp.minimum(dt_nl, dt_lin)
            # Ramp-up rule: dt grows at most by 5% per step
            ramp_up = jnp.minimum(curr_dt * 1.05, dt_input)
            next_dt = jnp.where(dt_cfl < curr_dt, dt_cfl, jnp.minimum(dt_cfl, ramp_up))
            dt_info_step = jnp.stack([curr_dt, dt_nl, dt_lin])
            return (next_df, next_state, next_dt), dt_info_step

        # init_dt must reflect the CURRENT NL amplitude, not just params.dt,
        # to avoid resetting dt at every block boundary when gksolve is called
        # in a block loop with growing NL fields (blow-up observed at β=0.01).
        phi_init, apar_init, _ = _compute_fields(df, geometry, params, pre)
        dt_nl_init = estimate_nl_timestep(
            phi_init, pre, pre["bessel"], dt_input, cfl_safety, apar=apar_init
        )
        dt_lin_init = estimate_linear_timestep(pre, params=params)
        init_dt = jnp.minimum(jnp.minimum(dt_input, dt_lin_init), dt_nl_init)
        (final_df, final_state, _), dt_stack = jax.lax.scan(
            _scan_body, (df, state, init_dt), None, length=n_steps
        )
        # dt_stack shape: (n_steps, 3) -> split into named arrays
        dt_info = {
            "dt_used": dt_stack[:, 0],
            "dt_nl": dt_stack[:, 1],
            "dt_lin": dt_stack[:, 2],
            "dt_input": dt_input_scalar,
        }
    else:
        # fixed dt path
        def _scan_body_fixed(carry, _):
            curr_df, curr_state = carry
            next_df, out, next_state = gkstep_single(
                curr_df, geometry, params, curr_state, pre, ops
            )
            return (next_df, next_state), None

        (final_df, final_state), _ = jax.lax.scan(
            _scan_body_fixed, (df, state), None, length=n_steps
        )
        dt_info = {
            "dt_used": jnp.full((n_steps,), dt_input_scalar, dtype=jnp.float64),
            "dt_nl": jnp.zeros((n_steps,), dtype=jnp.float64),
            "dt_lin": jnp.zeros((n_steps,), dtype=jnp.float64),
            "dt_input": dt_input_scalar,
        }

    # diagnostics use the physical distribution f, not the evolved mixed variable g
    # (GKW diagnos_fluxes_vspace.F90:444 applies get_f_from_g before fluxes/fields)
    diag_df = final_df
    if params.nlapar:
        _, apar_final, _ = _compute_fields(final_df, geometry, params, pre)
        diag_df = g_to_f(final_df, apar_final, params, pre)

    phi, fluxes = get_integrals(
        diag_df,
        geometry,
        params=params,
        pre=pre,
        adiabatic_electrons=params.adiabatic_electrons,
    )
    if return_dt_info:
        return final_df, (phi, fluxes), final_state, dt_info
    return final_df, (phi, fluxes), final_state
