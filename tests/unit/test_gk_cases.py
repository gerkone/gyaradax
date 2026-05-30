"""Verification tests against GKW standard reference cases and analytical benchmarks.

Includes:
- Rosenbluth-Hinton zonal flow residual (analytical)
"""

import os

import jax.numpy as jnp
import numpy as np
import pytest
from dataclasses import replace
from conftest import ALL_BACKENDS  # type: ignore[import-not-found]

from gyaradax.geometry import compute_geometry, compute_geometry_from_input
from gyaradax.params import gkparams_from_input_and_geometry
from gyaradax.params import GKParams
from gyaradax.precompute import linear_precompute
from gyaradax.solver import init_f, default_state
from gyaradax.simulate import gk_run
from gyaradax.integrals import calculate_phi


GKW_CASES_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "gkw_cases")
GKW_BENCHMARKS = os.path.join(os.path.dirname(__file__), "..", "..", "gkw_ref", "benchmarks")


def _rh_residual_xiao_catto(q, eps):
    """Analytical RH residual (Xiao-Catto, PoP 13 2006)."""
    theta = 1.6 * eps**1.5 + 0.5 * eps**2 + 0.36 * eps**2.5
    return 1.0 / (1.0 + q**2 * theta / eps**2)


@pytest.mark.parametrize("backend, use_z2z, mixed_precision", ALL_BACKENDS)
def test_rosenbluth_hinton_residual(backend, use_z2z, mixed_precision):
    """Rosenbluth-Hinton zonal flow test: residual converges to Xiao-Catto.

    Uses the GKW benchmark parameters (gkw_ref/benchmarks/zonal_flow/zonal01):
    q=1.3, shat=0.1592, eps=0.05, disp_par=0.01, kx≈0.025.
    GKW result: 0.078 (9.8% from analytical 0.0711).
    """
    zonal01 = os.path.join(GKW_BENCHMARKS, "zonal_flow", "zonal01", "input.dat")
    if not os.path.exists(zonal01):
        pytest.skip("GKW zonal_flow benchmark not available")

    geometry = compute_geometry_from_input(zonal01)
    params = gkparams_from_input_and_geometry(zonal01, geometry)
    # non_linear=False + large naverage → linear mode, no per-ky normalization
    params = replace(
        params,
        non_linear=False,
        naverage=100000,
        backend=backend,
        use_z2z=use_z2z,
        mixed_precision=mixed_precision,
    )

    pre = linear_precompute(geometry, params)
    df = init_f(geometry, finit="zonal", amp_init_real=params.amp_init)
    phi0 = calculate_phi(geometry, df, params=params, pre=pre)

    ints = jnp.asarray(geometry["ints"])
    ixzero = int(jnp.argmin(jnp.abs(jnp.asarray(geometry["kxrh"]))))
    ix_track = ixzero + 1

    def kxspec(phi):
        return float(jnp.sum(jnp.abs(phi[:, ix_track, 0]) ** 2 * ints))

    kx0 = kxspec(phi0)
    state = default_state(nky=len(geometry["krho"]))

    # Run 500 windows of 20 steps (t=100)
    trace = []
    times = []
    for _ in range(500):
        df, phi, _, state = gk_run(df, geometry, params, state, 20, pre=pre)
        trace.append(kxspec(phi) / kx0)
        times.append(float(state.time))

    trace = np.array(trace)
    times = np.array(times)

    # Late-time residual (t > 80)
    late = times > 80
    assert np.any(late), "simulation too short"
    residual = np.sqrt(np.mean(trace[late]))

    q_val = float(geometry["q"])
    eps_val = float(geometry["eps"])
    analytical = _rh_residual_xiao_catto(q_val, eps_val)

    assert abs(residual - analytical) / analytical < 0.15, (
        f"RH residual {residual:.4f} deviates from Xiao-Catto {analytical:.4f} "
        f"by {abs(residual - analytical) / analytical * 100:.1f}%"
    )


def test_adiabat_collisions_weak_1step_parity():
    """gyaradax vs GKW adiabat_collisions_weak at 1 step, no normalization.

    Tests the collision stencil + baseline ES operator against GKW's FDS.
    With normalization disabled in both codes the comparison is a clean
    stencil-for-stencil match.
    """
    from gyaradax.utils import load_gkw_dump
    from gyaradax.params import gkparams_from_input_and_geometry

    case_dir = os.path.join(GKW_CASES_DIR, "adiabat_collisions_weak_1step")
    input_dat = os.path.join(case_dir, "input.dat")
    if not os.path.exists(input_dat):
        pytest.skip("adiabat_collisions_weak_1step data not available")

    geom = compute_geometry(
        q=1.57,
        shat=1.07,
        eps=0.177,
        ns=50,
        nkx=1,
        nky=1,
        nvpar=16,
        nmu=4,
        vpar_max=3.0,
        nperiod=3,
        krhomax=0.5,
        geom_type="s-alpha",
    )
    params = gkparams_from_input_and_geometry(input_dat, geom)
    params = replace(params, non_linear=False, naverage=10**7)

    pre = linear_precompute(geom, params)
    df0 = init_f(geom, finit=params.finit, amp_init_real=params.amp_init)
    state = default_state(nky=1)

    df, _, _, _ = gk_run(df0, geom, params, state, 1, pre=pre)
    df_gkw, _ = load_gkw_dump(os.path.join(case_dir, "FDS"), (16, 4, 50, 1, 1), n_species=1)

    df_np = np.asarray(df)
    df_gkw_np = np.asarray(df_gkw)
    num = np.linalg.norm(df_np.ravel() - df_gkw_np.ravel())
    den = np.linalg.norm(df_gkw_np.ravel())
    rel_l2 = num / den
    assert rel_l2 < 1e-4, f"1-step rel L2 error {rel_l2:.4e} > 1e-4"


@pytest.mark.parametrize("backend, use_z2z, mixed_precision", ALL_BACKENDS)
def test_cbc_linear_itg_peak_growth(backend, use_z2z, mixed_precision):
    """CBC linear ITG at kt=0.5: growth rate matches GKW benchmark.

    Uses GKW benchmark parameters (gkw_ref/benchmarks/cyclone/linear):
    q=1.4, shat=0.78, eps=0.19, R/LT=6.9, R/Ln=2.2, s-alpha geometry,
    ns=144, nperiod=5, nvpar=64, nmu=16, disp_par=1.0.
    GKW reference (exact, identical params): gamma=0.1785 at kt=0.5.
    """
    geom = compute_geometry(
        q=1.4,
        shat=0.78,
        eps=0.19,
        ns=144,
        nvpar=64,
        nmu=16,
        vpar_max=3.0,
        nkx=1,
        nky=1,
        nperiod=5,
        kxmax=0.5,
        signB=1.0,
        Rref=1.0,
        krhomax=0.5,
        geom_type="s-alpha",
    )
    params = GKParams(
        dt=0.003,
        naverage=100,
        non_linear=False,
        adaptive_dt=False,
        adiabatic_electrons=True,
        disp_par=1.0,
        disp_vp=0.0,
        disp_x=0.0,
        disp_y=0.0,
        finit="cosine2",
        amp_init=1e-4,
        mas=1.0,
        signz=1.0,
        tmp=1.0,
        de=1.0,
        vthrat=1.0,
        rlt=6.9,
        rln=2.2,
        dgrid=1.0,
        tgrid=1.0,
        sgr_dist=float(geom["sgr_dist"]),
        dvp=float(geom["dvp"]),
        kxmax=0.5,
        kymax=0.5,
        norm_eps=1e-14,
        drive_scale=1.0,
        idisp=2,
        cfl_safety=0.95,
        mixed_precision=mixed_precision,
        backend=backend,
        use_z2z=use_z2z,
    )

    df = init_f(geom, finit="cosine2", amp_init_real=1e-4)
    pre = linear_precompute(geom, params)
    state = default_state(nky=1)

    for _ in range(300):
        df, phi, _, state = gk_run(df, geom, params, state, params.naverage, pre=pre)

    gamma = float(state.last_growth_rate[0])
    gamma_ref = 0.1785  # GKW (exact, identical parameters and grid)

    assert abs(gamma - gamma_ref) / gamma_ref < 0.05, (
        f"CBC growth rate {gamma:.4f} deviates from GKW {gamma_ref} "
        f"by {abs(gamma - gamma_ref) / gamma_ref * 100:.1f}%"
    )
