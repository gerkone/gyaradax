"""Tests for the eigenvalue formulation of the linear solve (issue #5).

Validates that gyaradax/eigenvalue.py finds eigenvalues of the EXACT same
operator the IVP (gksolve) integrates, for ES (adiabatic + kinetic), EM
(apar, apar+bpar) and collisional cases.  JAX backend only; GPU required
(same as the rest of the suite).

Comparison protocol
-------------------
The linear operator block-diagonalizes over ky, so all eigensolves here use
``ky_select`` to target the finite-ky block, and the IVP baseline is the
per-ky ``state.last_growth_rate`` from gksolve run until the window growth
rate converges.  The IVP complex eigenvalue (gamma + i*omega) is measured
independently from the phase/amplitude ratio of phi across one
normalization-free gkstep_single.
"""

import os
from dataclasses import replace

import jax.numpy as jnp
import numpy as np
import pytest

from gyaradax.backends import create_ops
from gyaradax.cfl import estimate_linear_timestep
from gyaradax.eigenvalue import (
    _build_rhs_matvec,
    eigensolve_linear,
)
from gyaradax.fields import _compute_fields
from gyaradax.geometry import compute_geometry, compute_geometry_from_input
from gyaradax.params import GKParams, gkparams_from_input_and_geometry
from gyaradax.precompute import linear_precompute
from gyaradax.solver import default_state, gkstep_single, gksolve, init_f

# ── case setups ──────────────────────────────────────────────────────────────


def _es_adiabatic_case(**param_overrides):
    """Small ES adiabatic ITG (eiv_simple-like physics, tiny grid)."""
    geom = compute_geometry(
        q=1.4,
        shat=0.78,
        eps=0.19,
        ns=16,
        nkx=1,
        nky=2,
        nvpar=16,
        nmu=8,
        nperiod=1,
        krhomax=0.5,
    )
    params = GKParams(
        dt=0.01,
        naverage=100,
        disp_par=1.0,
        disp_vp=0.2,
        dvp=float(geom["dvp"]),
        sgr_dist=float(geom["sgr_dist"]),
        kxmax=float(geom["kxmax"]),
        kymax=float(geom["kymax"]),
        rlt=6.9,
        rln=2.2,
        mas=1.0,
        tmp=1.0,
        de=1.0,
        signz=1.0,
        vthrat=1.0,
        shat=0.78,
        q=1.4,
        eps=0.19,
        kthnorm=float(np.asarray(geom["kthnorm"]).reshape(-1)[0]),
        adiabatic_electrons=True,
    )
    params = replace(params, **param_overrides)
    return geom, params, 1


_KINETIC_INPUT_TEMPLATE = """\
&CONTROL
 silent = .true.
 order_of_the_scheme = 'fourth_order'
 parallel_boundary_conditions = "open"
 NON_LINEAR = .false.
 zonal_adiabatic = .false.
 METHOD = 'EXP'
 METH   = 2
 DTIM   = {dtim}
 NTIME = 100
 NAVERAGE = {naverage}
 nlapar = {nlapar}
 nlbpar = {nlbpar}
 collisions = .false.
 disp_par = 1.0
 disp_vp  = 0.2
 disp_x   = 0.1
 disp_y   = 0.1
 io_format = 'ascii'
/
&GRIDSIZE
 NX = {nx}
 N_s_grid = 16
 N_mu_grid = 8
 nperiod = 1
 N_vpar_grid = 16
 NMOD = 2
 number_of_species = 2
/
&MODE
 mode_box = .true.
 krhomax = 0.4
 ikxspace = 1
/
&GEOM
 SHAT = 1.0
 Q    = 2.
 EPS  = 0.166
 GEOM_TYPE = 's-alpha'
 SIGNB = 1
 SIGNJ = 1
/
&SPCGENERAL
 beta = {beta}
 adiabatic_electrons = .false.
 amp_init = 1e-3
 finit = 'cosine2'
/
&SPECIES
 MASS  = 1.
 Z     = 1.
 TEMP  = 1.
 dens  = 1.
 rlt   = 9.
 rln   = 3.
 uprim = 0.0
/
&SPECIES
 MASS  = {mass_e}
 Z     = -1.0
 TEMP  = 1.0
 dens  = 1.0
 rlt   = 9.
 rln   = 3.
 uprim = 0.0
/
&ROTATION
 VCOR = 0.0
/
"""


def _kinetic_case(
    tmp_path, *, beta, nlapar, nlbpar, nx=1, mass_e=2.72e-4, dtim=0.005, naverage=200
):
    """Small kinetic 2-species case (Waltz-minrepro-like at reduced grid).

    dt is clipped to the von Neumann linear estimate so RK4 is stable for
    both the IVP run and the 'exp' eigensolve matvec.
    """
    input_path = os.path.join(tmp_path, "input.dat")
    with open(input_path, "w") as f:
        f.write(
            _KINETIC_INPUT_TEMPLATE.format(
                beta=beta,
                nlapar=".true." if nlapar else ".false.",
                nlbpar=".true." if nlbpar else ".false.",
                nx=nx,
                mass_e=mass_e,
                dtim=dtim,
                naverage=naverage,
            )
        )
    geom = compute_geometry_from_input(input_path)
    params = gkparams_from_input_and_geometry(input_path, geom)
    pre = linear_precompute(geom, params)
    dt_lin = float(estimate_linear_timestep(pre, params=params))
    if dt_lin < params.dt:
        params = replace(params, dt=dt_lin)
    nsp = int(np.atleast_1d(np.asarray(params.mas)).shape[0])
    return geom, params, nsp


# ── IVP baseline ─────────────────────────────────────────────────────────────


def _ivp_reference(geom, params, pre, n_species, max_blocks=400, tol=1e-5):
    """gksolve until the per-ky growth rate converges.

    Returns (gamma_per_ky, lambda_per_ky, n_steps_run).  lambda_per_ky is the
    complex per-ky eigenvalue measured from the phi ratio across one
    normalization-free gkstep_single — an eigensolver-independent estimate of
    gamma + i*omega.
    """
    df = init_f(geom, finit=params.finit, amp_init_real=params.amp_init, n_species=n_species)
    nky = len(geom["krho"])
    state = default_state(nky=nky)
    prev = None
    for _ in range(max_blocks):
        df, _, state = gksolve(df, geom, params, state, n_steps=params.naverage, pre=pre)
        g = np.asarray(state.last_growth_rate)
        if prev is not None and np.max(np.abs(g - prev)) < tol:
            break
        prev = g
    gamma = np.asarray(state.last_growth_rate)

    p_nonorm = replace(params, disable_per_ky_norm=True)
    phi0, _, _ = _compute_fields(df, geom, params, pre)
    df1, (phi1, _), _ = gkstep_single(df, geom, p_nonorm, state, pre)
    lam = np.zeros(nky, dtype=complex)
    for iky in range(nky):
        a0 = np.asarray(phi0[..., iky]).ravel()
        a1 = np.asarray(phi1[..., iky]).ravel()
        i = int(np.argmax(np.abs(a0)))
        # a dead ky (zonal) carries no phase to measure
        lam[iky] = (np.log(a1[i] / a0[i]) / float(params.dt)
                    if a0[i] != 0 else np.nan)
    return gamma, lam, int(state.step)


def _assert_eig_matches_ivp(eigval, gamma_ivp, lam_ivp, rtol=5e-3):
    """Dominant eigenvalue vs IVP growth rate + frequency."""
    assert np.isfinite(lam_ivp), f"IVP gave no eigenvalue for this ky: {lam_ivp}"
    assert np.isfinite(gamma_ivp) and gamma_ivp != 0, f"IVP gave no growth rate: {gamma_ivp}"
    scale = max(abs(lam_ivp), abs(gamma_ivp))
    rel_gamma = abs(eigval.real - gamma_ivp) / abs(gamma_ivp)
    rel_freq = abs(eigval.imag - lam_ivp.imag) / scale
    assert (
        rel_gamma < rtol
    ), f"growth mismatch: eig={eigval}, gamma_ivp={gamma_ivp}, rel={rel_gamma:.2e}"
    assert (
        rel_freq < rtol
    ), f"frequency mismatch: eig={eigval}, lam_ivp={lam_ivp}, rel={rel_freq:.2e}"
    # internal consistency of the two IVP measurements
    assert (
        abs(lam_ivp.real - gamma_ivp) / abs(gamma_ivp) < rtol
    ), f"IVP self-inconsistency: lam_ivp={lam_ivp}, gamma_ivp={gamma_ivp}"


def _run_case(geom, params, n_species, *, iky=1, k=4, n_steps_per_matvec=None, ivp_tol=1e-5):
    """Shared driver: IVP to convergence + per-ky scipy-ARPACK eigensolve."""
    pre = linear_precompute(geom, params)
    gamma, lam, n_steps = _ivp_reference(geom, params, pre, n_species, tol=ivp_tol)
    if n_steps_per_matvec is None:
        # ~0.25 time units of linear evolution per Arnoldi vector
        n_steps_per_matvec = max(1, int(round(0.25 / float(params.dt))))
    eigvals, eigvecs = eigensolve_linear(
        geom,
        params,
        pre=pre,
        k=k,
        mode="exp",
        tol=1e-10,
        n_steps_per_matvec=n_steps_per_matvec,
        ky_select=iky,
    )
    return pre, gamma, lam, eigvals, eigvecs


# ── (i) ES adiabatic ITG ─────────────────────────────────────────────────────


class TestESAdiabatic:
    def test_dominant_matches_ivp(self):
        geom, params, nsp = _es_adiabatic_case()
        pre, gamma, lam, eigvals, _ = _run_case(geom, params, nsp, iky=1)
        _assert_eig_matches_ivp(eigvals[0], gamma[1], lam[1])



# ── (ii) ES kinetic electrons ────────────────────────────────────────────────


class TestESKinetic:
    def test_dominant_matches_ivp(self, tmp_path):
        geom, params, nsp = _kinetic_case(str(tmp_path), beta=0.0, nlapar=False, nlbpar=False)
        pre, gamma, lam, eigvals, _ = _run_case(geom, params, nsp, iky=1)
        _assert_eig_matches_ivp(eigvals[0], gamma[1], lam[1])


# ── (iii) EM A_parallel (Waltz minrepro-like, beta=0.005) ───────────────────


class TestEMApar:
    def test_dominant_matches_ivp(self, tmp_path):
        geom, params, nsp = _kinetic_case(str(tmp_path), beta=0.005, nlapar=True, nlbpar=False)
        assert params.nlapar and not params.nlbpar and params.beta == 0.005
        pre, gamma, lam, eigvals, _ = _run_case(geom, params, nsp, iky=1)
        _assert_eig_matches_ivp(eigvals[0], gamma[1], lam[1])


# ── (iv) EM A_parallel + B_parallel ─────────────────────────────────────────


class TestEMAparBpar:
    def test_dominant_matches_ivp(self, tmp_path):
        geom, params, nsp = _kinetic_case(str(tmp_path), beta=0.005, nlapar=True, nlbpar=True)
        assert params.nlapar and params.nlbpar
        pre, gamma, lam, eigvals, _ = _run_case(geom, params, nsp, iky=1)
        _assert_eig_matches_ivp(eigvals[0], gamma[1], lam[1])


# ── (v) collisions (adiabatic + 1 ion MVP) ──────────────────────────────────


class TestCollisions:
    def test_dominant_matches_ivp(self):
        geom, params, nsp = _es_adiabatic_case(collisions=True, coll_freq=0.1)
        assert params.collisions and params.coll_freq > 0
        pre, gamma, lam, eigvals, _ = _run_case(geom, params, nsp, iky=1)
        _assert_eig_matches_ivp(eigvals[0], gamma[1], lam[1])

    def test_collisions_shift_the_eigenvalue(self):
        """The collisional operator is actually in the matvec (gamma changes)."""
        geom, params, nsp = _es_adiabatic_case()
        pre = linear_precompute(geom, params)
        ev0, _ = eigensolve_linear(
            geom,
            params,
            pre=pre,
            k=1,
            mode="exp",
            tol=1e-9,
            n_steps_per_matvec=25,
            ky_select=1,
        )
        geom_c, params_c, _ = _es_adiabatic_case(collisions=True, coll_freq=0.5)
        pre_c = linear_precompute(geom_c, params_c)
        ev1, _ = eigensolve_linear(
            geom_c,
            params_c,
            pre=pre_c,
            k=1,
            mode="exp",
            tol=1e-9,
            n_steps_per_matvec=25,
            ky_select=1,
        )
        assert abs(ev1[0] - ev0[0]) > 1e-3, f"no collisional shift: {ev0[0]} vs {ev1[0]}"


# ── (vi) subdominant-mode residuals (ARPACK-independent check) ──────────────


class TestSubdominantResiduals:
    @pytest.mark.parametrize("case", ["es_adiabatic", "em_apar"])
    def test_residuals(self, case, tmp_path):
        if case == "es_adiabatic":
            geom, params, nsp = _es_adiabatic_case()
        else:
            geom, params, nsp = _kinetic_case(str(tmp_path), beta=0.005, nlapar=True, nlbpar=False)
        pre = linear_precompute(geom, params)
        n_steps = max(1, int(round(0.25 / float(params.dt))))
        eigvals, eigvecs = eigensolve_linear(
            geom,
            params,
            pre=pre,
            k=4,
            mode="exp",
            tol=1e-12,
            n_steps_per_matvec=n_steps,
            ky_select=1,
        )
        ops = create_ops(pre, backend="jax", use_z2z=False, mixed_precision=False)
        mv = _build_rhs_matvec(geom, params, pre, ops)
        for i in range(4):
            v = jnp.asarray(eigvecs[i])
            lv = mv(v)
            res = float(jnp.linalg.norm(lv - eigvals[i] * v) / jnp.linalg.norm(eigvals[i] * v))
            assert res < 1e-4, f"pair {i}: lambda={eigvals[i]}, residual={res:.3e}"
        # eigenvalues are distinct and sorted by descending growth rate
        assert np.all(np.diff(eigvals.real) <= 1e-12)


# ── (vii) 'rhs' vs 'exp' mode agreement ──────────────────────────────────────


class TestModeAgreement:
    def test_rhs_vs_exp_dominant(self):
        geom, params, nsp = _es_adiabatic_case()
        pre = linear_precompute(geom, params)
        ev_exp, _ = eigensolve_linear(
            geom,
            params,
            pre=pre,
            k=2,
            mode="exp",
            tol=1e-10,
            n_steps_per_matvec=25,
            ky_select=1,
        )
        ev_rhs, _ = eigensolve_linear(
            geom,
            params,
            pre=pre,
            k=2,
            mode="rhs",
            tol=1e-8,
            ncv=80,
            ky_select=1,
        )
        rel = abs(ev_rhs[0] - ev_exp[0]) / abs(ev_exp[0])
        assert rel < 1e-5, f"rhs={ev_rhs[0]}, exp={ev_exp[0]}, rel={rel:.2e}"

    def test_exp_refinement_fixes_log_branch(self):
        """With dt_eff large enough that Im(lambda)*dt_eff > pi, the raw
        log(mu)/dt frequency wraps; Rayleigh refinement must not."""
        geom, params, nsp = _es_adiabatic_case()
        pre = linear_precompute(geom, params)
        # dominant ITG omega ~ 0.6; need |omega|*dt_eff > pi, so dt_eff = 8.0
        n_steps = int(round(8.0 / float(params.dt)))
        ev_ref, _ = eigensolve_linear(
            geom,
            params,
            pre=pre,
            k=1,
            mode="exp",
            tol=1e-10,
            n_steps_per_matvec=25,
            ky_select=1,
        )
        ev_raw, _ = eigensolve_linear(
            geom,
            params,
            pre=pre,
            k=1,
            mode="exp",
            tol=1e-10,
            n_steps_per_matvec=n_steps,
            ky_select=1,
            refine=False,
        )
        ev_fix, _ = eigensolve_linear(
            geom,
            params,
            pre=pre,
            k=1,
            mode="exp",
            tol=1e-10,
            n_steps_per_matvec=n_steps,
            ky_select=1,
            refine=True,
        )
        dt_eff = n_steps * float(params.dt)
        assert abs(ev_ref[0].imag) * dt_eff > np.pi, "case no longer exercises the branch"
        assert abs(ev_fix[0] - ev_ref[0]) / abs(ev_ref[0]) < 1e-8
        # the raw log value differs by a multiple of 2*pi/dt_eff in Im
        wrap = (ev_raw[0].imag - ev_ref[0].imag) * dt_eff / (2 * np.pi)
        assert abs(wrap - round(wrap)) < 1e-6 and round(wrap) != 0


# ── misc API behavior ────────────────────────────────────────────────────────


class TestAPI:
    def test_nonlinear_params_rejected(self):
        geom, params, _ = _es_adiabatic_case(non_linear=True)
        with pytest.raises(ValueError, match="non_linear"):
            eigensolve_linear(geom, params, k=1)

    def test_ky_select_out_of_range(self):
        geom, params, _ = _es_adiabatic_case()
        with pytest.raises(ValueError, match="ky_select"):
            eigensolve_linear(geom, params, k=1, ky_select=7)

    def test_kinetic_n_species_autoderived(self, tmp_path):
        """Eigenvector shape carries the species axis without passing n_species."""
        geom, params, nsp = _kinetic_case(str(tmp_path), beta=0.0, nlapar=False, nlbpar=False)
        pre = linear_precompute(geom, params)
        ev, evec = eigensolve_linear(
            geom,
            params,
            pre=pre,
            k=1,
            ncv=10,
            mode="exp",
            n_steps_per_matvec=10,
            ky_select=1,
        )
        assert evec.shape[1] == nsp
        assert np.all(np.isfinite(ev))
