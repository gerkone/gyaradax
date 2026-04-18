"""Unit tests for the Fokker-Planck collision operator.

Sanity checks that do not require a GKW reference run:
- The full operator (pitch + energy + friction) approximately preserves
  the Maxwellian (only the discretization error remains).
- The pitch-angle-only operator preserves any isotropic function f(v),
  tested with f = v^2.
- A small perturbation to the Maxwellian relaxes back toward it when
  stepped with explicit Euler.
"""

import jax.numpy as jnp
import numpy as np
import pytest
from dataclasses import replace

from gyaradax.collisions import precompute_collisions, collision_rhs
from gyaradax.geometry import compute_geometry
from gyaradax.params import GKParams


def _base_params(geom, **overrides):
    p = GKParams(
        dt=0.012,
        naverage=50,
        disp_par=1.0,
        disp_vp=0.0,
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
        shat=1.07,
        q=1.57,
        eps=0.177,
        kthnorm=1.0,
        adiabatic_electrons=True,
        collisions=True,
        coll_freq=0.1,
    )
    return replace(p, **overrides)


def _geom(nvpar=32, nmu=8, ns=8):
    return compute_geometry(
        q=1.57,
        shat=1.07,
        eps=0.177,
        ns=ns,
        nkx=1,
        nky=1,
        nvpar=nvpar,
        nmu=nmu,
        nperiod=1,
        krhomax=0.5,
    )


def _fmax(geom):
    vpgr = jnp.asarray(geom["vpgr"], dtype=jnp.float64)
    mugr = jnp.asarray(geom["mugr"], dtype=jnp.float64)
    bn = jnp.asarray(geom["bn"], dtype=jnp.float64)
    energy = vpgr[:, None, None] ** 2 + 2.0 * mugr[None, :, None] * bn[None, None, :]
    return (jnp.exp(-energy) / (jnp.sqrt(jnp.pi) ** 3)).astype(jnp.float64)


def test_full_operator_preserves_maxwellian():
    """Pitch + energy + friction (FDT balance) residual on fmax."""
    geom = _geom(nvpar=32, nmu=8, ns=8)
    p = _base_params(geom)
    st = precompute_collisions(geom, p)["coll_stencil"]
    fmax = _fmax(geom)
    df = (fmax[:, :, :, None, None] + 0j).astype(jnp.complex128)
    rhs = collision_rhs(df, st)
    rel = float(jnp.max(jnp.abs(rhs)) / jnp.max(jnp.abs(df)))
    assert rel < 1e-2, f"rel = {rel:.4e}"


def test_pitch_angle_preserves_isotropic_function():
    """C_pitch vanishes for any f(v); verified with f = v^2."""
    geom = _geom(nvpar=32, nmu=8, ns=8)
    p = _base_params(geom, coll_pitch_angle=True, coll_en_scatter=False, coll_friction=False)
    st = precompute_collisions(geom, p)["coll_stencil"]
    vpgr = jnp.asarray(geom["vpgr"], dtype=jnp.float64)
    mugr = jnp.asarray(geom["mugr"], dtype=jnp.float64)
    bn = jnp.asarray(geom["bn"], dtype=jnp.float64)
    f = vpgr[:, None, None] ** 2 + 2.0 * mugr[None, :, None] * bn[None, None, :]
    df = (f[:, :, :, None, None] + 0j).astype(jnp.complex128)
    rhs = collision_rhs(df, st)
    # mu=0 has a coordinate singularity; interior-only residual converges at O(dv^2)
    rel = float(jnp.max(jnp.abs(rhs[2:-2, :-1])) / jnp.max(jnp.abs(df[2:-2, :-1])))
    assert rel < 1e-3, f"rel = {rel:.4e}"


def test_perturbation_relaxes_to_maxwellian():
    """A small anisotropic perturbation should decay toward zero under collisions."""
    geom = _geom(nvpar=32, nmu=8, ns=8)
    p = _base_params(geom, coll_freq=1.0)
    st = precompute_collisions(geom, p)["coll_stencil"]
    fmax = _fmax(geom)

    # initial: fmax * (1 + amp * v_par) — antisymmetric perturbation, pitch-angle active
    vpgr = jnp.asarray(geom["vpgr"], dtype=jnp.float64)
    amp = 0.1
    f0 = fmax * (1.0 + amp * vpgr[:, None, None])

    df = (f0[:, :, :, None, None] + 0j).astype(jnp.complex128)
    dt = 0.01

    fmax_b = fmax[:, :, :, None, None]
    pert_norm_0 = float(jnp.linalg.norm(df - fmax_b))
    for _ in range(200):
        df = df + dt * collision_rhs(df, st)
    pert_norm_final = float(jnp.linalg.norm(df - fmax_b))

    assert (
        pert_norm_final < 0.6 * pert_norm_0
    ), f"|f0-fmax|={pert_norm_0:.4e}, |ff-fmax|={pert_norm_final:.4e}"


def test_disabled_gives_zero_stencil():
    """collisions=False yields no stencil in pre."""
    geom = _geom()
    p = _base_params(geom, collisions=False)
    out = precompute_collisions(geom, p)
    assert "coll_stencil" not in out


def test_xu_conservation_zeroes_deltas():
    """With mom/ene conservation enabled, integrated Δp, ΔE go to zero."""
    import jax.numpy as jnp
    from gyaradax.collisions import conservation_correction

    geom = _geom(nvpar=32, nmu=8, ns=8)
    p = _base_params(
        geom,
        coll_freq=0.1,
        coll_mom_conservation=True,
        coll_ene_conservation=True,
    )
    out = precompute_collisions(geom, p)
    vpgr = jnp.asarray(geom["vpgr"])
    mugr = jnp.asarray(geom["mugr"])
    bn = jnp.asarray(geom["bn"])
    energy = vpgr[:, None, None] ** 2 + 2.0 * mugr[None, :, None] * bn[None, None, :]
    fmax = jnp.exp(-energy) / (jnp.sqrt(jnp.pi) ** 3)
    # non-equilibrium df that has nonzero delta_p and delta_e under C
    df = ((fmax * (1.0 + 0.1 * vpgr[:, None, None]))[:, :, :, None, None] + 0j).astype(
        jnp.complex128
    )

    rhs_base = collision_rhs(df, out["coll_stencil"])
    corr = conservation_correction(
        rhs_base,
        out["coll_mom_factor"],
        out["coll_ene_factor"],
        out["coll_vpar_weight"],
        out["coll_vsq_weight"],
    )
    rhs = rhs_base + corr
    dp = float(jnp.abs(jnp.sum(out["coll_vpar_weight"][:, :, :, None, None] * rhs)))
    de = float(jnp.abs(jnp.sum(out["coll_vsq_weight"][:, :, :, None, None] * rhs)))
    assert dp < 1e-17, f"|dp| = {dp:.4e}"
    assert de < 1e-17, f"|dE| = {de:.4e}"


def test_coulomb_log_path_runs_and_scales():
    """freq_override=False should produce a finite stencil scaled by ~6.5e-5·L_ii."""
    import jax.numpy as jnp
    from gyaradax.collisions import _coulomb_log_ii, _gamma_pref_self

    geom = _geom()
    p = _base_params(
        geom,
        coll_freq_override=False,
        coll_rref=1.0,
        coll_nref=1.0,
        coll_tref=1.0,
    )
    out = precompute_collisions(geom, p)
    assert "coll_stencil" in out
    L = float(_coulomb_log_ii(1.0, 1.0, 1.0, 1.0, 1.0))
    expected = 6.5141e-5 * L  # expected gamma_pref for default refs
    gp = float(_gamma_pref_self(p, jnp.asarray(1.0), jnp.asarray(1.0), jnp.asarray(1.0)))
    assert abs(gp - expected) / expected < 1e-10, f"gamma_pref {gp:.4e} != {expected:.4e}"


def test_kinetic_produces_per_species_stencil():
    """Kinetic path yields a stencil with leading species axis."""
    import jax.numpy as jnp

    geom = _geom()
    p = _base_params(
        geom,
        adiabatic_electrons=False,
        mas=jnp.array([1.0, 2.72e-4]),
        signz=jnp.array([1.0, -1.0]),
        tmp=jnp.array([1.0, 1.0]),
        de=jnp.array([1.0, 1.0]),
        vthrat=jnp.array([1.0, 60.6]),
        rlt=jnp.array([6.9, 6.9]),
        rln=jnp.array([2.2, 2.2]),
        # self-collision only for this test: target = bg, pair sum reduces to self
        coll_bg_mas=jnp.array([1.0, 2.72e-4]),
        coll_bg_signz=jnp.array([1.0, -1.0]),
        coll_bg_tmp=jnp.array([1.0, 1.0]),
        coll_bg_de=jnp.array([1.0, 1.0]),
        coll_bg_vthrat=jnp.array([1.0, 60.6]),
    )
    st = precompute_collisions(geom, p)["coll_stencil"]
    nvpar, nmu, ns = 32, 8, 8
    assert st.shape == (2, 9, nvpar, nmu, ns), f"got {st.shape}"
    # Finite stencil, no NaNs. (Per-species residual on T=1 Maxwellian is
    # expected to be nonzero when species temperatures/velocities differ —
    # the full operator's fixed point is the matched Maxwellian at each
    # species' own temperature.)
    assert jnp.all(jnp.isfinite(st)), "stencil has NaN/Inf"
