"""Unit tests for the Fokker-Planck collision operator."""

import jax.numpy as jnp
import numpy as np
import pytest
from dataclasses import replace

from gyaradax.operators.collisions import precompute_collisions, collision_rhs
from gyaradax.geometry import compute_geometry
from gyaradax.core.params import GKParams


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
        coll_freq_override=True,
        coll_mass_conserve=True,
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
    # exact away from the mass-conserving vpar and mu_max edges, mu = 0 included
    rel = float(jnp.max(jnp.abs(rhs[2:-2, :-1])) / jnp.max(jnp.abs(df[2:-2, :-1])))
    assert rel < 1e-13, f"rel = {rel:.4e}"


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
        pert_norm_final < 0.7 * pert_norm_0
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
    from gyaradax.operators.collisions import conservation_correction

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
    def moments(r):
        dp = float(jnp.abs(jnp.sum(out["coll_vpar_weight"][:, :, :, None, None] * r)))
        de = float(jnp.abs(jnp.sum(out["coll_vsq_weight"][:, :, :, None, None] * r)))
        return dp, de

    (dp0, de0), (dp, de) = moments(rhs_base), moments(rhs)
    assert dp < 1e-12 * dp0, f"|dp| = {dp:.4e} of {dp0:.4e}"
    assert de < 1e-12 * de0, f"|dE| = {de:.4e} of {de0:.4e}"


def test_coulomb_log_path_runs_and_scales():
    """freq_override=False yields gamma_pref = 6.5e-5 * L_ii for default refs."""
    import jax.numpy as jnp
    from gyaradax.operators.collisions import _coulomb_log_ii, _gamma_pref_self

    geom = _geom()
    p = _base_params(
        geom, coll_freq_override=False,
        coll_rref=1.0, coll_nref=1.0, coll_tref=1.0,
    )
    assert "coll_stencil" in precompute_collisions(geom, p)
    L = float(_coulomb_log_ii(1.0, 1.0, 1.0, 1.0, 1.0))
    expected = 6.5141e-5 * L
    gp = float(_gamma_pref_self(p, jnp.asarray(1.0), jnp.asarray(1.0), jnp.asarray(1.0)))
    assert abs(gp - expected) / expected < 1e-13, f"rel = {abs(gp - expected) / expected:.4e}"


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


# GKW CoulombLog / CollFreqs (Coll_params.dat) of 1-step runs: ions (mass mi, T 1) and
# electrons (mass 2.72e-4, T te), equal densities, rref 6.229; rows scattered species
GKW_COLL_REFERENCE = [
    (
        dict(te=0.903, mi=1.0, tref=1.0, nref=10.0, fo=False, cf=0.0),
        [[16.953426409720027, 14.697967274434848], [14.697967274434848, 14.797967274434848]],
        [
            [6.8790780598274606e-2, 5.9638955428890500e-2],
            [7.3139927850796954e-2, 7.3637547191520450e-2],
        ],
    ),
    (
        dict(te=1.5, mi=1.0, tref=20.0, nref=10.0, fo=False, cf=0.0),
        [[21.447024820051016, 18.201197381662155], [18.201197381662155, 18.301197381662156]],
        [
            [2.1756038325621333e-4, 1.8463444283303560e-4],
            [8.2059752370238046e-5, 8.2510600469126944e-5],
        ],
    ),
    (
        dict(te=0.5, mi=1.5, tref=5.0, nref=3.0, fo=False, cf=0.0),
        [[19.969569680534146, 16.318277134037125], [16.318277134037125, 16.418277134037123]],
        [
            [9.7235019281858583e-4, 7.9456293608644779e-4],
            [3.1782517443457912e-3, 3.1977283822177908e-3],
        ],
    ),
    (
        dict(te=1.0, mi=1.0, tref=0.005, nref=1.0, fo=False, cf=0.0),
        [[10.157242906395000, 10.403816496674967], [10.403816496674967, 10.752975179948987]],
        [[164.85745155483019, 168.85947199373172], [168.85947199373172, 174.52650302205873]],
    ),
    (
        dict(te=0.7, mi=1.0, tref=8.0, nref=10.0, fo=True, cf=2e-3),
        [[20.072588722239782, 16.522766597741104], [16.522766597741104, 16.622766597741105]],
        [[2.0e-3, 1.6463015136093941e-3], [3.3597990073661109e-3, 3.3801333683672855e-3]],
    ),
]


def _kinetic_params(te=1.0, mi=1.0, **overrides):
    mas = jnp.array([mi, 2.72e-4])
    tmp = jnp.array([1.0, te])
    overrides = {"coll_freq_override": True, "coll_mass_conserve": True, **overrides}
    return GKParams(
        adiabatic_electrons=False,
        collisions=True,
        mas=mas,
        tmp=tmp,
        de=jnp.array([1.0, 1.0]),
        signz=jnp.array([1.0, -1.0]),
        vthrat=jnp.sqrt(tmp / mas),
        rlt=jnp.array([6.9, 6.9]),
        rln=jnp.array([2.2, 2.2]),
        **overrides,
    )


@pytest.mark.parametrize("case, coulomb_log, freqs", GKW_COLL_REFERENCE)
def test_collision_frequencies_match_gkw(case, coulomb_log, freqs):
    """Coulomb logs (e-e, e-i low/high T, i-e, i-i) and Gamma^(a/b) as GKW collision_init."""
    from gyaradax.operators.collisions import collision_frequencies

    p = _kinetic_params(
        te=case["te"],
        mi=case["mi"],
        coll_freq_override=case["fo"],
        coll_freq=case["cf"],
        coll_rref=6.229,
        coll_nref=case["nref"],
        coll_tref=case["tref"],
    )
    L, gamma = collision_frequencies(p)
    np.testing.assert_allclose(np.asarray(L), coulomb_log, rtol=1e-13)
    np.testing.assert_allclose(np.asarray(gamma), freqs, rtol=1e-13)


def test_yaml_collisions_section_sets_references():
    """The YAML section carries rref/nref/tref and freq_override false."""
    from omegaconf import OmegaConf
    from gyaradax.core.params import gkparams_from_config

    def cfg(coll):
        return OmegaConf.create(
            {
                "solver": {"dt": 0.01},
                "physics": {"mas": [1.0, 2.72e-4], "tmp": [1.0, 0.9], "signz": [1.0, -1.0]},
                "grid": {"adiabatic_electrons": False, "nvpar": 16, "nmu": 8, "ns": 16},
                "collisions": coll,
            }
        )

    refs = {"rref": 6.229, "nref": 10.0, "tref": 8.0}
    p = gkparams_from_config(cfg({"freq_override": False, **refs}))
    assert p.collisions and not p.coll_freq_override
    assert (p.coll_rref, p.coll_nref, p.coll_tref) == (6.229, 10.0, 8.0)
    assert p.coll_pitch_angle and p.coll_en_scatter and p.coll_friction
    # GKW defaults (collisionop_read_nml)
    p = gkparams_from_config(cfg({}))
    assert not (p.coll_freq_override or p.coll_mass_conserve)
    assert not (p.coll_mom_conservation or p.coll_ene_conservation)
    assert (p.coll_freq, p.coll_rref, p.coll_nref, p.coll_tref) == (0.0, 1.0, 1.0, 1.0)
    with pytest.raises(ValueError, match="needs freq > 0"):
        gkparams_from_config(cfg({"freq_override": True}))
    with pytest.raises(ValueError, match="only with freq_override"):
        gkparams_from_config(cfg({"freq": 0.01}))
    with pytest.raises(ValueError, match="positive"):
        gkparams_from_config(cfg({"freq_override": False, "tref": 0.0}))
    assert gkparams_from_config(cfg({"enabled": False})).collisions is False


def test_kinetic_timestep_bound_includes_electron_collisions():
    """The collision bound of the linear time step takes every species' diagonal."""
    from gyaradax.operators.precompute import linear_precompute
    from gyaradax.solver.cfl import estimate_linear_timestep
    from gyaradax.solver.driver import gk_init

    geom = _geom(nvpar=16, nmu=8, ns=8)
    p = replace(
        _kinetic_params(te=0.9, coll_freq=5.0),
        dvp=float(geom["dvp"]),
        sgr_dist=float(geom["sgr_dist"]),
    )
    _, geom, _ = gk_init(geom, p, n_species=2)
    pre = linear_precompute(geom, p)
    st = np.asarray(pre["coll_stencil"])
    tmax2 = np.abs(st[:, 0]).max()
    assert np.abs(st[1, 0]).max() > 10 * np.abs(st[0]).max()
    dt = float(estimate_linear_timestep(pre, params=p))
    np.testing.assert_allclose(dt, p.fac_dtim_est / tmax2, rtol=1e-12)


def test_kinetic_conservation_uses_self_collisions():
    """Momentum / energy corrections cancel the self-collision moments only (GKW selfcollcon)."""
    from gyaradax.operators.collisions import collision_term

    geom = _geom(nvpar=16, nmu=8, ns=4)
    p = _kinetic_params(
        te=0.8, coll_freq=0.1, coll_mom_conservation=True, coll_ene_conservation=True
    )
    pre = precompute_collisions(geom, p)
    assert pre["coll_self_stencil"].shape == pre["coll_stencil"].shape
    vp = jnp.asarray(geom["vpgr"])[:, None, None]
    fm = _fmax(geom)
    df = jnp.stack([fm * (1.0 + 0.1 * vp), fm * (1.0 - 0.2 * vp + 0.05 * vp**2)])
    df = (df[..., None, None] + 0j).astype(jnp.complex128)
    total = collision_term(df, p, pre)
    for sp in range(2):
        self_part = collision_rhs(df[sp], pre["coll_self_stencil"][sp])
        conserved = self_part + total[sp] - collision_rhs(df[sp], pre["coll_stencil"][sp])
        for key in ("coll_vpar_weight", "coll_vsq_weight"):
            w = pre[key][sp][..., None, None]
            before = float(jnp.abs(jnp.sum(w * self_part)))
            after = float(jnp.abs(jnp.sum(w * conserved)))
            assert after < 1e-12 * before, f"species {sp} {key}: {after:.3e} of {before:.3e}"
    # the inter-species momentum exchange (e-i friction) is kept
    w = pre["coll_vpar_weight"][1][..., None, None]
    assert float(jnp.abs(jnp.sum(w * total[1]))) > 1e-6 * float(jnp.sum(jnp.abs(w * total[1])))


def test_conservation_maxwellian_is_the_species_own():
    """The restoring terms use F_M = exp(-v^2) on the species' own thermal-speed grid."""
    geom = _geom(nvpar=16, nmu=8, ns=4)
    p = _kinetic_params(te=0.5, coll_freq=0.1, coll_mom_conservation=True)
    pre = precompute_collisions(geom, p)
    vp = jnp.asarray(geom["vpgr"])[:, None, None]
    expected = vp * _fmax(geom)
    for sp in range(2):
        ratio = np.asarray(pre["coll_mom_factor"][sp] / expected)
        np.testing.assert_allclose(ratio, np.broadcast_to(ratio[:1, :1], ratio.shape), rtol=1e-12)


@pytest.mark.parametrize(
    "extra, refused",
    [
        ("", None),
        ("lorentz = .true.", "lorentz"),
        ("selfcollcon = .false.", "selfcollcon"),
        ("cons_type = 'Lin'", "cons_type"),
        ("zeff = 1.5", "zeff"),
    ],
)
def test_gkw_import_collision_defaults(tmp_path, extra, refused):
    """GKW import takes GKW's namelist defaults and refuses options gyaradax lacks."""
    from gyaradax.io.gkw import load_runtime_params

    path = tmp_path / "input.dat"
    path.write_text(f"&control\n collisions = .true.\n/\n&collisions\n nref = 10.0\n {extra}\n/\n")
    if refused:
        with pytest.raises(NotImplementedError, match=refused):
            load_runtime_params(str(path))
        return
    rt = load_runtime_params(str(path))
    assert not (rt["coll_freq_override"] or rt["coll_mass_conserve"])
    assert not (rt["coll_mom_conservation"] or rt["coll_ene_conservation"])
    assert (rt["coll_freq"], rt["coll_rref"], rt["coll_nref"], rt["coll_tref"]) == (0, 1, 10, 1)
