"""Run parameters: the :class:`GKParams` pytree and its builders from YAML and GKW input."""

import math
import os
from dataclasses import dataclass
from typing import Any, Dict

import jax
import jax.numpy as jnp
import numpy as np
from omegaconf import OmegaConf

from gyaradax.core.jax_config import enable_x64
from gyaradax.io.gkw import as_int, load_runtime_params, load_scalars, parse_input_dat, species_blocks

enable_x64()


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class GKParams:
    """Physical parameters and solver controls of a gyrokinetic run.

    A frozen dataclass registered as a JAX pytree. Strings and the flags that
    select code paths are static (pytree auxiliary data), so changing them
    retraces; floats such as gradients stay leaves and can be differentiated.
    Species fields (``rlt``, ``rln``, ``mas``, ``tmp``, ``de``, ``signz``,
    ``vthrat``) are scalars for one species or arrays of length ``nsp``.
    Field names follow the GKW namelists (``control``, ``gridsize``,
    ``species``, ``geom``, ``collisions``, ``rotation``).

    Attributes:
        dt: Time step (``dtim``); the initial step when ``adaptive_dt`` is set.
        naverage: Steps per normalisation / growth-rate window.
        disp_par: Parallel (s) hyper-dissipation coefficient.
        disp_vp: Parallel-velocity dissipation coefficient.
        disp_x: Radial (kx) hyper-dissipation coefficient.
        disp_y: Binormal (ky) hyper-dissipation coefficient.
        idisp: Dissipation scaling, as GKW ``idisp`` (1 / -1: abs(vpar), 2: rms velocity).
        disp_par_conserve: Conservative parallel dissipation. 0 off, 1 removes
            the Ampere current moment from the dissipated part, 2 also the
            Poisson moment.
        disp_par_conserve_kycut: Apply the conservative projection only for
            ``krho`` at or below this value; ``<= 0`` applies it to every ky.
        drive_scale: Scale of the gradient drives (Terms V and VIII); keep at 1.
        norm_eps: Floor of the per-ky mode amplitude in the normalisation.
        non_linear: Include the ExB nonlinearity (Term III).
        disable_per_ky_norm: Skip the per-ky amplitude renormalisation of linear runs.
        finit: Initial condition (``cosine2``, ``cosine``, ``cosine3``, ``noise``,
            ``gnoise``, ``zonal``), see :func:`gyaradax.solver.rk4.init_f`.
        amp_init: Initial amplitude.
        adiabatic_electrons: Adiabatic (Boltzmann) electrons instead of a kinetic species.
        adaptive_dt: Adapt dt every block from the CFL estimate.
        cfl_safety: Safety factor of the nonlinear CFL estimate.
        fac_dtim_est: Safety factor of the linear CFL estimate (GKW ``fac_dtim_est``).
        fac_dtim_nl: Nonlinear CFL factor (GKW ``fac_dtim_nl``).
        spectral_radius: Use the RK4 spectral-radius stability bound in the linear CFL.
        mixed_precision: Run the nonlinear FFTs in FP32 (fields and linear terms stay FP64).
        backend: ``"jax"``, ``"cuda"`` or ``"auto"``.
        use_z2z: Z2Z instead of R2C FFTs in the JAX bracket (CUDA always uses Z2Z).
        fused_linear: CUDA: one fused electrostatic linear kernel (with
            collisions, RK4 stage updates and field moments).
        fused_bracket: CUDA: pack and unpack the mixed-precision v6 bracket
            inside the cuFFT callbacks.
        linear_arith: ``"fp64"`` or ``"ds"`` (double-single electrostatic linear RHS, CUDA).
        vcor: Rigid toroidal rotation ``Rref * Omega / vthref`` (Coriolis drift).
        uprim: Toroidal velocity gradient per species.
        nlapar: Evolve the parallel vector potential A_par.
        nlbpar: Evolve the parallel magnetic perturbation B_par.
        beta: Reference plasma beta ``2 mu0 n_ref T_ref / B_ref^2``.
        betaprime: Pressure-gradient correction of the curvature drift (GKW
            ``betaprime_ref``; negative for a peaked profile).
        drift_gradp_type: ``"full_drift"`` (vpar^2 term on the full drift) or
            ``"curv_only"`` (vpar^2 + B mu), as GKW ``drift_gradp_type``.
        collisions: Enable the linearised Fokker-Planck operator.
        coll_pitch_angle: Pitch-angle scattering.
        coll_en_scatter: Energy scattering.
        coll_friction: Friction.
        coll_freq: Collision frequency, used with ``coll_freq_override``.
        coll_freq_override: Use ``coll_freq`` instead of the Coulomb-log frequency.
        coll_mass_conserve: Enforce particle conservation.
        coll_rref: Reference major radius for the Coulomb log [m].
        coll_tref: Reference temperature for the Coulomb log [keV].
        coll_nref: Reference density for the Coulomb log [1e19 m^-3].
        coll_bg_mas: Masses of the collision backgrounds (kinetic and adiabatic species).
        coll_bg_signz: Charges of the collision backgrounds.
        coll_bg_tmp: Temperatures of the collision backgrounds.
        coll_bg_de: Densities of the collision backgrounds.
        coll_bg_vthrat: Thermal-velocity ratios of the collision backgrounds.
        coll_mom_conservation: Momentum conservation correction (Xu).
        coll_ene_conservation: Energy conservation correction (Xu).
        rlt: Normalised temperature gradient R/L_T.
        rln: Normalised density gradient R/L_n.
        mas: Mass, in reference masses.
        tmp: Temperature, in reference temperatures.
        de: Density, in reference densities.
        signz: Charge number.
        vthrat: Thermal-velocity ratio ``sqrt(tmp / mas)``.
        shat: Magnetic shear.
        q: Safety factor.
        eps: Inverse aspect ratio r/R.
        kthnorm: Binormal wavenumber normalisation.
        Rref: Reference major radius (normalised).
        d2X: Radial derivative normalisation of the geometry.
        signB: Sign of the magnetic field.
        dvp: Parallel-velocity grid spacing.
        sgr_dist: Parallel (s) grid spacing.
        kxmax: Largest radial wavenumber (internal units).
        kymax: Largest binormal wavenumber (internal units).
        dgrid: Grid density normalisation of the Maxwellian.
        tgrid: Grid temperature normalisation.
        n_gpus_sp: Devices on the species axis.
        n_gpus_vp: Devices on the vpar axis.
        n_gpus_mu: Devices on the mu axis.
    """

    dt: float = 0.01
    naverage: int = 40
    disp_par: float = 1.0
    disp_vp: float = 0.2
    disp_x: float = 0.1
    disp_y: float = 0.1
    idisp: int = 2
    disp_par_conserve: int = 0
    disp_par_conserve_kycut: float = 0.0
    drive_scale: float = 1.0
    norm_eps: float = 1.0e-14
    non_linear: bool = False
    disable_per_ky_norm: bool = False
    finit: str = "cosine2"
    amp_init: float = 1.0e-4
    adiabatic_electrons: bool = True
    adaptive_dt: bool = False
    cfl_safety: float = 0.95
    fac_dtim_est: float = 0.95
    fac_dtim_nl: float = 1.0
    spectral_radius: bool = True
    mixed_precision: bool = True
    backend: str = "jax"
    use_z2z: bool = False
    fused_linear: bool = False
    fused_bracket: bool = False
    linear_arith: str = "fp64"

    # rotation: centrifugal terms are not implemented
    vcor: float = 0.0
    uprim: Any = 0.0

    nlapar: bool = False
    nlbpar: bool = False
    beta: float = 0.0
    betaprime: float = 0.0
    drift_gradp_type: str = "full_drift"

    collisions: bool = False
    coll_pitch_angle: bool = True
    coll_en_scatter: bool = True
    coll_friction: bool = True
    coll_freq: float = 0.0
    coll_freq_override: bool = False
    coll_mass_conserve: bool = False
    coll_rref: float = 1.0
    coll_tref: float = 1.0
    coll_nref: float = 1.0
    coll_bg_mas: Any = None
    coll_bg_signz: Any = None
    coll_bg_tmp: Any = None
    coll_bg_de: Any = None
    coll_bg_vthrat: Any = None
    coll_mom_conservation: bool = False
    coll_ene_conservation: bool = False

    rlt: float = 1.0
    rln: float = 1.0
    mas: float = 1.0
    tmp: float = 1.0
    de: float = 1.0
    signz: float = 1.0
    vthrat: float = 1.0

    shat: float = 0.0
    q: float = 1.0
    eps: float = 0.0
    kthnorm: float = 1.0
    Rref: float = 1.0
    d2X: float = 1.0
    signB: float = 1.0

    dvp: float = 1.0
    sgr_dist: float = 1.0
    kxmax: float = 1.0
    kymax: float = 1.0
    dgrid: float = 1.0
    tgrid: float = 1.0

    n_gpus_sp: int = 1
    n_gpus_vp: int = 1
    n_gpus_mu: int = 1

    # stored as pytree auxiliary data, not leaves
    _STATIC_FIELDS = (
        "finit",
        "adiabatic_electrons",
        "non_linear",
        "disable_per_ky_norm",
        "adaptive_dt",
        "mixed_precision",
        "backend",
        "use_z2z",
        "fused_linear",
        "fused_bracket",
        "linear_arith",
        "nlapar",
        "nlbpar",
        "collisions",
        "coll_pitch_angle",
        "coll_en_scatter",
        "coll_friction",
        "coll_freq",
        "coll_freq_override",
        "coll_mass_conserve",
        "coll_mom_conservation",
        "coll_ene_conservation",
        "coll_rref",
        "coll_tref",
        "coll_nref",
        "coll_bg_mas",
        "coll_bg_signz",
        "coll_bg_tmp",
        "coll_bg_de",
        "coll_bg_vthrat",
        "n_gpus_sp",
        "n_gpus_vp",
        "n_gpus_mu",
        "dt",
        "naverage",
        "disp_par",
        "disp_vp",
        "disp_x",
        "disp_y",
        "idisp",
        "disp_par_conserve",
        "disp_par_conserve_kycut",
        "drive_scale",
        "norm_eps",
        "cfl_safety",
        "fac_dtim_est",
        "fac_dtim_nl",
        "spectral_radius",
        "amp_init",
        "dvp",
        "sgr_dist",
        "kxmax",
        "kymax",
        "mas",
        "tmp",
        "de",
        "signz",
        "vthrat",
        "dgrid",
        "tgrid",
        "vcor",
        "uprim",
        "betaprime",
        "drift_gradp_type",
    )

    def tree_flatten(self):
        d = vars(self)
        aux = {}
        leaves = []
        leaf_keys = []
        for k, v in d.items():
            if k.startswith("_"):
                continue
            # arrays must be leaves (aux entries need to be hashable)
            if k in self._STATIC_FIELDS and not hasattr(v, "shape"):
                aux[k] = v
            else:
                leaves.append(v)
                leaf_keys.append(k)
        aux["_leaf_keys"] = tuple(leaf_keys)
        return leaves, aux

    @classmethod
    def tree_unflatten(cls, aux, leaves):
        leaf_keys = aux["_leaf_keys"]
        kwargs = {k: v for k, v in aux.items() if k != "_leaf_keys"}
        kwargs.update(zip(leaf_keys, leaves))
        return cls(**kwargs)


def gkparams_from_runtime(runtime: Dict[str, Any], **overrides: Any) -> GKParams:
    """Build GKParams from a GKW-compatible runtime-controls dictionary."""
    non_linear = bool(runtime.get("non_linear", False))
    nl_dtim_est = bool(runtime.get("nl_dtim_est", True))
    fac_dtim_est = float(runtime.get("fac_dtim_est", 0.95))
    fac_dtim_nl = float(runtime.get("fac_dtim_nl", 1.0))
    params_dict: dict[str, Any] = {
        "dt": float(runtime.get("dtim", 0.01)),
        "naverage": int(runtime.get("naverage", 40)),
        "disp_par": float(runtime.get("disp_par", 1.0)),
        "disp_vp": float(runtime.get("disp_vp", 0.2)),
        "disp_x": float(runtime.get("disp_x", 0.1)),
        "disp_y": float(runtime.get("disp_y", 0.1)),
        "non_linear": non_linear,
        "finit": str(runtime.get("finit", "cosine2")),
        "amp_init": float(runtime.get("amp_init", 1.0e-4)),
        "adiabatic_electrons": bool(runtime.get("adiabatic_electrons", True)),
        "adaptive_dt": non_linear and nl_dtim_est,
        "cfl_safety": fac_dtim_nl,
        "fac_dtim_est": fac_dtim_est,
        "fac_dtim_nl": fac_dtim_nl,
        "spectral_radius": bool(runtime.get("spectral_radius", True)),
        "backend": str(runtime.get("backend", "jax")),
        "fused_linear": bool(runtime.get("fused_linear", False)),
        "fused_bracket": bool(runtime.get("fused_bracket", False)),
        "linear_arith": str(runtime.get("linear_arith", "fp64")),
        "nlapar": bool(runtime.get("nlapar", False)),
        "nlbpar": bool(runtime.get("nlbpar", False)),
        "beta": float(runtime.get("beta", 0.0)),
        "betaprime": float(runtime.get("betaprime", 0.0)),
        "drift_gradp_type": str(runtime.get("drift_gradp_type", "full_drift")),
        "collisions": bool(runtime.get("collisions", False)),
        "coll_pitch_angle": bool(runtime.get("coll_pitch_angle", True)),
        "coll_en_scatter": bool(runtime.get("coll_en_scatter", True)),
        "coll_friction": bool(runtime.get("coll_friction", True)),
        "coll_freq": float(runtime.get("coll_freq", 0.0)),
        "coll_freq_override": bool(runtime.get("coll_freq_override", False)),
        "coll_mass_conserve": bool(runtime.get("coll_mass_conserve", False)),
        "coll_mom_conservation": bool(runtime.get("coll_mom_conservation", False)),
        "coll_ene_conservation": bool(runtime.get("coll_ene_conservation", False)),
        "coll_rref": float(runtime.get("coll_rref", 1.0)),
        "coll_tref": float(runtime.get("coll_tref", 1.0)),
        "coll_nref": float(runtime.get("coll_nref", 1.0)),
        "n_gpus_sp": int(runtime.get("n_gpus_sp", 1)),
        "n_gpus_vp": int(runtime.get("n_gpus_vp", 1)),
        "n_gpus_mu": int(runtime.get("n_gpus_mu", 1)),
    }
    for k in ("coll_bg_mas", "coll_bg_signz", "coll_bg_tmp", "coll_bg_de", "coll_bg_vthrat"):
        if k in runtime:
            params_dict[k] = runtime[k]
    # species params may be arrays (multi-species) or scalars
    _SPECIES_PARAMS = {"rlt", "rln", "mas", "tmp", "de", "signz", "vthrat"}
    for k in _SPECIES_PARAMS:
        if k in runtime:
            v = runtime[k]
            params_dict[k] = v if hasattr(v, "__len__") else float(v)

    # geometry and grid scalars (always scalar)
    for k in [
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
        "dgrid",
        "tgrid",
    ]:
        if k in runtime:
            params_dict[k] = float(runtime[k])

    if overrides:
        params_dict.update(overrides)
    return GKParams(**params_dict)


def gkparams_from_input_dat(input_dat_path: str, **overrides: Any) -> GKParams:
    """Load all runtime, physics, and geometry scalars from a GKW run directory.

    Args:
        input_dat_path: Path to the GKW input.dat file.
        overrides: Manual parameter overrides.

    Returns:
        Configured GKParams instance.
    """
    directory = os.path.dirname(input_dat_path)
    scalars = load_scalars(directory)
    return gkparams_from_runtime(scalars, **overrides)


def gkparams_from_input_and_geometry(
    input_dat_path: str, geometry: Dict[str, Any], **overrides: Any
) -> GKParams:
    """Build GKParams from a GKW ``input.dat`` and a computed geometry dict.

    Unlike :func:`gkparams_from_input_dat` this needs no ``geom.dat`` or other
    GKW output: the geometry scalars come from ``geometry``. Adiabatic runs keep
    only the ion species; every species block becomes a collision background.

    Args:
        input_dat_path: Path to the GKW ``input.dat``.
        geometry: Geometry dict, e.g. from :func:`gyaradax.geometry.compute_geometry_from_input`.
        overrides: Parameter overrides.

    Returns:
        GKParams.
    """
    runtime = load_runtime_params(input_dat_path)
    inp = parse_input_dat(input_dat_path)

    scalars: dict[str, Any] = {}
    for k in (
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
    ):
        if k in geometry:
            scalars[k] = float(np.asarray(geometry[k]).reshape(-1)[0])

    # target species = first num_sp blocks (GKW convention). Collision backgrounds
    # = all species blocks (may include adiabatic-electron block beyond num_sp).
    # Kinetic-species set filters Z>0 when adiabatic.
    num_sp = as_int(inp.get("gridsize", {}).get("number_of_species", 1), 1)
    all_species = species_blocks(inp)
    species = species_blocks(inp, limit=num_sp)
    all_mas = np.array([float(sp.get("mass", 1.0)) for sp in all_species])
    all_tmp = np.array([float(sp.get("temp", 1.0)) for sp in all_species])
    all_de = np.array([float(sp.get("dens", 1.0)) for sp in all_species])
    all_signz = np.array([float(sp.get("z", 1.0)) for sp in all_species])
    all_vthrat = np.sqrt(all_tmp / all_mas)

    if species:
        sp_mas = np.array([float(sp.get("mass", 1.0)) for sp in species])
        sp_tmp = np.array([float(sp.get("temp", 1.0)) for sp in species])
        sp_de = np.array([float(sp.get("dens", 1.0)) for sp in species])
        sp_signz = np.array([float(sp.get("z", 1.0)) for sp in species])
        sp_rlt = np.array([float(sp.get("rlt", 0.0)) for sp in species])
        sp_rln = np.array([float(sp.get("rln", 0.0)) for sp in species])
        sp_vthrat = np.sqrt(sp_tmp / sp_mas)

        # adiabatic path evolves only kinetic (non-electron) species, dropping Z<0
        # from the target list; Boltzmann electron lives in the QN denominator.
        ae_val = inp.get("gridsize", {}).get("adiabatic_electrons")
        if ae_val is None:
            ae_val = inp.get("spcgeneral", {}).get("adiabatic_electrons", True)
        if bool(ae_val):
            keep = sp_signz > 0
            if keep.any():
                sp_mas = sp_mas[keep]
                sp_tmp = sp_tmp[keep]
                sp_de = sp_de[keep]
                sp_signz = sp_signz[keep]
                sp_rlt = sp_rlt[keep]
                sp_rln = sp_rln[keep]
                sp_vthrat = sp_vthrat[keep]

        def _maybe_scalar(arr):
            return float(arr[0]) if len(arr) == 1 else arr

        scalars.update(
            {
                "mas": _maybe_scalar(sp_mas),
                "tmp": _maybe_scalar(sp_tmp),
                "de": _maybe_scalar(sp_de),
                "signz": _maybe_scalar(sp_signz),
                "rlt": _maybe_scalar(sp_rlt),
                "rln": _maybe_scalar(sp_rln),
                "vthrat": _maybe_scalar(sp_vthrat),
            }
        )
        # collision backgrounds = ALL species (kinetic + adiabatic)
        if len(all_mas) > 0:
            scalars.update(
                {
                    "coll_bg_mas": all_mas,
                    "coll_bg_tmp": all_tmp,
                    "coll_bg_de": all_de,
                    "coll_bg_signz": all_signz,
                    "coll_bg_vthrat": all_vthrat,
                }
            )

    scalars.update(runtime)
    if overrides:
        scalars.update(overrides)
    return gkparams_from_runtime(scalars)


def load_config(config_path: str) -> Any:
    return OmegaConf.load(config_path)


def gkparams_from_config(config: Any, **overrides: Any) -> GKParams:
    """Build GKParams from a YAML run configuration.

    Reads the ``solver``, ``physics``, ``geometry``, ``grid``, ``sharding`` and
    ``collisions`` sections. Grid spacings (``sgr_dist``, ``dvp``) and the
    largest wavenumbers (``kxmax``, ``kymax``) are derived from ``grid`` when the
    config does not set them, as :func:`gyaradax.geometry.compute_geometry` does.

    Args:
        config: Config from :func:`load_config`.
        overrides: Parameter overrides, applied last.

    Returns:
        GKParams.
    """
    solver_cfg = config.solver
    physics_cfg = getattr(config, "physics", {})
    geometry_cfg = getattr(config, "geometry", {})

    fac_dtim_nl = float(
        getattr(solver_cfg, "fac_dtim_nl", getattr(solver_cfg, "cfl_safety", 0.95))
    )
    cfl_safety = fac_dtim_nl

    params_dict: dict[str, Any] = {
        "dt": float(getattr(solver_cfg, "dt", 0.01)),
        "naverage": int(getattr(solver_cfg, "naverage", 40)),
        "disp_par": float(getattr(solver_cfg, "disp_par", 1.0)),
        "disp_vp": float(getattr(solver_cfg, "disp_vp", 0.2)),
        "disp_x": float(getattr(solver_cfg, "disp_x", 0.1)),
        "disp_y": float(getattr(solver_cfg, "disp_y", 0.1)),
        "idisp": int(getattr(solver_cfg, "idisp", 2)),
        "disp_par_conserve": int(getattr(solver_cfg, "disp_par_conserve", 0)),
        "disp_par_conserve_kycut": float(
            getattr(solver_cfg, "disp_par_conserve_kycut", 0.0)
        ),
        "non_linear": bool(getattr(solver_cfg, "non_linear", False)),
        "finit": str(getattr(solver_cfg, "finit", "cosine2")),
        "amp_init": float(getattr(solver_cfg, "amp_init", 1.0e-4)),
        "drive_scale": float(getattr(solver_cfg, "drive_scale", 1.0)),
        "adiabatic_electrons": bool(getattr(config.grid, "adiabatic_electrons", True)),
        "adaptive_dt": bool(getattr(solver_cfg, "adaptive_dt", False)),
        "cfl_safety": cfl_safety,
        "fac_dtim_est": float(getattr(solver_cfg, "fac_dtim_est", 0.95)),
        "fac_dtim_nl": fac_dtim_nl,
        "spectral_radius": bool(getattr(solver_cfg, "spectral_radius", True)),
        "backend": str(getattr(solver_cfg, "backend", "jax")),
        "nlapar": bool(getattr(solver_cfg, "nlapar", False)),
        "nlbpar": bool(getattr(solver_cfg, "nlbpar", False)),
        "mixed_precision": bool(getattr(solver_cfg, "mixed_precision", True)),
        "use_z2z": bool(getattr(solver_cfg, "use_z2z", False)),
        "fused_linear": bool(getattr(solver_cfg, "fused_linear", False)),
        "fused_bracket": bool(getattr(solver_cfg, "fused_bracket", False)),
        "linear_arith": str(getattr(solver_cfg, "linear_arith", "fp64")),
        "disable_per_ky_norm": bool(getattr(solver_cfg, "disable_per_ky_norm", False)),
        "beta": float(getattr(physics_cfg, "beta", 0.0)),
        "betaprime": float(getattr(physics_cfg, "betaprime", 0.0)),
        "drift_gradp_type": str(getattr(physics_cfg, "drift_gradp_type", "full_drift")),
    }

    # optional sharding: config.sharding.{n_gpus_sp, n_gpus_vp, n_gpus_mu}
    shard_cfg = getattr(config, "sharding", None)
    if shard_cfg is not None:
        params_dict["n_gpus_sp"] = int(getattr(shard_cfg, "n_gpus_sp", 1))
        params_dict["n_gpus_vp"] = int(getattr(shard_cfg, "n_gpus_vp", 1))
        params_dict["n_gpus_mu"] = int(getattr(shard_cfg, "n_gpus_mu", 1))

    # collision operator: optional 'collisions' section in the YAML config
    coll_cfg = getattr(config, "collisions", None)
    if coll_cfg is not None:
        params_dict.update(
            {
                "collisions": bool(getattr(coll_cfg, "enabled", True)),
                "coll_pitch_angle": bool(getattr(coll_cfg, "pitch_angle", True)),
                "coll_en_scatter": bool(getattr(coll_cfg, "en_scatter", True)),
                "coll_friction": bool(getattr(coll_cfg, "friction", True)),
                "coll_freq": float(getattr(coll_cfg, "freq", 0.0)),
                "coll_freq_override": bool(getattr(coll_cfg, "freq_override", False)),
                "coll_mass_conserve": bool(getattr(coll_cfg, "mass_conserve", False)),
                "coll_mom_conservation": bool(getattr(coll_cfg, "mom_conservation", False)),
                "coll_ene_conservation": bool(getattr(coll_cfg, "ene_conservation", False)),
                "coll_rref": float(getattr(coll_cfg, "rref", 1.0)),
                "coll_nref": float(getattr(coll_cfg, "nref", 1.0)),
                "coll_tref": float(getattr(coll_cfg, "tref", 1.0)),
            }
        )
        if params_dict["collisions"]:
            if params_dict["coll_freq_override"] and params_dict["coll_freq"] <= 0.0:
                raise ValueError("collisions: freq_override: true needs freq > 0")
            if not params_dict["coll_freq_override"] and params_dict["coll_freq"] > 0.0:
                raise ValueError("collisions: freq is used only with freq_override: true")
            if min(params_dict[k] for k in ("coll_rref", "coll_nref", "coll_tref")) <= 0.0:
                raise ValueError("collisions: rref, nref and tref must be positive")

    # physics scalars (may be arrays for multi-species kinetic configs)
    _SPECIES_PARAMS = {"rlt", "rln", "mas", "tmp", "de", "signz", "vthrat"}
    _vthrat_explicit = hasattr(physics_cfg, "vthrat")
    for k in ["rlt", "rln", "mas", "tmp", "de", "signz", "vthrat", "dgrid", "tgrid"]:
        if hasattr(physics_cfg, k):
            v = getattr(physics_cfg, k)
            if k in _SPECIES_PARAMS and hasattr(v, "__iter__") and not isinstance(v, str):
                params_dict[k] = jnp.array([float(x) for x in v])
            elif hasattr(v, "__iter__") and not isinstance(v, str):
                # scalar param stored as list in yaml (e.g. dgrid: [1.0, 1.0]) — take first
                params_dict[k] = float(list(v)[0])
            else:
                params_dict[k] = float(v)

    # GKW defines vthrat = sqrt(T_s/m_s); old configs used sqrt(tgrid/mas) which
    # was wrong for electrons (missing T_e factor). Default to sqrt(tmp/mas).
    if not _vthrat_explicit and "tmp" in params_dict and "mas" in params_dict:
        _tmp = np.asarray(params_dict["tmp"], dtype=float)
        _mas = np.asarray(params_dict["mas"], dtype=float)
        params_dict["vthrat"] = np.sqrt(_tmp / _mas)

    # geometry scalars
    for k in ["shat", "q", "eps", "kthnorm", "Rref", "d2X", "signB"]:
        if hasattr(geometry_cfg, k):
            params_dict[k] = float(getattr(geometry_cfg, k))

    # scaling/grid scalars
    for k in ["dvp", "sgr_dist", "kxmax", "kymax"]:
        if hasattr(geometry_cfg, k):
            params_dict[k] = float(getattr(geometry_cfg, k))

    # derive sgr_dist/dvp from grid params when missing (matches compute_geometry).
    # Without these, the parallel stencil divides by sgr_dist=1.0 instead of 1/ns,
    # making terms I and VII ~ns times too weak. Same applies to dvp/kxmax/kymax.
    grid_cfg = getattr(config, "grid", None)
    if "sgr_dist" not in params_dict and grid_cfg is not None:
        ns = int(getattr(grid_cfg, "ns", 16))
        nperiod = int(getattr(grid_cfg, "nperiod", 1))
        if ns > 1:
            params_dict["sgr_dist"] = (2.0 * nperiod - 1.0) / ns
    if "dvp" not in params_dict and grid_cfg is not None:
        nvpar = int(getattr(grid_cfg, "nvpar", 32))
        vpar_max = float(getattr(grid_cfg, "vpar_max", 3.0))
        if nvpar > 0:
            params_dict["dvp"] = 2.0 * vpar_max / nvpar

    # derive kxmax/kymax from grid params (internal units: krho/kthnorm and kxrh).
    # Fallback is GKParams default 1.0.
    if grid_cfg is not None:
        nkx = int(getattr(grid_cfg, "nkx", 1))
        nky = int(getattr(grid_cfg, "nky", 1))
        krhomax = float(getattr(grid_cfg, "krhomax", 0.0))
        q = float(getattr(geometry_cfg, "q", 1.0))
        shat = float(getattr(geometry_cfg, "shat", 0.0))
        eps = float(getattr(geometry_cfg, "eps", 0.1))
        ikxspace = int(getattr(grid_cfg, "ikxspace", 5))
        geom_type = str(getattr(geometry_cfg, "geometry_model", "s-alpha"))
        if geom_type == "s-alpha":
            kthnorm = q / (2.0 * math.pi * eps)
        elif geom_type == "circ":
            # Lapillonne: kthnorm = 1/(2π(1+eps)) * sqrt(1 + (1-eps²)*(q/eps)²)
            kthnorm = (1.0 / (2.0 * math.pi * (1.0 + eps))) * math.sqrt(
                1.0 + (1.0 - eps**2) * (q / max(eps, 1e-12)) ** 2
            )
        else:
            kthnorm = 1.0
        if "kymax" not in params_dict and krhomax > 0 and nky > 1:
            params_dict["kymax"] = krhomax / kthnorm
        if (
            "kxmax" not in params_dict
            and nkx > 1
            and krhomax > 0
            and nky > 1
            and abs(shat) > 1e-12
            and abs(eps) > 1e-12
        ):
            ky_min_internal = krhomax / max(nky - 1, 1) / kthnorm
            kxspace = abs(q * shat * ky_min_internal / (eps * max(ikxspace, 1)))
            params_dict["kxmax"] = (nkx - 1) // 2 * kxspace

    if overrides:
        params_dict.update(overrides)
    return GKParams(**params_dict)
