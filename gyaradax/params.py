import jax
import jax.numpy as jnp

# enforce 64-bit precision
jax.config.update("jax_enable_x64", True)

import os
from omegaconf import OmegaConf
from dataclasses import dataclass
from typing import Dict, Any
from gyaradax.utils import load_scalars


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class GKParams:
    """
    Runtime controls and physical parameters for the gyrokinetic solver.

    This dataclass mirrors the GKW 'control', 'gridsize', and 'species' namelists,
    handling numerical hyperparameters and physical constants required for the
    gyrokinetic Vlasov-Poisson system (electrostatic and electromagnetic).

    Attributes:
        dt: Small time step for RK4 integration.
        naverage: Number of steps between mode normalization and growth rate calculation.
        disp_par: Parallel dissipation coefficient.
        disp_vp: Parallel velocity space dissipation coefficient.
        disp_x: Radial (kx) hyper-dissipation coefficient.
        disp_y: Binormal (ky) hyper-dissipation coefficient.
        idisp: Dissipation method selector.
        drive_scale: Scaling factor for equilibrium drive terms.
        norm_eps: Numerical floor for mode amplitude normalization.
        non_linear: Toggle for nonlinear term inclusion.
        rlt: Inverse temperature gradient scale length (R/LT).
        rln: Inverse density gradient scale length (R/LN).
        mas: Atomic mass of the kinetic species.
        tmp: Temperature of the kinetic species.
        de: Density of the kinetic species.
        signz: Charge sign of the species.
        vthrat: Thermal velocity ratio.
        shat: Magnetic shear parameter.
        q: Safety factor.
        eps: Local aspect ratio.
        kthnorm: Wavevector normalization factor.
        Rref: Reference major radius.
        d2X: Geometry-dependent scaling factor.
        signB: Direction of the magnetic field.
        dvp: Parallel velocity grid spacing.
        sgr_dist: Field-line grid spacing.
        kxmax: Maximum radial wavevector.
        kymax: Maximum binormal wavevector.
        dgrid: Global density scaling.
        tgrid: Global temperature scaling.
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
    finit: str = "cosine2"
    amp_init: float = 1.0e-4
    adiabatic_electrons: bool = True
    adaptive_dt: bool = False
    cfl_safety: float = 0.95
    mixed_precision: bool = True
    backend: str = "jax"
    use_z2z: bool = False

    # electromagnetic controls
    nlapar: bool = False  # enable A_parallel (shear Alfven)
    nlbpar: bool = False  # enable B_parallel (magnetic compression)
    beta: float = 0.0  # reference plasma beta = 2*mu0*n_ref*T_ref/B_ref^2

    # collision operator controls (GKW &collisions namelist)
    collisions: bool = False
    coll_pitch_angle: bool = True
    coll_en_scatter: bool = True
    coll_friction: bool = True
    coll_freq: float = 0.0
    coll_freq_override: bool = True
    coll_mass_conserve: bool = True
    # references for Coulomb log: Rref [m], Tref [keV], Nref [1e19 m^-3]
    coll_rref: float = 1.0
    coll_tref: float = 1.0
    coll_nref: float = 1.0
    # full species-pair collision backgrounds (incl. adiabatic species).
    # default None falls back to self-collision only.
    coll_bg_mas: Any = None
    coll_bg_signz: Any = None
    coll_bg_tmp: Any = None
    coll_bg_de: Any = None
    coll_bg_vthrat: Any = None
    # conservation corrections (Xu variant only; GKW default cons_type='Xu')
    coll_mom_conservation: bool = False
    coll_ene_conservation: bool = False

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

    # multi-GPU grid parallelism; sharding logic lives in gyaradax/sharding.py.
    # These fields only carry mesh shape and are static (part of trace signature).
    n_gpus_sp: int = 1
    n_gpus_vp: int = 1
    n_gpus_mu: int = 1

    # non-JAX-traceable fields (strings, control-flow booleans) — stored as
    # pytree auxiliary data rather than leaves.
    _STATIC_FIELDS = (
        "finit",
        "adiabatic_electrons",
        "non_linear",
        "adaptive_dt",
        "mixed_precision",
        "backend",
        "use_z2z",
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
        "drive_scale",
        "norm_eps",
        "cfl_safety",
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


def gkparams_from_runtime(runtime: Dict[str, Any], **overrides) -> GKParams:
    """Build GKParams from a GKW-compatible runtime-controls dictionary."""
    params_dict = {
        "dt": float(runtime.get("dtim", 0.01)),
        "naverage": int(runtime.get("naverage", 40)),
        "disp_par": float(runtime.get("disp_par", 1.0)),
        "disp_vp": float(runtime.get("disp_vp", 0.2)),
        "disp_x": float(runtime.get("disp_x", 0.1)),
        "disp_y": float(runtime.get("disp_y", 0.1)),
        "non_linear": bool(runtime.get("non_linear", False)),
        "finit": str(runtime.get("finit", "cosine2")),
        "amp_init": float(runtime.get("amp_init", 1.0e-4)),
        "adiabatic_electrons": bool(runtime.get("adiabatic_electrons", True)),
        "backend": str(runtime.get("backend", "jax")),
        "nlapar": bool(runtime.get("nlapar", False)),
        "nlbpar": bool(runtime.get("nlbpar", False)),
        "beta": float(runtime.get("beta", 0.0)),
        "collisions": bool(runtime.get("collisions", False)),
        "coll_pitch_angle": bool(runtime.get("coll_pitch_angle", True)),
        "coll_en_scatter": bool(runtime.get("coll_en_scatter", True)),
        "coll_friction": bool(runtime.get("coll_friction", True)),
        "coll_freq": float(runtime.get("coll_freq", 0.0)),
        "coll_freq_override": bool(runtime.get("coll_freq_override", True)),
        "coll_mass_conserve": bool(runtime.get("coll_mass_conserve", True)),
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


def gkparams_from_input_dat(input_dat_path: str, **overrides) -> GKParams:
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
    input_dat_path: str, geometry: Dict[str, Any], **overrides
) -> GKParams:
    """Build GKParams from input.dat + a pre-computed geometry dict.

    unlike gkparams_from_input_dat, this does not require geom.dat or
    other GKW output files — geometry scalars come from the dict.
    """
    import numpy as np
    from gyaradax.utils import load_runtime_params, parse_input_dat

    runtime = load_runtime_params(input_dat_path)
    inp = parse_input_dat(input_dat_path)

    scalars = {}
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
    num_sp = int(inp.get("gridsize", {}).get("number_of_species", 1))
    all_species_keys = [k for k in inp if k.startswith("species")]
    species_keys = all_species_keys[:num_sp]
    all_mas = (
        np.array([float(inp[k].get("mass", 1.0)) for k in all_species_keys])
        if all_species_keys
        else np.array([])
    )
    all_tmp = (
        np.array([float(inp[k].get("temp", 1.0)) for k in all_species_keys])
        if all_species_keys
        else np.array([])
    )
    all_de = (
        np.array([float(inp[k].get("dens", 1.0)) for k in all_species_keys])
        if all_species_keys
        else np.array([])
    )
    all_signz = (
        np.array([float(inp[k].get("z", 1.0)) for k in all_species_keys])
        if all_species_keys
        else np.array([])
    )
    all_vthrat = np.sqrt(all_tmp / all_mas) if len(all_mas) else np.array([])

    if species_keys:
        sp_mas = np.array([float(inp[k].get("mass", 1.0)) for k in species_keys])
        sp_tmp = np.array([float(inp[k].get("temp", 1.0)) for k in species_keys])
        sp_de = np.array([float(inp[k].get("dens", 1.0)) for k in species_keys])
        sp_signz = np.array([float(inp[k].get("z", 1.0)) for k in species_keys])
        sp_rlt = np.array([float(inp[k].get("rlt", 0.0)) for k in species_keys])
        sp_rln = np.array([float(inp[k].get("rln", 0.0)) for k in species_keys])
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


def gkparams_from_config(config: Any, **overrides) -> GKParams:
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
        "idisp": int(getattr(solver_cfg, "idisp", 2)),
        "non_linear": bool(getattr(solver_cfg, "non_linear", False)),
        "finit": str(getattr(solver_cfg, "finit", "cosine2")),
        "amp_init": float(getattr(solver_cfg, "amp_init", 1.0e-4)),
        "drive_scale": float(getattr(solver_cfg, "drive_scale", 1.0)),
        "adiabatic_electrons": bool(getattr(config.grid, "adiabatic_electrons", True)),
        "adaptive_dt": bool(getattr(solver_cfg, "adaptive_dt", False)),
        "cfl_safety": float(getattr(solver_cfg, "cfl_safety", 0.95)),
        "backend": str(getattr(solver_cfg, "backend", "jax")),
        "nlapar": bool(getattr(solver_cfg, "nlapar", False)),
        "nlbpar": bool(getattr(solver_cfg, "nlbpar", False)),
        "mixed_precision": bool(getattr(solver_cfg, "mixed_precision", True)),
        "beta": float(getattr(physics_cfg, "beta", 0.0)),
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
                "coll_freq_override": bool(getattr(coll_cfg, "freq_override", True)),
                "coll_mass_conserve": bool(getattr(coll_cfg, "mass_conserve", True)),
                "coll_mom_conservation": bool(getattr(coll_cfg, "mom_conservation", False)),
                "coll_ene_conservation": bool(getattr(coll_cfg, "ene_conservation", False)),
            }
        )
        if not params_dict["coll_freq_override"]:
            raise NotImplementedError("collisions: only freq_override=True is supported in MVP")

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
        import numpy as _np

        _tmp = _np.asarray(params_dict["tmp"], dtype=float)
        _mas = _np.asarray(params_dict["mas"], dtype=float)
        params_dict["vthrat"] = _np.sqrt(_tmp / _mas)

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
        import math

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
