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

    # fields that are not JAX-traceable (strings, booleans used for control flow)
    # and must be stored as pytree auxiliary data rather than leaves.
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
            # arrays must be leaves (not hashable for aux)
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
    }
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
    """
    Load all runtime, physics, and geometry scalars from a GKW run directory.

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

    # geometry scalars from the computed geometry
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

    # species from input.dat
    num_sp = int(inp.get("gridsize", {}).get("number_of_species", 1))
    species_keys = [k for k in inp if k.startswith("species")][:num_sp]
    if species_keys:
        sp_mas = np.array([float(inp[k].get("mass", 1.0)) for k in species_keys])
        sp_tmp = np.array([float(inp[k].get("temp", 1.0)) for k in species_keys])
        sp_de = np.array([float(inp[k].get("dens", 1.0)) for k in species_keys])
        sp_signz = np.array([float(inp[k].get("z", 1.0)) for k in species_keys])
        sp_rlt = np.array([float(inp[k].get("rlt", 0.0)) for k in species_keys])
        sp_rln = np.array([float(inp[k].get("rln", 0.0)) for k in species_keys])
        sp_vthrat = np.sqrt(sp_tmp / sp_mas)

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
        "adiabatic_electrons": bool(getattr(config.grid, "adiabatic_electrons", True)),
        "adaptive_dt": bool(getattr(solver_cfg, "adaptive_dt", False)),
        "cfl_safety": float(getattr(solver_cfg, "cfl_safety", 0.95)),
        "backend": str(getattr(solver_cfg, "backend", "jax")),
        "nlapar": bool(getattr(solver_cfg, "nlapar", False)),
        "nlbpar": bool(getattr(solver_cfg, "nlbpar", False)),
        "beta": float(getattr(physics_cfg, "beta", 0.0)),
    }

    # physics scalars (may be arrays for multi-species kinetic configs)
    _SPECIES_PARAMS = {"rlt", "rln", "mas", "tmp", "de", "signz", "vthrat"}
    for k in ["rlt", "rln", "mas", "tmp", "de", "signz", "vthrat", "dgrid", "tgrid"]:
        if hasattr(physics_cfg, k):
            v = getattr(physics_cfg, k)
            if k in _SPECIES_PARAMS and hasattr(v, "__iter__") and not isinstance(v, str):
                params_dict[k] = jnp.array([float(x) for x in v])
            else:
                params_dict[k] = float(v)

    # geometry scalars
    for k in ["shat", "q", "eps", "kthnorm", "Rref", "d2X", "signB"]:
        if hasattr(geometry_cfg, k):
            params_dict[k] = float(getattr(geometry_cfg, k))

    # scaling/grid scalars
    for k in ["dvp", "sgr_dist", "kxmax", "kymax"]:
        if hasattr(geometry_cfg, k):
            params_dict[k] = float(getattr(geometry_cfg, k))

    if overrides:
        params_dict.update(overrides)
    return GKParams(**params_dict)
