import jax
import jax.numpy as jnp

# enforce 64-bit precision
jax.config.update("jax_enable_x64", True)

import os
from omegaconf import OmegaConf
from dataclasses import dataclass
from typing import Dict, Any


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class GKParams:
    """
    Runtime controls and physical parameters for the electrostatic solver.

    This dataclass mirrors the GKW 'control', 'gridsize', and 'species' namelists,
    handling numerical hyperparameters and physical constants required for the
    gyrokinetic Vlasov-Poisson system.

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
    adiabatic_electrons: bool = True

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

    def tree_flatten(self):
        return tuple(vars(self).values()), None

    @classmethod
    def tree_unflatten(cls, aux_data, leaves):
        return cls(*leaves)


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
        "adiabatic_electrons": bool(runtime.get("adiabatic_electrons", True)),
    }
    # fill physical and geometry params if available
    for k in [
        "rlt",
        "rln",
        "mas",
        "tmp",
        "de",
        "signz",
        "vthrat",
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
    from gyaradax.geometry import load_scalars

    directory = os.path.dirname(input_dat_path)
    scalars = load_scalars(directory)
    return gkparams_from_runtime(scalars, **overrides)


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
        "adiabatic_electrons": bool(getattr(config.grid, "adiabatic_electrons", True)),
    }

    # physics scalars
    for k in ["rlt", "rln", "mas", "tmp", "de", "signz", "vthrat", "dgrid", "tgrid"]:
        if hasattr(physics_cfg, k):
            params_dict[k] = float(getattr(physics_cfg, k))

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
