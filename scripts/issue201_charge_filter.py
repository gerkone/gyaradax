"""FP64 JAX prototype of a charge correction for parallel dissipation on issue 201.

Run a linear two-field kinetic checkpoint and print the rate of its nonzonal
potential. The correction uses the solver's connected field-line stencils.
"""
import argparse
import csv
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from omegaconf import OmegaConf

from gyaradax.backends import create_ops
from gyaradax.core.jax_config import enable_x64
from gyaradax.core.params import gkparams_from_config
from gyaradax.io.gkw import load_geometry
from gyaradax.operators.fields import g_to_f
from gyaradax.operators.precompute import linear_precompute
from gyaradax.operators.stencils import D4_IPW_POS
from gyaradax.solver.driver import _ensure_species_arrays


def make_charge_correction(geometry, params, pre):
    """Remove charge created from a neutral perturbation by the centered damping rows."""
    if params.adiabatic_electrons or not params.nlapar or params.nlbpar:
        raise ValueError("This prototype requires kinetic species with phi and apar")
    if params.disp_par_conserve:
        raise ValueError("The prototype requires native parallel dissipation as its input")

    classes = np.asarray(pre["par_stencil_class"])
    _, nkx, nky = classes.shape
    nonzonal = np.asarray(geometry["krho"]).reshape(1, 1, nky) != 0
    valid = np.asarray(pre["valid_shift"])[2:7].all(axis=0)
    bulk = ((classes == 2) & nonzonal & valid).ravel()
    ky_index = np.arange(nky)[None, None, :]
    centers = np.flatnonzero(bulk)
    spatial_index = np.asarray(pre["s_shift"])[2:7] * nkx + np.asarray(pre["kx_shift"])[2:7]
    cells = (spatial_index * nky + ky_index).reshape(5, -1)[:, centers]
    # The two velocity signs share the same centered fourth-difference row.
    coeff = D4_IPW_POS[2, 2:7] / 12
    weight = pre["phi_weight"]
    species_factor = (np.asarray(params.signz) / np.asarray(params.tmp)).reshape(-1, 1, 1, 1, 1, 1)
    response = species_factor * pre["fmaxwl"] * pre["bessel"]
    axes = (0, 1, 2)
    charge_norm = jnp.sum(weight * response, axis=axes, keepdims=True)
    active = np.broadcast_to(nonzonal, classes.shape)
    if np.any(np.asarray(charge_norm).reshape(classes.shape)[active] <= 0):
        raise ValueError("The local charge response must be positive on each active row")
    response = response / jnp.where(active, charge_norm, 1.)
    speed = params.disp_par * pre["abs_dum2_par"] / pre["sgr_dist"]
    weights = jnp.broadcast_to(weight * speed, response.shape)
    center_weights = weights.reshape(*response.shape[:3], -1)[..., centers]

    def correction(f):
        neutral = f - response * jnp.sum(weight * f, axis=axes, keepdims=True)
        neutral = neutral.reshape(*neutral.shape[:3], -1)
        fourth = sum(c * neutral[..., neighbors] for c, neighbors in zip(coeff, cells))
        charge = jnp.sum(center_weights * fourth, axis=axes)
        removed = jnp.zeros(classes.size, dtype=f.dtype).at[centers].set(charge)
        return -response * removed.reshape(classes.shape)

    return correction


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--dt", type=float, required=True)
    parser.add_argument("--steps", type=int, nargs="+", required=True)
    parser.add_argument("--grid", type=Path, help="Use a separately exported GKW geometry")
    parser.add_argument("--native", action="store_true", help="Run the unmodified control")
    args = parser.parse_args()
    if args.dt <= 0 or any(n < 0 for n in args.steps):
        parser.error("dt must be positive and output steps must be nonnegative")
    enable_x64()
    cfg = OmegaConf.load(args.config)
    params = gkparams_from_config(cfg, backend="jax", mixed_precision=False)
    if params.non_linear:
        parser.error("The checkpoint driver evolves a linear right-hand side")
    geo = _ensure_species_arrays(load_geometry(str(args.grid or cfg.run.data_dir)), params)
    pre = linear_precompute(geo, params)
    ops = create_ops(pre, backend="jax", mixed_precision=False)
    correction = None if args.native else make_charge_correction(geo, params, pre)
    with np.load(args.checkpoint) as checkpoint:
        g = jnp.asarray(checkpoint["df"], dtype=jnp.complex128)
    active = np.asarray(geo["krho"]).ravel() != 0
    g = g * active

    def fields(x):
        return ops.compute_fields(x, geo, params, pre)[:2]

    def rhs(x):
        phi, apar = fields(x)
        rate = ops.linear_rhs_from_g(x, phi, geo, params, pre, apar=apar)
        if correction is None:
            return rate
        return rate + correction(g_to_f(x, apar, params, pre))

    @jax.jit
    def advance(x, count):
        dt = args.dt

        def step(_, y):
            k1 = rhs(y)
            k2 = rhs(y + dt * k1 / 2)
            k3 = rhs(y + dt * k2 / 2)
            k4 = rhs(y + dt * k3)
            return y + dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6
        return jax.lax.fori_loop(0, count, step, x)

    @jax.jit
    def measure(x):
        phi = fields(x)[0][..., active].ravel()
        rate = fields(rhs(x))[0][..., active].ravel()
        lam = jnp.vdot(phi, rate) / jnp.vdot(phi, phi)
        residual = jnp.linalg.norm(rate - lam * phi) / jnp.linalg.norm(rate)
        return jnp.array([lam.real, lam.imag, residual, jnp.linalg.norm(phi)])

    writer = csv.writer(sys.stdout)
    writer.writerow(["time", "gamma", "omega", "relative_phi_residual", "phi_norm"])
    prev = 0
    for step in sorted({0, *args.steps}):
        if step > prev:
            g = advance(g, step - prev)
        result = np.asarray(measure(g))
        if not np.all(np.isfinite(result)):
            raise FloatingPointError(f"Nonfinite field measurement at step {step}")
        writer.writerow([step * args.dt, *result.tolist()])
        sys.stdout.flush()
        prev = step


if __name__ == "__main__":
    main()
