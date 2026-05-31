"""Field wrapper solves and electromagnetic variable transforms."""

from __future__ import annotations

import jax.numpy as jnp

from gyaradax.jax_config import enable_x64

from gyaradax.integrals import calculate_phi, calculate_phi_adiabatic


enable_x64()


def g_to_f(dg, apar, params, pre):
    """Convert mixed variable g to physical distribution f.

    g = f + (2Z/T) * vthrat * vpar * J0 * A_par * F_M
    => f = g - (2Z/T) * vthrat * vpar * J0 * A_par * F_M
    => f = g + g2f_factor * A_par  (g2f_factor is negative of the coupling)

    When nlapar=False, returns dg unchanged (identity).
    """
    if not params.nlapar:
        return dg
    g2f = pre["g2f_factor"]
    if dg.ndim == 5:
        apar_b = apar[jnp.newaxis, jnp.newaxis, :, :, :]
    else:
        apar_b = apar[jnp.newaxis, jnp.newaxis, jnp.newaxis, :, :, :]
    return dg + g2f * apar_b


def f_to_g(df, apar, params, pre):
    """Convert physical distribution f to mixed variable g.

    g = f + (2Z/T) * vthrat * vpar * J0 * A_par * F_M
    => g = f - g2f_factor * A_par  (g2f_factor = -(2Z/T)*vthrat*vpar*J0*F_M/T)

    When nlapar=False, returns df unchanged (identity).
    """
    if not params.nlapar:
        return df
    g2f = pre["g2f_factor"]
    if df.ndim == 5:
        apar_b = apar[jnp.newaxis, jnp.newaxis, :, :, :]
    else:
        apar_b = apar[jnp.newaxis, jnp.newaxis, jnp.newaxis, :, :, :]
    return df - g2f * apar_b


def _compute_phi(df, geometry, params, pre):
    """Compute phi via the appropriate solver.

    The argument is the distribution supplied to the quasineutrality solve.  In
    the current A_parallel-only Maxwellian cases, using mixed ``g`` or physical
    ``f = g_to_f(g, A_parallel)`` gives the same phi to roundoff: the g2f
    correction is odd in v_parallel while the phi weights are even.  B_parallel
    uses a separate coupled solve and is not covered by that A-only cancellation.
    """
    if params.adiabatic_electrons and "phi_weight" in pre and "phi_corr_weight" in pre:
        return calculate_phi_adiabatic(
            df,
            phi_weight=pre["phi_weight"],
            phi_corr_weight=pre["phi_corr_weight"],
            tmp=pre["phi_tmp"],
            de=pre["phi_de"],
            signz=pre["phi_signz"],
            gamma=pre["phi_gamma"],
            ints=pre["phi_ints"],
            has_zonal=pre["phi_has_zonal"],
            ixzero=pre["phi_ixzero"],
            iyzero=pre["phi_iyzero"],
        )
    else:
        return calculate_phi(geometry, df, params=params, pre=pre)


def _compute_fields(dg, geometry, params, pre):
    """Compute all field variables (phi, apar, bpar) from evolved mixed ``dg``.

    When nlapar=True, Ampere's law is solved directly from mixed ``g``/``dg``;
    the g2f contribution belongs in the Ampere diagonal, not in the source.
    The code then forms physical ``f`` for the phi/RHS path.  GKW's field
    routine solves the coupled phi/B_parallel system from mixed ``g``.  For the
    currently supported kinetic Maxwellian cases on symmetric v_parallel grids,
    this ``f`` path is equivalent to the GKW ordering for phi/B_parallel: the
    g2f correction is odd in v_parallel while the phi and B_parallel weights
    are even, so its contribution cancels to roundoff.  This parity argument is
    not guaranteed for asymmetric grids or future non-Maxwellian/species models.
    """
    if params.nlbpar and not params.nlapar:
        raise NotImplementedError(
            "B_parallel without A_parallel (nlbpar=True, nlapar=False) is not supported yet"
        )
    if not params.nlapar:
        phi = _compute_phi(dg, geometry, params, pre)
        return phi, None, None

    # adiabatic + nlapar (GKW em_adiabat_apar): promote 5D dg to 6D so the
    # (nsp-indexed) apar/bpar einsums work uniformly, restore to 5D for phi.
    adiabatic_5d = dg.ndim == 5
    dg_6d = dg[jnp.newaxis] if adiabatic_5d else dg

    apar_weight = pre["apar_weight"]
    apar_diag = pre["apar_diag"]
    apar_num = jnp.einsum("avmjkl,avmjkl->jkl", apar_weight, dg_6d)
    apar = apar_num / apar_diag

    df_6d = g_to_f(dg_6d, apar, params, pre)
    df = df_6d[0] if adiabatic_5d else df_6d
    phi = _compute_phi(df, geometry, params, pre)

    bpar = None
    if params.nlbpar and "bpar_weight" in pre:
        bpar_num = jnp.einsum("avmjkl,avmjkl->jkl", pre["bpar_weight"], df_6d)
        bpar = -bpar_num / pre["phi_diag"]
    return phi, apar, bpar
