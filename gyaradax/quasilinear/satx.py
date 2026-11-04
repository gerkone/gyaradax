"""satx_mlp: SAT2's intensity with coefficients set by a small MLP of the operating point.

The saturated intensity per ky is SAT2's (TGLF kx width, zonal-mixing growth rate),
raised to an exponent ``p``. Electron modes (``omega < 0``) get a separate level
``exp(le)``. Each flux channel then has its own amplitude ``exp(lc)``:

    Q_ch = exp(lc_ch) sum_ky W_ch(ky) I(ky)^p (f_ion + (1 - f_ion) exp(le))

with ``f_ion = sigmoid(omega / omega_width)``. ``lc``, ``p`` and ``le`` are per
electron model, shifted by an MLP of the operating parameters and a few linear
scalars; the prediction is the mean over an ensemble of such MLPs. Every number
of the rule (fitted weights, feature scalings, the input ranges) comes from its
calibration record.

Operating parameters pass unchanged inside the training range. Between the training
range and the wider input box they are squashed smoothly (tanh, slope 1 at the training
edge) so that the rule extrapolates a little and then saturates at the box; ``te_ti``,
``shat`` and ``q`` are squashed in log space.
"""

import jax
import jax.numpy as jnp

from . import rules as R

CHANNELS = ("Qi", "Qe", "Gamma")
# operating parameters, in the order of the record's input box
OPERATING = ("rlt_i", "rlt_e", "rln", "te_ti", "shat", "q", "eps", "nu", "kinetic")


LOG_INPUTS = ("te_ti", "shat", "q")


def soft_box(x, train_range, input_box):
    """Identity inside ``train_range``, tanh-saturating to ``input_box`` outside it, per parameter."""
    log = jnp.asarray([k in LOG_INPUTS for k in OPERATING])
    tr = jnp.asarray(train_range, dtype=jnp.float64)
    bx = jnp.asarray(input_box, dtype=jnp.float64)

    def to(v):
        return jnp.where(log, jnp.log(jnp.maximum(v, 1e-12)), v)

    v, a, b = to(x), to(tr[:, 0]), to(tr[:, 1])
    lo, hi = to(bx[:, 0]), to(bx[:, 1])
    up, dn = jnp.maximum(hi - b, 1e-12), jnp.maximum(a - lo, 1e-12)
    v = jnp.where(v > b, b + up * jnp.tanh((v - b) / up), v)
    v = jnp.where(v < a, a - dn * jnp.tanh((a - v) / dn), v)
    return jnp.where(log, jnp.exp(v), v)


def sat2_intensity(growth_rate, ky, dlnpdr, grad_r0, gq, gate_sharpness, eps):
    """SAT2's per-ky intensity with the TGLF kx width, and the zonal-mixing peak kymax."""
    ghat = R._gated_gamma(growth_rate, 0.0, gate_sharpness)
    kymax, vzf, jmax = R._zonal_mixing(ky, ghat, eps, kymin=R._SAT2_KYMIN * grad_r0)
    ge = R._gamma_kymix(ky, ghat, kymax, vzf, jmax, 1.0, eps)
    kycut = R._SAT2_B0 * kymax
    kxw = jnp.where(ky < kycut, kycut / grad_r0, kycut / grad_r0 + R._SAT2_B1 * (ky - kycut) * gq)
    cnorm = 12.0 / jnp.maximum(dlnpdr, R._DLNPDR_FLOOR)
    inten = (1.0 / kymax) * cnorm * (ge / jnp.maximum(kxw * ky, eps)) ** 2
    return jnp.where(ky > eps, inten, 0.0), kymax


def features(operating, growth_rate, omega, ky, kymax, scale):
    """MLP inputs: the scaled operating parameters, then five linear scalars.

    ``operating`` holds :data:`OPERATING`, already clipped into the input box.
    """
    op = dict(zip(OPERATING, operating))
    kin = op["kinetic"]
    s = scale
    x_op = jnp.stack([
        op["rlt_i"] / s["rlt"] - 1.0,
        op["rlt_e"] / s["rlt"] - kin,
        op["rln"] / s["rln"] - 1.0,
        jnp.log(op["te_ti"]),
        jnp.log(op["shat"]),
        jnp.log(op["q"] / s["q"]),
        (op["eps"] - s["eps_center"]) / s["eps_width"],
        op["nu"] / s["nu"],
        kin,
    ])
    gp = jnp.maximum(growth_rate, 0.0)
    ion = jax.nn.sigmoid(omega / s["omega_width"])

    def log_rate(g):
        return jnp.log(g + s["gamma_floor"]) / 2.0 + 1.0

    x_lin = jnp.stack([
        log_rate(jnp.max(gp)),
        2.0 * jnp.sum(gp * ion) / jnp.maximum(jnp.sum(gp), s["gamma_sum_floor"]) - 1.0,
        jnp.log(kymax / s["kymax"]),
        log_rate(jnp.max(gp * ion)),
        log_rate(jnp.max(gp * (1.0 - ion))),
    ])
    return jnp.concatenate([x_op, x_lin]), ion


def _member(theta, x, kin, intensity, ion, weights):
    theta = {k: jnp.asarray(v, dtype=jnp.float64) for k, v in theta.items()}
    h = jnp.tanh(x @ theta["w1"] + theta["b1"])
    d = h @ theta["w2"] + theta["b2"]

    def per_model(v):
        return v[0] * (1.0 - kin) + v[1] * kin

    lc = jnp.stack([per_model(theta["lc_Qi"]) + d[0], per_model(theta["lc_Qe"]) + d[1],
                    per_model(theta["lc_Gamma"]) + d[1]])
    p = per_model(theta["p"]) + theta["p_range"] * jnp.tanh(d[2])
    le = theta["le"] + d[3]
    shaped = jnp.where(intensity > 0.0, jnp.exp(p * jnp.log(jnp.maximum(intensity, 1e-300))), 0.0)
    shaped = shaped * (ion + (1.0 - ion) * jnp.exp(le))
    return jnp.exp(lc) * jnp.sum(weights * shaped[None, :], axis=1)


def ql_flux_satx_mlp(
    growth_rate,
    omega,
    weights,
    krho,
    operating,
    dlnpdr,
    kthnorm,
    grad_r0,
    gq,
    members,
    scale,
    train_range,
    input_box,
    cn=1.0,
    gate_sharpness=20.0,
    eps=1e-30,
):
    """``(Qi, Qe, Gamma)`` of the satx_mlp rule at one point.

    Args:
        growth_rate: Converged growth rate per ky, (nky,).
        omega: Real frequency per ky, GKW sign (ion modes positive), (nky,).
        weights: QL weights ``W(ky)`` of Qi, Qe and Gamma, (3, nky); the zonal entry is ignored.
        krho: Binormal wavenumbers, (nky,), times ``kthnorm`` gives ``k_theta rho``.
        operating: Operating parameters in the order of :data:`OPERATING`, (9,).
        dlnpdr: SAT2's pressure-gradient length (:func:`~gyaradax.quasilinear.rules.dlnpdr_tglf`).
        kthnorm: Converts ``krho`` to ``k_theta rho``.
        grad_r0: SAT2's ``grad_r0``.
        gq: SAT2's ``Gq``.
        members: Fitted parameters of the ensemble, one dict per MLP.
        scale: Feature scalings of the record.
        train_range: ``[lo, hi]`` per operating parameter of the training rows.
        input_box: ``[lo, hi]`` per operating parameter the inputs saturate at (:func:`soft_box`).
        cn: Overall amplitude.
        gate_sharpness: Sharpness of the stability gate on the growth rate.
        eps: Floor of the denominators.

    Returns:
        (3,) array: Qi, Qe, Gamma in GKW gyro-Bohm units.
    """
    op = soft_box(jnp.asarray(operating, dtype=jnp.float64), train_range, input_box)
    ky = jnp.abs(krho) * kthnorm
    g = jnp.where(jnp.isfinite(growth_rate), growth_rate, 0.0)
    intensity, kymax = sat2_intensity(g, ky, dlnpdr, grad_r0, gq, gate_sharpness, eps)
    x, ion = features(op, g, omega, ky, kymax, scale)
    w = jnp.where((ky > eps)[None, :], jnp.nan_to_num(jnp.asarray(weights)), 0.0)
    kin = op[OPERATING.index("kinetic")]
    out = jnp.stack([_member({**m, "p_range": scale["p_range"]}, x, kin, intensity, ion, w)
                     for m in members])
    return cn * jnp.mean(out, axis=0)
