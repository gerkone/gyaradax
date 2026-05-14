"""Calibration of the QL amplitude C_n.

The QL rule produces an un-normalized prediction `X = saturation_rule(linear outputs)`.
The amplitude `C_n` is calibrated against nonlinear flux `Y` from a training set.

Two calibration forms:

  * scalar (`fit_cn`):       Y ≈ C_n · X          one free parameter
  * parametric (`fit_cn_parametric`):
        Y ≈ X · (a + b·ŝ + c·q + d·R/L_T + e·R/L_n)
        five free parameters; mimics TGLF SAT1/2 geometry prefactors.

Both fits are linear least squares — no exponentials, no log-space surprises,
no extrapolation blowups on held-out / OOD samples.
"""

from dataclasses import dataclass
from typing import Tuple

import jax
import jax.numpy as jnp
import numpy as np

from .data import FEATURE_NAMES


@jax.jit
def fit_cn(X, Y):
    """Scalar OLS: Y = C_n · X. Returns scalar C_n."""
    return jnp.sum(X * Y) / jnp.sum(X * X)


@jax.jit
def fit_cn_log(X, Y, eps=1e-12):
    """Log-space scalar fit: log(C_n) = mean(log Y − log X)."""
    X_safe = jnp.maximum(X, eps)
    Y_safe = jnp.maximum(Y, eps)
    return jnp.exp(jnp.mean(jnp.log(Y_safe) - jnp.log(X_safe)))


@jax.jit
def r2_score(y_true, y_pred):
    ss_res = jnp.sum((y_true - y_pred) ** 2)
    ss_tot = jnp.sum((y_true - jnp.mean(y_true)) ** 2)
    return 1.0 - ss_res / ss_tot


# itg drives + geometry; names index into data.FEATURE_NAMES
DEFAULT_PARAM_FEATURES = ("shat", "q", "rlt_i", "rln_i")


@dataclass
class ParametricCn:
    """Fitted parametric C_n = a + Σ_i β_i · feature_i.

    Apply via `.predict(X, F)` → returns X · C_n(F).
    """

    coef: np.ndarray
    feature_names: Tuple[str, ...]

    def cn(self, F):
        """Compute C_n(features) for a feature matrix F (n, len(FEATURE_NAMES))."""
        idx = [FEATURE_NAMES.index(nm) for nm in self.feature_names]
        out = float(self.coef[0]) * np.ones(F.shape[0])
        for i, ii in enumerate(idx):
            out = out + float(self.coef[i + 1]) * np.asarray(F[:, ii])
        return out

    def predict(self, X, F):
        """Y_pred = X · C_n(F)."""
        return np.asarray(X) * self.cn(F)

    def __repr__(self):
        s = f"ParametricCn(C_n = {self.coef[0]:+.4f}"
        for nm, c in zip(self.feature_names, self.coef[1:]):
            s += f" {'+' if c >= 0 else ''}{c:+.4f}·{nm}"
        return s + ")"


def fit_cn_parametric(X, Y, F, feature_names=DEFAULT_PARAM_FEATURES):
    """Fit parametric C_n by linear least squares.

    Args:
        X: (n,) QL prediction (un-normalized).
        Y: (n,) nonlinear target.
        F: (n, len(FEATURE_NAMES)) physics features in the order of FEATURE_NAMES.
        feature_names: subset of FEATURE_NAMES to use in the linear C_n.

    Returns: ParametricCn instance.
    """
    X = np.asarray(X)
    Y = np.asarray(Y)
    F = np.asarray(F)
    idx = [FEATURE_NAMES.index(nm) for nm in feature_names]
    cols = [X] + [X * F[:, i] for i in idx]
    A = np.stack(cols, axis=1)
    coef, *_ = np.linalg.lstsq(A, Y, rcond=None)
    return ParametricCn(coef=coef, feature_names=tuple(feature_names))
