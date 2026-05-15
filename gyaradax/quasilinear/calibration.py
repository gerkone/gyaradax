"""Calibration of the QL amplitude C_n.

The QL rule produces an un-normalized prediction `X = saturation_rule(linear outputs)`.
The amplitude `C_n` is calibrated against nonlinear flux `Y` from a training set.

Three calibration forms:

  * scalar (`fit_cn`):       Y ≈ C_n · X          one free parameter
  * parametric (`fit_cn_parametric`):
        Y ≈ X · (a + b·ŝ + c·q + d·R/L_T + e·R/L_n)
        five free parameters; mimics TGLF SAT1/2 geometry prefactors.
  * polynomial (`fit_cn_polynomial`):
        Y ≈ X · poly_degree_d(features)
        strict generalization of parametric (degree=1 reproduces it); adds
        cross terms and higher powers to absorb residual curvature.

All fits are linear least squares — no exponentials, no log-space surprises,
no extrapolation blowups on held-out / OOD samples.
"""

from dataclasses import dataclass
from itertools import combinations_with_replacement
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


def _poly_terms(n_features, degree):
    """Multi-index tuples for all monomials of degree 0 ... `degree`.

    Returns a list of tuples of feature indices. Empty tuple is the bias term;
    `(i,)` is feature i; `(i, j)` is feature i x feature j; etc. Uses
    combinations_with_replacement so each unique monomial appears once.
    """
    feats = list(range(n_features))
    terms = [()]
    for d in range(1, degree + 1):
        terms.extend(combinations_with_replacement(feats, d))
    return terms


def _term_label(term, feature_names):
    """Human-readable monomial label, e.g. (0, 0, 2) -> 'shat^2 * rlt_i'."""
    if not term:
        return "1"
    from collections import Counter
    c = Counter(term)
    parts = []
    for i, k in sorted(c.items()):
        parts.append(feature_names[i] if k == 1 else f"{feature_names[i]}^{k}")
    return " * ".join(parts)


@dataclass
class PolynomialCn:
    """Fitted polynomial C_n = poly_d(features). Apply via `.predict(X, F)`.

    Strict generalization of ParametricCn: degree=1 recovers the affine fit.
    Higher degree captures cross-terms (e.g. shat * rlt_i) and curvature in
    individual features. With more coefficients the fit is more flexible on
    train, more prone to OOD overshoot — track held-out RMSE.
    """

    coef: np.ndarray
    feature_names: Tuple[str, ...]
    degree: int

    def _terms(self):
        return _poly_terms(len(self.feature_names), self.degree)

    def cn(self, F):
        """Evaluate poly_d(F) for a feature matrix F (n, len(FEATURE_NAMES))."""
        idx = [FEATURE_NAMES.index(nm) for nm in self.feature_names]
        Fsub = np.asarray(F)[:, idx]
        terms = self._terms()
        out = np.zeros(Fsub.shape[0])
        for c, t in zip(self.coef, terms):
            prod = np.ones(Fsub.shape[0])
            for i in t:
                prod = prod * Fsub[:, i]
            out = out + float(c) * prod
        return out

    def predict(self, X, F):
        """Y_pred = X · poly_d(F)."""
        return np.asarray(X) * self.cn(F)

    def __repr__(self):
        terms = self._terms()
        n_show = min(len(terms), 8)
        parts = [
            f"{self.coef[i]:+.3e}·{_term_label(terms[i], self.feature_names)}"
            for i in range(n_show)
        ]
        suffix = f" + {len(terms) - n_show} more" if len(terms) > n_show else ""
        return f"PolynomialCn(degree={self.degree}, n_coef={len(terms)}: " + " ".join(parts) + suffix + ")"


def fit_cn_polynomial(X, Y, F, feature_names=DEFAULT_PARAM_FEATURES, degree=2):
    """Fit polynomial C_n by linear least squares.

    Args:
        X: (n,) QL prediction (un-normalized).
        Y: (n,) nonlinear target.
        F: (n, len(FEATURE_NAMES)) physics features in the order of FEATURE_NAMES.
        feature_names: subset of FEATURE_NAMES to use as polynomial variables.
        degree: max total degree of the polynomial. degree=1 is equivalent to
            fit_cn_parametric. degree=2 adds quadratic and bilinear cross-terms.

    Returns: PolynomialCn instance.
    """
    if degree < 1:
        raise ValueError(f"degree must be >= 1 (got {degree})")
    X = np.asarray(X)
    Y = np.asarray(Y)
    F = np.asarray(F)
    idx = [FEATURE_NAMES.index(nm) for nm in feature_names]
    Fsub = F[:, idx]
    terms = _poly_terms(len(feature_names), degree)
    cols = []
    for t in terms:
        prod = np.ones(Fsub.shape[0])
        for i in t:
            prod = prod * Fsub[:, i]
        cols.append(X * prod)
    A = np.stack(cols, axis=1)
    coef, *_ = np.linalg.lstsq(A, Y, rcond=None)
    return PolynomialCn(coef=coef, feature_names=tuple(feature_names), degree=degree)
