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
    `.cn_jax(F)` evaluates in JAX so the head can be used inside a jit'd path
    (e.g. the TORAX plug-in's per-radius call).
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

    def cn_jax(self, F):
        """JAX-native C_n(F). F is (n, len(FEATURE_NAMES))."""
        idx = [FEATURE_NAMES.index(nm) for nm in self.feature_names]
        Fsub = jnp.asarray(F)
        out = jnp.full((Fsub.shape[0],), float(self.coef[0]))
        for i, ii in enumerate(idx):
            out = out + float(self.coef[i + 1]) * Fsub[:, ii]
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

    `log_space=True` means the polynomial fits log(C_n), so C_n = exp(poly(F))
    and the head is strictly positive. Useful when Y spans many decades.
    """

    coef: np.ndarray
    feature_names: Tuple[str, ...]
    degree: int
    log_space: bool = False

    def _terms(self):
        return _poly_terms(len(self.feature_names), self.degree)

    def cn(self, F):
        """Evaluate C_n(F) = poly_d(F) (or exp(poly_d(F)) if log_space)."""
        idx = [FEATURE_NAMES.index(nm) for nm in self.feature_names]
        Fsub = np.asarray(F)[:, idx]
        terms = self._terms()
        out = np.zeros(Fsub.shape[0])
        for c, t in zip(self.coef, terms):
            prod = np.ones(Fsub.shape[0])
            for i in t:
                prod = prod * Fsub[:, i]
            out = out + float(c) * prod
        return np.exp(out) if self.log_space else out

    def cn_jax(self, F):
        """JAX-native C_n(F). Polynomial-term loop is unrolled at trace time."""
        idx = [FEATURE_NAMES.index(nm) for nm in self.feature_names]
        Fsub = jnp.asarray(F)[:, idx]
        terms = self._terms()
        out = jnp.zeros(Fsub.shape[0])
        for c, t in zip(self.coef, terms):
            prod = jnp.ones(Fsub.shape[0])
            for i in t:
                prod = prod * Fsub[:, i]
            out = out + float(c) * prod
        return jnp.exp(out) if self.log_space else out

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
        tag = "log " if self.log_space else ""
        return f"PolynomialCn({tag}degree={self.degree}, n_coef={len(terms)}: " + " ".join(parts) + suffix + ")"


def _poly_design(F, feature_names, degree):
    idx = [FEATURE_NAMES.index(nm) for nm in feature_names]
    Fsub = np.asarray(F)[:, idx]
    terms = _poly_terms(len(feature_names), degree)
    cols = []
    for t in terms:
        prod = np.ones(Fsub.shape[0])
        for i in t:
            prod = prod * Fsub[:, i]
        cols.append(prod)
    return np.stack(cols, axis=1), terms


def fit_cn_polynomial(X, Y, F, feature_names=DEFAULT_PARAM_FEATURES, degree=2):
    """Fit polynomial C_n by linear least squares: Y ≈ X · poly_d(F).

    degree=1 is equivalent to fit_cn_parametric. degree=2 adds quadratic
    and bilinear cross-terms (15 coefficients on 4 features — overfits fast
    on small training sets).
    """
    if degree < 1:
        raise ValueError(f"degree must be >= 1 (got {degree})")
    X = np.asarray(X); Y = np.asarray(Y)
    Phi, _ = _poly_design(F, feature_names, degree)
    A = np.asarray(X)[:, None] * Phi
    coef, *_ = np.linalg.lstsq(A, Y, rcond=None)
    return PolynomialCn(coef=coef, feature_names=tuple(feature_names),
                        degree=degree, log_space=False)


def fit_cn_polynomial_log(X, Y, F, feature_names=DEFAULT_PARAM_FEATURES,
                          degree=1, eps=1e-12):
    """Fit log-space polynomial: log(Y/X) ≈ poly_d(F) → C_n = exp(poly_d(F)).

    Robust to the wide dynamic range of Y typical in QL-vs-NL flux datasets
    (Y can span 10+ decades). Always strictly positive C_n. degree=1 is the
    log-space analogue of the parametric fit; bigger degrees overfit just as
    fast as the linear-space version on small data.
    """
    if degree < 1:
        raise ValueError(f"degree must be >= 1 (got {degree})")
    X = np.maximum(np.asarray(X), eps)
    Y = np.maximum(np.asarray(Y), eps)
    Phi, _ = _poly_design(F, feature_names, degree)
    rhs = np.log(Y) - np.log(X)
    coef, *_ = np.linalg.lstsq(Phi, rhs, rcond=None)
    return PolynomialCn(coef=coef, feature_names=tuple(feature_names),
                        degree=degree, log_space=True)


def fit_cn_heads(X, Y, F, *, degree=1, test_frac=0.2, seed=0, min_flux=None):
    """Fit the full set of Cn heads on a (X_QL, Y_NL, F) dataset.

    Convenience wrapper that produces the same payload dict the torax
    gyaradax-ql plugin loads via `cn_calibration_path`: a scalar (basic ql),
    a parametric and a polynomial head (cn version), a log-space polynomial,
    plus train/test R2. Pickle the result and point `cn_calibration_path` at
    it to use a custom calibration.

    Args:
      X: QL flux per sample at cn=1, shape (n,).
      Y: nonlinear / target flux per sample, shape (n,).
      F: features per sample, shape (n, n_features) in `FEATURE_NAMES` order.
      degree: polynomial degree for the polynomial heads (1 = affine).
      test_frac: held-out fraction for the reported R2.
      seed: split RNG seed.
      min_flux: if set, drop samples with Y < min_flux (unsaturated / noise).

    Returns:
      dict with keys scalar, parametric, polynomial, polynomial_log, r2_test,
      r2_train, n_train, n_test.
    """
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)
    F = np.asarray(F, dtype=float)
    keep = np.isfinite(X) & np.isfinite(Y) & (X > 0) & (Y > 0)
    if min_flux is not None:
        keep &= Y >= float(min_flux)
    X, Y, F = X[keep], Y[keep], F[keep]
    if len(X) < 4:
        raise ValueError(f"need >= 4 usable samples, got {len(X)}")

    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(X))
    n_test = max(1, int(test_frac * len(X)))
    te, tr = idx[:n_test], idx[n_test:]
    Xtr, Ytr, Ftr = X[tr], Y[tr], F[tr]
    Xte, Yte, Fte = X[te], Y[te], F[te]

    cn_scalar = float(fit_cn(jnp.asarray(Xtr), jnp.asarray(Ytr)))
    par = fit_cn_parametric(Xtr, Ytr, Ftr)
    poly = fit_cn_polynomial(Xtr, Ytr, Ftr, degree=degree)
    poly_log = fit_cn_polynomial_log(Xtr, Ytr, Ftr, degree=degree)

    def _r2(yt, yp):
        return float(r2_score(jnp.asarray(yt), jnp.asarray(yp)))

    return dict(
        scalar=cn_scalar,
        parametric=par,
        polynomial=poly,
        polynomial_log=poly_log,
        r2_test=dict(
            scalar=_r2(Yte, cn_scalar * Xte),
            parametric=_r2(Yte, par.predict(Xte, Fte)),
            polynomial=_r2(Yte, poly.predict(Xte, Fte)),
            polynomial_log=_r2(Yte, poly_log.predict(Xte, Fte)),
        ),
        r2_train=dict(
            scalar=_r2(Ytr, cn_scalar * Xtr),
            parametric=_r2(Ytr, par.predict(Xtr, Ftr)),
            polynomial=_r2(Ytr, poly.predict(Xtr, Ftr)),
            polynomial_log=_r2(Ytr, poly_log.predict(Xtr, Ftr)),
        ),
        n_train=int(len(tr)),
        n_test=int(len(te)),
    )
