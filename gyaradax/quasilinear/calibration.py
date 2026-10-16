"""Cn calibration head classes.

Only the containers the bundled heads were pickled as. The fitting that
produced them is gone: calibration now runs off the summary parquets, not a
walk over paired run directories. Kept so the shipped Cn payloads still
unpickle.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np


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
