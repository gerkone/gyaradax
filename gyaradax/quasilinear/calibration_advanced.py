"""Advanced C_n calibration methods for the QL model.

All fitters share the `.predict(X, F)` interface of `ParametricCn` and
`PolynomialCn` in calibration.py — they slot into the notebook eval loop
identically.

  * `fit_cn_ridge_polynomial` — RidgeCV on Y = X · poly_d(F). Cross-validated
    L2; strict generalization of `fit_cn_polynomial` (the alpha=0 case).
    Helps when degree x features pushes the parameter count above ~N/10.
  * `fit_cn_gbm`             — gradient-boosted trees on log(Y/X). Uses
    sklearn's HistGradientBoostingRegressor; no extra deps. Positive C_n
    by construction (exp transform). Strong tabular baseline.
  * `fit_cn_gp_log`          — Gaussian process on log(Y/X) with ARD
    Matern kernel on standardized features. Positive C_n + uncertainty
    estimates; well-calibrated on small N.

Note on positivity: GBM and GP both target log(Y/X), so predictions are
positive by construction. The ridge polynomial fits in linear space (to
match the existing fit_cn_polynomial baseline) and can in principle
produce negative C_n at OOD points — check the eval if that matters.
"""

from dataclasses import dataclass
from typing import Tuple

import numpy as np

from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import PolynomialFeatures

from .calibration import DEFAULT_PARAM_FEATURES
from .data import FEATURE_NAMES


def _feature_subset(F, feature_names):
    idx = [FEATURE_NAMES.index(nm) for nm in feature_names]
    return np.asarray(F)[:, idx]


# ─── (1) Ridge polynomial ─────────────────────────────────────────────────


@dataclass
class RidgePolynomialCn:
    """Ridge-regularized polynomial C_n. Fit by `fit_cn_ridge_polynomial`."""

    feature_names: Tuple[str, ...]
    degree: int
    alpha: float
    coef: np.ndarray
    poly: object  # sklearn PolynomialFeatures

    def cn(self, F):
        Fsub = _feature_subset(F, self.feature_names)
        Phi = self.poly.transform(Fsub)
        return Phi @ self.coef

    def predict(self, X, F):
        return np.asarray(X) * self.cn(F)

    def __repr__(self):
        return (
            f"RidgePolynomialCn(degree={self.degree}, "
            f"alpha={self.alpha:.3e}, n_coef={self.coef.size})"
        )


def fit_cn_ridge_polynomial(
    X, Y, F, feature_names=DEFAULT_PARAM_FEATURES, degree=2, alphas=None, cv=5
):
    """Fit Y ≈ X · poly_d(F) with L2 regularization, alpha by K-fold CV.

    Args:
        X, Y, F: as in `fit_cn_parametric`.
        feature_names: subset of FEATURE_NAMES for polynomial variables.
        degree: max total polynomial degree.
        alphas: candidate ridge strengths (default: log-spaced 1e-4 ... 1e4).
        cv: K-fold splits for picking alpha.
    """
    if alphas is None:
        alphas = np.logspace(-4, 4, 30)
    X = np.asarray(X)
    Y = np.asarray(Y)
    Fsub = _feature_subset(F, feature_names)

    poly = PolynomialFeatures(degree=degree, include_bias=True)
    Phi = poly.fit_transform(Fsub)
    D = X[:, None] * Phi

    ridge = RidgeCV(alphas=alphas, cv=cv, fit_intercept=False).fit(D, Y)

    return RidgePolynomialCn(
        feature_names=tuple(feature_names),
        degree=degree,
        alpha=float(ridge.alpha_),
        coef=np.asarray(ridge.coef_),
        poly=poly,
    )


# ─── (4) Gradient-boosted trees on log C_n ────────────────────────────────


@dataclass
class GBMCn:
    """Gradient-boosted-trees C_n on log(Y/X). Fit by `fit_cn_gbm`."""

    feature_names: Tuple[str, ...]
    model: object  # HistGradientBoostingRegressor

    def cn(self, F):
        Fsub = _feature_subset(F, self.feature_names)
        return np.exp(self.model.predict(Fsub))

    def predict(self, X, F):
        return np.asarray(X) * self.cn(F)

    def __repr__(self):
        return (
            f"GBMCn(max_iter={self.model.max_iter}, "
            f"max_depth={self.model.max_depth}, "
            f"lr={self.model.learning_rate})"
        )


def fit_cn_gbm(
    X,
    Y,
    F,
    feature_names=DEFAULT_PARAM_FEATURES,
    max_iter=300,
    max_depth=4,
    learning_rate=0.05,
    min_samples_leaf=10,
    random_state=0,
    eps=1e-6,
):
    """Fit log(Y/X) = GBM(F) via sklearn HistGradientBoostingRegressor.

    Only positive (X, Y) pairs contribute (log target requires positivity).
    """
    X = np.asarray(X)
    Y = np.asarray(Y)
    pos = (X > eps) & (Y > eps)
    if pos.sum() < 10:
        raise ValueError(f"fit_cn_gbm: only {pos.sum()} positive samples; need >= 10")
    Fsub = _feature_subset(F, feature_names)[pos]
    target = np.log(Y[pos] / X[pos])

    gbm = HistGradientBoostingRegressor(
        max_iter=max_iter,
        max_depth=max_depth,
        learning_rate=learning_rate,
        min_samples_leaf=min_samples_leaf,
        random_state=random_state,
    ).fit(Fsub, target)

    return GBMCn(feature_names=tuple(feature_names), model=gbm)


# ─── (6) Gaussian process on log C_n ──────────────────────────────────────


@dataclass
class GPLogCn:
    """GP-regressed log C_n with ARD Matern kernel. Fit by `fit_cn_gp_log`."""

    feature_names: Tuple[str, ...]
    gp: object  # GaussianProcessRegressor
    feature_mean: np.ndarray
    feature_std: np.ndarray

    def _normalize(self, F):
        Fsub = _feature_subset(F, self.feature_names)
        return (Fsub - self.feature_mean) / self.feature_std

    def cn(self, F):
        return np.exp(self.gp.predict(self._normalize(F)))

    def predict(self, X, F):
        return np.asarray(X) * self.cn(F)

    def predict_with_std(self, X, F):
        """Returns (Y_pred, sigma_log). sigma_log is the std on log C_n."""
        mu, sigma = self.gp.predict(self._normalize(F), return_std=True)
        return np.asarray(X) * np.exp(mu), sigma

    def __repr__(self):
        return f"GPLogCn(kernel={self.gp.kernel_})"


def fit_cn_gp_log(
    X,
    Y,
    F,
    feature_names=DEFAULT_PARAM_FEATURES,
    nu=2.5,
    noise_level=0.1,
    n_restarts=5,
    eps=1e-6,
):
    """Fit log(Y/X) = GP(F) with an ARD Matern(nu) kernel + white noise.

    Features standardized to unit variance per dimension before training so
    the per-feature length scales are comparable. `normalize_y=True` lets
    the GP fit a per-dataset mean offset.
    """
    X = np.asarray(X)
    Y = np.asarray(Y)
    pos = (X > eps) & (Y > eps)
    if pos.sum() < 10:
        raise ValueError(f"fit_cn_gp_log: only {pos.sum()} positive samples; need >= 10")
    Fsub = _feature_subset(F, feature_names)[pos]
    target = np.log(Y[pos] / X[pos])

    fmean = Fsub.mean(axis=0)
    fstd = Fsub.std(axis=0)
    fstd = np.where(fstd > 1e-9, fstd, 1.0)
    Fn = (Fsub - fmean) / fstd

    kernel = (
        ConstantKernel(1.0, (1e-3, 1e3))
        * Matern(length_scale=np.ones(Fn.shape[1]), length_scale_bounds=(1e-2, 1e3), nu=nu)
        + WhiteKernel(noise_level=noise_level, noise_level_bounds=(1e-5, 1e1))
    )
    gp = GaussianProcessRegressor(
        kernel=kernel,
        n_restarts_optimizer=n_restarts,
        normalize_y=True,
        random_state=0,
    ).fit(Fn, target)

    return GPLogCn(
        feature_names=tuple(feature_names),
        gp=gp,
        feature_mean=fmean,
        feature_std=fstd,
    )
