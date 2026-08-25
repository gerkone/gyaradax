"""Pure-JAX MLP flux surrogate conditioned on local operational parameters.

Custom alternative to QLKNN-style surrogates: a small MLP mapping the
campaign feature vector (subset of data.FEATURE_NAMES) to per-channel GKW
gyroBohm fluxes (default targets: Q_i, Q_e, Gamma_e), trained directly on
nonlinear gyaradax runs (scripts/train_mlp.py). Weights ship as an npz with
feature/target names, input standardization, target transform, and the
calibration grid metadata (adopted by the TORAX plugin like Cn heads).
"""

import json

import jax
import jax.numpy as jnp
import numpy as np

DEFAULT_TARGETS = ("q_i", "q_e", "pfe")


def init_mlp(sizes, key):
    """He-initialized (W, b) list for layer sizes like (10, 64, 64, 3)."""
    params = []
    for n_in, n_out in zip(sizes[:-1], sizes[1:]):
        key, sub = jax.random.split(key)
        w = jax.random.normal(sub, (n_in, n_out)) * jnp.sqrt(2.0 / n_in)
        params.append((w, jnp.zeros(n_out)))
    return params


def mlp_apply(params, x):
    """Forward pass, gelu hidden activations, linear output."""
    for w, b in params[:-1]:
        x = jax.nn.gelu(x @ w + b)
    w, b = params[-1]
    return x @ w + b


def gpr_apply(model, x):
    """Exact GP posterior mean: RBF kernel dot with precomputed alpha."""
    xt = model["x_train"]
    d2 = jnp.sum(((x[..., None, :] - xt) / model["length_scale"]) ** 2, axis=-1)
    k = model["sigma_f"] ** 2 * jnp.exp(-0.5 * d2)
    return k @ model["alpha"]


def predict(model, features):
    """Standardize -> surrogate (mlp or gpr) -> inverse target transform."""
    x = (jnp.asarray(features) - model["x_mean"]) / model["x_std"]
    y = gpr_apply(model, x) if model.get("kind") == "gpr" else mlp_apply(model["params"], x)
    if model["y_transform"] == "asinh":
        y = jnp.sinh(y) * model["y_scale"]
    return y


def save_model(path, params, *, feature_names, target_names, x_mean, x_std,
               y_transform, y_scale, grid=None, meta=None, kind="mlp", gpr=None):
    arrays = {"kind": np.asarray(kind)}
    if kind == "gpr":
        arrays.update({k: np.asarray(v) for k, v in gpr.items()})
        arrays["n_layers"] = np.asarray(0)
    else:
        for i, (w, b) in enumerate(params):
            arrays[f"w{i}"] = np.asarray(w)
            arrays[f"b{i}"] = np.asarray(b)
        arrays["n_layers"] = np.asarray(len(params))
    arrays["feature_names"] = np.asarray(feature_names)
    arrays["target_names"] = np.asarray(target_names)
    arrays["x_mean"] = np.asarray(x_mean)
    arrays["x_std"] = np.asarray(x_std)
    arrays["y_transform"] = np.asarray(y_transform)
    arrays["y_scale"] = np.asarray(y_scale)
    arrays["grid"] = np.asarray(json.dumps(grid or {}))
    arrays["meta"] = np.asarray(json.dumps(meta or {}))
    np.savez(path, **arrays)


def load_model(path):
    """Load an MLP surrogate npz into a jax-ready dict."""
    d = np.load(path, allow_pickle=True)
    n = int(d["n_layers"])
    params = [(jnp.asarray(d[f"w{i}"]), jnp.asarray(d[f"b{i}"])) for i in range(n)]
    extra = {}
    if "kind" in d.files and str(d["kind"]) == "gpr":
        extra = {k: jnp.asarray(d[k]) for k in ("x_train", "alpha", "length_scale", "sigma_f")}
    return {
        "kind": str(d["kind"]) if "kind" in d.files else "mlp",
        **extra,
        "params": params,
        "feature_names": tuple(str(s) for s in d["feature_names"]),
        "target_names": tuple(str(s) for s in d["target_names"]),
        "x_mean": jnp.asarray(d["x_mean"]),
        "x_std": jnp.asarray(d["x_std"]),
        "y_transform": str(d["y_transform"]),
        "y_scale": jnp.asarray(d["y_scale"]),
        "grid": json.loads(str(d["grid"])),
        "meta": json.loads(str(d["meta"])),
    }
