"""Run outcome: linear gate decision, heat-flux decay, s-Nyquist banding and the outcome summary.

``gyaradax run`` with ``--linear-gate``, ``--stop-on-decay``, ``--stop-on-banding``,
``--track-outcome`` or ``--dataset-from-flux`` keeps these files in the run directory::

    linear_gate.json     linear growth rates, real frequencies and the gate decision
    linear_mode.npz      df (complex64) and phi at the end of the linear run
    monitor_series.npz   per frame: time, step, fluxes, growth rate, s-Nyquist shares of phi
    outcome.json         summary: outcome class, flux averages, banding, stop reason

The datagen scheduler reads the JSON files without numpy (``scheduler.write_db``).
"""

from __future__ import annotations

import json
import math
import os
import time
from dataclasses import asdict, dataclass
from typing import Any, Mapping, Optional

import numpy as np

# exit codes of `gyaradax run`, mirrored in gyaradax/datagen/scheduler.py
EXIT_STABLE = 4
EXIT_DECAYED = 5
EXIT_BANDED = 6

GATE_FILE = "linear_gate.json"
MODE_FILE = "linear_mode.npz"
SERIES_FILE = "monitor_series.npz"
OUTCOME_FILE = "outcome.json"

# outcome classes counted as stable in the dataset metadata
STABLE_CLASSES = ("linear_stable", "decayed", "flat")


def _atomic_json(path: str, doc: Mapping[str, Any]) -> None:
    tmp = f"{path}.tmp{os.getpid()}"
    with open(tmp, "w") as fh:
        json.dump(doc, fh, indent=1)
    os.replace(tmp, path)


def _finite(x: Any) -> Optional[float]:
    x = float(x)
    return x if math.isfinite(x) else None


def gate_decision(result: Mapping[str, Any], stable: float = -0.01, gamma_min: float = 0.02) -> str:
    """``stable``, ``unstable`` or ``ambiguous`` from the growth rates of every ky > 0.

    Stable: the rates settled and the largest is below ``stable``. Unstable: the largest is
    above ``gamma_min``. Anything else (between the two, or not settled) is ambiguous.
    """
    ky, g = np.asarray(result["ky"], float), np.asarray(result["gamma"], float)
    sel = (ky > 0) & np.isfinite(g)
    if not sel.any():
        return "ambiguous"
    gmax = float(g[sel].max())
    if gmax > gamma_min:
        return "unstable"
    if result.get("converged") and gmax < stable:
        return "stable"
    return "ambiguous"


def gate_report(
    result: Mapping[str, Any], stable: float, gamma_min: float, low_ky: float = 1.0
) -> dict[str, Any]:
    """The ``linear_gate.json`` document of a ``growth.growth_rates`` result.

    ``gamma_max`` is over every ky > 0 (the decision uses it), ``gamma_max_lowky`` over
    0 < kθρ <= ``low_ky`` only.
    """
    ky, g = np.asarray(result["ky"], float), np.asarray(result["gamma"], float)
    sel = ky > 0
    i = int(np.flatnonzero(sel)[np.argmax(g[sel])]) if sel.any() else 0
    low = sel & (ky <= low_ky)
    j = int(np.flatnonzero(low)[np.argmax(g[low])]) if low.any() else 0
    return {
        **{k: v for k, v in result.items() if k != "peak"},
        "settled": bool(result["converged"]),
        "gamma_max": _finite(g[i]) if sel.any() else None,
        "peak_ky": float(ky[i]) if sel.any() else None,
        "omega_peak": _finite(result["omega"][i]) if "omega" in result and sel.any() else None,
        "low_ky": float(low_ky),
        "gamma_max_lowky": _finite(g[j]) if low.any() else None,
        "peak_ky_lowky": float(ky[j]) if low.any() else None,
        "stable_below": float(stable),
        "gamma_min": float(gamma_min),
        "decision": gate_decision(result, stable, gamma_min),
    }


def format_gate(gate: Mapping[str, Any]) -> str:
    settled = "settled" if gate["settled"] else "not settled"
    gmax = gate["gamma_max"]
    text = "-" if gmax is None else f"{gmax:+.4f} at kθρ {gate['peak_ky']:.3f}"
    omega = gate.get("omega_peak")
    freq = "" if omega is None else f", ω {omega:+.4f}"
    return f"linear gate: γmax {text}{freq} ({settled}, t {gate['t']:.1f}): {gate['decision']}"


def nyquist_shares(phi_real: Any) -> tuple[float, float]:
    """Share of the s-Nyquist power in the zonal (y-mean) part of a real phi (x, s, y) and in the rest.

    Each share is the power at FFT index ns // 2 along s, summed over x (and y), over the
    total power of that part.
    """
    phi = np.asarray(phi_real, np.float64)
    zonal = phi.mean(axis=2, keepdims=True)

    def share(f: np.ndarray) -> float:
        p = np.abs(np.fft.fft(f, axis=1)) ** 2
        total = float(p.sum())
        return float(p[:, f.shape[1] // 2].sum()) / total if total > 0 else 0.0

    return share(zonal), share(phi - zonal)


def heat_flux(fluxes: Any) -> np.ndarray:
    """Heat flux per species of a flux array: (nsp, 3) kinetic or (3,) adiabatic."""
    return np.asarray(fluxes, np.float64).reshape(-1, 3)[:, 1]


def decayed(t: Any, q: Any, *, t_min: float, floor: float, rel_floor: float, window: int) -> bool:
    """Whether the heat flux ``q(t)`` has decayed.

    After ``t_min``, the last ``window`` values all lie below the larger of ``floor`` and
    ``rel_floor`` times the peak, and still fall (negative slope of log q over them, and the
    last below the first).
    """
    t, q = np.asarray(t, float), np.abs(np.asarray(q, float))
    if len(q) < max(window, 2) or t[-1] < t_min:
        return False
    limit = max(floor, rel_floor * float(np.max(q)))
    last = q[-window:]
    if not np.all(last < limit):
        return False
    slope = np.polyfit(t[-window:], np.log(np.maximum(last, 1e-300)), 1)[0]
    return bool(slope < 0 and last[-1] < last[0])


def banded(shares: Any, threshold: float, window: int) -> bool:
    s = np.asarray(shares, float)
    return len(s) >= window and bool(np.all(s[-window:] > threshold))


def flux_average(t: Any, flux: Any, frac: float = 0.3, n_batches: int = 5) -> dict[str, Any]:
    """Mean of ``flux`` over the last ``frac`` of the run time, its batch-means error and trend.

    ``flux`` is (frames, channels). The window is split into ``n_batches`` consecutive batches;
    the error is the standard error of their means. ``drift`` is the change over the window of a
    straight line fitted to the batch means, relative to the mean, and ``drift_sigma`` that change
    in units of its standard error (nan with fewer than three batches).
    """
    t, x = np.asarray(t, float), np.asarray(flux, float).reshape(len(t), -1)
    span = t[-1] - t[0]
    sel = t >= t[-1] - frac * span - 1e-9 * max(span, 1.0)
    tw, xw = t[sel], x[sel]
    mean = xw.mean(axis=0)
    nb = min(n_batches, len(tw) // 2)
    err = np.full(mean.shape, np.nan)
    drift = np.full(mean.shape, np.nan)
    sigma = np.full(mean.shape, np.nan)
    if nb >= 2:
        idx = np.array_split(np.arange(len(tw)), nb)
        bm = np.array([xw[i].mean(axis=0) for i in idx])
        bt = np.array([tw[i].mean() for i in idx])
        err = bm.std(axis=0, ddof=1) / np.sqrt(nb)
        width = tw[-1] - tw[0]
        if nb >= 3 and width > 0:
            coef = np.polyfit(bt, bm, 1)
            res = bm - (coef[0][None] * bt[:, None] + coef[1][None])
            sxx = float(np.sum((bt - bt.mean()) ** 2))
            se = np.sqrt(np.sum(res**2, axis=0) / (nb - 2) / sxx)
            with np.errstate(divide="ignore", invalid="ignore"):
                drift = coef[0] * width / np.abs(mean)
                sigma = np.where(se > 0, coef[0] / se, np.sign(coef[0]) * np.inf)
    return {
        "window": [float(tw[0]), float(tw[-1])],
        "frames": int(len(tw)),
        "batches": int(nb),
        "mean": mean,
        "err": err,
        "drift": drift,
        "drift_sigma": sigma,
    }


def classify(
    stopped: Optional[str],
    banded_flag: bool,
    q_mean: float,
    drift: float,
    drift_sigma: float,
    *,
    floor: float,
    drift_tol: float = 0.2,
) -> str:
    """Outcome class of a nonlinear run from its total heat flux over the averaging window.

    decayed (stopped on decay) > banded > flat (mean below ``floor``) > growing / decaying (a
    drift beyond ``drift_tol`` of the mean, at two standard errors) > saturated.
    """
    if stopped == "decayed":
        return "decayed"
    if banded_flag:
        return "banded"
    if not math.isfinite(q_mean) or abs(q_mean) < floor:
        return "flat"
    if math.isfinite(drift) and math.isfinite(drift_sigma) and abs(drift) > drift_tol:
        if drift > 0 and drift_sigma > 2:
            return "growing"
        if drift < 0 and drift_sigma < -2:
            return "decaying"
    return "saturated"


@dataclass
class MonitorSettings:
    stop_on_decay: bool = False
    decay_t_min: float = 30.0
    decay_floor: float = 1e-3
    decay_rel_floor: float = 1e-3
    decay_window: int = 5
    decay_keep_frames: bool = False
    banding_threshold: float = 0.3
    banding_window: int = 3
    stop_on_banding: bool = False
    dataset_from_flux: Optional[float] = None
    average_fraction: float = 0.3


class RunMonitor:
    """Follows a nonlinear run frame by frame: heat-flux decay, s-Nyquist banding, outcome summary.

    ``observe`` takes each frame's fluxes and phi, appends them to ``monitor_series.npz``,
    rewrites ``outcome.json`` and returns the reason to stop (``decayed``, ``banded``) or None.
    A resumed run continues the series of the frames up to its start step. ``write_frame`` says
    whether a frame goes to the dataset (``dataset_from_flux``), ``finish`` stores the full
    traces and the outcome in the trajectory metadata.
    """

    def __init__(
        self,
        run_dir: str,
        settings: MonitorSettings,
        start_step: int = 0,
        ion: int = 0,
        electron: Optional[int] = None,
        gate: Optional[Mapping[str, Any]] = None,
    ):
        self.run_dir = run_dir
        self.settings = settings
        self.ion, self.electron = ion, electron
        self.gate = dict(gate) if gate else None
        self.series: dict[str, list] = {
            k: [] for k in ("time", "step", "flux", "growth", "nyq_zonal", "nyq_nonzonal")
        }
        self.stopped: Optional[str] = None
        self.message = ""
        path = os.path.join(run_dir, SERIES_FILE)
        if os.path.isfile(path):
            with np.load(path) as data:
                keep = np.asarray(data["step"]) <= start_step
                for k in self.series:
                    self.series[k] = list(np.asarray(data[k])[keep])

    @property
    def exit_code(self) -> int:
        return {"decayed": EXIT_DECAYED, "banded": EXIT_BANDED}.get(self.stopped or "", 0)

    def _q(self) -> np.ndarray:
        return np.array([heat_flux(f).sum() for f in self.series["flux"]])

    def observe(self, state: Any, fluxes: Any, phi: Any) -> Optional[str]:
        from gyaradax.io.dataset import phi_to_real

        s = self.settings
        zonal, nonzonal = nyquist_shares(phi_to_real(phi))
        self.series["time"].append(float(np.ravel(np.asarray(state.time))[0]))
        self.series["step"].append(int(np.ravel(np.asarray(state.step))[0]))
        self.series["flux"].append(np.asarray(fluxes, np.float64).reshape(-1, 3))
        self.series["growth"].append(np.asarray(state.last_growth_rate, np.float64))
        self.series["nyq_zonal"].append(zonal)
        self.series["nyq_nonzonal"].append(nonzonal)
        t, q = np.asarray(self.series["time"]), self._q()
        if s.stop_on_decay and decayed(
            t,
            q,
            t_min=s.decay_t_min,
            floor=s.decay_floor,
            rel_floor=s.decay_rel_floor,
            window=s.decay_window,
        ):
            self.stopped = "decayed"
            self.message = (
                f"heat flux {q[-1]:.3e} at t={t[-1]:.2f} decayed below the floor "
                f"(peak {np.max(np.abs(q)):.3e}) and still falls"
            )
        elif s.stop_on_banding and banded(
            self.series["nyq_zonal"], s.banding_threshold, s.banding_window
        ):
            self.stopped = "banded"
            self.message = (
                f"zonal s-Nyquist share {zonal:.2f} above {s.banding_threshold:g} for "
                f"{s.banding_window} frames at t={t[-1]:.2f}"
            )
        self._save()
        return self.stopped

    def write_frame(self, n_frames: int, final: bool = False) -> bool:
        """Whether this frame goes to the dataset: always, or once the heat flux has reached the threshold."""
        thr = self.settings.dataset_from_flux
        if thr is None or n_frames > 0 or final:
            return True
        return bool(len(self.series["flux"]) and np.max(np.abs(self._q())) >= thr)

    def banded_since(self) -> Optional[float]:
        s = self.settings
        z, t = self.series["nyq_zonal"], self.series["time"]
        for i in range(s.banding_window - 1, len(z)):
            if banded(z[: i + 1], s.banding_threshold, s.banding_window):
                return float(t[i])
        return None

    def summary(self, final: bool = False) -> dict[str, Any]:
        """The ``outcome.json`` document: class, label, flux averages, decay and banding."""
        s = self.settings
        t = np.asarray(self.series["time"], float)
        doc: dict[str, Any] = {
            "updated": time.time(),
            "final": bool(final),
            "stopped": self.stopped,
            "message": self.message,
            "frames": int(len(t)),
            "t_start": float(t[0]) if len(t) else None,
            "t": float(t[-1]) if len(t) else None,
            "gate": self.gate["decision"] if self.gate else None,
            "settings": asdict(s),
        }
        if not len(t):
            return {**doc, "class": "running", "label": "no frame yet"}
        flux = np.asarray(self.series["flux"])
        q = self._q()
        zonal = np.asarray(self.series["nyq_zonal"], float)
        since = self.banded_since()
        species = (
            {"ion": self.ion}
            if self.electron is None
            else {"ion": self.ion, "electron": self.electron}
        )
        doc.update(
            nyq_zonal=float(zonal[-1]),
            nyq_nonzonal=float(self.series["nyq_nonzonal"][-1]),
            nyq_zonal_max=float(zonal.max()),
            banded=since is not None,
            banded_t=since,
            q_peak=float(np.max(np.abs(q))),
            q_last=float(q[-1]),
            species=species,
        )
        avg = flux_average(
            t,
            np.concatenate([flux.reshape(len(t), -1), q[:, None]], axis=1),
            frac=s.average_fraction,
        )
        n_sp = flux.shape[1]

        def per_species(arr: np.ndarray) -> list[list[Optional[float]]]:
            return [[_finite(v) for v in row] for row in arr[:-1].reshape(n_sp, 3)]

        doc["average"] = {
            "window": avg["window"],
            "frames": avg["frames"],
            "batches": avg["batches"],
            "labels": ["pflux", "eflux", "vflux"],
            "mean": per_species(avg["mean"]),
            "err": per_species(avg["err"]),
            "drift": per_species(avg["drift"]),
            "q": _finite(avg["mean"][-1]),
            "q_err": _finite(avg["err"][-1]),
            "q_drift": _finite(avg["drift"][-1]),
            "q_drift_sigma": _finite(avg["drift_sigma"][-1]),
        }
        cls = classify(
            self.stopped,
            since is not None,
            float(avg["mean"][-1]),
            float(avg["drift"][-1]),
            float(avg["drift_sigma"][-1]),
            floor=s.decay_floor,
        )
        if not final and cls in ("saturated", "growing", "decaying", "flat"):
            cls = "running"
        doc["class"] = cls
        qi = avg["mean"][:-1].reshape(n_sp, 3)[self.ion, 1]
        qi_err = avg["err"][:-1].reshape(n_sp, 3)[self.ion, 1]
        doc["label"] = f"{cls} t {t[-1]:.1f} Qi {qi:.3g}±{qi_err:.2g}" + (
            f" banded (zonal Nyquist {zonal[-1]:.2f})"
            if since is not None and cls != "banded"
            else ""
        )
        return doc

    def _save(self, final: bool = False) -> None:
        os.makedirs(self.run_dir, exist_ok=True)
        path = os.path.join(self.run_dir, SERIES_FILE)
        tmp = f"{path}.tmp{os.getpid()}.npz"
        np.savez(tmp, **{k: np.asarray(v) for k, v in self.series.items()})
        os.replace(tmp, path)
        _atomic_json(os.path.join(self.run_dir, OUTCOME_FILE), self.summary(final))

    def traces(self) -> dict[str, Any]:
        """Per-frame series for the trajectory metadata (``trace_*`` keys)."""
        return {
            "trace_timesteps": np.asarray(self.series["time"], np.float64),
            "trace_solver_steps": np.asarray(self.series["step"], np.int64),
            "trace_flux": np.asarray(self.series["flux"], np.float64).reshape(
                len(self.series["flux"]), -1
            ),
            "trace_growth": np.asarray(self.series["growth"], np.float64),
            "trace_nyq_zonal": np.asarray(self.series["nyq_zonal"], np.float64),
            "trace_nyq_nonzonal": np.asarray(self.series["nyq_nonzonal"], np.float64),
        }

    def finish(self, dataset: Any = None, final: bool = True) -> dict[str, Any]:
        """``outcome.json`` at the end of a leg; with a dataset writer also the traces and outcome.

        ``final`` (the run is complete, not interrupted) settles the outcome class; a run that
        decayed then keeps only its last dataset frame unless ``decay_keep_frames``.
        """
        self._save(final=final)
        doc = self.summary(final=final)
        if dataset is not None:
            if final and self.stopped == "decayed" and not self.settings.decay_keep_frames:
                dataset.keep_last_frame()
            dataset.update_static(
                {**self.traces(), "outcome": doc, "stable": doc["class"] in STABLE_CLASSES}
            )
        return doc


def gate_outcome(run_dir: str, gate: Mapping[str, Any]) -> dict[str, Any]:
    """Write the ``outcome.json`` of a run the linear gate skipped."""
    doc = {
        "updated": time.time(),
        "final": True,
        "class": "linear_stable",
        "stopped": "linear_stable",
        "gate": gate["decision"],
        "frames": 0,
        "t": None,
        "label": "linear_stable " + format_gate(gate).split(": ", 1)[1].rsplit(":", 1)[0],
        "message": format_gate(gate),
    }
    _atomic_json(os.path.join(run_dir, OUTCOME_FILE), doc)
    return doc
