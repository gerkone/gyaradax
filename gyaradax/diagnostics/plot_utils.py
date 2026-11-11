"""Plotting helpers for fluxes, spectra, growth rates and phase-space slices."""

# mypy: ignore-errors

import json
import os
from typing import List, Mapping, Optional, Sequence, Union, Tuple
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import jax.numpy as jnp

GK_LABELS = {
    5: [r"v_{||}", r"\mu", r"s", r"k_x", r"k_y"],
    6: [r"sp", r"v_{||}", r"\mu", r"s", r"k_x", r"k_y"],
}

SPECIES_LABELS = {0: "ion", 1: "electron"}

JAX_COLORS = {
    "blue": "#4285F4",
    "red": "#EA4335",
    "yellow": "#FBBC05",
    "green": "#34A853",
    "cyan": "#24B6AD",
    "purple": "#9B51E0",
}

SPECIES_COLORS = [JAX_COLORS["cyan"], JAX_COLORS["cyan"]]

RUN_COLORS = [JAX_COLORS[c] for c in ("cyan", "purple", "red", "blue", "green", "yellow")]

FLUX_CHANNELS = ("particle", "heat", "momentum")

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica", "Arial", "Liberation Sans"],
        "font.size": 9,
        "axes.labelsize": 10,
        "axes.titlesize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "figure.titlesize": 12,
        "lines.linewidth": 1.25,
        "grid.alpha": 0.15,
        "grid.linestyle": ":",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "figure.figsize": (7.0, 2.8),
    }
)


def plot_flux_trace(
    time: np.ndarray,
    fluxes: Union[np.ndarray, Tuple[np.ndarray, ...]],
    labels: List[str] = ["Particle", "Heat", "Momentum"],
    ref_time: Optional[np.ndarray] = None,
    ref_fluxes: Optional[Union[np.ndarray, Tuple[np.ndarray, ...], List[np.ndarray]]] = None,
    title: str = "Flux Evolution",
    show_average: bool = False,
    avg_window: int = 80,
    n_species: int = 1,
    species_labels: Optional[List[str]] = None,
) -> plt.Figure:
    """Plot flux time traces: one row per flux, one column per species.

    Args:
        time: Times, shape (T,).
        fluxes: ``(n_species * len(labels), T)``, an ensemble
            ``(n_runs, n_species * len(labels), T)`` drawn as mean and spread, or
            a tuple of ``(T,)`` arrays.
        labels: Flux names, one row each.
        ref_time: Times of the reference traces.
        ref_fluxes: Reference (e.g. GKW) traces, same layout as ``fluxes``.
        title: Figure title.
        show_average: Mark the average over the last ``avg_window`` points.
        avg_window: Averaging window, in points.
        n_species: Number of species columns.
        species_labels: Column titles.

    Returns:
        The figure.
    """
    if isinstance(fluxes, (tuple, list)):
        min_len = min(f.shape[-1] for f in fluxes)
        fluxes = np.stack([f[..., :min_len] for f in fluxes])
        time = time[:min_len]

    if ref_fluxes is not None:
        if isinstance(ref_fluxes, (tuple, list)):
            ref_min_len = min(f.shape[-1] for f in ref_fluxes)
            ref_fluxes = np.stack([f[..., :ref_min_len] for f in ref_fluxes])
            if ref_time is not None:
                ref_time = ref_time[:ref_min_len]

        if ref_fluxes.ndim == 3:
            ref_fluxes_mean = np.mean(ref_fluxes, axis=0)
            ref_fluxes_std = np.std(ref_fluxes, axis=0)
        else:
            ref_fluxes_mean = ref_fluxes
            ref_fluxes_std = None
    else:
        ref_fluxes_mean = None
        ref_fluxes_std = None

    if fluxes.ndim == 3:
        fluxes_mean = np.mean(fluxes, axis=0)
        fluxes_std = np.std(fluxes, axis=0)
    else:
        fluxes_mean = fluxes
        fluxes_std = None

    if species_labels is None:
        species_labels = [SPECIES_LABELS.get(i, f"sp{i}") for i in range(n_species)]

    n_flux = len(labels)
    ncols = n_species
    fig, axes = plt.subplots(n_flux, ncols, figsize=(7.0, 1.6 * n_flux), sharex=True, squeeze=False)

    for isp in range(n_species):
        col_offset = isp * n_flux
        color = SPECIES_COLORS[isp % len(SPECIES_COLORS)]

        for i in range(n_flux):
            ax = axes[i, isp]
            flux_idx = col_offset + i
            if flux_idx >= fluxes_mean.shape[0]:
                continue

            if fluxes.ndim == 3:
                for run_idx in range(fluxes.shape[0]):
                    ax.plot(
                        time,
                        fluxes[run_idx, flux_idx],
                        color=color,
                        alpha=0.15,
                        lw=1.0,
                        linestyle="-",
                        zorder=1,
                    )

            ax.plot(
                time,
                fluxes_mean[flux_idx],
                label="gyaradax",
                color=color,
                lw=1.5,
                alpha=1.0,
                zorder=3,
            )

            if fluxes_std is not None:
                ax.fill_between(
                    time,
                    fluxes_mean[flux_idx] - fluxes_std[flux_idx],
                    fluxes_mean[flux_idx] + fluxes_std[flux_idx],
                    color=color,
                    alpha=0.3,
                    lw=0,
                    zorder=2,
                )

            if show_average and len(fluxes_mean[flux_idx]) >= avg_window:
                avg_val = np.mean(fluxes_mean[flux_idx][-avg_window:])
                t_start = time[-avg_window]
                t_end = time[-1]

                ax.plot(
                    [t_start, t_end],
                    [avg_val, avg_val],
                    color=color,
                    linestyle="-",
                    lw=2.0,
                    zorder=4,
                )

                ax.text(
                    t_end,
                    avg_val,
                    f"{avg_val:.4g}",
                    color=color,
                    va="bottom",
                    ha="right",
                    fontweight="bold",
                    fontsize=9,
                    zorder=5,
                    bbox=dict(facecolor="white", alpha=0.8, edgecolor="none", pad=0.2),
                )

            if ref_fluxes_mean is not None and ref_time is not None:
                ref_idx = col_offset + i
                if ref_idx < ref_fluxes_mean.shape[0]:
                    if ref_fluxes.ndim == 3:
                        for run_idx in range(ref_fluxes.shape[0]):
                            ax.plot(
                                ref_time,
                                ref_fluxes[run_idx, ref_idx],
                                color="black",
                                alpha=0.15,
                                lw=1.0,
                                linestyle="--",
                                zorder=0,
                            )

                    ax.plot(
                        ref_time,
                        ref_fluxes_mean[ref_idx],
                        color="black",
                        linestyle="--",
                        label="GKW",
                        alpha=0.8,
                        lw=1.4,
                        zorder=2,
                    )

                    if ref_fluxes_std is not None:
                        ax.fill_between(
                            ref_time,
                            ref_fluxes_mean[ref_idx] - ref_fluxes_std[ref_idx],
                            ref_fluxes_mean[ref_idx] + ref_fluxes_std[ref_idx],
                            color="black",
                            alpha=0.2,
                            lw=0,
                            zorder=1,
                        )

                    if show_average and len(ref_fluxes_mean[ref_idx]) >= avg_window:
                        ref_avg_val = np.mean(ref_fluxes_mean[ref_idx][-3 * avg_window :])
                        ref_t_start = ref_time[-3 * avg_window]
                        ref_t_end = ref_time[-1]

                        ax.plot(
                            [ref_t_start, ref_t_end],
                            [ref_avg_val, ref_avg_val],
                            color="black",
                            linestyle="-",
                            lw=2.0,
                            zorder=4,
                        )

                        ax.text(
                            ref_t_end,
                            ref_avg_val,
                            f"{ref_avg_val:.4g}",
                            color="black",
                            va="top",
                            ha="right",
                            fontweight="bold",
                            fontsize=9,
                            zorder=5,
                            bbox=dict(facecolor="white", alpha=0.8, edgecolor="none", pad=0.2),
                        )

            if isp == 0:
                ax.set_ylabel(labels[i], fontsize=12)
            ax.grid(True, axis="y")
            if i == 0:
                if n_species > 1:
                    ax.set_title(species_labels[isp], fontsize=9)
                if isp == 0:
                    ax.legend(frameon=False, loc="upper left", fontsize=10)

        axes[-1, isp].set_xlabel(r"time $[R/v_{th}]$", fontsize=12)

    fig.suptitle(title, fontweight="bold", fontsize=14)
    fig.align_ylabels(axes[:, 0])
    fig.tight_layout()
    return fig


def plot_spectra(
    kx: np.ndarray,
    ky: np.ndarray,
    phi: Optional[jnp.ndarray] = None,
    kx_spec: Optional[np.ndarray] = None,
    ky_spec: Optional[np.ndarray] = None,
    ref_phi: Optional[jnp.ndarray] = None,
    ref_kx_spec: Optional[np.ndarray] = None,
    ref_ky_spec: Optional[np.ndarray] = None,
    kx_spec_std: Optional[np.ndarray] = None,
    ky_spec_std: Optional[np.ndarray] = None,
    ref_kx_spec_std: Optional[np.ndarray] = None,
    ref_ky_spec_std: Optional[np.ndarray] = None,
    title: str = "",
    shade_alpha: float = 0.25,
) -> plt.Figure:
    """Plot the kx and ky spectra, from ``phi`` or from precomputed 1D spectra.

    Args:
        kx: Radial wavenumbers.
        ky: Binormal wavenumbers.
        phi: Potential, (ns, nkx, nky); used when the spectra are not given.
        kx_spec: kx spectrum.
        ky_spec: ky spectrum.
        ref_phi: Reference potential (e.g. GKW).
        ref_kx_spec: Reference kx spectrum.
        ref_ky_spec: Reference ky spectrum.
        kx_spec_std: Spread of ``kx_spec``, drawn as a band.
        ky_spec_std: Spread of ``ky_spec``.
        ref_kx_spec_std: Spread of ``ref_kx_spec``.
        ref_ky_spec_std: Spread of ``ref_ky_spec``.
        title: Figure title.
        shade_alpha: Opacity of the bands.

    Returns:
        The figure.
    """
    if kx_spec is None or ky_spec is None:
        if phi is None:
            raise ValueError("provide either phi or both (kx_spec, ky_spec)")
        phi_sq = np.abs(np.array(phi)) ** 2
        kx_spec = np.sum(phi_sq, axis=(0, 2))
        ky_spec = np.sum(phi_sq, axis=(0, 1))

    if ref_phi is not None:
        ref_phi_sq = np.abs(np.array(ref_phi)) ** 2
        ref_kx_spec = np.sum(ref_phi_sq, axis=(0, 2))
        ref_ky_spec = np.sum(ref_phi_sq, axis=(0, 1))

    def _shade(ax, x, mean, std, color):
        _band(ax, x, mean, std, color, shade_alpha)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.0, 2.2))

    ax1.semilogy(
        ky,
        ky_spec,
        "o-",
        color=JAX_COLORS["purple"],
        markersize=3,
        lw=1,
        label="gyaradax",
    )
    if ky_spec_std is not None:
        _shade(ax1, ky, ky_spec, ky_spec_std, JAX_COLORS["purple"])
    if ref_ky_spec is not None:
        ax1.semilogy(
            ky,
            ref_ky_spec,
            marker="x",
            linestyle="None",
            color="black",
            markersize=4,
            markeredgewidth=1,
            label="GKW",
        )
        if ref_ky_spec_std is not None:
            _shade(ax1, ky, ref_ky_spec, ref_ky_spec_std, "black")
    ax1.set_xlabel(r"$k_y \rho_{ref}$", fontsize=12)
    ax1.set_ylabel(r"$k_y^{\text{spec}}$", fontsize=12)
    ax1.set_title(r"$k_y$ spectrum", fontsize=12)
    ax1.grid(True, which="both")
    ax1.legend(fontsize=10)

    ax2.semilogy(kx, kx_spec, "o-", color=JAX_COLORS["purple"], markersize=3, lw=1)
    if kx_spec_std is not None:
        _shade(ax2, kx, kx_spec, kx_spec_std, JAX_COLORS["purple"])
    if ref_kx_spec is not None:
        idx_x = _progressive_sparse_indices(ref_kx_spec)
        ax2.semilogy(
            kx[idx_x],
            ref_kx_spec[idx_x],
            marker="x",
            linestyle="None",
            color="black",
            markersize=4,
            markeredgewidth=1,
        )
        if ref_kx_spec_std is not None:
            _shade(ax2, kx, ref_kx_spec, ref_kx_spec_std, "black")
        ax2.set_xlabel(r"$k_x \rho_{ref}$", fontsize=12)
    ax2.set_ylabel(r"$k_x^{\text{spec}}$", fontsize=12)
    ax2.set_title(r"$k_x$ spectrum", fontsize=12)
    ax2.grid(True, which="both")

    if len(title) > 0:
        fig.suptitle(title, fontweight="bold", fontsize=14)
    fig.tight_layout()
    return fig


def _progressive_sparse_indices(spec):
    """Generate progressively sparse indices moving outward from the peak."""
    N = len(spec)
    peak_idx = int(np.argmax(spec))
    idx = [peak_idx]
    for direction in [-1, 1]:
        curr = peak_idx + direction
        step = 1
        count = 0
        while 0 <= curr < N:
            idx.append(curr)
            count += 1
            if count % 3 == 0:
                step += 1
            curr += direction * step
    return np.sort(idx)


def _band(ax, x, mean, std, color, alpha):
    ax.fill_between(x, np.maximum(mean - std, 1e-30), mean + std, color=color, alpha=alpha, lw=0)


def _in_window(t, window):
    t = np.asarray(t)
    return (t >= window[0] - 1e-6) & (t <= window[1] + 1e-6)


def _legend_below(fig, ax):
    handles, labels = ax.get_legend_handles_labels()
    ncol = len(labels) if len(labels) <= 4 else (len(labels) + 1) // 2
    fig.legend(handles, labels, loc="outside lower center", ncol=ncol, frameon=False, fontsize=10)


def _load_reference(reference):
    if isinstance(reference, (str, os.PathLike)):
        with open(reference) as fh:
            return json.load(fh)
    return reference


def _load_run(run):
    """Fluxes and spectra of a run directory; a run that is already loaded is returned as is."""
    if not isinstance(run, (str, os.PathLike)):
        return run
    with np.load(os.path.join(run, "fluxes.npz")) as f:
        out = {"time": f["time"], "fluxes": f["fluxes"]}
    for name, key in (("kyspec", "ky_spec"), ("kxspec", "kx_spec")):
        path = os.path.join(run, f"{name}.npz")
        if os.path.exists(path):
            with np.load(path) as f:
                out[key], out["spec_time"] = f[key], f["time"]
    return out


def plot_run_fluxes(
    runs: Mapping[str, Union[str, os.PathLike, dict]],
    reference: Optional[Union[str, os.PathLike, dict]] = None,
    window: Optional[Sequence[float]] = None,
    channels: Sequence[str] = FLUX_CHANNELS,
    title: str = "",
    reference_label: str = "reference",
    colors: Optional[Sequence[str]] = None,
) -> plt.Figure:
    """Flux traces of runs against a reference: one row per channel, one column per species.

    Args:
        runs: Legend label to a ``gyaradax run`` output directory, or to a loaded run with
            ``time`` (T,) and ``fluxes`` (T, 3) or (T, n_species, 3).
        reference: Reference diagnostics, a ``nonlinear.json`` of gerkone/gyaradax-tests as a
            dict or a path, drawn dashed in black. Its ``species`` name the columns.
        window: Time window shaded on every panel; the reference ``average.window`` when None.
        channels: Rows, from ``"particle"``, ``"heat"`` and ``"momentum"``.
        title: Figure title.
        reference_label: Legend label of the reference.
        colors: Line colors in the order of ``runs``; ``RUN_COLORS`` when None.

    Returns:
        The figure.
    """
    ref = _load_reference(reference)
    colors = RUN_COLORS if colors is None else colors
    loaded = {label: _load_run(run) for label, run in runs.items()}
    fluxes = {
        label: np.asarray(r["fluxes"]).reshape(len(r["time"]), -1, 3) for label, r in loaded.items()
    }
    if ref is not None:
        species = ref["species"]
        window = ref["average"]["window"] if window is None else window
    else:
        species = [
            SPECIES_LABELS.get(i, f"sp{i}") for i in range(max(f.shape[1] for f in fluxes.values()))
        ]

    fig, axes = plt.subplots(
        len(channels),
        len(species),
        figsize=(7.0, 1.6 * len(channels) + 0.8),
        sharex=True,
        squeeze=False,
        layout="constrained",
    )
    for isp, sp in enumerate(species):
        for row, channel in enumerate(channels):
            ax = axes[row, isp]
            ich = FLUX_CHANNELS.index(channel)
            if window is not None:
                ax.axvspan(*window, color="0.93", lw=0, zorder=0)
            for color, (label, run) in zip(colors, loaded.items()):
                if isp < fluxes[label].shape[1]:
                    ax.plot(
                        run["time"],
                        fluxes[label][:, isp, ich],
                        color=color,
                        lw=1.0,
                        label=label,
                        zorder=2,
                    )
            if ref is not None:
                ax.plot(
                    ref["time"],
                    ref["fluxes"][sp][channel],
                    color="black",
                    linestyle="--",
                    lw=1.2,
                    alpha=0.8,
                    label=reference_label,
                    zorder=3,
                )
            ax.grid(True, axis="y")
            if isp == 0:
                ax.set_ylabel(channel.capitalize(), fontsize=12)
            if row == 0 and len(species) > 1:
                ax.set_title(sp, fontsize=9)
        axes[-1, isp].set_xlabel(r"time $[R/v_{th}]$", fontsize=12)

    _legend_below(fig, axes[0, 0])
    if title:
        fig.suptitle(title, fontweight="bold", fontsize=14)
    fig.align_ylabels(axes[:, 0])
    return fig


def plot_run_spectra(
    runs: Mapping[str, Union[str, os.PathLike, dict, Sequence[Union[str, os.PathLike, dict]]]],
    reference: Optional[Union[str, os.PathLike, dict]] = None,
    window: Optional[Sequence[float]] = None,
    ky: Optional[np.ndarray] = None,
    kx: Optional[np.ndarray] = None,
    title: str = "",
    reference_label: str = "reference",
    shade_alpha: float = 0.2,
    colors: Optional[Sequence[str]] = None,
) -> plt.Figure:
    """Time-averaged ky and kx spectra of one or more runs against a reference.

    Each spectrum is the mean over the samples in the averaging window, with their standard
    deviation as a band.

    Args:
        runs: Legend label to a ``gyaradax run`` output directory, a loaded run with ``time``
            (T,), ``ky_spec`` (T, nky) and ``kx_spec`` (T, nkx), or a list of them whose
            samples are pooled (e.g. realisations of one case).
        reference: Reference diagnostics, a ``nonlinear.json`` of gerkone/gyaradax-tests as a
            dict or a path, averaged over its own ``average.window``.
        window: Averaging window of the runs; the reference ``average.window`` when None.
        ky: Binormal wavenumbers; the reference ``ky`` when None.
        kx: Radial wavenumbers; the reference ``kx`` when None.
        title: Figure title.
        reference_label: Legend label of the reference.
        shade_alpha: Opacity of the bands.
        colors: Line colors in the order of ``runs``; ``RUN_COLORS`` when None.

    Returns:
        The figure.
    """
    ref = _load_reference(reference)
    colors = RUN_COLORS if colors is None else colors
    if ref is not None:
        window = ref["average"]["window"] if window is None else window
        ky = ref["ky"] if ky is None else ky
        kx = ref["kx"] if kx is None else kx
    if window is None or ky is None or kx is None:
        raise ValueError("without a reference, pass window, ky and kx")
    order = {"ky": np.argsort(ky), "kx": np.argsort(kx)}
    grids = {"ky": np.asarray(ky)[order["ky"]], "kx": np.asarray(kx)[order["kx"]]}

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.8), layout="constrained")
    panels = list(zip(axes, ("ky", "kx")))
    bands = []
    for color, (label, run) in zip(colors, runs.items()):
        group = [_load_run(r) for r in (run if isinstance(run, (list, tuple)) else [run])]
        for ax, name in panels:
            samples = np.concatenate(
                [
                    np.asarray(r[f"{name}_spec"])[_in_window(r.get("spec_time", r["time"]), window)]
                    for r in group
                ]
            )
            if len(samples) == 0:
                raise ValueError(f"{label}: no spectra in the window {window}")
            mean, std = samples.mean(0)[order[name]], samples.std(0)[order[name]]
            ax.semilogy(grids[name], mean, "o-", color=color, markersize=3, lw=1, label=label)
            bands.append((ax, grids[name], mean, std, color))
    if ref is not None:
        mask = _in_window(ref["time"], ref["average"]["window"])
        for ax, name in panels:
            samples = np.asarray(ref[f"{name}_spectrum"])[mask]
            mean, std = samples.mean(0)[order[name]], samples.std(0)[order[name]]
            idx = np.arange(len(mean)) if name == "ky" else _progressive_sparse_indices(mean)
            ax.semilogy(
                grids[name][idx],
                mean[idx],
                marker="x",
                linestyle="None",
                color="black",
                markersize=4,
                markeredgewidth=1,
                label=reference_label,
            )
            bands.append((ax, grids[name], mean, std, "black"))
    # bands that reach zero do not set the log range
    for ax in axes:
        ax.set_ylim(ax.get_ylim())
    for ax, x, mean, std, color in bands:
        _band(ax, x, mean, std, color, shade_alpha)
    for ax, name in panels:
        k = f"k_{name[1]}"
        ax.set_xlabel(rf"${k} \rho_{{ref}}$", fontsize=12)
        ax.set_ylabel(rf"${k}^{{\text{{spec}}}}$", fontsize=12)
        ax.set_title(rf"${k}$ spectrum", fontsize=12)
        ax.grid(True, which="both")

    _legend_below(fig, axes[0])
    if title:
        fig.suptitle(title, fontweight="bold", fontsize=14)
    return fig


def plot_growth_rates(
    time: np.ndarray,
    growth: np.ndarray,
    ky: Optional[np.ndarray] = None,
    ref_time: Optional[np.ndarray] = None,
    ref_growth: Optional[np.ndarray] = None,
    title: str = "",
    max_modes: int = 8,
) -> plt.Figure:
    """Plot per-ky growth rates over time, optionally against a GKW reference.

    Args:
        time: Window times, (n_windows,).
        growth: Growth rates, (n_windows,) or (n_windows, nky).
        ky: Wavenumbers for the legend, (nky,).
        ref_time: GKW ``time.dat`` columns ``[time, growth, ...]``.
        ref_growth: GKW growth rates; column 1 of ``ref_time`` when None.
        title: Figure title.
        max_modes: Most ky modes drawn individually.

    Returns:
        The figure.
    """
    fig, ax = plt.subplots(figsize=(7.0, 2.8))

    if growth.ndim == 2:
        nky = growth.shape[1]
        colors = plt.cm.viridis(np.linspace(0.15, 0.85, min(nky, max_modes)))
        for iy in range(min(nky, max_modes)):
            label = rf"$k_y={float(ky[iy]):.2f}$" if ky is not None and iy < len(ky) else f"ky={iy}"
            ax.plot(time, growth[:, iy], lw=1, color=colors[iy], alpha=0.8, label=label)
    else:
        ax.plot(time, growth, "-", color=JAX_COLORS["blue"], lw=1.5, label="gyaradax")

    if ref_growth is not None:
        if ref_growth.ndim == 1:
            ax.plot(
                ref_time if ref_time is not None else np.arange(len(ref_growth)),
                ref_growth,
                "kx",
                ms=4,
                alpha=0.7,
                label="GKW",
            )
    elif ref_time is not None and ref_time.ndim == 2 and ref_time.shape[1] >= 2:
        ax.plot(ref_time[:, 0], ref_time[:, 1], "kx", ms=4, alpha=0.7, label="GKW (mean)")

    ax.set_xlabel(r"time $[R/v_{th}]$", fontsize=12)
    ax.set_ylabel(r"$\gamma$", fontsize=12)
    ax.legend(frameon=False, ncol=3, fontsize=10)
    ax.grid(True)
    if title:
        ax.set_title(title, fontsize=14)
    fig.tight_layout()
    return fig


def plot_growth_snapshots(
    ky: np.ndarray,
    sim_growth: np.ndarray,
    sim_time: np.ndarray,
    ref_growth: Optional[np.ndarray] = None,
    ref_time: Optional[np.ndarray] = None,
    title: str = "",
) -> plt.Figure:
    """Growth-rate spectra at window 5, at 10 % and 50 % of the run, and averaged over time.

    Args:
        ky: Wavenumbers, (nky,).
        sim_growth: Growth rates, (n_windows, nky).
        sim_time: Window times.
        ref_growth: Reference growth rates, or None.
        ref_time: Reference times, or None.
        title: Figure title.

    Returns:
        The figure.
    """
    n_total = len(sim_time)

    # detect saturation: mean growth over ky>0 modes drops near zero
    sat_idx = n_total // 10
    sat_idx = max(2, min(sat_idx, n_total - 3))

    # panels: early linear, saturation onset, half-run, time-average
    snap_indices = [
        5,
        sat_idx,
        n_total // 2,
    ]
    snap_indices = [min(i, n_total - 1) for i in snap_indices]

    fig, axes = plt.subplots(2, 2, figsize=(7.0, 4.2), sharex=True, sharey=True)
    axes = axes.flatten()

    for i, idx in enumerate(snap_indices):
        ax = axes[i]
        t_sim = sim_time[idx]

        if ref_growth is not None and ref_time is not None:
            ref_idx = np.argmin(np.abs(ref_time - t_sim))
            ax.plot(
                ky[: ref_growth.shape[1]],
                ref_growth[ref_idx],
                "kx",
                ms=5,
                alpha=0.9,
                label="GKW",
                zorder=100,
            )

        ax.plot(
            ky,
            sim_growth[idx],
            "o-",
            color=JAX_COLORS["blue"],
            ms=3,
            lw=1.2,
            label="gyaradax",
        )

        ax.set_title(rf"$t = {t_sim:.1f}$", fontsize=12)
        ax.grid(True)
        if i == 0:
            ax.legend(frameon=False, fontsize=10)
        if i >= 2:
            ax.set_xlabel(r"$k_y \rho_{ref}$", fontsize=12)
        if i % 2 == 0:
            ax.set_ylabel(r"$\gamma$", fontsize=12)

    # 4th panel: time-averaged growth rate
    ax = axes[3]
    sim_avg = np.mean(sim_growth, axis=0)
    if ref_growth is not None:
        ref_avg = np.mean(ref_growth, axis=0)
        ax.plot(ky[: len(ref_avg)], ref_avg, "kx", ms=5, alpha=0.9, label="GKW", zorder=100)
    ax.plot(ky, sim_avg, "o-", color=JAX_COLORS["blue"], ms=3, lw=1.2, label="gyaradax")
    ax.set_title("Time average", fontsize=12)
    ax.set_xlabel(r"$k_y \rho_{ref}$", fontsize=12)
    ax.grid(True)

    if title:
        fig.suptitle(title, fontweight="bold", fontsize=14)
    fig.tight_layout()
    return fig


def force_aspect(ax: plt.Axes, aspect: float = 1.0):
    """Set the axes aspect ratio from the extent of its first image.

    Args:
        ax: Matplotlib axes.
        aspect: Target width-to-height ratio.
    """
    im = ax.get_images()
    if not im:
        return
    extent = im[0].get_extent()
    ax.set_aspect(abs((extent[1] - extent[0]) / (extent[3] - extent[2])) / aspect)


def plot_nd(
    x: np.ndarray,
    y: Optional[np.ndarray] = None,
    labels: Optional[List[str]] = None,
    cmap: str = "RdBu_r",
    aggregate: str = "mean",
    aspect: float = 1.0,
    mark_bad: bool = False,
    **kwargs,
):
    """Grid of 2D slices for every pair of dimensions; ``y`` is drawn next to ``x``.

    Args:
        x: N-dimensional array.
        y: Second array of the same shape, or None.
        labels: Dimension names.
        cmap: Colormap.
        aggregate: Reduction over the other dimensions: ``"mean"``, ``"std"`` or
            ``"slice"`` (middle index).
        aspect: Panel aspect ratio.
        mark_bad: Blank cells whose values do not vary.
        kwargs: Accepted and ignored.

    Returns:
        The figure.
    """
    if labels is not None:
        ndim = len(labels)
        has_channel = x.ndim > ndim
    else:
        if x.ndim in [5, 6]:
            ndim = x.ndim - 1
            has_channel = True
        else:
            ndim = x.ndim
            has_channel = False

    if ndim == 0:
        return None

    if labels is None:
        labels = GK_LABELS.get(ndim, [f"d_{i}" for i in range(ndim)])

    comb = []
    for i in range(ndim):
        for j in range(i + 1, ndim):
            comb.append([i, j])

    fig, axes = plt.subplots(
        ndim,
        ndim,
        figsize=(ndim * (3.5 if y is not None else 2), ndim * 1.8),
        squeeze=False,
    )

    c_map = matplotlib.colormaps[cmap].copy()
    c_map.set_bad("gray")

    for i in range(ndim):
        for j in range(ndim):
            ax = axes[i, j]
            if [i, j] not in comb:
                ax.remove()
                continue

            other_dims = tuple(o for o in range(ndim) if o != i and o != j)

            def get_2d_slice(data):
                d = data.sum(0) if has_channel and data.ndim > ndim else data
                if aggregate == "mean":
                    res = d.mean(axis=other_dims)
                elif aggregate == "std":
                    res = d.std(axis=other_dims)
                elif aggregate == "slice":
                    slices = [slice(None)] * ndim
                    for o in other_dims:
                        slices[o] = d.shape[o] // 2
                    res = d[tuple(slices)]
                else:
                    res = d.mean(axis=other_dims)

                if mark_bad:
                    s = d.std(axis=other_dims)
                    res = np.where(s == 0, np.nan, res)
                return res

            xx = get_2d_slice(np.asarray(x))
            if np.iscomplexobj(xx):
                xx = xx.real

            if y is not None:
                yy = get_2d_slice(np.asarray(y))
                if np.iscomplexobj(yy):
                    yy = yy.real
                vmin = min(np.nanmin(xx), np.nanmin(yy))
                vmax = max(np.nanmax(xx), np.nanmax(yy))

                spacer = np.full((xx.shape[0], max(1, xx.shape[1] // 15)), np.nan)
                display_img = np.concatenate([xx, spacer, yy], axis=1)
                ax.matshow(display_img, cmap=c_map, vmin=vmin, vmax=vmax)
            else:
                ax.matshow(xx, cmap=c_map)

            if j == i + 1:
                ax.set_ylabel(rf"${labels[i]}$", labelpad=2)
            if i == j - 1:
                ax.set_xlabel(rf"${labels[j]}$", labelpad=2)

            ax.set_xticks([])
            ax.set_yticks([])
            force_aspect(ax, aspect=aspect * (2.1 if y is not None else 1.0))

    plt.subplots_adjust(left=0.01, right=0.99, bottom=0.01, top=0.99, wspace=0, hspace=0)
    return fig

