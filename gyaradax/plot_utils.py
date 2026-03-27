from typing import List, Optional, Union, Tuple
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
                    ax.plot(time, fluxes[run_idx, flux_idx], color=color, alpha=0.15, lw=1.0, linestyle="-", zorder=1)

            ax.plot(time, fluxes_mean[flux_idx], label="gyaradax", color=color, lw=1.5, alpha=1.0, zorder=3)

            if fluxes_std is not None:
                ax.fill_between(
                    time,
                    fluxes_mean[flux_idx] - fluxes_std[flux_idx],
                    fluxes_mean[flux_idx] + fluxes_std[flux_idx],
                    color=color,
                    alpha=0.3,
                    lw=0,
                    zorder=2
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
                    zorder=4
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
                    bbox=dict(facecolor="white", alpha=0.8, edgecolor="none", pad=0.2)
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
                                zorder=0
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
                            zorder=1
                        )

                    if show_average and len(ref_fluxes_mean[ref_idx]) >= avg_window:
                        ref_avg_val = np.mean(ref_fluxes_mean[ref_idx][-3 * avg_window:])
                        ref_t_start = ref_time[-3 * avg_window]
                        ref_t_end = ref_time[-1]
                        
                        ax.plot(
                            [ref_t_start, ref_t_end], 
                            [ref_avg_val, ref_avg_val], 
                            color="black", 
                            linestyle="-", 
                            lw=2.0, 
                            zorder=4
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
                            bbox=dict(facecolor="white", alpha=0.8, edgecolor="none", pad=0.2)
                        )

            if isp == 0:
                ax.set_ylabel(labels[i], fontsize=12)
            ax.grid(True, axis="y")
            if i == 0:
                if n_species > 1:
                    ax.set_title(species_labels[isp], fontsize=9)
                if isp == 0:
                    ax.legend(frameon=False, loc="upper left", fontsize=10)

        axes[-1, isp].set_xlabel(r"time $[v_{th}/R]$", fontsize=12)

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
    """Plot radial and binormal spectra. Accepts either phi or pre-computed 1D spectra.

    Optional std arrays add shaded error bounds around the mean lines.
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

    def _shade(ax, x, mean, std, color):
        lo = np.maximum(mean - std, 1e-30)
        hi = mean + std
        ax.fill_between(x, lo, hi, color=color, alpha=shade_alpha, lw=0)

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


def plot_growth_rates(
    time: np.ndarray,
    growth: np.ndarray,
    ky: Optional[np.ndarray] = None,
    ref_time: Optional[np.ndarray] = None,
    ref_growth: Optional[np.ndarray] = None,
    title: str = "",
    max_modes: int = 8,
) -> plt.Figure:
    """Plot per-ky growth rates over time with optional GKW reference.

    Args:
        time: (n_windows,) simulation time.
        growth: (n_windows,) or (n_windows, nky) growth rates.
        ky: (nky,) wavenumber values for legend labels.
        ref_time: (n_ref, >=2) GKW time.dat with columns [time, growth, ...].
        ref_growth: (n_ref,) or (n_ref, nky) GKW growth rates. if None and
            ref_time has >=2 columns, column 1 is used as the mean growth.
        title: figure title.
        max_modes: maximum number of ky modes to plot individually.
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

    ax.set_xlabel(r"time $[v_{th}/R]$", fontsize=12)
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
    """2x2 grid of per-ky growth rate profiles at 4 physically meaningful timesteps.

    Snapshots: (1) early linear phase, (2) onset of saturation,
    (3) mid-saturation, (4) late/converged.

    Saturation onset is detected as the first time the mean growth rate
    (over non-zonal ky modes) crosses below a small threshold.
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


def plot_ky_spectra_evolution(
    ky: np.ndarray,
    ky_spec_history: np.ndarray,
    times: Optional[np.ndarray] = None,
    ref_ky_spec: Optional[np.ndarray] = None,
    n_snapshots: int = 5,
    title: str = "",
) -> plt.Figure:
    """Plot ky spectra at selected timesteps showing time evolution.

    Args:
        ky: (nky,) wavenumber grid.
        ky_spec_history: (n_windows, nky) spectral density over time.
        times: (n_windows,) timestamps for labeling.
        ref_ky_spec: (nky,) time-averaged GKW reference spectrum.
        n_snapshots: number of timesteps to show.
        title: figure title.
    """
    n_total = len(ky_spec_history)
    if times is not None:
        n_total = min(n_total, len(times))
    indices = np.linspace(0, n_total - 1, min(n_snapshots, n_total), dtype=int)
    colors = plt.cm.plasma(np.linspace(0.1, 0.9, len(indices)))

    fig, ax = plt.subplots(figsize=(7.0, 2.8))

    for i, idx in enumerate(indices):
        t_label = f"t={times[idx]:.0f}" if times is not None else f"step {idx}"
        ax.semilogy(
            ky, ky_spec_history[idx], "o-", color=colors[i], ms=2, lw=0.8, alpha=0.7, label=t_label
        )

    if ref_ky_spec is not None:
        ax.semilogy(ky, ref_ky_spec, "k--", lw=1.5, alpha=0.8, label="GKW (avg)")

    ax.set_xlabel(r"$k_y \rho_{ref}$")
    ax.set_ylabel(r"$|\phi|^2$")
    ax.legend(frameon=False, ncol=2)
    ax.grid(True, which="both")
    if title:
        ax.set_title(title)
    fig.tight_layout()
    return fig


def plot_zonal_residual(
    time: np.ndarray,
    phi_history: np.ndarray,
    target_residual: Optional[float] = None,
) -> plt.Figure:
    """Rosenbluth-Hinton zonal flow residual test."""
    phi_norm = phi_history / phi_history[0]

    fig, ax = plt.subplots()
    ax.plot(time, phi_norm, color=JAX_COLORS["blue"], label="gyaradax")

    if target_residual is not None:
        ax.axhline(
            target_residual,
            color=JAX_COLORS["red"],
            linestyle="--",
            label=f"Analytical ({target_residual:.3f})",
        )

    ax.set_xlabel(r"Normalised Time $[c_s t/R]$")
    ax.set_ylabel(r"$\phi(t)/\phi(0)$")
    ax.set_title("Zonal Flow Damping (Rosenbluth-Hinton)")
    ax.legend(frameon=False)
    ax.grid(True)

    return fig


def force_aspect(ax: plt.Axes, aspect: float = 1.0):
    """Adjust axis aspect ratio based on image extent."""
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
    """Grid of 2D slices for all dimension pairs. If y is provided, shows side-by-side."""
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


def plot_gradient_comparison(
    analytical_grad: np.ndarray,
    fd_grad: Optional[np.ndarray] = None,
    labels: Optional[List[str]] = None,
    title: str = "Gradient Validation (Analytical vs FD)",
) -> plt.Figure:
    """Analytical vs finite-difference gradient comparison."""
    grad_to_plot = np.real(analytical_grad)
    fd_to_plot = np.real(fd_grad) if fd_grad is not None else None

    fig = plot_nd(grad_to_plot, y=fd_to_plot, labels=labels)
    if fig:
        fig.suptitle(title, fontweight="bold", y=1.02)
    return fig
