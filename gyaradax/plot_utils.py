"""Publication-quality visualization for gyaradax gyrokinetics data."""

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

SPECIES_COLORS = [JAX_COLORS["cyan"], JAX_COLORS["purple"]]

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica", "Arial", "Liberation Sans"],
        "font.size": 8,
        "axes.labelsize": 9,
        "axes.titlesize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "figure.titlesize": 10,
        "lines.linewidth": 1.25,
        "grid.alpha": 0.15,
        "grid.linestyle": ":",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "figure.figsize": (3.5, 2.8),
    }
)


def plot_flux_trace(
    time: np.ndarray,
    fluxes: Union[np.ndarray, Tuple[np.ndarray, ...]],
    labels: List[str] = ["Particle", "Heat", "Momentum"],
    ref_time: Optional[np.ndarray] = None,
    ref_fluxes: Optional[Union[np.ndarray, Tuple[np.ndarray, ...]]] = None,
    title: str = "Flux Evolution",
    show_average: bool = False,
    avg_window: int = 80,
    n_species: int = 1,
    species_labels: Optional[List[str]] = None,
) -> plt.Figure:
    """Plot flux traces over time with optional multi-species side-by-side layout.

    For n_species > 1, creates two columns (one per species). Fluxes should
    have shape (n_species * n_flux, n_time) with species interleaved:
    [pflux_i, eflux_i, vflux_i, pflux_e, eflux_e, vflux_e, ...].
    """
    if isinstance(fluxes, tuple):
        fluxes = np.stack(fluxes)
    if ref_fluxes is not None and isinstance(ref_fluxes, tuple):
        ref_fluxes = np.stack(ref_fluxes)

    # if species_labels is None:
    #     species_labels = [SPECIES_LABELS.get(i, f"sp{i}") for i in range(n_species)]

    n_flux = len(labels)
    ncols = n_species
    fig, axes = plt.subplots(
        n_flux, ncols, figsize=(6 * ncols, 1.5 * n_flux), sharex=True, squeeze=False
    )

    for isp in range(n_species):
        col_offset = isp * n_flux
        color = SPECIES_COLORS[isp % len(SPECIES_COLORS)]

        for i in range(n_flux):
            ax = axes[i, isp]
            flux_idx = col_offset + i
            if flux_idx >= fluxes.shape[0]:
                continue

            ax.plot(time, fluxes[flux_idx], label="gyaradax", color=color, lw=1.5)

            if show_average and len(fluxes[flux_idx]) >= avg_window:
                avg_val = np.mean(fluxes[flux_idx][-avg_window:])
                ax.axhline(
                    avg_val,
                    color=JAX_COLORS["red"],
                    linestyle=":",
                    lw=2.0,
                    label=f"avg (last {avg_window})",
                    zorder=-1,
                )

            if ref_fluxes is not None and ref_time is not None:
                ref_idx = col_offset + i
                if ref_idx < ref_fluxes.shape[0]:
                    ax.plot(
                        ref_time,
                        ref_fluxes[ref_idx],
                        color="black",
                        linestyle="--",
                        label="GKW",
                        alpha=0.8,
                        lw=1.4,
                        zorder=0,
                    )

            if isp == 0:
                ax.set_ylabel(labels[i], fontsize=12)
            ax.grid(True, axis="y")
            if i == 0:
                # ax.set_title(species_labels[isp])
                ax.legend(frameon=False, loc="best")

        axes[-1, isp].set_xlabel(r"time $[v_{th}/R]$", fontsize=12)

    fig.suptitle(title, fontweight="bold")
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
    title: str = "",
) -> plt.Figure:
    """Plot radial and binormal spectra. Accepts either phi or pre-computed 1D spectra."""
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

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.0, 2.8))

    ax1.semilogy(
        ky,
        ky_spec,
        "o-",
        color=JAX_COLORS["purple"],
        markersize=3,
        lw=1,
        label="gyaradax",
    )
    if ref_ky_spec is not None:
        ax1.semilogy(
            ky, ref_ky_spec, "x--", color="black", markersize=4, lw=1, label="GKW"
        )
    ax1.set_xlabel(r"$k_y \rho_{ref}$", fontsize=12)
    ax1.set_ylabel(r"$\sum_{s, k_x} |\phi|^2$", fontsize=10)
    ax1.set_title(r"$k_y^{\text{spec}}$", fontsize=16)
    ax1.grid(True, which="both")
    ax1.legend()

    ax2.semilogy(kx, kx_spec, "o-", color=JAX_COLORS["purple"], markersize=3, lw=1)
    if ref_kx_spec is not None:
        ax2.semilogy(kx, ref_kx_spec, "x--", color="black", markersize=4, lw=1)
    ax2.set_xlabel(r"$k_x \rho_{ref}$", fontsize=12)
    ax2.set_ylabel(r"$\sum_{s, k_y} |\phi|^2$", fontsize=10)
    ax2.set_title(r"$k_x^{\text{spec}}$", fontsize=16)
    ax2.grid(True, which="both")
    if len(title) > 0:
        fig.suptitle(title, fontweight="bold")
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
                ax.set_ylabel(rf"${labels[i]}$", fontsize=12, labelpad=2)
            if i == j - 1:
                ax.set_xlabel(rf"${labels[j]}$", fontsize=12, labelpad=2)

            ax.set_xticks([])
            ax.set_yticks([])
            force_aspect(ax, aspect=aspect * (2.1 if y is not None else 1.0))

    plt.subplots_adjust(
        left=0.01, right=0.99, bottom=0.01, top=0.99, wspace=0, hspace=0
    )
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
