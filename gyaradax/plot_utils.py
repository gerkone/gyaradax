"""Publication-quality visualization functions for gyrokinetics data."""

from typing import List, Optional, Union, Tuple
import matplotlib.pyplot as plt
import numpy as np
import jax.numpy as jnp

# Standard styling for scientific plots
plt.rcParams.update(
    {
        "font.size": 12,
        "axes.labelsize": 14,
        "axes.titlesize": 14,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 12,
        "figure.titlesize": 16,
        "lines.linewidth": 2,
        "grid.alpha": 0.3,
    }
)


def force_aspect(ax: plt.Axes, aspect: float = 1.0):
    """Adjust axis aspect ratio based on image extent."""
    im = ax.get_images()
    if not im:
        return
    extent = im[0].get_extent()
    ax.set_aspect(abs((extent[1] - extent[0]) / (extent[3] - extent[2])) / aspect)


def plot_flux_trace(
    time: np.ndarray,
    fluxes: Union[np.ndarray, Tuple[np.ndarray, ...]],
    labels: List[str] = ["Particle", "Heat", "Momentum"],
    ref_time: Optional[np.ndarray] = None,
    ref_fluxes: Optional[Union[np.ndarray, Tuple[np.ndarray, ...]]] = None,
    title: str = "Flux Evolution",
) -> plt.Figure:
    """Plot flux traces over time with optional reference comparison."""
    if isinstance(fluxes, tuple):
        fluxes = np.stack(fluxes)
    if ref_fluxes is not None and isinstance(ref_fluxes, tuple):
        ref_fluxes = np.stack(ref_fluxes)

    n_flux = fluxes.shape[0]
    fig, axes = plt.subplots(n_flux, 1, figsize=(10, 3 * n_flux), sharex=True)
    if n_flux == 1:
        axes = [axes]

    for i in range(n_flux):
        ax = axes[i]
        ax.plot(time, fluxes[i], label="Gyaradax", color="tab:blue")
        if ref_fluxes is not None and ref_time is not None:
            ax.plot(ref_time, ref_fluxes[i], "k--", label="Reference", alpha=0.7)

        ax.set_ylabel(labels[i])
        ax.grid(True)
        if i == 0:
            ax.legend()

    axes[-1].set_xlabel("Time $[v_{th}/R]$")
    fig.suptitle(title)
    fig.tight_layout()
    return fig


def plot_spectra(
    kx: np.ndarray,
    ky: np.ndarray,
    phi: jnp.ndarray,
    title: str = "Potential Spectra",
) -> plt.Figure:
    """Plot kx and ky spectra summed/averaged over other dimensions."""
    phi_sq = np.abs(np.array(phi)) ** 2
    # phi is [ns, nkx, nky]
    kx_spec = np.sum(phi_sq, axis=(0, 2))
    ky_spec = np.sum(phi_sq, axis=(0, 1))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.semilogy(ky, ky_spec, "o-", color="tab:red")
    ax1.set_xlabel(r"$k_y \rho_{ref}$")
    ax1.set_ylabel(r"$\sum_{s, k_x} |\phi|^2$")
    ax1.set_title("$k_y$ Spectrum")
    ax1.grid(True)

    ax2.semilogy(kx, kx_spec, "o-", color="tab:green")
    ax2.set_xlabel(r"$k_x \rho_{ref}$")
    ax2.set_ylabel(r"$\sum_{s, k_y} |\phi|^2$")
    ax2.set_title("$k_x$ Spectrum")
    ax2.grid(True)

    fig.suptitle(title)
    fig.tight_layout()
    return fig


def plot_mode_growth(
    time: np.ndarray,
    phi_hist: np.ndarray,
    ky_indices: List[int],
    ky_values: np.ndarray,
    ds: float = 1.0,
    title: str = "Mode Growth Analysis",
) -> plt.Figure:
    """Plot log-amplitude of specific ky modes over time."""
    # phi_hist: [ntime, ns, nkx, nky]
    fig, ax = plt.subplots(figsize=(10, 6))

    for idx in ky_indices:
        # L2 amplitude over (s, kx) for this ky
        amp = np.sqrt(ds * np.sum(np.abs(phi_hist[..., idx]) ** 2, axis=(1, 2)))
        ax.plot(
            time,
            np.log(np.maximum(amp, 1e-20)),
            label=fr"$k_y \rho = {ky_values[idx]:.3f}$",
        )

    ax.set_xlabel("Time $[v_{th}/R]$")
    ax.set_ylabel(r"$\log(\mathrm{Amplitude})$")
    ax.set_title(title)
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    ax.grid(True)
    fig.tight_layout()
    return fig


def plot_nd(
    x: Union[np.ndarray, jnp.ndarray],
    labels: Optional[List[str]] = None,
    cmap: str = "RdBu_r",
    title: Optional[str] = None,
):
    """Generic n-dimensional slice visualizer."""
    x = np.abs(np.array(x))
    ndim = x.ndim
    if labels is None:
        labels = [f"d_{i}" for i in range(ndim)]

    fig, axes = plt.subplots(1, ndim, figsize=(3 * ndim, 3))
    if ndim == 1:
        axes = [axes]

    for i in range(ndim):
        other_dims = tuple(o for o in range(ndim) if o != i)
        axes[i].plot(x.mean(axis=other_dims))
        axes[i].set_title(f"Mean vs {labels[i]}")

    if title:
        fig.suptitle(title)
    fig.tight_layout()
    return fig
