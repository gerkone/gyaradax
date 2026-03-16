"""Publication-quality visualization functions for gyaradax gyrokinetics data."""

from typing import List, Optional, Union, Tuple
import matplotlib.pyplot as plt
import numpy as np
import jax.numpy as jnp

# Updated JAX/Google brand color scheme from image
JAX_COLORS = {
    "blue": "#4285F4",   # Google Blue
    "red": "#EA4335",    # Google Red
    "yellow": "#FBBC05", # Google Yellow
    "green": "#34A853",  # Google Green
    "cyan": "#24B6AD",   # Teal/Cyan from JAX logo image
    "purple": "#9B51E0", # Purple from JAX logo image
}

# Strict "Nature-ready" styling for scientific plots
plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica", "Arial", "Liberation Sans"],
        "font.size": 8,            # Standard Nature font size
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
        "savefig.dpi": 300,        # High resolution
        "savefig.bbox": "tight",
        "figure.figsize": (3.5, 2.8), # Single-column width (~89mm)
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
) -> plt.Figure:
    """Plot flux traces over time with GKW reference comparison and optional averaging."""
    if isinstance(fluxes, tuple):
        fluxes = np.stack(fluxes)
    if ref_fluxes is not None and isinstance(ref_fluxes, tuple):
        ref_fluxes = np.stack(ref_fluxes)

    n_flux = fluxes.shape[0]
    fig, axes = plt.subplots(n_flux, 1, figsize=(6, 1.5 * n_flux), sharex=True)
    if n_flux == 1:
        axes = [axes]

    colors = [JAX_COLORS["green"]]

    for i in range(n_flux):
        ax = axes[i]
        # Main trace
        ax.plot(time, fluxes[i], label="gyaradax", color=colors[i % len(colors)], lw=1.5)
        
        # Optional average line for the last N timesteps
        if show_average and len(fluxes[i]) >= avg_window:
            avg_val = np.mean(fluxes[i][-avg_window:])
            # Plotting from the start of the window to the end of time
            ax.axhline(
                avg_val, 
                color=JAX_COLORS["red"], 
                linestyle=":", 
                lw=2.0, 
                label=f"Avg (last {avg_window})",
                zorder=-1
            )
            # Optional: add text label for the value
            ax.text(
                time[-1], avg_val, f"{avg_val:.2e}", 
                va="bottom", ha="right", color=JAX_COLORS["red"], fontsize=7,
            )

        if ref_fluxes is not None and ref_time is not None:
            ax.plot(
                ref_time, ref_fluxes[i], color="black", linestyle="--", 
                label="GKW", alpha=0.8, lw=1.4, 
                zorder=0
            )

        ax.set_ylabel(labels[i])
        ax.grid(True, axis="y")
        if i == 0:
            ax.legend(frameon=False, loc="best")

    axes[-1].set_xlabel(r"Time $[v_{th}/R]$")
    fig.suptitle(title, fontweight="bold")
    fig.tight_layout()
    return fig


def plot_spectra(
    kx: np.ndarray,
    ky: np.ndarray,
    phi: jnp.ndarray,
    title: str = "Potential Spectra",
) -> plt.Figure:
    """Plot radial and bi-normal spectra with publication styling."""
    phi_sq = np.abs(np.array(phi)) ** 2
    # phi indices: [s, kx, ky]
    kx_spec = np.sum(phi_sq, axis=(0, 2))
    ky_spec = np.sum(phi_sq, axis=(0, 1))

    # Nature single-column width might be tight for side-by-side; 
    # using a slightly wider figure for 1x2 layout
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.0, 2.8))

    ax1.semilogy(ky, ky_spec, "o-", color=JAX_COLORS["red"], markersize=3, lw=1)
    ax1.set_xlabel(r"$k_y \rho_{ref}$")
    ax1.set_ylabel(r"$\sum_{s, k_x} |\phi|^2$")
    ax1.set_title(r"$k_y$ Spectrum")
    ax1.grid(True, which="both")

    ax2.semilogy(kx, kx_spec, "o-", color=JAX_COLORS["cyan"], markersize=3, lw=1)
    ax2.set_xlabel(r"$k_x \rho_{ref}$")
    ax2.set_ylabel(r"$\sum_{s, k_y} |\phi|^2$")
    ax2.set_title(r"$k_x$ Spectrum")
    ax2.grid(True, which="both")

    fig.suptitle(title, fontweight="bold")
    fig.tight_layout()
    return fig

def plot_zonal_residual(
    time: np.ndarray,
    phi_history: np.ndarray,
    target_residual: Optional[float] = None,
) -> plt.Figure:
    """Specific Nature-ready plot for Rosenbluth-Hinton Zonal Flow test."""
    # Normalize potential to t=0
    phi_norm = phi_history / phi_history[0]
    
    fig, ax = plt.subplots()
    ax.plot(time, phi_norm, color=JAX_COLORS["blue"], label="gyaradax")
    
    if target_residual is not None:
        ax.axhline(target_residual, color=JAX_COLORS["red"], linestyle="--", 
                   label=f"Analytical ({target_residual:.3f})")
    
    ax.set_xlabel(r"Normalised Time $[c_s t/R]$")
    ax.set_ylabel(r"$\phi(t)/\phi(0)$")
    ax.set_title("Zonal Flow Damping (Rosenbluth-Hinton)")
    ax.legend(frameon=False)
    ax.grid(True)
    
    return fig