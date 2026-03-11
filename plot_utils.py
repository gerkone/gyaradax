"""Qualitative visualization functions for n-dimensional gyrokinetics data in JAX."""

import io
from typing import Dict, List, Optional, Sequence, Union

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import jax.numpy as jnp
from PIL import Image as PILImage

# Standard labels for gyrokinetics dimensions in this JAX port: (vpar, mu, s, kx, ky)
GK_LABELS = {
    6: [r"t", r"v_{\parallel}", r"\mu", r"s", r"k_x", r"k_y"],
    5: [r"v_{\parallel}", r"\mu", r"s", r"k_x", r"k_y"],
    4: [r"v_{\parallel}", r"s", r"k_x", r"k_y"],
    3: [r"s", r"k_x", r"k_y"],
}


def force_aspect(ax: plt.Axes, aspect: float = 1.0):
    """Adjust axis aspect ratio based on image extent."""
    im = ax.get_images()
    if not im:
        return
    extent = im[0].get_extent()
    ax.set_aspect(abs((extent[1] - extent[0]) / (extent[3] - extent[2])) / aspect)


def plot_nd(
    x: Union[np.ndarray, jnp.ndarray],
    y: Optional[Union[np.ndarray, jnp.ndarray]] = None,
    labels: Optional[List[str]] = None,
    cmap: str = "RdBu_r",
    aggregate: str = "mean",
    aspect: float = 1.0,
    mark_bad: bool = False,
    title: Optional[str] = None,
    **kwargs,
):
    """
    Generic n-dimensional plotting function for JAX/NumPy.
    Creates a grid of 2D slices for all combinations of dimensions.
    If 'y' is provided, shows side-by-side comparison.
    """
    if hasattr(x, "device_buffer"):  # Check for JAX array
        x = np.array(x)
    if y is not None and hasattr(y, "device_buffer"):
        y = np.array(y)

    # Handle complex data by taking magnitude
    if np.iscomplexobj(x):
        x = np.abs(x)
    if y is not None and np.iscomplexobj(y):
        y = np.abs(y)

    ndim = x.ndim
    if ndim == 0:
        return None

    if labels is None:
        labels = GK_LABELS.get(ndim, [f"d_{i}" for i in range(ndim)])

    # Get all pairs of dimensions
    comb_list = []
    for i in range(ndim):
        for j in range(i + 1, ndim):
            comb_list.append([i, j])

    fig, axes = plt.subplots(
        ndim,
        ndim,
        figsize=(ndim * (3.5 if y is not None else 2), ndim * 1.8),
        squeeze=False,
    )

    if title:
        fig.suptitle(title, fontsize=16)

    c_map = matplotlib.colormaps[cmap].copy()
    c_map.set_bad("gray")

    for i in range(ndim):
        for j in range(ndim):
            ax = axes[i, j]
            if [i, j] not in comb_list:
                ax.remove()
                continue

            other_dims = tuple(o for o in range(ndim) if o != i and o != j)

            def get_2d_slice(data):
                # mean/std/slice over the 'other' spatial dims
                if aggregate == "mean":
                    res = data.mean(axis=other_dims)
                elif aggregate == "std":
                    res = data.std(axis=other_dims)
                elif aggregate == "slice":
                    slices = [slice(None)] * ndim
                    for o in other_dims:
                        slices[o] = data.shape[o] // 2
                    res = data[tuple(slices)]
                else:
                    res = data.mean(axis=other_dims)

                if mark_bad:
                    s = data.std(axis=other_dims)
                    res = np.where(s == 0, np.nan, res)
                return res

            xx = get_2d_slice(x)

            if y is not None:
                yy = get_2d_slice(y)
                vmin = min(np.nanmin(xx), np.nanmin(yy))
                vmax = max(np.nanmax(xx), np.nanmax(yy))

                spacer = np.full((xx.shape[0], max(1, xx.shape[1] // 15)), np.nan)
                display_img = np.concatenate([xx, spacer, yy], axis=1)
                im = ax.matshow(display_img, cmap=c_map, vmin=vmin, vmax=vmax)
            else:
                im = ax.matshow(xx, cmap=c_map)

            # Y-label on the first plot of each row
            if j == i + 1 or (ndim == 2 and j == 1):
                ax.set_ylabel(rf"${labels[i]}$", fontsize=14, labelpad=2)
            # X-label on the bottom plots
            if i == j - 1 or (ndim == 2 and i == 0):
                ax.set_xlabel(rf"${labels[j]}$", fontsize=14, labelpad=2)

            ax.set_xticks([])
            ax.set_yticks([])

            force_aspect(ax, aspect=aspect * (2.1 if y is not None else 1.0))

    plt.subplots_adjust(
        left=0.05,
        right=0.95,
        bottom=0.05,
        top=0.90 if title else 0.95,
        wspace=0.1,
        hspace=0.1,
    )

    return fig
