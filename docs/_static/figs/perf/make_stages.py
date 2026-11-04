"""Speed-up of the optimisation stages on one B300, as a waterfall per grid."""

import os

import matplotlib.pyplot as plt

OUT = os.path.dirname(os.path.abspath(__file__))
GAIN, LOSS, LINK, INK, MUTED = "#0d5940", "#9a9a9a", "#c8c8c8", "#1f1f1f", "#5f5f5f"
plt.rcParams["font.family"] = "DejaVu Sans"

STAGES = [
    "JAX, FP64", "+ mixed precision", "+ Z2Z packing", "+ CUDA, v5 bracket", "+ v6 bracket",
    "+ kx padding", "+ fused_lin", "+ ds", "+ fused_nl",
]
STEPS = {
    "adiabatic (32, 8, 16, 85, 32)": [10.29, 31.18, 47.07, 71.33, 81.39, 85.58, 100.51, 137.67, 136.77],
    "kinetic (2, 32, 12, 18, 192, 32)": [1.18, 4.22, 5.24, 7.98, 5.50, 9.35, 13.02, 17.80, 20.18],
}


def waterfall(ax, title, steps):
    sp = [s / steps[0] for s in steps]
    y = list(range(len(sp)))[::-1]
    for i, (yi, v) in enumerate(zip(y, sp)):
        lo = sp[i - 1] if i else 0.0
        color = GAIN if v >= lo else LOSS
        ax.barh(yi, v - lo, left=lo, height=0.62, color=color, edgecolor="white", linewidth=2)
        if i:
            ax.plot([lo, lo], [yi + 1 - 0.31, yi + 0.31], color=LINK, lw=1, zorder=0)
        label = "baseline" if i == 0 else f"{v / lo:.2f}x"
        if i == len(sp) - 1:
            label += f"  ({v:.1f}x total)"
        ax.text(max(v, lo) + sp[-1] * 0.015, yi, label, va="center", ha="left", fontsize=10, color=INK)
    ax.set_yticks(y, STAGES, fontsize=10.5, color=INK)
    ax.set_xlim(0, max(sp) * 1.3)
    ax.set_xlabel("speed-up over JAX, FP64", fontsize=10.5, color=MUTED)
    ax.set_title(title, fontsize=12, color=INK, loc="left")
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(LINK)
    ax.tick_params(axis="y", length=0)
    ax.tick_params(axis="x", colors=MUTED, labelsize=9.5)
    ax.grid(axis="x", color="#ececec", lw=0.8)
    ax.set_axisbelow(True)


fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
for ax, (title, steps) in zip(axes, STEPS.items()):
    waterfall(ax, title, steps)
fig.tight_layout(w_pad=3)
fig.savefig(os.path.join(OUT, "stages.png"), dpi=160)
