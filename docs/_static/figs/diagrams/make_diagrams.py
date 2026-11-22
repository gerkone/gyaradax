"""Pipeline and grid diagrams of the docs, in the style of the paper figures."""

import os

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

GREEN, GRAY, PURPLE, INNER, ARROW, BRACE, LINE = (
    "#0d5940", "#4b4b4b", "#3f378a", "#5b54b0", "#888888", "#d4a48c", "#e5e7eb"
)
OUT = os.path.dirname(os.path.abspath(__file__))
plt.rcParams["font.family"] = "DejaVu Sans"


def box(ax, x, y, w, h, text, color, size=13, sub=None):
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0,rounding_size=0.1",
                                fc=color, ec="none"))
    if sub is None:
        ax.text(x, y, text, ha="center", va="center", color="white", fontsize=size)
    else:
        ax.text(x, y + 0.14, text, ha="center", va="center", color="white", fontsize=size)
        ax.text(x, y - 0.17, sub, ha="center", va="center", color="#c3bfee", fontsize=size - 2)


def arrow(ax, x, y0, y1):
    ax.add_patch(FancyArrowPatch((x, y0), (x, y1), arrowstyle="-|>", mutation_scale=16, color=ARROW, lw=2))


def column(ax, x, title, stages, w, top, gap=0.55):
    ax.text(x, top + 0.45, title, ha="center", va="center", fontsize=15)
    rows, y = [], top
    for st in stages:
        h = 0.5 if "sub" not in st else 0.72
        if "inner" in st:
            h = 0.62 + 0.46 * len(st["inner"])
        if rows:
            y -= gap
            arrow(ax, x, rows[-1][1], y)
        if "inner" in st:
            ax.add_patch(FancyBboxPatch((x - w / 2, y - h), w, h, boxstyle="round,pad=0,rounding_size=0.1",
                                        fc=st["color"], ec="none"))
            ax.text(x, y - 0.28, st["text"], ha="center", va="center", color="white", fontsize=13)
            for i, t in enumerate(st["inner"]):
                box(ax, x, y - 0.75 - 0.46 * i, w - 0.3, 0.36, t, INNER, size=11)
        else:
            box(ax, x, y - h / 2, w, h, st["text"], st["color"], sub=st.get("sub"))
        rows.append((y, y - h))
        y -= h
    return rows


def figure(name, left, right, brace_rows=None, w=3.0, brace_right=None, gap=0.55):
    fig, ax = plt.subplots(figsize=(8.0, 1))
    xl, xr = w / 2 + 0.5, w * 1.5 + 1.3
    top = 0.0
    rl = column(ax, xl, left[0], left[1], w, top, gap)
    rr = column(ax, xr, right[0], right[1], w, top, gap)
    bottom = min(rl[-1][1], rr[-1][1]) - 0.15
    ax.plot([(xl + xr) / 2] * 2, [bottom, top + 0.7], color=LINE, lw=1.5, ls="--")
    if brace_rows:
        a, b = brace_rows
        bx = xl - w / 2 - 0.15
        ax.plot([bx + 0.12, bx, bx, bx + 0.12], [rl[b][1], rl[b][1], rl[a][0], rl[a][0]], color=BRACE, lw=1.8)
        ax.text(bx - 0.17, (rl[a][0] + rl[b][1]) / 2, "HBM round-trips", rotation=90, ha="center",
                va="center", color=BRACE, fontsize=13)
    x_max = xr + w / 2 + 0.2
    if brace_right:
        a, b = brace_right
        bx = xr + w / 2 + 0.15
        ax.plot([bx - 0.12, bx, bx, bx - 0.12], [rr[b][1], rr[b][1], rr[a][0], rr[a][0]], color=BRACE, lw=1.8)
        ax.text(bx + 0.17, (rr[a][0] + rr[b][1]) / 2, "HBM round-trips", rotation=270, ha="center",
                va="center", color=BRACE, fontsize=13)
        x_max += 0.45
    ax.set_xlim(-0.2, x_max)
    ax.set_ylim(bottom - 0.1, top + 0.8)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.set_size_inches(8.0, 8.0 * (top + 0.9 - bottom) / (x_max + 0.2))
    fig.savefig(os.path.join(OUT, name), dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def bracket_v6():
    left = ("v5: 2D transforms", [
        {"text": "Spectral input", "color": GREEN},
        {"text": "2x inverse 2D FFT", "color": PURPLE, "sub": "derivatives in load callbacks", "h": 0.75},
        {"text": "Forward 2D FFT", "color": PURPLE, "sub": "bracket in load callback", "h": 0.75},
        {"text": "Spectral output", "color": GREEN},
    ])
    right = ("v6: 1D passes", [
        {"text": "Spectral input", "color": GREEN},
        {"text": "Column FFT along kx", "color": PURPLE, "sub": "retained ky columns only", "h": 0.75},
        {"text": "Row kernel (cuFFTDx)", "color": PURPLE, "h": 1.75,
         "inner": ["inverse row FFT", "bracket with potential row", "forward row FFT"]},
        {"text": "Column FFT, kept columns", "color": PURPLE, "sub": "unpack to (nkx, nky)", "h": 0.75},
        {"text": "Spectral output", "color": GREEN},
    ])
    figure("bracket_v6.png", left, right, w=3.3)


def fused_linear():
    left = ("Default RK4 stage", [
        {"text": "df", "color": GREEN},
        {"text": "Linear RHS kernel", "color": GRAY},
        {"text": "Collision stencil", "color": GRAY},
        {"text": "RK update (XLA)", "color": GRAY},
        {"text": "df next", "color": GREEN},
    ])
    right = ("fused_linear", [
        {"text": "df", "color": GREEN},
        {"text": "Fused stage kernel", "color": PURPLE, "h": 2.35,
         "inner": ["vpar ring in shared memory", "stencils and drives", "collision stencil", "RK update in registers"]},
        {"text": "df next", "color": GREEN},
    ])
    figure("fused_linear.png", left, right, brace_rows=(1, 3), w=3.3)


def hip_bracket():
    left = ("CUDA v6", [
        {"text": "Spectral input", "color": GREEN},
        {"text": "Column FFT along kx", "color": PURPLE, "sub": "cuFFT, retained ky columns"},
        {"text": "Row kernel (cuFFTDx)", "color": PURPLE,
         "inner": ["inverse row FFT", "bracket with potential row", "forward row FFT"]},
        {"text": "Column FFT, kept columns", "color": PURPLE, "sub": "unpack to (nkx, nky)"},
        {"text": "Spectral output", "color": GREEN},
    ])
    right = ("HIP", [
        {"text": "Spectral input", "color": GREEN},
        {"text": "Pack", "color": GRAY, "sub": "tiled transpose to columns"},
        {"text": "Column FFT along kx", "color": PURPLE, "sub": "hipFFT, contiguous"},
        {"text": "Expand", "color": GRAY, "sub": "derivatives, transpose to rows"},
        {"text": "Inverse row FFT", "color": PURPLE, "sub": "hipFFT"},
        {"text": "Bracket kernel", "color": GRAY, "sub": "product with potential rows"},
        {"text": "Forward row FFT", "color": PURPLE, "sub": "hipFFT"},
        {"text": "Split", "color": GRAY, "sub": "tiled transpose to columns"},
        {"text": "Column FFT, kept columns", "color": PURPLE, "sub": "hipFFT"},
        {"text": "Unpack", "color": GRAY, "sub": "to (nkx, nky)"},
        {"text": "Spectral output", "color": GREEN},
    ])
    figure("hip_bracket.png", left, right, w=3.8, brace_right=(3, 7), gap=0.4)


def connectivity():
    ns, segs = 8, 3
    fig, ax = plt.subplots(figsize=(9.0, 2.6))
    ax.axis("off")
    cls = {0: "-2", 1: "-1", ns * segs - 2: "+1", ns * segs - 1: "+2"}
    x = 0.0
    for k in range(segs):
        for i in range(ns):
            j = k * ns + i
            c = PURPLE if j in cls else "#c9c6ea"
            ax.add_patch(Rectangle((x, 0), 0.9, 0.9, fc=c, ec="white", lw=1.5))
            if j in cls:
                ax.text(x + 0.45, 0.45, cls[j], ha="center", va="center", color="white", fontsize=10)
            x += 1.0
        ax.text(x - ns / 2 - 0.05, 1.25, ["$k_x - \\Delta k_x$", "$k_x$", "$k_x + \\Delta k_x$"][k],
                ha="center", fontsize=13)
        if k < segs - 1:
            ax.add_patch(FancyArrowPatch((x - 0.6, 1.0), (x + 0.6, 1.0), connectionstyle="arc3,rad=-0.6",
                                         arrowstyle="-|>", mutation_scale=14, color=ARROW, lw=1.8))
        x += 0.3
    ax.set_xlim(-0.3, x)
    ax.set_ylim(-0.2, 1.6)
    fig.savefig(os.path.join(OUT, "connectivity.png"), dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    bracket_v6()
    fused_linear()
    hip_bracket()
    connectivity()
