"""Export gyrokinetic snapshots as an animation.

Usage:
    python scripts/animate_sim.py output_dir/ -o torus.mp4 --fps 12

Reads step_*.npz snapshots. Converts spectral phi/df to real space using
the same ifftshift+ifftn convention as warm_restart_eval.ipynb.

MP4/GIF output: matplotlib 3D rendering.
"""

import argparse
import glob
import os
import sys

import numpy as np


def load_snapshots(output_dir, last_only=False):
    files = sorted(glob.glob(os.path.join(output_dir, "step_*.npz")))
    if not files:
        print(f"No step_*.npz in {output_dir}")
        sys.exit(1)
    if last_only:
        snapshots = [dict(np.load(files[-1]))]
        print(f"Loaded last snapshot: {os.path.basename(files[-1])}")
    else:
        snapshots = [dict(np.load(f)) for f in files]
        print(f"Loaded {len(snapshots)} snapshots")
    return snapshots


def spectral_to_real(field_spectral):
    """Convert spectral (kx, ky) to real-space (x, y).

    Same convention as warm_restart_eval.ipynb:
      ifftshift on kx axis, then ifftn on (kx, ky), take real part.

    Input shape:  (..., nkx, nky) complex
    Output shape: (..., nkx, nky) real
    """
    shifted = np.fft.ifftshift(field_spectral, axes=-2)
    return np.fft.ifftn(shifted, axes=(-2, -1), norm="forward").real


def extract_frames(snapshots, dry_run=False):
    """Extract per-frame 2D real-space data from snapshots.

    Returns: list of (n_theta, n_zeta) arrays, list of times,
             and (ns, nkx, nky) shapes for info display.
    """
    frames_phi_real = []  # real-space phi(x, y) for torus coloring
    frames_s_kx_ky = []  # velocity-averaged |df|(s, kx, ky)
    times = []
    info = {}

    if dry_run:
        snapshots = [snapshots[-1]]

    for snap in snapshots:
        t = float(snap["time"])
        times.append(t)

        phi = snap["phi"]  # (ns, nkx, nky) complex
        ns, nkx, nky = phi.shape
        info["ns"], info["nkx"], info["nky"] = ns, nkx, nky

        # real-space phi for torus surface coloring:
        # average over s, then spectral -> real on (kx, ky)
        phi_avg_s = np.mean(phi, axis=0)  # (nkx, nky)
        phi_real = spectral_to_real(phi_avg_s)  # (nkx, nky) real
        frames_phi_real.append(phi_real)

        # velocity-averaged |df|(s, kx, ky) for info panel
        df = snap["df"]  # (nvpar, nmu, ns, nkx, nky) or (nsp, nvpar, nmu, ns, nkx, nky)
        if df.ndim == 6:
            df = df[0]  # take first species
        df_vavg = np.mean(np.abs(df), axis=(0, 1))  # (ns, nkx, nky)
        frames_s_kx_ky.append(df_vavg)

    return frames_phi_real, frames_s_kx_ky, times, info


def generate_mp4(
    snapshots, output_path, R0=3.0, a=1.0, fps=12, dpi=150, dry_run=False, diag_dir="."
):
    """Render to mp4/gif using matplotlib."""
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation
    from matplotlib.colors import Normalize, LightSource

    frames_phi, frames_skk, times, info = extract_frames(snapshots, dry_run=dry_run)
    nx, ny = frames_phi[0].shape
    ns, nkx, nky = info["ns"], info["nkx"], info["nky"]

    vmax_phi = max(np.max(np.abs(f)) for f in frames_phi)
    if vmax_phi < 1e-30:
        vmax_phi = 1.0
    vmax_skk = max(np.max(f) for f in frames_skk)
    if vmax_skk < 1e-30:
        vmax_skk = 1.0

    # load diagnostics from the snapshot directory
    # diag_dir is passed as a parameter
    eflux_trace = growth_all = kyspec_all = kxspec_all = diag_times = None
    try:
        fd = np.load(os.path.join(diag_dir, "fluxes.npz"))
        eflux_trace = fd["fluxes"][:, 1]
        diag_times = fd.get("time", np.load(os.path.join(diag_dir, "growth.npz"))["time"])
    except (FileNotFoundError, KeyError):
        pass
    try:
        growth_all = np.load(os.path.join(diag_dir, "growth.npz"))["growth"]
    except (FileNotFoundError, KeyError):
        pass
    try:
        kyspec_all = np.load(os.path.join(diag_dir, "kyspec.npz"))["ky_spec"]
        kxspec_all = np.load(os.path.join(diag_dir, "kxspec.npz"))["kx_spec"]
    except (FileNotFoundError, KeyError):
        pass

    if eflux_trace is None:
        eflux_trace = np.array([float(np.sum(np.abs(s["phi"]) ** 2)) for s in snapshots])
        diag_times = np.array(times)
    if kyspec_all is None:
        kyspec_all = np.array([np.sum(np.abs(s["phi"]) ** 2, axis=(0, 1)) for s in snapshots])
        kxspec_all = np.array([np.sum(np.abs(s["phi"]) ** 2, axis=(0, 2)) for s in snapshots])
    if growth_all is None:
        growth_all = np.zeros((len(times), nky))

    # torus mesh — seamless: use endpoint=False, then append first column
    n_t, n_z = max(nx, 80), max(ny, 200)
    theta = np.linspace(0, 2 * np.pi, n_t, endpoint=False)
    zeta = np.linspace(0, 2 * np.pi, n_z, endpoint=False)
    # append first point to close the surface without a seam
    theta_c = np.append(theta, theta[0] + 2 * np.pi)
    zeta_c = np.append(zeta, zeta[0] + 2 * np.pi)
    T, Z = np.meshgrid(theta_c, zeta_c, indexing="ij")
    Rm = R0 + a * np.cos(T)
    X = Rm * np.cos(Z)
    Y = Rm * np.sin(Z)
    Zc = a * np.sin(T)

    from scipy.ndimage import zoom

    frames_torus = []
    for f in frames_phi:
        zf = zoom(f, (n_t / f.shape[0], n_z / f.shape[1]), order=1, mode="wrap")
        # pad to close: append first row and column
        zf = np.pad(zf, ((0, 1), (0, 1)), mode="wrap")
        frames_torus.append(zf)

    # --- LAYOUT ---
    # Left: torus (MASSIVE, fills most of figure), two small 2d projections (bottom-left)
    # Right: heat flux, ky+kx spectra (large), phi(x,y) (compact)
    proj_h = 0.30  # height of projection strip
    fig = plt.figure(figsize=(16, 8), facecolor="white")

    # torus — MASSIVE: nearly full figure, shifted left so right side is for diagnostics
    ax3d = fig.add_axes([-0.25, 0.20, 0.95, 0.95], projection="3d", facecolor="white")

    # 2d projections: small bottom-left strip
    gs_proj = fig.add_gridspec(
        1,
        2,
        wspace=0.08,
        left=0.03,
        right=0.42,
        top=proj_h + 0.04,
        bottom=0.04,
    )
    ax_skx = fig.add_subplot(gs_proj[0])
    ax_sky = fig.add_subplot(gs_proj[1])

    # right column: flux (tall), spectra (tall), phi(x,y) (compact, same height as projections)
    gs_right = fig.add_gridspec(
        3,
        1,
        height_ratios=[0.5, 0.7, 0.8],
        hspace=0.40,
        left=0.54,
        right=0.97,
        top=0.95,
        bottom=0.04,
    )
    ax_flux = fig.add_subplot(gs_right[0])
    gs_spec = gs_right[1].subgridspec(1, 2, wspace=0.35)
    ax_kyspec = fig.add_subplot(gs_spec[0])
    ax_kxspec = fig.add_subplot(gs_spec[1])
    ax_phi2d = fig.add_subplot(gs_right[2])

    norm_phi = Normalize(-vmax_phi, vmax_phi)
    lim = R0 + a
    z_lim = a

    # 2d projection panels — RdBu_r colormap
    im_skx = ax_skx.imshow(
        np.zeros((ns, nkx)),
        aspect="auto",
        cmap="RdBu_r",
        vmin=0,
        vmax=vmax_skk,
        origin="lower",
        interpolation="bilinear",
    )
    im_sky = ax_sky.imshow(
        np.zeros((ns, nky)),
        aspect="auto",
        cmap="RdBu_r",
        vmin=0,
        vmax=vmax_skk,
        origin="lower",
        interpolation="bilinear",
    )
    for ax, title in [(ax_skx, r"$|\delta f|\;(s, k_x)$"), (ax_sky, r"$|\delta f|\;(s, k_y)$")]:
        ax.set_title(title, fontsize=14, pad=3)
        ax.tick_params(
            axis="both", which="both", length=0, labelsize=0, labelbottom=False, labelleft=False
        )

    # heat flux — axes grow with time
    (flux_line,) = ax_flux.plot([], [], "-", color="#24B6AD", lw=1.2)
    flux_dot = ax_flux.plot([], [], "o", color="#EA4335", ms=5, zorder=5)[0]
    ax_flux.set_ylabel("heat flux", fontsize=13)
    ax_flux.set_title("heat flux", fontsize=14, pad=3)
    ax_flux.grid(True, alpha=0.3)
    ax_flux.tick_params(labelsize=11)

    # ky spectrum — axes grow with time
    (kyspec_line,) = ax_kyspec.semilogy([], [], "o-", color="#9B51E0", ms=2, lw=1)
    ax_kyspec.set_xlim(0, kyspec_all.shape[1] - 1)
    ax_kyspec.set_title(r"$k_y$ spec", fontsize=14, pad=3)
    ax_kyspec.grid(True, which="both", alpha=0.3)
    ax_kyspec.tick_params(labelsize=11)

    # kx spectrum — axes grow with time
    (kxspec_line,) = ax_kxspec.semilogy([], [], "o-", color="#9B51E0", ms=2, lw=1)
    ax_kxspec.set_xlim(0, kxspec_all.shape[1] - 1)
    ax_kxspec.set_title(r"$k_x$ spec", fontsize=14, pad=3)
    ax_kxspec.grid(True, which="both", alpha=0.3)
    ax_kxspec.tick_params(labelsize=11)

    # phi(x,y) — plasma, tall
    im_phi2d = ax_phi2d.imshow(
        np.zeros((ny, nx)),
        aspect="auto",
        cmap="plasma",
        vmin=-vmax_phi,
        vmax=vmax_phi,
        origin="lower",
        interpolation="bilinear",
    )
    ax_phi2d.set_title(r"$\phi(x, y)$", fontsize=14, pad=3)
    ax_phi2d.tick_params(
        axis="both", which="both", length=0, labelsize=0, labelbottom=False, labelleft=False
    )

    def draw(fi):
        ax3d.clear()

        torus_data = frames_torus[fi]
        torus_colors = plt.cm.plasma(norm_phi(torus_data))
        ax3d.plot_surface(
            X,
            Y,
            Zc,
            facecolors=torus_colors,
            shade=True,
            lightsource=LightSource(azdeg=315, altdeg=50),
            rstride=1,
            cstride=2,
            antialiased=False,
            alpha=0.95,
        )
        ax3d.set_xlim(-lim, lim)
        ax3d.set_ylim(-lim, lim)
        ax3d.set_zlim(-z_lim, z_lim)
        ax3d.set_box_aspect([1, 1, a / lim])
        ax3d.view_init(elev=25, azim=20 + fi * 0.5)
        ax3d.dist = 4.2
        ax3d.axis("off")

        fig.texts.clear()
        fig.text(
            0.02,
            0.97,
            f"t = {times[fi]:.1f}",
            fontsize=20,
            fontfamily="monospace",
            fontweight="bold",
            color="#333",
            bbox=dict(facecolor="#eeeeee", edgecolor="none", pad=4, alpha=0.85),
        )

        # 2d projections
        im_skx.set_data(np.sum(frames_skk[fi], axis=-1))
        im_sky.set_data(np.sum(frames_skk[fi], axis=-2))

        # diagnostics — animated axes that grow with data
        di = np.argmin(np.abs(diag_times - times[fi]))
        mask = diag_times <= times[fi]

        # flux: xlim and ylim grow
        flux_line.set_data(diag_times[mask], eflux_trace[mask])
        flux_dot.set_data([diag_times[di]], [eflux_trace[di]])
        t_now = float(diag_times[di])
        ax_flux.set_xlim(float(diag_times[0]), max(t_now * 1.05, 1.0))
        ef_now = float(np.max(eflux_trace[mask])) if mask.any() else 1.0
        ax_flux.set_ylim(0, max(ef_now * 1.3, 1e-6))

        # spectra: ylim adapts to current frame
        ky_now = kyspec_all[di]
        kx_now = kxspec_all[di]
        kyspec_line.set_data(np.arange(len(ky_now)), ky_now)
        kxspec_line.set_data(np.arange(len(kx_now)), kx_now)
        ky_max_now = max(float(np.max(ky_now)), 1e-30)
        kx_max_now = max(float(np.max(kx_now)), 1e-30)
        ax_kyspec.set_ylim(ky_max_now * 1e-5, ky_max_now * 3)
        ax_kxspec.set_ylim(kx_max_now * 1e-5, kx_max_now * 3)

        im_phi2d.set_data(frames_phi[fi].T)
        return []

    if dry_run:
        draw(len(times) - 1)
        out_png = os.path.join(os.path.dirname(output_path) or ".", "torus_preview.png")
        fig.savefig(out_png, dpi=dpi, facecolor="white")
        print(f"Dry run: saved {out_png}")
        plt.close(fig)
        return

    print(f"Rendering {len(times)} frames at {dpi} dpi...")
    anim = animation.FuncAnimation(fig, draw, frames=len(times), interval=1000 // fps, blit=False)

    ext = os.path.splitext(output_path)[1].lower()
    if ext == ".gif":
        anim.save(output_path, writer=animation.PillowWriter(fps=fps), dpi=dpi)
    else:
        anim.save(
            output_path,
            writer=animation.FFMpegWriter(
                fps=fps, bitrate=6000, extra_args=["-pix_fmt", "yuv420p"]
            ),
            dpi=dpi,
        )
    size_mb = os.path.getsize(output_path) / 1e6
    print(f"Saved {output_path} ({size_mb:.1f} MB)")
    plt.close(fig)


def generate_html_viewer(snapshots, output_path, R0=3.0, a=1.0, diag_dir=".", dry_run=False):
    """Generate a self-contained HTML viewer with Three.js torus and SVG diagnostics."""
    import json as _json

    frames_phi, frames_skk, times, info = extract_frames(snapshots, dry_run=dry_run)
    ns, nkx, nky = info["ns"], info["nkx"], info["nky"]
    nx, ny = frames_phi[0].shape

    vmax = max(np.max(np.abs(f)) for f in frames_phi)
    if vmax < 1e-30:
        vmax = 1.0

    # load diagnostics
    eflux_trace = kyspec_all = kxspec_all = diag_times = None
    try:
        fd = np.load(os.path.join(diag_dir, "fluxes.npz"))
        eflux_trace = fd["fluxes"][:, 1].tolist()
        diag_times = fd.get("time", np.load(os.path.join(diag_dir, "growth.npz"))["time"]).tolist()
    except (FileNotFoundError, KeyError):
        pass
    try:
        kyspec_all = np.load(os.path.join(diag_dir, "kyspec.npz"))["ky_spec"].tolist()
        kxspec_all = np.load(os.path.join(diag_dir, "kxspec.npz"))["kx_spec"].tolist()
    except (FileNotFoundError, KeyError):
        pass

    if eflux_trace is None:
        eflux_trace = [float(np.sum(np.abs(s["phi"]) ** 2)) for s in snapshots]
        diag_times = times
    if kyspec_all is None:
        kyspec_all = [np.sum(np.abs(s["phi"]) ** 2, axis=(0, 1)).tolist() for s in snapshots]
        kxspec_all = [np.sum(np.abs(s["phi"]) ** 2, axis=(0, 2)).tolist() for s in snapshots]

    # serialize frames as flat float lists
    phi_flat = [f.ravel().tolist() for f in frames_phi]

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Gyrokinetic Torus Viewer</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
  * {{ margin:0; padding:0; box-sizing:border-box; }}
  body {{ background:#0d0d1a; color:#ccc; font-family:'SF Mono','Fira Code',monospace; font-size:16px; overflow-x:hidden; }}
  #app {{ display:grid; grid-template-columns:3fr 2fr; grid-template-rows:1fr; height:70vh; max-width:1400px; margin:0 auto; gap:0; }}
  #torus-panel {{ position:relative; background:#0d0d1a; min-height:0; }}
  #side {{ background:#111122; border-left:1px solid #222; display:flex; flex-direction:column; padding:12px; gap:6px; overflow:hidden; }}
  #side > div {{ display:flex; flex-direction:column; min-height:0; }}
  #side > div:first-child {{ flex:0 0 auto; }}
  #side > div:nth-child(2) {{ flex:3 1 0; }}
  #side > div:nth-child(3) {{ flex:2 1 0; }}
  #side > div:nth-child(4) {{ flex:2 1 0; }}
  #side > div:last-child {{ flex:3 1 0; }}
  .panel-title {{ color:#778; font-size:13px; text-transform:uppercase; letter-spacing:1.5px; margin-bottom:6px; }}
  .stat {{ color:#9fcefd; font-size:20px; font-weight:600; }}
  .stat-label {{ color:#556; font-size:12px; }}
  canvas.diag {{ width:100%; flex:1; min-height:0; border-radius:4px; background:#0a0a16; border:1px solid #1a1a2e; }}
  canvas.phi2d {{ width:100%; flex:1; min-height:0; border-radius:4px; image-rendering:pixelated; border:1px solid #1a1a2e; }}
  #controls {{ position:absolute; bottom:16px; left:50%; transform:translateX(-50%); width:80%; display:flex; align-items:center; gap:10px; z-index:10; background:rgba(13,13,26,0.9); padding:8px 18px; border-radius:8px; border:1px solid #222; backdrop-filter:blur(8px); }}
  #controls button {{ background:#1a1a2e; color:#9fcefd; border:1px solid #334; padding:5px 16px; border-radius:4px; cursor:pointer; font-family:inherit; font-size:13px; }}
  #controls button:hover {{ background:#222244; border-color:#9fcefd; }}
  #slider {{ flex:1; accent-color:#9fcefd; height:4px; }}
  #time-label {{ color:#9fcefd; font-size:17px; min-width:80px; font-weight:600; }}
  .grid-info {{ color:#556; font-size:14px; }}
  @media (max-width:800px) {{
    #app {{ grid-template-columns:1fr; grid-template-rows:50vh 1fr; }}
    #side {{ border-left:none; border-top:1px solid #222; }}
  }}
</style>
</head>
<body>
<div id="app">
  <div id="torus-panel">
    <div id="controls">
      <button id="play-btn" onclick="togglePlay()">&#9654;</button>
      <input type="range" id="slider" min="0" max="{len(times)-1}" value="{len(times)-1}" oninput="setFrame(+this.value)">
      <span id="time-label">t = {times[-1]:.1f}</span>
    </div>
  </div>
  <div id="side">
    <div>
      <div class="panel-title">simulation</div>
      <span class="grid-info">{ns}s &times; {nkx}kx &times; {nky}ky &nbsp; | &nbsp; <span id="frame-info">{len(times)}/{len(times)}</span></span>
    </div>
    <div>
      <div class="panel-title">heat flux</div>
      <canvas id="c-flux" class="diag"></canvas>
    </div>
    <div>
      <div class="panel-title">k<sub>y</sub> spectrum</div>
      <canvas id="c-ky" class="diag"></canvas>
    </div>
    <div>
      <div class="panel-title">k<sub>x</sub> spectrum</div>
      <canvas id="c-kx" class="diag"></canvas>
    </div>
    <div>
      <div class="panel-title">&phi;(x, y)</div>
      <canvas id="c-phi" class="phi2d" width="{nx}" height="{ny}"></canvas>
    </div>
  </div>
</div>

<script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
<script>
const R0={R0}, AA={a}, NX={nx}, NY={ny};
const VMAX={vmax};
const TIMES={_json.dumps([round(t,3) for t in times])};
const PHI={_json.dumps(phi_flat)};
const FLUX_T={_json.dumps([round(t,3) for t in diag_times])};
const FLUX={_json.dumps([round(v,6) for v in eflux_trace])};
const KYSPEC={_json.dumps(kyspec_all)};
const KXSPEC={_json.dumps(kxspec_all)};
const NF=TIMES.length;

let scene,camera,renderer,mesh,frameIdx=NF-1,playing=false,playInterval;

function plasma(v,vmax) {{
  let t=Math.max(0,Math.min(1,(v/vmax+1)*0.5));
  // dark blue -> white -> dark red
  let r,g,b;
  if(t<0.5){{ let s=t*2; r=s*0.3; g=s*0.3+0.05; b=0.15+s*0.7; }}
  else{{ let s=(t-0.5)*2; r=0.3+s*0.7; g=0.35-s*0.3; b=0.85-s*0.7; }}
  return [r,g,b];
}}

function drawLine(canvasId,data,highlight,isLog,rawVal) {{
  let c=document.getElementById(canvasId);
  let W=c.width=c.clientWidth*2, H=c.height=c.clientHeight*2;
  let ctx=c.getContext('2d');
  ctx.clearRect(0,0,W,H);
  if(!data||data.length<2) return;

  let vals=isLog?data.map(v=>Math.log10(Math.max(v,1e-30))):data;
  let mn=Math.min(...vals), mx=Math.max(...vals);
  if(mx-mn<1e-20) mx=mn+1;

  ctx.beginPath();
  for(let i=0;i<vals.length;i++){{
    let x=i/(vals.length-1)*W;
    let y=H-((vals[i]-mn)/(mx-mn))*H*0.85-H*0.05;
    i===0?ctx.moveTo(x,y):ctx.lineTo(x,y);
  }}
  ctx.strokeStyle='rgba(159,206,253,0.6)';
  ctx.lineWidth=1.5;
  ctx.stroke();

  if(highlight>=0&&highlight<vals.length){{
    let x=highlight/(vals.length-1)*W;
    let y=H-((vals[highlight]-mn)/(mx-mn))*H*0.85-H*0.05;
    ctx.beginPath(); ctx.arc(x,y,7,0,Math.PI*2);
    ctx.fillStyle='#c27721'; ctx.fill();
    if(rawVal!==undefined){{
      let label=rawVal<10?rawVal.toFixed(2):rawVal.toFixed(1);
      ctx.font=(H*0.12)+'px monospace';
      ctx.fillStyle='rgba(159,206,253,0.8)';
      ctx.textAlign=x>W*0.75?'right':'left';
      ctx.fillText(label,x>W*0.75?x-12:x+12,y+4);
    }}
  }}
}}

function drawSpectrum(canvasId,spec,isLog) {{
  if(!spec) return;
  let c=document.getElementById(canvasId);
  let W=c.width=c.clientWidth*2, H=c.height=c.clientHeight*2;
  let ctx=c.getContext('2d');
  ctx.clearRect(0,0,W,H);

  let vals=isLog?spec.map(v=>Math.log10(Math.max(v,1e-30))):spec;
  let mn=Math.min(...vals), mx=Math.max(...vals);
  if(mx-mn<1e-20) mx=mn+1;

  let bw=W/vals.length;
  for(let i=0;i<vals.length;i++){{
    let h=((vals[i]-mn)/(mx-mn))*H*0.85;
    ctx.fillStyle='rgba(159,206,253,0.5)';
    ctx.fillRect(i*bw,H-h,bw-1,h);
  }}
}}

function drawPhi(fi) {{
  let c=document.getElementById('c-phi');
  c.width=NY; c.height=NX;
  c.style.imageRendering='pixelated';
  let ctx=c.getContext('2d');
  let img=ctx.createImageData(NY,NX);
  let d=PHI[fi];
  for(let r=0;r<NX;r++) for(let j=0;j<NY;j++) {{
    let v=d[r*NY+j];
    let [cr,cg,cb]=plasma(v,VMAX);
    let idx=(r*NY+j)*4;
    img.data[idx]=cr*255; img.data[idx+1]=cg*255; img.data[idx+2]=cb*255; img.data[idx+3]=255;
  }}
  ctx.putImageData(img,0,0);
}}

function init() {{
  let container=document.getElementById('torus-panel');
  let W=container.clientWidth, H=container.clientHeight;
  scene=new THREE.Scene();
  scene.background=new THREE.Color(0x0d0d1a);
  camera=new THREE.PerspectiveCamera(50,W/H,0.1,300);
  camera.position.set(0,R0*2.7,R0*2.7);
  camera.lookAt(0,0,0);
  renderer=new THREE.WebGLRenderer({{antialias:true}});
  renderer.setSize(W,H);
  renderer.setPixelRatio(window.devicePixelRatio);
  container.insertBefore(renderer.domElement,container.firstChild);

  let amb=new THREE.AmbientLight(0xffffff,0.5);
  scene.add(amb);
  let dir=new THREE.DirectionalLight(0xffffff,0.8);
  dir.position.set(R0*2,R0*3,R0*2);
  scene.add(dir);

  let geom=new THREE.TorusGeometry(R0,AA,Math.max(NX,64),Math.max(NY,128));
  let colors=new Float32Array(geom.attributes.position.count*3);
  geom.setAttribute('color',new THREE.BufferAttribute(colors,3));
  let mat=new THREE.MeshPhongMaterial({{vertexColors:true,shininess:20,specular:0x111111}});
  mesh=new THREE.Mesh(geom,mat);
  mesh.rotation.x=Math.PI/2;
  scene.add(mesh);

  updateFrame(NF-1);
  render();

  window.addEventListener('resize',()=>{{
    let W2=container.clientWidth, H2=container.clientHeight;
    camera.aspect=W2/H2;
    camera.updateProjectionMatrix();
    renderer.setSize(W2,H2);
  }});
}}

function updateFrame(fi) {{
  frameIdx=fi;
  let phiData=PHI[fi];

  let colors=mesh.geometry.attributes.color;
  let pos=mesh.geometry.attributes.position;
  for(let i=0;i<pos.count;i++){{
    let x=pos.getX(i),y=pos.getY(i),z=pos.getZ(i);
    let zeta=Math.atan2(y,x);
    let Rxy=Math.sqrt(x*x+y*y);
    let theta=Math.atan2(z,Rxy-R0);
    let it=Math.round(((theta/(2*Math.PI))%1+1)%1*NX)%NX;
    let iz=Math.round(((zeta/(2*Math.PI))%1+1)%1*NY)%NY;
    let val=phiData[it*NY+iz];
    let [r,g,b]=plasma(val,VMAX);
    colors.setXYZ(i,r,g,b);
  }}
  colors.needsUpdate=true;

  // find closest diagnostic frame
  let di=0; let minD=1e20;
  for(let i=0;i<FLUX_T.length;i++){{ let d=Math.abs(FLUX_T[i]-TIMES[fi]); if(d<minD){{minD=d;di=i;}} }}

  drawLine('c-flux',FLUX.slice(0,di+1),di,false,FLUX[di]);
  if(KYSPEC[di]) drawSpectrum('c-ky',KYSPEC[di],true);
  if(KXSPEC[di]) drawSpectrum('c-kx',KXSPEC[di],true);
  drawPhi(fi);

  document.getElementById('time-label').textContent='t = '+TIMES[fi].toFixed(1);
  document.getElementById('slider').value=fi;
  document.getElementById('frame-info').textContent=(fi+1)+'/'+NF;
}}

function render() {{
  requestAnimationFrame(render);
  renderer.render(scene,camera);
}}
function setFrame(i) {{ updateFrame(parseInt(i)); }}
function togglePlay() {{
  playing=!playing;
  document.getElementById('play-btn').innerHTML=playing?'&#9646;&#9646;':'&#9654;';
  if(playing) playInterval=setInterval(()=>updateFrame((frameIdx+1)%NF),24);
  else clearInterval(playInterval);
}}
init();
</script>
</body>
</html>"""

    with open(output_path, "w") as f:
        f.write(html)
    size_mb = os.path.getsize(output_path) / 1e6
    print(f"Saved {output_path} ({size_mb:.1f} MB, {len(times)} frames)")


def main():
    parser = argparse.ArgumentParser(description="Export torus animation.")
    parser.add_argument("output_dir", help="directory with step_*.npz snapshots")
    parser.add_argument("-o", "--output", default="torus.mp4")
    parser.add_argument("--R0", type=float, default=3.0, help="major radius")
    parser.add_argument("--a", type=float, default=1.0, help="minor radius")
    parser.add_argument("--fps", type=int, default=12, help="frames per second (mp4/gif)")
    parser.add_argument("--dpi", type=int, default=120, help="resolution (mp4/gif)")
    parser.add_argument("--dry-run", action="store_true", help="last frame only (no video)")
    args = parser.parse_args()

    snapshots = load_snapshots(args.output_dir, last_only=args.dry_run)
    ext = os.path.splitext(args.output)[1].lower()

    if ext == ".html":
        generate_html_viewer(
            snapshots,
            args.output,
            R0=args.R0,
            a=args.a,
            diag_dir=args.output_dir,
            dry_run=args.dry_run,
        )
    else:
        generate_mp4(
            snapshots,
            args.output,
            R0=args.R0,
            a=args.a,
            fps=args.fps,
            dpi=args.dpi,
            dry_run=args.dry_run,
            diag_dir=args.output_dir,
        )


if __name__ == "__main__":
    main()
