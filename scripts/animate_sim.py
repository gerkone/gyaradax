"""Export gyrokinetic snapshots as a self-contained HTML torus animation.

Usage:
    python scripts/animate_sim.py output_dir/ -o torus.html
    python scripts/animate_sim.py output_dir/ -o torus.mp4 --fps 12

Reads step_*.npz snapshots. Converts spectral phi/df to real space using
the same ifftshift+ifftn convention as warm_restart_eval.ipynb.

HTML output: self-contained Three.js viewer (opens in any browser).
MP4/GIF output: matplotlib 3D rendering.
"""

import argparse
import glob
import json
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


def extract_frames(snapshots, quantity="phi"):
    """Extract per-frame 2D real-space data from snapshots.

    Returns: list of (n_theta, n_zeta) arrays, list of times,
             and (ns, nkx, nky) shapes for info display.
    """
    frames_phi_real = []  # real-space phi(x, y) for torus coloring
    frames_s_kx_ky = []  # velocity-averaged |df|(s, kx, ky)
    times = []
    info = {}

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


def generate_html(snapshots, output_path, R0=3.0, a=1.0):
    frames_phi, frames_skk, times, info = extract_frames(snapshots)
    ns, nkx, nky = info["ns"], info["nkx"], info["nky"]

    # real-space grid size = nkx x nky (from FFT)
    nx, ny = frames_phi[0].shape

    vmax = max(np.max(np.abs(f)) for f in frames_phi)
    if vmax < 1e-30:
        vmax = 1.0

    # also compute vmax for the s-kx-ky panels
    vmax_skk = max(np.max(f) for f in frames_skk)
    if vmax_skk < 1e-30:
        vmax_skk = 1.0

    # serialize frames as flat lists for JS
    phi_flat = [f.ravel().tolist() for f in frames_phi]
    # for the info panels: take central s-slice of df, and ky-sum for s-kx view
    skx_flat = [np.sum(f, axis=-1).ravel().tolist() for f in frames_skk]  # (ns, nkx) per frame
    sky_flat = [np.sum(f, axis=-2).ravel().tolist() for f in frames_skk]  # (ns, nky) per frame

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Gyrokinetic Torus</title>
<style>
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{ background: #ffffff; overflow: hidden; font-family: -apple-system, system-ui, sans-serif; color: #333; }}
  #main {{ display: flex; width: 100vw; height: 100vh; }}
  #torus-panel {{ flex: 1; position: relative; }}
  #side-panel {{
    width: 320px; background: #f8f9fa; border-left: 1px solid #dee2e6;
    display: flex; flex-direction: column; padding: 12px; gap: 8px; overflow-y: auto;
  }}
  #side-panel h3 {{ font-size: 13px; font-weight: 600; color: #555; margin: 4px 0 2px; }}
  #side-panel canvas {{ width: 100%; border-radius: 4px; border: 1px solid #dee2e6; }}
  #controls {{
    position: absolute; bottom: 16px; left: 50%; transform: translateX(-50%);
    display: flex; align-items: center; gap: 10px; z-index: 10;
    background: rgba(255,255,255,0.9); padding: 8px 16px; border-radius: 8px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.1); backdrop-filter: blur(6px);
  }}
  #controls button {{
    background: #2563eb; color: white; border: none; padding: 5px 14px;
    border-radius: 5px; cursor: pointer; font-size: 12px;
  }}
  #controls button:hover {{ background: #1d4ed8; }}
  #slider {{ width: 240px; accent-color: #2563eb; }}
  #time-label {{ font-size: 12px; min-width: 70px; color: #555; }}
  .info-row {{ display: flex; justify-content: space-between; font-size: 11px; color: #777; }}
</style>
</head>
<body>
<div id="main">
  <div id="torus-panel">
    <div id="controls">
      <button id="play-btn" onclick="togglePlay()">Play</button>
      <input type="range" id="slider" min="0" max="{len(times)-1}" value="0" oninput="setFrame(+this.value)">
      <span id="time-label">t = {times[0]:.2f}</span>
    </div>
  </div>
  <div id="side-panel">
    <div class="info-row"><span>grid: {ns}s x {nkx}kx x {nky}ky</span><span id="frame-info">0/{len(times)}</span></div>
    <h3>|df|(s, kx) summed over ky</h3>
    <canvas id="c-skx" width="{nkx}" height="{ns}"></canvas>
    <h3>|df|(s, ky) summed over kx</h3>
    <canvas id="c-sky" width="{nky}" height="{ns}"></canvas>
    <h3>phi real-space (x, y)</h3>
    <canvas id="c-phi" width="{nx}" height="{ny}"></canvas>
  </div>
</div>

<script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
<script>
const R0={R0}, a={a}, NX={nx}, NY={ny}, NS={ns}, NKX={nkx}, NKY={nky};
const VMAX={vmax}, VMAX_SKK={vmax_skk};
const TIMES={json.dumps([round(t,4) for t in times])};
const PHI_FRAMES={json.dumps(phi_flat)};
const SKX_FRAMES={json.dumps(skx_flat)};
const SKY_FRAMES={json.dumps(sky_flat)};
const NFRAMES=TIMES.length;

let scene, camera, renderer, mesh, frameIdx=0, playing=false, playInterval;

function colormap(v, vmax) {{
  let t = (v/vmax+1)*0.5;
  t = Math.max(0, Math.min(1, t));
  if (t<0.5) {{ let s=t*2; return [0.15+s*0.85, 0.3+s*0.7, 0.8+s*0.2]; }}
  else {{ let s=(t-0.5)*2; return [1, 1-s*0.6, 1-s*0.85]; }}
}}

function viridis(t) {{
  t = Math.max(0, Math.min(1, t));
  let r = 0.267+t*(0.329+t*(-1.426+t*(3.024+t*(-2.392+t*0.785))));
  let g = 0.004+t*(1.314+t*(-0.489+t*(-0.477+t*(0.654-t*0.232))));
  let b = 0.329+t*(1.527+t*(-3.866+t*(5.669+t*(-3.727+t*0.906))));
  return [Math.max(0,Math.min(1,r)), Math.max(0,Math.min(1,g)), Math.max(0,Math.min(1,b))];
}}

function drawHeatmap(canvasId, data, rows, cols, vmax, useDiverging) {{
  const c = document.getElementById(canvasId);
  c.width = cols; c.height = rows;
  c.style.height = (rows*3)+'px'; c.style.imageRendering = 'pixelated';
  const ctx = c.getContext('2d');
  const img = ctx.createImageData(cols, rows);
  for (let r=0; r<rows; r++) {{
    for (let c2=0; c2<cols; c2++) {{
      const v = data[r*cols+c2];
      let rgb;
      if (useDiverging) rgb = colormap(v, vmax);
      else rgb = viridis(v/vmax);
      const i = (r*cols+c2)*4;
      img.data[i]=rgb[0]*255; img.data[i+1]=rgb[1]*255; img.data[i+2]=rgb[2]*255; img.data[i+3]=255;
    }}
  }}
  ctx.putImageData(img, 0, 0);
}}

function init() {{
  const container = document.getElementById('torus-panel');
  const W = container.clientWidth, H = container.clientHeight;

  scene = new THREE.Scene();
  scene.background = new THREE.Color(0xffffff);

  camera = new THREE.PerspectiveCamera(40, W/H, 0.1, 100);
  // flat side view: look from above-ish, no spin
  camera.position.set(0, R0*2.5, R0*1.2);
  camera.lookAt(0, 0, 0);

  renderer = new THREE.WebGLRenderer({{ antialias: true }});
  renderer.setSize(W, H);
  renderer.setPixelRatio(window.devicePixelRatio);
  container.insertBefore(renderer.domElement, container.firstChild);

  const ambient = new THREE.AmbientLight(0xffffff, 0.6);
  scene.add(ambient);
  const dir = new THREE.DirectionalLight(0xffffff, 0.8);
  dir.position.set(R0*2, R0*3, R0*2);
  scene.add(dir);
  const dir2 = new THREE.DirectionalLight(0xaabbdd, 0.3);
  dir2.position.set(-R0*2, -R0, R0);
  scene.add(dir2);

  // torus lies flat: rotate 90 degrees around X
  const geom = new THREE.TorusGeometry(R0, a, NX, NY);
  const colors = new Float32Array(geom.attributes.position.count*3);
  geom.setAttribute('color', new THREE.BufferAttribute(colors, 3));
  const mat = new THREE.MeshPhongMaterial({{ vertexColors: true, shininess: 30, specular: 0x111111 }});
  mesh = new THREE.Mesh(geom, mat);
  mesh.rotation.x = Math.PI / 2;  // lie flat
  scene.add(mesh);

  updateFrame(0);
  render();

  window.addEventListener('resize', () => {{
    const W2 = container.clientWidth, H2 = container.clientHeight;
    camera.aspect = W2/H2;
    camera.updateProjectionMatrix();
    renderer.setSize(W2, H2);
  }});
}}

function updateFrame(fi) {{
  frameIdx = fi;
  const phiData = PHI_FRAMES[fi];
  const skxData = SKX_FRAMES[fi];
  const skyData = SKY_FRAMES[fi];

  // update torus colors
  const colors = mesh.geometry.attributes.color;
  const pos = mesh.geometry.attributes.position;
  const n = pos.count;
  for (let i=0; i<n; i++) {{
    const x=pos.getX(i), y=pos.getY(i), z=pos.getZ(i);
    const zeta = Math.atan2(y, x);
    const Rxy = Math.sqrt(x*x+y*y);
    const theta = Math.atan2(z, Rxy-R0);
    let it = Math.round(((theta/(2*Math.PI))%1+1)%1*NX)%NX;
    let iz = Math.round(((zeta/(2*Math.PI))%1+1)%1*NY)%NY;
    const val = phiData[it*NY+iz];
    const [r,g,b] = colormap(val, VMAX);
    colors.setXYZ(i, r, g, b);
  }}
  colors.needsUpdate = true;

  // update side panels
  drawHeatmap('c-skx', skxData, NS, NKX, VMAX_SKK, false);
  drawHeatmap('c-sky', skyData, NS, NKY, VMAX_SKK, false);
  drawHeatmap('c-phi', phiData, NX, NY, VMAX, true);

  document.getElementById('time-label').textContent = 't = '+TIMES[fi].toFixed(2);
  document.getElementById('slider').value = fi;
  document.getElementById('frame-info').textContent = (fi+1)+'/'+NFRAMES;
}}

function render() {{ requestAnimationFrame(render); renderer.render(scene, camera); }}
function setFrame(i) {{ updateFrame(parseInt(i)); }}
function togglePlay() {{
  playing = !playing;
  document.getElementById('play-btn').textContent = playing ? 'Pause' : 'Play';
  if (playing) playInterval = setInterval(()=>{{ updateFrame((frameIdx+1)%NFRAMES); }}, 120);
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


def generate_mp4(snapshots, output_path, R0=3.0, a=1.0, fps=12, dpi=150, dry_run=False):
    """Render to mp4/gif using matplotlib."""
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation
    from matplotlib.colors import Normalize, LightSource

    frames_phi, frames_skk, times, info = extract_frames(snapshots)
    nx, ny = frames_phi[0].shape
    ns, nkx, nky = info["ns"], info["nkx"], info["nky"]

    vmax_phi = max(np.max(np.abs(f)) for f in frames_phi)
    if vmax_phi < 1e-30:
        vmax_phi = 1.0
    vmax_skk = max(np.max(f) for f in frames_skk)
    if vmax_skk < 1e-30:
        vmax_skk = 1.0

    # load diagnostics from the snapshot directory
    diag_dir = sys.argv[1] if len(sys.argv) > 1 else "."
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

    kyspec_max = max(np.max(kyspec_all), 1e-30)
    kxspec_max = max(np.max(kxspec_all), 1e-30)

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
    proj_h = 0.20  # height of projection strip
    fig = plt.figure(figsize=(16, 8), facecolor="white")

    # torus — MASSIVE: nearly full figure, shifted left so right side is for diagnostics
    ax3d = fig.add_axes([-0.25, 0.10, 0.95, 0.95], projection="3d", facecolor="white")

    # 2d projections: small bottom-left strip
    gs_proj = fig.add_gridspec(
        1, 2, wspace=0.08,
        left=0.03, right=0.42, top=proj_h + 0.04, bottom=0.04,
    )
    ax_skx = fig.add_subplot(gs_proj[0])
    ax_sky = fig.add_subplot(gs_proj[1])

    # right column: flux (tall), spectra (tall), phi(x,y) (compact, same height as projections)
    gs_right = fig.add_gridspec(
        3, 1, height_ratios=[1.0, 1.0, 0.5], hspace=0.40,
        left=0.54, right=0.97, top=0.95, bottom=0.04,
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
        np.zeros((ns, nkx)), aspect="auto", cmap="RdBu_r",
        vmin=0, vmax=vmax_skk, origin="lower", interpolation="bilinear",
    )
    im_sky = ax_sky.imshow(
        np.zeros((ns, nky)), aspect="auto", cmap="RdBu_r",
        vmin=0, vmax=vmax_skk, origin="lower", interpolation="bilinear",
    )
    for ax, title in [(ax_skx, r"$|\delta f|\;(s, k_x)$"), (ax_sky, r"$|\delta f|\;(s, k_y)$")]:
        ax.set_title(title, fontsize=14, pad=3)
        ax.tick_params(axis="both", which="both", length=0, labelsize=0,
                       labelbottom=False, labelleft=False)

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
        np.zeros((ny, nx)), aspect="auto", cmap="plasma",
        vmin=-vmax_phi, vmax=vmax_phi, origin="lower", interpolation="bilinear",
    )
    ax_phi2d.set_title(r"$\phi(x, y)$", fontsize=14, pad=3)
    ax_phi2d.tick_params(axis="both", which="both", length=0, labelsize=0,
                         labelbottom=False, labelleft=False)

    def draw(fi):
        ax3d.clear()

        torus_data = frames_torus[fi]
        torus_colors = plt.cm.plasma(norm_phi(torus_data))
        ax3d.plot_surface(
            X, Y, Zc,
            facecolors=torus_colors, shade=True,
            lightsource=LightSource(azdeg=315, altdeg=50),
            rstride=1, cstride=2, antialiased=False, alpha=0.95,
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
            0.02, 0.97, f"t = {times[fi]:.1f}",
            fontsize=20, fontfamily="monospace", fontweight="bold", color="#333",
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


def main():
    parser = argparse.ArgumentParser(description="Export torus animation.")
    parser.add_argument("output_dir", help="directory with step_*.npz snapshots")
    parser.add_argument("-o", "--output", default="torus.html")
    parser.add_argument("--R0", type=float, default=3.0, help="major radius")
    parser.add_argument("--a", type=float, default=1.0, help="minor radius")
    parser.add_argument("--fps", type=int, default=12, help="frames per second (mp4/gif)")
    parser.add_argument("--dpi", type=int, default=120, help="resolution (mp4/gif)")
    parser.add_argument("--dry-run", action="store_true", help="show last frame only (no video)")
    args = parser.parse_args()

    snapshots = load_snapshots(args.output_dir, last_only=False)
    ext = os.path.splitext(args.output)[1].lower()
    if args.dry_run or ext in (".mp4", ".gif", ".png"):
        generate_mp4(
            snapshots,
            args.output,
            R0=args.R0,
            a=args.a,
            fps=args.fps,
            dpi=args.dpi,
            dry_run=args.dry_run,
        )
    else:
        generate_html(snapshots, args.output, R0=args.R0, a=args.a)


if __name__ == "__main__":
    main()
