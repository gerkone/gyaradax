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

    # torus mesh — higher resolution for smooth rendering
    n_t, n_z = max(nx, 80), max(ny, 160)
    theta = np.linspace(0, 2 * np.pi, n_t, endpoint=False)
    zeta = np.linspace(0, 2 * np.pi, n_z, endpoint=False)
    T, Z = np.meshgrid(theta, zeta, indexing="ij")
    R = R0 + a * np.cos(T)
    X = R * np.cos(Z)
    Y = R * np.sin(Z)
    Zc = a * np.sin(T)

    # interpolate phi frames to torus resolution
    from scipy.ndimage import zoom

    frames_torus = []
    for f in frames_phi:
        zf = zoom(f, (n_t / f.shape[0], n_z / f.shape[1]), order=1)
        frames_torus.append(zf)

    fig = plt.figure(figsize=(14, 5), facecolor="white")
    ax3d = fig.add_axes([-0.18, -0.1, 0.85, 1.2], projection="3d", facecolor="white")
    gs_r = fig.add_gridspec(
        3,
        2,
        height_ratios=[1, 1, 1.1],
        hspace=0.2,
        wspace=0.06,
        left=0.50,
        right=0.96,
        top=0.93,
        bottom=0.02,
    )
    ax_skx = fig.add_subplot(gs_r[0:2, 0])
    ax_sky = fig.add_subplot(gs_r[0:2, 1])
    ax_phi = fig.add_subplot(gs_r[2, :])

    norm_phi = Normalize(-vmax_phi, vmax_phi)
    _ = Normalize(0, vmax_skk)  # norm_df reserved for future use
    lim = R0 + a
    z_lim = a

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
    im_phi = ax_phi.imshow(
        np.zeros((ny, nx)),
        aspect="auto",
        cmap="plasma",
        vmin=-vmax_phi,
        vmax=vmax_phi,
        origin="lower",
        interpolation="bilinear",
    )

    for ax, title in [
        (ax_skx, r"$|\delta f|\;(s,\,k_x)$"),
        (ax_sky, r"$|\delta f|\;(s,\,k_y)$"),
        (ax_phi, r"$\phi\;(x,\,y)$"),
    ]:
        ax.set_title(title, fontsize=11, pad=3)
        ax.tick_params(
            axis="both", which="both", length=0, labelsize=0, labelbottom=False, labelleft=False
        )

    # build torus with a wedge cutout (remove 60 degrees to show cross-section)
    cutout_start, cutout_end = 0, int(n_z * 0.83)  # keep 300 of 360 degrees
    X_cut = X[:, cutout_start:cutout_end]
    Y_cut = Y[:, cutout_start:cutout_end]
    Zc_cut = Zc[:, cutout_start:cutout_end]

    def draw(fi):
        ax3d.clear()

        # torus with cutout
        torus_data = frames_torus[fi][:, cutout_start:cutout_end]
        colors = plt.cm.plasma(norm_phi(torus_data))
        ax3d.plot_surface(
            X_cut,
            Y_cut,
            Zc_cut,
            facecolors=colors,
            shade=True,
            lightsource=LightSource(azdeg=315, altdeg=50),
            rstride=1,
            cstride=2,
            antialiased=False,
            alpha=0.95,
        )

        # cross-section disk at the cutout edge
        theta_cs = np.linspace(0, 2 * np.pi, n_t)
        zeta_cut = 2 * np.pi * cutout_end / n_z
        r_cs = a * np.cos(theta_cs)
        z_cs = a * np.sin(theta_cs)
        x_cs = (R0 + r_cs) * np.cos(zeta_cut)
        y_cs = (R0 + r_cs) * np.sin(zeta_cut)

        # fill cross-section with phi color
        phi_cs = frames_torus[fi][:, cutout_end % n_z]
        cs_colors = plt.cm.plasma(norm_phi(phi_cs))
        for j in range(len(theta_cs) - 1):
            ax3d.plot(
                [x_cs[j], x_cs[j + 1]],
                [y_cs[j], y_cs[j + 1]],
                [z_cs[j], z_cs[j + 1]],
                color=cs_colors[j % len(cs_colors)],
                lw=2.5,
            )

        ax3d.set_xlim(-lim, lim)
        ax3d.set_ylim(-lim, lim)
        ax3d.set_zlim(-z_lim, z_lim)
        ax3d.set_box_aspect([1, 1, a / lim])
        ax3d.view_init(elev=30, azim=20)
        ax3d.dist = 5.0  # zoom in (default ~10)
        ax3d.axis("off")
        fig.texts.clear()
        fig.text(
            0.02,
            0.93,
            f"t = {times[fi]:.2f}",
            fontsize=13,
            fontfamily="monospace",
            color="#444444",
            bbox=dict(facecolor="#eeeeee", edgecolor="none", pad=3, alpha=0.8),
        )

        im_skx.set_data(np.sum(frames_skk[fi], axis=-1))
        im_sky.set_data(np.sum(frames_skk[fi], axis=-2))
        im_phi.set_data(frames_phi[fi].T)
        return []

    if dry_run:
        draw(len(times) - 1)
        out_png = output_path.rsplit(".", 1)[0] + "_preview.png"
        fig.savefig(out_png, dpi=dpi, facecolor="white")
        print(f"Dry run: saved {out_png}")
        plt.show()
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

    snapshots = load_snapshots(args.output_dir, last_only=args.dry_run)
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
