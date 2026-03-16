
import json
import os

notebook_path = "notebooks/examine_validation.ipynb"

# Read the notebook content
with open(notebook_path, "r") as f:
    notebook_content = json.load(f)

# Modify the 'plot_fluxes' cell
for cell in notebook_content["cells"]:
    if cell.get("id") == "plot_fluxes":
        # New source for the 'plot_fluxes' cell
        new_source = [
            "
",
            "for out_dir in output_dirs:
",
            "    path = os.path.join("..", out_dir)
",
            "
",
            "    flux_path = os.path.join(path, "fluxes.npz")
",
            "    growth_path = os.path.join(path, "growth.npz")
",
            "    kx_spec_path = os.path.join(path, "kxspec.npz")
",
            "    ky_spec_path = os.path.join(path, "kyspec.npz")
",
            "
",
            "    if not os.path.exists(flux_path) or not os.path.exists(growth_path):
",
            "        print(f"Skipping {out_dir}: missing .npz files")
",
            "        continue
",
            "
",
            "    sim_flux = np.load(flux_path)["fluxes"]
",
            "    sim_time = np.load(growth_path)["time"]
",
            "
",
            "    config_name = out_dir.replace("validation_outputs_", "")
",
            "    config_path = os.path.join("..", "configs", f"{config_name}.yaml")
",
            "
",
            "    ref_time = None
",
            "    ref_fluxes = None
",
            "    ref_kx_spec = None
",
            "    ref_ky_spec = None
",
            "
",
            "    if os.path.exists(config_path):
",
            "        cfg = load_config(config_path)
",
            "        ref_dir = cfg.run.data_dir
",
            "
",
            "        ref_time_path = os.path.join(ref_dir, "time.dat")
",
            "        ref_flux_path = os.path.join(ref_dir, "fluxes.dat")
",
            "
",
            "        if os.path.exists(ref_time_path) and os.path.exists(ref_flux_path):
",
            "            ref_time = np.loadtxt(ref_time_path)
",
            "            ref_fluxes = np.loadtxt(ref_flux_path).T
",
            "
",
            "        # Load full geometry from config
",
            "        geom = load_geometry(ref_dir)
",
            "        kx = np.asarray(geom["kxrh"])
",
            "        ky = np.asarray(geom["krho"])
",
            "
",
            "        # Time-averaged reference spectra
",
            "        avg_count = 80
",
            "        ref_kx_path = os.path.join(ref_dir, "kxspec")
",
            "        ref_ky_path = os.path.join(ref_dir, "kyspec")
",
            "
",
            "        if os.path.exists(ref_kx_path) and os.path.exists(ref_ky_path):
",
            "            try:
",
            "                rkx = np.loadtxt(ref_kx_path)
",
            "                rky = np.loadtxt(ref_ky_path)
",
            "                ref_kx_spec = np.mean(rkx[-avg_count:], axis=0)
",
            "                ref_ky_spec = np.mean(rky[-avg_count:], axis=0)
",
            "            except Exception as e:
",
            "                print(f"Warning: failed to load reference spectra from {ref_dir}: {e}")
",
            "
",
            "    # 3. Plot Fluxes
",
            "    fig = plot_flux_trace(
",
            "        sim_time,
",
            "        sim_flux.T[[1, 2]],
",
            "        ref_time=ref_time,
",
            "        ref_fluxes=ref_fluxes[[1, 2]],
",
            "        labels=["Heat", "Momentum"],
",
            "        title=(
",
            "            f"Fluxes: R/LT={cfg.physics.rlt:.1f}, R/LN={cfg.physics.rln:.1f}, "
",
            "            f"shat={cfg.geometry.shat:.1f}, q={cfg.geometry.q:.1f}"
",
            "        )
",
            "    )
",
            "    fig.savefig(f"figs/fluxes_{config_name}.pdf")
",
            "
",
            "    # 4. Plot Spectra Comparison (Time-averaged)
",
            "    if os.path.exists(kx_spec_path) and os.path.exists(ky_spec_path):
",
            "        kx_data = np.load(kx_spec_path)
",
            "        ky_data = np.load(ky_spec_path)
",
            "        
",
            "        kx_spec_hist = kx_data["kx_spec"]
",
            "        ky_spec_hist = ky_data["ky_spec"]
",
            "        
",
            "        kx_spec_avg = np.mean(kx_spec_hist[-avg_count:], axis=0)
",
            "        ky_spec_avg = np.mean(ky_spec_hist[-avg_count:], axis=0)
",
            "        
",
            "        fig_spec = plot_spectra(
",
            "            kx=kx,
",
            "            ky=ky,
",
            "            kx_spec=kx_spec_avg,
",
            "            ky_spec=ky_spec_avg,
",
            "            ref_kx_spec=ref_kx_spec,
",
            "            ref_ky_spec=ref_ky_spec,
",
            "            title=(
",
            "                f"Time-averaged Spectra (last {avg_count}): R/LT={cfg.physics.rlt:.1f}, "
",
            "                f"R/LN={cfg.physics.rln:.1f}, shat={cfg.geometry.shat:.1f}, q={cfg.geometry.q:.1f}"
",
            "            )
",
            "        )
",
            "        fig_spec.savefig(f"figs/spectra_{config_name}.pdf")
",
            "    else:
",
            "        # Fallback to last dump if history not available
",
            "        last_step_files = sorted([f for f in os.listdir(path) if f.startswith("step_") and f.endswith(".npz")])
",
            "        if last_step_files:
",
            "            last_dump = np.load(os.path.join(path, last_step_files[-1]))
",
            "            phi_final = last_dump["phi"]
",
            "            
",
            "            fig_spec = plot_spectra(
",
            "                kx=kx,
",
            "                ky=ky,
",
            "                phi=phi_final,
",
            "                ref_kx_spec=ref_kx_spec,
",
            "                ref_ky_spec=ref_ky_spec,
",
            "                title=(
",
            "                    f"Saturated Spectra (last dump): R/LT={cfg.physics.rlt:.1f}, "
",
            "                    f"R/LN={cfg.physics.rln:.1f}, shat={cfg.geometry.shat:.1f}, q={cfg.geometry.q:.1f}"
",
            "                )
",
            "            )
",
            "            fig_spec.savefig(f"figs/spectra_{config_name}.pdf")
",
        ]
        cell["source"] = new_source
        break

# Write the modified notebook content back to the file
with open(notebook_path, "w") as f:
    json.dump(notebook_content, f, indent=2)
