import os
import shutil
import pytest
import numpy as np

from gyaradax import simulate, load_config


@pytest.mark.skip(
    reason="Long integration test: requires significant compute time for full saturation."
)
def test_simulation_saturation():
    """
    Integration test: Runs a full nonlinear trajectory into the saturated phase,
    and compares the time-averaged fluxes over the final 80 against the GKW reference.
    """
    config_path = "configs/iteration_13.yaml"
    cfg = load_config(config_path)
    data_dir = cfg.run.data_dir

    # Identify available K dumps in the reference directory
    all_files = os.listdir(data_dir)
    dumps = sorted([int(f) for f in all_files if f.isdigit()])
    assert len(dumps) > 0, "No numeric K dumps found in reference directory."

    start_dump = dumps[0]

    dump_interval = getattr(cfg.solver, "dump_interval", 40)
    n_steps = 265 * dump_interval

    output_dir = "integration_test_outputs"
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)

    start_k_file = os.path.join(data_dir, str(start_dump))

    # Run the simulation, skipping heavy 5D/3D state dumps
    _, _ = simulate(
        config_path,
        output_dir=output_dir,
        resume_k_file=start_k_file,
        n_steps=n_steps,
        checkpoint_interval=dump_interval,
        save_dumps=False,  # do not save intermediate df and phi
        verbose=True,
    )

    # simulated history
    fluxes_path = os.path.join(output_dir, "fluxes.npz")
    growth_path = os.path.join(output_dir, "growth.npz")
    assert os.path.exists(fluxes_path), "Fluxes file was not generated."
    assert os.path.exists(growth_path), "Growth file was not generated."

    hist_flux = np.load(fluxes_path)
    hist_growth = np.load(growth_path)

    # fluxes column 1 is heat flux
    sim_eflux = hist_flux["fluxes"][:, 1]

    # reference fluxes.dat (pflux, eflux, vflux)
    ref_fluxes = np.loadtxt(os.path.join(data_dir, "fluxes.dat"))
    ref_time = np.loadtxt(os.path.join(data_dir, "time.dat"))

    # average over the last 80 big timesteps
    sim_times = hist_growth["time"][-80:]
    ref_eflux_samples = []
    for t in sim_times:
        idx = np.argmin(np.abs(ref_time - t))
        ref_eflux_samples.append(ref_fluxes[idx, 1])

    ref_eflux_avg = np.mean(ref_eflux_samples)
    sim_eflux_avg = np.mean(sim_eflux[-80:])

    err_eflux = np.abs(sim_eflux_avg - ref_eflux_avg) / (np.abs(ref_eflux_avg) + 1e-8)
    print(f"Heat flux - Sim: {sim_eflux_avg:.4e}, Ref: {ref_eflux_avg:.4e}")
    assert err_eflux < 0.01, f"Heat flux error {err_eflux:.2%} exceeds 1% tolerance."
