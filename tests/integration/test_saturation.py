import os
import shutil
import pytest
import numpy as np
import jax.numpy as jnp

from gyaradax import load_config
from gyaradax.simulate import (
    gk_from_config,
    gksimulate,
    default_log,
    _compute_phi_for_init,
)
from gyaradax.solver import GKState, mode_amplitude
from gyaradax.utils import load_gkw_k_dump, read_gkw_dump_time


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

    # Load config and geometry
    df, geometry, params, state, pre = gk_from_config(config_path)

    # Resume from K-file
    n_species = 1
    if not params.adiabatic_electrons:
        n_species = int(jnp.asarray(params.mas).shape[0])
    res = (
        len(geometry["intvp"]),
        len(geometry["intmu"]),
        len(geometry["ints"]),
        len(geometry["kxrh"]),
        len(geometry["krho"]),
    )
    df_k = load_gkw_k_dump(start_k_file, res, n_species=n_species)
    nky = len(geometry["krho"])
    dat_path = start_k_file + ".dat"
    t_start = read_gkw_dump_time(dat_path) if os.path.exists(dat_path) else 0.0
    phi0 = _compute_phi_for_init(df_k, geometry, params)
    amp0 = mode_amplitude(phi0, geometry, params.norm_eps)
    state_k = GKState(
        time=jnp.array(t_start, dtype=jnp.float64),
        step=jnp.array(0, dtype=jnp.int32),
        accumulated_norm_factor=jnp.ones(nky, dtype=jnp.float64),
        window_start_amp=amp0,
        last_growth_rate=jnp.zeros(nky, dtype=jnp.float64),
    )

    # Run the simulation
    gksimulate(
        df_k,
        geometry,
        params,
        state_k,
        n_steps,
        pre=pre,
        output_dir=output_dir,
        checkpoint_interval=dump_interval,
        save_snapshots=False,
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
