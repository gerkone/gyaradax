import os
import time
import numpy as np
from gyaradax import load_config, simulate

def validate_time_averaged(config_paths):
    """
    Execute a batch of simulations from YAML configs and validate
    time-averaged fluxes against GKW references.
    """
    for config_path in config_paths:
        print(f"\n{'=' * 88}")
        print(f"Validating Trajectory: {config_path}")
        print(f"{'=' * 88}")

        cfg = load_config(config_path)
        data_dir = cfg.run.data_dir
        start_k_file = os.path.join(data_dir, "K01")

        # run parameters
        # run for 266 big steps, average last 80
        target_big_steps = 266
        dump_interval = getattr(cfg.solver, "dump_interval", 120)
        n_steps = target_big_steps * dump_interval

        output_dir = f"validation_outputs_{cfg.run.name}"

        # execute
        t0 = time.time()
        _, _, perf = simulate(
            config_path,
            output_dir=output_dir,
            resume_k_file=start_k_file,
            n_steps=n_steps,
            checkpoint_interval=dump_interval,
            save_dumps=False,  # Do not dump heavy 5D files
            verbose=True,
        )
        runtime = time.time() - t0

        print(f"\nRun complete in {runtime:.2f}s")
        print(f"Performance: {n_steps/runtime:.2f} steps/s")

        # load history and calculate averages
        history_path = os.path.join(output_dir, "fluxes.npz")
        growth_path = os.path.join(output_dir, "growth.npz")

        if not os.path.exists(history_path):
            print("Error: Diagnostic history not found. Skipping validation.")
            continue

        hist_flux = np.load(history_path)
        hist_growth = np.load(growth_path)

        sim_eflux = hist_flux["fluxes"][:, 1]
        sim_times = hist_growth["time"]

        # average over the last 80 big timesteps
        if len(sim_eflux) < 80:
            print(
                f"Warning: Simulation only produced {len(sim_eflux)} big steps. Averaging all."
            )
            avg_count = len(sim_eflux)
        else:
            avg_count = 80

        sim_eflux_avg = np.mean(sim_eflux[-avg_count:])

        # GKW reference for comparison
        try:
            ref_fluxes = np.loadtxt(os.path.join(data_dir, "fluxes.dat"))
            ref_time = np.loadtxt(os.path.join(data_dir, "time.dat"))

            ref_eflux_samples = []
            for t in sim_times[-avg_count:]:
                idx = np.argmin(np.abs(ref_time - t))
                ref_eflux_samples.append(ref_fluxes[idx, 1])

            ref_eflux_avg = np.mean(ref_eflux_samples)

            err_eflux = np.sqrt(sim_eflux_avg**2 - ref_eflux_avg**2)

            print(f"\nValidation Results for {cfg.run.name}:")
            print(f"  Time-averaged Heat Flux (last {avg_count} steps):")
            print(f"    Simulated: {sim_eflux_avg:.4e}")
            print(f"    Reference: {ref_eflux_avg:.4e}")
            print(f"    Rel. Error: {err_eflux:.2%}")

        except Exception as e:
            print(f"Error during reference comparison: {e}")


if __name__ == "__main__":
    configs = ["configs/iteration_13.yaml", "configs/iteration_8.yaml"]
    validate_time_averaged(configs)
