import os
import time
import numpy as np
from gyaradax import (
    load_config,
    simulate,
)


def validate_time_averaged(config_paths):
    """
    Execute a batch of simulations from YAML configs and validate
    time-averaged fluxes against GKW references.
    """
    for config_path in config_paths:
        print(f"\n{'='*60}")
        print(f"Validating Trajectory: {config_path}")
        print(f"{'='*60}")

        cfg = load_config(config_path)
        data_dir = cfg.run.data_dir

        # 1. Identify starting K-file (K01 equivalent in this dataset)
        all_files = os.listdir(data_dir)
        numeric_dumps = sorted([int(f) for f in all_files if f.isdigit()])
        if not numeric_dumps:
            print(f"Error: No numeric dumps found in {data_dir}. Skipping.")
            continue

        start_dump = numeric_dumps[0]
        start_k_file = os.path.join(data_dir, str(start_dump))
        print(f"Starting from K-file: {start_k_file}")

        # 2. Setup run parameters
        # Run for 266 big steps, average last 80
        target_big_steps = 266
        dump_interval = getattr(cfg.solver, "dump_interval", 40)
        n_steps = target_big_steps * dump_interval

        output_dir = f"validation_outputs_{cfg.run.name}"

        # 3. Execute Simulation with timing
        t0 = time.time()
        final_df, final_state = simulate(
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

        # 4. Load history and calculate averages
        history_path = os.path.join(output_dir, "fluxes.npz")
        growth_path = os.path.join(output_dir, "growth.npz")

        if not os.path.exists(history_path):
            print("Error: Diagnostic history not found. Skipping validation.")
            continue

        hist_flux = np.load(history_path)
        hist_growth = np.load(growth_path)

        # Heat flux is column 1
        sim_eflux = hist_flux["fluxes"][:, 1]
        sim_times = hist_growth["time"]

        # Average over the last 80 big timesteps
        if len(sim_eflux) < 80:
            print(
                f"Warning: Simulation only produced {len(sim_eflux)} big steps. Averaging all."
            )
            avg_count = len(sim_eflux)
        else:
            avg_count = 80

        sim_eflux_avg = np.mean(sim_eflux[-avg_count:])

        # 5. Load GKW reference for comparison
        try:
            ref_fluxes = np.loadtxt(os.path.join(data_dir, "fluxes.dat"))
            ref_time = np.loadtxt(os.path.join(data_dir, "time.dat"))

            # Map simulated times to reference data
            ref_eflux_samples = []
            for t in sim_times[-avg_count:]:
                idx = np.argmin(np.abs(ref_time - t))
                ref_eflux_samples.append(ref_fluxes[idx, 1])

            ref_eflux_avg = np.mean(ref_eflux_samples)

            # 6. Compare with 1% tolerance
            err_eflux = np.abs(sim_eflux_avg - ref_eflux_avg) / max(
                np.abs(ref_eflux_avg), 1e-12
            )

            print(f"\nValidation Results for {cfg.run.name}:")
            print(f"  Time-averaged Heat Flux (last {avg_count} steps):")
            print(f"    Simulated: {sim_eflux_avg:.4e}")
            print(f"    Reference: {ref_eflux_avg:.4e}")
            print(f"    Rel. Error: {err_eflux:.2%}")

            if err_eflux < 0.01:
                print("  Status: PASSED (within 1% tolerance)")
            else:
                print("  Status: FAILED (exceeds 1% tolerance)")

        except Exception as e:
            print(f"Error during reference comparison: {e}")


if __name__ == "__main__":
    configs = ["configs/iteration_13.yaml", "configs/iteration_8.yaml"]
    validate_time_averaged(configs)
