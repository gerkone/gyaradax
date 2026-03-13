import os
import time
import numpy as np
import jax
import jax.numpy as jnp
from gyaradax import (
    load_config,
    gkparams_from_config,
    load_geometry,
    load_gkw_dump,
    GKState,
    gkstep_single,
    save_checkpoint,
    get_integrals,
)
from gyaradax.diag import get_diagnostics


def run_advanced():
    # 1. Setup paths
    config_path = "configs/iteration_13.yaml"
    start_dump = "/restricteddata/ukaea/gyrokinetics/raw/iteration_13/100"
    output_dir = "advanced_run_outputs"
    os.makedirs(output_dir, exist_ok=True)

    # 2. Load Config and Geometry
    cfg = load_config(config_path)
    params = gkparams_from_config(cfg)
    geometry = load_geometry(cfg.run.data_dir)
    res = (
        len(geometry["intvp"]),
        len(geometry["intmu"]),
        len(geometry["ints"]),
        len(geometry["kxrh"]),
        len(geometry["krho"]),
    )

    # 3. Load Starting Point (K01) with unified loader
    print(f"Loading start dump from {start_dump}...")
    df_start, info = load_gkw_dump(start_dump, res)
    state = GKState(
        time=jnp.array(info["time"], dtype=jnp.float64),
        step=jnp.array(0, dtype=jnp.int32),
        accumulated_norm_factor=jnp.array(1.0, dtype=jnp.float64),
        window_start_amp=jnp.array(1.0, dtype=jnp.float64),
        last_growth_rate=jnp.array(0.0, dtype=jnp.float64),
    )
    print(f"Loaded start time: {float(state.time):.4f}")

    # 4. Define trajectory (Dump 1 to Dump 200)
    # We execute in one big jitted scan to maximize performance
    total_steps = 40 * 200  # ~8000 steps

    print(f"Starting advanced run for {total_steps} steps...")

    # 5. Define Scan with custom diagnostic collection using diag.py
    def _scan_with_diag(carry, _):
        d, s = carry
        dn, (pn, fl), sn = gkstep_single(d, geometry, params, s)
        # Use the decoupled diagnostic logic from diag.py
        d_out = get_diagnostics(pn, fl, sn)
        return (dn, sn), d_out

    @jax.jit
    def run_trajectory(d0, s0):
        return jax.lax.scan(_scan_with_diag, (d0, s0), None, length=total_steps)

    # 6. Execute with Timing
    print("Compiling and running trajectory...")
    t0 = time.time()
    (final_df, final_state), history = run_trajectory(df_start, state)

    # Force execution for timing
    final_df.block_until_ready()
    runtime = time.time() - t0

    print(f"Run complete in {runtime:.2f}s")
    print(f"Performance: {total_steps/runtime:.2f} steps/s")
    print(f"Final simulation time: {float(final_state.time):.4f}")

    # 7. Save Final State and History
    print(f"Saving results to {output_dir}...")

    phi_f, fluxes_f = get_integrals(final_df, geometry, params=params)
    save_checkpoint(
        os.path.join(output_dir, "final_state.npz"),
        final_df,
        phi_f,
        fluxes_f,
        final_state,
        geometry,
    )

    # Save high-level history (small file)
    history_np = {k: np.array(v) for k, v in history.items()}
    np.savez(os.path.join(output_dir, "history.npz"), **history_np)

    print("All diagnostics saved.")


if __name__ == "__main__":
    run_advanced()
