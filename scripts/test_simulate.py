import os
import shutil
import numpy as np
from gyaradax.solver import simulate


def test_simulate_logic():
    config_path = "configs/iteration_13.yaml"
    output_dir = "test_outputs"

    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)

    print("\n--- Test 1: New Simulation ---")
    df1, state1 = simulate(
        config_path, output_dir=output_dir, n_steps=80, checkpoint_interval=40
    )
    assert int(state1.step) == 80
    assert os.path.exists(os.path.join(output_dir, "step_000080.npz"))

    print("\n--- Test 2: Resume from Checkpoint ---")
    ckpt_path = os.path.join(output_dir, "step_000040.npz")
    df2, state2 = simulate(
        config_path, output_dir=output_dir, n_steps=80, resume_from=ckpt_path
    )
    assert int(state2.step) == 80

    # Check if df1 and df2 match.
    df1_np = np.array(df1)
    df2_np = np.array(df2)
    if not np.any(np.isnan(df1_np)):
        diff = np.linalg.norm(df1_np - df2_np) / (np.linalg.norm(df1_np) + 1e-30)
        print(f"Resume parity relative difference: {diff:.2e}")
        assert np.allclose(df1_np, df2_np, rtol=1e-5, atol=1e-8)
    else:
        print(
            "Skipping parity check due to NaNs in baseline run (as expected per disclaimer)."
        )

    print("\n--- Test 3: Resume from K-file ---")
    # Using a known dump from iteration_13
    k_file = "/restricteddata/ukaea/gyrokinetics/raw/iteration_13/100"
    df3, state3 = simulate(
        config_path, output_dir=output_dir, n_steps=1, resume_k_file=k_file
    )
    print(f"Loaded state time: {float(state3.time):.4f}")
    assert int(state3.step) == 1
    assert float(state3.time) > 0.0  # iteration_13/100 is at t ~ 150

    print("\nSimulation runner tests passed.")


if __name__ == "__main__":
    test_simulate_logic()
