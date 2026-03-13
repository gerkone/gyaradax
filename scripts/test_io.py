import os
import shutil
import jax.numpy as jnp
from gyaradax.utils import save_dumps, load_checkpoint
from gyaradax.solver import GKState


def test_io_roundtrip():
    # create dummy data
    df = jnp.ones((2, 2, 2, 2, 2), dtype=jnp.complex128) * (1.0 + 2j)
    phi = jnp.ones((2, 2, 2), dtype=jnp.complex128) * 0.5
    fluxes = (
        jnp.array(0.1, dtype=jnp.float64),
        jnp.array(0.2, dtype=jnp.float64),
        jnp.array(0.3, dtype=jnp.float64),
    )
    state = GKState(
        time=jnp.array(1.5, dtype=jnp.float64),
        step=jnp.array(10, dtype=jnp.int32),
        accumulated_norm_factor=jnp.array(0.8, dtype=jnp.float64),
        window_start_amp=jnp.array(1.1, dtype=jnp.float64),
        last_growth_rate=jnp.array(0.05, dtype=jnp.float64),
    )

    test_dir = "test_io_out"
    if os.path.exists(test_dir):
        shutil.rmtree(test_dir)

    try:
        save_dumps(test_dir, df, phi, fluxes, state, geometry={}, save_dumps=True)
        ckpt_path = os.path.join(test_dir, f"step_{int(state.step):06d}.npz")
        loaded = load_checkpoint(ckpt_path)

        # assertions
        assert jnp.allclose(df, loaded["df"]), "DF mismatch"
        assert jnp.allclose(phi, loaded["phi"]), "Phi mismatch"
        assert jnp.allclose(state.time, loaded["time"]), "Time mismatch"
        assert jnp.allclose(state.step, loaded["step"]), "Step mismatch"
        assert "kx_spec" in loaded, "KX spectrum missing"
        assert "ky_spec" in loaded, "KY spectrum missing"

        # check persistent files
        assert os.path.exists(os.path.join(test_dir, "fluxes.npz"))
        assert os.path.exists(os.path.join(test_dir, "growth.npz"))

        print("I/O Roundtrip test passed.")
    finally:
        if os.path.exists(test_dir):
            shutil.rmtree(test_dir)


if __name__ == "__main__":
    test_io_roundtrip()
