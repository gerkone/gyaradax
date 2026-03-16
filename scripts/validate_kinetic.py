"""long-term kinetic electron trajectory validation.

runs the kinetic solver from K01 for the full trajectory length and compares
time-averaged heat fluxes against GKW reference data.
"""

import os
import re
import time
import numpy as np

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "5")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

jax.config.update("jax_enable_x64", True)

from gyaradax import load_geometry  # noqa: E402
from gyaradax.solver import gksolve, GKState  # noqa: E402
from gyaradax.params import gkparams_from_input_dat  # noqa: E402
from gyaradax.utils import load_gkw_k_dump, K_files  # noqa: E402
from gyaradax.integrals import (
    calculate_fluxes_kinetic,
)  # noqa: E402


KINETIC_DIR = "/restricteddata/ukaea/gyrokinetics/raw/kinetic_electrons"
CASES = [
    "v3_kiteration_991_half_rlt",
]
SHAPE = (32, 8, 16, 85, 32)
N_SPECIES = 2
BLOCK_SIZE = 300  # steps per gksolve call (matches naverage * dump_interval)


def read_dtim(dat_path):
    with open(dat_path) as f:
        text = f.read()
    return float(re.search(r"DTIM\s*=\s*([0-9eE+\-.]+)", text).group(1))


def read_time(dat_path):
    with open(dat_path) as f:
        text = f.read()
    return float(re.search(r"TIME\s*=\s*([0-9eE+\-.]+)", text).group(1))


def validate_case(case_name):
    kin_dir = os.path.join(KINETIC_DIR, case_name)
    print(f"\n{'='*72}")
    print(f"validating: {case_name}")
    print(f"{'='*72}")

    geom = load_geometry(kin_dir)
    ks = K_files(kin_dir)

    # load starting distribution
    start_file = ks[0]
    df = load_gkw_k_dump(os.path.join(kin_dir, start_file), SHAPE, n_species=N_SPECIES)
    t_start = read_time(os.path.join(kin_dir, f"{start_file}.dat"))
    dtim = read_dtim(os.path.join(kin_dir, f"{start_file}.dat"))

    params = gkparams_from_input_dat(
        os.path.join(kin_dir, "input.dat"),
        non_linear=True,
        adiabatic_electrons=False,
        dt=dtim,
    )
    nky = len(geom["krho"])
    state = GKState(
        time=jnp.array(t_start, dtype=jnp.float64),
        step=jnp.array(0, dtype=jnp.int32),
        accumulated_norm_factor=jnp.ones(nky, dtype=jnp.float64),
        window_start_amp=jnp.ones(nky, dtype=jnp.float64),
        last_growth_rate=jnp.zeros(nky, dtype=jnp.float64),
    )

    # reference data
    ref_fluxes = np.loadtxt(os.path.join(kin_dir, "fluxes.dat"))
    ref_times = np.loadtxt(os.path.join(kin_dir, "time.dat"))

    # compute how many blocks to cover ~80% of the trajectory
    total_ref_time = ref_times[-1] - t_start
    target_time = total_ref_time * 0.8
    n_blocks = max(1, int(target_time / (BLOCK_SIZE * dtim)))

    print(f"dt={dtim:.6e}, blocks={n_blocks}, steps/block={BLOCK_SIZE}")
    print(f"total simulated time: {n_blocks * BLOCK_SIZE * dtim:.2f}")

    # run simulation in blocks, collecting eflux diagnostics
    sim_eflux_ion = []
    sim_eflux_elec = []
    sim_times = []
    t0 = time.time()

    gksolve_jit = jax.jit(gksolve, static_argnames="n_steps")

    for block in range(n_blocks):
        df, (phi, fluxes), state = gksolve_jit(
            df, geom, params, state, n_steps=BLOCK_SIZE
        )

        # compute per-species fluxes
        fl = calculate_fluxes_kinetic(geom, df, phi)
        sim_eflux_ion.append(float(fl[0, 1]))
        sim_eflux_elec.append(float(fl[1, 1]))
        sim_times.append(float(state.time))

        if (block + 1) % 10 == 0 or block == 0:
            elapsed = time.time() - t0
            steps_done = (block + 1) * BLOCK_SIZE
            print(
                f"  [{block+1:4d}/{n_blocks}] t={float(state.time):.2f} "
                f"eflux_i={sim_eflux_ion[-1]:.4e} eflux_e={sim_eflux_elec[-1]:.4e} "
                f"({steps_done/elapsed:.1f} steps/s)"
            )

    runtime = time.time() - t0
    total_steps = n_blocks * BLOCK_SIZE
    print(f"\ncompleted in {runtime:.1f}s ({total_steps/runtime:.1f} steps/s)")

    # time-averaged comparison over last 25% of blocks
    avg_start = max(0, len(sim_eflux_ion) - len(sim_eflux_ion) // 4)
    sim_avg_ion = np.mean(sim_eflux_ion[avg_start:])
    sim_avg_elec = np.mean(sim_eflux_elec[avg_start:])

    # match reference at same time points
    ref_eflux_ion = []
    ref_eflux_elec = []
    for t in sim_times[avg_start:]:
        idx = np.argmin(np.abs(ref_times - t))
        ref_eflux_ion.append(ref_fluxes[idx, 1])
        ref_eflux_elec.append(ref_fluxes[idx, 4])

    ref_avg_ion = np.mean(ref_eflux_ion)
    ref_avg_elec = np.mean(ref_eflux_elec)

    print(f"\ntime-averaged eflux (last {len(sim_eflux_ion) - avg_start} blocks):")
    print(
        f"  ion:      sim={sim_avg_ion:.4e}  ref={ref_avg_ion:.4e}  "
        f"rel_err={abs(sim_avg_ion - ref_avg_ion) / max(abs(ref_avg_ion), 1e-15):.2e}"
    )
    print(
        f"  electron: sim={sim_avg_elec:.4e}  ref={ref_avg_elec:.4e}  "
        f"rel_err={abs(sim_avg_elec - ref_avg_elec) / max(abs(ref_avg_elec), 1e-15):.2e}"
    )

    # save results
    out_dir = f"validation_kinetic_{case_name}"
    os.makedirs(out_dir, exist_ok=True)
    np.savez(
        os.path.join(out_dir, "results.npz"),
        sim_times=np.array(sim_times),
        sim_eflux_ion=np.array(sim_eflux_ion),
        sim_eflux_elec=np.array(sim_eflux_elec),
    )
    print(f"results saved to {out_dir}/results.npz")


if __name__ == "__main__":
    for case in CASES:
        validate_case(case)
