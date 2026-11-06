# gyaradax in TORAX

`gyaradax.transport` adds two TORAX transport models. Importing it registers them.

- **`ql-gyx`**: quasilinear. At each `rho_match` radius it runs one linear gyaradax solve
  (all ky at once, stopped when the growth rates converge), then applies a saturation rule.
- **`nl-gyx`**: nonlinear. At each `rho_match` radius it runs a nonlinear simulation and
  averages the fluxes over its last blocks, with adiabatic or kinetic electrons. This is ground
  truth for validation, not for routine transport runs.

Both interpolate the fluxes between the radii onto the TORAX grid and convert them from GKW
gyroBohm units (a factor 2 sqrt 2 from GKW's v_th = sqrt(2T/m)).

## Install

The models need TORAX 2. They register through its plugin API (`torax.transport.register_transport_model`) and reuse its quasilinear helpers, so the extra pins a minor release.

```bash
python3.12 -m venv .venv
.venv/bin/pip install -e "path/to/gyaradax[torax]"
.venv/bin/pip install "jax[cuda13]==<the installed jax version>"   # for a GPU
```

## Example: ITER hybrid with ql-gyx

`gyaradax/transport/cases/iterhybrid.py` is TORAX's `iterhybrid_predictor_corrector` example with
ql-gyx in QLKNN's place:

```python
QL = {
    "model_name": "ql-gyx",
    "rho_min": 0.15,
    "rho_max": 0.95,
    "rho_match": (0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9),
    "kinetic": True,  # 32x12x18x191x32, two species
    "rule": "sat2",
    "backend": "cuda",
    "fused_linear": True,
    "n_steps_linear": 404250,  # cold first call: t = 300 at the smallest CFL step
    "warm_start": True,
    "warm_start_steps": 10000,
    "early_stop_block": 250,
    "early_stop_atol": 1e-5,
    "early_stop_rtol": 1e-5,
    "early_stop_min_steps": 2000,
    "early_stop_patience": 24,
}
```

The eight radii run one per GPU on an 8-GPU node. The first transport call is cold and runs to t = 300; every later call starts from the previous call's distributions and runs at most 10000 steps.

Run it as a TORAX config file, or from Python:

```bash
run_torax --config=gyaradax/transport/cases/iterhybrid.py
```

```python
import torax
from gyaradax.transport.cases import iterhybrid

history = torax.run_simulation(torax.ToraxConfig.from_dict(iterhybrid.CONFIG))
output = history.simulation_output_to_xr()
```

Any other TORAX config uses the models the same way, after `import gyaradax.transport`.

## Configuration

| field | models | default | meaning |
|---|---|---|---|
| `rho_match` | both | 0.35, 0.55, 0.75, 0.875 | the radii that run gyaradax |
| `grid_preset` | both | from `kinetic` | `adiabatic` (32x8x16x85x32) or `kinetic` (32x12x18x191x32, two species) |
| `nvpar`, `nmu`, `ns`, `nkx`, `nky`, `ikxspace`, `krhomax` | both | from the preset | flux-tube grid; never below nkx 85, nky 32 |
| `backend`, `mixed_precision`, `fused_linear` | both | `jax`, true, true | `jax` keeps the call differentiable; `cuda` + `fused_linear` is the fastest |
| `linear_arith` | both | `fp64` | CUDA: `ds` runs the linear RHS in double-single arithmetic, faster on weak-FP64 GPUs such as B300 |
| `batching` | both | `devices` | how the radii run: `devices` splits them evenly over the local GPUs and vmaps each share (the last radius is repeated to fill the final share and its result dropped), `vmap` vmaps all on one GPU, `loop` runs one at a time |
| `kinetic`, `em` | both | false | kinetic electrons, A_par (experimental) |
| `collisions` | both | true | linearised Fokker-Planck collisions, with frequencies from the local n_e and T_i (as `coll_nref`, `coll_tref`) and R_major (`coll_rref`) |
| `ikxspace_choices`, `ikxspace_per_shat` | both | kinetic preset: 1, 2, 3, 4, 6, 8, 12, 16; 6.53 | per radius, ikxspace is the largest choice below floor(6.53 abs(s_hat)); empty keeps the fixed `ikxspace`. Each choice adds one prebuilt topology (about 18 MB at the kinetic grid) |
| `diagnostics_path` | both | | one jsonl row per call and radius (padding radii included): local parameters, fluxes, steps, time |
| `rule`, `rule_options` | ql | `sat2` | saturation rule (`sat2`, `canonical`, `sat0`), and its options |
| `calibration`, `cn` | ql | `auto` | calibration name or JSON path (`auto`: the rule's default in `gyaradax/quasilinear/weights/`, `sat2_v3` or `canonical_v3`), with parameters per electron model and flux channel; `cn` instead sets one amplitude for every channel |
| `n_steps_linear`, `early_stop` | ql | 2000, true | cap on RK4 steps, and stopping on converged growth rates |
| `warm_start`, `warm_start_steps` | ql | true, 1000 | start each radius from its final distribution of the previous transport call, kept in host memory (cold when the ikxspace changes), and the step cap of such a warm solve |
| `n_steps_nl`, `nl_tail_blocks`, `nl_block` | nl | 30000, 12, 500 | run length and averaging window |

The local inputs at each radius are R/L_Ti, R/L_Te, R/L_n, q, s_hat, eps, T_e/T_i, n_e and T_i from
the TORAX profiles, clipped into `clip_*`. The flux tube is circular with a pure deuterium plasma:
shaping, impurities, Z_eff and a DT main ion are not modelled yet. Remaining fields (geometry and solver settings, clip boxes,
particle-transport options) are listed in `gyaradax/transport/base.py`.
