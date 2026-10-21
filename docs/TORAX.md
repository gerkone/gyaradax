# gyaradax in TORAX

`gyaradax.transport` adds two TORAX transport models. Importing it registers them.

- **`gyaradax-ql`**: quasilinear. At each `rho_match` radius it runs one linear gyaradax solve
  (all ky at once, stopped when the growth rates converge), then applies a saturation rule.
- **`gyaradax-nl`**: nonlinear. At each `rho_match` radius it runs a nonlinear simulation and
  averages the fluxes over its last blocks. This is ground truth for validation, not for routine
  transport runs.

Both interpolate the fluxes between the radii onto the TORAX grid and convert them from GKW
gyroBohm units (a factor 2 sqrt 2 from GKW's v_th = sqrt(2T/m)).

## Install

The models need TORAX main; the 1.4.3 release on PyPI predates its plugin API.

```bash
python3.12 -m venv .venv
.venv/bin/pip install -e "path/to/gyaradax[torax]"
.venv/bin/pip install "jax[cuda13]==<the installed jax version>"   # for a GPU
```

## Example: ITER hybrid with gyaradax-ql

`gyaradax/transport/cases/iterhybrid.py` is TORAX's `iterhybrid_predictor_corrector` example with
gyaradax-ql in QLKNN's place:

```python
QL = {
    "model_name": "gyaradax-ql",
    "rho_min": 0.25,
    "rho_max": 0.95,
    "rho_match": (0.3, 0.5, 0.7, 0.9),
    "grid_preset": "adiabatic",  # 85x32
    "rule": "sat2",
    "backend": "cuda",
    "fused_linear": True,
}
```

Run it as a TORAX config file, or from Python:

```bash
run_torax --config=gyaradax/transport/cases/iterhybrid.py
```

```python
import torax
from gyaradax.transport.cases import iterhybrid

history = torax.run_simulation(torax.ToraxConfig.from_dict(iterhybrid.CONFIG))
```

Any other TORAX config uses the models the same way, after `import gyaradax.transport`.

## Configuration

| field | models | default | meaning |
|---|---|---|---|
| `rho_match` | both | 0.35, 0.55, 0.75, 0.875 | the radii that run gyaradax |
| `grid_preset` | both | from `kinetic` | `adiabatic` (32x8x16x85x32) or `kinetic` (32x12x18x191x32, two species) |
| `nvpar`, `nmu`, `ns`, `nkx`, `nky`, `ikxspace`, `krhomax` | both | from the preset | flux-tube grid; never below nkx 85, nky 32 |
| `backend`, `mixed_precision`, `fused_linear` | both | `jax`, true, false | `jax` keeps the call differentiable; `cuda` + `fused_linear` is the fastest |
| `batching` | both | `devices` | how the radii run: `devices` splits them evenly over the local GPUs and vmaps each share (the last radius is repeated to fill the final share and its result dropped), `vmap` vmaps all on one GPU, `loop` runs one at a time |
| `kinetic`, `em` | both | false | kinetic electrons, A_par (experimental) |
| `diagnostics_path` | both | | one jsonl row per call and radius (padding radii included): local parameters, fluxes, steps, time |
| `rule`, `rule_options` | ql | `canonical` | saturation rule (`canonical`, `sat0`, `sat2`), and its options |
| `calibration`, `cn` | ql | `auto` | calibration name or JSON path (`auto`: the rule's default in `gyaradax/quasilinear/weights/`); `cn` overrides its amplitude |
| `n_steps_linear`, `early_stop` | ql | 2000, true | cap on RK4 steps, and stopping on converged growth rates |
| `n_steps_nl`, `nl_tail_blocks`, `nl_block` | nl | 30000, 12, 500 | run length and averaging window |

The local inputs at each radius are R/L_Ti, R/L_Te, R/L_n, q, s_hat, eps and T_e/T_i from the
TORAX profiles, clipped into `clip_*`. Remaining fields (geometry and solver settings, clip boxes,
particle-transport options) are listed in `gyaradax/transport/base.py`.
