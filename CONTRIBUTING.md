# Contributing

Contributions are welcome: bug reports, fixes, new physics, faster kernels,
documentation. Open an issue or a pull request on
[GitHub](https://github.com/gerkone/gyaradax), or write to
galletti[at]ml[dot]jku[dot]at.

`gyaradax` is released under the GNU Lesser General Public License v3.0
([LICENSE](https://github.com/gerkone/gyaradax/blob/master/LICENSE)). Contributions are accepted under the same license.

## Running the tests

The suite needs a GPU. Install the `test` extra and run it on one device:

```bash
pip install -e ".[test]"
CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_PREALLOCATE=false python -m pytest tests/ -x -q
```

Most tests compare against GKW reference runs, which are stored on Hugging Face
([gerkone/gyaradax-tests](https://huggingface.co/datasets/gerkone/gyaradax-tests)).
The tests read them from `$GYARADAX_TEST_DATA`, or from
`~/.cache/gyaradax/test_data` when it is not set. When the runs are missing, the
tests that need them skip. Any of these lets the tests download the cases they use:

- the `--download-test-data` flag,
- `GYARADAX_DOWNLOAD_TEST_DATA=1` in the environment,
- `download_test_data = true` in the pytest ini options, for example in your
  `pytest.ini` or under `[tool.pytest.ini_options]`.

`python -m gyaradax.io.reference_data` downloads them explicitly (`--kind`,
`--name` for a subset, `--root` for another location, `--diagnostics-only` for the
JSON diagnostics and configs without the dumps). One case is 1.1 GB (adiabatic) or
2.1 GB (kinetic), 15 GB in total.

The reference runs are short excerpts of GKW runs: three pairs of consecutive
distribution dumps per case, early, mid and late in the run. The tests restart
from the first dump of each pair and compare with the second, about 120 steps
later. They also compare fluxes, spectra and growth rates at the dumps with GKW's
own diagnostics.

Some tests skip for other reasons:

- CUDA backend tests skip without the compiled kernels. `-k cuda` runs only them.
- Sharding tests need two or four GPUs in `CUDA_VISIBLE_DEVICES`.
- The generated electromagnetic cases and the GKW benchmarks (zonal flow) need a
  local GKW checkout and runs: `$GKW_EM_DATA_ROOT` and `gkw_ref/benchmarks`.

After editing `gyaradax/`, run the tests in a fresh Python process: JAX's
compilation cache does not see the change after `importlib.reload`.

## Showing parity

The tests check single evaluations and runs of about a hundred steps. That is not
enough for a change to the solver. Turbulence is chaotic, so two codes that differ
by rounding separate after a few tens of R/v_th, and a small bias only shows in the
statistics of a long run. A pull request that touches the nonlinear or linear
solver shows one of the following.

**Bitwise identical.** For refactors and optimisations that should not change the
numbers. Restart a saturated run, adiabatic and kinetic, from a snapshot, step it
with the old and the new code for thousands of steps, and compare the distribution
functions. They must be identical. Do not use the flux or k_x spectrum diagnostics
for this: their GPU reductions differ at 1e-14 between two runs of the same code.
Turn autotuning off for the comparison.

**Same statistics.** For changes that are not bitwise, such as a different FFT
length, a new arithmetic or a reformulated term. Run adiabatic and kinetic cases
long enough to saturate, and compare with the old code and with the references:

- time-averaged fluxes per species, over the last 30% of the run or a stated
  saturated window, with an autocorrelation-corrected standard error, and a check
  that the flux has stopped growing,
- the time-averaged k_y and k_x spectra over the same window,
- for linear changes, the converged growth rate and frequency of every k_y.

Show the flux traces with the averaging window, and the spectra, as figures in the
pull request. Earlier pull requests show the format.

The dataset holds the references for this, as JSON diagnostics with no
distribution functions:

- **The test cases.** Every test case has a `nonlinear.json` or `linear.json` with
  the diagnostics of its whole GKW run: five adiabatic nonlinear, three adiabatic
  linear and three kinetic nonlinear runs.
- **`long_horizon/`.** Six parameter points, two adiabatic (GKW) and four kinetic
  (`gyaradax`), each with a long nonlinear run and a converged linear run, and
  their `gyaradax` configs.

The JSON files hold the flux and spectrum traces with their time averages and
errors, the converged growth rates and frequencies, the quasilinear weights and
the eigenmode along the field line. A rerun should reproduce the averages within
the stated errors and the converged linear spectrum, not the trajectory.
[`notebooks/validation.ipynb`](https://github.com/gerkone/gyaradax/blob/master/notebooks/validation.ipynb) compares a set of runs
with them, and the
[dataset README](https://huggingface.co/datasets/gerkone/gyaradax-tests/blob/main/long_horizon/README.md)
describes the files. There is no pytest for these: each run takes hours on a GPU.

## Open work

- **Documentation.** More worked examples, notebooks that run from the dataset,
  and the API pages beyond the docstrings.
- **Electromagnetic testing.** A_parallel and B_parallel are checked against GKW
  over short runs and linear cases. Long nonlinear electromagnetic runs against
  GKW, and electromagnetic cases in the Hugging Face dataset, are missing.
- **Bigger runs.** Sharding covers the species, mu and v_parallel axes. Grids that
  do not fit on one node, and strong scaling past a few GPUs, are not covered.
- **Optimisation.** Sharding along s, as in GKW, is the next axis. The blocks per
  GPU get small, so it needs care for strong scaling.
- **Quasilinear.** The saturation rules are calibrated on a limited set of runs.
  More nonlinear references, kinetic electrons and electromagnetic cases would make
  the calibration more general.
- **Transport.** Coupling `gyaradax` to a transport solver, quasilinear or with
  nonlinear runs in the loop.
