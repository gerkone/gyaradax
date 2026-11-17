# How to contribute

We welcome bug reports, fixes, new physics, faster kernels and documentation. Open an issue or a pull request on [GitHub](https://github.com/gerkone/gyaradax), or write to galletti[at]ml[dot]jku[dot]at.

License: [LGPL-3.0](https://github.com/gerkone/gyaradax/blob/master/LICENSE).

## Running the tests

The suite needs a GPU:

```bash
pip install -e ".[test]"
CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_PREALLOCATE=false python -m pytest tests/ -x -q --download-test-data
```

Most tests compare against GKW reference runs on Hugging Face ([gerkone/gyaradax-tests](https://huggingface.co/datasets/gerkone/gyaradax-tests)). They restart from early, mid and late dumps of each run and compare with the next dump about 120 steps later. The runs are read from `$GYARADAX_TEST_DATA` (default `~/.cache/gyaradax/test_data`). The tests download the cases they use when `--download-test-data`, `GYARADAX_DOWNLOAD_TEST_DATA=1` or the ini option `download_test_data = true` is set, and skip them otherwise. The full set is 15 GB; `hf download gerkone/gyaradax-tests --repo-type dataset --local-dir ~/.cache/gyaradax/test_data` fetches it at once.

CUDA tests need the compiled kernels, sharding tests two or four GPUs, and the generated electromagnetic cases a local GKW checkout (`$GKW_EM_DATA_ROOT`).

## Showing parity

Turbulence is chaotic: two codes that differ by rounding separate after a few tens of $R/v_{th}$. A change to the nonlinear or linear solver therefore shows one of:

- **Bitwise identical results**, for refactors and optimisations: restart a saturated adiabatic and a kinetic run from a snapshot, step both codes for thousands of steps and compare the distribution functions.
- **The same statistics**, for everything else: time-averaged fluxes per species over a saturated window, with an autocorrelation-corrected error, the time-averaged $k_y$ and $k_x$ spectra, and the converged linear growth rates, for adiabatic and kinetic cases, against the old code.

The dataset holds references for the second kind as JSON diagnostics: every test case covers its whole GKW run, and `long_horizon/` adds long adiabatic (GKW) and kinetic (`gyaradax`) runs with their configs. [`notebooks/validation.ipynb`](https://github.com/gerkone/gyaradax/blob/master/notebooks/validation.ipynb) compares a set of runs with them. A rerun should reproduce the averages within their errors, not the trajectory.

## Documentation

The documentation is a Sphinx site in `docs/`, written in Markdown (MyST). Build it with `pip install -e ".[docs]"` and `sphinx-build docs docs/_build`. Each section is a folder with an `index.md` whose `toctree` lists its pages. `docs/api/` is generated from the docstrings (Google style). Figures sit in `docs/_static/figs/<section>/` next to the script that makes them; commit both.

## Open work

- **Documentation:** worked examples and notebooks that run from the dataset.
- **Electromagnetic testing:** long nonlinear $A_\parallel$ and $B_\parallel$ runs against GKW, and electromagnetic cases in the dataset.
- **Bigger runs:** grids that do not fit on one node, and strong scaling past a few GPUs.
- **Optimisation:** sharding along $s$, as in GKW.
- **Quasilinear:** calibrating the saturation rules on more nonlinear references, with kinetic electrons and electromagnetic cases.
- **Transport:** coupling to a transport solver, quasilinear or with nonlinear runs in the loop.
