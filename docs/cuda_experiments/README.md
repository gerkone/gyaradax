# CUDA Experiments

Historical optimization experiments for the CUDA backend. These files
are **not** part of the build — the production kernels live in
`gyaradax/backends/cuda_kernels/`.

Python scripts in this directory are experimental analysis harnesses. They use
the centralized `gyaradax/runtime_config.py` loader pattern before importing
JAX, so device defaults and `XLA_PYTHON_CLIENT_PREALLOCATE=false` match the
main script entrypoints without importing the `gyaradax` package too early.
HLO dump scripts still set their `XLA_FLAGS --xla_dump_to=...` path directly
when `--full-dump` is requested, because that output location is specific to
those experiments.

## Contents

- **`cufft_graph_bracket*.cu`** — iterations on the cuFFT graph-captured
  Poisson bracket (FP32, FP64, mixed precision, direct variants).
- **`bracket_*_cb.cu`** — cuFFT LTO load/store callback prototypes
  (D2Z, Z2Z, versioned iterations from v1 through v5).
- **`apply_vpar*.cu`, `apply_parallel*.cu`** — stencil kernel drafts
  and optimization notes (`apply_parallel_opt.md`).
- **`linear_rhs_fused.cu`, `linear_rhs_vtiled.cu`** — fused linear RHS
  kernel experiments (velocity-tiled and fully fused).
- **`compute_scale_factors.cu`** — dealiasing scale factor kernel.
- **`cuFFT_LTO_example/`** — standalone cuFFT LTO callback example.
- **`test_z2z_cb*`** — host/device test harnesses for Z2Z callbacks.
- **`hlo_dumps/`** — XLA HLO dumps used for performance analysis.
- **`jax_ffi_benchmark.py`** — JAX FFI microbenchmark script.
