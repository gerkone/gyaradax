# Experiment kernels

Put active CUDA experiment sources in this directory as `*.cu` files.

Guidelines:

- Keep each file focused on one experiment or kernel family.
- Document the exported C ABI or JAX FFI target name in the source header.
- Keep generated files, fatbins, and binaries in `_build/`, not here.
- Add a JAX reference comparison in `../compare_against_jax.py` before using
  results to guide production changes.
- Promote successful kernels to `gyaradax/backends/cuda_kernels/` with backend
  parity tests.
