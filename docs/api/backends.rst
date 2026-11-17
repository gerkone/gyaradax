Backends
========

The solver calls its field solve, linear RHS and nonlinear bracket through a
:class:`~gyaradax.backends.ops.SolverOps` object. :func:`~gyaradax.backends.create_ops`
returns the pure-JAX implementation or the CUDA one (fused FFI kernels, cuFFT and cuFFTDx).

``gyaradax.backends``
---------------------

.. automodule:: gyaradax.backends

``gyaradax.backends.ops``
-------------------------

.. automodule:: gyaradax.backends.ops

JAX backend
-----------

.. autoclass:: gyaradax.backends._jax.JAXOps
   :members: nonlinear_term_iii, linear_rhs

CUDA backend
------------

.. autoclass:: gyaradax.backends._cuda.CUDAOps
   :members: nonlinear_term_iii, linear_rhs, linear_rhs_from_g, compute_fields

.. autofunction:: gyaradax.backends._cuda.is_available
