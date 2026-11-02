"""JAX FFI bindings of the retired standalone stencil kernels, as they were in ``CUDAOps``.

Reference only: nothing imports this module and the kernels next to it are not built.
To revive them, add the ``.cu`` files to a build, register the ``*_ffi`` handlers and
mix ``LegacyStencilOps`` into ``CUDAOps`` (it uses ``CUDAOps._pack_shift_maps``).
"""

from typing import Tuple

import jax
import jax.numpy as jnp
import numpy as np
from jax import ffi


class LegacyStencilOps:
    def _prepare_parallel_coeffs(self, c, nv, nmu, ns, nkx, nky):
        """Reshape stencil coefficients to (9, nv*nmu, ns, nkx, nky) for FFI."""
        if c.ndim == 2:
            c = c.reshape(9, 1, 1, ns, 1, 1)
        elif c.ndim == 4:
            c = c.reshape(9, nv, nmu, ns, 1, 1)
        elif c.ndim != 6:
            while c.ndim < 6:
                c = c[..., None]
        nv_nmu = nv * nmu
        return (
            jnp.broadcast_to(c, (9, nv, nmu, ns, nkx, nky)).reshape(9, nv_nmu, ns, nkx, nky).copy()
        )


    def _apply_vpar(self, field: jnp.ndarray, coeffs) -> jnp.ndarray:
        """Apply 5-point vpar stencil via CUDA kernel."""
        nv = field.shape[0]
        inner_size = field.size // nv
        return ffi.ffi_call(
            "apply_vpar_stencil_ffi",
            [jax.ShapeDtypeStruct(field.shape, field.dtype)],
            vmap_method="sequential",
        )(
            field,
            c0=float(coeffs[0]),
            c1=float(coeffs[1]),
            c2=float(coeffs[2]),
            c3=float(coeffs[3]),
            c4=float(coeffs[4]),
            nv=np.int32(nv),
            inner_size=np.int32(inner_size),
        )[0]

    def _apply_vpar_dual(
        self, field: jnp.ndarray, coeffs_d1, coeffs_d4
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Apply d1 and d4 vpar stencils in a single fused kernel."""
        nv = field.shape[0]
        inner_size = field.size // nv
        out = ffi.ffi_call(
            "apply_vpar_dual_stencil_ffi",
            [
                jax.ShapeDtypeStruct(field.shape, field.dtype),
                jax.ShapeDtypeStruct(field.shape, field.dtype),
            ],
            vmap_method="sequential",
        )(
            field,
            c0_d1=float(coeffs_d1[0]),
            c1_d1=float(coeffs_d1[1]),
            c2_d1=float(coeffs_d1[2]),
            c3_d1=float(coeffs_d1[3]),
            c4_d1=float(coeffs_d1[4]),
            c0_d4=float(coeffs_d4[0]),
            c1_d4=float(coeffs_d4[1]),
            c2_d4=float(coeffs_d4[2]),
            c3_d4=float(coeffs_d4[3]),
            c4_d4=float(coeffs_d4[4]),
            nv=np.int32(nv),
            inner_size=np.int32(inner_size),
        )
        return out[0], out[1]

    def _apply_parallel(self, field: jnp.ndarray, coeffs: jnp.ndarray) -> jnp.ndarray:
        """Apply 9-point parallel stencil via CUDA kernel."""
        nv, nmu, ns, nkx, nky = field.shape
        c_1d = self._prepare_parallel_coeffs(coeffs, nv, nmu, ns, nkx, nky).reshape(-1)
        field_b = jnp.broadcast_to(field, (nv, nmu, ns, nkx, nky)).copy()
        packed_maps = self._pack_shift_maps()
        return ffi.ffi_call(
            "apply_parallel_ffi",
            [jax.ShapeDtypeStruct(field_b.shape, field_b.dtype)],
            vmap_method="sequential",
        )(
            field_b,
            c_1d,
            packed_maps,
            nv_nmu=np.int32(nv * nmu),
            nkx=np.int32(nkx),
            ns=np.int32(ns),
            nky=np.int32(nky),
            nmu=np.int32(nmu),
        )[0]

    def _apply_parallel_dual(
        self,
        field1: jnp.ndarray,
        field2: jnp.ndarray,
        coeffs1: jnp.ndarray,
        coeffs2: jnp.ndarray,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Apply parallel stencils to two fields in a single fused kernel."""
        nv1, nmu1, ns1, nkx1, nky1 = field1.shape
        nv2, nmu2, ns2, nkx2, nky2 = field2.shape
        assert ns1 == ns2 and nkx1 == nkx2 and nky1 == nky2, (
            f"spatial mismatch: field1={(ns1, nkx1, nky1)}, field2={(ns2, nkx2, nky2)}"
        )
        nv, nmu = max(nv1, nv2), max(nmu1, nmu2)
        ns, nkx, nky = ns1, nkx1, nky1

        target_shape = (nv, nmu, ns, nkx, nky)
        f1_b = jnp.broadcast_to(field1, target_shape).copy()
        f2_b = jnp.broadcast_to(field2, target_shape).copy()

        c1_1d = self._prepare_parallel_coeffs(coeffs1, nv, nmu, ns, nkx, nky).reshape(-1)
        c2_1d = self._prepare_parallel_coeffs(coeffs2, nv, nmu, ns, nkx, nky).reshape(-1)
        packed_maps = self._pack_shift_maps()

        out = ffi.ffi_call(
            "apply_parallel_dual_ffi",
            [
                jax.ShapeDtypeStruct(target_shape, field1.dtype),
                jax.ShapeDtypeStruct(target_shape, field2.dtype),
            ],
            vmap_method="sequential",
        )(
            f1_b,
            f2_b,
            c1_1d,
            c2_1d,
            packed_maps,
            nv_nmu=np.int32(nv * nmu),
            nkx=np.int32(nkx),
            ns=np.int32(ns),
            nky=np.int32(nky),
            nmu=np.int32(nmu),
        )
        return out[0], out[1]
