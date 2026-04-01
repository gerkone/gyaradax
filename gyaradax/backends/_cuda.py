"""CUDA backend for solver operations using custom FFI kernels.

provides fused stencil application and nonlinear bracket kernels
compiled from cuda_augmentations/. falls back gracefully if the
shared library is not compiled.
"""

import ctypes
from pathlib import Path
from typing import Dict, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np
from jax import ffi

from gyaradax import stencils
from gyaradax.backends.ops import SolverOps
from gyaradax.types import GKPre

LIB_PATH = Path(__file__).parent.parent.parent / "cuda_augmentations" / "liblto_bracket.so"
_ffi_registered = False


def _register_ffi():
    global _ffi_registered
    if _ffi_registered:
        return True

    if not LIB_PATH.exists():
        return False

    try:
        _lib = ctypes.cdll.LoadLibrary(str(LIB_PATH))
    except Exception:
        return False

    targets = {
        "apply_vpar_stencil_ffi": _lib.apply_vpar_stencil_ffi,
        "apply_vpar_dual_stencil_ffi": _lib.apply_vpar_dual_stencil_ffi,
        "apply_parallel_ffi": _lib.apply_parallel_ffi,
        "apply_parallel_dual_ffi": _lib.apply_parallel_dual_ffi,
        "lto_fft_bracket_v2_ffi": _lib.lto_fft_bracket_v2_ffi,
        "lto_fft_bracket_v4_ffi": _lib.lto_fft_bracket_v4_ffi,
        "cufft_graph_bracket_ffi": _lib.cufft_graph_bracket_ffi,
        "linear_rhs_vtiled_ffi": _lib.linear_rhs_vtiled_ffi,
        "linear_rhs_fused_ffi": _lib.linear_rhs_fused_ffi,
    }

    for name, symbol in targets.items():
        try:
            ffi.register_ffi_target(name, ffi.pycapsule(symbol), platform="CUDA")
        except Exception:
            pass

    _ffi_registered = True
    return True


def is_available():
    """check if CUDA FFI kernels are compiled and a GPU is present."""
    return LIB_PATH.exists() and jax.devices("cuda")


@jax.tree_util.register_pytree_node_class
class CUDAOps(SolverOps):
    """CUDA backend using custom FFI kernels for stencils and FFT bracket."""

    def __init__(self, pre: GKPre, field_template: Optional[jnp.ndarray] = None):
        _register_ffi()
        self.pre = pre
        if field_template is not None:
            self.template_meta = (field_template.shape, field_template.dtype)
        else:
            self.template_meta = (None, None)

    def tree_flatten(self):
        return (self.pre,), self.template_meta

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        (pre,) = children
        obj = cls(pre, None)
        obj.template_meta = aux_data
        return obj

    def _apply_vpar(self, field: jnp.ndarray, coeffs) -> jnp.ndarray:
        """apply 5-point vpar stencil via CUDA kernel."""
        nv = field.shape[0]
        inner_size = field.size // nv

        return ffi.ffi_call(
            "apply_vpar_stencil_ffi",
            [jax.ShapeDtypeStruct(field.shape, field.dtype)],
        )(
            field,
            c0=float(coeffs[0]),
            c1=float(coeffs[1]),
            c2=float(coeffs[2]),
            c3=float(coeffs[3]),
            c4=float(coeffs[4]),
            nv=np.int32(nv),
            inner_size=np.int32(inner_size),
        )[
            0
        ]

    def _apply_vpar_dual(
        self, field: jnp.ndarray, coeffs_d1, coeffs_d4
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """apply d1 and d4 vpar stencils in a single fused kernel."""
        nv = field.shape[0]
        inner_size = field.size // nv

        return ffi.ffi_call(
            "apply_vpar_dual_stencil_ffi",
            [
                jax.ShapeDtypeStruct(field.shape, field.dtype),
                jax.ShapeDtypeStruct(field.shape, field.dtype),
            ],
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

    def _apply_parallel(self, field: jnp.ndarray, coeffs: jnp.ndarray) -> jnp.ndarray:
        """apply 9-point parallel stencil via CUDA kernel."""
        nv, nmu, ns, nkx, nky = field.shape
        nv_nmu = np.int32(nv * nmu)

        def prepare_c(c):
            if c.ndim == 2:
                c = c.reshape(9, 1, 1, ns, 1, 1)
            elif c.ndim == 4:
                c = c.reshape(9, nv, nmu, ns, 1, 1)
            elif c.ndim != 6:
                while c.ndim < 6:
                    c = c[..., None]
            return (
                jnp.broadcast_to(c, (9, nv, nmu, ns, nkx, nky))
                .reshape(9, nv_nmu, ns, nkx, nky)
                .copy()
            )

        c_1d = prepare_c(coeffs).reshape(-1)
        field_b = jnp.broadcast_to(field, (nv, nmu, ns, nkx, nky)).copy()

        valid_jax = jnp.array(self.pre["valid_shift"])
        s_map_jax = jnp.where(valid_jax, self.pre["s_shift"], -1).astype(jnp.int32)
        kx_map_jax = jnp.array(self.pre["kx_shift"]).astype(jnp.int32)
        packed_maps = jnp.stack([s_map_jax, kx_map_jax], axis=-1).copy()

        return ffi.ffi_call(
            "apply_parallel_ffi",
            [jax.ShapeDtypeStruct(field_b.shape, field_b.dtype)],
        )(
            field_b,
            c_1d,
            packed_maps,
            nv_nmu=np.int32(nv_nmu),
            nkx=np.int32(nkx),
            ns=np.int32(ns),
            nky=np.int32(nky),
            nmu=np.int32(nmu),
        )[
            0
        ]

    def _apply_parallel_dual(
        self,
        field1: jnp.ndarray,
        field2: jnp.ndarray,
        coeffs1: jnp.ndarray,
        coeffs2: jnp.ndarray,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """apply parallel stencils to two fields in a single fused kernel."""
        nv1, nmu1, ns1, nkx1, nky1 = field1.shape
        nv2, nmu2, ns2, nkx2, nky2 = field2.shape
        assert (
            ns1 == ns2 and nkx1 == nkx2 and nky1 == nky2
        ), f"spatial mismatch: field1={(ns1, nkx1, nky1)}, field2={(ns2, nkx2, nky2)}"
        nv, nmu = max(nv1, nv2), max(nmu1, nmu2)
        ns, nkx, nky = ns1, nkx1, nky1

        target_shape = (nv, nmu, ns, nkx, nky)
        f1_b = jnp.broadcast_to(field1, target_shape).copy()
        f2_b = jnp.broadcast_to(field2, target_shape).copy()

        nv_nmu = np.int32(nv * nmu)

        def prepare_c(c):
            if c.ndim == 2:
                c = c.reshape(9, 1, 1, ns, 1, 1)
            elif c.ndim == 4:
                c = c.reshape(9, nv, nmu, ns, 1, 1)
            elif c.ndim != 6:
                while c.ndim < 6:
                    c = c[..., None]
            return (
                jnp.broadcast_to(c, (9, nv, nmu, ns, nkx, nky))
                .reshape(9, nv_nmu, ns, nkx, nky)
                .copy()
            )

        c1_1d = prepare_c(coeffs1).reshape(-1)
        c2_1d = prepare_c(coeffs2).reshape(-1)

        valid_jax = jnp.array(self.pre["valid_shift"])
        s_map_jax = jnp.where(valid_jax, self.pre["s_shift"], -1).astype(jnp.int32)
        kx_map_jax = jnp.array(self.pre["kx_shift"]).astype(jnp.int32)
        packed_maps = jnp.stack([s_map_jax, kx_map_jax], axis=-1).copy()

        return ffi.ffi_call(
            "apply_parallel_dual_ffi",
            [
                jax.ShapeDtypeStruct(target_shape, field1.dtype),
                jax.ShapeDtypeStruct(target_shape, field2.dtype),
            ],
        )(
            f1_b,
            f2_b,
            c1_1d,
            c2_1d,
            packed_maps,
            nv_nmu=np.int32(nv_nmu),
            nkx=np.int32(nkx),
            ns=np.int32(ns),
            nky=np.int32(nky),
            nmu=np.int32(nmu),
        )

    def _linear_rhs_fused(
        self,
        df: jnp.ndarray,
        phi: jnp.ndarray,
        pre: Dict[str, jnp.ndarray],
        params_dvp: float,
        params_disp_vp: float,
        params_drive_scale: float,
        target_name: str = "linear_rhs_fused_ffi",
    ) -> jnp.ndarray:
        """fused linear RHS kernel for 5D data."""
        nv, nmu, ns, nkx, nky = df.shape
        nv_nmu = np.int32(nv * nmu)
        V_TILE = 8

        if target_name == "linear_rhs_vtiled_ffi" and nv_nmu % V_TILE != 0:
            target_name = "linear_rhs_fused_ffi"

        f_b = df.copy()
        phi_b = phi.copy()

        bessel = jnp.broadcast_to(pre["bessel"].squeeze(), (nmu, ns, nkx, nky)).copy()

        c_upar_in = pre["s_total_upar"]
        if c_upar_in.ndim == 6 and c_upar_in.shape[2] > 1:
            c_upar_in = c_upar_in[:, :, 0:1, ...]
        elif c_upar_in.ndim == 7 and c_upar_in.shape[3] > 1:
            c_upar_in = c_upar_in[:, :, :, 0:1, ...]

        c_upar = jnp.broadcast_to(c_upar_in, (9, nv, 1, ns, nkx, nky)).copy()
        c_t7 = jnp.broadcast_to(pre["s_total_t7"], (9, nv, nmu, ns, nkx, nky)).copy()

        utrap = jnp.broadcast_to(pre["utrap"].squeeze(), (nmu, ns)).copy()
        abs_vp = jnp.broadcast_to(pre["abs_dum2_vp"].squeeze(), (nmu, ns)).copy()

        drift_x = jnp.broadcast_to(pre["drift_x"].squeeze(), (nv, nmu, ns)).copy()
        drift_y = jnp.broadcast_to(pre["drift_y"].squeeze(), (nv, nmu, ns)).copy()
        fmaxwl = jnp.broadcast_to(pre["fmaxwl"].squeeze(), (nv, nmu, ns)).copy()
        dmaxwel = jnp.broadcast_to(pre["dmaxwel_fm_ek"].squeeze(), (nv, nmu, ns, nky)).copy()

        hyper = jnp.broadcast_to(pre["hyper"].squeeze(), (ns, nkx, nky)).copy()
        kx_vals = pre["kx_b"].reshape(-1)[:nkx].copy()
        ky_vals = pre["ky_b"].reshape(-1)[:nky].copy()

        valid_jax = jnp.array(pre["valid_shift"])
        s_map_jax = jnp.where(valid_jax, pre["s_shift"], -1).astype(jnp.int32)
        kx_map_jax = jnp.array(pre["kx_shift"]).astype(jnp.int32)
        packed_maps = jnp.stack([s_map_jax, kx_map_jax], axis=-1).reshape(9, ns, nkx, nky, 2).copy()

        d1 = stencils.VPAR_D1
        d4 = stencils.VPAR_D4

        attrs = dict(
            nv=np.int32(nv),
            nmu=np.int32(nmu),
            ns=np.int32(ns),
            nkx=np.int32(nkx),
            nky=np.int32(nky),
            nv_nmu=nv_nmu,
            c_d1_0=float(d1[0]),
            c_d1_1=float(d1[1]),
            c_d1_2=float(d1[2]),
            c_d1_3=float(d1[3]),
            c_d1_4=float(d1[4]),
            c_d4_0=float(d4[0]),
            c_d4_1=float(d4[1]),
            c_d4_2=float(d4[2]),
            c_d4_3=float(d4[3]),
            c_d4_4=float(d4[4]),
            dvp=float(params_dvp),
            disp_vp=float(params_disp_vp),
            drive_scale=float(params_drive_scale),
            signz0=float(pre["signz0"]),
            tmp0=float(pre["tmp0"]),
        )
        if target_name == "linear_rhs_vtiled_ffi":
            attrs["v_tile"] = np.int32(V_TILE)

        _register_ffi()
        return ffi.ffi_call(target_name, [jax.ShapeDtypeStruct(df.shape, df.dtype)])(
            f_b,
            phi_b,
            bessel,
            c_upar,
            c_t7,
            packed_maps,
            utrap,
            abs_vp,
            drift_x,
            drift_y,
            dmaxwel,
            fmaxwl,
            hyper,
            kx_vals,
            ky_vals,
            **attrs,
        )[0]

    def linear_rhs(
        self,
        df: jnp.ndarray,
        phi: jnp.ndarray,
        geometry: Dict[str, jnp.ndarray],
        params,
        pre: Dict[str, jnp.ndarray],
    ) -> jnp.ndarray:
        """fused linear RHS for single or multi-species."""
        if df.ndim == 6:
            nsp, nv, nmu, ns, nkx, nky = df.shape
            df_5d = df.reshape(nsp * nv * nmu, 1, ns, nkx, nky)

            uni_pre = pre.copy()

            def r6to5(arr):
                if arr.ndim == 6:
                    return arr.reshape(nsp * nv * nmu, 1, ns, nkx, nky)
                elif arr.ndim == 5:
                    return jnp.broadcast_to(arr[None, ...], (nsp, nv, nmu, ns, nkx, nky)).reshape(
                        nsp * nv * nmu, 1, ns, nkx, nky
                    )
                return arr

            def r4to3(arr):
                return arr.reshape(nsp * nv * nmu, 1, ns)

            uni_pre["bessel"] = r6to5(pre["bessel"])
            uni_pre["s_total_upar"] = r6to5(pre["s_total_upar"])
            uni_pre["s_total_t7"] = r6to5(pre["s_total_t7"])
            uni_pre["fmaxwl"] = r4to3(pre["fmaxwl"])
            uni_pre["dmaxwel_fm_ek"] = pre["dmaxwel_fm_ek"].reshape(nsp * nv * nmu, 1, ns, nky)
            uni_pre["drift_x"] = r4to3(pre["drift_x"])
            uni_pre["drift_y"] = r4to3(pre["drift_y"])

            utrap_4d = jnp.broadcast_to(pre["utrap"][:, None, :, :], (nsp, nv, nmu, ns))
            abs_vp_4d = jnp.broadcast_to(pre["abs_dum2_vp"][:, None, :, :], (nsp, nv, nmu, ns))
            uni_pre["utrap"] = utrap_4d.reshape(nsp * nv * nmu, 1, ns)
            uni_pre["abs_dum2_vp"] = abs_vp_4d.reshape(nsp * nv * nmu, 1, ns)

            if jnp.unique(pre["signz0"]).size > 1 or jnp.unique(pre["tmp0"]).size > 1:
                return None

            uni_pre["signz0"] = pre["signz0"][0]
            uni_pre["tmp0"] = pre["tmp0"][0]

            out_5d = self._linear_rhs_fused(
                df_5d, phi, uni_pre, params.dvp, params.disp_vp, params.drive_scale
            )
            return out_5d.reshape(nsp, nv, nmu, ns, nkx, nky)
        else:
            return self._linear_rhs_fused(
                df, phi, pre, params.dvp, params.disp_vp, params.drive_scale
            )

    def nonlinear_term_iii(
        self,
        df: jnp.ndarray,
        phi: jnp.ndarray,
        geometry: Dict[str, jnp.ndarray],
        *,
        efun_sign: float = 1.0,
        fft_prefactor: complex = 1.0 + 0.0j,
        exclude_zero_mode: bool = True,
        mixed_precision: bool = True,
        bessel: Optional[jnp.ndarray] = None,
    ) -> jnp.ndarray:
        """nonlinear bracket via CUDA graph-captured cuFFT pipeline.

        uses z2z 2-for-1 packing with phi at its natural (nmu*ns) batch
        size, avoiding the nvpar duplication that dominates memory bandwidth.
        """
        pre = self.pre
        mrad, mphi = pre["nl_mrad"], pre["nl_mphi"]
        nv, nmu, ns, nkx, nky = df.shape

        jind = pre["nl_jind"]
        inverse_jind = jnp.full((mrad,), -1, dtype=jnp.int32)
        inverse_jind = inverse_jind.at[jind].set(jnp.arange(jind.shape[0], dtype=jnp.int32))

        kx_vec = pre["nl_kx2d"][:, 0]
        ky_vec = pre["nl_ky2d"][0, :]
        dum_s = pre["nl_dum_s"]

        batch_total = nv * nmu * ns
        df_flat = df.reshape(-1, nkx, nky) * efun_sign

        p_b = phi.reshape(1, 1, ns, nkx, nky)
        if bessel is None:
            bessel = pre["bessel"]
        p_phi = (bessel * p_b).reshape(-1, nkx, nky)

        _register_ffi()
        out_raw = ffi.ffi_call(
            "cufft_graph_bracket_ffi",
            jax.ShapeDtypeStruct((batch_total, nkx, nky), jnp.complex128),
        )(
            df_flat,
            p_phi,
            kx_vec,
            ky_vec,
            jnp.asarray(jind, dtype=jnp.int32),
            inverse_jind,
            dum_s,
            batch=np.int32(batch_total // ns),
            mrad=np.int32(mrad),
            mphi=np.int32(mphi),
            nkx=np.int32(nkx),
            nky=np.int32(nky),
            nspec=np.int32(ns),
            ixzero=np.int32(pre["ixzero"]),
            iyzero=np.int32(pre["iyzero"]),
        )

        fft_scale = pre["nl_fft_scale"]
        nl_5d = (fft_prefactor * fft_scale * out_raw).reshape(df.shape)

        if exclude_zero_mode:
            ixzero, iyzero = pre["ixzero"], pre["iyzero"]
            return nl_5d.at[:, :, :, ixzero, iyzero].set(0.0)
        return nl_5d
