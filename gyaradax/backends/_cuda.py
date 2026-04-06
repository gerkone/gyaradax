"""CUDA backend for solver operations using custom FFI kernels.

Provides fused stencil application and nonlinear bracket kernels
from cuda_kernels/. Falls back gracefully if the shared library
is not compiled.
"""

import ctypes
import logging
from pathlib import Path
from typing import Dict, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np
from jax import ffi

from gyaradax import stencils
from gyaradax.backends.ops import SolverOps
from gyaradax.types import GKPre

log = logging.getLogger(__name__)

_CUDA_KERNELS_DIR = Path(__file__).parent / "cuda_kernels"
LIB_PATH = _CUDA_KERNELS_DIR / "libgyaradax_cuda.so"
_ffi_registered = False


def _register_ffi():
    global _ffi_registered
    if _ffi_registered:
        return True

    if not LIB_PATH.exists():
        return False

    try:
        _lib = ctypes.cdll.LoadLibrary(str(LIB_PATH))
    except (OSError, AttributeError):
        return False

    targets = {
        "apply_vpar_stencil_ffi": _lib.apply_vpar_stencil_ffi,
        "apply_vpar_dual_stencil_ffi": _lib.apply_vpar_dual_stencil_ffi,
        "apply_parallel_ffi": _lib.apply_parallel_ffi,
        "apply_parallel_dual_ffi": _lib.apply_parallel_dual_ffi,
        # Nonlinear bracket kernels (production variants)
        "cufft_graph_bracket_true_fp32_ffi": _lib.cufft_graph_bracket_true_fp32_ffi,
        "cufft_graph_bracket_fp64_ffi": _lib.cufft_graph_bracket_fp64_ffi,
        "linear_rhs_vtiled_ffi": _lib.linear_rhs_vtiled_ffi,
        "linear_rhs_fused_ffi": _lib.linear_rhs_fused_ffi,
    }

    for name, symbol in targets.items():
        try:
            ffi.register_ffi_target(name, ffi.pycapsule(symbol), platform="CUDA")
        except (AttributeError, RuntimeError):
            pass

    _ffi_registered = True
    return True


def is_available():
    """Check if CUDA FFI kernels are compiled and a GPU is present."""
    return LIB_PATH.exists() and bool(jax.devices("cuda"))


# velocity tiling factor for the vtiled linear RHS kernel
_V_TILE = 8


@jax.tree_util.register_pytree_node_class
class CUDAOps(SolverOps):
    """CUDA backend using custom FFI kernels for stencils and FFT bracket.
    
    Note: CUDA backend is Z2Z (complex-to-complex) only. The use_z2z flag
    is ignored for CUDA operations and will emit a warning if set to True.
    """

    def __init__(self, pre: GKPre, use_z2z: bool = False, mixed_precision: bool = True):
        _register_ffi()
        if use_z2z:
            log.warning("CUDA backend: use_z2z=True ignored (CUDA is Z2Z-only)")
        
        super().__init__(pre, use_z2z, mixed_precision)

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

    def _pack_shift_maps(self):
        """Pack precomputed shift maps into (9, ns, nkx, nky, 2) int32 array."""
        valid_jax = jnp.array(self.pre["valid_shift"])
        s_map_jax = jnp.where(valid_jax, self.pre["s_shift"], -1).astype(jnp.int32)
        kx_map_jax = jnp.array(self.pre["kx_shift"]).astype(jnp.int32)
        return jnp.stack([s_map_jax, kx_map_jax], axis=-1).copy()

    def _apply_vpar(self, field: jnp.ndarray, coeffs) -> jnp.ndarray:
        """Apply 5-point vpar stencil via CUDA kernel."""
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
        """Apply d1 and d4 vpar stencils in a single fused kernel."""
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
        """Apply 9-point parallel stencil via CUDA kernel."""
        nv, nmu, ns, nkx, nky = field.shape
        c_1d = self._prepare_parallel_coeffs(coeffs, nv, nmu, ns, nkx, nky).reshape(-1)
        field_b = jnp.broadcast_to(field, (nv, nmu, ns, nkx, nky)).copy()
        packed_maps = self._pack_shift_maps()
        return ffi.ffi_call(
            "apply_parallel_ffi",
            [jax.ShapeDtypeStruct(field_b.shape, field_b.dtype)],
        )(
            field_b,
            c_1d,
            packed_maps,
            nv_nmu=np.int32(nv * nmu),
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
        """Apply parallel stencils to two fields in a single fused kernel."""
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

        c1_1d = self._prepare_parallel_coeffs(coeffs1, nv, nmu, ns, nkx, nky).reshape(-1)
        c2_1d = self._prepare_parallel_coeffs(coeffs2, nv, nmu, ns, nkx, nky).reshape(-1)
        packed_maps = self._pack_shift_maps()

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
            nv_nmu=np.int32(nv * nmu),
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
        """Fused linear RHS kernel for 5D data.

        Dispatches to either the basic fused kernel or the velocity-tiled
        variant depending on target_name. Falls back to fused if the
        velocity dimension is not divisible by _V_TILE.
        """
        nv, nmu, ns, nkx, nky = df.shape
        nv_nmu = np.int32(nv * nmu)

        # vtiled requires nv*nmu divisible by tile size
        if target_name == "linear_rhs_vtiled_ffi" and nv_nmu % _V_TILE != 0:
            target_name = "linear_rhs_fused_ffi"

        # reshape precomputed arrays to expected shapes for FFI
        bessel = pre["bessel"].reshape(nmu, ns, nkx, nky).copy()

        c_upar_in = pre["s_total_upar"]
        if c_upar_in.ndim == 6 and c_upar_in.shape[2] > 1:
            c_upar_in = c_upar_in[:, :, 0:1, ...]
        elif c_upar_in.ndim == 7 and c_upar_in.shape[3] > 1:
            c_upar_in = c_upar_in[:, :, :, 0:1, ...]
        c_upar = jnp.broadcast_to(c_upar_in, (9, nv, 1, ns, nkx, nky)).copy()
        c_t7 = jnp.broadcast_to(pre["s_total_t7"], (9, nv, nmu, ns, nkx, nky)).copy()

        utrap = pre["utrap"].reshape(nmu, ns).copy()
        abs_vp = pre["abs_dum2_vp"].reshape(nmu, ns).copy()
        drift_x = pre["drift_x"].reshape(nv, nmu, ns).copy()
        drift_y = pre["drift_y"].reshape(nv, nmu, ns).copy()
        fmaxwl = pre["fmaxwl"].reshape(nv, nmu, ns).copy()
        dmaxwel = pre["dmaxwel_fm_ek"].reshape(nv, nmu, ns, nky).copy()

        hyper = jnp.broadcast_to(pre["hyper"].squeeze(), (ns, nkx, nky)).copy()
        kx_vals = pre["kx_b"].reshape(-1)[:nkx].copy()
        ky_vals = pre["ky_b"].reshape(-1)[:nky].copy()

        packed_maps = self._pack_shift_maps().reshape(9, ns, nkx, nky, 2).copy()

        # signz0/tmp0 are buffer args (F64) in the kernel, not scalar attrs
        signz0_buf = jnp.asarray(pre["signz0"], dtype=jnp.float64).reshape(1)
        tmp0_buf   = jnp.asarray(pre["tmp0"],   dtype=jnp.float64).reshape(1)

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
        )
        if target_name == "linear_rhs_vtiled_ffi":
            attrs["v_tile"] = np.int32(_V_TILE)

        _register_ffi()
        return ffi.ffi_call(target_name, [jax.ShapeDtypeStruct(df.shape, df.dtype)])(
            df.copy(),
            phi.copy(),
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
            signz0_buf,
            tmp0_buf,
            **attrs,
        )[0]

    def _linear_rhs_kinetic_loop(
        self,
        df: jnp.ndarray,
        phi: jnp.ndarray,
        params,
        pre: Dict[str, jnp.ndarray],
    ) -> jnp.ndarray:
        """Handle 6D kinetic df with non-uniform species params.
        
        Loops over species at Python level, calling fused kernel per-species
        with correct scalar signz0[i] and tmp0[i]. Results stacked to 6D.
        
        Note: This incurs ~5-10% overhead vs single fused call (2 kernel launches
        for typical 2-species case) but correctly handles kinetic electrons with
        different species parameters.
        """
        nsp = df.shape[0]
        results = []
        
        for i in range(nsp):
            # Extract 5D slice for this species
            df_sp = df[i]
            
            # Build species-specific pre dict with SCALAR params for kernel
            # Kernel expects 1-element buffer (const double*), not Python scalar
            sp_pre = {
                "bessel": pre["bessel"][i],
                "s_total_upar": pre["s_total_upar"][:, i],
                "s_total_t7": pre["s_total_t7"][:, i],
                "fmaxwl": pre["fmaxwl"][i],
                "dmaxwel_fm_ek": pre["dmaxwel_fm_ek"][i],
                "drift_x": pre["drift_x"][i],
                "drift_y": pre["drift_y"][i],
                "utrap": pre["utrap"][i],
                "abs_dum2_vp": pre["abs_dum2_vp"][i],
                "signz0": jnp.asarray(pre["signz0"][i]).reshape(1),  # 1-element buffer for kernel
                "tmp0": jnp.asarray(pre["tmp0"][i]).reshape(1),      # 1-element buffer for kernel
                # Shared arrays (species-independent)
                "hyper": pre["hyper"],
                "kx_b": pre["kx_b"],
                "ky_b": pre["ky_b"],
                "valid_shift": pre["valid_shift"],
                "s_shift": pre["s_shift"],
                "kx_shift": pre["kx_shift"],
            }
            
            result_sp = self._linear_rhs_fused(
                df_sp, phi, sp_pre, params.dvp, params.disp_vp, params.drive_scale
            )
            results.append(result_sp)
        
        return jnp.stack(results)

    def linear_rhs(
        self,
        df: jnp.ndarray,
        phi: jnp.ndarray,
        geometry: Dict[str, jnp.ndarray],
        params,
        pre: Dict[str, jnp.ndarray],
    ) -> jnp.ndarray:
        """Fused linear RHS for single or multi-species.

        For 5D df (adiabatic): direct fused kernel call.
        For 6D df (kinetic): per-species loop to handle non-uniform species params
        (ions + electrons have different signz, tmp, etc.).
        """
        if df.ndim == 5:
            return self._linear_rhs_fused(
                df, phi, pre, params.dvp, params.disp_vp, params.drive_scale
            )

        # 6D kinetic: always use per-species loop (ions and electrons have different params)
        log.debug("CUDA linear_rhs: kinetic electrons, using per-species loop")
        return self._linear_rhs_kinetic_loop(df, phi, params, pre)

    def nonlinear_term_iii(
        self,
        df: jnp.ndarray,
        phi: jnp.ndarray,
        geometry: Dict[str, jnp.ndarray],
        *,
        efun_sign: float = 1.0,
        fft_prefactor: complex = 1.0 + 0.0j,
        exclude_zero_mode: bool = True,
        bessel: Optional[jnp.ndarray] = None,
    ) -> jnp.ndarray:
        """Nonlinear bracket via CUDA graph-captured cuFFT pipeline.

        Uses z2z 2-for-1 packing with phi at its natural (nmu*ns) batch
        size. Dispatches to mixed precision (FP32 FFTs) or full precision
        (FP64) kernel based on self.mixed_precision flag.
        
        Supports both 5D (adiabatic) and 6D (kinetic) df. For kinetic,
        flattens (nsp, nv, nmu) into a single batch dimension.
        
        Args:
            df: Distribution function, 5D (nv, nmu, ns, nkx, nky) or 
                6D (nsp, nv, nmu, ns, nkx, nky)
            phi: Electrostatic potential (ns, nkx, nky)
            geometry: Geometry dict
            efun_sign: Sign factor for ExB bracket
            fft_prefactor: Prefactor for FFT
            exclude_zero_mode: Zero out (kx=0, ky=0) mode
            bessel: Optional Bessel function array
        
        Returns:
            Nonlinear RHS term III
        """
        pre = self.pre
        mrad, mphi = pre["nl_mrad"], pre["nl_mphi"]
        
        # Handle both 5D (adiabatic) and 6D (kinetic) df
        if df.ndim == 5:
            nv, nmu, ns, nkx, nky = df.shape
        elif df.ndim == 6:
            nsp, nv, nmu, ns, nkx, nky = df.shape
        else:
            raise ValueError(f"nonlinear_term_iii: expected df with ndim 5 or 6, got {df.ndim}")

        jind = pre["nl_jind"]
        inverse_jind = jnp.full((mrad,), -1, dtype=jnp.int32)
        inverse_jind = inverse_jind.at[jind].set(jnp.arange(jind.shape[0], dtype=jnp.int32))

        kx_vec = pre["nl_kx2d"][:, 0]
        ky_vec = pre["nl_ky2d"][0, :]
        dum_s = pre["nl_dum_s"]

        if self.mixed_precision:
            kernel_name = "cufft_graph_bracket_true_fp32_ffi"
        else:
            kernel_name = "cufft_graph_bracket_fp64_ffi"
        _register_ffi()

        def _call_5d(df5, bessel5):
            batch = nv * nmu * ns
            df_flat = df5.reshape(-1, nkx, nky) * efun_sign
            p_phi = (bessel5 * phi.reshape(1, 1, ns, nkx, nky)).reshape(-1, nkx, nky)
            return ffi.ffi_call(
                kernel_name,
                jax.ShapeDtypeStruct((batch, nkx, nky), jnp.complex128),
            )(
                df_flat,
                p_phi,
                kx_vec,
                ky_vec,
                jnp.asarray(jind, dtype=jnp.int32),
                inverse_jind,
                dum_s,
                batch=np.int32(batch // ns),
                mrad=np.int32(mrad),
                mphi=np.int32(mphi),
                nkx=np.int32(nkx),
                nky=np.int32(nky),
                nspec=np.int32(ns),
                ixzero=np.int32(pre["ixzero"]),
                iyzero=np.int32(pre["iyzero"]),
            )

        if bessel is None:
            bessel = pre["bessel"]

        if df.ndim == 5:
            out_raw = _call_5d(df, bessel)
            nl = (fft_prefactor * pre["nl_fft_scale"] * out_raw).reshape(df.shape)
        else:  # 6D kinetic: loop over species, each gets its own 5D call
            results = []
            for i in range(nsp):
                bessel_sp = bessel[i] if bessel.ndim >= df.ndim else bessel
                out_sp = _call_5d(df[i], bessel_sp)
                results.append(out_sp.reshape(nv, nmu, ns, nkx, nky))
            nl = fft_prefactor * pre["nl_fft_scale"] * jnp.stack(results, axis=0)

        if exclude_zero_mode:
            return nl.at[..., pre["ixzero"], pre["iyzero"]].set(0.0)
        return nl
