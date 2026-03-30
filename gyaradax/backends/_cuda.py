import jax
import jax.numpy as jnp
import numpy as np
import ctypes
from pathlib import Path
from typing import Optional, Tuple, Dict
from jax import ffi

from gyaradax.backends.ops import SolverOps
from gyaradax.types import GKPre
from gyaradax.utils import pack_half_spectrum, unpack_half_spectrum
from gyaradax import stencils

# --- FFI --
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
        "apply_vpar_stencil_ffi":      _lib.apply_vpar_stencil_ffi,
        "apply_vpar_dual_stencil_ffi": _lib.apply_vpar_dual_stencil_ffi,
        "apply_parallel_ffi":          _lib.apply_parallel_ffi,
        "apply_parallel_dual_ffi":     _lib.apply_parallel_dual_ffi,
        "lto_fft_bracket_v2_ffi":      _lib.lto_fft_bracket_v2_ffi,
        "linear_rhs_vtiled_ffi":       _lib.linear_rhs_vtiled_ffi,
        "linear_rhs_fused_ffi":        _lib.linear_rhs_fused_ffi,
    }
    
    for name, symbol in targets.items():
        try:
            ffi.register_ffi_target(name, ffi.pycapsule(symbol), platform="CUDA")
        except Exception:
            pass # ignore re-registration
    
    _ffi_registered = True
    return True

def is_available():
    """ Check if CUDA is reachable and FFI kernels are compiled. """
    return LIB_PATH.exists() and jax.devices("cuda")

@jax.tree_util.register_pytree_node_class
class CUDAOps(SolverOps):
    """ CUDA backend for solver operations using high-performance FFI kernels. """

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
        pre, = children
        obj = cls(pre, None)
        obj.template_meta = aux_data
        return obj

    def _apply_vpar(self, field: jnp.ndarray, coeffs) -> jnp.ndarray:
        nv = field.shape[0]
        inner_size = field.size // nv
        
        # Stencil coefficients for D1 or D4 (len 5)
        return ffi.ffi_call(
            "apply_vpar_stencil_ffi",
            [jax.ShapeDtypeStruct(field.shape, field.dtype)]
        )(field, 
          c0=float(coeffs[0]), c1=float(coeffs[1]), c2=float(coeffs[2]), 
          c3=float(coeffs[3]), c4=float(coeffs[4]),
          nv=np.int32(nv), inner_size=np.int32(inner_size))[0]


    def _apply_vpar_dual(
        self, field: jnp.ndarray, coeffs_d1, coeffs_d4
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        nv = field.shape[0]
        inner_size = field.size // nv
        
        return ffi.ffi_call(
            "apply_vpar_dual_stencil_ffi",
            [jax.ShapeDtypeStruct(field.shape, field.dtype),
             jax.ShapeDtypeStruct(field.shape, field.dtype)]
        )(field,
          c0_d1=float(coeffs_d1[0]), c1_d1=float(coeffs_d1[1]), c2_d1=float(coeffs_d1[2]), 
          c3_d1=float(coeffs_d1[3]), c4_d1=float(coeffs_d1[4]),
          c0_d4=float(coeffs_d4[0]), c1_d4=float(coeffs_d4[1]), c2_d4=float(coeffs_d4[2]), 
          c3_d4=float(coeffs_d4[3]), c4_d4=float(coeffs_d4[4]),
          nv=np.int32(nv), inner_size=np.int32(inner_size))


    def _apply_parallel(self, field: jnp.ndarray, coeffs: jnp.ndarray) -> jnp.ndarray:
        nv, nmu, ns, nkx, nky = field.shape
        nv_nmu = np.int32(nv * nmu)
        
        def prepare_c(c):
            if c.ndim == 2: # (9, ns)
                c = c.reshape(9, 1, 1, ns, 1, 1)
            elif c.ndim == 4: # (9, nv, nmu, ns)
                c = c.reshape(9, nv, nmu, ns, 1, 1)
            elif c.ndim == 6: # Already 6D
                pass
            else:
                while c.ndim < 6:
                    c = c[..., None]
            # No nky-slicing: keep (9, nv_nmu, ns, nkx, nky)
            return jnp.broadcast_to(c, (9, nv, nmu, ns, nkx, nky)).reshape(9, nv_nmu, ns, nkx, nky).copy()

        c_1d = prepare_c(coeffs).reshape(-1)
        field_b = jnp.broadcast_to(field, (nv, nmu, ns, nkx, nky)).copy()

        # Packed maps: keep (9, ns, nkx, nky, 2)
        valid_jax = jnp.array(self.pre["valid_shift"])
        s_map_jax = jnp.where(valid_jax, self.pre["s_shift"], -1).astype(jnp.int32)
        kx_map_jax = jnp.array(self.pre["kx_shift"]).astype(jnp.int32)
        packed_maps = jnp.stack([s_map_jax, kx_map_jax], axis=-1).copy()

        return ffi.ffi_call(
            "apply_parallel_ffi",
            [jax.ShapeDtypeStruct(field_b.shape, field_b.dtype)]
        )(field_b, c_1d, packed_maps,
          nv_nmu=np.int32(nv_nmu), nkx=np.int32(nkx), ns=np.int32(ns), 
          nky=np.int32(nky), nmu=np.int32(nmu))[0]

    def _apply_parallel_dual(
        self, field1: jnp.ndarray, field2: jnp.ndarray, coeffs1: jnp.ndarray, coeffs2: jnp.ndarray
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        nv1, nmu1, ns1, nkx1, nky1 = field1.shape
        nv2, nmu2, ns2, nkx2, nky2 = field2.shape
        assert ns1 == ns2 and nkx1 == nkx2 and nky1 == nky2, \
            f"Spatial mismatch: field1={(ns1, nkx1, nky1)}, field2={(ns2, nkx2, nky2)}"
        nv, nmu = max(nv1, nv2), max(nmu1, nmu2)
        ns, nkx, nky = ns1, nkx1, nky1
        
        target_shape = (nv, nmu, ns, nkx, nky)
        f1_b = jnp.broadcast_to(field1, target_shape).copy()
        f2_b = jnp.broadcast_to(field2, target_shape).copy()
        
        nv_nmu = np.int32(nv * nmu)

        def prepare_c(c):
            if c.ndim == 2: # (9, ns)
                c = c.reshape(9, 1, 1, ns, 1, 1)
            elif c.ndim == 4: # (9, nv, nmu, ns)
                c = c.reshape(9, nv, nmu, ns, 1, 1)
            elif c.ndim == 6: # Already 6D
                pass
            else:
                while c.ndim < 6:
                    c = c[..., None]
            return jnp.broadcast_to(c, (9, nv, nmu, ns, nkx, nky)).reshape(9, nv_nmu, ns, nkx, nky).copy()

        c1_1d = prepare_c(coeffs1).reshape(-1)
        c2_1d = prepare_c(coeffs2).reshape(-1)

        valid_jax = jnp.array(self.pre["valid_shift"])
        s_map_jax = jnp.where(valid_jax, self.pre["s_shift"], -1).astype(jnp.int32)
        kx_map_jax = jnp.array(self.pre["kx_shift"]).astype(jnp.int32)
        packed_maps = jnp.stack([s_map_jax, kx_map_jax], axis=-1).copy()

        return ffi.ffi_call(
            "apply_parallel_dual_ffi",
            [jax.ShapeDtypeStruct(target_shape, field1.dtype),
             jax.ShapeDtypeStruct(target_shape, field2.dtype)]
        )(f1_b, f2_b, c1_1d, c2_1d, packed_maps,
          nv_nmu=np.int32(nv_nmu), nkx=np.int32(nkx), ns=np.int32(ns), 
          nky=np.int32(nky), nmu=np.int32(nmu))

    def _linear_rhs_fused(
        self,
        df: jnp.ndarray,
        phi: jnp.ndarray, # (ns, nkx, nky)
        pre: Dict[str, jnp.ndarray],
        params_dvp: float,
        params_disp_vp: float,
        params_drive_scale: float,
        target_name: str = "linear_rhs_fused_ffi"
    ) -> jnp.ndarray:
        # Internal fused call for 5D data (single species or bulk species)
        nv, nmu, ns, nkx, nky = df.shape
        nv_nmu = np.int32(nv * nmu)
        V_TILE = 8 # Sweet spot on A100

        # Fallback for non-tiled sizes
        if target_name == "linear_rhs_vtiled_ffi" and nv_nmu % V_TILE != 0:
            target_name = "linear_rhs_fused_ffi"
            
        if target_name == "linear_rhs_vtiled_ffi":
            assert nv_nmu % V_TILE == 0, f"nv_nmu={nv_nmu} must be divisible by V_TILE={V_TILE}"

        # Fields and 5D coefficients
        f_b      = df.copy()
        phi_b    = phi.copy()

        # Minimal-shape preparations
        # bessel: (nmu, ns, nkx, nky)
        bessel   = jnp.broadcast_to(pre["bessel"].squeeze(), (nmu, ns, nkx, nky)).copy()
        
        # s_total_upar: (9, nv, 1, ns, nkx, nky) (Independent of mu)
        c_upar_in = pre["s_total_upar"]
        if c_upar_in.ndim == 6 and c_upar_in.shape[2] > 1:
            c_upar_in = c_upar_in[:, :, 0:1, ...]
        elif c_upar_in.ndim == 7 and c_upar_in.shape[3] > 1: # (9, nsp, nv, nmu, ns, nkx, nky)
            c_upar_in = c_upar_in[:, :, :, 0:1, ...]
            
        c_upar   = jnp.broadcast_to(c_upar_in, (9, nv, 1, ns, nkx, nky)).copy()
        
        # s_total_t7: (9, nv, nmu, ns, nkx, nky)
        c_t7     = jnp.broadcast_to(pre["s_total_t7"], (9, nv, nmu, ns, nkx, nky)).copy()
        
        # utrap, abs_vp: (nmu, ns)
        utrap    = jnp.broadcast_to(pre["utrap"].squeeze(), (nmu, ns)).copy()
        abs_vp   = jnp.broadcast_to(pre["abs_dum2_vp"].squeeze(), (nmu, ns)).copy()
        
        # drift_x, drift_y, fmaxwl: (nv, nmu, ns)
        drift_x  = jnp.broadcast_to(pre["drift_x"].squeeze(), (nv, nmu, ns)).copy()
        drift_y  = jnp.broadcast_to(pre["drift_y"].squeeze(), (nv, nmu, ns)).copy()
        fmaxwl   = jnp.broadcast_to(pre["fmaxwl"].squeeze(), (nv, nmu, ns)).copy()
        
        # dmaxwel: (nv, nmu, ns, nky)
        dmaxwel  = jnp.broadcast_to(pre["dmaxwel_fm_ek"].squeeze(), (nv, nmu, ns, nky)).copy()

        # 3D and 1D arrays
        hyper    = jnp.broadcast_to(pre["hyper"].squeeze(), (ns, nkx, nky)).copy()
        kx_vals  = pre["kx_b"].reshape(-1)[:nkx].copy()
        ky_vals  = pre["ky_b"].reshape(-1)[:nky].copy()

        # Packed maps
        valid_jax = jnp.array(pre["valid_shift"])
        s_map_jax = jnp.where(valid_jax, pre["s_shift"], -1).astype(jnp.int32)
        kx_map_jax = jnp.array(pre["kx_shift"]).astype(jnp.int32)
        packed_maps = jnp.stack([s_map_jax, kx_map_jax], axis=-1).reshape(9, ns, nkx, nky, 2).copy()

        d1 = stencils.VPAR_D1
        d4 = stencils.VPAR_D4

        attrs = dict(
          nv=np.int32(nv), nmu=np.int32(nmu), ns=np.int32(ns), 
          nkx=np.int32(nkx), nky=np.int32(nky), nv_nmu=nv_nmu,
          c_d1_0=float(d1[0]), c_d1_1=float(d1[1]), c_d1_2=float(d1[2]),
          c_d1_3=float(d1[3]), c_d1_4=float(d1[4]),
          c_d4_0=float(d4[0]), c_d4_1=float(d4[1]), c_d4_2=float(d4[2]),
          c_d4_3=float(d4[3]), c_d4_4=float(d4[4]),
          dvp=float(params_dvp), disp_vp=float(params_disp_vp), 
          drive_scale=float(params_drive_scale),
          signz0=float(pre["signz0"]), tmp0=float(pre["tmp0"])
        )
        if target_name == "linear_rhs_vtiled_ffi":
            attrs["v_tile"] = np.int32(V_TILE)

        _register_ffi()
        return ffi.ffi_call(
            target_name,
            [jax.ShapeDtypeStruct(df.shape, df.dtype)]
        )(f_b, phi_b, bessel, c_upar, c_t7, packed_maps,
          utrap, abs_vp, drift_x, drift_y, dmaxwel, fmaxwl,
          hyper, kx_vals, ky_vals, **attrs)[0]

    def linear_rhs(
        self,
        df: jnp.ndarray,
        phi: jnp.ndarray, # (ns, nkx, nky)
        geometry: Dict[str, jnp.ndarray],
        params,
        pre: Dict[str, jnp.ndarray],
    ) -> jnp.ndarray:
        """ 6D Unified Linear RHS for multiple species. """
        if df.ndim == 6: # (nsp, nv, nmu, ns, nkx, nky)
            nsp, nv, nmu, ns, nkx, nky = df.shape
            # Reshape to 5D where nv' = nsp * nv * nmu, nmu' = 1
            # Wait, easier to flatten nsp, nv, nmu into the first dimension
            df_5d = df.reshape(nsp * nv * nmu, 1, ns, nkx, nky)
            
            # Prepare a unified 'pre' by broadcasting kinetic arrays
            uni_pre = pre.copy()
            # Most arrays in 'pre' for kinetic (nsp, nv, nmu, ns, nkx, nky)
            # We need to reshape them to (nsp*nv*nmu, 1, ns, nkx, nky)
            def r6to5(arr):
                if arr.ndim == 6:
                    return arr.reshape(nsp * nv * nmu, 1, ns, nkx, nky)
                elif arr.ndim == 5: # (nv, nmu, ns, nkx, nky)
                    return jnp.broadcast_to(arr[None, ...], (nsp, nv, nmu, ns, nkx, nky)).reshape(nsp * nv * nmu, 1, ns, nkx, nky)
                return arr

            uni_pre["bessel"] = r6to5(pre["bessel"])
            uni_pre["s_total_upar"] = r6to5(pre["s_total_upar"])
            uni_pre["s_total_t7"] = r6to5(pre["s_total_t7"])
            # Some are (nsp, nv, nmu, ns) -> reshape to (nsp*nv*nmu, 1, ns)
            def r4to3(arr):
                return arr.reshape(nsp * nv * nmu, 1, ns)
            
            uni_pre["fmaxwl"] = r4to3(pre["fmaxwl"])
            uni_pre["dmaxwel_fm_ek"] = pre["dmaxwel_fm_ek"].reshape(nsp * nv * nmu, 1, ns, nky)
            uni_pre["drift_x"] = r4to3(pre["drift_x"])
            uni_pre["drift_y"] = r4to3(pre["drift_y"])
            
            # utrap, abs_vp are (nsp, nmu, ns) -> broadcast to (nsp, nv, nmu, ns) then reshape
            utrap_4d = jnp.broadcast_to(pre["utrap"][:, None, :, :], (nsp, nv, nmu, ns))
            abs_vp_4d = jnp.broadcast_to(pre["abs_dum2_vp"][:, None, :, :], (nsp, nv, nmu, ns))
            uni_pre["utrap"] = utrap_4d.reshape(nsp * nv * nmu, 1, ns)
            uni_pre["abs_dum2_vp"] = abs_vp_4d.reshape(nsp * nv * nmu, 1, ns)
            
            # Scalars signz0, tmp0 are (nsp,) -> broadcast to (nsp, nv, nmu)? No, need a single value for FFI
            # Actually our kernel handles them per-species? No, they are passed as scalars.
            # WAIT! If they differ per species, we can't use the single value FFI.
            if jnp.unique(pre["signz0"]).size > 1 or jnp.unique(pre["tmp0"]).size > 1:
                # Fallback to vmap if species parameters differ (rare in GKW, but possible)
                # log.info("CUDAOps.linear_rhs falling back to vmap due to per-species signz0/tmp0")
                return None # Signal solver.py to use vmap

            uni_pre["signz0"] = pre["signz0"][0]
            uni_pre["tmp0"] = pre["tmp0"][0]

            out_5d = self._linear_rhs_fused(
                df_5d, phi, uni_pre, params.dvp, params.disp_vp, params.drive_scale
            )
            return out_5d.reshape(nsp, nv, nmu, ns, nkx, nky)
        else:
            # 5D case (adiabatic)
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
    ) -> jnp.ndarray:
        """ Nonlinear FFT Poisson bracket using FFI fusion. """
        pre = self.pre
        mrad, mphi = pre["nl_mrad"], pre["nl_mphi"]
        nv, nmu, ns, nkx, nky = df.shape
        
        # Build inverse_jind for the FFI callback (JIT-compatible)
        jind = pre["nl_jind"]
        inverse_jind = jnp.full((mrad,), -1, dtype=jnp.int32)
        inverse_jind = inverse_jind.at[jind].set(jnp.arange(jind.shape[0], dtype=jnp.int32))
        
        # Prepare inputs
        kx_vec = pre["nl_kx2d"][:, 0]
        ky_vec = pre["nl_ky2d"][0, :]
        dum_s = pre["nl_dum_s"]
        
        # Reshape to 3D batches for FFI
        batch_total = df.shape[0] * df.shape[1] * df.shape[2]
        df_lto = df.reshape(-1, nkx, nky)
        
        # Apply bessel once in JAX space before FFI (mirroring benchmark)
        # Apply efun_sign to df inside the bracket (as JAX logic)
        # Note: applying to df only is equivalent to scaling the whole bracket since it is linear in df.
        df_lto = df_lto * efun_sign
        
        # p_lto: must match df_lto (batch, nkx, nky) 
        # pre["bessel"] is (1, nmu, ns, nkx, nky) or (nmu, ns, nkx, nky)
        p_b = phi.reshape(1, 1, ns, nkx, nky)
        p_gyro = (pre["bessel"] * p_b)
        p_lto = jnp.broadcast_to(p_gyro, (nv, nmu, ns, nkx, nky)).reshape(-1, nkx, nky)
        
        out_raw = ffi.ffi_call(
            "lto_fft_bracket_v2_ffi",
            jax.ShapeDtypeStruct((batch_total, mrad, (mphi // 2 + 1)), jnp.complex128)
        )(df_lto, p_lto, kx_vec, ky_vec, inverse_jind, dum_s,
          batch=np.int32(batch_total), mrad=np.int32(mrad), mphi=np.int32(mphi), 
          nkx=np.int32(nkx), nky=np.int32(nky))
        
        # Apply physics normalization and unpacking
        # Note: efun_sign is applied inside the wrapper in benchmark, 
        # but here we follow JAXOps logic if possible or check benchmark.
        # Benchmark v2 uses: nl_half = (fft_prefactor * fft_scale) * out_normalized
        # where out_normalized = out_raw / (N * N)
        N = mrad * mphi
        fft_scale = pre["nl_fft_scale"]
        
        # Combine scales
        total_scale = (fft_prefactor * fft_scale) / (N * N)
        nl_half = total_scale * out_raw
        
        # Unpack back to 5D
        nl = unpack_half_spectrum(nl_half, jind, nky)
        nl_5d = nl.reshape(df.shape)
        
        if exclude_zero_mode:
            ixzero, iyzero = pre["ixzero"], pre["iyzero"]
            return nl_5d.at[:, :, :, ixzero, iyzero].set(0.0)
        return nl_5d

