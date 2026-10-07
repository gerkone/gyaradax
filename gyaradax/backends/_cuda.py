"""CUDA backend for solver operations using custom FFI kernels.

Provides fused stencil application and nonlinear bracket kernels
from cuda_kernels/. Falls back gracefully if the shared library
is not compiled.
"""

import ctypes
import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np
from jax import ffi
from jax.sharding import PartitionSpec

import gyaradax.sharding as sharding
import gyaradax.stencils as stencils
from gyaradax.backends.ops import SolverOps
from gyaradax.collisions import collision_term, collisions_on
from gyaradax.fields import g_to_f
from gyaradax.state import GKPre

log = logging.getLogger(__name__)

_CUDA_KERNELS_DIR = Path(__file__).parent / "cuda_kernels"
LIB_PATH = _CUDA_KERNELS_DIR / "libgyaradax_cuda.so"
_ffi_registered = False
# set when the library carries the cuFFTDx (v6) bracket; GYARADAX_BRACKET=v5 forces the v5 pipeline
_has_bracket_v6 = False


def _concrete_int_or_none(value: Any) -> int | None:
    """Return a Python int for concrete scalar metadata, or None for tracers."""
    try:
        arr = np.asarray(value)
    except (jax.errors.ConcretizationTypeError, jax.errors.TracerArrayConversionError, TypeError):
        return None
    if arr.shape != ():
        raise ValueError(f"expected scalar metadata, got shape {arr.shape}")
    return int(arr.item())


def _concrete_float(value: Any, name: str) -> float:
    """Return a Python float for scalar metadata passed as an FFI attribute."""
    try:
        return float(np.asarray(value))
    except (
        jax.errors.ConcretizationTypeError,
        jax.errors.TracerArrayConversionError,
        TypeError,
    ) as exc:
        raise ValueError(f"CUDA backend needs a concrete {name}") from exc


def _cuda_zero_mode_index(pre: GKPre, key: str, fallback: int, size: int) -> np.int32:
    """Get concrete zero-mode metadata for CUDA FFI static attributes.

    CUDA FFI attributes must be Python-concrete at trace time. Prefer the
    precomputed zero-mode index when available; when ``pre`` was constructed
    inside a JIT trace, scalar metadata can be traced, so fall back to the
    current mode-box convention while still validating all concrete metadata.
    """
    concrete = _concrete_int_or_none(pre[key])
    if concrete is None:
        if key == "ixzero" and size % 2 != 1:
            raise ValueError("CUDA nonlinear mode-box fallback requires odd nkx")
        idx = fallback
    else:
        idx = concrete
    if not 0 <= idx < size:
        raise ValueError(f"{key}={idx} out of bounds for size {size}")
    return np.int32(idx)


def _register_ffi():
    global _ffi_registered, _has_bracket_v6
    if _ffi_registered:
        return True

    if not LIB_PATH.exists():
        return False

    try:
        _lib = ctypes.cdll.LoadLibrary(str(LIB_PATH))
    except (OSError, AttributeError):
        return False

    targets = {
        # Nonlinear bracket kernels (production variants)
        "cufft_graph_bracket_true_fp32_ffi": _lib.cufft_graph_bracket_true_fp32_ffi,
        "cufft_graph_bracket_fp64_ffi": _lib.cufft_graph_bracket_fp64_ffi,
        "linear_rhs_fused_ffi": _lib.linear_rhs_fused_ffi,
        "field_moments_ffi": _lib.field_moments_ffi,
    }

    for name in ("cufft_bracket_v6_fp32_ffi", "cufft_bracket_v6_fp64_ffi"):
        if hasattr(_lib, name):
            targets[name] = getattr(_lib, name)
    _has_bracket_v6 = "cufft_bracket_v6_fp32_ffi" in targets

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


@jax.tree_util.register_pytree_node_class
class CUDAOps(SolverOps):
    """CUDA backend using custom FFI kernels for stencils and FFT bracket.

    Note: CUDA backend is Z2Z (complex-to-complex) only. The use_z2z flag
    is ignored for CUDA operations and will emit a warning if set to True.
    """

    def __init__(self, pre: GKPre, use_z2z: bool = False, mixed_precision: bool = True, mesh=None):
        _register_ffi()
        if use_z2z:
            log.warning("CUDA backend: use_z2z=True ignored (CUDA is Z2Z-only)")

        super().__init__(pre, use_z2z, mixed_precision, mesh)

    def _pack_shift_maps(self):
        """Pack precomputed shift maps into (9, ns, nkx, nky, 2) int32 array."""
        valid_jax = jnp.array(self.pre["valid_shift"])
        s_map_jax = jnp.where(valid_jax, self.pre["s_shift"], -1).astype(jnp.int32)
        kx_map_jax = jnp.array(self.pre["kx_shift"]).astype(jnp.int32)
        return jnp.stack([s_map_jax, kx_map_jax], axis=-1).copy()

    def _linear_rhs_fused(
        self,
        df: jnp.ndarray,
        phi: jnp.ndarray,
        pre: Dict[str, jnp.ndarray],
        params,
        apar: Optional[jnp.ndarray] = None,
        bpar: Optional[jnp.ndarray] = None,
        df_is_g: bool = False,
        dproj: Optional[Tuple[jnp.ndarray, jnp.ndarray]] = None,
    ) -> jnp.ndarray:
        """Fused linear RHS kernel, all species in one launch.

        df is 5D (adiabatic) or 6D (kinetic). With apar, df_is_g selects whether
        df holds the mixed variable g (the kernel forms f = g + g2f*apar) or f.
        dproj = (m0, m1) enables the conservative parallel dissipation projection.
        """
        kinetic = df.ndim == 6
        if kinetic:
            nsp, nv, nmu, ns, nkx, nky = df.shape
        else:
            nsp = 1
            nv, nmu, ns, nkx, nky = df.shape

        def sp6(x):
            x = jnp.asarray(x)
            return x if kinetic else x[None]

        def table(key, shape, out_shape):
            return jnp.broadcast_to(sp6(pre[key]), shape).reshape(out_shape)

        if "s_upar_tab" not in pre:
            raise ValueError("CUDA linear_rhs needs the stencil class tables of linear_precompute")
        n_class = int(pre["s_upar_tab"].shape[-2])

        def class_table(key):
            # ([nsp,] nv, 1, ns, n_class, 9) -> (nsp, nv, ns, n_class, 9)
            c = sp6(pre[key])
            if c.shape[2] != 1:
                raise ValueError(f"CUDA linear_rhs expects mu-independent {key}, got {c.shape}")
            return jnp.broadcast_to(c, (nsp, nv, 1, ns, n_class, 9)).reshape(
                nsp, nv, ns, n_class, 9
            )

        v_mu_s = (nsp, nv, nmu, ns, 1, 1)
        mu_s = (nsp, 1, nmu, ns, 1, 1)
        bessel = table("bessel", (nsp, 1, nmu, ns, nkx, nky), (nsp, nmu, ns, nkx, nky))
        s_upar_tab = class_table("s_upar_tab")
        s_t7_tab = table("s_t7_tab", (nsp, nv, nmu, ns, n_class, 9), (nsp, nv, nmu, ns, n_class, 9))
        par_class = jnp.broadcast_to(pre["par_stencil_class"], (ns, nkx, nky)).astype(jnp.int32)
        utrap = table("utrap", mu_s, (nsp, nmu, ns))
        abs_vp = table("abs_dum2_vp", mu_s, (nsp, nmu, ns))
        drift_x = table("drift_x", v_mu_s, (nsp, nv, nmu, ns))
        drift_y = table("drift_y", v_mu_s, (nsp, nv, nmu, ns))
        fmaxwl = table("fmaxwl", v_mu_s, (nsp, nv, nmu, ns))
        dmaxwel = table("dmaxwel_fm_ek", (nsp, nv, nmu, ns, 1, nky), (nsp, nv, nmu, ns, nky))
        # keeps the ky axis when nky == 1
        hyper = jnp.broadcast_to(pre["hyper"], (1, 1, 1, nkx, nky)).reshape(nkx, nky)
        kx_vals = pre["kx_b"].reshape(-1)[:nkx]
        ky_vals = pre["ky_b"].reshape(-1)[:nky]
        packed_maps = self._pack_shift_maps().reshape(9, ns, nkx, nky, 2)

        # signz0/tmp0 are per-species buffer args (F64) in the kernel, not scalar attrs
        def per_species(key):
            return jnp.broadcast_to(jnp.asarray(pre[key], dtype=jnp.float64).reshape(-1), (nsp,))

        signz0 = per_species("signz0")
        tmp0 = per_species("tmp0")

        dummy_c = jnp.zeros((1,), dtype=jnp.complex128)
        dummy_r = jnp.zeros((1,), dtype=jnp.float64)

        has_apar = apar is not None and "apar_chi_factor" in pre
        if has_apar:
            if "apar_chi_vfac" not in pre:
                raise ValueError(
                    "CUDA EM linear_rhs needs pre['apar_chi_vfac'] (linear_precompute)"
                )
            vel = (nsp, nv, 1, 1, 1, 1)
            chi_vfac = table("apar_chi_vfac", vel, (nsp, nv))
            # g2f = 0 makes the in-kernel g -> f conversion the identity
            if df_is_g:
                g2f_vfac = table("g2f_vfac", vel, (nsp, nv))
            else:
                g2f_vfac = jnp.zeros((nsp, nv), dtype=jnp.float64)
            apar_buf = apar.astype(jnp.complex128)
        else:
            if df_is_g:
                raise ValueError("df_is_g requires apar")
            chi_vfac = g2f_vfac = dummy_r
            apar_buf = dummy_c

        has_bpar = bpar is not None and "bpar_chi_factor" in pre
        if has_bpar:
            bpar_chi = table(
                "bpar_chi_factor", (nsp, 1, nmu, ns, nkx, nky), (nsp, nmu, ns, nkx, nky)
            )
            bpar_buf = bpar.astype(jnp.complex128)
        else:
            bpar_chi, bpar_buf = dummy_r, dummy_c

        has_dpc = dproj is not None
        if has_dpc:
            full = (nsp, nv, nmu, ns, nkx, nky)
            s_disp_tab = class_table("s_disp_par_tab")
            dproj_m0, dproj_m1 = dproj
            dproj_e0 = table("dproj_e0", full, full)
            dproj_e1 = table("dproj_e1", full, full)
        else:
            s_disp_tab = dproj_e0 = dproj_e1 = dummy_r
            dproj_m0 = dproj_m1 = dummy_c

        d1 = stencils.VPAR_D1
        d4 = stencils.VPAR_D4
        attrs = dict(
            ns=np.int32(ns),
            nkx=np.int32(nkx),
            nky=np.int32(nky),
            n_class=np.int32(n_class),
            has_apar=np.int32(has_apar),
            has_bpar=np.int32(has_bpar),
            has_dpc=np.int32(has_dpc),
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
            dvp=_concrete_float(pre["dvp"], "pre['dvp']"),
            disp_vp=float(params.disp_vp),
            drive_scale=float(params.drive_scale),
        )
        vel = ("sp", "vp", "mu") if kinetic else ("vp", "mu")
        sp_v, sp_mu, sp_v_mu = ("sp", "vp"), ("sp", "mu"), ("sp", "vp", "mu")
        # kernel arguments in FFI order, with the mesh axes of their leading dims
        inputs = {
            "df": (df.astype(jnp.complex128), vel),
            "phi": (phi.astype(jnp.complex128), ()),
            "bessel": (bessel, sp_mu),
            "s_upar_tab": (s_upar_tab, sp_v),
            "s_t7_tab": (s_t7_tab, sp_v_mu),
            "par_class": (par_class, ()),
            "packed_maps": (packed_maps, ()),
            "utrap": (utrap, sp_mu),
            "abs_vp": (abs_vp, sp_mu),
            "drift_x": (drift_x, sp_v_mu),
            "drift_y": (drift_y, sp_v_mu),
            "dmaxwel": (dmaxwel, sp_v_mu),
            "fmaxwl": (fmaxwl, sp_v_mu),
            "hyper": (hyper, ()),
            "kx_vals": (kx_vals, ()),
            "ky_vals": (ky_vals, ()),
            "signz0": (signz0, ("sp",)),
            "tmp0": (tmp0, ("sp",)),
            "apar": (apar_buf, ()),
            "chi_vfac": (chi_vfac, sp_v),
            "g2f_vfac": (g2f_vfac, sp_v),
            "bpar": (bpar_buf, ()),
            "bpar_chi": (bpar_chi, sp_mu),
            "s_disp_tab": (s_disp_tab, sp_v),
            "dproj_m0": (dproj_m0, ("sp",)),
            "dproj_m1": (dproj_m1, ("sp",)),
            "dproj_e0": (dproj_e0, sp_v_mu),
            "dproj_e1": (dproj_e1, sp_v_mu),
        }
        names = tuple(inputs)
        args = tuple(x for x, _ in inputs.values())
        axes = tuple(ax for _, ax in inputs.values())
        n_vp = self.mesh.shape["vp"] if self.mesh is not None else 1
        v_axis = 1 if kinetic else 0
        _register_ffi()

        def kernel(*xs):
            halo = (dummy_c, dummy_r, dummy_r)
            if n_vp > 1:
                # vpar neighbours beyond the shard edges, for the in-kernel g -> f and vpar stencil
                halo = (sharding.halo_planes(xs[0], v_axis, "vp", n_vp), dummy_r, dummy_r)
                if has_apar:
                    fm, g2f = xs[names.index("fmaxwl")], xs[names.index("g2f_vfac")]
                    halo = (
                        halo[0],
                        sharding.halo_planes(fm, 1, "vp", n_vp),
                        sharding.halo_planes(g2f, 1, "vp", n_vp),
                    )
            d = xs[0]
            nsp_l, nv_l, nmu_l = d.shape[:3] if kinetic else (1,) + d.shape[:2]
            return ffi.ffi_call(
                "linear_rhs_fused_ffi",
                [jax.ShapeDtypeStruct(d.shape, jnp.complex128)],
                vmap_method="sequential",
            )(
                *xs,
                *halo,
                nsp=np.int32(nsp_l),
                nv=np.int32(nv_l),
                nmu=np.int32(nmu_l),
                has_halo=np.int32(n_vp > 1),
                **attrs,
            )[0]

        if self.mesh is None:
            return kernel(*args)
        return sharding.velocity_map(kernel, self.mesh, args, axes, PartitionSpec(*vel))

    def _field_moments(self, g, weights, g2f=None, apar=None):
        """Moments sum_{sp,v,mu} w * f per spatial point for each weight, f = g + g2f*apar if given."""
        ns, nkx, nky = g.shape[3:]
        n_w = len(weights)
        use_g2f = g2f is not None
        w = [jnp.broadcast_to(x, g.shape).astype(jnp.float64) for x in weights]
        args = (
            g.astype(jnp.complex128),
            w[0],
            w[1] if n_w > 1 else jnp.zeros((1,), dtype=jnp.float64),
            jnp.broadcast_to(g2f, g.shape).astype(jnp.float64)
            if use_g2f
            else jnp.zeros((1,), dtype=jnp.float64),
            apar.astype(jnp.complex128) if use_g2f else jnp.zeros((1,), dtype=jnp.complex128),
        )

        def moments(g_l, w0, w1, g2f_l, apar_l):
            n_rows, n_spatial = int(np.prod(g_l.shape[:3])), ns * nkx * nky
            n_chunks = max(1, min(n_rows, 2048 // -(-n_spatial // 128)))
            partial = ffi.ffi_call(
                "field_moments_ffi",
                jax.ShapeDtypeStruct((n_chunks, n_w, n_spatial), jnp.complex128),
                vmap_method="sequential",
            )(
                g_l,
                w0,
                w1,
                g2f_l,
                apar_l,
                n_rows=np.int32(n_rows),
                n_spatial=np.int32(n_spatial),
                n_chunks=np.int32(n_chunks),
                n_w=np.int32(n_w),
                use_g2f=np.int32(use_g2f),
            )
            m = jnp.sum(partial, axis=0)
            if self.mesh is not None:
                m = jax.lax.psum(m, axis_name=tuple(self.mesh.axis_names))
            return m

        if self.mesh is None:
            m = moments(*args)
        else:
            vel = ("sp", "vp", "mu")
            m = sharding.velocity_map(
                moments, self.mesh, args, (vel, vel, vel, vel, ()), PartitionSpec()
            )
        m = m.reshape(n_w, ns, nkx, nky)
        return [m[k] for k in range(n_w)]

    def compute_fields(self, dg, geometry, params, pre):
        """Field solve (phi, apar, bpar) from the mixed variable g.

        Kinetic electromagnetic runs take the velocity moments with the fused
        field_moments kernel; all other configurations use the JAX field solve.
        """
        if dg.ndim != 6 or params.adiabatic_electrons or not (params.nlapar or params.nlbpar):
            return super().compute_fields(dg, geometry, params, pre)
        _register_ffi()
        apar = None
        if params.nlapar:
            (apar_num,) = self._field_moments(dg, [pre["apar_weight"]])
            apar = apar_num / pre["apar_diag"]
        has_bpar = params.nlbpar and "bpar_weight" in pre
        weights = [pre["phi_weight"]] + ([pre["bpar_weight"]] if has_bpar else [])
        g2f = pre["g2f_factor"] if apar is not None else None
        moments = self._field_moments(dg, weights, g2f=g2f, apar=apar)
        phi = -moments[0] / pre["phi_diag"]
        bpar = -moments[1] / pre["phi_diag"] if has_bpar else None
        return phi, apar, bpar

    @staticmethod
    def _dproj_moments(df: jnp.ndarray, pre) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Velocity moments (m0, m1) of f for the conservative dissipation, (nsp, ns, nkx, nky)."""
        axes = (1, 2) if df.ndim == 6 else (0, 1)
        m0 = jnp.sum(pre["dproj_w0"] * df, axis=axes)
        m1 = jnp.sum(pre["dproj_w1"] * df, axis=axes)
        if df.ndim == 5:
            m0, m1 = m0[None], m1[None]
        return m0, m1

    def _linear_rhs(self, df, phi, params, pre, apar, bpar, df_is_g):
        if df.ndim not in (5, 6):
            raise ValueError(f"linear_rhs: expected df with ndim 5 or 6, got {df.ndim}")

        f_phys = None

        def physical_f():
            return g_to_f(df, apar, params, pre) if df_is_g else df

        dproj = None
        if "s_disp_par_tab" in pre:
            f_phys = physical_f()
            dproj = self._dproj_moments(f_phys, pre)

        rhs = self._linear_rhs_fused(
            df, phi, pre, params, apar=apar, bpar=bpar, df_is_g=df_is_g, dproj=dproj
        )
        if collisions_on(params, pre):
            f_phys = physical_f() if f_phys is None else f_phys
            rhs = rhs + collision_term(f_phys, params, pre, self.mesh)
        return rhs

    def linear_rhs(
        self,
        df: jnp.ndarray,
        phi: jnp.ndarray,
        geometry: Dict[str, jnp.ndarray],
        params,
        pre: Dict[str, jnp.ndarray],
        apar: Optional[jnp.ndarray] = None,
        bpar: Optional[jnp.ndarray] = None,
    ) -> jnp.ndarray:
        return self._linear_rhs(df, phi, params, pre, apar, bpar, df_is_g=False)

    def linear_rhs_from_g(
        self,
        dg: jnp.ndarray,
        phi: jnp.ndarray,
        geometry: Dict[str, jnp.ndarray],
        params,
        pre: Dict[str, jnp.ndarray],
        apar: Optional[jnp.ndarray] = None,
        bpar: Optional[jnp.ndarray] = None,
    ) -> jnp.ndarray:
        df_is_g = apar is not None and bool(params.nlapar) and "apar_chi_factor" in pre
        return self._linear_rhs(dg, phi, params, pre, apar, bpar, df_is_g=df_is_g)

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
        chi_correction: Optional[jnp.ndarray] = None,
        apar: Optional[jnp.ndarray] = None,
        bpar: Optional[jnp.ndarray] = None,
    ) -> jnp.ndarray:
        """Nonlinear bracket via the cuFFT LTO-callback pipeline.

        Uses z2z 2-for-1 packing with the gyro-averaged potential at its natural
        (nsp*nmu*ns) batch size; all species run in one cuFFT batch. Dispatches to
        mixed precision (FP32 FFTs) or full precision (FP64) based on
        self.mixed_precision.

        Electromagnetic chi = J0*phi + bpar_chi*bpar + vfac[sp, v]*J0*apar is
        separable in vpar: the potentials A = J0*phi + bpar_chi*bpar and
        B = J0*apar are transformed once per (sp, mu, s) and combined in real
        space. An explicit chi_correction falls back to one potential per df plane.

        Args:
            df: Distribution function, 5D (nv, nmu, ns, nkx, nky) or
                6D (nsp, nv, nmu, ns, nkx, nky)
            phi: Electrostatic potential (ns, nkx, nky)
            geometry: Geometry dict
            efun_sign: Sign factor for ExB bracket
            fft_prefactor: Prefactor for FFT
            exclude_zero_mode: Zero out (kx=0, ky=0) mode
            bessel: Optional Bessel function array
            chi_correction: Optional explicit EM correction (same shape as df)
            apar, bpar: Optional EM fields (ns, nkx, nky)

        Returns:
            Nonlinear RHS term III
        """
        pre = self.pre
        mrad, mphi = pre["nl_mrad"], pre["nl_mphi"]

        if df.ndim == 5:
            nsp = 1
            nv, nmu, ns, nkx, nky = df.shape
        elif df.ndim == 6:
            nsp, nv, nmu, ns, nkx, nky = df.shape
        else:
            raise ValueError(f"nonlinear_term_iii: expected df with ndim 5 or 6, got {df.ndim}")
        if chi_correction is not None and (apar is not None or bpar is not None):
            raise ValueError("pass either chi_correction or apar/bpar, not both")

        jind = pre["nl_jind"]
        inverse_jind = jnp.full((mrad,), -1, dtype=jnp.int32)
        inverse_jind = inverse_jind.at[jind].set(jnp.arange(jind.shape[0], dtype=jnp.int32))

        kx_vec = pre["nl_kx2d"][:, 0]
        ky_vec = pre["nl_ky2d"][0, :]
        dum_s = pre["nl_dum_s"]

        _register_ffi()
        # v6 runs v5 itself for grids it does not instantiate
        if _has_bracket_v6 and os.environ.get("GYARADAX_BRACKET", "auto") != "v5":
            kernel_name = (
                "cufft_bracket_v6_fp32_ffi" if self.mixed_precision else "cufft_bracket_v6_fp64_ffi"
            )
        elif self.mixed_precision:
            kernel_name = "cufft_graph_bracket_true_fp32_ffi"
        else:
            kernel_name = "cufft_graph_bracket_fp64_ffi"

        # Fallback convention for currently supported CUDA mode-box grids.
        # _cuda_zero_mode_index uses concrete precomputed metadata when it can,
        # and only falls back here when the metadata is traced under JIT.
        ixzero_static = _cuda_zero_mode_index(pre, "ixzero", nkx // 2, nkx)
        iyzero_static = _cuda_zero_mode_index(pre, "iyzero", 0, nky)

        def to6(x):
            x = jnp.asarray(x)
            while x.ndim < 6:
                x = x[None]
            return x

        # gyro-averaged potential per (sp, [v,] mu, s); v axis only when bessel carries it
        bes = to6(pre["bessel"] if bessel is None else bessel)
        pot = bes * phi
        if bpar is not None and "bpar_chi_factor" in pre:
            pot = pot + to6(pre["bpar_chi_factor"]) * bpar
        vfac = jnp.zeros((1,), dtype=jnp.float64)
        em = 0
        if chi_correction is not None:
            pot = pot + to6(chi_correction)
        elif apar is not None and "apar_chi_factor" in pre:
            if pot.shape[1] == 1 and "apar_chi_vfac" in pre:
                pot = jnp.broadcast_to(pot, (nsp, 1, nmu, ns, nkx, nky))
                pot_apar = jnp.broadcast_to(bes * apar, pot.shape)
                vfac = jnp.broadcast_to(to6(pre["apar_chi_vfac"]), (nsp, nv, 1, 1, 1, 1)).reshape(
                    -1
                )
                em = 1
            else:
                pot = pot + to6(pre["apar_chi_factor"]) * apar
        n_pot_v = pot.shape[1]
        pot = jnp.broadcast_to(pot, (nsp, n_pot_v, nmu, ns, nkx, nky))
        if em:
            vfac = vfac.reshape(nsp, nv)
        else:
            pot_apar = jnp.zeros((1,), dtype=jnp.complex128)
        static = (kx_vec, ky_vec, jnp.asarray(jind, dtype=jnp.int32), inverse_jind, dum_s)

        def bracket(d, a_pot, b_pot, vf):
            nsp_l, nv_l, nmu_l = d.shape[:3] if d.ndim == 6 else (1,) + d.shape[:2]
            n_pot_v_l = a_pot.shape[1]
            a_pot = a_pot.reshape(-1, nkx, nky)
            if em:
                # [A; B] stacked along the species axis
                a_pot = jnp.concatenate([a_pot, b_pot.reshape(-1, nkx, nky)], axis=0)
            out = ffi.ffi_call(
                kernel_name,
                jax.ShapeDtypeStruct((nsp_l * nv_l * nmu_l * ns, nkx, nky), jnp.complex128),
                vmap_method="sequential",
            )(
                d.reshape(-1, nkx, nky).astype(jnp.complex128),
                a_pot.astype(jnp.complex128),
                *static,
                vf.reshape(-1),
                batch=np.int32(nsp_l * nv_l * nmu_l),
                mrad=np.int32(mrad),
                mphi=np.int32(mphi),
                nkx=np.int32(nkx),
                nky=np.int32(nky),
                nspec=np.int32(ns),
                # the store callback zeroes (ixzero, iyzero); -1 disables the masking
                ixzero=ixzero_static if exclude_zero_mode else np.int32(-1),
                iyzero=iyzero_static,
                nsp=np.int32(nsp_l),
                nv=np.int32(nv_l),
                b_inner=np.int32(n_pot_v_l * nmu_l * ns),
                em=np.int32(em),
            )
            return out.reshape(d.shape)

        args = (df, pot, pot_apar, vfac)
        if self.mesh is None:
            out = bracket(*args)
        else:
            vel = ("sp", "vp", "mu") if df.ndim == 6 else ("vp", "mu")
            pot_ax = ("sp", "vp", "mu")
            out = sharding.velocity_map(
                bracket, self.mesh, args, (vel, pot_ax, pot_ax, ("sp", "vp")), PartitionSpec(*vel)
            )
        # bracket is linear in df: efun_sign folds into the output scale
        return fft_prefactor * pre["nl_fft_scale"] * efun_sign * out
