#include "xla/ffi/api/ffi.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

namespace xla_ffi = xla::ffi;

// linear_rhs_fused_kernel:
// Fuses parallel stencils (term_par, term_vii), vpar stencils (out_d1, out_d4),
// and all elementwise terms into a single pass.
// One block per (species, v_idx, kx, ky tile); one thread per (s, ky) element of the tile.

struct LinearRhsArgs {
    const double2* df;            // (nsp, nv, nmu, ns, nkx, nky): f, or g when APAR
    const double2* phi;           // (ns, nkx, nky)
    const double*  bessel;        // (nsp, nmu, ns, nkx, nky)
    const double*  s_upar_tab;    // (nsp, nv, ns, n_class, 9)
    const double*  s_t7_tab;      // (nsp, nv, nmu, ns, n_class, 9)
    const int*     par_class;     // (ns, nkx, nky)
    const int2*    packed_maps;   // (9, ns, nkx, nky)
    const double*  utrap;         // (nsp, nmu, ns)
    const double*  abs_dum2_vp;   // (nsp, nmu, ns)
    const double*  drift_x;       // (nsp, nv, nmu, ns)
    const double*  drift_y;       // (nsp, nv, nmu, ns)
    const double*  dmaxwel_fm_ek; // (nsp, nv, nmu, ns, nky)
    const double*  fmaxwl;        // (nsp, nv, nmu, ns)
    const double*  hyper;         // (nkx, nky)
    const double*  kx_vals;       // (nkx,)
    const double*  ky_vals;       // (nky,)
    const double*  signz0;        // (nsp,)
    const double*  tmp0;          // (nsp,)
    const double2* apar;          // (ns, nkx, nky)
    const double*  chi_vfac;      // (nsp, nv)
    const double*  g2f_vfac;      // (nsp, nv)
    const double2* bpar;          // (ns, nkx, nky)
    const double*  bpar_chi;      // (nsp, nmu, ns, nkx, nky)
    const double*  s_disp_tab;    // (nsp, nv, ns, n_class, 9)
    const double2* dproj_m0;      // (nsp, ns, nkx, nky)
    const double2* dproj_m1;      // (nsp, ns, nkx, nky)
    const double*  dproj_e0;      // (nsp, nv, nmu, ns, nkx, nky)
    const double*  dproj_e1;      // (nsp, nv, nmu, ns, nkx, nky)
    double2*       rhs_out;       // (nsp, nv, nmu, ns, nkx, nky)
    int nsp, nv, nmu, ns, nkx, nky, n_class;
    int ky_tile;                  // ky per block (nky unless ns * nky > 1024)
    double c_d1_0, c_d1_1, c_d1_2, c_d1_3, c_d1_4;
    double c_d4_0, c_d4_1, c_d4_2, c_d4_3, c_d4_4;
    double dvp, disp_vp, drive_scale;
};

// g2f_correct: g2f = -2 signz vthrat vpgr J0 fmaxwl / tmp
__device__ __forceinline__ double2 g_to_f(double2 g, double g2f_v, double bes, double fm,
                                          double inv_tmp0, double2 apar) {
    const double g2f = g2f_v * bes * fm * inv_tmp0;
    return make_double2(g.x + g2f * apar.x, g.y + g2f * apar.y);
}

// APAR: df is g, f = g + g2f*apar formed in-kernel; BPAR: psi = J0*phi + bpar_chi*bpar; NS == 0: runtime-sized
template <int NS, int NKY, bool APAR, bool BPAR, bool DPC>
__device__ __forceinline__ void linear_rhs_fused_body(const LinearRhsArgs& a) {
    // NS > 0: compile-time sizes; NS == 0: runtime sizes; NS < 0: runtime sizes with ky tiles
    constexpr bool kDyn   = (NS <= 0);
    constexpr bool kTiled = (NS < 0);
    const int ns  = kDyn ? a.ns  : NS;
    const int nky = kDyn ? a.nky : NKY;
    const int nv = a.nv, nmu = a.nmu, nkx = a.nkx, nsp = a.nsp;
    const int nv_nmu = nv * nmu;

    const int ky_tile = kTiled ? a.ky_tile : nky;

    extern __shared__ double2 smem_total[];
    double2* smem_df   = smem_total;
    double2* smem_gyro = smem_total + ns * ky_tile;
    double2* smem_proj = smem_total + 2 * ns * ky_tile;

    const int local_tid = threadIdx.x;
    int kx, v_idx, sp, s, ky, ky_local;
    bool live = true;
    if constexpr (kTiled) {
        // the parallel stencil couples s (and kx at the connections) but not ky, so ky tiles are independent
        const int n_tiles = (nky + ky_tile - 1) / ky_tile;
        const int tile    = blockIdx.x % n_tiles;
        const int blk     = blockIdx.x / n_tiles;
        kx       = blk % nkx;
        v_idx    = (blk / nkx) % nv_nmu;
        sp       = blk / (nkx * nv_nmu);
        s        = local_tid / ky_tile;
        ky_local = local_tid % ky_tile;
        // a partial last tile keeps its idle threads for the barriers, on a valid ky
        live = tile * ky_tile + ky_local < nky;
        ky   = live ? tile * ky_tile + ky_local : nky - 1;
    } else {
        kx       = blockIdx.x % nkx;
        v_idx    = (blockIdx.x / nkx) % nv_nmu; // This is (v * nmu + mu)
        sp       = blockIdx.x / (nkx * nv_nmu);
        s        = local_tid / nky;
        ky       = local_tid % nky;
        ky_local = ky;
    }

    const int mu_idx    = v_idx % nmu;
    const int v_phys    = v_idx / nmu;

    const size_t spatial_stride = (size_t)ns * nkx * nky;
    const size_t spatial_idx    = (size_t)s * (nkx * nky) + (size_t)kx * nky + ky;
    const size_t slab_base      = ((size_t)sp * nv_nmu + v_idx) * spatial_stride;
    const size_t field_idx      = slab_base + spatial_idx;

    // (sp, v, mu, s) and (sp, mu, s) table offsets
    const size_t v_mu_s_idx = (((size_t)sp * nv + v_phys) * nmu + mu_idx) * ns + s;
    const size_t mu_s_idx   = ((size_t)sp * nmu + mu_idx) * ns + s;
    const size_t sp_v_idx   = (size_t)sp * nv + v_phys;

    const double signz0 = __ldg(&a.signz0[sp]);
    const double tmp0   = __ldg(&a.tmp0[sp]);
    const double inv_tmp0 = 1.0 / tmp0;

    // bessel dependency: (sp, mu, s, kx, ky)
    const double* bessel_mu = a.bessel + ((size_t)sp * nmu + mu_idx) * spatial_stride;
    const double* bpar_chi_mu = BPAR ? a.bpar_chi + ((size_t)sp * nmu + mu_idx) * spatial_stride
                                     : nullptr;
    const double bes        = __ldg(&bessel_mu[spatial_idx]);
    const double fmaxwl_val = __ldg(&a.fmaxwl[v_mu_s_idx]);
    const double g2f_v      = APAR ? __ldg(&a.g2f_vfac[sp_v_idx]) : 0.0;
    const double2 apar_val  = APAR ? __ldg(&a.apar[spatial_idx]) : make_double2(0.0, 0.0);

    // Load df (f, reconstructed from g in the A_par variant)
    double2 my_df = __ldg(&a.df[field_idx]);
    if (APAR) my_df = g_to_f(my_df, g2f_v, bes, fmaxwl_val, inv_tmp0, apar_val);
    smem_df[local_tid] = my_df;

    // Compute and store gyro_phi (psi = J0 phi + bpar_chi bpar with B_par)
    double2 phi_val = __ldg(&a.phi[spatial_idx]);
    double2 my_gyro_phi = make_double2(bes * phi_val.x, bes * phi_val.y);
    if (BPAR) {
        const double bc   = __ldg(&bpar_chi_mu[spatial_idx]);
        const double2 bp  = __ldg(&a.bpar[spatial_idx]);
        my_gyro_phi.x += bc * bp.x;
        my_gyro_phi.y += bc * bp.y;
    }
    smem_gyro[local_tid] = my_gyro_phi;

    if (DPC) {
        const size_t m_idx = (size_t)sp * spatial_stride + spatial_idx;
        const double e0 = __ldg(&a.dproj_e0[field_idx]);
        const double e1 = __ldg(&a.dproj_e1[field_idx]);
        const double2 m0 = __ldg(&a.dproj_m0[m_idx]);
        const double2 m1 = __ldg(&a.dproj_m1[m_idx]);
        smem_proj[local_tid] = make_double2(e0 * m0.x + e1 * m1.x, e0 * m0.y + e1 * m1.y);
    }

    __syncthreads();

    // ── Parallel Stencils (Phase 1) ──
    // fused stencil coefficients per (sp, v, [mu,] s) row, stencil class and shift
    const int cls = __ldg(&a.par_class[spatial_idx]);
    const double* c_upar_row = a.s_upar_tab + (((size_t)sp_v_idx * ns + s) * a.n_class + cls) * 9;
    const double* c_t7_row   = a.s_t7_tab + ((size_t)v_mu_s_idx * a.n_class + cls) * 9;
    const double* c_dp_row   = DPC ? a.s_disp_tab + (((size_t)sp_v_idx * ns + s) * a.n_class + cls) * 9
                                   : nullptr;

    double acc_par_r = 0.0, acc_par_i = 0.0;
    double acc_t7_r  = 0.0, acc_t7_i  = 0.0;
    double acc_dp_r  = 0.0, acc_dp_i  = 0.0;

    #pragma unroll
    for (int i = 0; i < 9; ++i) {
        const int2   map_val = __ldg(&a.packed_maps[(size_t)i * spatial_stride + spatial_idx]);
        const int    src_s   = map_val.x;
        if (src_s >= 0) {
            const int    src_kx = map_val.y;

            const double c_upar = __ldg(&c_upar_row[i]);
            const double c_t7   = __ldg(&c_t7_row[i]);

            double2 v_df, v_gyro, v_proj;
            if (src_kx == kx) {
                v_df   = smem_df[src_s * ky_tile + ky_local];
                v_gyro = smem_gyro[src_s * ky_tile + ky_local];
                if (DPC) v_proj = smem_proj[src_s * ky_tile + ky_local];
            } else {
                const size_t src_spatial_idx = (size_t)src_s * (nkx * nky)
                                             + (size_t)src_kx * nky + ky;
                v_df = __ldg(&a.df[slab_base + src_spatial_idx]);

                double bes_src = __ldg(&bessel_mu[src_spatial_idx]);
                if (APAR) {
                    const double fm_src = __ldg(&a.fmaxwl[v_mu_s_idx - s + src_s]);
                    v_df = g_to_f(v_df, g2f_v, bes_src, fm_src, inv_tmp0,
                                  __ldg(&a.apar[src_spatial_idx]));
                }
                double2 phi_src = __ldg(&a.phi[src_spatial_idx]);
                v_gyro = make_double2(bes_src * phi_src.x, bes_src * phi_src.y);
                if (BPAR) {
                    const double bc  = __ldg(&bpar_chi_mu[src_spatial_idx]);
                    const double2 bp = __ldg(&a.bpar[src_spatial_idx]);
                    v_gyro.x += bc * bp.x;
                    v_gyro.y += bc * bp.y;
                }
                if (DPC) {
                    const size_t m_src = (size_t)sp * spatial_stride + src_spatial_idx;
                    const double e0 = __ldg(&a.dproj_e0[slab_base + src_spatial_idx]);
                    const double e1 = __ldg(&a.dproj_e1[slab_base + src_spatial_idx]);
                    const double2 m0 = __ldg(&a.dproj_m0[m_src]);
                    const double2 m1 = __ldg(&a.dproj_m1[m_src]);
                    v_proj = make_double2(e0 * m0.x + e1 * m1.x, e0 * m0.y + e1 * m1.y);
                }
            }
            acc_par_r += v_df.x * c_upar;
            acc_par_i += v_df.y * c_upar;
            acc_t7_r  += v_gyro.x * c_t7;
            acc_t7_i  += v_gyro.y * c_t7;
            if (DPC) {
                const double c_dp = __ldg(&c_dp_row[i]);
                acc_dp_r += v_proj.x * c_dp;
                acc_dp_i += v_proj.y * c_dp;
            }
        }
    }

    // ── Vpar Stencils (Phase 2) ──
    const size_t vpar_stride = (size_t)nmu * spatial_stride;
    const size_t fm_vstride  = (size_t)nmu * ns;
    auto load_vpar = [&](int dv) -> double2 {
        const double2 g = __ldg(&a.df[field_idx + dv * (long long)vpar_stride]);
        if (!APAR) return g;
        const double fm = __ldg(&a.fmaxwl[v_mu_s_idx + dv * (long long)fm_vstride]);
        return g_to_f(g, __ldg(&a.g2f_vfac[sp_v_idx + dv]), bes, fm, inv_tmp0, apar_val);
    };

    double2 df_vm2 = (v_phys >= 2)      ? load_vpar(-2) : make_double2(0.0, 0.0);
    double2 df_vm1 = (v_phys >= 1)      ? load_vpar(-1) : make_double2(0.0, 0.0);
    double2 df_vp1 = (v_phys <= nv - 2) ? load_vpar(+1) : make_double2(0.0, 0.0);
    double2 df_vp2 = (v_phys <= nv - 3) ? load_vpar(+2) : make_double2(0.0, 0.0);

    double2 out_d1 = make_double2(
        a.c_d1_0 * df_vm2.x + a.c_d1_1 * df_vm1.x + a.c_d1_2 * my_df.x + a.c_d1_3 * df_vp1.x + a.c_d1_4 * df_vp2.x,
        a.c_d1_0 * df_vm2.y + a.c_d1_1 * df_vm1.y + a.c_d1_2 * my_df.y + a.c_d1_3 * df_vp1.y + a.c_d1_4 * df_vp2.y
    );
    double2 out_d4 = make_double2(
        a.c_d4_0 * df_vm2.x + a.c_d4_1 * df_vm1.x + a.c_d4_2 * my_df.x + a.c_d4_3 * df_vp1.x + a.c_d4_4 * df_vp2.x,
        a.c_d4_0 * df_vm2.y + a.c_d4_1 * df_vm1.y + a.c_d4_2 * my_df.y + a.c_d4_3 * df_vp1.y + a.c_d4_4 * df_vp2.y
    );

    // ── Elementwise Assembly (Phase 3) ──
    double utrap_val   = __ldg(&a.utrap[mu_s_idx]);
    double abs_vp_val  = __ldg(&a.abs_dum2_vp[mu_s_idx]);
    double drift_x_val = __ldg(&a.drift_x[v_mu_s_idx]);
    double drift_y_val = __ldg(&a.drift_y[v_mu_s_idx]);
    double dmaxwel_val = __ldg(&a.dmaxwel_fm_ek[v_mu_s_idx * nky + ky]);

    double kx_val      = __ldg(&a.kx_vals[kx]);
    double ky_val      = __ldg(&a.ky_vals[ky]);
    double hyper_val   = __ldg(&a.hyper[(size_t)kx * nky + ky]);

    double kdotvd  = drift_x_val * kx_val + drift_y_val * ky_val;
    double inv_dvp = 1.0 / a.dvp;
    double inv_tmp = 1.0 / fmax(tmp0, 1e-15);

    // term_iv = utrap * out_d1 / dvp
    double2 term_iv = make_double2(utrap_val * out_d1.x * inv_dvp, utrap_val * out_d1.y * inv_dvp);

    // term_vp_diss = disp_vp * abs_vp * out_d4 / dvp
    double vp_diss_coeff = a.disp_vp * abs_vp_val * inv_dvp;
    double2 term_vp_diss = make_double2(vp_diss_coeff * out_d4.x, vp_diss_coeff * out_d4.y);

    // -1j * kdotvd * df => (kdotvd * df.y, -kdotvd * df.x)
    double2 drift_term = make_double2(kdotvd * my_df.y, -kdotvd * my_df.x);

    // hyper * df
    double2 hyper_term = make_double2(hyper_val * my_df.x, hyper_val * my_df.y);

    double2 drive_term;
    if (!APAR && !BPAR) {
        // drive = 1j * drive_scale * (dmaxwel - signz0 * kdotvd * fmaxwl / tmp0) * gyro_phi
        double drive_tot_coeff = a.drive_scale * (dmaxwel_val - signz0 * kdotvd * fmaxwl_val * inv_tmp);
        drive_term = make_double2(-drive_tot_coeff * my_gyro_phi.y, drive_tot_coeff * my_gyro_phi.x);
    } else {
        // term V on chi, terms VIII + XI on psi
        double2 chi = my_gyro_phi;
        if (APAR) {
            const double chi_a = __ldg(&a.chi_vfac[sp_v_idx]) * bes;
            chi.x += chi_a * apar_val.x;
            chi.y += chi_a * apar_val.y;
        }
        const double c_v    = a.drive_scale * dmaxwel_val;
        const double c_curv = a.drive_scale * signz0 * kdotvd * fmaxwl_val * inv_tmp;
        drive_term = make_double2(-c_v * chi.y + c_curv * my_gyro_phi.y,
                                   c_v * chi.x - c_curv * my_gyro_phi.x);
    }

    // Final Sum
    double2 res;
    res.x = acc_par_r + term_iv.x + term_vp_diss.x + drift_term.x + hyper_term.x + drive_term.x + acc_t7_r;
    res.y = acc_par_i + term_iv.y + term_vp_diss.y + drift_term.y + hyper_term.y + drive_term.y + acc_t7_i;
    // conservative parallel dissipation: drop the dissipation of the field-sourcing projection
    if (DPC) {
        res.x -= acc_dp_r;
        res.y -= acc_dp_i;
    }

    if (live) a.rhs_out[field_idx] = res;
}

template <int NS, int NKY, int MAX_THREADS, bool DPC>
__global__ __launch_bounds__(MAX_THREADS)
void linear_rhs_fused_kernel(const LinearRhsArgs a) {
    linear_rhs_fused_body<NS, NKY, false, false, DPC>(a);
}

// the EM variants are capped at 64 registers so that 1024 threads stay resident per SM
template <int NS, int NKY, int MAX_THREADS, bool APAR, bool BPAR, bool DPC>
__global__ __launch_bounds__(MAX_THREADS, 1024 / MAX_THREADS)
void linear_rhs_fused_em_kernel(const LinearRhsArgs a) {
    linear_rhs_fused_body<NS, NKY, APAR, BPAR, DPC>(a);
}

template <int NS, int NKY, int MAX_THREADS, bool APAR, bool BPAR, bool DPC>
static void launch_variant(const LinearRhsArgs& args, int num_blocks, int threads, size_t smem,
                           cudaStream_t stream) {
    if constexpr (APAR || BPAR)
        linear_rhs_fused_em_kernel<NS, NKY, MAX_THREADS, APAR, BPAR, DPC>
            <<<num_blocks, threads, smem, stream>>>(args);
    else
        linear_rhs_fused_kernel<NS, NKY, MAX_THREADS, DPC><<<num_blocks, threads, smem, stream>>>(args);
}


// ── FFI Implementation ──────────────────────────────────────────────────────

template <bool APAR, bool BPAR, bool DPC>
static cudaError_t launch_linear_rhs(const LinearRhsArgs& args, cudaStream_t stream) {
    const int n_tiles    = (args.nky + args.ky_tile - 1) / args.ky_tile;
    const int num_blocks = args.nsp * args.nv * args.nmu * args.nkx * n_tiles;
    const int threads    = args.ns * args.ky_tile;
    const int n_smem     = DPC ? 3 : 2;
    const size_t smem    = (size_t)n_smem * threads * sizeof(double2);

#define DISPATCH_CASE(NS_VAL, NKY_VAL)                                                   \
    case (((NS_VAL) << 16) | (NKY_VAL)):                                                 \
        launch_variant<NS_VAL, NKY_VAL, (NS_VAL) * (NKY_VAL), APAR, BPAR, DPC>(          \
            args, num_blocks, (NS_VAL) * (NKY_VAL), smem, stream);                       \
        break;

    if (args.ky_tile != args.nky) {
        launch_variant<-1, 0, 1024, APAR, BPAR, DPC>(args, num_blocks, threads, smem, stream);
        return cudaGetLastError();
    }
    switch ((args.ns << 16) | args.nky) {
        DISPATCH_CASE(16, 32)
        DISPATCH_CASE(32, 32)
        DISPATCH_CASE(16, 64)
        default:
            // small blocks get the register budget of a 256-thread launch
            if (threads <= 256)
                launch_variant<0, 0, 256, APAR, BPAR, DPC>(args, num_blocks, threads, smem, stream);
            else
                launch_variant<0, 0, 1024, APAR, BPAR, DPC>(args, num_blocks, threads, smem, stream);
    }
#undef DISPATCH_CASE
    return cudaGetLastError();
}

xla_ffi::Error LinearRhsFusedImpl(
    cudaStream_t stream,
    xla_ffi::Buffer<xla_ffi::DataType::C128> df,
    xla_ffi::Buffer<xla_ffi::DataType::C128> phi,
    xla_ffi::Buffer<xla_ffi::DataType::F64>  bessel,
    xla_ffi::Buffer<xla_ffi::DataType::F64>  s_upar_tab,
    xla_ffi::Buffer<xla_ffi::DataType::F64>  s_t7_tab,
    xla_ffi::Buffer<xla_ffi::DataType::S32>  par_class,
    xla_ffi::Buffer<xla_ffi::DataType::S32>  packed_maps,
    xla_ffi::Buffer<xla_ffi::DataType::F64>  utrap,
    xla_ffi::Buffer<xla_ffi::DataType::F64>  abs_dum2_vp,
    xla_ffi::Buffer<xla_ffi::DataType::F64>  drift_x,
    xla_ffi::Buffer<xla_ffi::DataType::F64>  drift_y,
    xla_ffi::Buffer<xla_ffi::DataType::F64>  dmaxwel_fm_ek,
    xla_ffi::Buffer<xla_ffi::DataType::F64>  fmaxwl,
    xla_ffi::Buffer<xla_ffi::DataType::F64>  hyper,
    xla_ffi::Buffer<xla_ffi::DataType::F64>  kx_vals,
    xla_ffi::Buffer<xla_ffi::DataType::F64>  ky_vals,
    xla_ffi::Buffer<xla_ffi::DataType::F64>  signz0_buf,
    xla_ffi::Buffer<xla_ffi::DataType::F64>  tmp0_buf,
    xla_ffi::Buffer<xla_ffi::DataType::C128> apar,
    xla_ffi::Buffer<xla_ffi::DataType::F64>  chi_vfac,
    xla_ffi::Buffer<xla_ffi::DataType::F64>  g2f_vfac,
    xla_ffi::Buffer<xla_ffi::DataType::C128> bpar,
    xla_ffi::Buffer<xla_ffi::DataType::F64>  bpar_chi,
    xla_ffi::Buffer<xla_ffi::DataType::F64>  s_disp_tab,
    xla_ffi::Buffer<xla_ffi::DataType::C128> dproj_m0,
    xla_ffi::Buffer<xla_ffi::DataType::C128> dproj_m1,
    xla_ffi::Buffer<xla_ffi::DataType::F64>  dproj_e0,
    xla_ffi::Buffer<xla_ffi::DataType::F64>  dproj_e1,
    xla_ffi::Result<xla_ffi::Buffer<xla_ffi::DataType::C128>> rhs_out,
    int32_t nsp, int32_t nv, int32_t nmu, int32_t ns, int32_t nkx, int32_t nky, int32_t n_class,
    int32_t has_apar, int32_t has_bpar, int32_t has_dpc,
    double c_d1_0, double c_d1_1, double c_d1_2, double c_d1_3, double c_d1_4,
    double c_d4_0, double c_d4_1, double c_d4_2, double c_d4_3, double c_d4_4,
    double dvp, double disp_vp, double drive_scale
) {
    if (ns > 1024)
        return xla_ffi::Error(XLA_FFI_Error_Code_INVALID_ARGUMENT,
            "ns exceeds maximum CUDA block size of 1024");
    const int ky_tile = ns * nky <= 1024 ? nky : 1024 / ns;

    LinearRhsArgs args = {
        (const double2*)df.typed_data(), (const double2*)phi.typed_data(),
        bessel.typed_data(), s_upar_tab.typed_data(), s_t7_tab.typed_data(),
        par_class.typed_data(),
        (const int2*)packed_maps.typed_data(),
        utrap.typed_data(), abs_dum2_vp.typed_data(),
        drift_x.typed_data(), drift_y.typed_data(),
        dmaxwel_fm_ek.typed_data(), fmaxwl.typed_data(),
        hyper.typed_data(), kx_vals.typed_data(), ky_vals.typed_data(),
        signz0_buf.typed_data(), tmp0_buf.typed_data(),
        (const double2*)apar.typed_data(), chi_vfac.typed_data(), g2f_vfac.typed_data(),
        (const double2*)bpar.typed_data(), bpar_chi.typed_data(),
        s_disp_tab.typed_data(),
        (const double2*)dproj_m0.typed_data(), (const double2*)dproj_m1.typed_data(),
        dproj_e0.typed_data(), dproj_e1.typed_data(),
        (double2*)rhs_out->typed_data(),
        nsp, nv, nmu, ns, nkx, nky, n_class, ky_tile,
        c_d1_0, c_d1_1, c_d1_2, c_d1_3, c_d1_4,
        c_d4_0, c_d4_1, c_d4_2, c_d4_3, c_d4_4,
        dvp, disp_vp, drive_scale
    };

    cudaError_t err;
    const int variant = (has_apar ? 4 : 0) | (has_bpar ? 2 : 0) | (has_dpc ? 1 : 0);
    switch (variant) {
        case 0: err = launch_linear_rhs<false, false, false>(args, stream); break;
        case 1: err = launch_linear_rhs<false, false, true >(args, stream); break;
        case 2: err = launch_linear_rhs<false, true,  false>(args, stream); break;
        case 3: err = launch_linear_rhs<false, true,  true >(args, stream); break;
        case 4: err = launch_linear_rhs<true,  false, false>(args, stream); break;
        case 5: err = launch_linear_rhs<true,  false, true >(args, stream); break;
        case 6: err = launch_linear_rhs<true,  true,  false>(args, stream); break;
        default: err = launch_linear_rhs<true, true,  true >(args, stream); break;
    }
    if (err != cudaSuccess)
        return xla_ffi::Error(XLA_FFI_Error_Code_INTERNAL, cudaGetErrorString(err));

    return xla_ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    linear_rhs_fused_ffi, LinearRhsFusedImpl,
    xla_ffi::Ffi::Bind()
        .Ctx<xla_ffi::PlatformStream<cudaStream_t>>()
        .Arg<xla_ffi::Buffer<xla_ffi::DataType::C128>>() // df
        .Arg<xla_ffi::Buffer<xla_ffi::DataType::C128>>() // phi
        .Arg<xla_ffi::Buffer<xla_ffi::DataType::F64>>()  // bessel
        .Arg<xla_ffi::Buffer<xla_ffi::DataType::F64>>()  // s_upar_tab
        .Arg<xla_ffi::Buffer<xla_ffi::DataType::F64>>()  // s_t7_tab
        .Arg<xla_ffi::Buffer<xla_ffi::DataType::S32>>()  // par_class
        .Arg<xla_ffi::Buffer<xla_ffi::DataType::S32>>()  // packed_maps
        .Arg<xla_ffi::Buffer<xla_ffi::DataType::F64>>()  // utrap
        .Arg<xla_ffi::Buffer<xla_ffi::DataType::F64>>()  // abs_dum2_vp
        .Arg<xla_ffi::Buffer<xla_ffi::DataType::F64>>()  // drift_x
        .Arg<xla_ffi::Buffer<xla_ffi::DataType::F64>>()  // drift_y
        .Arg<xla_ffi::Buffer<xla_ffi::DataType::F64>>()  // dmaxwel_fm_ek
        .Arg<xla_ffi::Buffer<xla_ffi::DataType::F64>>()  // fmaxwl
        .Arg<xla_ffi::Buffer<xla_ffi::DataType::F64>>()  // hyper
        .Arg<xla_ffi::Buffer<xla_ffi::DataType::F64>>()  // kx_vals
        .Arg<xla_ffi::Buffer<xla_ffi::DataType::F64>>()  // ky_vals
        .Arg<xla_ffi::Buffer<xla_ffi::DataType::F64>>()  // signz0
        .Arg<xla_ffi::Buffer<xla_ffi::DataType::F64>>()  // tmp0
        .Arg<xla_ffi::Buffer<xla_ffi::DataType::C128>>() // apar
        .Arg<xla_ffi::Buffer<xla_ffi::DataType::F64>>()  // chi_vfac
        .Arg<xla_ffi::Buffer<xla_ffi::DataType::F64>>()  // g2f_vfac
        .Arg<xla_ffi::Buffer<xla_ffi::DataType::C128>>() // bpar
        .Arg<xla_ffi::Buffer<xla_ffi::DataType::F64>>()  // bpar_chi
        .Arg<xla_ffi::Buffer<xla_ffi::DataType::F64>>()  // s_disp_tab
        .Arg<xla_ffi::Buffer<xla_ffi::DataType::C128>>() // dproj_m0
        .Arg<xla_ffi::Buffer<xla_ffi::DataType::C128>>() // dproj_m1
        .Arg<xla_ffi::Buffer<xla_ffi::DataType::F64>>()  // dproj_e0
        .Arg<xla_ffi::Buffer<xla_ffi::DataType::F64>>()  // dproj_e1
        .Ret<xla_ffi::Buffer<xla_ffi::DataType::C128>>() // rhs_out
        .Attr<int32_t>("nsp")
        .Attr<int32_t>("nv")
        .Attr<int32_t>("nmu")
        .Attr<int32_t>("ns")
        .Attr<int32_t>("nkx")
        .Attr<int32_t>("nky")
        .Attr<int32_t>("n_class")
        .Attr<int32_t>("has_apar")
        .Attr<int32_t>("has_bpar")
        .Attr<int32_t>("has_dpc")
        .Attr<double>("c_d1_0").Attr<double>("c_d1_1").Attr<double>("c_d1_2")
        .Attr<double>("c_d1_3").Attr<double>("c_d1_4")
        .Attr<double>("c_d4_0").Attr<double>("c_d4_1").Attr<double>("c_d4_2")
        .Attr<double>("c_d4_3").Attr<double>("c_d4_4")
        .Attr<double>("dvp")
        .Attr<double>("disp_vp")
        .Attr<double>("drive_scale")
);
