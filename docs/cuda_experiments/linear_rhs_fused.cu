#include "xla/ffi/api/ffi.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

namespace xla_ffi = xla::ffi;

// linear_rhs_fused_kernel:
// Fuses parallel stencils (term_par, term_vii), vpar stencils (out_d1, out_d4),
// and all elementwise terms into a single pass.
// One block per (v_idx, kx) pair; one thread per (s, ky) element.

template <int NS, int NKY>
__global__ __launch_bounds__(NS * NKY)
void linear_rhs_fused_kernel(
    const double2* __restrict__ df,          // (nv_nmu, ns, nkx, nky)
    const double2* __restrict__ phi,         // (ns, nkx, nky)
    const double*  __restrict__ bessel,      // (nmu, ns, nkx, nky)
    const double*  __restrict__ s_total_upar,// (9, nv, 1, ns, nkx, nky)
    const double*  __restrict__ s_total_t7,  // (9, nv, nmu, ns, nkx, nky)
    const int2*    __restrict__ packed_maps,  // (9, ns, nkx, nky)
    const double*  __restrict__ utrap,       // (nmu, ns)
    const double*  __restrict__ abs_dum2_vp, // (nmu, ns)
    const double*  __restrict__ drift_x,     // (nv, nmu, ns)
    const double*  __restrict__ drift_y,     // (nv, nmu, ns)
    const double*  __restrict__ dmaxwel_fm_ek, // (nv, nmu, ns, nky)
    const double*  __restrict__ fmaxwl,      // (nv, nmu, ns)
    const double*  __restrict__ hyper,       // (ns, nkx, nky)
    const double*  __restrict__ kx_vals,     // (nkx,)
    const double*  __restrict__ ky_vals,     // (nky,)
    double2*       __restrict__ rhs_out,
    int nv, int nmu, int nkx, int nky_param, int nv_total,
    double c_d1_0, double c_d1_1, double c_d1_2, double c_d1_3, double c_d1_4,
    double c_d4_0, double c_d4_1, double c_d4_2, double c_d4_3, double c_d4_4,
    double dvp, double disp_vp, double drive_scale, double signz0, double tmp0
) {
    __shared__ double2 smem_df[NS * NKY];
    __shared__ double2 smem_gyro[NS * NKY];

    const int local_tid = threadIdx.x;
    const int v_idx     = blockIdx.x / nkx; // This is (v * nmu + mu)
    const int kx        = blockIdx.x % nkx;
    const int s         = local_tid / NKY;
    const int ky        = local_tid % NKY;

    const int mu_idx    = v_idx % nmu;
    const int v_phys    = v_idx / nmu;

    const size_t spatial_stride = (size_t)NS * nkx * NKY;
    const size_t spatial_idx    = (size_t)s * (nkx * NKY) + (size_t)kx * NKY + ky;
    const size_t field_idx      = (size_t)v_idx * spatial_stride + spatial_idx;

    // Load df
    double2 my_df = __ldg(&df[field_idx]);
    smem_df[local_tid] = my_df;

    // Compute and store gyro_phi
    // bessel dependency: (mu, s, kx, ky)
    const size_t bessel_idx = (((size_t)mu_idx * NS + s) * nkx + kx) * NKY + ky;
    double2 phi_val = __ldg(&phi[spatial_idx]);
    double bes = __ldg(&bessel[bessel_idx]);
    double2 my_gyro_phi = make_double2(bes * phi_val.x, bes * phi_val.y);
    smem_gyro[local_tid] = my_gyro_phi;

    __syncthreads();

    // ── Parallel Stencils (Phase 1) ──
    const size_t spatial_idx_2d = (size_t)s * (nkx * NKY) + (size_t)kx * NKY + ky;
    
    double acc_par_r = 0.0, acc_par_i = 0.0;
    double acc_t7_r  = 0.0, acc_t7_i  = 0.0;

    #pragma unroll
    for (int i = 0; i < 9; ++i) {
        const int2   map_val = __ldg(&packed_maps[(size_t)i * spatial_stride + spatial_idx]);
        const int    src_s   = map_val.x;
        if (src_s >= 0) {
            const int    src_kx = map_val.y;
            
            // s_total_upar: (9, nv, 1, ns, nkx, nky)
            const size_t upar_stride = (size_t)nv * 1 * NS * nkx * NKY;
            const size_t upar_idx = (size_t)i * upar_stride 
                                  + (((size_t)v_phys * 1 + 0) * NS + s) * (nkx * NKY)
                                  + (size_t)kx * NKY + ky;

            // s_total_t7: (9, nv, nmu, ns, nkx, nky)
            const size_t t7_stride = (size_t)nv * nmu * NS * nkx * NKY;
            const size_t t7_idx = (size_t)i * t7_stride + field_idx;

            const double c_upar = __ldg(&s_total_upar[upar_idx]);
            const double c_t7   = __ldg(&s_total_t7[t7_idx]);
            
            double2 v_df, v_gyro;
            if (src_kx == kx) {
                v_df   = smem_df[src_s * NKY + ky];
                v_gyro = smem_gyro[src_s * NKY + ky];
            } else {
                const size_t src_field_idx = (size_t)v_idx * spatial_stride
                                           + (size_t)src_s * (nkx * NKY)
                                           + (size_t)src_kx * NKY + ky;
                const size_t src_spatial_idx = (size_t)src_s * (nkx * NKY)
                                             + (size_t)src_kx * NKY + ky;
                v_df = __ldg(&df[src_field_idx]);
                
                const size_t bessel_src_idx = (((size_t)mu_idx * NS + src_s) * nkx + src_kx) * NKY + ky;
                double bes_src = __ldg(&bessel[bessel_src_idx]);
                double2 phi_src = __ldg(&phi[src_spatial_idx]);
                v_gyro = make_double2(bes_src * phi_src.x, bes_src * phi_src.y);
            }
            acc_par_r += v_df.x * c_upar;
            acc_par_i += v_df.y * c_upar;
            acc_t7_r  += v_gyro.x * c_t7;
            acc_t7_i  += v_gyro.y * c_t7;
        }
    }

    // ── Vpar Stencils (Phase 2) ──
    const size_t vpar_stride = (size_t)nmu * spatial_stride;

    double2 df_vm2 = (v_phys >= 2)      ? __ldg(&df[field_idx - 2 * vpar_stride]) : make_double2(0.0, 0.0);
    double2 df_vm1 = (v_phys >= 1)      ? __ldg(&df[field_idx - 1 * vpar_stride]) : make_double2(0.0, 0.0);
    double2 df_vp1 = (v_phys <= nv - 2) ? __ldg(&df[field_idx + 1 * vpar_stride]) : make_double2(0.0, 0.0);
    double2 df_vp2 = (v_phys <= nv - 3) ? __ldg(&df[field_idx + 2 * vpar_stride]) : make_double2(0.0, 0.0);

    double2 out_d1 = make_double2(
        c_d1_0 * df_vm2.x + c_d1_1 * df_vm1.x + c_d1_2 * my_df.x + c_d1_3 * df_vp1.x + c_d1_4 * df_vp2.x,
        c_d1_0 * df_vm2.y + c_d1_1 * df_vm1.y + c_d1_2 * my_df.y + c_d1_3 * df_vp1.y + c_d1_4 * df_vp2.y
    );
    double2 out_d4 = make_double2(
        c_d4_0 * df_vm2.x + c_d4_1 * df_vm1.x + c_d4_2 * my_df.x + c_d4_3 * df_vp1.x + c_d4_4 * df_vp2.x,
        c_d4_0 * df_vm2.y + c_d4_1 * df_vm1.y + c_d4_2 * my_df.y + c_d4_3 * df_vp1.y + c_d4_4 * df_vp2.y
    );

    // ── Elementwise Assembly (Phase 3) ──
    // utrap, abs_dum2_vp: (nmu, ns)
    const size_t mu_s_idx = (size_t)mu_idx * NS + s;
    double utrap_val   = __ldg(&utrap[mu_s_idx]);
    double abs_vp_val  = __ldg(&abs_dum2_vp[mu_s_idx]);
    
    // drift_x, drift_y, fmaxwl: (nv, nmu, ns)
    const size_t v_mu_s_idx = ((size_t)v_phys * nmu + mu_idx) * NS + s;
    double drift_x_val = __ldg(&drift_x[v_mu_s_idx]);
    double drift_y_val = __ldg(&drift_y[v_mu_s_idx]);
    double fmaxwl_val  = __ldg(&fmaxwl[v_mu_s_idx]);

    // dmaxwel_fm_ek: (nv, nmu, ns, nky)
    const size_t dmaxwel_idx = (((size_t)v_phys * nmu + mu_idx) * NS + s) * NKY + ky;
    double dmaxwel_val = __ldg(&dmaxwel_fm_ek[dmaxwel_idx]);
    
    double kx_val      = __ldg(&kx_vals[kx]);
    double ky_val      = __ldg(&ky_vals[ky]);
    double hyper_val   = __ldg(&hyper[spatial_idx]);

    double kdotvd  = drift_x_val * kx_val + drift_y_val * ky_val;
    double inv_dvp = 1.0 / dvp;
    double inv_tmp = 1.0 / fmax(tmp0, 1e-15);

    // term_iv = utrap * out_d1 / dvp
    double2 term_iv = make_double2(utrap_val * out_d1.x * inv_dvp, utrap_val * out_d1.y * inv_dvp);

    // term_vp_diss = disp_vp * abs_vp * out_d4 / dvp
    double vp_diss_coeff = disp_vp * abs_vp_val * inv_dvp;
    double2 term_vp_diss = make_double2(vp_diss_coeff * out_d4.x, vp_diss_coeff * out_d4.y);

    // -1j * kdotvd * df => (kdotvd * df.y, -kdotvd * df.x)
    double2 drift_term = make_double2(kdotvd * my_df.y, -kdotvd * my_df.x);

    // hyper * df
    double2 hyper_term = make_double2(hyper_val * my_df.x, hyper_val * my_df.y);

    // drive = 1j * drive_scale * (dmaxwel - signz0 * kdotvd * fmaxwl / tmp0) * gyro_phi
    double drive_tot_coeff = drive_scale * (dmaxwel_val - signz0 * kdotvd * fmaxwl_val * inv_tmp);
    double2 drive_term = make_double2(-drive_tot_coeff * my_gyro_phi.y, drive_tot_coeff * my_gyro_phi.x);

    // Final Sum
    double2 res;
    res.x = acc_par_r + term_iv.x + term_vp_diss.x + drift_term.x + hyper_term.x + drive_term.x + acc_t7_r;
    res.y = acc_par_i + term_iv.y + term_vp_diss.y + drift_term.y + hyper_term.y + drive_term.y + acc_t7_i;

    rhs_out[field_idx] = res;
}

// Dynamic fallback kernel
__global__ void linear_rhs_fused_dynamic_kernel(
    const double2* __restrict__ df,
    const double2* __restrict__ phi,
    const double*  __restrict__ bessel,
    const double*  __restrict__ s_total_upar,
    const double*  __restrict__ s_total_t7,
    const int2*    __restrict__ packed_maps,
    const double*  __restrict__ utrap,
    const double*  __restrict__ abs_dum2_vp,
    const double*  __restrict__ drift_x,
    const double*  __restrict__ drift_y,
    const double*  __restrict__ dmaxwel_fm_ek,
    const double*  __restrict__ fmaxwl,
    const double*  __restrict__ hyper,
    const double*  __restrict__ kx_vals,
    const double*  __restrict__ ky_vals,
    double2*       __restrict__ rhs_out,
    int nv, int nmu, int ns, int nkx, int nky, int nv_total,
    double c_d1_0, double c_d1_1, double c_d1_2, double c_d1_3, double c_d1_4,
    double c_d4_0, double c_d4_1, double c_d4_2, double c_d4_3, double c_d4_4,
    double dvp, double disp_vp, double drive_scale, double signz0, double tmp0
) {
    extern __shared__ double2 smem_total[];
    double2* smem_df = smem_total;
    double2* smem_gyro = &smem_total[ns * nky];

    const int local_tid = threadIdx.x;
    const int v_idx     = blockIdx.x / nkx;
    const int kx        = blockIdx.x % nkx;
    const int s         = local_tid / nky;
    const int ky        = local_tid % nky;

    const int mu_idx    = v_idx % nmu;
    const int v_phys    = v_idx / nmu;

    const size_t spatial_stride = (size_t)ns * nkx * nky;
    const size_t spatial_idx    = (size_t)s * (nkx * nky) + (size_t)kx * nky + ky;
    const size_t field_idx      = (size_t)v_idx * spatial_stride + spatial_idx;

    double2 my_df = __ldg(&df[field_idx]);
    smem_df[local_tid] = my_df;

    const size_t bessel_idx = (((size_t)mu_idx * ns + s) * nkx + kx) * nky + ky;
    double2 phi_val = __ldg(&phi[spatial_idx]);
    double bes = __ldg(&bessel[bessel_idx]);
    double2 my_gyro_phi = make_double2(bes * phi_val.x, bes * phi_val.y);
    smem_gyro[local_tid] = my_gyro_phi;

    __syncthreads();

    double acc_par_r = 0.0, acc_par_i = 0.0;
    double acc_t7_r  = 0.0, acc_t7_i  = 0.0;

    #pragma unroll
    for (int i = 0; i < 9; ++i) {
        const int2   map_val = __ldg(&packed_maps[(size_t)i * spatial_stride + spatial_idx]);
        const int    src_s   = map_val.x;
        if (src_s >= 0) {
            const int    src_kx = map_val.y;
            
            const size_t upar_stride = (size_t)nv * 1 * ns * nkx * nky;
            const size_t upar_idx = (size_t)i * upar_stride 
                                  + (((size_t)v_phys * 1 + 0) * ns + s) * (nkx * nky)
                                  + (size_t)kx * nky + ky;

            const size_t t7_stride = (size_t)nv * nmu * ns * nkx * nky;
            const size_t t7_idx = (size_t)i * t7_stride + field_idx;

            const double c_upar = __ldg(&s_total_upar[upar_idx]);
            const double c_t7   = __ldg(&s_total_t7[t7_idx]);
            
            double2 v_df, v_gyro;
            if (src_kx == kx) {
                v_df   = smem_df[src_s * nky + ky];
                v_gyro = smem_gyro[src_s * nky + ky];
            } else {
                const size_t src_field_idx = (size_t)v_idx * spatial_stride
                                           + (size_t)src_s * (nkx * nky)
                                           + (size_t)src_kx * nky + ky;
                const size_t src_spatial_idx = (size_t)src_s * (nkx * nky)
                                             + (size_t)src_kx * nky + ky;
                v_df = __ldg(&df[src_field_idx]);
                
                const size_t bessel_src_idx = (((size_t)mu_idx * ns + src_s) * nkx + src_kx) * nky + ky;
                double bes_src = __ldg(&bessel[bessel_src_idx]);
                double2 phi_src = __ldg(&phi[src_spatial_idx]);
                v_gyro = make_double2(bes_src * phi_src.x, bes_src * phi_src.y);
            }
            acc_par_r += v_df.x * c_upar;
            acc_par_i += v_df.y * c_upar;
            acc_t7_r  += v_gyro.x * c_t7;
            acc_t7_i  += v_gyro.y * c_t7;
        }
    }

    const size_t vpar_stride = (size_t)nmu * spatial_stride;

    double2 df_vm2 = (v_phys >= 2)      ? __ldg(&df[field_idx - 2 * vpar_stride]) : make_double2(0.0, 0.0);
    double2 df_vm1 = (v_phys >= 1)      ? __ldg(&df[field_idx - 1 * vpar_stride]) : make_double2(0.0, 0.0);
    double2 df_vp1 = (v_phys <= nv - 2) ? __ldg(&df[field_idx + 1 * vpar_stride]) : make_double2(0.0, 0.0);
    double2 df_vp2 = (v_phys <= nv - 3) ? __ldg(&df[field_idx + 2 * vpar_stride]) : make_double2(0.0, 0.0);

    double2 out_d1 = make_double2(
        c_d1_0 * df_vm2.x + c_d1_1 * df_vm1.x + c_d1_2 * my_df.x + c_d1_3 * df_vp1.x + c_d1_4 * df_vp2.x,
        c_d1_0 * df_vm2.y + c_d1_1 * df_vm1.y + c_d1_2 * my_df.y + c_d1_3 * df_vp1.y + c_d1_4 * df_vp2.y
    );
    double2 out_d4 = make_double2(
        c_d4_0 * df_vm2.x + c_d4_1 * df_vm1.x + c_d4_2 * my_df.x + c_d4_3 * df_vp1.x + c_d4_4 * df_vp2.x,
        c_d4_0 * df_vm2.y + c_d4_1 * df_vm1.y + c_d4_2 * my_df.y + c_d4_3 * df_vp1.y + c_d4_4 * df_vp2.y
    );

    const size_t mu_s_idx = (size_t)mu_idx * ns + s;
    double utrap_val   = __ldg(&utrap[mu_s_idx]);
    double abs_vp_val  = __ldg(&abs_dum2_vp[mu_s_idx]);
    
    const size_t v_mu_s_idx = ((size_t)v_phys * nmu + mu_idx) * ns + s;
    double drift_x_val = __ldg(&drift_x[v_mu_s_idx]);
    double drift_y_val = __ldg(&drift_y[v_mu_s_idx]);
    double fmaxwl_val  = __ldg(&fmaxwl[v_mu_s_idx]);

    const size_t dmaxwel_idx = (((size_t)v_phys * nmu + mu_idx) * ns + s) * nky + ky;
    double dmaxwel_val = __ldg(&dmaxwel_fm_ek[dmaxwel_idx]);
    
    double kx_val      = __ldg(&kx_vals[kx]);
    double ky_val      = __ldg(&ky_vals[ky]);
    double hyper_val   = __ldg(&hyper[spatial_idx]);

    double kdotvd  = drift_x_val * kx_val + drift_y_val * ky_val;
    double inv_dvp = 1.0 / dvp;
    double inv_tmp = 1.0 / fmax(tmp0, 1e-15);

    double2 term_iv = make_double2(utrap_val * out_d1.x * inv_dvp, utrap_val * out_d1.y * inv_dvp);
    double vp_diss_coeff = disp_vp * abs_vp_val * inv_dvp;
    double2 term_vp_diss = make_double2(vp_diss_coeff * out_d4.x, vp_diss_coeff * out_d4.y);
    double2 drift_term = make_double2(kdotvd * my_df.y, -kdotvd * my_df.x);
    double2 hyper_term = make_double2(hyper_val * my_df.x, hyper_val * my_df.y);
    double drive_tot_coeff = drive_scale * (dmaxwel_val - signz0 * kdotvd * fmaxwl_val * inv_tmp);
    double2 drive_term = make_double2(-drive_tot_coeff * my_gyro_phi.y, drive_tot_coeff * my_gyro_phi.x);

    double2 res;
    res.x = acc_par_r + term_iv.x + term_vp_diss.x + drift_term.x + hyper_term.x + drive_term.x + acc_t7_r;
    res.y = acc_par_i + term_iv.y + term_vp_diss.y + drift_term.y + hyper_term.y + drive_term.y + acc_t7_i;
    rhs_out[field_idx] = res;
}


// ── FFI Implementation ──────────────────────────────────────────────────────

#define DISPATCH_CASE(NS_VAL, NKY_VAL)                                                   \
    case (((NS_VAL) << 16) | (NKY_VAL)):                                                 \
        linear_rhs_fused_kernel<NS_VAL, NKY_VAL>                                         \
            <<<num_blocks, (NS_VAL) * (NKY_VAL), 0, stream>>>(                           \
                (const double2*)df.typed_data(), (const double2*)phi.typed_data(),       \
                bessel.typed_data(), s_total_upar.typed_data(),                          \
                s_total_t7.typed_data(), (const int2*)packed_maps.typed_data(),          \
                utrap.typed_data(), abs_dum2_vp.typed_data(),                            \
                drift_x.typed_data(), drift_y.typed_data(),                              \
                dmaxwel_fm_ek.typed_data(), fmaxwl.typed_data(),                         \
                hyper.typed_data(), kx_vals.typed_data(), ky_vals.typed_data(),          \
                (double2*)rhs_out->typed_data(),                                         \
                nv, nmu, nkx, nky, nv_nmu,                                               \
                c_d1_0, c_d1_1, c_d1_2, c_d1_3, c_d1_4,                                  \
                c_d4_0, c_d4_1, c_d4_2, c_d4_3, c_d4_4,                                  \
                dvp, disp_vp, drive_scale, signz0, tmp0);                                \
        break;

xla_ffi::Error LinearRhsFusedImpl(
    cudaStream_t stream,
    xla_ffi::Buffer<xla_ffi::DataType::C128> df,
    xla_ffi::Buffer<xla_ffi::DataType::C128> phi,
    xla_ffi::Buffer<xla_ffi::DataType::F64>  bessel,
    xla_ffi::Buffer<xla_ffi::DataType::F64>  s_total_upar,
    xla_ffi::Buffer<xla_ffi::DataType::F64>  s_total_t7,
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
    xla_ffi::Result<xla_ffi::Buffer<xla_ffi::DataType::C128>> rhs_out,
    int32_t nv, int32_t nmu, int32_t ns, int32_t nkx, int32_t nky, int32_t nv_nmu,
    double c_d1_0, double c_d1_1, double c_d1_2, double c_d1_3, double c_d1_4,
    double c_d4_0, double c_d4_1, double c_d4_2, double c_d4_3, double c_d4_4,
    double dvp, double disp_vp, double drive_scale, double signz0, double tmp0
) {
    const int num_blocks = nv_nmu * nkx;

    switch ((ns << 16) | nky) {
        DISPATCH_CASE(16, 32)
        DISPATCH_CASE(32, 32)
        DISPATCH_CASE(16, 64)
        default: {
            const int threads = ns * nky;
            if (threads > 1024)
                return xla_ffi::Error(XLA_FFI_Error_Code_INVALID_ARGUMENT,
                    "ns * nky exceeds maximum CUDA block size of 1024");
            linear_rhs_fused_dynamic_kernel
                <<<num_blocks, threads, 2 * (size_t)threads * sizeof(double2), stream>>>(
                    (const double2*)df.typed_data(), (const double2*)phi.typed_data(),
                    bessel.typed_data(), s_total_upar.typed_data(),
                    s_total_t7.typed_data(), (const int2*)packed_maps.typed_data(),
                    utrap.typed_data(), abs_dum2_vp.typed_data(),
                    drift_x.typed_data(), drift_y.typed_data(),
                    dmaxwel_fm_ek.typed_data(), fmaxwl.typed_data(),
                    hyper.typed_data(), kx_vals.typed_data(), ky_vals.typed_data(),
                    (double2*)rhs_out->typed_data(),
                    nv, nmu, ns, nkx, nky, nv_nmu,
                    c_d1_0, c_d1_1, c_d1_2, c_d1_3, c_d1_4,
                    c_d4_0, c_d4_1, c_d4_2, c_d4_3, c_d4_4,
                    dvp, disp_vp, drive_scale, signz0, tmp0);
        }
    }

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
        return xla_ffi::Error(XLA_FFI_Error_Code_INTERNAL, cudaGetErrorString(err));

    return xla_ffi::Error::Success();
}

#undef DISPATCH_CASE

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    linear_rhs_fused_ffi, LinearRhsFusedImpl,
    xla_ffi::Ffi::Bind()
        .Ctx<xla_ffi::PlatformStream<cudaStream_t>>()
        .Arg<xla_ffi::Buffer<xla_ffi::DataType::C128>>() // df
        .Arg<xla_ffi::Buffer<xla_ffi::DataType::C128>>() // phi
        .Arg<xla_ffi::Buffer<xla_ffi::DataType::F64>>()  // bessel
        .Arg<xla_ffi::Buffer<xla_ffi::DataType::F64>>()  // s_total_upar
        .Arg<xla_ffi::Buffer<xla_ffi::DataType::F64>>()  // s_total_t7
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
        .Ret<xla_ffi::Buffer<xla_ffi::DataType::C128>>() // rhs_out
        .Attr<int32_t>("nv")
        .Attr<int32_t>("nmu")
        .Attr<int32_t>("ns")
        .Attr<int32_t>("nkx")
        .Attr<int32_t>("nky")
        .Attr<int32_t>("nv_nmu")
        .Attr<double>("c_d1_0").Attr<double>("c_d1_1").Attr<double>("c_d1_2")
        .Attr<double>("c_d1_3").Attr<double>("c_d1_4")
        .Attr<double>("c_d4_0").Attr<double>("c_d4_1").Attr<double>("c_d4_2")
        .Attr<double>("c_d4_3").Attr<double>("c_d4_4")
        .Attr<double>("dvp")
        .Attr<double>("disp_vp")
        .Attr<double>("drive_scale")
        .Attr<double>("signz0")
        .Attr<double>("tmp0")
);
