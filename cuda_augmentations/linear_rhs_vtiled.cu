#include "xla/ffi/api/ffi.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

namespace xla_ffi = xla::ffi;

// ── V-Tiled Layout Design (Phase 8 - Final) ──────────────────────────────────
// blockDim: 256 (fixed) — fits up to 255 registers/thread.
// Adjacent threads have contiguous ky for perfect coalescing (as long as blockDim % NKY == 0).
// Each block handles blockDim spatial points.
// ────────────────────────────────────────────────────────────────────────────

__global__ void __launch_bounds__(256)
linear_rhs_vtiled_kernel(
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
    int nv, int nmu, int ns, int nkx, int nky, int nv_nmu, int v_tile,
    double c_d1_0, double c_d1_1, double c_d1_2, double c_d1_3, double c_d1_4,
    double c_d4_0, double c_d4_1, double c_d4_2, double c_d4_3, double c_d4_4,
    double dvp, double disp_vp, double drive_scale, double signz0, double tmp0
) {
    extern __shared__ double2 smem_df[]; 

    const int tid         = threadIdx.x;
    const int tile_v_idx  = blockIdx.x % (nv_nmu / v_tile);
    const int tile_s_idx  = blockIdx.x / (nv_nmu / v_tile); // spatial block linear index
    
    // Each block processes 256 spatial points.
    // Linear spatial index within the block: tid
    // Global spatial index: tile_s_idx * 256 + tid
    const size_t spatial_idx_global = (size_t)tile_s_idx * 256 + tid;
    const size_t n_spatial = (size_t)ns * nkx * nky;
    if (spatial_idx_global >= n_spatial) return;

    // Decode (s, kx, ky)
    const int s           = spatial_idx_global / (nkx * nky);
    const int rem         = spatial_idx_global % (nkx * nky);
    const int kx          = rem / nky;
    const int ky          = rem % nky;
    const int v_base      = tile_v_idx * v_tile;
    const size_t spatial_stride = (size_t)ns * nkx * nky;

        #pragma unroll
        for (int vv = 0; vv < 8; vv++) { // Fixed at 8 for final
            const int v_idx = v_base + vv;
            if (v_idx < nv_nmu && spatial_idx_global < spatial_stride) {
                smem_df[vv * 256 + tid] = __ldg(&df[(size_t)v_idx * spatial_stride + spatial_idx_global]);
            } else {
                smem_df[vv * 256 + tid] = {0.0, 0.0};
            }
        }
    __syncthreads();

    const double2 phi_val = __ldg(&phi[spatial_idx_global]);
    const double  kx_val  = __ldg(&kx_vals[kx]);
    const double  ky_val  = __ldg(&ky_vals[ky]);
    const double  hyp_val = __ldg(&hyper[spatial_idx_global]);
    const double  inv_dvp = 1.0 / dvp, inv_tmp = 1.0 / fmax(tmp0, 1e-15);

    int src_s_arr[9], src_kx_arr[9]; bool valid_arr[9]; size_t src_spatial_arr[9];
    #pragma unroll
    for (int i = 0; i < 9; i++) {
        int2 m = __ldg(&packed_maps[(size_t)i * spatial_stride + spatial_idx_global]);
        src_s_arr[i] = m.x; src_kx_arr[i] = m.y; valid_arr[i] = (m.x >= 0);
        if (valid_arr[i]) src_spatial_arr[i] = ((size_t)m.x * nkx + m.y) * nky + ky;
    }

    const int v_for_upar = v_base / nmu;
    double c_upar_arr[9]; double2 phi_src_arr[9];
    #pragma unroll
    for (int i = 0; i < 9; i++) {
        if (valid_arr[i]) {
            c_upar_arr[i] = __ldg(&s_total_upar[(size_t)i * ((size_t)nv * spatial_stride) + (size_t)v_for_upar * spatial_stride + spatial_idx_global]);
            phi_src_arr[i] = __ldg(&phi[src_spatial_arr[i]]);
        }
    }

    #pragma unroll
    for (int vv = 0; vv < 8; vv++) {
        const int v_idx_global = v_base + vv;
        if (v_idx_global >= nv_nmu) break;

        const int v    = v_idx_global / nmu;
        const int mu   = v_idx_global % nmu;
        const size_t field_idx = (size_t)v_idx_global * spatial_stride + spatial_idx_global;

        double2 my_df = smem_df[vv * 256 + tid];
        const size_t bes_spatial = (((size_t)mu * ns + s) * nkx + kx) * nky + ky;
        const double bes_val = __ldg(&bessel[bes_spatial]);
        double2 my_gyro_phi = make_double2(bes_val * phi_val.x, bes_val * phi_val.y);

        double acc_par_r = 0.0, acc_par_i = 0.0, acc_t7_r = 0.0, acc_t7_i = 0.0;
        #pragma unroll
        for (int i = 0; i < 9; i++) {
            if (valid_arr[i]) {
                const double c_upar = c_upar_arr[i], c_t7 = __ldg(&s_total_t7[(size_t)i * ((size_t)nv_nmu * spatial_stride) + field_idx]);
                double2 v_df;
                if (src_kx_arr[i] == kx) {
                    // If we index by spatial_idx_global, we can check if it's within [tile_s_idx*256, (tile_s_idx+1)*256)
                    const size_t src_idx = src_spatial_arr[i];
                    if (src_idx >= (size_t)tile_s_idx * 256 && src_idx < (size_t)tile_s_idx * 256 + 256) {
                        v_df = smem_df[vv * 256 + (src_idx - (size_t)tile_s_idx * 256)];
                    } else {
                        v_df = __ldg(&df[(size_t)v_idx_global * spatial_stride + src_idx]);
                    }
                } else {
                    v_df = __ldg(&df[(size_t)v_idx_global * spatial_stride + src_spatial_arr[i]]);
                }
                const double bes_src = __ldg(&bessel[(((size_t)mu * ns + src_s_arr[i]) * nkx + src_kx_arr[i]) * nky + ky]);
                acc_par_r += v_df.x * c_upar; acc_par_i += v_df.y * c_upar;
                acc_t7_r  += bes_src * phi_src_arr[i].x * c_t7; acc_t7_i  += bes_src * phi_src_arr[i].y * c_t7;
            }
        }

        const size_t vs = (size_t)nmu * spatial_stride;
        double2 df_vm2 = (v >= 2) ? __ldg(&df[field_idx - 2*vs]) : make_double2(0,0);
        double2 df_vm1 = (v >= 1) ? __ldg(&df[field_idx - 1*vs]) : make_double2(0,0);
        double2 df_vp1 = (v <= nv - 2) ? __ldg(&df[field_idx + 1*vs]) : make_double2(0,0);
        double2 df_vp2 = (v <= nv - 3) ? __ldg(&df[field_idx + 2*vs]) : make_double2(0,0);

        double2 d1 = make_double2(c_d1_0*df_vm2.x+c_d1_1*df_vm1.x+c_d1_2*my_df.x+c_d1_3*df_vp1.x+c_d1_4*df_vp2.x, c_d1_0*df_vm2.y+c_d1_1*df_vm1.y+c_d1_2*my_df.y+c_d1_3*df_vp1.y+c_d1_4*df_vp2.y);
        double2 d4 = make_double2(c_d4_0*df_vm2.x+c_d4_1*df_vm1.x+c_d4_2*my_df.x+c_d4_3*df_vp1.x+c_d4_4*df_vp2.x, c_d4_0*df_vm2.y+c_d4_1*df_vm1.y+c_d4_2*my_df.y+c_d4_3*df_vp1.y+c_d4_4*df_vp2.y);

        const size_t m_s = (size_t)mu * ns + s, min_3d = ((size_t)v * nmu + mu) * ns + s;
        const double kdotvd = drift_x[min_3d] * kx_val + drift_y[min_3d] * ky_val;
        const double drive_c = drive_scale * (__ldg(&dmaxwel_fm_ek[min_3d * nky + ky]) - signz0 * kdotvd * fmaxwl[min_3d] * inv_tmp);
        rhs_out[field_idx].x = acc_par_r + utrap[m_s]*d1.x*inv_dvp + disp_vp*abs_dum2_vp[m_s]*inv_dvp*d4.x + kdotvd*my_df.y + hyp_val*my_df.x - drive_c*my_gyro_phi.y + acc_t7_r;
        rhs_out[field_idx].y = acc_par_i + utrap[m_s]*d1.y*inv_dvp + disp_vp*abs_dum2_vp[m_s]*inv_dvp*d4.y - kdotvd*my_df.x + hyp_val*my_df.y + drive_c*my_gyro_phi.x + acc_t7_i;
    }
}

xla_ffi::Error LinearRhsVtiledImpl(
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
    int32_t nv, int32_t nmu, int32_t ns, int32_t nkx, int32_t nky, int32_t nv_nmu, int32_t v_tile,
    double c_d1_0, double c_d1_1, double c_d1_2, double c_d1_3, double c_d1_4,
    double c_d4_0, double c_d4_1, double c_d4_2, double c_d4_3, double c_d4_4,
    double dvp, double disp_vp, double drive_scale, double signz0, double tmp0
) {
    const size_t n_spatial = (size_t)ns * nkx * nky;
    dim3 block(256);
    dim3 grid((nv_nmu / v_tile), (n_spatial + 255) / 256);
    // Flatten grid for simplicity
    dim3 grid_flat(grid.x * grid.y);
    
    // Shared memory: v_tile * 256 * sizeof(double2)
    const size_t smem_bytes = (size_t)v_tile * 256 * sizeof(double2);
    
    linear_rhs_vtiled_kernel<<<grid_flat, block, smem_bytes, stream>>>(
        (const double2*)df.typed_data(), (const double2*)phi.typed_data(),
        bessel.typed_data(), s_total_upar.typed_data(),
        s_total_t7.typed_data(), (const int2*)packed_maps.typed_data(),
        utrap.typed_data(), abs_dum2_vp.typed_data(),
        drift_x.typed_data(), drift_y.typed_data(),
        dmaxwel_fm_ek.typed_data(), fmaxwl.typed_data(),
        hyper.typed_data(), kx_vals.typed_data(), ky_vals.typed_data(),
        (double2*)rhs_out->typed_data(),
        nv, nmu, ns, nkx, nky, nv_nmu, v_tile,
        c_d1_0, c_d1_1, c_d1_2, c_d1_3, c_d1_4,
        c_d4_0, c_d4_1, c_d4_2, c_d4_3, c_d4_4,
        dvp, disp_vp, drive_scale, signz0, tmp0);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
        return xla_ffi::Error(XLA_FFI_Error_Code_INTERNAL, cudaGetErrorString(err));

    return xla_ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    linear_rhs_vtiled_ffi, LinearRhsVtiledImpl,
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
        .Attr<int32_t>("v_tile")
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
