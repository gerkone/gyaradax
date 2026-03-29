#include "xla/ffi/api/ffi.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

namespace xla_ffi = xla::ffi;

// One block per (v_idx, kx) pair; one thread per (s, ky) element.
// packed_maps: int2[9][nv_nmu][ns][nkx][nky]  — .x = src_s (-1 = inactive), .y = src_kx
// coeffs:      double[9][nv_raw][ns][nkx][nky] — nv_raw = nv_nmu / nmu

template <int NS, int NKY>
__global__ __launch_bounds__(NS * NKY)
void apply_parallel_kernel(
    const double2* __restrict__ field,
    const double*  __restrict__ coeffs,
    const int2*    __restrict__ packed_maps,
    double2*       __restrict__ out,
    int nv_nmu, int nkx, int nmu
) {
    __shared__ double2 smem[NS * NKY];

    const int local_tid = threadIdx.x;
    const int v_idx     = blockIdx.x / nkx;
    const int kx        = blockIdx.x % nkx;
    const int s         = local_tid / NKY;
    const int ky        = local_tid % NKY;

    const size_t spatial_stride = (size_t)NS * nkx * NKY;
    const size_t spatial_idx    = (size_t)s * (nkx * NKY) + (size_t)kx * NKY + ky;
    const size_t field_idx      = (size_t)v_idx * spatial_stride + spatial_idx;

    // All threads are valid: block size == NS * NKY, so s < NS and ky < NKY always hold.
    smem[local_tid] = __ldg(&field[field_idx]);
    __syncthreads();

    const int    nv_raw     = nv_nmu / nmu;
    const size_t c_idx_base = (size_t)(v_idx / nmu) * spatial_stride + spatial_idx;
    const size_t c_i_stride = (size_t)nv_raw * spatial_stride;

    double acc_r = 0.0, acc_i = 0.0;

    #pragma unroll
    for (int i = 0; i < 9; ++i) {
        const int2   map_val = __ldg(&packed_maps[(size_t)i * spatial_stride + spatial_idx]);
        const int    src_s   = map_val.x;
        if (src_s >= 0) {
            const int    src_kx = map_val.y;
            const double c      = __ldg(&coeffs[(size_t)i * c_i_stride + c_idx_base]);
            double2 val;
            if (src_kx == kx) {
                val = smem[src_s * NKY + ky];
            } else {
                val = __ldg(&field[(size_t)v_idx * spatial_stride
                                   + (size_t)src_s * (nkx * NKY)
                                   + (size_t)src_kx * NKY + ky]);
            }
            acc_r += val.x * c;
            acc_i += val.y * c;
        }
    }
    out[field_idx] = make_double2(acc_r, acc_i);
}

__global__ void apply_parallel_dynamic_kernel(
    const double2* __restrict__ field,
    const double*  __restrict__ coeffs,
    const int2*    __restrict__ packed_maps,
    double2*       __restrict__ out,
    int nv_nmu, int nkx, int ns, int nky, int nmu
) {
    extern __shared__ double2 smem[];

    const int local_tid = threadIdx.x;
    const int v_idx     = blockIdx.x / nkx;
    const int kx        = blockIdx.x % nkx;
    const int s         = local_tid / nky;
    const int ky        = local_tid % nky;

    const size_t spatial_stride = (size_t)ns * nkx * nky;
    const size_t spatial_idx    = (size_t)s * (nkx * nky) + (size_t)kx * nky + ky;
    const size_t field_idx      = (size_t)v_idx * spatial_stride + spatial_idx;

    smem[local_tid] = __ldg(&field[field_idx]);
    __syncthreads();

    const int    nv_raw     = nv_nmu / nmu;
    const size_t c_idx_base = (size_t)(v_idx / nmu) * spatial_stride + spatial_idx;
    const size_t c_i_stride = (size_t)nv_raw * spatial_stride;

    double acc_r = 0.0, acc_i = 0.0;

    #pragma unroll
    for (int i = 0; i < 9; ++i) {
        const int2   map_val = __ldg(&packed_maps[(size_t)i * spatial_stride + spatial_idx]);
        const int    src_s   = map_val.x;
        if (src_s >= 0) {
            const int    src_kx = map_val.y;
            const double c      = __ldg(&coeffs[(size_t)i * c_i_stride + c_idx_base]);
            double2 val;
            if (src_kx == kx) {
                val = smem[src_s * nky + ky];
            } else {
                val = __ldg(&field[(size_t)v_idx * spatial_stride
                                   + (size_t)src_s * (nkx * nky)
                                   + (size_t)src_kx * nky + ky]);
            }
            acc_r += val.x * c;
            acc_i += val.y * c;
        }
    }
    out[field_idx] = make_double2(acc_r, acc_i);
}

// ── Dispatch ────────────────────────────────────────────────────────────────

#define DISPATCH_CASE(NS_VAL, NKY_VAL)                                              \
    case (((NS_VAL) << 16) | (NKY_VAL)):                                            \
        apply_parallel_kernel<NS_VAL, NKY_VAL>                                      \
            <<<num_blocks, (NS_VAL) * (NKY_VAL), 0, stream>>>(                      \
                (const double2*)field.typed_data(), coeffs.typed_data(),             \
                (const int2*)packed_maps.typed_data(), (double2*)out->typed_data(), \
                nv_nmu, nkx, nmu);                                                  \
        break;

xla_ffi::Error ApplyParallelImpl(
    cudaStream_t stream,
    xla_ffi::Buffer<xla_ffi::DataType::C128> field,
    xla_ffi::Buffer<xla_ffi::DataType::F64>  coeffs,
    xla_ffi::Buffer<xla_ffi::DataType::S32>  packed_maps,  // int2 pairs: .x=src_s, .y=src_kx
    xla_ffi::Result<xla_ffi::Buffer<xla_ffi::DataType::C128>> out,
    int32_t nv_nmu, int32_t nkx, int32_t ns, int32_t nky, int32_t nmu
) {
    const int num_blocks = nv_nmu * nkx;

    switch ((ns << 16) | nky) {
        DISPATCH_CASE(16, 32)
        DISPATCH_CASE(32, 32)
        DISPATCH_CASE(16, 64)
        DISPATCH_CASE(32, 64)
        default: {
            const int threads = ns * nky;
            if (threads > 1024)
                return xla_ffi::Error(XLA_FFI_Error_Code_INVALID_ARGUMENT,
                    "ns * nky exceeds maximum CUDA block size of 1024");
            apply_parallel_dynamic_kernel
                <<<num_blocks, threads, (size_t)threads * sizeof(double2), stream>>>(
                    (const double2*)field.typed_data(), coeffs.typed_data(),
                    (const int2*)packed_maps.typed_data(), (double2*)out->typed_data(),
                    nv_nmu, nkx, ns, nky, nmu);
        }
    }

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
        return xla_ffi::Error(XLA_FFI_Error_Code_INTERNAL, cudaGetErrorString(err));

    return xla_ffi::Error::Success();
}

#undef DISPATCH_CASE

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    apply_parallel_ffi, ApplyParallelImpl,
    xla_ffi::Ffi::Bind()
        .Ctx<xla_ffi::PlatformStream<cudaStream_t>>()
        .Arg<xla_ffi::Buffer<xla_ffi::DataType::C128>>() // field
        .Arg<xla_ffi::Buffer<xla_ffi::DataType::F64>>()  // coeffs
        .Arg<xla_ffi::Buffer<xla_ffi::DataType::S32>>()  // packed_maps (int2 pairs)
        .Ret<xla_ffi::Buffer<xla_ffi::DataType::C128>>() // out
        .Attr<int32_t>("nv_nmu")
        .Attr<int32_t>("nkx")
        .Attr<int32_t>("ns")
        .Attr<int32_t>("nky")
        .Attr<int32_t>("nmu")
);
