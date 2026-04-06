#include "xla/ffi/api/ffi.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

namespace xla_ffi = xla::ffi;

// One block per (v_idx, kx) pair; one thread per (s, ky) element.
// packed_maps: int2[9][ns][nkx][nky]  — .x = src_s (-1 = inactive), .y = src_kx
// coeffs:      double[9][nv_raw][ns][nkx][nky] — nv_raw = nv_nmu / nmu

template <int NS, int NKY>
__global__ __launch_bounds__(NS * NKY)
void apply_parallel_dual_kernel(
    const double2* __restrict__ field1, // e.g. df
    const double2* __restrict__ field2, // e.g. gyro_phi
    const double*  __restrict__ coeffs1, // weight for field1
    const double*  __restrict__ coeffs2, // weight for field2
    const int2*    __restrict__ packed_maps,
    double2*       __restrict__ out1,
    double2*       __restrict__ out2,
    int nv_nmu, int nkx, int nmu
) {
    __shared__ double2 smem1[NS * NKY];
    __shared__ double2 smem2[NS * NKY];

    const int local_tid = threadIdx.x;
    const int v_idx     = blockIdx.x / nkx;
    const int kx        = blockIdx.x % nkx;
    const int s         = local_tid / NKY;
    const int ky        = local_tid % NKY;

    const size_t spatial_stride = (size_t)NS * nkx * NKY;
    const size_t spatial_idx    = (size_t)s * (nkx * NKY) + (size_t)kx * NKY + ky;
    const size_t field_idx      = (size_t)v_idx * spatial_stride + spatial_idx;

    smem1[local_tid] = __ldg(&field1[field_idx]);
    smem2[local_tid] = __ldg(&field2[field_idx]);
    __syncthreads();

    const size_t c_idx_base   = (size_t)v_idx * spatial_stride + spatial_idx;
    const size_t c_i_stride   = (size_t)nv_nmu * spatial_stride;

    double acc1_r = 0.0, acc1_i = 0.0;
    double acc2_r = 0.0, acc2_i = 0.0;

    #pragma unroll
    for (int i = 0; i < 9; ++i) {
        const int2   map_val = __ldg(&packed_maps[(size_t)i * spatial_stride + spatial_idx]);
        const int    src_s   = map_val.x;
        if (src_s >= 0) {
            const int    src_kx = map_val.y;
            const double c1     = __ldg(&coeffs1[(size_t)i * c_i_stride + c_idx_base]);
            const double c2     = __ldg(&coeffs2[(size_t)i * c_i_stride + c_idx_base]);
            
            double2 v1, v2;
            if (src_kx == kx) {
                v1 = smem1[src_s * NKY + ky];
                v2 = smem2[src_s * NKY + ky];
            } else {
                const size_t src_idx = (size_t)v_idx * spatial_stride
                                     + (size_t)src_s * (nkx * NKY)
                                     + (size_t)src_kx * NKY + ky;
                v1 = __ldg(&field1[src_idx]);
                v2 = __ldg(&field2[src_idx]);
            }
            acc1_r += v1.x * c1;
            acc1_i += v1.y * c1;
            acc2_r += v2.x * c2;
            acc2_i += v2.y * c2;
        }
    }
    out1[field_idx] = make_double2(acc1_r, acc1_i);
    out2[field_idx] = make_double2(acc2_r, acc2_i);
}

__global__ void apply_parallel_dual_dynamic_kernel(
    const double2* __restrict__ field1,
    const double2* __restrict__ field2,
    const double*  __restrict__ coeffs1,
    const double*  __restrict__ coeffs2,
    const int2*    __restrict__ packed_maps,
    double2*       __restrict__ out1,
    double2*       __restrict__ out2,
    int nv_nmu, int nkx, int ns, int nky, int nmu
) {
    extern __shared__ double2 smem_total[];
    double2* smem1 = smem_total;
    double2* smem2 = &smem_total[ns * nky];

    const int local_tid = threadIdx.x;
    const int v_idx     = blockIdx.x / nkx;
    const int kx        = blockIdx.x % nkx;
    const int s         = local_tid / nky;
    const int ky        = local_tid % nky;

    const size_t spatial_stride = (size_t)ns * nkx * nky;
    const size_t spatial_idx    = (size_t)s * (nkx * nky) + (size_t)kx * nky + ky;
    const size_t field_idx      = (size_t)v_idx * spatial_stride + spatial_idx;

    smem1[local_tid] = __ldg(&field1[field_idx]);
    smem2[local_tid] = __ldg(&field2[field_idx]);
    __syncthreads();

    const size_t c_idx_base   = (size_t)v_idx * spatial_stride + spatial_idx;
    const size_t c_i_stride   = (size_t)nv_nmu * spatial_stride;

    double acc1_r = 0.0, acc1_i = 0.0;
    double acc2_r = 0.0, acc2_i = 0.0;

    #pragma unroll
    for (int i = 0; i < 9; ++i) {
        const int2   map_val = __ldg(&packed_maps[(size_t)i * spatial_stride + spatial_idx]);
        const int    src_s   = map_val.x;
        if (src_s >= 0) {
            const int    src_kx = map_val.y;
            const double c1     = __ldg(&coeffs1[(size_t)i * c_i_stride + c_idx_base]);
            const double c2     = __ldg(&coeffs2[(size_t)i * c_i_stride + c_idx_base]);
            
            double2 v1, v2;
            if (src_kx == kx) {
                v1 = smem1[src_s * nky + ky];
                v2 = smem2[src_s * nky + ky];
            } else {
                const size_t src_idx = (size_t)v_idx * spatial_stride
                                     + (size_t)src_s * (nkx * nky)
                                     + (size_t)src_kx * nky + ky;
                v1 = __ldg(&field1[src_idx]);
                v2 = __ldg(&field2[src_idx]);
            }
            acc1_r += v1.x * c1;
            acc1_i += v1.y * c1;
            acc2_r += v2.x * c2;
            acc2_i += v2.y * c2;
        }
    }
    out1[field_idx] = make_double2(acc1_r, acc1_i);
    out2[field_idx] = make_double2(acc2_r, acc2_i);
}

// ── Dispatch ────────────────────────────────────────────────────────────────

#define DISPATCH_CASE(NS_VAL, NKY_VAL)                                                   \
    case (((NS_VAL) << 16) | (NKY_VAL)):                                                 \
        apply_parallel_dual_kernel<NS_VAL, NKY_VAL>                                      \
            <<<num_blocks, (NS_VAL) * (NKY_VAL), 0, stream>>>(                           \
                (const double2*)field1.typed_data(), (const double2*)field2.typed_data(), \
                coeffs1.typed_data(), coeffs2.typed_data(),                              \
                (const int2*)packed_maps.typed_data(),                                   \
                (double2*)out1->typed_data(), (double2*)out2->typed_data(),              \
                nv_nmu, nkx, nmu);                                                       \
        break;

xla_ffi::Error ApplyParallelDualImpl(
    cudaStream_t stream,
    xla_ffi::Buffer<xla_ffi::DataType::C128> field1,
    xla_ffi::Buffer<xla_ffi::DataType::C128> field2,
    xla_ffi::Buffer<xla_ffi::DataType::F64>  coeffs1,
    xla_ffi::Buffer<xla_ffi::DataType::F64>  coeffs2,
    xla_ffi::Buffer<xla_ffi::DataType::S32>  packed_maps,
    xla_ffi::Result<xla_ffi::Buffer<xla_ffi::DataType::C128>> out1,
    xla_ffi::Result<xla_ffi::Buffer<xla_ffi::DataType::C128>> out2,
    int32_t nv_nmu, int32_t nkx, int32_t ns, int32_t nky, int32_t nmu
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
            // Shared memory: 2 * threads * sizeof(double2)
            apply_parallel_dual_dynamic_kernel
                <<<num_blocks, threads, 2 * (size_t)threads * sizeof(double2), stream>>>(
                    (const double2*)field1.typed_data(), (const double2*)field2.typed_data(),
                    coeffs1.typed_data(), coeffs2.typed_data(),
                    (const int2*)packed_maps.typed_data(),
                    (double2*)out1->typed_data(), (double2*)out2->typed_data(),
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
    apply_parallel_dual_ffi, ApplyParallelDualImpl,
    xla_ffi::Ffi::Bind()
        .Ctx<xla_ffi::PlatformStream<cudaStream_t>>()
        .Arg<xla_ffi::Buffer<xla_ffi::DataType::C128>>() // field1
        .Arg<xla_ffi::Buffer<xla_ffi::DataType::C128>>() // field2
        .Arg<xla_ffi::Buffer<xla_ffi::DataType::F64>>()  // coeffs1
        .Arg<xla_ffi::Buffer<xla_ffi::DataType::F64>>()  // coeffs2
        .Arg<xla_ffi::Buffer<xla_ffi::DataType::S32>>()  // packed_maps
        .Ret<xla_ffi::Buffer<xla_ffi::DataType::C128>>() // out1
        .Ret<xla_ffi::Buffer<xla_ffi::DataType::C128>>() // out2
        .Attr<int32_t>("nv_nmu")
        .Attr<int32_t>("nkx")
        .Attr<int32_t>("ns")
        .Attr<int32_t>("nky")
        .Attr<int32_t>("nmu")
);
