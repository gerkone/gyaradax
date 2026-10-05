#include "xla/ffi/api/ffi.h"
#include <cuda_runtime.h>

namespace xla_ffi = xla::ffi;

// partial field moments sum_r w_k[r, x] * f[r, x] over a chunk of (sp, v, mu) rows, f = g + g2f*apar if G2F

template <int NW, bool G2F>
__global__ __launch_bounds__(128)
void field_moments_kernel(
    const double2* __restrict__ g,        // (n_rows, n_spatial)
    const double*  __restrict__ w0,       // (n_rows, n_spatial)
    const double*  __restrict__ w1,       // (n_rows, n_spatial), NW == 2
    const double*  __restrict__ g2f,      // (n_rows, n_spatial), G2F
    const double2* __restrict__ apar,     // (n_spatial,), G2F
    double2*       __restrict__ partial,  // (n_chunks, NW, n_spatial)
    int n_rows, int n_spatial, int rows_per_chunk
) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    if (x >= n_spatial) return;
    const int chunk = blockIdx.y;
    const int r0 = chunk * rows_per_chunk;
    const int r1 = min(r0 + rows_per_chunk, n_rows);
    const double2 a = G2F ? __ldg(&apar[x]) : make_double2(0.0, 0.0);

    double acc0_r = 0.0, acc0_i = 0.0, acc1_r = 0.0, acc1_i = 0.0;
    for (int r = r0; r < r1; ++r) {
        const size_t idx = (size_t)r * n_spatial + x;
        double2 f = __ldg(&g[idx]);
        if (G2F) {
            const double c = __ldg(&g2f[idx]);
            f.x += c * a.x;
            f.y += c * a.y;
        }
        const double c0 = __ldg(&w0[idx]);
        acc0_r += c0 * f.x;
        acc0_i += c0 * f.y;
        if (NW > 1) {
            const double c1 = __ldg(&w1[idx]);
            acc1_r += c1 * f.x;
            acc1_i += c1 * f.y;
        }
    }
    partial[((size_t)chunk * NW) * n_spatial + x] = make_double2(acc0_r, acc0_i);
    if (NW > 1) partial[((size_t)chunk * NW + 1) * n_spatial + x] = make_double2(acc1_r, acc1_i);
}


// ── FFI Implementation ──────────────────────────────────────────────────────

xla_ffi::Error FieldMomentsImpl(
    cudaStream_t stream,
    xla_ffi::Buffer<xla_ffi::DataType::C128> g,
    xla_ffi::Buffer<xla_ffi::DataType::F64>  w0,
    xla_ffi::Buffer<xla_ffi::DataType::F64>  w1,
    xla_ffi::Buffer<xla_ffi::DataType::F64>  g2f,
    xla_ffi::Buffer<xla_ffi::DataType::C128> apar,
    xla_ffi::Result<xla_ffi::Buffer<xla_ffi::DataType::C128>> partial,
    int32_t n_rows, int32_t n_spatial, int32_t n_chunks, int32_t n_w, int32_t use_g2f
) {
    const int threads = 128;
    const int rows_per_chunk = (n_rows + n_chunks - 1) / n_chunks;
    dim3 grid((n_spatial + threads - 1) / threads, n_chunks);

    const double2* g_p    = (const double2*)g.typed_data();
    const double2* apar_p = (const double2*)apar.typed_data();
    double2* out = (double2*)partial->typed_data();
    const int variant = (n_w == 2 ? 2 : 0) | (use_g2f ? 1 : 0);
    switch (variant) {
        case 0: field_moments_kernel<1, false><<<grid, threads, 0, stream>>>(
            g_p, w0.typed_data(), w1.typed_data(), g2f.typed_data(), apar_p, out, n_rows, n_spatial, rows_per_chunk); break;
        case 1: field_moments_kernel<1, true><<<grid, threads, 0, stream>>>(
            g_p, w0.typed_data(), w1.typed_data(), g2f.typed_data(), apar_p, out, n_rows, n_spatial, rows_per_chunk); break;
        case 2: field_moments_kernel<2, false><<<grid, threads, 0, stream>>>(
            g_p, w0.typed_data(), w1.typed_data(), g2f.typed_data(), apar_p, out, n_rows, n_spatial, rows_per_chunk); break;
        default: field_moments_kernel<2, true><<<grid, threads, 0, stream>>>(
            g_p, w0.typed_data(), w1.typed_data(), g2f.typed_data(), apar_p, out, n_rows, n_spatial, rows_per_chunk); break;
    }

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
        return xla_ffi::Error(XLA_FFI_Error_Code_INTERNAL, cudaGetErrorString(err));
    return xla_ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    field_moments_ffi, FieldMomentsImpl,
    xla_ffi::Ffi::Bind()
        .Ctx<xla_ffi::PlatformStream<cudaStream_t>>()
        .Arg<xla_ffi::Buffer<xla_ffi::DataType::C128>>() // g
        .Arg<xla_ffi::Buffer<xla_ffi::DataType::F64>>()  // w0
        .Arg<xla_ffi::Buffer<xla_ffi::DataType::F64>>()  // w1
        .Arg<xla_ffi::Buffer<xla_ffi::DataType::F64>>()  // g2f
        .Arg<xla_ffi::Buffer<xla_ffi::DataType::C128>>() // apar
        .Ret<xla_ffi::Buffer<xla_ffi::DataType::C128>>() // partial
        .Attr<int32_t>("n_rows")
        .Attr<int32_t>("n_spatial")
        .Attr<int32_t>("n_chunks")
        .Attr<int32_t>("n_w")
        .Attr<int32_t>("use_g2f")
);
