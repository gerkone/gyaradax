#include "xla/ffi/api/ffi.h"
#include <cuda_runtime.h>
#include <iostream>

#define CHECK_CUDA(call) { if ((call) != cudaSuccess) return xla::ffi::Error::Internal("CUDA Error"); }

namespace xla_ffi = xla::ffi;

__global__ void apply_vpar_stencil_kernel(
    const double2* __restrict__ field,
    double2*       __restrict__ output,
    int nv, int inner_size,
    double c0, double c1, double c2, double c3, double c4)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= inner_size) return;

    double coeffs[5] = {c0, c1, c2, c3, c4};

    double2 w[5];
    w[0] = {0.0, 0.0};
    w[1] = {0.0, 0.0};
    w[2] = field[0 * inner_size + idx];
    w[3] = (nv > 1) ? field[1 * inner_size + idx] : make_double2(0.0, 0.0);
    w[4] = (nv > 2) ? field[2 * inner_size + idx] : make_double2(0.0, 0.0);

    for (int v = 0; v < nv; v++) {
        double2 result = {0.0, 0.0};
        #pragma unroll
        for (int s = 0; s < 5; s++) {
            result.x += coeffs[s] * w[s].x;
            result.y += coeffs[s] * w[s].y;
        }
        output[v * inner_size + idx] = result;

        w[0] = w[1];
        w[1] = w[2];
        w[2] = w[3];
        w[3] = w[4];

        int next_v = v + 3;
        w[4] = (next_v < nv) ? field[next_v * inner_size + idx] : make_double2(0.0, 0.0);
    }
}

xla_ffi::Error ApplyVparStencilImpl(
    cudaStream_t stream,
    xla_ffi::Buffer<xla_ffi::DataType::C128> field,
    xla_ffi::Result<xla_ffi::Buffer<xla_ffi::DataType::C128>> output,
    double c0, double c1, double c2, double c3, double c4,
    int32_t nv, int32_t inner_size)
{
    int threads = 256;
    int blocks = (inner_size + threads - 1) / threads;

    apply_vpar_stencil_kernel<<<blocks, threads, 0, stream>>>(
        (const double2*)field.typed_data(),
        (double2*)output->typed_data(),
        nv, inner_size,
        c0, c1, c2, c3, c4);

    CHECK_CUDA(cudaGetLastError());

    return xla_ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(apply_vpar_stencil_ffi, ApplyVparStencilImpl,
    xla_ffi::Ffi::Bind()
    .Ctx<xla_ffi::PlatformStream<cudaStream_t>>()
    .Arg<xla_ffi::Buffer<xla_ffi::DataType::C128>>()
    .Ret<xla_ffi::Buffer<xla_ffi::DataType::C128>>()
    .Attr<double>("c0")
    .Attr<double>("c1")
    .Attr<double>("c2")
    .Attr<double>("c3")
    .Attr<double>("c4")
    .Attr<int32_t>("nv")
    .Attr<int32_t>("inner_size")
);
