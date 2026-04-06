#include "xla/ffi/api/ffi.h"
#include <cuda_runtime.h>
#include <iostream>

#define CHECK_CUDA(call) { if ((call) != cudaSuccess) return xla::ffi::Error::Internal("CUDA Error"); }

namespace xla_ffi = xla::ffi;

__global__ void apply_vpar_dual_stencil_kernel(
    const double2* __restrict__ field,
    double2*       __restrict__ output_d1,
    double2*       __restrict__ output_d4,
    int nv, int inner_size,
    double c0_d1, double c1_d1, double c2_d1, double c3_d1, double c4_d1,
    double c0_d4, double c1_d4, double c2_d4, double c3_d4, double c4_d4)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= inner_size) return;

    double coeffs_d1[5] = {c0_d1, c1_d1, c2_d1, c3_d1, c4_d1};
    double coeffs_d4[5] = {c0_d4, c1_d4, c2_d4, c3_d4, c4_d4};

    double2 w[5];
    w[0] = {0.0, 0.0};
    w[1] = {0.0, 0.0};
    w[2] = field[idx];
    w[3] = (nv > 1) ? field[inner_size + idx] : make_double2(0.0, 0.0);
    w[4] = (nv > 2) ? field[2 * inner_size + idx] : make_double2(0.0, 0.0);

    size_t offset = idx;
    for (int v = 0; v < nv; v++) {
        double2 result_d1 = {0.0, 0.0};
        double2 result_d4 = {0.0, 0.0};
        
        #pragma unroll
        for (int s = 0; s < 5; s++) {
            result_d1.x += coeffs_d1[s] * w[s].x;
            result_d1.y += coeffs_d1[s] * w[s].y;
            result_d4.x += coeffs_d4[s] * w[s].x;
            result_d4.y += coeffs_d4[s] * w[s].y;
        }
        output_d1[offset] = result_d1;
        output_d4[offset] = result_d4;

        w[0] = w[1];
        w[1] = w[2];
        w[2] = w[3];
        w[3] = w[4];

        int next_v = v + 3;
        w[4] = (next_v < nv) ? field[offset + 3 * inner_size] : make_double2(0.0, 0.0);
        offset += inner_size;
    }
}

xla_ffi::Error ApplyVparDualStencilImpl(
    cudaStream_t stream,
    xla_ffi::Buffer<xla_ffi::DataType::C128> field,
    xla_ffi::Result<xla_ffi::Buffer<xla_ffi::DataType::C128>> output_d1,
    xla_ffi::Result<xla_ffi::Buffer<xla_ffi::DataType::C128>> output_d4,
    double c0_d1, double c1_d1, double c2_d1, double c3_d1, double c4_d1,
    double c0_d4, double c1_d4, double c2_d4, double c3_d4, double c4_d4,
    int32_t nv, int32_t inner_size)
{
    int threads = 128;
    int blocks = (inner_size + threads - 1) / threads;

    apply_vpar_dual_stencil_kernel<<<blocks, threads, 0, stream>>>(
        (const double2*)field.typed_data(),
        (double2*)output_d1->typed_data(),
        (double2*)output_d4->typed_data(),
        nv, inner_size,
        c0_d1, c1_d1, c2_d1, c3_d1, c4_d1,
        c0_d4, c1_d4, c2_d4, c3_d4, c4_d4);

    CHECK_CUDA(cudaGetLastError());

    return xla_ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(apply_vpar_dual_stencil_ffi, ApplyVparDualStencilImpl,
    xla_ffi::Ffi::Bind()
    .Ctx<xla_ffi::PlatformStream<cudaStream_t>>()
    .Arg<xla_ffi::Buffer<xla_ffi::DataType::C128>>()
    .Ret<xla_ffi::Buffer<xla_ffi::DataType::C128>>()
    .Ret<xla_ffi::Buffer<xla_ffi::DataType::C128>>()
    .Attr<double>("c0_d1")
    .Attr<double>("c1_d1")
    .Attr<double>("c2_d1")
    .Attr<double>("c3_d1")
    .Attr<double>("c4_d1")
    .Attr<double>("c0_d4")
    .Attr<double>("c1_d4")
    .Attr<double>("c2_d4")
    .Attr<double>("c3_d4")
    .Attr<double>("c4_d4")
    .Attr<int32_t>("nv")
    .Attr<int32_t>("inner_size")
);
