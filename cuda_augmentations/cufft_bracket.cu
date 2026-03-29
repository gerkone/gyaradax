#include "xla/ffi/api/ffi.h"
#include <cuda_runtime.h>
#include <cufft.h>
#include <cstdint>

// Global plan + workspace cache (4 real-space scratch buffers)
static cufftHandle plan_z2d = 0;
static cufftHandle plan_d2z = 0;
static int    cached_batch = -1;
static int    cached_mrad  = -1;
static int    cached_mphi  = -1;
static double cached_inv_n2 = 1.0;
static double* ws_a = nullptr;  // phi_y real → nl_real (in-place)
static double* ws_b = nullptr;  // f_x real
static double* ws_c = nullptr;  // phi_x real
static double* ws_d = nullptr;  // f_y real

// Fused bracket + per-species dum scaling (in-place into ws_a).
// ws_a: phi_y real (input, then nl_real output)
// dum_s_eff[spec] = dum_s[spec] * fft_scale * efun_sign * real(fft_prefactor), pre-computed by caller.
// Note: fft_prefactor is assumed real; if complex, fold only the real part here.
__global__ void fused_bracket_scale(
    double* __restrict__ ws_a,
    const double* __restrict__ ws_b,
    const double* __restrict__ ws_c,
    const double* __restrict__ ws_d,
    const double* __restrict__ dum_s_eff,
    size_t n,
    int real_stride,   // mrad * mphi
    int nspec,
    double inv_n2
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        int spec_idx = (int)(idx / real_stride) % nspec;
        double2 v0 = ((const double2*)ws_a)[idx];
        double2 v1 = ((const double2*)ws_b)[idx];
        ws_a[idx] = inv_n2 * dum_s_eff[spec_idx] * (v0.x * v0.y - v1.x * v1.y);
    }
}

// vZ2Z Path A bracket (direction-based pairing):
// ws0 = phi_y + i*f_y, ws1 = f_x + i*phi_x
// bracket = phi_y*f_x - phi_x*f_y = v0.x*v1.x - v1.y*v0.y
__global__ void vz2z_bracket_kernel(
    double* ws_out, const double* ws_a, const double* ws_b, const double* dum_s,
    size_t n, int real_stride, int nspec, double scale
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        int spec_idx = (int)(idx / real_stride) % nspec;
        double2 v0 = ((const double2*)ws_a)[idx];
        double2 v1 = ((const double2*)ws_b)[idx];
        ws_out[idx] = scale * dum_s[spec_idx] * (v0.x * v1.x - v1.y * v0.y);
    }
}

namespace xla_ffi = xla::ffi;

xla_ffi::Error CufftBracketImpl(
    cudaStream_t stream,
    xla_ffi::Buffer<xla_ffi::DataType::C128> phi_y_k,
    xla_ffi::Buffer<xla_ffi::DataType::C128> f_x_k,
    xla_ffi::Buffer<xla_ffi::DataType::C128> phi_x_k,
    xla_ffi::Buffer<xla_ffi::DataType::C128> f_y_k,
    xla_ffi::Buffer<xla_ffi::DataType::F64>  dum_s_eff,   // [nspec]
    xla_ffi::Result<xla_ffi::Buffer<xla_ffi::DataType::C128>> out,
    int32_t batch,
    int32_t mrad,
    int32_t mphi,
    int32_t nspec
) {
    // Plan + workspace management
    if (plan_z2d == 0 || cached_batch != batch || cached_mrad != mrad || cached_mphi != mphi) {
        if (plan_z2d != 0) {
            cufftDestroy(plan_z2d); cufftDestroy(plan_d2z);
            cudaFree(ws_a); cudaFree(ws_b); cudaFree(ws_c); cudaFree(ws_d);
        }
        int n[2] = {mrad, mphi};
        cufftPlanMany(&plan_z2d, 2, n, NULL, 1, 0, NULL, 1, 0, CUFFT_Z2D, batch);
        cufftPlanMany(&plan_d2z, 2, n, NULL, 1, 0, NULL, 1, 0, CUFFT_D2Z, batch);
        size_t nbytes = (size_t)batch * mrad * mphi * sizeof(double);
        cudaMalloc(&ws_a, nbytes); cudaMalloc(&ws_b, nbytes);
        cudaMalloc(&ws_c, nbytes); cudaMalloc(&ws_d, nbytes);
        cached_batch  = batch;
        cached_mrad   = mrad;
        cached_mphi   = mphi;
        cached_inv_n2 = 1.0 / ((double)mrad * mphi * (double)mrad * mphi);
    }
    cufftSetStream(plan_z2d, stream);
    cufftSetStream(plan_d2z, stream);

    // 4× inverse FFTs (Z2D) into owned workspaces
    cufftExecZ2D(plan_z2d, (cufftDoubleComplex*)phi_y_k.typed_data(), ws_a);
    cufftExecZ2D(plan_z2d, (cufftDoubleComplex*)f_x_k  .typed_data(), ws_b);
    cufftExecZ2D(plan_z2d, (cufftDoubleComplex*)phi_x_k.typed_data(), ws_c);
    cufftExecZ2D(plan_z2d, (cufftDoubleComplex*)f_y_k  .typed_data(), ws_d);

    // Fused bracket + dum_s scaling in real space (in-place into ws_a)
    size_t total = (size_t)batch * mrad * mphi;
    int threads = 512;
    int blocks  = (int)((total + threads - 1) / threads);
    fused_bracket_scale<<<blocks, threads, 0, stream>>>(
        ws_a, ws_b, ws_c, ws_d,
        dum_s_eff.typed_data(),
        total, mrad * mphi, nspec, cached_inv_n2
    );

    // Forward FFT (D2Z) → output
    cufftExecD2Z(plan_d2z, ws_a, (cufftDoubleComplex*)out->typed_data());

    return xla_ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    cufft_bracket_ffi, CufftBracketImpl,
    xla_ffi::Ffi::Bind()
        .Ctx<xla_ffi::PlatformStream<cudaStream_t>>()
        .Arg<xla_ffi::Buffer<xla_ffi::DataType::C128>>()  // phi_y_k
        .Arg<xla_ffi::Buffer<xla_ffi::DataType::C128>>()  // f_x_k
        .Arg<xla_ffi::Buffer<xla_ffi::DataType::C128>>()  // phi_x_k
        .Arg<xla_ffi::Buffer<xla_ffi::DataType::C128>>()  // f_y_k
        .Arg<xla_ffi::Buffer<xla_ffi::DataType::F64>>()   // dum_s_eff [nspec]
        .Ret<xla_ffi::Buffer<xla_ffi::DataType::C128>>()  // out
        .Attr<int32_t>("batch")
        .Attr<int32_t>("mrad")
        .Attr<int32_t>("mphi")
        .Attr<int32_t>("nspec")
);
