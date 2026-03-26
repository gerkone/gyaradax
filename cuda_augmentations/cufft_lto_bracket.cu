// Link-Time Optimization (LTO) implementations of Gyrokinetics Nonlinear Bracket
#include "xla/ffi/api/ffi.h"
#include <cuda_runtime.h>
#include <cufft.h>
#include <cufftXt.h>
#include <iostream>
#include <cstddef>   // offsetof

#include "bracket_load_cb_fatbin.h"
#include "bracket_d2z_load_cb_fatbin.h"
#include "bracket_z2z_load_cb_fatbin.h"
#include "bracket_d2z_load_v2_cb_fatbin.h"

#define CHECK_CUDA(call) { if ((call) != cudaSuccess) return xla::ffi::Error::Internal("CUDA Error"); }
#define CHECK_CUFFT(call) { if ((call) != CUFFT_SUCCESS) return xla::ffi::Error::Internal("cuFFT Error"); }

namespace xla_ffi = xla::ffi;

// Callback Configuration Structs
struct CallbackInfo {
    const double2* df_packed;
    const double2* phi_packed;
    const double*  kx;
    const double*  ky;
    const int*     inverse_jind;
    int mrad, mphi_half, nkx, nky, gradient_type, n_df_batches, n_phi_batches;
};

struct CallbackInfoZ2Z {
    const double2* df_packed;
    const double2* phi_packed;
    const double*  kx;
    const double*  ky;
    const int*     inverse_jind;
    int mrad, mphi, nkx, nky, pair_type, n_df_batches, n_phi_batches;
};

struct BracketD2zV2Info {
    const double2 *ws0, *ws1;
    const double* dum_s;
    int nspec, mrad, mphi;
    double scale;
};

// Bracket kernels for non-LTO stages
__global__ void lto_bracket_explicit_kernel(const double* py, const double* fx, const double* px, const double* fy, double* nl, size_t n, int mrad, int mphi, const double* dum_s, int nspec, double scale) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        int batch_idx = (int)(i / ((size_t)mrad * mphi));
        nl[i] = scale * dum_s[batch_idx % nspec] * (py[i] * fx[i] - px[i] * fy[i]);
    }
}

// -----------------------------------------------------------------------------
// Version 0: Real-space bracket kernel (Works)
// Uses Z2D load callbacks for derivatives, then explicit kernel.
// -----------------------------------------------------------------------------
static struct {
    cufftHandle z2d = 0, d2z = 0;
    int batch = -1;
    CallbackInfo* d_cb = nullptr;
    void* d_cb_ptr = nullptr;
    double *d_dum = nullptr;
    cufftDoubleComplex* d_in = nullptr;
    double *d_py = nullptr, *d_fx = nullptr, *d_px = nullptr, *d_fy = nullptr, *d_nl = nullptr;
} v0;

xla_ffi::Error LtoFftBracketImpl(cudaStream_t stream, xla_ffi::Buffer<xla_ffi::DataType::C128> df, xla_ffi::Buffer<xla_ffi::DataType::C128> phi,
    xla_ffi::Buffer<xla_ffi::DataType::F64> kx, xla_ffi::Buffer<xla_ffi::DataType::F64> ky, xla_ffi::Buffer<xla_ffi::DataType::S32> jind,
    xla_ffi::Buffer<xla_ffi::DataType::F64> dum_s,
    xla_ffi::Result<xla_ffi::Buffer<xla_ffi::DataType::C128>> out, int32_t batch, int32_t mrad, int32_t mphi, int32_t nkx, int32_t nky) {

    int mphi_half = mphi/2 + 1;
    size_t r_dist = (size_t)mrad * mphi, c_dist = (size_t)mrad * mphi_half;
    if (v0.z2d == 0 || v0.batch != batch) {
        int n_int[2] = {(int)mrad, (int)mphi}; long long int n_ll[2] = {mrad, mphi};
        CHECK_CUDA(cudaMalloc(&v0.d_cb, sizeof(CallbackInfo))); v0.d_cb_ptr = (void*)v0.d_cb;
        CHECK_CUDA(cudaMalloc(&v0.d_dum, dum_s.dimensions()[0]*8));
        CHECK_CUDA(cudaMalloc(&v0.d_in, (size_t)batch * c_dist * 16));
        CHECK_CUDA(cudaMalloc(&v0.d_py, (size_t)batch * r_dist * 8)); CHECK_CUDA(cudaMalloc(&v0.d_fx, (size_t)batch * r_dist * 8));
        CHECK_CUDA(cudaMalloc(&v0.d_px, (size_t)batch * r_dist * 8)); CHECK_CUDA(cudaMalloc(&v0.d_fy, (size_t)batch * r_dist * 8));
        CHECK_CUDA(cudaMalloc(&v0.d_nl, (size_t)batch * r_dist * 8));

        CHECK_CUFFT(cufftCreate(&v0.z2d));
        CHECK_CUFFT(cufftXtSetJITCallback(v0.z2d, "d_load_cb_ptr", (void*)bracket_load_cb_fatbin, sizeof(bracket_load_cb_fatbin), CUFFT_CB_LD_COMPLEX_DOUBLE, &v0.d_cb_ptr));
        size_t ws=0; CHECK_CUFFT(cufftXtMakePlanMany(v0.z2d, 2, n_ll, NULL,1, (long long)c_dist, CUDA_C_64F, NULL,1, (long long)r_dist, CUDA_R_64F, batch, &ws, CUDA_C_64F));

        CHECK_CUFFT(cufftPlanMany(&v0.d2z, 2, n_int, NULL,1, (int)r_dist, NULL,1, (int)c_dist, CUFFT_D2Z, batch));
        v0.batch = batch;
    }

    CHECK_CUFFT(cufftSetStream(v0.z2d, stream)); CHECK_CUFFT(cufftSetStream(v0.d2z, stream));
    CHECK_CUDA(cudaMemcpyAsync(v0.d_dum, dum_s.typed_data(), dum_s.dimensions()[0]*8, cudaMemcpyHostToDevice, stream));

    static const int grad_types[4] = {0, 1, 2, 3};
    double* const ws_v0[4] = {v0.d_py, v0.d_fx, v0.d_px, v0.d_fy};
    CallbackInfo h_ci = {(const double2*)df.typed_data(), (const double2*)phi.typed_data(), kx.typed_data(), ky.typed_data(), jind.typed_data(), mrad, mphi_half, nkx, nky, 0, (int)df.dimensions()[0], (int)phi.dimensions()[0]};
    CHECK_CUDA(cudaMemcpyAsync(v0.d_cb, &h_ci, sizeof(CallbackInfo), cudaMemcpyHostToDevice, stream));
    CHECK_CUFFT(cufftExecZ2D(v0.z2d, v0.d_in, v0.d_py));
    for (int i=1; i<4; ++i) {
        CHECK_CUDA(cudaMemcpyAsync((char*)v0.d_cb + offsetof(CallbackInfo, gradient_type), &grad_types[i], sizeof(int), cudaMemcpyHostToDevice, stream));
        CHECK_CUFFT(cufftExecZ2D(v0.z2d, v0.d_in, ws_v0[i]));
    }

    size_t total_n = (size_t)batch * mrad * mphi;
    lto_bracket_explicit_kernel<<<(total_n+511)/512, 512, 0, stream>>>(v0.d_py, v0.d_fx, v0.d_px, v0.d_fy, v0.d_nl, total_n, mrad, mphi, v0.d_dum, (int)dum_s.dimensions()[0], 1.0);

    CHECK_CUFFT(cufftExecD2Z(v0.d2z, v0.d_nl, (double2*)out->typed_data()));
    return xla_ffi::Error::Success();
}

// -----------------------------------------------------------------------------
// Version 1: Working LTO Baseline
// Like V0, but consolidate memory management and ensure stability.
// -----------------------------------------------------------------------------
static struct {
    cufftHandle z2d = 0, d2z = 0;
    int batch = -1;
    CallbackInfo* d_cb = nullptr;
    void* d_cb_ptr = nullptr;
    double *d_dum = nullptr;
    cufftDoubleComplex* d_in = nullptr;
    double *d_py = nullptr, *d_fx = nullptr, *d_px = nullptr, *d_fy = nullptr, *d_nl = nullptr;
} v1;

xla_ffi::Error LtoFftBracketV1Impl(cudaStream_t stream, xla_ffi::Buffer<xla_ffi::DataType::C128> df, xla_ffi::Buffer<xla_ffi::DataType::C128> phi,
    xla_ffi::Buffer<xla_ffi::DataType::F64> kx, xla_ffi::Buffer<xla_ffi::DataType::F64> ky, xla_ffi::Buffer<xla_ffi::DataType::S32> jind,
    xla_ffi::Buffer<xla_ffi::DataType::F64> dum_s,
    xla_ffi::Result<xla_ffi::Buffer<xla_ffi::DataType::C128>> out, int32_t batch, int32_t mrad, int32_t mphi, int32_t nkx, int32_t nky) {

    int mphi_half = mphi/2 + 1;
    size_t r_dist = (size_t)mrad * mphi, c_dist = (size_t)mrad * mphi_half;
    if (v1.z2d == 0 || v1.batch != batch) {
        int n_int[2] = {(int)mrad, (int)mphi}; long long int n_ll[2] = {mrad, mphi};
        CHECK_CUDA(cudaMalloc(&v1.d_cb, sizeof(CallbackInfo))); v1.d_cb_ptr = (void*)v1.d_cb;
        CHECK_CUDA(cudaMalloc(&v1.d_dum, dum_s.dimensions()[0]*8));
        CHECK_CUDA(cudaMalloc(&v1.d_in, (size_t)batch * c_dist * 16));
        CHECK_CUDA(cudaMalloc(&v1.d_py, (size_t)batch * r_dist * 8)); CHECK_CUDA(cudaMalloc(&v1.d_fx, (size_t)batch * r_dist * 8));
        CHECK_CUDA(cudaMalloc(&v1.d_px, (size_t)batch * r_dist * 8)); CHECK_CUDA(cudaMalloc(&v1.d_fy, (size_t)batch * r_dist * 8));
        CHECK_CUDA(cudaMalloc(&v1.d_nl, (size_t)batch * r_dist * 8));

        CHECK_CUFFT(cufftCreate(&v1.z2d));
        CHECK_CUFFT(cufftXtSetJITCallback(v1.z2d, "d_load_cb_ptr", (void*)bracket_load_cb_fatbin, sizeof(bracket_load_cb_fatbin), CUFFT_CB_LD_COMPLEX_DOUBLE, &v1.d_cb_ptr));
        size_t ws=0; CHECK_CUFFT(cufftXtMakePlanMany(v1.z2d, 2, n_ll, NULL,1, (long long)c_dist, CUDA_C_64F, NULL,1, (long long)r_dist, CUDA_R_64F, batch, &ws, CUDA_C_64F));
        CHECK_CUFFT(cufftPlanMany(&v1.d2z, 2, n_int, NULL,1, (int)r_dist, NULL,1, (int)c_dist, CUFFT_D2Z, batch));
        v1.batch = batch;
    }

    CHECK_CUFFT(cufftSetStream(v1.z2d, stream)); CHECK_CUFFT(cufftSetStream(v1.d2z, stream));
    CHECK_CUDA(cudaMemcpyAsync(v1.d_dum, dum_s.typed_data(), dum_s.dimensions()[0]*8, cudaMemcpyHostToDevice, stream));

    static const int grad_types[4] = {0, 1, 2, 3};
    double* const ws_v1[4] = {v1.d_py, v1.d_fx, v1.d_px, v1.d_fy};
    CallbackInfo h_ci = {(const double2*)df.typed_data(), (const double2*)phi.typed_data(), kx.typed_data(), ky.typed_data(), jind.typed_data(), mrad, mphi_half, nkx, nky, 0, (int)df.dimensions()[0], (int)phi.dimensions()[0]};
    CHECK_CUDA(cudaMemcpyAsync(v1.d_cb, &h_ci, sizeof(CallbackInfo), cudaMemcpyHostToDevice, stream));
    CHECK_CUFFT(cufftExecZ2D(v1.z2d, v1.d_in, v1.d_py));
    for (int i=1; i<4; ++i) {
        CHECK_CUDA(cudaMemcpyAsync((char*)v1.d_cb + offsetof(CallbackInfo, gradient_type), &grad_types[i], sizeof(int), cudaMemcpyHostToDevice, stream));
        CHECK_CUFFT(cufftExecZ2D(v1.z2d, v1.d_in, ws_v1[i]));
    }

    size_t total_n = (size_t)batch * mrad * mphi;
    lto_bracket_explicit_kernel<<<(total_n+511)/512, 512, 0, stream>>>(v1.d_py, v1.d_fx, v1.d_px, v1.d_fy, v1.d_nl, total_n, mrad, mphi, v1.d_dum, (int)dum_s.dimensions()[0], 1.0);
    CHECK_CUFFT(cufftExecD2Z(v1.d2z, v1.d_nl, (double2*)out->typed_data()));

    return xla_ffi::Error::Success();
}

struct BracketD2zInfo {
    const double *py, *fx, *px, *fy;
    const double* dum_s;
    int nspec, mrad, mphi;
    double scale;
};

// -----------------------------------------------------------------------------
// Version 2: Fixed D2Z-Bracket-Fusion LTO
// Uses Z2D load callbacks for derivatives, then fused D2Z load callback for bracket.
// -----------------------------------------------------------------------------
static struct {
    cufftHandle z2d = 0, d2z = 0;
    int batch = -1;
    CallbackInfo*    d_cb      = nullptr;   // Z2D load callback info
    void*            d_cb_ptr  = nullptr;
    BracketD2zInfo*  d_d2z_cb  = nullptr;   // D2Z bracket-fusion callback info
    void*            d_d2z_ptr = nullptr;
    double*          d_dum     = nullptr;
    cufftDoubleComplex* d_in   = nullptr;   // Z2D dummy input (half-spectrum)
    double *d_py = nullptr, *d_fx = nullptr,
           *d_px = nullptr, *d_fy = nullptr;  // real-space workspaces (no d_nl)
} v2;

xla_ffi::Error LtoFftBracketV2Impl(cudaStream_t stream, xla_ffi::Buffer<xla_ffi::DataType::C128> df, xla_ffi::Buffer<xla_ffi::DataType::C128> phi,
    xla_ffi::Buffer<xla_ffi::DataType::F64> kx, xla_ffi::Buffer<xla_ffi::DataType::F64> ky, xla_ffi::Buffer<xla_ffi::DataType::S32> jind,
    xla_ffi::Buffer<xla_ffi::DataType::F64> dum_s,
    xla_ffi::Result<xla_ffi::Buffer<xla_ffi::DataType::C128>> out, int32_t batch, int32_t mrad, int32_t mphi, int32_t nkx, int32_t nky) {
    
    int mphi_half = mphi/2 + 1;
    size_t r_dist = (size_t)mrad * mphi, c_dist = (size_t)mrad * mphi_half;

    if (v2.batch != -1 && v2.batch != batch) {
        cufftDestroy(v2.z2d); cufftDestroy(v2.d2z);
        cudaFree(v2.d_cb);    cudaFree(v2.d_d2z_cb);
        cudaFree(v2.d_dum);   cudaFree(v2.d_in);
        cudaFree(v2.d_py);    cudaFree(v2.d_fx);
        cudaFree(v2.d_px);    cudaFree(v2.d_fy);
        v2.z2d = v2.d2z = 0;
        v2.d_cb = nullptr; v2.d_d2z_cb = nullptr;
        v2.d_dum = nullptr; v2.d_in = nullptr;
        v2.d_py = v2.d_fx = v2.d_px = v2.d_fy = nullptr;
        v2.batch = -1;
    }

    if (v2.z2d == 0) {
        long long int n_ll[2] = {mrad, mphi};
        CHECK_CUDA(cudaMalloc(&v2.d_cb, sizeof(CallbackInfo))); v2.d_cb_ptr = (void*)v2.d_cb;
        CHECK_CUDA(cudaMalloc(&v2.d_d2z_cb, sizeof(BracketD2zInfo))); v2.d_d2z_ptr = (void*)v2.d_d2z_cb;
        CHECK_CUDA(cudaMalloc(&v2.d_dum, dum_s.dimensions()[0]*8));
        CHECK_CUDA(cudaMalloc(&v2.d_in, (size_t)batch * c_dist * 16));
        CHECK_CUDA(cudaMalloc(&v2.d_py, (size_t)batch * r_dist * 8));
        CHECK_CUDA(cudaMalloc(&v2.d_fx, (size_t)batch * r_dist * 8));
        CHECK_CUDA(cudaMalloc(&v2.d_px, (size_t)batch * r_dist * 8));
        CHECK_CUDA(cudaMalloc(&v2.d_fy, (size_t)batch * r_dist * 8));
        
        // Z2D plan — identical to V1
        CHECK_CUFFT(cufftCreate(&v2.z2d));
        CHECK_CUFFT(cufftXtSetJITCallback(v2.z2d, "d_load_cb_ptr", (void*)bracket_load_cb_fatbin, sizeof(bracket_load_cb_fatbin), CUFFT_CB_LD_COMPLEX_DOUBLE, &v2.d_cb_ptr));
        size_t ws=0;
        CHECK_CUFFT(cufftXtMakePlanMany(v2.z2d, 2, n_ll, NULL,1, (long long)c_dist, CUDA_C_64F, NULL,1, (long long)r_dist, CUDA_R_64F, batch, &ws, CUDA_C_64F));
        
        // D2Z plan with bracket-fusion LTO load callback
        CHECK_CUFFT(cufftCreate(&v2.d2z));
        CHECK_CUFFT(cufftXtSetJITCallback(v2.d2z, "d_bracket_d2z_load", (void*)bracket_d2z_load_cb_fatbin, sizeof(bracket_d2z_load_cb_fatbin), CUFFT_CB_LD_REAL_DOUBLE, &v2.d_d2z_ptr));
        CHECK_CUFFT(cufftXtMakePlanMany(v2.d2z, 2, n_ll, NULL,1, (long long)r_dist, CUDA_R_64F, NULL,1, (long long)c_dist, CUDA_C_64F, batch, &ws, CUDA_R_64F));
        v2.batch = batch;
    }
    
    CHECK_CUFFT(cufftSetStream(v2.z2d, stream)); CHECK_CUFFT(v2.d2z != 0 ? cufftSetStream(v2.d2z, stream) : CUFFT_SUCCESS);
    CHECK_CUDA(cudaMemcpyAsync(v2.d_dum, dum_s.typed_data(), dum_s.dimensions()[0]*8, cudaMemcpyHostToDevice, stream));
    
    static const int grad_types[4] = {0, 1, 2, 3};
    double* const ws_v2[4] = {v2.d_py, v2.d_fx, v2.d_px, v2.d_fy};
    CallbackInfo h_ci = {(const double2*)df.typed_data(), (const double2*)phi.typed_data(), kx.typed_data(), ky.typed_data(), jind.typed_data(), mrad, mphi_half, nkx, nky, 0, (int)df.dimensions()[0], (int)phi.dimensions()[0]};
    CHECK_CUDA(cudaMemcpyAsync(v2.d_cb, &h_ci, sizeof(CallbackInfo), cudaMemcpyHostToDevice, stream));
    CHECK_CUFFT(cufftExecZ2D(v2.z2d, v2.d_in, v2.d_py));
    for (int i=1; i<4; ++i) {
        CHECK_CUDA(cudaMemcpyAsync((char*)v2.d_cb + offsetof(CallbackInfo, gradient_type), &grad_types[i], sizeof(int), cudaMemcpyHostToDevice, stream));
        CHECK_CUFFT(cufftExecZ2D(v2.z2d, v2.d_in, ws_v2[i]));
    }

    // Populate D2Z bracket-fusion callback struct
    BracketD2zInfo h_dci = { v2.d_py, v2.d_fx, v2.d_px, v2.d_fy, v2.d_dum, (int)dum_s.dimensions()[0], mrad, mphi, 1.0 };
    CHECK_CUDA(cudaMemcpyAsync(v2.d_d2z_cb, &h_dci, sizeof(BracketD2zInfo), cudaMemcpyHostToDevice, stream));

    // D2Z with bracket fusion — pass v2.d_py as the non-null dummy; callback overrides the read
    CHECK_CUFFT(cufftExecD2Z(v2.d2z, v2.d_py, (double2*)out->typed_data()));
    
    return xla_ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(lto_fft_bracket_ffi, LtoFftBracketImpl, xla_ffi::Ffi::Bind().Ctx<xla_ffi::PlatformStream<cudaStream_t>>().Arg<xla_ffi::Buffer<xla_ffi::DataType::C128>>().Arg<xla_ffi::Buffer<xla_ffi::DataType::C128>>().Arg<xla_ffi::Buffer<xla_ffi::DataType::F64>>().Arg<xla_ffi::Buffer<xla_ffi::DataType::F64>>().Arg<xla_ffi::Buffer<xla_ffi::DataType::S32>>().Arg<xla_ffi::Buffer<xla_ffi::DataType::F64>>().Ret<xla_ffi::Buffer<xla_ffi::DataType::C128>>().Attr<int32_t>("batch").Attr<int32_t>("mrad").Attr<int32_t>("mphi").Attr<int32_t>("nkx").Attr<int32_t>("nky"));
XLA_FFI_DEFINE_HANDLER_SYMBOL(lto_fft_bracket_v1_ffi, LtoFftBracketV1Impl, xla_ffi::Ffi::Bind().Ctx<xla_ffi::PlatformStream<cudaStream_t>>().Arg<xla_ffi::Buffer<xla_ffi::DataType::C128>>().Arg<xla_ffi::Buffer<xla_ffi::DataType::C128>>().Arg<xla_ffi::Buffer<xla_ffi::DataType::F64>>().Arg<xla_ffi::Buffer<xla_ffi::DataType::F64>>().Arg<xla_ffi::Buffer<xla_ffi::DataType::S32>>().Arg<xla_ffi::Buffer<xla_ffi::DataType::F64>>().Ret<xla_ffi::Buffer<xla_ffi::DataType::C128>>().Attr<int32_t>("batch").Attr<int32_t>("mrad").Attr<int32_t>("mphi").Attr<int32_t>("nkx").Attr<int32_t>("nky"));
XLA_FFI_DEFINE_HANDLER_SYMBOL(lto_fft_bracket_v2_ffi, LtoFftBracketV2Impl, xla_ffi::Ffi::Bind().Ctx<xla_ffi::PlatformStream<cudaStream_t>>().Arg<xla_ffi::Buffer<xla_ffi::DataType::C128>>().Arg<xla_ffi::Buffer<xla_ffi::DataType::C128>>().Arg<xla_ffi::Buffer<xla_ffi::DataType::F64>>().Arg<xla_ffi::Buffer<xla_ffi::DataType::F64>>().Arg<xla_ffi::Buffer<xla_ffi::DataType::S32>>().Arg<xla_ffi::Buffer<xla_ffi::DataType::F64>>().Ret<xla_ffi::Buffer<xla_ffi::DataType::C128>>().Attr<int32_t>("batch").Attr<int32_t>("mrad").Attr<int32_t>("mphi").Attr<int32_t>("nkx").Attr<int32_t>("nky"));
