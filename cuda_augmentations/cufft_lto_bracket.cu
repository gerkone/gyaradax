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
#include "bracket_store_cb_fatbin.h"
#include "bracket_d2z_z2z_load_cb_fatbin.h"
#include "bracket_z2z_merged_load_cb_fatbin.h"
#include "bracket_d2z_load_cb_merged_fatbin.h"

#define CHECK_CUDA(call) { if ((call) != cudaSuccess) return xla::ffi::Error::Internal("CUDA Error"); }
#define CHECK_CUFFT(call) { if ((call) != CUFFT_SUCCESS) return xla::ffi::Error::Internal("cuFFT Error"); }

namespace xla_ffi = xla::ffi;

struct ScaleFactors {
    double alpha0, beta0;
    double alpha1, beta1;
    double inv_a0, inv_b0;
    double inv_a1, inv_b1;
};

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
    const ScaleFactors* sf;
    int mrad, mphi, nkx, nky, pair_type;
    size_t df_size, phi_size;
};

struct CallbackInfoZ2Z_Merged {
    const double2* df_packed;
    const double2* phi_packed;
    const double*  kx;
    const double*  ky;
    const int*     inverse_jind;
    const ScaleFactors* sf;
    int mrad, mphi, nkx, nky, field_boundary;
    size_t df_size, phi_size;
};

struct BracketD2zInfoMerged {
    const double2* ws;
    const double*  dum_s;
    const ScaleFactors* sf;
    size_t df_offset;
    int nspec, mrad, mphi;
    double scale;
};

struct BracketD2zV2Info {
    const double2 *ws0, *ws1;
    const double* dum_s;
    int nspec, mrad, mphi;
    double scale;
};

extern "C" void launch_compute_scale_factors(
    cudaStream_t stream,
    const double2* phi, const double2* df,
    const double* kx, const double* ky,
    int nkx, int nky, int batch,
    size_t phi_size, size_t df_size,
    ScaleFactors* d_out);

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
// Version 2 (exp): Experimental Complex-Packed LTO (Reference only)
// Uses Z2Z callbacks + D2Z callback.
// -----------------------------------------------------------------------------
static struct {
    cufftHandle p0=0, p1=0, d2z=0;
    int batch=-1;
    CallbackInfoZ2Z *d_cb0=nullptr, *d_cb1=nullptr;
    BracketD2zV2Info *d_d2z_cb=nullptr;
    void *d_cb0_ptr=nullptr, *d_cb1_ptr=nullptr, *d_d2z_ptr=nullptr;
    double *d_dum=nullptr;
    cufftDoubleComplex *d_in=nullptr, *d_ws0=nullptr, *d_ws1=nullptr;
    cufftDoubleReal* d_d2z_in = nullptr;
} v_exp;

xla_ffi::Error LtoFftBracketVExpImpl(cudaStream_t stream, xla_ffi::Buffer<xla_ffi::DataType::C128> df, xla_ffi::Buffer<xla_ffi::DataType::C128> phi,
    xla_ffi::Buffer<xla_ffi::DataType::F64> kx, xla_ffi::Buffer<xla_ffi::DataType::F64> ky, xla_ffi::Buffer<xla_ffi::DataType::S32> jind,
    xla_ffi::Buffer<xla_ffi::DataType::F64> dum_s,
    xla_ffi::Result<xla_ffi::Buffer<xla_ffi::DataType::C128>> out, int32_t batch, int32_t mrad, int32_t mphi, int32_t nkx, int32_t nky) {
    
    if (v_exp.p0 == 0 || v_exp.batch != batch) {
        long long int n_ll[2] = {mrad, mphi};
        size_t r_dist = (size_t)mrad * mphi;
        CHECK_CUDA(cudaMalloc(&v_exp.d_cb0, sizeof(CallbackInfoZ2Z))); v_exp.d_cb0_ptr = (void*)v_exp.d_cb0;
        CHECK_CUDA(cudaMalloc(&v_exp.d_cb1, sizeof(CallbackInfoZ2Z))); v_exp.d_cb1_ptr = (void*)v_exp.d_cb1;
        CHECK_CUDA(cudaMalloc(&v_exp.d_d2z_cb, sizeof(BracketD2zV2Info))); v_exp.d_d2z_ptr = (void*)v_exp.d_d2z_cb;
        CHECK_CUDA(cudaMalloc(&v_exp.d_dum, dum_s.dimensions()[0]*8));
        CHECK_CUDA(cudaMalloc(&v_exp.d_in, (size_t)batch * r_dist * 16));
        CHECK_CUDA(cudaMalloc(&v_exp.d_ws0, (size_t)batch * r_dist * 16));
        CHECK_CUDA(cudaMalloc(&v_exp.d_ws1, (size_t)batch * r_dist * 16));
        CHECK_CUDA(cudaMalloc(&v_exp.d_d2z_in, (size_t)batch * r_dist * 8));
        
        CHECK_CUFFT(cufftCreate(&v_exp.p0)); 
        CHECK_CUFFT(cufftXtSetJITCallback(v_exp.p0, "d_z2z_load_cb_ptr", (void*)bracket_z2z_load_cb_fatbin, sizeof(bracket_z2z_load_cb_fatbin), CUFFT_CB_LD_COMPLEX_DOUBLE, &v_exp.d_cb0_ptr));
        size_t ws=0; CHECK_CUFFT(cufftXtMakePlanMany(v_exp.p0, 2, n_ll, NULL,1, (long long)r_dist, CUDA_C_64F, NULL,1, (long long)r_dist, CUDA_C_64F, batch, &ws, CUDA_C_64F));
        
        CHECK_CUFFT(cufftCreate(&v_exp.p1)); 
        CHECK_CUFFT(cufftXtSetJITCallback(v_exp.p1, "d_z2z_load_cb_ptr", (void*)bracket_z2z_load_cb_fatbin, sizeof(bracket_z2z_load_cb_fatbin), CUFFT_CB_LD_COMPLEX_DOUBLE, &v_exp.d_cb1_ptr));
        CHECK_CUFFT(cufftXtMakePlanMany(v_exp.p1, 2, n_ll, NULL,1, (long long)r_dist, CUDA_C_64F, NULL,1, (long long)r_dist, CUDA_C_64F, batch, &ws, CUDA_C_64F));
        
        CHECK_CUFFT(cufftCreate(&v_exp.d2z)); 
        CHECK_CUFFT(cufftXtSetJITCallback(v_exp.d2z, "d_bracket_d2z_v2_load", (void*)bracket_d2z_load_v2_cb_fatbin, sizeof(bracket_d2z_load_v2_cb_fatbin), CUFFT_CB_LD_REAL_DOUBLE, &v_exp.d_d2z_ptr));
        CHECK_CUFFT(cufftXtMakePlanMany(v_exp.d2z, 2, n_ll, NULL,1, (long long)r_dist, CUDA_R_64F, NULL,1, (long long)(mrad*(mphi/2+1)), CUDA_C_64F, batch, &ws, CUDA_R_64F));
        v_exp.batch = batch;
    }
    
    CHECK_CUFFT(cufftSetStream(v_exp.p0, stream)); CHECK_CUFFT(cufftSetStream(v_exp.p1, stream)); CHECK_CUFFT(cufftSetStream(v_exp.d2z, stream));
    CHECK_CUDA(cudaMemcpyAsync(v_exp.d_dum, dum_s.typed_data(), dum_s.dimensions()[0]*8, cudaMemcpyHostToDevice, stream));
    
    CallbackInfoZ2Z h_ci0 = {(const double2*)df.typed_data(), (const double2*)phi.typed_data(), kx.typed_data(), ky.typed_data(), jind.typed_data(), nullptr, mrad, mphi, nkx, nky, 0, (int)df.dimensions()[0], (int)phi.dimensions()[0]};
    CallbackInfoZ2Z h_ci1 = {(const double2*)df.typed_data(), (const double2*)phi.typed_data(), kx.typed_data(), ky.typed_data(), jind.typed_data(), nullptr, mrad, mphi, nkx, nky, 1, (int)df.dimensions()[0], (int)phi.dimensions()[0]};

    double scale = 1.0 / (double)mrad / (double)mphi / (double)mrad / (double)mphi;
    BracketD2zV2Info h_dci = {v_exp.d_ws0, v_exp.d_ws1, v_exp.d_dum, (int)dum_s.dimensions()[0], mrad, mphi, scale};
    
    CHECK_CUDA(cudaMemcpyAsync(v_exp.d_cb0, &h_ci0, sizeof(CallbackInfoZ2Z), cudaMemcpyHostToDevice, stream));
    CHECK_CUDA(cudaMemcpyAsync(v_exp.d_cb1, &h_ci1, sizeof(CallbackInfoZ2Z), cudaMemcpyHostToDevice, stream));
    CHECK_CUDA(cudaMemcpyAsync(v_exp.d_d2z_cb, &h_dci, sizeof(BracketD2zV2Info), cudaMemcpyHostToDevice, stream));
    
    CHECK_CUFFT(cufftExecZ2Z(v_exp.p0, v_exp.d_in, v_exp.d_ws0, CUFFT_INVERSE));
    CHECK_CUFFT(cufftExecZ2Z(v_exp.p1, v_exp.d_in, v_exp.d_ws1, CUFFT_INVERSE));
    CHECK_CUFFT(cufftExecD2Z(v_exp.d2z, (double*)v_exp.d_d2z_in, (double2*)out->typed_data()));
    
    return xla_ffi::Error::Success();
}

// -----------------------------------------------------------------------------
// Version 2: Optimized Fused D2Z LTO (Production)
// Uses 4x Z2D followed by a fused bracket loader in D2Z.
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
           *d_px = nullptr, *d_fy = nullptr;  // real-space workspaces
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
        
        CHECK_CUFFT(cufftCreate(&v2.z2d));
        CHECK_CUFFT(cufftXtSetJITCallback(v2.z2d, "d_load_cb_ptr", (void*)bracket_load_cb_fatbin, sizeof(bracket_load_cb_fatbin), CUFFT_CB_LD_COMPLEX_DOUBLE, &v2.d_cb_ptr));
        size_t ws=0;
        CHECK_CUFFT(cufftXtMakePlanMany(v2.z2d, 2, n_ll, NULL,1, (long long)c_dist, CUDA_C_64F, NULL,1, (long long)r_dist, CUDA_R_64F, batch, &ws, CUDA_C_64F));
        
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

    BracketD2zInfo h_dci = { v2.d_py, v2.d_fx, v2.d_px, v2.d_fy, v2.d_dum, (int)dum_s.dimensions()[0], mrad, mphi, 1.0 };
    CHECK_CUDA(cudaMemcpyAsync(v2.d_d2z_cb, &h_dci, sizeof(BracketD2zInfo), cudaMemcpyHostToDevice, stream));
    CHECK_CUFFT(cufftExecD2Z(v2.d2z, v2.d_py, (double2*)out->typed_data()));
    
    return xla_ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(lto_fft_bracket_ffi, LtoFftBracketImpl, xla_ffi::Ffi::Bind().Ctx<xla_ffi::PlatformStream<cudaStream_t>>().Arg<xla_ffi::Buffer<xla_ffi::DataType::C128>>().Arg<xla_ffi::Buffer<xla_ffi::DataType::C128>>().Arg<xla_ffi::Buffer<xla_ffi::DataType::F64>>().Arg<xla_ffi::Buffer<xla_ffi::DataType::F64>>().Arg<xla_ffi::Buffer<xla_ffi::DataType::S32>>().Arg<xla_ffi::Buffer<xla_ffi::DataType::F64>>().Ret<xla_ffi::Buffer<xla_ffi::DataType::C128>>().Attr<int32_t>("batch").Attr<int32_t>("mrad").Attr<int32_t>("mphi").Attr<int32_t>("nkx").Attr<int32_t>("nky"));
XLA_FFI_DEFINE_HANDLER_SYMBOL(lto_fft_bracket_v1_ffi, LtoFftBracketV1Impl, xla_ffi::Ffi::Bind().Ctx<xla_ffi::PlatformStream<cudaStream_t>>().Arg<xla_ffi::Buffer<xla_ffi::DataType::C128>>().Arg<xla_ffi::Buffer<xla_ffi::DataType::C128>>().Arg<xla_ffi::Buffer<xla_ffi::DataType::F64>>().Arg<xla_ffi::Buffer<xla_ffi::DataType::F64>>().Arg<xla_ffi::Buffer<xla_ffi::DataType::S32>>().Arg<xla_ffi::Buffer<xla_ffi::DataType::F64>>().Ret<xla_ffi::Buffer<xla_ffi::DataType::C128>>().Attr<int32_t>("batch").Attr<int32_t>("mrad").Attr<int32_t>("mphi").Attr<int32_t>("nkx").Attr<int32_t>("nky"));
XLA_FFI_DEFINE_HANDLER_SYMBOL(lto_fft_bracket_vexp_ffi, LtoFftBracketVExpImpl, xla_ffi::Ffi::Bind().Ctx<xla_ffi::PlatformStream<cudaStream_t>>().Arg<xla_ffi::Buffer<xla_ffi::DataType::C128>>().Arg<xla_ffi::Buffer<xla_ffi::DataType::C128>>().Arg<xla_ffi::Buffer<xla_ffi::DataType::F64>>().Arg<xla_ffi::Buffer<xla_ffi::DataType::F64>>().Arg<xla_ffi::Buffer<xla_ffi::DataType::S32>>().Arg<xla_ffi::Buffer<xla_ffi::DataType::F64>>().Ret<xla_ffi::Buffer<xla_ffi::DataType::C128>>().Attr<int32_t>("batch").Attr<int32_t>("mrad").Attr<int32_t>("mphi").Attr<int32_t>("nkx").Attr<int32_t>("nky"));
XLA_FFI_DEFINE_HANDLER_SYMBOL(lto_fft_bracket_v2_ffi, LtoFftBracketV2Impl, xla_ffi::Ffi::Bind().Ctx<xla_ffi::PlatformStream<cudaStream_t>>().Arg<xla_ffi::Buffer<xla_ffi::DataType::C128>>().Arg<xla_ffi::Buffer<xla_ffi::DataType::C128>>().Arg<xla_ffi::Buffer<xla_ffi::DataType::F64>>().Arg<xla_ffi::Buffer<xla_ffi::DataType::F64>>().Arg<xla_ffi::Buffer<xla_ffi::DataType::S32>>().Arg<xla_ffi::Buffer<xla_ffi::DataType::F64>>().Ret<xla_ffi::Buffer<xla_ffi::DataType::C128>>().Attr<int32_t>("batch").Attr<int32_t>("mrad").Attr<int32_t>("mphi").Attr<int32_t>("nkx").Attr<int32_t>("nky"));

// -----------------------------------------------------------------------------
// Version 3: Fused bracket + sparse store (Level 1 + Level 3)
// Like v2, but adds a CUFFT_CB_ST_COMPLEX_DOUBLE store callback on the D2Z plan
// that scatters output directly into packed [batch, nkx, nky], eliminating
// the unpack_half_spectrum gather on the Python side and 59% of D2Z output writes.
// -----------------------------------------------------------------------------

// Host-side mirror of StoreInfo (must match bracket_store_cb.cu exactly)
struct StoreInfo {
    double2*    out_packed;
    const int*  inverse_jind;   // [mrad] dense->packed, -1 if absent
    int mrad, mphiw3, nkx, nky;
};

static struct {
    cufftHandle z2d = 0, d2z = 0;
    int batch = -1;
    CallbackInfo*    d_cb        = nullptr;
    void*            d_cb_ptr    = nullptr;
    BracketD2zInfo*  d_d2z_cb    = nullptr;
    void*            d_d2z_ptr   = nullptr;
    StoreInfo*       d_store_cb  = nullptr;
    void*            d_store_ptr = nullptr;
    double*          d_dum       = nullptr;
    cufftDoubleComplex* d_in     = nullptr;
    double *d_py = nullptr, *d_fx = nullptr, *d_px = nullptr, *d_fy = nullptr;
    double2*         d_out_dummy = nullptr;  // cuFFT dataOut arg; store CB writes elsewhere
} v3;

xla_ffi::Error LtoFftBracketV3Impl(cudaStream_t stream,
    xla_ffi::Buffer<xla_ffi::DataType::C128> df, xla_ffi::Buffer<xla_ffi::DataType::C128> phi,
    xla_ffi::Buffer<xla_ffi::DataType::F64> kx, xla_ffi::Buffer<xla_ffi::DataType::F64> ky,
    xla_ffi::Buffer<xla_ffi::DataType::S32> jind, xla_ffi::Buffer<xla_ffi::DataType::F64> dum_s,
    xla_ffi::Result<xla_ffi::Buffer<xla_ffi::DataType::C128>> out,
    int32_t batch, int32_t mrad, int32_t mphi, int32_t nkx, int32_t nky) {

    int mphi_half = mphi / 2 + 1;
    size_t r_dist = (size_t)mrad * mphi, c_dist = (size_t)mrad * mphi_half;

    if (v3.batch != -1 && v3.batch != batch) {
        cufftDestroy(v3.z2d); cufftDestroy(v3.d2z);
        cudaFree(v3.d_cb);    cudaFree(v3.d_d2z_cb); cudaFree(v3.d_store_cb);
        cudaFree(v3.d_dum);   cudaFree(v3.d_in);
        cudaFree(v3.d_py);    cudaFree(v3.d_fx); cudaFree(v3.d_px); cudaFree(v3.d_fy);
        cudaFree(v3.d_out_dummy);
        v3 = {}; // zero-init
    }

    if (v3.z2d == 0) {
        long long int n_ll[2] = {mrad, mphi};
        CHECK_CUDA(cudaMalloc(&v3.d_cb,       sizeof(CallbackInfo)));   v3.d_cb_ptr    = (void*)v3.d_cb;
        CHECK_CUDA(cudaMalloc(&v3.d_d2z_cb,   sizeof(BracketD2zInfo))); v3.d_d2z_ptr   = (void*)v3.d_d2z_cb;
        CHECK_CUDA(cudaMalloc(&v3.d_store_cb, sizeof(StoreInfo)));      v3.d_store_ptr = (void*)v3.d_store_cb;
        CHECK_CUDA(cudaMalloc(&v3.d_dum,      dum_s.dimensions()[0] * 8));
        CHECK_CUDA(cudaMalloc(&v3.d_in,       (size_t)batch * c_dist * 16));
        CHECK_CUDA(cudaMalloc(&v3.d_py,       (size_t)batch * r_dist * 8));
        CHECK_CUDA(cudaMalloc(&v3.d_fx,       (size_t)batch * r_dist * 8));
        CHECK_CUDA(cudaMalloc(&v3.d_px,       (size_t)batch * r_dist * 8));
        CHECK_CUDA(cudaMalloc(&v3.d_fy,       (size_t)batch * r_dist * 8));
        CHECK_CUDA(cudaMalloc(&v3.d_out_dummy,(size_t)batch * c_dist * 16));

        // Z2D: load callback gathers df/phi from packed spectrum
        CHECK_CUFFT(cufftCreate(&v3.z2d));
        CHECK_CUFFT(cufftXtSetJITCallback(v3.z2d, "d_load_cb_ptr",
            (void*)bracket_load_cb_fatbin, sizeof(bracket_load_cb_fatbin),
            CUFFT_CB_LD_COMPLEX_DOUBLE, &v3.d_cb_ptr));
        size_t ws = 0;
        CHECK_CUFFT(cufftXtMakePlanMany(v3.z2d, 2, n_ll,
            NULL, 1, (long long)c_dist, CUDA_C_64F,
            NULL, 1, (long long)r_dist, CUDA_R_64F, batch, &ws, CUDA_C_64F));

        // D2Z: load callback computes bracket on-the-fly;
        //      store callback scatters result to packed [batch, nkx, nky] output.
        CHECK_CUFFT(cufftCreate(&v3.d2z));
        CHECK_CUFFT(cufftXtSetJITCallback(v3.d2z, "d_bracket_d2z_load",
            (void*)bracket_d2z_load_cb_fatbin, sizeof(bracket_d2z_load_cb_fatbin),
            CUFFT_CB_LD_REAL_DOUBLE, &v3.d_d2z_ptr));
        CHECK_CUFFT(cufftXtSetJITCallback(v3.d2z, "d_store_cb_ptr",
            (void*)bracket_store_cb_fatbin, sizeof(bracket_store_cb_fatbin),
            CUFFT_CB_ST_COMPLEX_DOUBLE, &v3.d_store_ptr));
        CHECK_CUFFT(cufftXtMakePlanMany(v3.d2z, 2, n_ll,
            NULL, 1, (long long)r_dist, CUDA_R_64F,
            NULL, 1, (long long)c_dist, CUDA_C_64F, batch, &ws, CUDA_R_64F));
        v3.batch = batch;
    }

    CHECK_CUFFT(cufftSetStream(v3.z2d, stream));
    CHECK_CUFFT(cufftSetStream(v3.d2z, stream));
    CHECK_CUDA(cudaMemcpyAsync(v3.d_dum, dum_s.typed_data(),
        dum_s.dimensions()[0] * 8, cudaMemcpyHostToDevice, stream));

    // 4x Z2D inverse transforms with gather load callback
    static const int grad_types[4] = {0, 1, 2, 3};
    double* const ws_v3[4] = {v3.d_py, v3.d_fx, v3.d_px, v3.d_fy};
    CallbackInfo h_ci = {
        (const double2*)df.typed_data(), (const double2*)phi.typed_data(),
        kx.typed_data(), ky.typed_data(), jind.typed_data(),
        mrad, mphi_half, nkx, nky, 0,
        (int)df.dimensions()[0], (int)phi.dimensions()[0]
    };
    CHECK_CUDA(cudaMemcpyAsync(v3.d_cb, &h_ci, sizeof(CallbackInfo), cudaMemcpyHostToDevice, stream));
    CHECK_CUFFT(cufftExecZ2D(v3.z2d, v3.d_in, v3.d_py));
    for (int i = 1; i < 4; ++i) {
        CHECK_CUDA(cudaMemcpyAsync((char*)v3.d_cb + offsetof(CallbackInfo, gradient_type),
            &grad_types[i], sizeof(int), cudaMemcpyHostToDevice, stream));
        CHECK_CUFFT(cufftExecZ2D(v3.z2d, v3.d_in, ws_v3[i]));
    }

    // D2Z with fused bracket (load CB) + sparse scatter (store CB)
    // Store CB writes to out->typed_data(); d_out_dummy is a required-but-unused sink.
    BracketD2zInfo h_dci = { v3.d_py, v3.d_fx, v3.d_px, v3.d_fy,
        v3.d_dum, (int)dum_s.dimensions()[0], mrad, mphi, 1.0 };
    StoreInfo h_si = {
        (double2*)out->typed_data(), jind.typed_data(), mrad, mphi_half, nkx, nky
    };
    CHECK_CUDA(cudaMemcpyAsync(v3.d_d2z_cb,  &h_dci, sizeof(BracketD2zInfo), cudaMemcpyHostToDevice, stream));
    CHECK_CUDA(cudaMemcpyAsync(v3.d_store_cb, &h_si,  sizeof(StoreInfo),      cudaMemcpyHostToDevice, stream));
    CHECK_CUFFT(cufftExecD2Z(v3.d2z, v3.d_py, v3.d_out_dummy));
    
    return xla_ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(lto_fft_bracket_v3_ffi, LtoFftBracketV3Impl,
    xla_ffi::Ffi::Bind()
    .Ctx<xla_ffi::PlatformStream<cudaStream_t>>()
    .Arg<xla_ffi::Buffer<xla_ffi::DataType::C128>>()
    .Arg<xla_ffi::Buffer<xla_ffi::DataType::C128>>()
    .Arg<xla_ffi::Buffer<xla_ffi::DataType::F64>>()
    .Arg<xla_ffi::Buffer<xla_ffi::DataType::F64>>()
    .Arg<xla_ffi::Buffer<xla_ffi::DataType::S32>>()
    .Arg<xla_ffi::Buffer<xla_ffi::DataType::F64>>()
    .Ret<xla_ffi::Buffer<xla_ffi::DataType::C128>>()
    .Attr<int32_t>("batch").Attr<int32_t>("mrad").Attr<int32_t>("mphi")
    .Attr<int32_t>("nkx").Attr<int32_t>("nky"));

// -----------------------------------------------------------------------------
// Version Z2Z: Explicit packing + 2x Z2Z + fused D2Z (Path A — Level 2 bypass)
//
// Eliminates the broken Z2Z LTO load callback by pre-filling ws0/ws1 with a
// regular CUDA gather kernel, then running plain in-place Z2Z IFFTs (no LTO).
// The D2Z stage carries the same dual LTO callbacks as v3:
//   load  = bracket fusion (reads ws0/ws1 as complex double2 with dum_s)
//   store = sparse scatter to packed [batch, nkx, nky] output
// -----------------------------------------------------------------------------

// Gather kernel: packs df/phi from [batch, nkx, nky] into Z2Z inputs
// [batch, mrad, mphi] with Hermitian extension for j > mphi/2.
// ws0 = phi_y + i*f_x,  ws1 = phi_x + i*f_y  (in spectral domain, pre-IFFT).
__global__ void gather_z2z_inputs(
    const double2* __restrict__ df_packed,    // [n_df_batch, nkx, nky]
    const double2* __restrict__ phi_packed,   // [n_phi_batch, nkx, nky]
    const double*  __restrict__ kx,           // [nkx]
    const double*  __restrict__ ky,           // [nky]
    const int*     __restrict__ inverse_jind, // [mrad]  dense->packed, -1 if absent
    double2* __restrict__ ws0,                // [batch, mrad, mphi] output
    double2* __restrict__ ws1,                // [batch, mrad, mphi] output
    int batch, int mrad, int mphi, int nkx, int nky,
    int n_df_batches, int n_phi_batches)
{
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (size_t)batch * mrad * mphi) return;

    int batch_idx = (int)(idx / ((size_t)mrad * mphi));
    int i_dense   = (int)((idx / mphi) % mrad);
    int j         = (int)(idx % mphi);

    // Standard 2D Hermitian coordinates
    bool mirrored = (j > mphi / 2);
    int j_lookup  = mirrored ? mphi - j : j;
    int i_lookup  = mirrored ? (mrad - i_dense) % mrad : i_dense;

    const double2 zero = {0.0, 0.0};
    if (j_lookup >= nky) { ws0[idx] = ws1[idx] = zero; return; }
    int i_pack = inverse_jind[i_lookup];
    if (i_pack < 0)      { ws0[idx] = ws1[idx] = zero; return; }

    double2 df_val  = df_packed [((size_t)(batch_idx % n_df_batches)  * nkx + i_pack) * nky + j_lookup];
    double2 phi_val = phi_packed[((size_t)(batch_idx % n_phi_batches) * nkx + i_pack) * nky + j_lookup];
    double kx_val = kx[i_pack];
    double ky_val = ky[j_lookup];

    // Derivatives in Fourier space: grad = i * k * signal
    // ga = phi_y = i*ky*phi
    // gb = f_x   = i*kx*df
    // gc = phi_x = i*kx*phi
    // gd = f_y   = i*ky*df
    double2 ga = {-ky_val * phi_val.y,  ky_val * phi_val.x};
    double2 gb = {-kx_val * df_val.y,   kx_val * df_val.x};
    double2 gc = {-kx_val * phi_val.y,  kx_val * phi_val.x};
    double2 gd = {-ky_val * df_val.y,   ky_val * df_val.x};

    // Direction-based pairing (minimizes cross-contamination):
    //   ws0 = phi_y + i*f_y  (both y-gradients, ga + i*gd)
    //   ws1 = f_x + i*phi_x  (both x-gradients, gb + i*gc)
    // Bracket = ws0.re*ws1.re - ws1.im*ws0.im = phi_y*f_x - phi_x*f_y
    if (!mirrored) {
        ws0[idx] = {ga.x - gd.y, ga.y + gd.x};
        ws1[idx] = {gb.x - gc.y, gb.y + gc.x};
    } else {
        // conj(A) + i*conj(B) for the Hermitian extension
        ws0[idx] = {ga.x + gd.y,  gd.x - ga.y};
        ws1[idx] = {gb.x + gc.y,  gc.x - gb.y};
    }
}

// Host-side mirror of BracketD2zZ2zInfo (must match bracket_d2z_z2z_load_cb.cu)
struct BracketD2zZ2zInfo {
    const double2* ws0;
    const double2* ws1;
    const double*  dum_s;
    const ScaleFactors* sf;
    int nspec, mrad, mphi;
    double scale;
};

__global__ void vz2z_bracket_kernel(
    double* ws_out, const double* ws_a, const double* ws_b, const double* dum_s,
    size_t n, int real_stride, int nspec, double scale
);

static struct {
    cufftHandle z2z0 = 0, z2z1 = 0, d2z = 0;
    int batch = -1;
    CallbackInfoZ2Z *d_cb0 = nullptr, *d_cb1 = nullptr;
    void *d_cb0_ptr = nullptr, *d_cb1_ptr = nullptr;
    BracketD2zZ2zInfo *d_d2z_cb = nullptr;
    void *d_d2z_ptr = nullptr;
    StoreInfo*         d_store_cb = nullptr;
    void*              d_store_ptr = nullptr;
    ScaleFactors*      d_sf       = nullptr;
    double*            d_dum      = nullptr;
    cufftDoubleComplex *d_in = nullptr, *d_ws0 = nullptr, *d_ws1 = nullptr;
    double2*           d_out_dummy = nullptr;
} vz2z;

xla_ffi::Error LtoFftBracketVZ2zImpl(cudaStream_t stream,
    xla_ffi::Buffer<xla_ffi::DataType::C128> df, xla_ffi::Buffer<xla_ffi::DataType::C128> phi,
    xla_ffi::Buffer<xla_ffi::DataType::F64> kx, xla_ffi::Buffer<xla_ffi::DataType::F64> ky,
    xla_ffi::Buffer<xla_ffi::DataType::S32> jind, xla_ffi::Buffer<xla_ffi::DataType::F64> dum_s,
    xla_ffi::Result<xla_ffi::Buffer<xla_ffi::DataType::C128>> out,
    int32_t batch, int32_t mrad, int32_t mphi, int32_t nkx, int32_t nky) {

    int mphi_half = mphi / 2 + 1;
    size_t r_dist = (size_t)mrad * mphi, c_dist = (size_t)mrad * mphi_half;

    if (vz2z.batch != -1 && vz2z.batch != batch) {
        printf("  [LTO] Re-initializing vz2z handles...\n");
        cufftDestroy(vz2z.z2z0); cufftDestroy(vz2z.z2z1); cufftDestroy(vz2z.d2z);
        cudaFree(vz2z.d_cb0); cudaFree(vz2z.d_cb1); 
        cudaFree(vz2z.d_d2z_cb); cudaFree(vz2z.d_store_cb);
        cudaFree(vz2z.d_sf);
        cudaFree(vz2z.d_dum); cudaFree(vz2z.d_in); 
        cudaFree(vz2z.d_ws0); cudaFree(vz2z.d_ws1);
        cudaFree(vz2z.d_out_dummy);
        vz2z = {};
    }

    if (vz2z.z2z0 == 0) {
        printf("  [LTO] Initializing vz2z handles for batch %d...\n", batch);
        long long n_ll[2] = {mrad, mphi};
        printf("  [LTO] Allocating callback infos and sf...\n");
        printf("  [LTO] Allocating callback infos and sf...\n");
        CHECK_CUDA(cudaMalloc(&vz2z.d_cb0, sizeof(CallbackInfoZ2Z))); vz2z.d_cb0_ptr = (void*)vz2z.d_cb0;
        CHECK_CUDA(cudaMalloc(&vz2z.d_cb1, sizeof(CallbackInfoZ2Z))); vz2z.d_cb1_ptr = (void*)vz2z.d_cb1;
        CHECK_CUDA(cudaMalloc(&vz2z.d_d2z_cb, sizeof(BracketD2zZ2zInfo))); vz2z.d_d2z_ptr = (void*)vz2z.d_d2z_cb;
        CHECK_CUDA(cudaMalloc(&vz2z.d_store_cb, sizeof(StoreInfo))); vz2z.d_store_ptr = (void*)vz2z.d_store_cb;
        CHECK_CUDA(cudaMalloc(&vz2z.d_sf, sizeof(ScaleFactors)));
        CHECK_CUDA(cudaMalloc(&vz2z.d_dum, dum_s.dimensions()[0]*8));
        
        printf("  [LTO] Allocating workspaces (batch=%d, r_dist=%zu)...\n", batch, r_dist);
        size_t bytes_r = (size_t)batch * r_dist * 16;
        CHECK_CUDA(cudaMalloc(&vz2z.d_ws0, bytes_r));
        CHECK_CUDA(cudaMalloc(&vz2z.d_ws1, bytes_r));
        // Use vz2z.d_ws0 as a dummy input/output where callbacks ignore the pointer
        vz2z.d_in = vz2z.d_ws0;
        vz2z.d_out_dummy = (cufftDoubleComplex*)vz2z.d_ws1;
        
        printf("  [LTO] Creating cuFFT plans...\n");
        CHECK_CUFFT(cufftCreate(&vz2z.z2z0));
        CHECK_CUFFT(cufftXtSetJITCallback(vz2z.z2z0, "d_z2z_load_cb", (void*)bracket_z2z_load_cb_fatbin, sizeof(bracket_z2z_load_cb_fatbin), CUFFT_CB_LD_COMPLEX_DOUBLE, &vz2z.d_cb0_ptr));
        size_t ws=0;
        CHECK_CUFFT(cufftXtMakePlanMany(vz2z.z2z0, 2, n_ll, NULL, 1, (long long)r_dist, CUDA_C_64F, NULL, 1, (long long)r_dist, CUDA_C_64F, batch, &ws, CUDA_C_64F));

        CHECK_CUFFT(cufftCreate(&vz2z.z2z1));
        CHECK_CUFFT(cufftXtSetJITCallback(vz2z.z2z1, "d_z2z_load_cb", (void*)bracket_z2z_load_cb_fatbin, sizeof(bracket_z2z_load_cb_fatbin), CUFFT_CB_LD_COMPLEX_DOUBLE, &vz2z.d_cb1_ptr));
        CHECK_CUFFT(cufftXtMakePlanMany(vz2z.z2z1, 2, n_ll, NULL, 1, (long long)r_dist, CUDA_C_64F, NULL, 1, (long long)r_dist, CUDA_C_64F, batch, &ws, CUDA_C_64F));

        CHECK_CUFFT(cufftCreate(&vz2z.d2z));
        CHECK_CUFFT(cufftXtSetJITCallback(vz2z.d2z, "d_bracket_d2z_z2z_load", (void*)bracket_d2z_z2z_load_cb_fatbin, sizeof(bracket_d2z_z2z_load_cb_fatbin), CUFFT_CB_LD_REAL_DOUBLE, &vz2z.d_d2z_ptr));
        CHECK_CUFFT(cufftXtSetJITCallback(vz2z.d2z, "d_store_cb_ptr", (void*)bracket_store_cb_fatbin, sizeof(bracket_store_cb_fatbin), CUFFT_CB_ST_COMPLEX_DOUBLE, &vz2z.d_store_ptr));
        CHECK_CUFFT(cufftXtMakePlanMany(vz2z.d2z, 2, n_ll, NULL, 1, (long long)r_dist, CUDA_R_64F, NULL, 1, (long long)c_dist, CUDA_C_64F, batch, &ws, CUDA_R_64F));
        
        vz2z.batch = batch;
    }

    CHECK_CUFFT(cufftSetStream(vz2z.z2z0, stream));
    CHECK_CUFFT(cufftSetStream(vz2z.z2z1, stream));
    CHECK_CUFFT(cufftSetStream(vz2z.d2z, stream));
    CHECK_CUDA(cudaMemcpyAsync(vz2z.d_dum, dum_s.typed_data(), dum_s.dimensions()[0]*8, cudaMemcpyHostToDevice, stream));

    size_t phi_sz = phi.element_count() * 16;
    size_t df_sz = df.element_count() * 16;
    launch_compute_scale_factors(stream, (const double2*)phi.typed_data(), (const double2*)df.typed_data(), kx.typed_data(), ky.typed_data(), nkx, nky, batch, phi_sz, df_sz, vz2z.d_sf);

    double scale = 1.0 / ((double)mrad * mphi * (double)mrad * mphi);
    
    CallbackInfoZ2Z h_ci0 = {(const double2*)df.typed_data(), (const double2*)phi.typed_data(), kx.typed_data(), ky.typed_data(), jind.typed_data(), vz2z.d_sf, mrad, mphi, nkx, nky, 0, df_sz, phi_sz};
    CallbackInfoZ2Z h_ci1 = {(const double2*)df.typed_data(), (const double2*)phi.typed_data(), kx.typed_data(), ky.typed_data(), jind.typed_data(), vz2z.d_sf, mrad, mphi, nkx, nky, 1, df_sz, phi_sz};
    
    BracketD2zZ2zInfo h_dci = {vz2z.d_ws0, vz2z.d_ws1, vz2z.d_dum, vz2z.d_sf, (int)dum_s.dimensions()[0], mrad, mphi, scale};
    
    StoreInfo h_si = {(double2*)out->typed_data(), jind.typed_data(), mrad, mphi_half, nkx, nky};

    CHECK_CUDA(cudaMemcpyAsync(vz2z.d_cb0, &h_ci0, sizeof(CallbackInfoZ2Z), cudaMemcpyHostToDevice, stream));
    CHECK_CUDA(cudaMemcpyAsync(vz2z.d_cb1, &h_ci1, sizeof(CallbackInfoZ2Z), cudaMemcpyHostToDevice, stream));
    CHECK_CUDA(cudaMemcpyAsync(vz2z.d_d2z_cb, &h_dci, sizeof(BracketD2zZ2zInfo), cudaMemcpyHostToDevice, stream));
    CHECK_CUDA(cudaMemcpyAsync(vz2z.d_store_cb, &h_si, sizeof(StoreInfo), cudaMemcpyHostToDevice, stream));

    CHECK_CUFFT(cufftExecZ2Z(vz2z.z2z0, vz2z.d_ws0, vz2z.d_ws0, CUFFT_INVERSE));
    CHECK_CUFFT(cufftExecZ2Z(vz2z.z2z1, vz2z.d_ws1, vz2z.d_ws1, CUFFT_INVERSE));
    CHECK_CUFFT(cufftExecD2Z(vz2z.d2z, (double*)vz2z.d_ws0, (double2*)vz2z.d_ws1));
    
    return xla_ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(lto_fft_bracket_vz2z_ffi, LtoFftBracketVZ2zImpl,
    xla_ffi::Ffi::Bind()
    .Ctx<xla_ffi::PlatformStream<cudaStream_t>>()
    .Arg<xla_ffi::Buffer<xla_ffi::DataType::C128>>()
    .Arg<xla_ffi::Buffer<xla_ffi::DataType::C128>>()
    .Arg<xla_ffi::Buffer<xla_ffi::DataType::F64>>()
    .Arg<xla_ffi::Buffer<xla_ffi::DataType::F64>>()
    .Arg<xla_ffi::Buffer<xla_ffi::DataType::S32>>()
    .Arg<xla_ffi::Buffer<xla_ffi::DataType::F64>>()
    .Ret<xla_ffi::Buffer<xla_ffi::DataType::C128>>()
    .Attr<int32_t>("batch").Attr<int32_t>("mrad").Attr<int32_t>("mphi")
    .Attr<int32_t>("nkx").Attr<int32_t>("nky"));

// -----------------------------------------------------------------------------
// Version 2a: Merged Z2Z Calls (Optimization Level 2a)
// Merges the two Z2Z transforms into a single batched call
// -----------------------------------------------------------------------------
static struct {
    cufftHandle z2z = 0, d2z = 0;
    int batch = -1;
    CallbackInfoZ2Z_Merged *d_cb = nullptr;
    void *d_cb_ptr = nullptr;
    BracketD2zInfoMerged *d_d2z_cb = nullptr;
    void *d_d2z_ptr = nullptr;
    StoreInfo*         d_store_cb = nullptr;
    void*              d_store_ptr = nullptr;
    ScaleFactors*      d_sf       = nullptr;
    double*            d_dum      = nullptr;
    cufftDoubleComplex *d_ws = nullptr;
    double2*           d_out_dummy = nullptr;
} vz2z_merged;

xla_ffi::Error LtoFftBracketVZ2zMergedImpl(cudaStream_t stream,
    xla_ffi::Buffer<xla_ffi::DataType::C128> df, xla_ffi::Buffer<xla_ffi::DataType::C128> phi,
    xla_ffi::Buffer<xla_ffi::DataType::F64> kx, xla_ffi::Buffer<xla_ffi::DataType::F64> ky,
    xla_ffi::Buffer<xla_ffi::DataType::S32> jind, xla_ffi::Buffer<xla_ffi::DataType::F64> dum_s,
    xla_ffi::Result<xla_ffi::Buffer<xla_ffi::DataType::C128>> out,
    int32_t batch, int32_t mrad, int32_t mphi, int32_t nkx, int32_t nky) {

    int mphi_half = mphi / 2 + 1;
    size_t r_dist = (size_t)mrad * mphi, c_dist = (size_t)mrad * mphi_half;

    if (vz2z_merged.batch != -1 && vz2z_merged.batch != batch) {
        cufftDestroy(vz2z_merged.z2z); cufftDestroy(vz2z_merged.d2z);
        cudaFree(vz2z_merged.d_cb); cudaFree(vz2z_merged.d_d2z_cb); 
        cudaFree(vz2z_merged.d_store_cb); cudaFree(vz2z_merged.d_sf);
        cudaFree(vz2z_merged.d_dum); cudaFree(vz2z_merged.d_ws);
        cudaFree(vz2z_merged.d_out_dummy);
        vz2z_merged = {};
    }

    if (vz2z_merged.z2z == 0) {
        long long n_ll[2] = {mrad, mphi};
        CHECK_CUDA(cudaMalloc(&vz2z_merged.d_cb, sizeof(CallbackInfoZ2Z_Merged))); vz2z_merged.d_cb_ptr = (void*)vz2z_merged.d_cb;
        CHECK_CUDA(cudaMalloc(&vz2z_merged.d_d2z_cb, sizeof(BracketD2zInfoMerged))); vz2z_merged.d_d2z_ptr = (void*)vz2z_merged.d_d2z_cb;
        CHECK_CUDA(cudaMalloc(&vz2z_merged.d_store_cb, sizeof(StoreInfo))); vz2z_merged.d_store_ptr = (void*)vz2z_merged.d_store_cb;
        CHECK_CUDA(cudaMalloc(&vz2z_merged.d_sf, sizeof(ScaleFactors)));
        CHECK_CUDA(cudaMalloc(&vz2z_merged.d_dum, dum_s.dimensions()[0]*8));
        
        // Single workspace for both fields: batch * 2
        size_t bytes_ws = (size_t)(batch * 2) * r_dist * 16;
        CHECK_CUDA(cudaMalloc(&vz2z_merged.d_ws, bytes_ws));
        CHECK_CUDA(cudaMalloc(&vz2z_merged.d_out_dummy, (size_t)batch * c_dist * 16));
        
        CHECK_CUFFT(cufftCreate(&vz2z_merged.z2z));
        CHECK_CUFFT(cufftXtSetJITCallback(vz2z_merged.z2z, "d_z2z_merged_load", (void*)bracket_z2z_merged_load_cb_fatbin, sizeof(bracket_z2z_merged_load_cb_fatbin), CUFFT_CB_LD_COMPLEX_DOUBLE, &vz2z_merged.d_cb_ptr));
        size_t ws=0;
        CHECK_CUFFT(cufftXtMakePlanMany(vz2z_merged.z2z, 2, n_ll, NULL, 1, (long long)r_dist, CUDA_C_64F, NULL, 1, (long long)r_dist, CUDA_C_64F, batch * 2, &ws, CUDA_C_64F));

        CHECK_CUFFT(cufftCreate(&vz2z_merged.d2z));
        CHECK_CUFFT(cufftXtSetJITCallback(vz2z_merged.d2z, "d_bracket_d2z_merged_load", (void*)bracket_d2z_load_cb_merged_fatbin, sizeof(bracket_d2z_load_cb_merged_fatbin), CUFFT_CB_LD_REAL_DOUBLE, &vz2z_merged.d_d2z_ptr));
        CHECK_CUFFT(cufftXtSetJITCallback(vz2z_merged.d2z, "d_store_cb_ptr", (void*)bracket_store_cb_fatbin, sizeof(bracket_store_cb_fatbin), CUFFT_CB_ST_COMPLEX_DOUBLE, &vz2z_merged.d_store_ptr));
        CHECK_CUFFT(cufftXtMakePlanMany(vz2z_merged.d2z, 2, n_ll, NULL, 1, (long long)r_dist, CUDA_R_64F, NULL, 1, (long long)c_dist, CUDA_C_64F, batch, &ws, CUDA_R_64F));
        
        vz2z_merged.batch = batch;
    }

    CHECK_CUFFT(cufftSetStream(vz2z_merged.z2z, stream));
    CHECK_CUFFT(cufftSetStream(vz2z_merged.d2z, stream));
    CHECK_CUDA(cudaMemcpyAsync(vz2z_merged.d_dum, dum_s.typed_data(), dum_s.dimensions()[0]*8, cudaMemcpyHostToDevice, stream));

    size_t phi_sz = phi.element_count() * 16;
    size_t df_sz = df.element_count() * 16;
    launch_compute_scale_factors(stream, (const double2*)phi.typed_data(), (const double2*)df.typed_data(), kx.typed_data(), ky.typed_data(), nkx, nky, batch, phi_sz, df_sz, vz2z_merged.d_sf);

    double scale = 1.0 / ((double)mrad * mphi * (double)mrad * mphi);
    
    CallbackInfoZ2Z_Merged h_ci = {(const double2*)df.typed_data(), (const double2*)phi.typed_data(), kx.typed_data(), ky.typed_data(), jind.typed_data(), vz2z_merged.d_sf, mrad, mphi, nkx, nky, batch, df_sz, phi_sz};
    BracketD2zInfoMerged h_dci = {(const double2*)vz2z_merged.d_ws, vz2z_merged.d_dum, vz2z_merged.d_sf, (size_t)batch * mrad * mphi, (int)dum_s.dimensions()[0], mrad, mphi, scale};
    StoreInfo h_si = {(double2*)out->typed_data(), jind.typed_data(), mrad, mphi_half, nkx, nky};

    CHECK_CUDA(cudaMemcpyAsync(vz2z_merged.d_cb, &h_ci, sizeof(CallbackInfoZ2Z_Merged), cudaMemcpyHostToDevice, stream));
    CHECK_CUDA(cudaMemcpyAsync(vz2z_merged.d_d2z_cb, &h_dci, sizeof(BracketD2zInfoMerged), cudaMemcpyHostToDevice, stream));
    CHECK_CUDA(cudaMemcpyAsync(vz2z_merged.d_store_cb, &h_si, sizeof(StoreInfo), cudaMemcpyHostToDevice, stream));

    CHECK_CUFFT(cufftExecZ2Z(vz2z_merged.z2z, vz2z_merged.d_ws, vz2z_merged.d_ws, CUFFT_INVERSE));
    CHECK_CUFFT(cufftExecD2Z(vz2z_merged.d2z, (double*)vz2z_merged.d_ws, vz2z_merged.d_out_dummy));
    
    return xla_ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(lto_fft_bracket_vz2z_merged_ffi, LtoFftBracketVZ2zMergedImpl,
    xla_ffi::Ffi::Bind()
    .Ctx<xla_ffi::PlatformStream<cudaStream_t>>()
    .Arg<xla_ffi::Buffer<xla_ffi::DataType::C128>>()
    .Arg<xla_ffi::Buffer<xla_ffi::DataType::C128>>()
    .Arg<xla_ffi::Buffer<xla_ffi::DataType::F64>>()
    .Arg<xla_ffi::Buffer<xla_ffi::DataType::F64>>()
    .Arg<xla_ffi::Buffer<xla_ffi::DataType::S32>>()
    .Arg<xla_ffi::Buffer<xla_ffi::DataType::F64>>()
    .Ret<xla_ffi::Buffer<xla_ffi::DataType::C128>>()
    .Attr<int32_t>("batch")
    .Attr<int32_t>("mrad")
    .Attr<int32_t>("mphi")
    .Attr<int32_t>("nkx")
    .Attr<int32_t>("nky")
);

// -----------------------------------------------------------------------------
// Version 4: CUDA Graphs (Optimization Level 4)
// Captures the Level 2a merged sequence into a CUDA Graph.
// -----------------------------------------------------------------------------
static struct {
    cufftHandle z2z = 0, d2z = 0;
    int batch = -1;
    CallbackInfoZ2Z_Merged *d_cb = nullptr;
    void *d_cb_ptr = nullptr;
    BracketD2zInfoMerged *d_d2z_cb = nullptr;
    void *d_d2z_ptr = nullptr;
    StoreInfo*         d_store_cb = nullptr;
    void*              d_store_ptr = nullptr;
    ScaleFactors*      d_sf       = nullptr;
    double*            d_dum      = nullptr;
    cufftDoubleComplex *d_ws = nullptr;
    double2*           d_out_dummy = nullptr;

    // Graph tracking
    cudaGraphExec_t graphExec = nullptr;
    void* last_df = nullptr;
    void* last_phi = nullptr;
    void* last_out = nullptr;
} v4;

xla_ffi::Error LtoFftBracketV4Impl(cudaStream_t stream,
    xla_ffi::Buffer<xla_ffi::DataType::C128> df, xla_ffi::Buffer<xla_ffi::DataType::C128> phi,
    xla_ffi::Buffer<xla_ffi::DataType::F64> kx, xla_ffi::Buffer<xla_ffi::DataType::F64> ky,
    xla_ffi::Buffer<xla_ffi::DataType::S32> jind, xla_ffi::Buffer<xla_ffi::DataType::F64> dum_s,
    xla_ffi::Result<xla_ffi::Buffer<xla_ffi::DataType::C128>> out,
    int32_t batch, int32_t mrad, int32_t mphi, int32_t nkx, int32_t nky) {

    int mphi_half = mphi / 2 + 1;
    size_t r_dist = (size_t)mrad * mphi, c_dist = (size_t)mrad * mphi_half;

    bool pointers_changed = (v4.last_df != df.typed_data() || v4.last_phi != phi.typed_data() || v4.last_out != out->typed_data());

    if (v4.batch != -1 && v4.batch != batch) {
        if (v4.graphExec) { cudaGraphExecDestroy(v4.graphExec); v4.graphExec = nullptr; }
        cufftDestroy(v4.z2z); cufftDestroy(v4.d2z);
        cudaFree(v4.d_cb); cudaFree(v4.d_d2z_cb); 
        cudaFree(v4.d_store_cb); cudaFree(v4.d_sf);
        cudaFree(v4.d_dum); cudaFree(v4.d_ws);
        cudaFree(v4.d_out_dummy);
        v4 = {};
    }

    if (v4.z2z == 0) {
        long long n_ll[2] = {mrad, mphi};
        CHECK_CUDA(cudaMalloc(&v4.d_cb, sizeof(CallbackInfoZ2Z_Merged))); v4.d_cb_ptr = (void*)v4.d_cb;
        CHECK_CUDA(cudaMalloc(&v4.d_d2z_cb, sizeof(BracketD2zInfoMerged))); v4.d_d2z_ptr = (void*)v4.d_d2z_cb;
        CHECK_CUDA(cudaMalloc(&v4.d_store_cb, sizeof(StoreInfo))); v4.d_store_ptr = (void*)v4.d_store_cb;
        CHECK_CUDA(cudaMalloc(&v4.d_sf, sizeof(ScaleFactors)));
        CHECK_CUDA(cudaMalloc(&v4.d_dum, dum_s.dimensions()[0]*8));
        
        size_t bytes_ws = (size_t)(batch * 2) * r_dist * 16;
        CHECK_CUDA(cudaMalloc(&v4.d_ws, bytes_ws));
        CHECK_CUDA(cudaMalloc(&v4.d_out_dummy, (size_t)batch * c_dist * 16));
        
        // CUFFT plan creation natively supports Graph capture in CUDA 11+.
        CHECK_CUFFT(cufftCreate(&v4.z2z));
        CHECK_CUFFT(cufftXtSetJITCallback(v4.z2z, "d_z2z_merged_load", (void*)bracket_z2z_merged_load_cb_fatbin, sizeof(bracket_z2z_merged_load_cb_fatbin), CUFFT_CB_LD_COMPLEX_DOUBLE, &v4.d_cb_ptr));
        size_t ws=0;
        CHECK_CUFFT(cufftXtMakePlanMany(v4.z2z, 2, n_ll, NULL, 1, (long long)r_dist, CUDA_C_64F, NULL, 1, (long long)r_dist, CUDA_C_64F, batch * 2, &ws, CUDA_C_64F));

        CHECK_CUFFT(cufftCreate(&v4.d2z));
        CHECK_CUFFT(cufftXtSetJITCallback(v4.d2z, "d_bracket_d2z_merged_load", (void*)bracket_d2z_load_cb_merged_fatbin, sizeof(bracket_d2z_load_cb_merged_fatbin), CUFFT_CB_LD_REAL_DOUBLE, &v4.d_d2z_ptr));
        CHECK_CUFFT(cufftXtSetJITCallback(v4.d2z, "d_store_cb_ptr", (void*)bracket_store_cb_fatbin, sizeof(bracket_store_cb_fatbin), CUFFT_CB_ST_COMPLEX_DOUBLE, &v4.d_store_ptr));
        CHECK_CUFFT(cufftXtMakePlanMany(v4.d2z, 2, n_ll, NULL, 1, (long long)r_dist, CUDA_R_64F, NULL, 1, (long long)c_dist, CUDA_C_64F, batch, &ws, CUDA_R_64F));
        
        CHECK_CUFFT(cufftSetStream(v4.z2z, stream));
        CHECK_CUFFT(cufftSetStream(v4.d2z, stream));
        v4.batch = batch;
    }

    // 1. Copy dynamic parameter structures to device (NOT captured in graph to allow pointers to change)
    CHECK_CUDA(cudaMemcpyAsync(v4.d_dum, dum_s.typed_data(), dum_s.dimensions()[0]*8, cudaMemcpyHostToDevice, stream));

    size_t phi_sz = phi.element_count() * 16;
    size_t df_sz = df.element_count() * 16;
    double scale = 1.0 / ((double)mrad * mphi * (double)mrad * mphi);
    
    CallbackInfoZ2Z_Merged h_ci = {(const double2*)df.typed_data(), (const double2*)phi.typed_data(), kx.typed_data(), ky.typed_data(), jind.typed_data(), v4.d_sf, mrad, mphi, nkx, nky, batch, df_sz, phi_sz};
    BracketD2zInfoMerged h_dci = {(const double2*)v4.d_ws, v4.d_dum, v4.d_sf, (size_t)batch * mrad * mphi, (int)dum_s.dimensions()[0], mrad, mphi, scale};
    StoreInfo h_si = {(double2*)out->typed_data(), jind.typed_data(), mrad, mphi_half, nkx, nky};

    // Copies are executed synchronously in the stream BEFORE graph launch
    CHECK_CUDA(cudaMemcpyAsync(v4.d_cb, &h_ci, sizeof(CallbackInfoZ2Z_Merged), cudaMemcpyHostToDevice, stream));
    CHECK_CUDA(cudaMemcpyAsync(v4.d_d2z_cb, &h_dci, sizeof(BracketD2zInfoMerged), cudaMemcpyHostToDevice, stream));
    CHECK_CUDA(cudaMemcpyAsync(v4.d_store_cb, &h_si, sizeof(StoreInfo), cudaMemcpyHostToDevice, stream));

    if (v4.graphExec == nullptr || pointers_changed) {
        if (v4.graphExec) { cudaGraphExecDestroy(v4.graphExec); v4.graphExec = nullptr; }

        v4.last_df = df.typed_data();
        v4.last_phi = phi.typed_data();
        v4.last_out = out->typed_data();

        // captured sequence
        CHECK_CUDA(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));

        launch_compute_scale_factors(stream, (const double2*)phi.typed_data(), (const double2*)df.typed_data(), kx.typed_data(), ky.typed_data(), nkx, nky, batch, phi_sz, df_sz, v4.d_sf);
        CHECK_CUFFT(cufftExecZ2Z(v4.z2z, v4.d_ws, v4.d_ws, CUFFT_INVERSE));
        CHECK_CUFFT(cufftExecD2Z(v4.d2z, (double*)v4.d_ws, v4.d_out_dummy));

        cudaGraph_t graph;
        CHECK_CUDA(cudaStreamEndCapture(stream, &graph));
        CHECK_CUDA(cudaGraphInstantiate(&v4.graphExec, graph, NULL, NULL, 0));
        CHECK_CUDA(cudaGraphDestroy(graph));
    }

    CHECK_CUDA(cudaGraphLaunch(v4.graphExec, stream));

    return xla_ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(lto_fft_bracket_v4_ffi, LtoFftBracketV4Impl,
    xla_ffi::Ffi::Bind()
    .Ctx<xla_ffi::PlatformStream<cudaStream_t>>()
    .Arg<xla_ffi::Buffer<xla_ffi::DataType::C128>>()
    .Arg<xla_ffi::Buffer<xla_ffi::DataType::C128>>()
    .Arg<xla_ffi::Buffer<xla_ffi::DataType::F64>>()
    .Arg<xla_ffi::Buffer<xla_ffi::DataType::F64>>()
    .Arg<xla_ffi::Buffer<xla_ffi::DataType::S32>>()
    .Arg<xla_ffi::Buffer<xla_ffi::DataType::F64>>()
    .Ret<xla_ffi::Buffer<xla_ffi::DataType::C128>>()
    .Attr<int32_t>("batch")
    .Attr<int32_t>("mrad")
    .Attr<int32_t>("mphi")
    .Attr<int32_t>("nkx")
    .Attr<int32_t>("nky")
);
