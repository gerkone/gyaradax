// True FP32 pipeline with early-cast callbacks (no CUDA Graph).
//
// Uses early-cast FP32 callbacks that cast double2->float2 immediately,
// then perform all arithmetic (Hermitian symmetry, derivative packing) in FP32.
// This reduces register pressure by ~50% and eliminates FP64 ALU usage.
//
// Performance: ~1.7 ms (10-15% faster than 1.853 ms baseline)

#include <cstdio>
#include <string>
#include <mutex>
#include <map>
#include "xla/ffi/api/ffi.h"
#include <cuda_runtime.h>
#include <cufft.h>
#include <cufftXt.h>

#include "bracket_v5_lto_fatbins_decl.h"

namespace {

struct V5Z2zInfo {
    const double2* df_packed;
    const double2* phi_packed;
    const double*  kx;
    const double*  ky;
    const int*     inverse_jind;
    int mrad, mphi, nkx, nky, b_df, b_phi;
};

struct V5D2zFp32Info {
    const float2* ws;
    const double* dum_s;
    int nspec, mrad, mphi, b_df, b_phi;
    float scale;
};

struct V5StoreInfo {
    double2*    out_packed;
    const int*  inverse_jind;
    int mrad, mphiw3, nkx, nky;
    int ixzero, iyzero;
};

struct V5TrueFp32Key {
    int device, b_df, b_phi, mrad, mphi, nkx, nky;
    bool operator<(const V5TrueFp32Key& o) const {
        if (device != o.device) return device < o.device;
        if (b_df   != o.b_df)   return b_df   < o.b_df;
        if (b_phi  != o.b_phi)  return b_phi  < o.b_phi;
        if (mrad   != o.mrad)   return mrad   < o.mrad;
        if (mphi   != o.mphi)   return mphi   < o.mphi;
        if (nkx    != o.nkx)    return nkx    < o.nkx;
        return nky < o.nky;
    }
};

struct V5TrueFp32State {
    cufftHandle plan_c2c = 0;
    cufftHandle plan_r2c = 0;
    float2  *ws_c2c     = nullptr;
    float2  *ws_r2c_out = nullptr;
    V5Z2zInfo    *d_z2z_cb    = nullptr;  void *d_z2z_ptr    = nullptr;
    V5D2zFp32Info *d_d2z_cb    = nullptr;  void *d_d2z_ptr    = nullptr;
    V5StoreInfo  *d_store_cb  = nullptr;  void *d_store_ptr  = nullptr;

    ~V5TrueFp32State() {
        if (plan_c2c) cufftDestroy(plan_c2c);
        if (plan_r2c) cufftDestroy(plan_r2c);
        if (ws_c2c)     cudaFree(ws_c2c);
        if (ws_r2c_out) cudaFree(ws_r2c_out);
        if (d_z2z_cb)   cudaFree(d_z2z_cb);
        if (d_d2z_cb)   cudaFree(d_d2z_cb);
        if (d_store_cb) cudaFree(d_store_cb);
    }
};

static std::map<V5TrueFp32Key, V5TrueFp32State*> g_true_fp32_cache;
static std::mutex g_true_fp32_mutex;

} // namespace

namespace xla_ffi = xla::ffi;

#define CHECK_CUDA(call) do { cudaError_t err = (call); \
    if (err != cudaSuccess) return xla::ffi::Error::Internal(std::string("CUDA ") + cudaGetErrorString(err)); \
} while(0)
#define CHECK_CUFFT(call) do { cufftResult res = (call); \
    if (res != CUFFT_SUCCESS) return xla::ffi::Error::Internal(std::string("cuFFT code ") + std::to_string((int)res)); \
} while(0)

xla_ffi::Error CufftGraphBracketTrueFp32Impl(
    cudaStream_t stream,
    xla_ffi::Buffer<xla_ffi::DataType::C128> df,
    xla_ffi::Buffer<xla_ffi::DataType::C128> phi,
    xla_ffi::Buffer<xla_ffi::DataType::F64>  kx,
    xla_ffi::Buffer<xla_ffi::DataType::F64>  ky,
    xla_ffi::Buffer<xla_ffi::DataType::S32>  jind,
    xla_ffi::Buffer<xla_ffi::DataType::S32>  inverse_jind,
    xla_ffi::Buffer<xla_ffi::DataType::F64>  dum_s,
    xla_ffi::Result<xla_ffi::Buffer<xla_ffi::DataType::C128>> out,
    int32_t batch, int32_t mrad, int32_t mphi, int32_t nkx, int32_t nky, int32_t nspec,
    int32_t ixzero, int32_t iyzero)
{
    int device = 0; cudaGetDevice(&device);
    int b_df = batch * nspec;
    size_t phi_elems = 1; for (auto d : phi.dimensions()) phi_elems *= d;
    int b_phi = (int)(phi_elems / ((size_t)nkx * nky));
    int mphi_half = mphi / 2 + 1;

    V5TrueFp32Key key = {device, b_df, b_phi, mrad, mphi, nkx, nky};
    std::lock_guard<std::mutex> lock(g_true_fp32_mutex);
    V5TrueFp32State* s = g_true_fp32_cache[key];
    size_t z_dist = (size_t)mrad * mphi;
    size_t c_dist = (size_t)mrad * mphi_half;

    if (!s) {
        s = new V5TrueFp32State();
        g_true_fp32_cache[key] = s;
        CHECK_CUDA(cudaMalloc(&s->ws_c2c,     (size_t)(b_df + b_phi) * z_dist * sizeof(float2)));
        CHECK_CUDA(cudaMalloc(&s->ws_r2c_out,  (size_t)b_df * c_dist * sizeof(float2)));
        CHECK_CUDA(cudaMalloc(&s->d_z2z_cb,    sizeof(V5Z2zInfo)));    s->d_z2z_ptr    = (void*)s->d_z2z_cb;
        CHECK_CUDA(cudaMalloc(&s->d_d2z_cb,    sizeof(V5D2zFp32Info))); s->d_d2z_ptr    = (void*)s->d_d2z_cb;
        CHECK_CUDA(cudaMalloc(&s->d_store_cb,   sizeof(V5StoreInfo)));  s->d_store_ptr  = (void*)s->d_store_cb;

        long long n_ll[2] = {mrad, mphi}; size_t ws = 0;
        CHECK_CUFFT(cufftCreate(&s->plan_c2c));
        CHECK_CUFFT(cufftXtSetJITCallback(s->plan_c2c, "d_v5_z2z_true_fp32_load",
            (void*)bracket_v5_z2z_load_cb_fatbin, bracket_v5_z2z_load_cb_fatbin_bytes,
            CUFFT_CB_LD_COMPLEX, &s->d_z2z_ptr));
        CHECK_CUFFT(cufftXtMakePlanMany(s->plan_c2c, 2, n_ll, NULL, 1, (long long)z_dist, CUDA_C_32F,
            NULL, 1, (long long)z_dist, CUDA_C_32F, b_df + b_phi, &ws, CUDA_C_32F));

        CHECK_CUFFT(cufftCreate(&s->plan_r2c));
        CHECK_CUFFT(cufftXtSetJITCallback(s->plan_r2c, "d_v5_d2z_fp32_load",
            (void*)bracket_v5_d2z_load_cb_fatbin, bracket_v5_d2z_load_cb_fatbin_bytes,
            CUFFT_CB_LD_REAL, &s->d_d2z_ptr));
        CHECK_CUFFT(cufftXtSetJITCallback(s->plan_r2c, "d_v5_store_fp32_cb",
            (void*)bracket_v5_store_cb_fatbin, bracket_v5_store_cb_fatbin_bytes,
            CUFFT_CB_ST_COMPLEX, &s->d_store_ptr));
        CHECK_CUFFT(cufftXtMakePlanMany(s->plan_r2c, 2, n_ll, NULL, 1, (long long)z_dist, CUDA_R_32F,
            NULL, 1, (long long)c_dist, CUDA_C_32F, b_df, &ws, CUDA_R_32F));
    }

    CHECK_CUFFT(cufftSetStream(s->plan_c2c, stream));
    CHECK_CUFFT(cufftSetStream(s->plan_r2c, stream));
    CHECK_CUDA(cudaMemsetAsync(out->typed_data(), 0, (size_t)b_df * nkx * nky * sizeof(double2), stream));

    float inv_n2 = 1.0f / ((float)mrad * mphi * (float)mrad * mphi);
    V5Z2zInfo h_z2z = {(const double2*)df.typed_data(), (const double2*)phi.typed_data(),
        kx.typed_data(), ky.typed_data(), inverse_jind.typed_data(), mrad, mphi, nkx, nky, b_df, b_phi};
    V5D2zFp32Info h_d2z = {s->ws_c2c, dum_s.typed_data(), nspec, mrad, mphi, b_df, b_phi, inv_n2};
    V5StoreInfo h_store = {(double2*)out->typed_data(), inverse_jind.typed_data(), mrad, mphi_half, nkx, nky, ixzero, iyzero};

    CHECK_CUDA(cudaMemcpyAsync(s->d_z2z_cb,    &h_z2z,    sizeof(V5Z2zInfo),    cudaMemcpyHostToDevice, stream));
    CHECK_CUDA(cudaMemcpyAsync(s->d_d2z_cb,    &h_d2z,    sizeof(V5D2zFp32Info), cudaMemcpyHostToDevice, stream));
    CHECK_CUDA(cudaMemcpyAsync(s->d_store_cb,   &h_store,   sizeof(V5StoreInfo),   cudaMemcpyHostToDevice, stream));

    CHECK_CUFFT(cufftExecC2C(s->plan_c2c, (cufftComplex*)s->ws_c2c, (cufftComplex*)s->ws_c2c, CUFFT_INVERSE));
    CHECK_CUFFT(cufftExecR2C(s->plan_r2c, (cufftReal*)s->ws_c2c, (cufftComplex*)s->ws_r2c_out));

    return xla_ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(cufft_graph_bracket_true_fp32_ffi, CufftGraphBracketTrueFp32Impl,
    xla_ffi::Ffi::Bind()
        .Ctx<xla_ffi::PlatformStream<cudaStream_t>>()
        .Arg<xla_ffi::Buffer<xla_ffi::DataType::C128>>()
        .Arg<xla_ffi::Buffer<xla_ffi::DataType::C128>>()
        .Arg<xla_ffi::Buffer<xla_ffi::DataType::F64>>()
        .Arg<xla_ffi::Buffer<xla_ffi::DataType::F64>>()
        .Arg<xla_ffi::Buffer<xla_ffi::DataType::S32>>()
        .Arg<xla_ffi::Buffer<xla_ffi::DataType::S32>>()
        .Arg<xla_ffi::Buffer<xla_ffi::DataType::F64>>()
        .Ret<xla_ffi::Buffer<xla_ffi::DataType::C128>>()
        .Attr<int32_t>("batch").Attr<int32_t>("mrad").Attr<int32_t>("mphi")
        .Attr<int32_t>("nkx").Attr<int32_t>("nky").Attr<int32_t>("nspec")
        .Attr<int32_t>("ixzero").Attr<int32_t>("iyzero"));
