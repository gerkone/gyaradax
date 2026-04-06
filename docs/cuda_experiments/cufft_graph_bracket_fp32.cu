// v5-FP32: Pure FP32 pipeline matching JAX Z2Z fp32 behavior.
//
// Entire pipeline is two cuFFT calls:
//   1. cufftExecC2C (FP32, merged b_df+b_phi)
//      - Load callback fuses pack: Hermitian gather, fy+i*fx, FP64->FP32
//      - In-place on ws_c2c
//   2. cufftExecR2C (FP32, b_df)
//      - Load callback fuses assembly: FP32 bracket, dum_s, 1/N^2
//      - Store callback fuses unpack: scatter to packed (with FP32->FP64 conversion)
//
// This matches JAX's fp32 pipeline: downcast→fp32 compute→upcast at end.

#include <cstdio>
#include <string>
#include <mutex>
#include <map>
#include "xla/ffi/api/ffi.h"
#include <cuda_runtime.h>
#include <cufft.h>
#include <cufftXt.h>

// LTO callback fatbins (defined in bracket_v5_lto_fatbins.cu)
#include "bracket_v5_lto_fatbins_decl.h"

namespace {

// ── Callback info structs ────────────────────────────────────────────
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
    const double* dum_s;  // FP64 input from JAX, cast to FP32 in callback
    int nspec, mrad, mphi, b_df, b_phi;
    float scale;
};

struct V5StoreInfo {
    double2*    out_packed;
    const int*  inverse_jind;
    int mrad, mphiw3, nkx, nky;
    int ixzero, iyzero;
};

struct V5Fp32Key {
    int device, b_df, b_phi, mrad, mphi, nkx, nky;
    bool operator<(const V5Fp32Key& o) const {
        if (device != o.device) return device < o.device;
        if (b_df   != o.b_df)   return b_df   < o.b_df;
        if (b_phi  != o.b_phi)  return b_phi  < o.b_phi;
        if (mrad   != o.mrad)   return mrad   < o.mrad;
        if (mphi   != o.mphi)   return mphi   < o.mphi;
        if (nkx    != o.nkx)    return nkx    < o.nkx;
        return nky < o.nky;
    }
};

struct V5Fp32State {
    cufftHandle plan_c2c = 0;    // FP32 C2C, b_df+b_phi, with load callback
    cufftHandle plan_r2c = 0;    // FP32 R2C, b_df, with load + store callbacks

    float2  *ws_c2c     = nullptr;  // [(b_df+b_phi), mrad, mphi]
    float2  *ws_r2c_out = nullptr;  // [b_df, mrad, mphi_half] dummy R2C output

    V5Z2zInfo    *d_z2z_cb    = nullptr;  void *d_z2z_ptr    = nullptr;
    V5D2zFp32Info *d_d2z_cb    = nullptr;  void *d_d2z_ptr    = nullptr;
    V5StoreInfo  *d_store_cb  = nullptr;  void *d_store_ptr  = nullptr;

    ~V5Fp32State() {
        if (plan_c2c) cufftDestroy(plan_c2c);
        if (plan_r2c) cufftDestroy(plan_r2c);
        if (ws_c2c)     cudaFree(ws_c2c);
        if (ws_r2c_out) cudaFree(ws_r2c_out);
        if (d_z2z_cb)   cudaFree(d_z2z_cb);
        if (d_d2z_cb)   cudaFree(d_d2z_cb);
        if (d_store_cb) cudaFree(d_store_cb);
    }
};

static std::map<V5Fp32Key, V5Fp32State*> g_fp32_cache;
static std::mutex g_fp32_mutex;

} // namespace

namespace xla_ffi = xla::ffi;

#define CHECK_CUDA(call) do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) return xla::ffi::Error::Internal( \
        std::string("CUDA ") + cudaGetErrorString(err)); \
} while(0)
#define CHECK_CUFFT(call) do { \
    cufftResult res = (call); \
    if (res != CUFFT_SUCCESS) return xla::ffi::Error::Internal( \
        std::string("cuFFT code ") + std::to_string((int)res)); \
} while(0)

xla_ffi::Error CufftGraphBracketFp32Impl(
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
    int32_t ixzero, int32_t iyzero
) {
    int device = 0;
    cudaGetDevice(&device);
    int b_df = batch * nspec;
    size_t phi_elems = 1;
    for (auto d : phi.dimensions()) phi_elems *= d;
    int b_phi = (int)(phi_elems / ((size_t)nkx * nky));
    int mphi_half = mphi / 2 + 1;

    V5Fp32Key key = {device, b_df, b_phi, mrad, mphi, nkx, nky};
    std::lock_guard<std::mutex> lock(g_fp32_mutex);
    V5Fp32State* s = g_fp32_cache[key];

    size_t z_dist = (size_t)mrad * mphi;
    size_t c_dist = (size_t)mrad * mphi_half;

    if (!s) {
        s = new V5Fp32State();
        g_fp32_cache[key] = s;

        // Workspaces
        CHECK_CUDA(cudaMalloc(&s->ws_c2c,     (size_t)(b_df + b_phi) * z_dist * sizeof(float2)));
        CHECK_CUDA(cudaMalloc(&s->ws_r2c_out,  (size_t)b_df * c_dist * sizeof(float2)));

        // Callback info structs (device)
        CHECK_CUDA(cudaMalloc(&s->d_z2z_cb,    sizeof(V5Z2zInfo)));    s->d_z2z_ptr    = (void*)s->d_z2z_cb;
        CHECK_CUDA(cudaMalloc(&s->d_d2z_cb,    sizeof(V5D2zFp32Info))); s->d_d2z_ptr    = (void*)s->d_d2z_cb;
        CHECK_CUDA(cudaMalloc(&s->d_store_cb,   sizeof(V5StoreInfo)));  s->d_store_ptr  = (void*)s->d_store_cb;

        long long n_ll[2] = {mrad, mphi};
        size_t ws = 0;

        // C2C plan with FP32 load callback
        CHECK_CUFFT(cufftCreate(&s->plan_c2c));
        CHECK_CUFFT(cufftXtSetJITCallback(s->plan_c2c,
            "d_v5_z2z_fp32_load",
            (void*)bracket_v5_z2z_load_cb_fatbin,
            bracket_v5_z2z_load_cb_fatbin_bytes,
            CUFFT_CB_LD_COMPLEX, &s->d_z2z_ptr));
        CHECK_CUFFT(cufftXtMakePlanMany(s->plan_c2c, 2, n_ll,
            NULL, 1, (long long)z_dist, CUDA_C_32F,
            NULL, 1, (long long)z_dist, CUDA_C_32F,
            b_df + b_phi, &ws, CUDA_C_32F));

        // R2C plan with FP32 load + store callbacks
        CHECK_CUFFT(cufftCreate(&s->plan_r2c));
        CHECK_CUFFT(cufftXtSetJITCallback(s->plan_r2c,
            "d_v5_d2z_fp32_load",
            (void*)bracket_v5_d2z_load_cb_fatbin,
            bracket_v5_d2z_load_cb_fatbin_bytes,
            CUFFT_CB_LD_REAL, &s->d_d2z_ptr));
        CHECK_CUFFT(cufftXtSetJITCallback(s->plan_r2c,
            "d_v5_store_fp32_cb",
            (void*)bracket_v5_store_cb_fatbin,
            bracket_v5_store_cb_fatbin_bytes,
            CUFFT_CB_ST_COMPLEX, &s->d_store_ptr));
        CHECK_CUFFT(cufftXtMakePlanMany(s->plan_r2c, 2, n_ll,
            NULL, 1, (long long)z_dist, CUDA_R_32F,
            NULL, 1, (long long)c_dist, CUDA_C_32F,
            b_df, &ws, CUDA_R_32F));
    }

    CHECK_CUFFT(cufftSetStream(s->plan_c2c, stream));
    CHECK_CUFFT(cufftSetStream(s->plan_r2c, stream));

    // Zero output buffer (store callback only writes valid entries)
    CHECK_CUDA(cudaMemsetAsync(out->typed_data(), 0,
        (size_t)b_df * nkx * nky * sizeof(double2), stream));

    // Update callback info structs (pointers change each FFI call)
    float inv_n2 = 1.0f / ((float)mrad * mphi * (float)mrad * mphi);

    V5Z2zInfo h_z2z = {
        (const double2*)df.typed_data(),
        (const double2*)phi.typed_data(),
        kx.typed_data(), ky.typed_data(),
        inverse_jind.typed_data(),
        mrad, mphi, nkx, nky, b_df, b_phi
    };
    
    V5D2zFp32Info h_d2z = {
        s->ws_c2c, 
        dum_s.typed_data(),  // FP64 pointer, cast to FP32 in callback
        nspec, mrad, mphi, b_df, b_phi, inv_n2
    };
    
    V5StoreInfo h_store = {
        (double2*)out->typed_data(),
        inverse_jind.typed_data(),
        mrad, mphi_half, nkx, nky,
        ixzero, iyzero
    };

    CHECK_CUDA(cudaMemcpyAsync(s->d_z2z_cb,    &h_z2z,    sizeof(V5Z2zInfo),    cudaMemcpyHostToDevice, stream));
    CHECK_CUDA(cudaMemcpyAsync(s->d_d2z_cb,    &h_d2z,    sizeof(V5D2zFp32Info), cudaMemcpyHostToDevice, stream));
    CHECK_CUDA(cudaMemcpyAsync(s->d_store_cb,   &h_store,   sizeof(V5StoreInfo),   cudaMemcpyHostToDevice, stream));

    // ── 1. FP32 C2C inverse with load callback (fuses pack) ────────
    CHECK_CUFFT(cufftExecC2C(s->plan_c2c,
        (cufftComplex*)s->ws_c2c, (cufftComplex*)s->ws_c2c, CUFFT_INVERSE));

    // ── 2. FP32 R2C with load+store callbacks (fuses assembly+unpack)
    CHECK_CUFFT(cufftExecR2C(s->plan_r2c,
        (cufftReal*)s->ws_c2c, (cufftComplex*)s->ws_r2c_out));

    return xla_ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    cufft_graph_bracket_fp32_ffi, CufftGraphBracketFp32Impl,
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
        .Attr<int32_t>("batch")
        .Attr<int32_t>("mrad")
        .Attr<int32_t>("mphi")
        .Attr<int32_t>("nkx")
        .Attr<int32_t>("nky")
        .Attr<int32_t>("nspec")
        .Attr<int32_t>("ixzero")
        .Attr<int32_t>("iyzero")
);
