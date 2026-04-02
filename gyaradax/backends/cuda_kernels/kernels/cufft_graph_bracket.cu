// v5 LTO: Z2Z 2-for-1 + phi broadcast + mixed precision + LTO callbacks.
//
// Entire pipeline is two cuFFT calls:
//   1. cufftExecC2C (FP32, merged b_df+b_phi)
//      - Load callback fuses pack: Hermitian gather, fy+i*fx, FP64->FP32
//      - In-place on ws_c2c
//   2. cufftExecD2Z (FP64, b_df)
//      - Load callback fuses assembly: FP32->FP64 bracket, dum_s, 1/N^2
//      - Store callback fuses unpack: scatter to packed, ixzero masking
//
// Eliminated vs non-LTO v5: pack kernel, assembly kernel, unpack kernel, ws_nl_r.
// HBM savings: ~b_df * mrad * mphi * 8 bytes (ws_nl_r eliminated).

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

// ── Callback info structs (must match callback .cu definitions) ────
struct V5Z2zInfo {
    const double2* df_packed;
    const double2* phi_packed;
    const double*  kx;
    const double*  ky;
    const int*     inverse_jind;
    int mrad, mphi, nkx, nky, b_df, b_phi;
};

struct V5D2zMpInfo {
    const float2* ws;
    const double* dum_s;
    int nspec, mrad, mphi, b_df, b_phi;
    double scale;
};

struct V5StoreInfo {
    double2*    out_packed;
    const int*  inverse_jind;
    int mrad, mphiw3, nkx, nky;
    int ixzero, iyzero;
};

struct V5Key {
    int device, b_df, b_phi, mrad, mphi, nkx, nky;
    bool operator<(const V5Key& o) const {
        if (device != o.device) return device < o.device;
        if (b_df   != o.b_df)   return b_df   < o.b_df;
        if (b_phi  != o.b_phi)  return b_phi  < o.b_phi;
        if (mrad   != o.mrad)   return mrad   < o.mrad;
        if (mphi   != o.mphi)   return mphi   < o.mphi;
        if (nkx    != o.nkx)    return nkx    < o.nkx;
        return nky < o.nky;
    }
};

struct V5State {
    cufftHandle plan_c2c = 0;    // FP32 C2C, b_df+b_phi, with load callback
    cufftHandle plan_d2z = 0;    // FP64 D2Z, b_df, with load + store callbacks

    float2  *ws_c2c     = nullptr;  // [(b_df+b_phi), mrad, mphi]
    double2 *ws_d2z_out = nullptr;  // [b_df, mrad, mphi_half] dummy D2Z output

    V5Z2zInfo   *d_z2z_cb   = nullptr;  void *d_z2z_ptr   = nullptr;
    V5D2zMpInfo *d_d2z_cb   = nullptr;  void *d_d2z_ptr   = nullptr;
    V5StoreInfo *d_store_cb = nullptr;  void *d_store_ptr = nullptr;

    ~V5State() {
        if (plan_c2c) cufftDestroy(plan_c2c);
        if (plan_d2z) cufftDestroy(plan_d2z);
        if (ws_c2c)     cudaFree(ws_c2c);
        if (ws_d2z_out) cudaFree(ws_d2z_out);
        if (d_z2z_cb)   cudaFree(d_z2z_cb);
        if (d_d2z_cb)   cudaFree(d_d2z_cb);
        if (d_store_cb) cudaFree(d_store_cb);
    }
};

static std::map<V5Key, V5State*> g_cache;
static std::mutex g_mutex;

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

xla_ffi::Error CufftGraphBracketImpl(
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

    V5Key key = {device, b_df, b_phi, mrad, mphi, nkx, nky};
    std::lock_guard<std::mutex> lock(g_mutex);
    V5State* s = g_cache[key];

    size_t z_dist = (size_t)mrad * mphi;
    size_t c_dist = (size_t)mrad * mphi_half;

    if (!s) {
        s = new V5State();
        g_cache[key] = s;

        // Workspaces
        CHECK_CUDA(cudaMalloc(&s->ws_c2c,     (size_t)(b_df + b_phi) * z_dist * sizeof(float2)));
        CHECK_CUDA(cudaMalloc(&s->ws_d2z_out,  (size_t)b_df * c_dist * sizeof(double2)));

        // Callback info structs (device)
        CHECK_CUDA(cudaMalloc(&s->d_z2z_cb,   sizeof(V5Z2zInfo)));   s->d_z2z_ptr   = (void*)s->d_z2z_cb;
        CHECK_CUDA(cudaMalloc(&s->d_d2z_cb,   sizeof(V5D2zMpInfo))); s->d_d2z_ptr   = (void*)s->d_d2z_cb;
        CHECK_CUDA(cudaMalloc(&s->d_store_cb,  sizeof(V5StoreInfo))); s->d_store_ptr = (void*)s->d_store_cb;

        long long n_ll[2] = {mrad, mphi};
        size_t ws = 0;

        // C2C plan with Z2Z FP32 load callback
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

        // D2Z plan with load + store callbacks
        CHECK_CUFFT(cufftCreate(&s->plan_d2z));
        CHECK_CUFFT(cufftXtSetJITCallback(s->plan_d2z,
            "d_v5_d2z_mp_load",
            (void*)bracket_v5_d2z_load_cb_fatbin,
            bracket_v5_d2z_load_cb_fatbin_bytes,
            CUFFT_CB_LD_REAL_DOUBLE, &s->d_d2z_ptr));
        CHECK_CUFFT(cufftXtSetJITCallback(s->plan_d2z,
            "d_v5_store_cb",
            (void*)bracket_v5_store_cb_fatbin,
            bracket_v5_store_cb_fatbin_bytes,
            CUFFT_CB_ST_COMPLEX_DOUBLE, &s->d_store_ptr));
        CHECK_CUFFT(cufftXtMakePlanMany(s->plan_d2z, 2, n_ll,
            NULL, 1, (long long)z_dist, CUDA_R_64F,
            NULL, 1, (long long)c_dist, CUDA_C_64F,
            b_df, &ws, CUDA_R_64F));
    }

    CHECK_CUFFT(cufftSetStream(s->plan_c2c, stream));
    CHECK_CUFFT(cufftSetStream(s->plan_d2z, stream));

    // Zero output buffer (store callback only writes valid entries)
    CHECK_CUDA(cudaMemsetAsync(out->typed_data(), 0,
        (size_t)b_df * nkx * nky * sizeof(double2), stream));

    // Update callback info structs (pointers change each FFI call)
    double inv_n2 = 1.0 / ((double)mrad * mphi * (double)mrad * mphi);

    V5Z2zInfo h_z2z = {
        (const double2*)df.typed_data(),
        (const double2*)phi.typed_data(),
        kx.typed_data(), ky.typed_data(),
        inverse_jind.typed_data(),
        mrad, mphi, nkx, nky, b_df, b_phi
    };
    V5D2zMpInfo h_d2z = {
        s->ws_c2c, dum_s.typed_data(),
        nspec, mrad, mphi, b_df, b_phi, inv_n2
    };
    V5StoreInfo h_store = {
        (double2*)out->typed_data(),
        inverse_jind.typed_data(),
        mrad, mphi_half, nkx, nky,
        ixzero, iyzero
    };

    CHECK_CUDA(cudaMemcpyAsync(s->d_z2z_cb,   &h_z2z,   sizeof(V5Z2zInfo),   cudaMemcpyHostToDevice, stream));
    CHECK_CUDA(cudaMemcpyAsync(s->d_d2z_cb,   &h_d2z,   sizeof(V5D2zMpInfo), cudaMemcpyHostToDevice, stream));
    CHECK_CUDA(cudaMemcpyAsync(s->d_store_cb,  &h_store,  sizeof(V5StoreInfo), cudaMemcpyHostToDevice, stream));

    // ── 1. FP32 C2C inverse with load callback (fuses pack) ────────
    CHECK_CUFFT(cufftExecC2C(s->plan_c2c,
        (cufftComplex*)s->ws_c2c, (cufftComplex*)s->ws_c2c, CUFFT_INVERSE));

    // ── 2. FP64 D2Z with load+store callbacks (fuses assembly+unpack)
    CHECK_CUFFT(cufftExecD2Z(s->plan_d2z,
        (cufftDoubleReal*)s->ws_c2c, (cufftDoubleComplex*)s->ws_d2z_out));

    return xla_ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    cufft_graph_bracket_ffi, CufftGraphBracketImpl,
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
