#include <cstdio>
#include <string>
#include <mutex>
#include <map>
#include "xla/ffi/api/ffi.h"
#include <cuda_runtime.h>
#include <cufft.h>

// v5: Z2Z 2-for-1 + phi broadcast + mixed precision.
//
// Mixed precision: FP32 inverse C2C (halves FFT I/O bandwidth),
// FP64 bracket accumulation + forward D2Z (preserves spectral accuracy).
// This matches JAX's mixed_precision=True path.
//
// Pipeline (5 launches, 2 cuFFT calls):
//   1a/b. v5_pack_z2z  (FP64→FP32 cast on output)
//   2.    cufftExecC2C  (FP32, merged b_df+b_phi, in-place)
//   3.    v5_assembly   (FP32→FP64 bracket, writes FP64)
//   4.    cufftExecD2Z  (FP64)
//   5.    v5_unpack     (FP64)

namespace {

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
    cufftHandle plan_c2c = 0;   // FP32 C2C, b_df + b_phi (merged, in-place)
    cufftHandle plan_d2z = 0;   // FP64 D2Z, b_df

    float2  *ws_c2c  = nullptr;  // [(b_df+b_phi), mrad, mphi] FP32 complex
    double  *ws_nl_r = nullptr;  // [b_df, mrad, mphi] FP64 real
    double2 *ws_nl_k = nullptr;  // [b_df, mrad, mphi_half] FP64 complex

    ~V5State() {
        if (plan_c2c) cufftDestroy(plan_c2c);
        if (plan_d2z) cufftDestroy(plan_d2z);
        if (ws_c2c)  cudaFree(ws_c2c);
        if (ws_nl_r) cudaFree(ws_nl_r);
        if (ws_nl_k) cudaFree(ws_nl_k);
    }
};

static std::map<V5Key, V5State*> g_cache;
static std::mutex g_mutex;
static constexpr int KX_TILE = 4;

// ── Pack: FP64 input → FP32 Z2Z workspace ──────────────────────
// Hermitian extension + j=0 symmetrization + 2-for-1 packing.
// All arithmetic in FP64, cast to FP32 on final store.
__global__ void v5_pack_z2z_kernel(
    const double2* __restrict__ field,       // [batch, nkx, nky] FP64
    const double*  __restrict__ kx,
    const double*  __restrict__ ky,
    const int*     __restrict__ inverse_jind,
    float2* __restrict__ out,                // [batch, mrad, mphi] FP32
    int mrad, int mphi, int nkx, int nky
) {
    int b = blockIdx.x;
    int m = blockIdx.y * KX_TILE + threadIdx.y;
    int j = blockIdx.z * blockDim.x + threadIdx.x;
    if (m >= mrad || j >= mphi) return;

    bool mirror = (j > mphi / 2);
    int j_src = mirror ? (mphi - j) : j;
    int m_src = mirror ? ((mrad - m) % mrad) : m;

    size_t out_idx = (size_t)b * mrad * mphi + (size_t)m * mphi + j;

    if (j_src >= nky) { out[out_idx] = {0.0f, 0.0f}; return; }

    int kx_idx = __ldg(&inverse_jind[m_src]);
    if (kx_idx < 0) { out[out_idx] = {0.0f, 0.0f}; return; }

    size_t row = (size_t)nkx * nky;
    double2 val = __ldg(&field[(size_t)b * row + (size_t)kx_idx * nky + j_src]);
    double kxv  = __ldg(&kx[kx_idx]);
    double kyv  = __ldg(&ky[j_src]);

    // Hermitian symmetrization at ky=0
    if ((j_src == 0 || j_src == mphi / 2) && !mirror) {
        int m_pair = (mrad - m) % mrad;
        int kx_pair = __ldg(&inverse_jind[m_pair]);
        if (kx_pair >= 0 && m_pair != m) {
            double2 vp = __ldg(&field[(size_t)b * row + (size_t)kx_pair * nky]);
            val.x = 0.5 * (val.x + vp.x);
            val.y = 0.5 * (val.y - vp.y);
        }
    }

    double fy_re = -kyv * val.y, fy_im = kyv * val.x;
    double fx_re = -kxv * val.y, fx_im = kxv * val.x;

    // Pack ws = fy + i*fx, cast to FP32
    float2 packed;
    if (!mirror) {
        packed = {(float)(fy_re - fx_im), (float)(fy_im + fx_re)};
    } else {
        packed = {(float)(fy_re + fx_im), (float)(fx_re - fy_im)};
    }
    out[out_idx] = packed;
}

// ── Assembly: FP32 C2C output → FP64 bracket ───────────────────
__global__ void v5_assembly_z2z_kernel(
    const float2* __restrict__ ws_df,
    const float2* __restrict__ ws_phi,
    double* __restrict__ nl,
    const double* __restrict__ dum_s,
    int plane, int b_phi, int nspec, double scale
) {
    int b   = blockIdx.x;
    int off = blockIdx.y * blockDim.x + threadIdx.x;
    if (off >= plane) return;

    float2 d = ws_df[(size_t)b * plane + off];
    float2 p = ws_phi[(size_t)(b % b_phi) * plane + off];

    // Promote to FP64 for bracket accumulation
    double bracket = (double)p.x * (double)d.y - (double)p.y * (double)d.x;
    nl[(size_t)b * plane + off] = scale * __ldg(&dum_s[b % nspec]) * bracket;
}

// ── Unpack (FP64, unchanged) ────────────────────────────────────
__global__ void v5_unpack_kernel(
    const double2* __restrict__ nl_dense_k,
    double2* __restrict__ out_packed,
    const int* __restrict__ jind,
    int mrad, int mphi_half, int nkx, int nky,
    int ixzero, int iyzero
) {
    int b  = blockIdx.x;
    int kx = blockIdx.y * KX_TILE + threadIdx.y;
    int ky = threadIdx.x;
    if (kx >= nkx || ky >= nky) return;

    int m = __ldg(&jind[kx]);
    double2 val = {0.0, 0.0};
    if (m >= 0) {
        val = nl_dense_k[((size_t)b * mrad + m) * mphi_half + ky];
        if (kx == ixzero && ky == iyzero) val = {0.0, 0.0};
    }
    out_packed[((size_t)b * nkx + kx) * nky + ky] = val;
}

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

    size_t z_dist = (size_t)mrad * mphi;       // elements per C2C transform
    size_t c_dist = (size_t)mrad * mphi_half;   // complex elements per D2Z output
    size_t r_dist = (size_t)mrad * mphi;         // real elements per D2Z input

    if (!s) {
        s = new V5State();
        g_cache[key] = s;

        // FP32 workspace for C2C (half the size of FP64 Z2Z)
        CHECK_CUDA(cudaMalloc(&s->ws_c2c,  (size_t)(b_df + b_phi) * z_dist * sizeof(float2)));
        // FP64 workspace for assembly output + D2Z
        CHECK_CUDA(cudaMalloc(&s->ws_nl_r, (size_t)b_df * r_dist * sizeof(double)));
        CHECK_CUDA(cudaMalloc(&s->ws_nl_k, (size_t)b_df * c_dist * sizeof(double2)));

        int n[2] = {mrad, mphi};

        // FP32 C2C plan (replaces FP64 Z2Z — half the I/O bandwidth)
        CHECK_CUFFT(cufftCreate(&s->plan_c2c));
        CHECK_CUFFT(cufftPlanMany(&s->plan_c2c, 2, n,
            NULL, 1, (int)z_dist, NULL, 1, (int)z_dist,
            CUFFT_C2C, b_df + b_phi));

        // FP64 D2Z for forward FFT (preserves spectral accuracy)
        CHECK_CUFFT(cufftCreate(&s->plan_d2z));
        CHECK_CUFFT(cufftPlanMany(&s->plan_d2z, 2, n,
            NULL, 1, (int)r_dist, NULL, 1, (int)c_dist,
            CUFFT_D2Z, b_df));
    }

    CHECK_CUFFT(cufftSetStream(s->plan_c2c, stream));
    CHECK_CUFFT(cufftSetStream(s->plan_d2z, stream));

    dim3 blk(32, KX_TILE);
    int m_blks = (mrad + KX_TILE - 1) / KX_TILE;
    int j_blks = (mphi + 31) / 32;

    float2* ws_df  = s->ws_c2c;
    float2* ws_phi = s->ws_c2c + (size_t)b_df * z_dist;

    // ── 1a. Pack df (FP64 input → FP32 output) ─────────────────
    v5_pack_z2z_kernel<<<dim3(b_df, m_blks, j_blks), blk, 0, stream>>>(
        (const double2*)df.typed_data(), kx.typed_data(), ky.typed_data(),
        inverse_jind.typed_data(), ws_df, mrad, mphi, nkx, nky);

    // ── 1b. Pack phi ────────────────────────────────────────────
    v5_pack_z2z_kernel<<<dim3(b_phi, m_blks, j_blks), blk, 0, stream>>>(
        (const double2*)phi.typed_data(), kx.typed_data(), ky.typed_data(),
        inverse_jind.typed_data(), ws_phi, mrad, mphi, nkx, nky);

    // ── 2. FP32 C2C inverse (in-place) ──────────────────────────
    CHECK_CUFFT(cufftExecC2C(s->plan_c2c,
        (cufftComplex*)s->ws_c2c, (cufftComplex*)s->ws_c2c, CUFFT_INVERSE));

    // ── 3. Assembly: FP32→FP64 bracket ──────────────────────────
    // C2C unnormalized: output *= N. Combined with D2Z: scale = 1/N^2.
    double inv_n2 = 1.0 / ((double)mrad * mphi * (double)mrad * mphi);
    int plane = (int)z_dist;
    v5_assembly_z2z_kernel<<<dim3(b_df, (plane+255)/256), 256, 0, stream>>>(
        ws_df, ws_phi, s->ws_nl_r, dum_s.typed_data(),
        plane, b_phi, nspec, inv_n2);

    // ── 4. FP64 forward D2Z ─────────────────────────────────────
    CHECK_CUFFT(cufftExecD2Z(s->plan_d2z, s->ws_nl_r, s->ws_nl_k));

    // ── 5. Unpack ───────────────────────────────────────────────
    int kx_blks = (nkx + KX_TILE - 1) / KX_TILE;
    v5_unpack_kernel<<<dim3(b_df, kx_blks), blk, 0, stream>>>(
        s->ws_nl_k, (double2*)out->typed_data(), jind.typed_data(),
        mrad, mphi_half, nkx, nky, ixzero, iyzero);

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
