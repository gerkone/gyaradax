// v5-FP64: Z2Z 2-for-1 + phi broadcast, full double precision.
//
// Identical pipeline to cufft_graph_bracket.cu (v5) but with the FP32
// C2C inverse replaced by FP64 Z2Z throughout.  No precision loss anywhere.
//
// Pipeline (5 launches, 2 cuFFT calls):
//   1a/b. v5fp64_pack_z2z   (FP64 → FP64, no cast)
//   2.    cufftExecZ2Z       (FP64)
//   3.    v5fp64_assembly    (FP64 bracket, FP64 in/out)
//   4.    cufftExecD2Z       (FP64)
//   5.    v5fp64_unpack      (FP64)
//
// vs v5 mixed-precision:
//   - Z2Z doubles the inverse FFT I/O vs C2C  → slower FFT step
//   - pack kernel drops the float cast         → no precision loss
//   - expected: ~1.6x faster than JAX FP64, ~1.7x slower than v5 MP

#include <cstdio>
#include <string>
#include <mutex>
#include <map>
#include "xla/ffi/api/ffi.h"
#include <cuda_runtime.h>
#include <cufft.h>

namespace {

struct V5Fp64Key {
    int device, b_df, b_phi, mrad, mphi, nkx, nky;
    bool operator<(const V5Fp64Key& o) const {
        if (device != o.device) return device < o.device;
        if (b_df   != o.b_df)   return b_df   < o.b_df;
        if (b_phi  != o.b_phi)  return b_phi  < o.b_phi;
        if (mrad   != o.mrad)   return mrad   < o.mrad;
        if (mphi   != o.mphi)   return mphi   < o.mphi;
        if (nkx    != o.nkx)    return nkx    < o.nkx;
        return nky < o.nky;
    }
};

struct V5Fp64State {
    cufftHandle plan_z2z = 0;    // FP64 Z2Z, b_df + b_phi (merged, in-place)
    cufftHandle plan_d2z = 0;    // FP64 D2Z, b_df

    double2 *ws_z2z  = nullptr;  // [(b_df+b_phi), mrad, mphi] FP64 complex
    double  *ws_nl_r = nullptr;  // [b_df, mrad, mphi] FP64 real
    double2 *ws_nl_k = nullptr;  // [b_df, mrad, mphi_half] FP64 complex

    ~V5Fp64State() {
        if (plan_z2z) cufftDestroy(plan_z2z);
        if (plan_d2z) cufftDestroy(plan_d2z);
        if (ws_z2z)  cudaFree(ws_z2z);
        if (ws_nl_r) cudaFree(ws_nl_r);
        if (ws_nl_k) cudaFree(ws_nl_k);
    }
};

static std::map<V5Fp64Key, V5Fp64State*> g_fp64_cache;
static std::mutex g_fp64_mutex;
static constexpr int KX_TILE = 4;

// ── Pack: FP64 input → FP64 Z2Z workspace (no precision loss) ───
// Identical to v5_pack_z2z_kernel but stores double2 instead of float2.
__global__ void v5fp64_pack_z2z_kernel(
    const double2* __restrict__ field,   // [batch, nkx, nky] FP64
    const double*  __restrict__ kx,
    const double*  __restrict__ ky,
    const int*     __restrict__ inverse_jind,
    double2* __restrict__ out,           // [batch, mrad, mphi] FP64
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

    if (j_src >= nky) { out[out_idx] = {0.0, 0.0}; return; }

    int kx_idx = __ldg(&inverse_jind[m_src]);
    if (kx_idx < 0) { out[out_idx] = {0.0, 0.0}; return; }

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

    // Pack ws = fy + i*fx, store as FP64 (no cast to float)
    if (!mirror) {
        out[out_idx] = {fy_re - fx_im, fy_im + fx_re};
    } else {
        out[out_idx] = {fy_re + fx_im, fx_re - fy_im};
    }
}

// ── Assembly: FP64 Z2Z output → FP64 bracket ─────────────────────
// No promotion needed: inputs are already FP64.
__global__ void v5fp64_assembly_z2z_kernel(
    const double2* __restrict__ ws_df,
    const double2* __restrict__ ws_phi,
    double* __restrict__ nl,
    const double* __restrict__ dum_s,
    int plane, int b_phi, int nspec, double scale
) {
    int b   = blockIdx.x;
    int off = blockIdx.y * blockDim.x + threadIdx.x;
    if (off >= plane) return;

    double2 d = ws_df[(size_t)b * plane + off];
    double2 p = ws_phi[(size_t)(b % b_phi) * plane + off];

    double bracket = p.x * d.y - p.y * d.x;
    nl[(size_t)b * plane + off] = scale * __ldg(&dum_s[b % nspec]) * bracket;
}

// ── Unpack (FP64, identical to v5) ────────────────────────────────
__global__ void v5fp64_unpack_kernel(
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

xla_ffi::Error CufftGraphBracketFp64Impl(
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

    V5Fp64Key key = {device, b_df, b_phi, mrad, mphi, nkx, nky};
    std::lock_guard<std::mutex> lock(g_fp64_mutex);
    V5Fp64State* s = g_fp64_cache[key];

    size_t z_dist = (size_t)mrad * mphi;       // complex elements per Z2Z transform
    size_t c_dist = (size_t)mrad * mphi_half;  // complex elements per D2Z output
    size_t r_dist = (size_t)mrad * mphi;       // real elements per D2Z input

    if (!s) {
        s = new V5Fp64State();
        g_fp64_cache[key] = s;

        // FP64 workspace: 2x the memory of FP32 C2C
        CHECK_CUDA(cudaMalloc(&s->ws_z2z,  (size_t)(b_df + b_phi) * z_dist * sizeof(double2)));
        CHECK_CUDA(cudaMalloc(&s->ws_nl_r, (size_t)b_df * r_dist * sizeof(double)));
        CHECK_CUDA(cudaMalloc(&s->ws_nl_k, (size_t)b_df * c_dist * sizeof(double2)));

        int n[2] = {mrad, mphi};

        // FP64 Z2Z plan (replaces FP32 C2C)
        CHECK_CUFFT(cufftCreate(&s->plan_z2z));
        CHECK_CUFFT(cufftPlanMany(&s->plan_z2z, 2, n,
            NULL, 1, (int)z_dist, NULL, 1, (int)z_dist,
            CUFFT_Z2Z, b_df + b_phi));

        // FP64 D2Z for forward FFT
        CHECK_CUFFT(cufftCreate(&s->plan_d2z));
        CHECK_CUFFT(cufftPlanMany(&s->plan_d2z, 2, n,
            NULL, 1, (int)r_dist, NULL, 1, (int)c_dist,
            CUFFT_D2Z, b_df));
    }

    CHECK_CUFFT(cufftSetStream(s->plan_z2z, stream));
    CHECK_CUFFT(cufftSetStream(s->plan_d2z, stream));

    dim3 blk(32, KX_TILE);
    int m_blks = (mrad + KX_TILE - 1) / KX_TILE;
    int j_blks = (mphi + 31) / 32;

    double2* ws_df  = s->ws_z2z;
    double2* ws_phi = s->ws_z2z + (size_t)b_df * z_dist;

    // ── 1a. Pack df (FP64, no cast) ──────────────────────────────
    v5fp64_pack_z2z_kernel<<<dim3(b_df, m_blks, j_blks), blk, 0, stream>>>(
        (const double2*)df.typed_data(), kx.typed_data(), ky.typed_data(),
        inverse_jind.typed_data(), ws_df, mrad, mphi, nkx, nky);

    // ── 1b. Pack phi ─────────────────────────────────────────────
    v5fp64_pack_z2z_kernel<<<dim3(b_phi, m_blks, j_blks), blk, 0, stream>>>(
        (const double2*)phi.typed_data(), kx.typed_data(), ky.typed_data(),
        inverse_jind.typed_data(), ws_phi, mrad, mphi, nkx, nky);

    // ── 2. FP64 Z2Z inverse (in-place) ───────────────────────────
    CHECK_CUFFT(cufftExecZ2Z(s->plan_z2z,
        (cufftDoubleComplex*)s->ws_z2z,
        (cufftDoubleComplex*)s->ws_z2z,
        CUFFT_INVERSE));

    // ── 3. Assembly: FP64 bracket ────────────────────────────────
    // Z2Z unnormalized: output *= N. Combined with D2Z: scale = 1/N^2.
    double inv_n2 = 1.0 / ((double)mrad * mphi * (double)mrad * mphi);
    int plane = (int)z_dist;
    v5fp64_assembly_z2z_kernel<<<dim3(b_df, (plane + 255) / 256), 256, 0, stream>>>(
        ws_df, ws_phi, s->ws_nl_r, dum_s.typed_data(),
        plane, b_phi, nspec, inv_n2);

    // ── 4. FP64 forward D2Z ──────────────────────────────────────
    CHECK_CUFFT(cufftExecD2Z(s->plan_d2z, s->ws_nl_r, s->ws_nl_k));

    // ── 5. Unpack ─────────────────────────────────────────────────
    int kx_blks = (nkx + KX_TILE - 1) / KX_TILE;
    v5fp64_unpack_kernel<<<dim3(b_df, kx_blks), blk, 0, stream>>>(
        s->ws_nl_k, (double2*)out->typed_data(), jind.typed_data(),
        mrad, mphi_half, nkx, nky, ixzero, iyzero);

    return xla_ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    cufft_graph_bracket_fp64_ffi, CufftGraphBracketFp64Impl,
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
