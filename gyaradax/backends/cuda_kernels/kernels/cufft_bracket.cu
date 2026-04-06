#include <cstdio>
#include <string>
#include <mutex>
#include <map>
#include "xla/ffi/api/ffi.h"
#include <cuda_runtime.h>
#include <cufft.h>

namespace xla_ffi = xla::ffi;

#define CHECK_CUDA(call) do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) return xla::ffi::Error::Internal( \
        std::string("CUDA ") + cudaGetErrorString(err)); \
} while(0)

#define CHECK_CUFFT(call) do { \
    cufftResult res = (call); \
    if (res != CUFFT_SUCCESS) return xla::ffi::Error::Internal( \
        std::string("cuFFT Error: ") + std::to_string((int)res)); \
} while(0)

namespace {

static constexpr int KX_TILE = 4;

// ── Shared Unpack Kernel ─────────────────────────────────────────────────────
__global__ void bracket_unpack_kernel(
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

// =============================================================================
// Variant 1: v5 Mixed Precision (Z2Z 2-for-1 + phi broadcast)
// =============================================================================

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
    cufftHandle plan_c2c = 0;
    cufftHandle plan_d2z = 0;
    float2  *ws_c2c  = nullptr;
    double  *ws_nl_r = nullptr;
    double2 *ws_nl_k = nullptr;

    ~V5State() {
        if (plan_c2c) cufftDestroy(plan_c2c);
        if (plan_d2z) cufftDestroy(plan_d2z);
        if (ws_c2c)  cudaFree(ws_c2c);
        if (ws_nl_r) cudaFree(ws_nl_r);
        if (ws_nl_k) cudaFree(ws_nl_k);
    }
};

static std::map<V5Key, V5State*> g_v5_cache;
static std::mutex g_v5_mutex;

__global__ void v5_pack_z2z_kernel(
    const double2* __restrict__ field,
    const double*  __restrict__ kx,
    const double*  __restrict__ ky,
    const int*     __restrict__ inverse_jind,
    float2* __restrict__ out,
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
    double kxv = __ldg(&kx[kx_idx]);
    double kyv = __ldg(&ky[j_src]);

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

    float2 packed;
    if (!mirror) packed = {(float)(fy_re - fx_im), (float)(fy_im + fx_re)};
    else         packed = {(float)(fy_re + fx_im), (float)(fx_re - fy_im)};
    out[out_idx] = packed;
}

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

    double bracket = (double)p.x * (double)d.y - (double)p.y * (double)d.x;
    nl[(size_t)b * plane + off] = scale * __ldg(&dum_s[b % nspec]) * bracket;
}

// =============================================================================
// Variant 2: v5-FP64 (Full Double Precision Z2Z)
// =============================================================================

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
    cufftHandle plan_z2z = 0;
    cufftHandle plan_d2z = 0;
    double2 *ws_z2z  = nullptr;
    double  *ws_nl_r = nullptr;
    double2 *ws_nl_k = nullptr;

    ~V5Fp64State() {
        if (plan_z2z) cufftDestroy(plan_z2z);
        if (plan_d2z) cufftDestroy(plan_d2z);
        if (ws_z2z)  cudaFree(ws_z2z);
        if (ws_nl_r) cudaFree(ws_nl_r);
        if (ws_nl_k) cudaFree(ws_nl_k);
    }
};

static std::map<V5Fp64Key, V5Fp64State*> g_v5fp64_cache;
static std::mutex g_v5fp64_mutex;

__global__ void v5fp64_pack_z2z_kernel(
    const double2* __restrict__ field,
    const double*  __restrict__ kx,
    const double*  __restrict__ ky,
    const int*     __restrict__ inverse_jind,
    double2* __restrict__ out,
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
    double kxv = __ldg(&kx[kx_idx]);
    double kyv = __ldg(&ky[j_src]);

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

    if (!mirror) out[out_idx] = {fy_re - fx_im, fy_im + fx_re};
    else         out[out_idx] = {fy_re + fx_im, fx_re - fy_im};
}

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

// =============================================================================
// Variant 3: v6-FP64 (Direct Separate Transforms)
// =============================================================================

struct V6Fp64Key {
    int device, b_df, b_phi, mrad, mphi, nkx, nky;
    bool operator<(const V6Fp64Key& o) const {
        if (device != o.device) return device < o.device;
        if (b_df   != o.b_df)   return b_df   < o.b_df;
        if (b_phi  != o.b_phi)  return b_phi  < o.b_phi;
        if (mrad   != o.mrad)   return mrad   < o.mrad;
        if (mphi   != o.mphi)   return mphi   < o.mphi;
        if (nkx    != o.nkx)    return nkx    < o.nkx;
        return nky < o.nky;
    }
};

struct V6Fp64State {
    cufftHandle plan_fy  = 0;
    cufftHandle plan_fx  = 0;
    cufftHandle plan_d2z = 0;
    double2 *ws_fy  = nullptr;
    double2 *ws_fx  = nullptr;
    double  *ws_nl_r = nullptr;
    double2 *ws_nl_k = nullptr;

    ~V6Fp64State() {
        if (plan_fy)  cufftDestroy(plan_fy);
        if (plan_fx)  cufftDestroy(plan_fx);
        if (plan_d2z) cufftDestroy(plan_d2z);
        if (ws_fy)   cudaFree(ws_fy);
        if (ws_fx)   cudaFree(ws_fx);
        if (ws_nl_r) cudaFree(ws_nl_r);
        if (ws_nl_k) cudaFree(ws_nl_k);
    }
};

static std::map<V6Fp64Key, V6Fp64State*> g_v6fp64_cache;
static std::mutex g_v6fp64_mutex;

__global__ void v6fp64_pack_deriv_kernel(
    const double2* __restrict__ field,
    const double*  __restrict__ kx,
    const double*  __restrict__ ky,
    const int*     __restrict__ inverse_jind,
    double2* __restrict__ out_y,
    double2* __restrict__ out_x,
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
    if (j_src >= nky) { out_y[out_idx] = {0.0, 0.0}; out_x[out_idx] = {0.0, 0.0}; return; }

    int kx_idx = __ldg(&inverse_jind[m_src]);
    if (kx_idx < 0) { out_y[out_idx] = {0.0, 0.0}; out_x[out_idx] = {0.0, 0.0}; return; }

    size_t row = (size_t)nkx * nky;
    double2 val = __ldg(&field[(size_t)b * row + (size_t)kx_idx * nky + j_src]);
    double kxv = __ldg(&kx[kx_idx]);
    double kyv = __ldg(&ky[j_src]);

    if ((j_src == 0 || j_src == mphi / 2) && !mirror) {
        int m_pair  = (mrad - m) % mrad;
        int kx_pair = __ldg(&inverse_jind[m_pair]);
        if (kx_pair >= 0 && m_pair != m) {
            double2 vp = __ldg(&field[(size_t)b * row + (size_t)kx_pair * nky]);
            val.x = 0.5 * (val.x + vp.x);
            val.y = 0.5 * (val.y - vp.y);
        }
    }

    double fy_re = -kyv * val.y, fy_im = kyv * val.x;
    double fx_re = -kxv * val.y, fx_im = kxv * val.x;

    if (!mirror) {
        out_y[out_idx] = {fy_re,  fy_im};
        out_x[out_idx] = {fx_re,  fx_im};
    } else {
        out_y[out_idx] = {fy_re, -fy_im};
        out_x[out_idx] = {fx_re, -fx_im};
    }
}

__global__ void v6fp64_assembly_kernel(
    const double2* __restrict__ ws_fy,
    const double2* __restrict__ ws_fx,
    double* __restrict__ nl,
    const double* __restrict__ dum_s,
    int plane, int b_df, int b_phi, int nspec, double scale
) {
    int b   = blockIdx.x;
    int off = blockIdx.y * blockDim.x + threadIdx.x;
    if (off >= plane) return;

    size_t df_off  = (size_t)b * plane + off;
    size_t phi_off = (size_t)(b_df + b % b_phi) * plane + off;

    double df_fy  = ws_fy[df_off ].x;
    double df_fx  = ws_fx[df_off ].x;
    double phi_fy = ws_fy[phi_off].x;
    double phi_fx = ws_fx[phi_off].x;

    double bracket = phi_fy * df_fx - phi_fx * df_fy;
    nl[(size_t)b * plane + off] = scale * __ldg(&dum_s[b % nspec]) * bracket;
}

} // namespace

// =============================================================================
// XLA FFI Implementation Handlers
// =============================================================================

xla_ffi::Error CufftGraphBracketMpImpl(
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
    int b_phi = (int)(phi.element_count() / ((size_t)nkx * nky));
    int mphi_half = mphi / 2 + 1;

    V5Key key = {device, b_df, b_phi, mrad, mphi, nkx, nky};
    std::lock_guard<std::mutex> lock(g_v5_mutex);
    V5State* s = g_v5_cache[key];

    size_t z_dist = (size_t)mrad * mphi;
    size_t c_dist = (size_t)mrad * mphi_half;
    size_t r_dist = (size_t)mrad * mphi;

    if (!s) {
        s = new V5State();
        g_v5_cache[key] = s;
        CHECK_CUDA(cudaMalloc(&s->ws_c2c,  (size_t)(b_df + b_phi) * z_dist * sizeof(float2)));
        CHECK_CUDA(cudaMalloc(&s->ws_nl_r, (size_t)b_df * r_dist * sizeof(double)));
        CHECK_CUDA(cudaMalloc(&s->ws_nl_k, (size_t)b_df * c_dist * sizeof(double2)));

        int n[2] = {mrad, mphi};
        CHECK_CUFFT(cufftCreate(&s->plan_c2c));
        CHECK_CUFFT(cufftPlanMany(&s->plan_c2c, 2, n, NULL, 1, (int)z_dist, NULL, 1, (int)z_dist, CUFFT_C2C, b_df + b_phi));

        CHECK_CUFFT(cufftCreate(&s->plan_d2z));
        CHECK_CUFFT(cufftPlanMany(&s->plan_d2z, 2, n, NULL, 1, (int)r_dist, NULL, 1, (int)c_dist, CUFFT_D2Z, b_df));
    }

    CHECK_CUFFT(cufftSetStream(s->plan_c2c, stream));
    CHECK_CUFFT(cufftSetStream(s->plan_d2z, stream));

    dim3 blk(32, KX_TILE);
    int m_blks = (mrad + KX_TILE - 1) / KX_TILE;
    int j_blks = (mphi + 31) / 32;

    float2* ws_df  = s->ws_c2c;
    float2* ws_phi = s->ws_c2c + (size_t)b_df * z_dist;

    v5_pack_z2z_kernel<<<dim3(b_df, m_blks, j_blks), blk, 0, stream>>>(
        (const double2*)df.typed_data(), kx.typed_data(), ky.typed_data(), inverse_jind.typed_data(), ws_df, mrad, mphi, nkx, nky);
    v5_pack_z2z_kernel<<<dim3(b_phi, m_blks, j_blks), blk, 0, stream>>>(
        (const double2*)phi.typed_data(), kx.typed_data(), ky.typed_data(), inverse_jind.typed_data(), ws_phi, mrad, mphi, nkx, nky);

    CHECK_CUFFT(cufftExecC2C(s->plan_c2c, (cufftComplex*)s->ws_c2c, (cufftComplex*)s->ws_c2c, CUFFT_INVERSE));

    double inv_n2 = 1.0 / ((double)mrad * mphi * (double)mrad * mphi);
    v5_assembly_z2z_kernel<<<dim3(b_df, ((int)z_dist + 255)/256), 256, 0, stream>>>(
        ws_df, ws_phi, s->ws_nl_r, dum_s.typed_data(), (int)z_dist, b_phi, nspec, inv_n2);

    CHECK_CUFFT(cufftExecD2Z(s->plan_d2z, s->ws_nl_r, s->ws_nl_k));

    bracket_unpack_kernel<<<dim3(b_df, (nkx + KX_TILE - 1) / KX_TILE), blk, 0, stream>>>(
        s->ws_nl_k, (double2*)out->typed_data(), jind.typed_data(), mrad, mphi_half, nkx, nky, ixzero, iyzero);

    return xla_ffi::Error::Success();
}

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
    int b_phi = (int)(phi.element_count() / ((size_t)nkx * nky));
    int mphi_half = mphi / 2 + 1;

    V5Fp64Key key = {device, b_df, b_phi, mrad, mphi, nkx, nky};
    std::lock_guard<std::mutex> lock(g_v5fp64_mutex);
    V5Fp64State* s = g_v5fp64_cache[key];

    size_t z_dist = (size_t)mrad * mphi;
    size_t c_dist = (size_t)mrad * mphi_half;
    size_t r_dist = (size_t)mrad * mphi;

    if (!s) {
        s = new V5Fp64State();
        g_v5fp64_cache[key] = s;
        CHECK_CUDA(cudaMalloc(&s->ws_z2z,  (size_t)(b_df + b_phi) * z_dist * sizeof(double2)));
        CHECK_CUDA(cudaMalloc(&s->ws_nl_r, (size_t)b_df * r_dist * sizeof(double)));
        CHECK_CUDA(cudaMalloc(&s->ws_nl_k, (size_t)b_df * c_dist * sizeof(double2)));

        int n[2] = {mrad, mphi};
        CHECK_CUFFT(cufftCreate(&s->plan_z2z));
        CHECK_CUFFT(cufftPlanMany(&s->plan_z2z, 2, n, NULL, 1, (int)z_dist, NULL, 1, (int)z_dist, CUFFT_Z2Z, b_df + b_phi));

        CHECK_CUFFT(cufftCreate(&s->plan_d2z));
        CHECK_CUFFT(cufftPlanMany(&s->plan_d2z, 2, n, NULL, 1, (int)r_dist, NULL, 1, (int)c_dist, CUFFT_D2Z, b_df));
    }

    CHECK_CUFFT(cufftSetStream(s->plan_z2z, stream));
    CHECK_CUFFT(cufftSetStream(s->plan_d2z, stream));

    dim3 blk(32, KX_TILE);
    int m_blks = (mrad + KX_TILE - 1) / KX_TILE;
    int j_blks = (mphi + 31) / 32;

    double2* ws_df  = s->ws_z2z;
    double2* ws_phi = s->ws_z2z + (size_t)b_df * z_dist;

    v5fp64_pack_z2z_kernel<<<dim3(b_df, m_blks, j_blks), blk, 0, stream>>>(
        (const double2*)df.typed_data(), kx.typed_data(), ky.typed_data(), inverse_jind.typed_data(), ws_df, mrad, mphi, nkx, nky);
    v5fp64_pack_z2z_kernel<<<dim3(b_phi, m_blks, j_blks), blk, 0, stream>>>(
        (const double2*)phi.typed_data(), kx.typed_data(), ky.typed_data(), inverse_jind.typed_data(), ws_phi, mrad, mphi, nkx, nky);

    CHECK_CUFFT(cufftExecZ2Z(s->plan_z2z, (cufftDoubleComplex*)s->ws_z2z, (cufftDoubleComplex*)s->ws_z2z, CUFFT_INVERSE));

    double inv_n2 = 1.0 / ((double)mrad * mphi * (double)mrad * mphi);
    v5fp64_assembly_z2z_kernel<<<dim3(b_df, ((int)z_dist + 255)/256), 256, 0, stream>>>(
        ws_df, ws_phi, s->ws_nl_r, dum_s.typed_data(), (int)z_dist, b_phi, nspec, inv_n2);

    CHECK_CUFFT(cufftExecD2Z(s->plan_d2z, s->ws_nl_r, s->ws_nl_k));

    bracket_unpack_kernel<<<dim3(b_df, (nkx + KX_TILE - 1) / KX_TILE), blk, 0, stream>>>(
        s->ws_nl_k, (double2*)out->typed_data(), jind.typed_data(), mrad, mphi_half, nkx, nky, ixzero, iyzero);

    return xla_ffi::Error::Success();
}

xla_ffi::Error CufftGraphBracketFp64DirectImpl(
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
    int b_phi = (int)(phi.element_count() / ((size_t)nkx * nky));
    int mphi_half = mphi / 2 + 1;

    V6Fp64Key key = {device, b_df, b_phi, mrad, mphi, nkx, nky};
    std::lock_guard<std::mutex> lock(g_v6fp64_mutex);
    V6Fp64State* s = g_v6fp64_cache[key];

    size_t z_dist = (size_t)mrad * mphi;
    size_t c_dist = (size_t)mrad * mphi_half;
    size_t r_dist = (size_t)mrad * mphi;

    if (!s) {
        s = new V6Fp64State();
        g_v6fp64_cache[key] = s;
        CHECK_CUDA(cudaMalloc(&s->ws_fy,  (size_t)(b_df + b_phi) * z_dist * sizeof(double2)));
        CHECK_CUDA(cudaMalloc(&s->ws_fx,  (size_t)(b_df + b_phi) * z_dist * sizeof(double2)));
        CHECK_CUDA(cudaMalloc(&s->ws_nl_r, (size_t)b_df * r_dist * sizeof(double)));
        CHECK_CUDA(cudaMalloc(&s->ws_nl_k, (size_t)b_df * c_dist * sizeof(double2)));

        int n[2] = {mrad, mphi};
        CHECK_CUFFT(cufftCreate(&s->plan_fy));
        CHECK_CUFFT(cufftPlanMany(&s->plan_fy, 2, n, NULL, 1, (int)z_dist, NULL, 1, (int)z_dist, CUFFT_Z2Z, b_df + b_phi));
        CHECK_CUFFT(cufftCreate(&s->plan_fx));
        CHECK_CUFFT(cufftPlanMany(&s->plan_fx, 2, n, NULL, 1, (int)z_dist, NULL, 1, (int)z_dist, CUFFT_Z2Z, b_df + b_phi));

        CHECK_CUFFT(cufftCreate(&s->plan_d2z));
        CHECK_CUFFT(cufftPlanMany(&s->plan_d2z, 2, n, NULL, 1, (int)r_dist, NULL, 1, (int)c_dist, CUFFT_D2Z, b_df));
    }

    CHECK_CUFFT(cufftSetStream(s->plan_fy,  stream));
    CHECK_CUFFT(cufftSetStream(s->plan_fx,  stream));
    CHECK_CUFFT(cufftSetStream(s->plan_d2z, stream));

    dim3 blk(32, KX_TILE);
    int m_blks = (mrad + KX_TILE - 1) / KX_TILE;
    int j_blks = (mphi + 31) / 32;

    double2* fy_df  = s->ws_fy;
    double2* fy_phi = s->ws_fy + (size_t)b_df * z_dist;
    double2* fx_df  = s->ws_fx;
    double2* fx_phi = s->ws_fx + (size_t)b_df * z_dist;

    v6fp64_pack_deriv_kernel<<<dim3(b_df, m_blks, j_blks), blk, 0, stream>>>(
        (const double2*)df.typed_data(), kx.typed_data(), ky.typed_data(), inverse_jind.typed_data(), fy_df, fx_df, mrad, mphi, nkx, nky);
    v6fp64_pack_deriv_kernel<<<dim3(b_phi, m_blks, j_blks), blk, 0, stream>>>(
        (const double2*)phi.typed_data(), kx.typed_data(), ky.typed_data(), inverse_jind.typed_data(), fy_phi, fx_phi, mrad, mphi, nkx, nky);

    CHECK_CUFFT(cufftExecZ2Z(s->plan_fy, (cufftDoubleComplex*)s->ws_fy, (cufftDoubleComplex*)s->ws_fy, CUFFT_INVERSE));
    CHECK_CUFFT(cufftExecZ2Z(s->plan_fx, (cufftDoubleComplex*)s->ws_fx, (cufftDoubleComplex*)s->ws_fx, CUFFT_INVERSE));

    double inv_n2 = 1.0 / ((double)mrad * mphi * (double)mrad * mphi);
    v6fp64_assembly_kernel<<<dim3(b_df, ((int)z_dist + 255)/256), 256, 0, stream>>>(
        s->ws_fy, s->ws_fx, s->ws_nl_r, dum_s.typed_data(), (int)z_dist, b_df, b_phi, nspec, inv_n2);

    CHECK_CUFFT(cufftExecD2Z(s->plan_d2z, s->ws_nl_r, s->ws_nl_k));

    bracket_unpack_kernel<<<dim3(b_df, (nkx + KX_TILE - 1) / KX_TILE), blk, 0, stream>>>(
        s->ws_nl_k, (double2*)out->typed_data(), jind.typed_data(), mrad, mphi_half, nkx, nky, ixzero, iyzero);

    return xla_ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(cufft_graph_bracket_mp_ffi, CufftGraphBracketMpImpl,
    xla_ffi::Ffi::Bind().Ctx<xla_ffi::PlatformStream<cudaStream_t>>().Arg<xla_ffi::Buffer<xla_ffi::DataType::C128>>().Arg<xla_ffi::Buffer<xla_ffi::DataType::C128>>().Arg<xla_ffi::Buffer<xla_ffi::DataType::F64>>().Arg<xla_ffi::Buffer<xla_ffi::DataType::F64>>().Arg<xla_ffi::Buffer<xla_ffi::DataType::S32>>().Arg<xla_ffi::Buffer<xla_ffi::DataType::S32>>().Arg<xla_ffi::Buffer<xla_ffi::DataType::F64>>().Ret<xla_ffi::Buffer<xla_ffi::DataType::C128>>().Attr<int32_t>("batch").Attr<int32_t>("mrad").Attr<int32_t>("mphi").Attr<int32_t>("nkx").Attr<int32_t>("nky").Attr<int32_t>("nspec").Attr<int32_t>("ixzero").Attr<int32_t>("iyzero"));

XLA_FFI_DEFINE_HANDLER_SYMBOL(cufft_graph_bracket_fp64_ffi, CufftGraphBracketFp64Impl,
    xla_ffi::Ffi::Bind().Ctx<xla_ffi::PlatformStream<cudaStream_t>>().Arg<xla_ffi::Buffer<xla_ffi::DataType::C128>>().Arg<xla_ffi::Buffer<xla_ffi::DataType::C128>>().Arg<xla_ffi::Buffer<xla_ffi::DataType::F64>>().Arg<xla_ffi::Buffer<xla_ffi::DataType::F64>>().Arg<xla_ffi::Buffer<xla_ffi::DataType::S32>>().Arg<xla_ffi::Buffer<xla_ffi::DataType::S32>>().Arg<xla_ffi::Buffer<xla_ffi::DataType::F64>>().Ret<xla_ffi::Buffer<xla_ffi::DataType::C128>>().Attr<int32_t>("batch").Attr<int32_t>("mrad").Attr<int32_t>("mphi").Attr<int32_t>("nkx").Attr<int32_t>("nky").Attr<int32_t>("nspec").Attr<int32_t>("ixzero").Attr<int32_t>("iyzero"));

XLA_FFI_DEFINE_HANDLER_SYMBOL(cufft_graph_bracket_fp64_direct_ffi, CufftGraphBracketFp64DirectImpl,
    xla_ffi::Ffi::Bind().Ctx<xla_ffi::PlatformStream<cudaStream_t>>().Arg<xla_ffi::Buffer<xla_ffi::DataType::C128>>().Arg<xla_ffi::Buffer<xla_ffi::DataType::C128>>().Arg<xla_ffi::Buffer<xla_ffi::DataType::F64>>().Arg<xla_ffi::Buffer<xla_ffi::DataType::F64>>().Arg<xla_ffi::Buffer<xla_ffi::DataType::S32>>().Arg<xla_ffi::Buffer<xla_ffi::DataType::S32>>().Arg<xla_ffi::Buffer<xla_ffi::DataType::F64>>().Ret<xla_ffi::Buffer<xla_ffi::DataType::C128>>().Attr<int32_t>("batch").Attr<int32_t>("mrad").Attr<int32_t>("mphi").Attr<int32_t>("nkx").Attr<int32_t>("nky").Attr<int32_t>("nspec").Attr<int32_t>("ixzero").Attr<int32_t>("iyzero"));
