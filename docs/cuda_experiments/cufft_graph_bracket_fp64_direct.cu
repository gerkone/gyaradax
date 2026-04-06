// v6-FP64: FP64 Poisson bracket WITHOUT the 2-for-1 derivative trick.
//
// Structure is identical to cufft_graph_bracket_fp64.cu (v5-FP64) except
// fy and fx are kept in SEPARATE workspaces and transformed independently.
// The merged-batch layout (b_df + b_phi in one plan) and all other
// optimisations (phi broadcast, dum_s, ixzero masking) are preserved.
//
// Pipeline (5 launches, 3 cuFFT calls):
//   1a.   v6fp64_pack_deriv  (df  → ws_fy[0..b_df-1],     ws_fx[0..b_df-1])
//   1b.   v6fp64_pack_deriv  (phi → ws_fy[b_df..b_df+b_phi-1], ws_fx[...])
//   2a.   cufftExecZ2Z(plan_fy, ws_fy, INVERSE)   FP64, b_df+b_phi batches
//   2b.   cufftExecZ2Z(plan_fx, ws_fx, INVERSE)   FP64, b_df+b_phi batches
//   3.    v6fp64_assembly    (bracket: phi_fy*df_fx − phi_fx*df_fy, FP64)
//   4.    cufftExecD2Z       (FP64 forward, b_df batches)
//   5.    v6fp64_unpack      (FP64, identical to v5)
//
// Cost vs v5-FP64 (with 2-for-1): 2× Z2Z calls + 2× Z2Z workspace memory.
// The gap quantifies the value of the 2-for-1 trick.

#include <cstdio>
#include <string>
#include <mutex>
#include <map>
#include "xla/ffi/api/ffi.h"
#include <cuda_runtime.h>
#include <cufft.h>

namespace {

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
    cufftHandle plan_fy  = 0;   // FP64 Z2Z, b_df+b_phi batches, for fy workspace
    cufftHandle plan_fx  = 0;   // FP64 Z2Z, b_df+b_phi batches, for fx workspace
    cufftHandle plan_d2z = 0;   // FP64 D2Z, b_df batches

    // Two merged workspaces — same layout as v5 ws_z2z but split by derivative.
    // ws_fy[0..b_df-1]         = i*ky*df after IFFT (fy of df in real space)
    // ws_fy[b_df..b_df+b_phi-1]= i*ky*phi after IFFT (fy of phi, broadcast)
    double2 *ws_fy  = nullptr;  // [(b_df+b_phi), mrad, mphi] FP64 complex
    double2 *ws_fx  = nullptr;  // [(b_df+b_phi), mrad, mphi] FP64 complex

    double  *ws_nl_r = nullptr; // [b_df, mrad, mphi]      FP64 real
    double2 *ws_nl_k = nullptr; // [b_df, mrad, mphi_half] FP64 complex

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
static constexpr int KX_TILE = 4;

// ── Pack: FP64 input → separate fy and fx workspaces ─────────────
//
// Same indexing and Hermitian symmetrisation as v5fp64_pack_z2z_kernel.
// Instead of packing fy + i*fx into one double2, stores them separately.
//
// For a Hermitian-packed sequence the IFFT is real-valued:
//   non-mirror: out_y = { fy_re,  fy_im },  out_x = { fx_re,  fx_im }
//   mirror:     out_y = { fy_re, -fy_im },  out_x = { fx_re, -fx_im }
// (conjugate at the mirrored point → Hermitian → IFFT gives real result)
__global__ void v6fp64_pack_deriv_kernel(
    const double2* __restrict__ field,   // [batch, nkx, nky] FP64
    const double*  __restrict__ kx,
    const double*  __restrict__ ky,
    const int*     __restrict__ inverse_jind,
    double2* __restrict__ out_y,         // [batch, mrad, mphi] ky-derivative
    double2* __restrict__ out_x,         // [batch, mrad, mphi] kx-derivative
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

    if (j_src >= nky) {
        out_y[out_idx] = {0.0, 0.0};
        out_x[out_idx] = {0.0, 0.0};
        return;
    }

    int kx_idx = __ldg(&inverse_jind[m_src]);
    if (kx_idx < 0) {
        out_y[out_idx] = {0.0, 0.0};
        out_x[out_idx] = {0.0, 0.0};
        return;
    }

    size_t row = (size_t)nkx * nky;
    double2 val = __ldg(&field[(size_t)b * row + (size_t)kx_idx * nky + j_src]);
    double kxv  = __ldg(&kx[kx_idx]);
    double kyv  = __ldg(&ky[j_src]);

    // Hermitian symmetrisation at ky=0 (same as v5)
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

// ── Assembly: bracket from separate fy/fx workspaces (FP64) ──────
// After IFFT of Hermitian-packed fy/fx, the physical derivative is in
// the real (.x) component; imaginary part ≈ 0.
// Phi broadcast: ws_fy/ws_fx phi batches start at offset b_df*plane.
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

// ── Unpack (FP64, identical to v5) ────────────────────────────────
__global__ void v6fp64_unpack_kernel(
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
    size_t phi_elems = 1;
    for (auto d : phi.dimensions()) phi_elems *= d;
    int b_phi = (int)(phi_elems / ((size_t)nkx * nky));
    int mphi_half = mphi / 2 + 1;

    V6Fp64Key key = {device, b_df, b_phi, mrad, mphi, nkx, nky};
    std::lock_guard<std::mutex> lock(g_v6fp64_mutex);
    V6Fp64State* s = g_v6fp64_cache[key];

    size_t z_dist = (size_t)mrad * mphi;       // complex elements per Z2Z transform
    size_t c_dist = (size_t)mrad * mphi_half;  // complex elements per D2Z output
    size_t r_dist = (size_t)mrad * mphi;       // real elements per D2Z input

    if (!s) {
        s = new V6Fp64State();
        g_v6fp64_cache[key] = s;

        // Two merged workspaces — same size as v5 ws_z2z but separate per derivative
        CHECK_CUDA(cudaMalloc(&s->ws_fy,  (size_t)(b_df + b_phi) * z_dist * sizeof(double2)));
        CHECK_CUDA(cudaMalloc(&s->ws_fx,  (size_t)(b_df + b_phi) * z_dist * sizeof(double2)));
        CHECK_CUDA(cudaMalloc(&s->ws_nl_r, (size_t)b_df * r_dist * sizeof(double)));
        CHECK_CUDA(cudaMalloc(&s->ws_nl_k, (size_t)b_df * c_dist * sizeof(double2)));

        int n[2] = {mrad, mphi};

        // Both Z2Z plans share the same batch size (b_df+b_phi) — mirrors v5 Z2Z plan
        CHECK_CUFFT(cufftCreate(&s->plan_fy));
        CHECK_CUFFT(cufftPlanMany(&s->plan_fy, 2, n,
            NULL, 1, (int)z_dist, NULL, 1, (int)z_dist,
            CUFFT_Z2Z, b_df + b_phi));

        CHECK_CUFFT(cufftCreate(&s->plan_fx));
        CHECK_CUFFT(cufftPlanMany(&s->plan_fx, 2, n,
            NULL, 1, (int)z_dist, NULL, 1, (int)z_dist,
            CUFFT_Z2Z, b_df + b_phi));

        CHECK_CUFFT(cufftCreate(&s->plan_d2z));
        CHECK_CUFFT(cufftPlanMany(&s->plan_d2z, 2, n,
            NULL, 1, (int)r_dist, NULL, 1, (int)c_dist,
            CUFFT_D2Z, b_df));
    }

    CHECK_CUFFT(cufftSetStream(s->plan_fy,  stream));
    CHECK_CUFFT(cufftSetStream(s->plan_fx,  stream));
    CHECK_CUFFT(cufftSetStream(s->plan_d2z, stream));

    dim3 blk(32, KX_TILE);
    int m_blks = (mrad + KX_TILE - 1) / KX_TILE;
    int j_blks = (mphi + 31) / 32;

    // df phi sections in each workspace — mirrors v5's ws_df / ws_phi split
    double2* fy_df  = s->ws_fy;
    double2* fy_phi = s->ws_fy + (size_t)b_df * z_dist;
    double2* fx_df  = s->ws_fx;
    double2* fx_phi = s->ws_fx + (size_t)b_df * z_dist;

    // ── 1a. Pack df → ws_fy[0..b_df-1] and ws_fx[0..b_df-1] ─────
    v6fp64_pack_deriv_kernel<<<dim3(b_df, m_blks, j_blks), blk, 0, stream>>>(
        (const double2*)df.typed_data(), kx.typed_data(), ky.typed_data(),
        inverse_jind.typed_data(),
        fy_df, fx_df,
        mrad, mphi, nkx, nky);

    // ── 1b. Pack phi → ws_fy[b_df..] and ws_fx[b_df..] ──────────
    v6fp64_pack_deriv_kernel<<<dim3(b_phi, m_blks, j_blks), blk, 0, stream>>>(
        (const double2*)phi.typed_data(), kx.typed_data(), ky.typed_data(),
        inverse_jind.typed_data(),
        fy_phi, fx_phi,
        mrad, mphi, nkx, nky);

    // ── 2a. Z2Z inverse for fy (b_df+b_phi merged) ───────────────
    CHECK_CUFFT(cufftExecZ2Z(s->plan_fy,
        (cufftDoubleComplex*)s->ws_fy,
        (cufftDoubleComplex*)s->ws_fy, CUFFT_INVERSE));

    // ── 2b. Z2Z inverse for fx (b_df+b_phi merged) ───────────────
    CHECK_CUFFT(cufftExecZ2Z(s->plan_fx,
        (cufftDoubleComplex*)s->ws_fx,
        (cufftDoubleComplex*)s->ws_fx, CUFFT_INVERSE));

    // ── 3. Assembly: FP64 bracket ─────────────────────────────────
    double inv_n2 = 1.0 / ((double)mrad * mphi * (double)mrad * mphi);
    int plane = (int)z_dist;
    v6fp64_assembly_kernel<<<dim3(b_df, (plane + 255) / 256), 256, 0, stream>>>(
        s->ws_fy, s->ws_fx, s->ws_nl_r, dum_s.typed_data(),
        plane, b_df, b_phi, nspec, inv_n2);

    // ── 4. FP64 forward D2Z ───────────────────────────────────────
    CHECK_CUFFT(cufftExecD2Z(s->plan_d2z, s->ws_nl_r, s->ws_nl_k));

    // ── 5. Unpack ─────────────────────────────────────────────────
    int kx_blks = (nkx + KX_TILE - 1) / KX_TILE;
    v6fp64_unpack_kernel<<<dim3(b_df, kx_blks), blk, 0, stream>>>(
        s->ws_nl_k, (double2*)out->typed_data(), jind.typed_data(),
        mrad, mphi_half, nkx, nky, ixzero, iyzero);

    return xla_ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    cufft_graph_bracket_fp64_direct_ffi, CufftGraphBracketFp64DirectImpl,
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
