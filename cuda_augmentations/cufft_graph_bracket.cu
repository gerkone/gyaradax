#include <cstdio>
#include <string>
#include <mutex>
#include <map>
#include "xla/ffi/api/ffi.h"
#include <cuda_runtime.h>
#include <cufft.h>

// v5 with Z2Z 2-for-1 packing + phi broadcast.
//
// Packs two spectral derivatives into one complex signal:
//   ws = field_y + i*field_x
// After Z2Z IFFT: Re = field_y_real, Im = field_x_real.
// This halves the IRFFT count vs the Z2D approach.
//
// Hermitian symmetrization at ky=0: the gyro-averaged potential has a
// symmetry defect at j=0 from the Bessel multiplication.  Z2D discards
// the resulting imaginary leakage; Z2Z does not.  We fix this by averaging
// each (kx, -kx) mirror pair at j=0 before packing, forcing exact
// Hermitian symmetry where it matters.
//
// Pipeline (5 launches, 2 cuFFT calls):
//   1a. v5_pack_z2z (df)   → ws_z2z[0 : b_df]
//   1b. v5_pack_z2z (phi)  → ws_z2z[b_df : b_df+b_phi]
//   2.  cufftExecZ2Z        (b_df+b_phi, in-place, merged)
//   3.  v5_assembly_z2z     bracket with phi broadcast
//   4.  cufftExecD2Z        (b_df)
//   5.  v5_unpack

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
    cufftHandle plan_z2z = 0;   // b_df + b_phi transforms (merged, in-place)
    cufftHandle plan_d2z = 0;   // b_df transforms

    double2 *ws_z2z  = nullptr;  // [(b_df+b_phi), mrad, mphi] merged
    double  *ws_nl_r = nullptr;  // [b_df, mrad, mphi]
    double2 *ws_nl_k = nullptr;  // [b_df, mrad, mphi_half]

    ~V5State() {
        if (plan_z2z) cufftDestroy(plan_z2z);
        if (plan_d2z) cufftDestroy(plan_d2z);
        if (ws_z2z)  cudaFree(ws_z2z);
        if (ws_nl_r) cudaFree(ws_nl_r);
        if (ws_nl_k) cudaFree(ws_nl_k);
    }
};

static std::map<V5Key, V5State*> g_cache;
static std::mutex g_mutex;
static constexpr int KX_TILE = 4;

// ── Z2Z Pack with Hermitian extension + j=0 symmetrization ─────
// Grid: (batch, ceil(mrad/KX_TILE), ceil(mphi/32))
// Block: (32, KX_TILE) = 128 threads
//
// For j in primary half (j <= mphi/2):
//   - At j=0: read both (m) and mirror (m'), average for Hermitian symmetry
//   - At j>0: read source directly
// For j in mirror half (j > mphi/2):
//   - Read source at (m', j'), conjugate both derivatives, then pack
__global__ void v5_pack_z2z_kernel(
    const double2* __restrict__ field,
    const double*  __restrict__ kx,
    const double*  __restrict__ ky,
    const int*     __restrict__ inverse_jind,  // [mrad] dense→packed
    double2* __restrict__ out,                 // [batch, mrad, mphi]
    int mrad, int mphi, int nkx, int nky
) {
    int b = blockIdx.x;
    int m = blockIdx.y * KX_TILE + threadIdx.y;
    int j = blockIdx.z * blockDim.x + threadIdx.x;
    if (m >= mrad || j >= mphi) return;

    bool mirror = (j > mphi / 2);
    int j_src = mirror ? (mphi - j) : j;
    int m_src = mirror ? ((mrad - m) % mrad) : m;

    const double2 zero = {0.0, 0.0};
    size_t out_idx = (size_t)b * mrad * mphi + (size_t)m * mphi + j;

    if (j_src >= nky) { out[out_idx] = zero; return; }

    int kx_idx = __ldg(&inverse_jind[m_src]);
    if (kx_idx < 0) { out[out_idx] = zero; return; }

    size_t row = (size_t)nkx * nky;
    double2 val = __ldg(&field[(size_t)b * row + (size_t)kx_idx * nky + j_src]);
    double kxv  = __ldg(&kx[kx_idx]);
    double kyv  = __ldg(&ky[j_src]);

    // ── Hermitian symmetrization at j=0 ──────────────────────────
    // The Bessel-multiplied phi has a symmetry defect at ky=0.
    // For each mirror pair (m, m'), enforce: val_sym = (val + conj(val_mirror)) / 2
    if ((j_src == 0 || j_src == mphi / 2) && !mirror) {
        int m_pair = (mrad - m) % mrad;
        int kx_pair = __ldg(&inverse_jind[m_pair]);
        if (kx_pair >= 0 && m_pair != m) {
            double2 val_pair = __ldg(&field[(size_t)b * row + (size_t)kx_pair * nky]);
            // Hermitian average: (val + conj(val_pair)) / 2
            val.x = 0.5 * (val.x + val_pair.x);
            val.y = 0.5 * (val.y - val_pair.y);
            // Use partner's kx (should be -kxv, but we recompute kxv for m_pair)
            // Actually kxv is for m_src=m, which is correct for this thread's position
        }
    }

    // ── Spectral derivatives ─────────────────────────────────────
    // field_y = i*ky*val, field_x = i*kx*val
    double2 fy = {-kyv * val.y,  kyv * val.x};
    double2 fx = {-kxv * val.y,  kxv * val.x};

    // ── Pack: ws = field_y + i*field_x ───────────────────────────
    // Primary: ws = {fy.x - fx.y, fy.y + fx.x}
    // Mirror:  ws = conj(fy) + i*conj(fx) = {fy.x + fx.y, fx.x - fy.y}
    double2 packed;
    if (!mirror) {
        packed = {fy.x - fx.y, fy.y + fx.x};
    } else {
        // At source: fy, fx computed from val at (m_src, j_src)
        // Conjugate both, then pack: conj(fy) + i*conj(fx)
        packed = {fy.x + fx.y, fx.x - fy.y};
    }
    out[out_idx] = packed;
}

// ── Assembly: bracket from Z2Z Re/Im parts ──────────────────────
// After Z2Z: Re(ws_df) = df_y_real, Im(ws_df) = df_x_real
//            Re(ws_phi)= phi_y_real, Im(ws_phi)= phi_x_real
// Bracket = phi_y * df_x - phi_x * df_y
//         = Re(phi) * Im(df) - Im(phi) * Re(df)
//
// Grid: (b_df, ceil(plane/256))  Block: (256)
__global__ void v5_assembly_z2z_kernel(
    const double2* __restrict__ ws_df,
    const double2* __restrict__ ws_phi,
    double* __restrict__ nl,
    const double* __restrict__ dum_s,
    int plane, int b_phi, int nspec, double scale
) {
    int b  = blockIdx.x;
    int off = blockIdx.y * blockDim.x + threadIdx.x;
    if (off >= plane) return;

    double2 d = ws_df[(size_t)b * plane + off];
    double2 p = ws_phi[(size_t)(b % b_phi) * plane + off];

    // phi_y*df_x - phi_x*df_y = Re(p)*Im(d) - Im(p)*Re(d)
    nl[(size_t)b * plane + off] = scale * __ldg(&dum_s[b % nspec]) * (p.x * d.y - p.y * d.x);
}

// ── Unpack (unchanged) ──────────────────────────────────────────
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

    size_t z_dist = (size_t)mrad * mphi;
    size_t c_dist = (size_t)mrad * mphi_half;
    size_t r_dist = (size_t)mrad * mphi;

    if (!s) {
        s = new V5State();
        g_cache[key] = s;

        CHECK_CUDA(cudaMalloc(&s->ws_z2z,  (size_t)(b_df + b_phi) * z_dist * sizeof(double2)));
        CHECK_CUDA(cudaMalloc(&s->ws_nl_r, (size_t)b_df * r_dist * sizeof(double)));
        CHECK_CUDA(cudaMalloc(&s->ws_nl_k, (size_t)b_df * c_dist * sizeof(double2)));

        int n[2] = {mrad, mphi};
        CHECK_CUFFT(cufftCreate(&s->plan_z2z));
        CHECK_CUFFT(cufftPlanMany(&s->plan_z2z, 2, n,
            NULL, 1, (int)z_dist, NULL, 1, (int)z_dist,
            CUFFT_Z2Z, b_df + b_phi));

        CHECK_CUFFT(cufftCreate(&s->plan_d2z));
        CHECK_CUFFT(cufftPlanMany(&s->plan_d2z, 2, n,
            NULL, 1, (int)r_dist, NULL, 1, (int)c_dist,
            CUFFT_D2Z, b_df));
    }

    CHECK_CUFFT(cufftSetStream(s->plan_z2z, stream));
    CHECK_CUFFT(cufftSetStream(s->plan_d2z, stream));

    dim3 blk(32, KX_TILE);  // 128 threads
    int m_blks = (mrad + KX_TILE - 1) / KX_TILE;
    int j_blks = (mphi + 31) / 32;

    double2* ws_df  = s->ws_z2z;
    double2* ws_phi = s->ws_z2z + (size_t)b_df * z_dist;

    // ── 1a. Pack df (Z2Z, Hermitian-extended, j=0 symmetrized) ──
    v5_pack_z2z_kernel<<<dim3(b_df, m_blks, j_blks), blk, 0, stream>>>(
        (const double2*)df.typed_data(), kx.typed_data(), ky.typed_data(),
        inverse_jind.typed_data(), ws_df, mrad, mphi, nkx, nky);

    // ── 1b. Pack phi ────────────────────────────────────────────
    v5_pack_z2z_kernel<<<dim3(b_phi, m_blks, j_blks), blk, 0, stream>>>(
        (const double2*)phi.typed_data(), kx.typed_data(), ky.typed_data(),
        inverse_jind.typed_data(), ws_phi, mrad, mphi, nkx, nky);

    // ── 2. Merged Z2Z IFFT (in-place) ───────────────────────────
    CHECK_CUFFT(cufftExecZ2Z(s->plan_z2z, s->ws_z2z, s->ws_z2z, CUFFT_INVERSE));

    // ── 3. Assembly ─────────────────────────────────────────────
    double inv_n2 = 1.0 / ((double)mrad * mphi * (double)mrad * mphi);
    int plane = (int)z_dist;
    v5_assembly_z2z_kernel<<<dim3(b_df, (plane+255)/256), 256, 0, stream>>>(
        ws_df, ws_phi, s->ws_nl_r, dum_s.typed_data(),
        plane, b_phi, nspec, inv_n2);

    // ── 4. Forward FFT ──────────────────────────────────────────
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
