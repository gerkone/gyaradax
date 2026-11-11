// v6 bracket on hipfft with contiguous ffts and lds-tiled pack, expand and split; exports the cuda v6 ffi symbols

#include <cmath>
#include <map>
#include <mutex>
#include <string>
#include <vector>
#include "xla/ffi/api/ffi.h"
#include <hipfft/hipfft.h>

#include "bracket_v6_info.cuh"

namespace xla_ffi = xla::ffi;

namespace {

template <class T> struct Prec;
template <> struct Prec<float> {
    using cplx = float2;
    static constexpr hipfftType type = HIPFFT_C2C;
    static hipfftResult exec(hipfftHandle p, cplx* x, int dir) {
        return hipfftExecC2C(p, (hipfftComplex*)x, (hipfftComplex*)x, dir);
    }
};
template <> struct Prec<double> {
    using cplx = double2;
    static constexpr hipfftType type = HIPFFT_Z2Z;
    static hipfftResult exec(hipfftHandle p, cplx* x, int dir) {
        return hipfftExecZ2Z(p, (hipfftDoubleComplex*)x, (hipfftDoubleComplex*)x, dir);
    }
};

// flat launches: one thread per element, 32-bit element index
#define GYX_INDEX(k) const unsigned k = blockIdx.x * blockDim.x + threadIdx.x

constexpr unsigned GYX_WG = 1024;
// lds tile edge and thread rows of the tiled transposes
constexpr int GYX_T = 32;
constexpr int GYX_TR = 16;
static inline unsigned gyx_blocks(unsigned long long n) { return (unsigned)((n + GYX_WG - 1) / GYX_WG); }
static inline unsigned gyx_tiles(int n) { return (unsigned)((n + GYX_T - 1) / GYX_T); }

template <class C>
__device__ __forceinline__ void gyx_store(C* p, const C v) {
    __builtin_nontemporal_store(v.x, &p->x);
    __builtin_nontemporal_store(v.y, &p->y);
}

template <class T>
struct BracketArgs {
    const typename Prec<T>::cplx* rows;  // [n_planes][mrad][mphi]
    typename Prec<T>::cplx*       z;     // [n_pairs][mrad][mphi]
    const double* dum_s;
    const double* vfac;
    int mrad, b_df, b_phi, b_sp, b_inner, nv, nspec, mphi, em;
    T scale;
};

// real bracket value of df plane gb at offset rij inside a plane
template <class T>
__device__ __forceinline__ T bracket_at(const BracketArgs<T>& a, unsigned plane, unsigned rij, int gb) {
    using C = typename Prec<T>::cplx;
    const C x = a.rows[gb * plane + rij];
    const int pidx = (gb / a.b_sp) * a.b_inner + gb % a.b_inner;
    const unsigned pot = (a.b_df + pidx) * plane + rij;
    C p = a.rows[pot];
    if (a.em) {
        const T w = (T)a.vfac[(gb / a.b_sp) * a.nv + (gb % a.b_sp) / a.b_inner];
        const C q = a.rows[pot + a.b_phi * plane];
        p = C{p.x + w * q.x, p.y + w * q.y};
    }
    return a.scale * (T)a.dum_s[gb % a.nspec] * (p.x * x.y - p.y * x.x);
}

// two real bracket planes as the real and imaginary part of one complex row
template <class T>
__global__ void bracket_kernel(const BracketArgs<T> a, unsigned n) {
    GYX_INDEX(k);
    if (k >= n) return;
    const unsigned plane = (unsigned)a.mrad * a.mphi;
    const unsigned pair = k / plane;
    const unsigned rij = k - pair * plane;
    const int gb0 = 2 * pair;
    const T br0 = bracket_at<T>(a, plane, rij, gb0);
    const T br1 = gb0 + 1 < a.b_df ? bracket_at<T>(a, plane, rij, gb0 + 1) : T(0);
    gyx_store(&a.z[k], typename Prec<T>::cplx{br0, br1});
}

template <class T>
__global__ void unpack_kernel(const typename Prec<T>::cplx* __restrict__ b, const V6StoreInfo si,
                              const int* __restrict__ jind, int mrad, unsigned n) {
    GYX_INDEX(t);
    if (t >= n) return;
    const unsigned r = t / si.nky;
    const int k = t - r * si.nky;
    const unsigned gb = r / si.nkx;
    const int i_pack = r - gb * si.nkx;
    const typename Prec<T>::cplx v = b[(gb * si.nky + k) * mrad + jind[i_pack]];
    const double2 u = (i_pack == si.ixzero && k == si.iyzero) ? make_double2(0.0, 0.0)
                                                              : make_double2((double)v.x, (double)v.y);
    gyx_store(&si.out_packed[t], si.fold ? v6_scale_c(si.scale_re, si.scale_im, u.x, u.y) : u);
}

// tiled transposes: lanes along the contiguous side of the source, then of the destination
template <class T>
__global__ __launch_bounds__(GYX_T * GYX_TR) void pack_tiled_kernel(typename Prec<T>::cplx* a, const V6ColInfo ci,
                                                                    unsigned tiles_c) {
    using C = typename Prec<T>::cplx;
    __shared__ C tile[GYX_T][GYX_T + 1];
    const unsigned gb = blockIdx.x / tiles_c;
    const int c0 = (blockIdx.x - gb * tiles_c) * GYX_T, i0 = blockIdx.y * GYX_T;
    const int mrad = ci.z.mrad, n_cols = ci.n_cols;
    const int c = c0 + threadIdx.x;
    if (c < n_cols) {
        const int j = v6_col_to_j(c, ci.z.nky, ci.z.mphi);
        for (int il = threadIdx.y; il < GYX_T; il += GYX_TR) {
            const int i = i0 + il;
            if (i >= mrad) break;
            if constexpr (sizeof(T) == 4) tile[il][threadIdx.x] = v5_z2z_true_fp32_pack_at(gb, i, j, &ci.z);
            else tile[il][threadIdx.x] = v5_z2z_pack_at(gb, i, j, &ci.z);
        }
    }
    __syncthreads();
    const int i = i0 + threadIdx.x;
    if (i >= mrad) return;
    for (int cl = threadIdx.y; cl < GYX_T; cl += GYX_TR) {
        const int cc = c0 + cl;
        if (cc >= n_cols) break;
        gyx_store(&a[((size_t)gb * n_cols + cc) * mrad + i], tile[threadIdx.x][cl]);
    }
}

template <class T>
__global__ __launch_bounds__(GYX_T * GYX_TR) void expand_tiled_kernel(const typename Prec<T>::cplx* __restrict__ a,
                                                                      typename Prec<T>::cplx* __restrict__ rows,
                                                                      int n_cols, int mrad, int mphi, int nky,
                                                                      unsigned tiles_i) {
    using C = typename Prec<T>::cplx;
    __shared__ C tile[GYX_T][GYX_T + 1];
    const unsigned gb = blockIdx.x / tiles_i;
    const int i0 = (blockIdx.x - gb * tiles_i) * GYX_T, j0 = blockIdx.y * GYX_T;
    const int i = i0 + threadIdx.x;
    if (i < mrad) {
        for (int jl = threadIdx.y; jl < GYX_T; jl += GYX_TR) {
            const int j = j0 + jl;
            if (j >= mphi) break;
            int jj = -1;
            if (j < nky) jj = j;
            else if (j > mphi - nky) jj = j - (mphi - 2 * nky + 1);
            tile[jl][threadIdx.x] = jj < 0 ? C{T(0), T(0)} : a[((size_t)gb * n_cols + jj) * mrad + i];
        }
    }
    __syncthreads();
    const int j = j0 + threadIdx.x;
    if (j >= mphi) return;
    for (int il = threadIdx.y; il < GYX_T; il += GYX_TR) {
        const int ii = i0 + il;
        if (ii >= mrad) break;
        gyx_store(&rows[((size_t)gb * mrad + ii) * mphi + j], tile[threadIdx.x][il]);
    }
}

// one tile of z gives both parities of b
template <class T>
__global__ __launch_bounds__(GYX_T * GYX_TR) void split_tiled_kernel(const typename Prec<T>::cplx* __restrict__ z,
                                                                     typename Prec<T>::cplx* __restrict__ b,
                                                                     int mrad, int mphi, int nky, int b_df,
                                                                     unsigned tiles_q) {
    using C = typename Prec<T>::cplx;
    __shared__ C te[GYX_T][GYX_T + 1];
    __shared__ C to[GYX_T][GYX_T + 1];
    const unsigned pair = blockIdx.x / tiles_q;
    const int q0 = (blockIdx.x - pair * tiles_q) * GYX_T, i0 = blockIdx.y * GYX_T;
    const int q = q0 + threadIdx.x;
    if (q < nky) {
        for (int il = threadIdx.y; il < GYX_T; il += GYX_TR) {
            const int i = i0 + il;
            if (i >= mrad) break;
            const size_t row = ((size_t)pair * mrad + i) * mphi;
            const C zk = z[row + q], zm = z[row + (mphi - q) % mphi];
            te[il][threadIdx.x] = C{T(0.5) * (zk.x + zm.x), T(0.5) * (zk.y - zm.y)};
            to[il][threadIdx.x] = C{T(0.5) * (zk.y + zm.y), T(0.5) * (zm.x - zk.x)};
        }
    }
    __syncthreads();
    const int i = i0 + threadIdx.x;
    if (i >= mrad) return;
    const unsigned g0 = 2 * pair;
    const bool odd = g0 + 1 < (unsigned)b_df;
    for (int ql = threadIdx.y; ql < GYX_T; ql += GYX_TR) {
        const int qq = q0 + ql;
        if (qq >= nky) break;
        gyx_store(&b[((size_t)g0 * nky + qq) * mrad + i], te[threadIdx.x][ql]);
        if (odd) gyx_store(&b[((size_t)(g0 + 1) * nky + qq) * mrad + i], to[threadIdx.x][ql]);
    }
}

#ifdef GYX_ROWFUSE
// fused row pass (expand, row ffts, bracket, split) in lds per (pair, kx tile); stockham radices 4, 3, 2
constexpr int GYX_RF_THREADS = 256;
constexpr int GYX_RF_TI = 16;
constexpr size_t GYX_RF_LDS = 16384;

// workgroups go round-robin over the 8 xcds; consecutive logical blocks on one xcd share its l2
__device__ __forceinline__ unsigned gyx_xcd_block(unsigned bid, unsigned nb) {
    const unsigned per = nb / 8;
    return bid < per * 8 ? (bid % 8) * per + bid / 8 : bid;
}

__host__ __device__ constexpr int gyx_rf_radix(int n) { return n % 4 == 0 ? 4 : n % 3 == 0 ? 3 : n % 2 == 0 ? 2 : 0; }

template <class C, class T>
__device__ __forceinline__ C gyx_cmul(C a, C b) { return C{a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x}; }

// one stockham stage of radix R on n_tr transforms of length N, x -> y; w: exp(-2 pi i m / N), sign +1 = inverse
template <class C, class T, int N, int R, int NS, int SIGN>
__device__ __forceinline__ void gyx_rf_stage(const C* __restrict__ x, C* __restrict__ y, const C* __restrict__ w,
                                             int n_tr) {
    constexpr int M = N / R;
    for (int u = threadIdx.x; u < n_tr * M; u += blockDim.x) {
        const int tr = u / M, j = u - tr * M;
        const C* xt = x + tr * N;
        C* yt = y + tr * N;
        const int k = j % NS;
        C v[R];
#pragma unroll
        for (int t = 0; t < R; ++t) {
            const C a = xt[j + t * M];
            if (t == 0) { v[0] = a; continue; }
            C tw = w[(t * k * (N / (NS * R))) % N];
            if (SIGN > 0) tw.y = -tw.y;
            v[t] = gyx_cmul<C, T>(a, tw);
        }
        constexpr T sg = SIGN > 0 ? T(1) : T(-1);
        if constexpr (R == 2) {
            const C a = v[0], b = v[1];
            v[0] = C{a.x + b.x, a.y + b.y};
            v[1] = C{a.x - b.x, a.y - b.y};
        } else if constexpr (R == 3) {
            const T c3 = T(-0.5), s3 = sg * T(0.86602540378443864676);
            const C a = v[0], b = v[1], c = v[2];
            const C sbc = C{b.x + c.x, b.y + c.y}, dbc = C{b.x - c.x, b.y - c.y};
            const C m1 = C{a.x + c3 * sbc.x, a.y + c3 * sbc.y};
            v[0] = C{a.x + sbc.x, a.y + sbc.y};
            v[1] = C{m1.x - s3 * dbc.y, m1.y + s3 * dbc.x};
            v[2] = C{m1.x + s3 * dbc.y, m1.y - s3 * dbc.x};
        } else {
            const C a = v[0], b = v[1], c = v[2], d = v[3];
            const C s0 = C{a.x + c.x, a.y + c.y}, d0 = C{a.x - c.x, a.y - c.y};
            const C s1 = C{b.x + d.x, b.y + d.y}, d1 = C{b.x - d.x, b.y - d.y};
            const C jd1 = C{-sg * d1.y, sg * d1.x};
            v[0] = C{s0.x + s1.x, s0.y + s1.y};
            v[1] = C{d0.x + jd1.x, d0.y + jd1.y};
            v[2] = C{s0.x - s1.x, s0.y - s1.y};
            v[3] = C{d0.x - jd1.x, d0.y - jd1.y};
        }
        const int base = (j / NS) * NS * R + k;
#pragma unroll
        for (int t = 0; t < R; ++t) yt[base + t * NS] = v[t];
    }
}

// stages from NS upward; the result ends in x when the stage count is even, else in y
template <class C, class T, int N, int NS, int SIGN>
__device__ __forceinline__ void gyx_rf_stages(C* x, C* y, const C* w, int n_tr) {
    if constexpr (NS < N) {
        constexpr int R = gyx_rf_radix(N / NS);
        __syncthreads();
        gyx_rf_stage<C, T, N, R, NS, SIGN>(x, y, w, n_tr);
        gyx_rf_stages<C, T, N, NS * R, SIGN>(y, x, w, n_tr);
    }
}

template <int N>
__host__ __device__ constexpr int gyx_rf_count() {
    int c = 0, r = N;
    while (r > 1) { r /= gyx_rf_radix(r); ++c; }
    return c;
}

// full transform; returns the buffer holding the result
template <class C, class T, int N, int SIGN>
__device__ C* gyx_rf_fft(C* x, C* y, const C* w, int n_tr) {
    gyx_rf_stages<C, T, N, 1, SIGN>(x, y, w, n_tr);
    __syncthreads();
    return gyx_rf_count<N>() % 2 == 0 ? x : y;
}

template <class T>
struct RowFuseArgs {
    const typename Prec<T>::cplx* a;  // [n_planes][n_cols][mrad]
    typename Prec<T>::cplx*       b;  // [b_df][nky][mrad]
    const typename Prec<T>::cplx* w;  // [mphi] exp(-2 pi i m / mphi)
    const double* dum_s;
    const double* vfac;
    int mrad, n_cols, nky, b_df, b_phi, b_sp, b_inner, nv, nspec, em, ti, tiles_i;
    T scale;
};

// slots per kx row: df0, df1, pot0, pot1 (+ apar0, apar1 when em)
template <class T, int N>
__global__ __launch_bounds__(GYX_RF_THREADS) void rowfuse_kernel(const RowFuseArgs<T> a) {
    using C = typename Prec<T>::cplx;
    extern __shared__ unsigned char gyx_rf_smem[];
    const int slots = a.em ? 6 : 4, ti = a.ti;
    const int n_tr = slots * ti;
    C* buf0 = reinterpret_cast<C*>(gyx_rf_smem);
    C* buf1 = buf0 + n_tr * N;
    const unsigned bid = gyx_xcd_block(blockIdx.x, gridDim.x);
    const int pair = bid / a.tiles_i;
    const int i0 = (bid - pair * a.tiles_i) * ti;
    const int gb0 = 2 * pair;
    const bool has1 = gb0 + 1 < a.b_df;
    int plane[6];
#pragma unroll
    for (int h = 0; h < 2; ++h) {
        const int gb = has1 || h == 0 ? gb0 + h : gb0;
        const int pidx = (gb / a.b_sp) * a.b_inner + gb % a.b_inner;
        plane[h] = gb;
        plane[2 + h] = a.b_df + pidx;
        plane[4 + h] = a.b_df + pidx + a.b_phi;
    }
    const int nky = a.nky, n_cols = a.n_cols;
    // expand: retained columns into rows, zero band, lanes along kx rows
    for (int u = threadIdx.x; u < n_tr * N; u += blockDim.x) {
        const int il = u % ti, r = u / ti;
        const int jpos = r % N, sl = r / N;
        const int i = i0 + il;
        int jj = -1;
        if (jpos < nky) jj = jpos;
        else if (jpos > N - nky) jj = jpos - (N - 2 * nky + 1);
        C v = C{T(0), T(0)};
        if (jj >= 0 && i < a.mrad) v = a.a[((size_t)plane[sl] * n_cols + jj) * a.mrad + i];
        buf0[(sl * ti + il) * N + jpos] = v;
    }
    C* x = gyx_rf_fft<C, T, N, +1>(buf0, buf1, a.w, n_tr);
    C* zb = x == buf0 ? buf1 : buf0;
    // bracket into pairs, z rows in the other buffer
    for (int u = threadIdx.x; u < ti * N; u += blockDim.x) {
        const int il = u / N, jpos = u - il * N;
        T br[2];
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            if (h == 1 && !has1) { br[1] = T(0); continue; }
            const int gb = gb0 + h;
            const C xv = x[(h * ti + il) * N + jpos];
            C p = x[((2 + h) * ti + il) * N + jpos];
            if (a.em) {
                const T wv = (T)a.vfac[(gb / a.b_sp) * a.nv + (gb % a.b_sp) / a.b_inner];
                const C q = x[((4 + h) * ti + il) * N + jpos];
                p = C{p.x + wv * q.x, p.y + wv * q.y};
            }
            br[h] = a.scale * (T)a.dum_s[gb % a.nspec] * (p.x * xv.y - p.y * xv.x);
        }
        zb[il * N + jpos] = C{br[0], br[1]};
    }
    C* z = gyx_rf_fft<C, T, N, -1>(zb, x, a.w, ti);
    // split, lanes along kx rows
    for (int u = threadIdx.x; u < 2 * nky * ti; u += blockDim.x) {
        const int il = u % ti, r = u / ti;
        const int q = r % nky, h = r / nky;
        const int i = i0 + il;
        if (i >= a.mrad || (h == 1 && !has1)) continue;
        const C zk = z[il * N + q], zm = z[il * N + (N - q) % N];
        const C v = h == 0 ? C{T(0.5) * (zk.x + zm.x), T(0.5) * (zk.y - zm.y)}
                           : C{T(0.5) * (zk.y + zm.y), T(0.5) * (zm.x - zk.x)};
        a.b[((size_t)(gb0 + h) * nky + q) * a.mrad + i] = v;
    }
}

template <class T>
bool gyx_rowfuse_launch(const RowFuseArgs<T>& args0, int mphi, int n_pairs, hipStream_t stream) {
    using C = typename Prec<T>::cplx;
    RowFuseArgs<T> args = args0;
    const int slots = args.em ? 6 : 4;
    int ti = GYX_RF_TI;
    while (ti > 1 && (size_t)2 * slots * ti * mphi * sizeof(C) > GYX_RF_LDS) ti /= 2;
    args.ti = ti;
    args.tiles_i = (args.mrad + ti - 1) / ti;
    const size_t smem = (size_t)2 * slots * ti * mphi * sizeof(C);
    const dim3 grid(n_pairs * args.tiles_i);
    switch (mphi) {
#define GYX_RF_CASE(n) case n: rowfuse_kernel<T, n><<<grid, GYX_RF_THREADS, smem, stream>>>(args); return true;
        GYX_RF_CASE(48) GYX_RF_CASE(64) GYX_RF_CASE(72) GYX_RF_CASE(96) GYX_RF_CASE(128)
        GYX_RF_CASE(144) GYX_RF_CASE(192) GYX_RF_CASE(256)
#undef GYX_RF_CASE
        default: return false;
    }
}
#endif

struct Key {
    int device, b_df, b_pot, mrad, mphi, nkx, nky, em, fp64;
    bool operator<(const Key& o) const {
        const int l[9] = {device, b_df, b_pot, mrad, mphi, nkx, nky, em, fp64};
        const int r[9] = {o.device, o.b_df, o.b_pot, o.mrad, o.mphi, o.nkx, o.nky, o.em, o.fp64};
        for (int k = 0; k < 9; ++k)
            if (l[k] != r[k]) return l[k] < r[k];
        return false;
    }
};

struct State {
    hipfftHandle col_inv = {}, row_inv = {}, row_fwd = {}, col_fwd = {};
    void *a = nullptr, *rows = nullptr, *z = nullptr, *b = nullptr, *w = nullptr;
    ~State() {
        for (hipfftHandle p : {col_inv, row_inv, row_fwd, col_fwd}) if (p) hipfftDestroy(p);
        for (void* x : {a, rows, z, b, w}) if (x) hipFree(x);
    }
};

std::map<Key, State*> g_cache;
std::mutex g_mutex;

#define CHECK_HIP(call) do { hipError_t err_ = (call); \
    if (err_ != hipSuccess) return xla_ffi::Error::Internal(std::string("HIP ") + hipGetErrorString(err_)); \
} while (0)
#define CHECK_FFT(call) do { hipfftResult res_ = (call); \
    if (res_ != HIPFFT_SUCCESS) return xla_ffi::Error::Internal(std::string("hipFFT code ") + std::to_string((int)res_)); \
} while (0)

// batch of contiguous transforms of length len
template <class T>
hipfftResult make_plan(hipfftHandle* plan, int len, int batch) {
    int n[1] = {len};
    size_t ws = 0;
    hipfftResult r = hipfftCreate(plan);
    if (r != HIPFFT_SUCCESS) return r;
    return hipfftMakePlanMany(*plan, 1, n, nullptr, 1, len, nullptr, 1, len, Prec<T>::type, batch, &ws);
}

template <class T>
xla_ffi::Error BracketImpl(
    hipStream_t stream,
    xla_ffi::Buffer<xla_ffi::DataType::C128> df,
    xla_ffi::Buffer<xla_ffi::DataType::C128> phi,
    xla_ffi::Buffer<xla_ffi::DataType::F64>  kx,
    xla_ffi::Buffer<xla_ffi::DataType::F64>  ky,
    xla_ffi::Buffer<xla_ffi::DataType::S32>  jind,
    xla_ffi::Buffer<xla_ffi::DataType::S32>  inverse_jind,
    xla_ffi::Buffer<xla_ffi::DataType::F64>  dum_s,
    xla_ffi::Buffer<xla_ffi::DataType::F64>  vfac,
    xla_ffi::Result<xla_ffi::Buffer<xla_ffi::DataType::C128>> out,
    int32_t batch, int32_t mrad, int32_t mphi, int32_t nkx, int32_t nky, int32_t nspec,
    int32_t ixzero, int32_t iyzero, int32_t nsp, int32_t nv, int32_t b_inner, int32_t em,
    int32_t fold, double out_scale_re, double out_scale_im, int32_t fused)
{
    using C = typename Prec<T>::cplx;
    (void)fused;
    int device = 0;
    CHECK_HIP(hipGetDevice(&device));

    const int b_df = batch * nspec;
    size_t phi_elems = 1;
    for (auto d : phi.dimensions()) phi_elems *= d;
    const int b_pot = (int)(phi_elems / ((size_t)nkx * nky));
    const int b_phi = em ? b_pot / 2 : b_pot;
    if (nsp <= 0 || b_inner <= 0 || b_df % nsp != 0 || (b_df / nsp) % b_inner != 0 || b_phi != nsp * b_inner
        || (em && b_pot != 2 * b_phi) || 2 * nky - 1 > mphi)
        return xla_ffi::Error(XLA_FFI_Error_Code_INVALID_ARGUMENT,
                              "bracket: potential planes inconsistent with nsp/b_inner");
    const int b_sp = b_df / nsp;
    const int n_planes = b_df + b_pot;
    const int n_cols = 2 * nky - 1;
    const int n_pairs = (b_df + 1) / 2;
    const unsigned long long n_a = (unsigned long long)mrad * n_planes * n_cols;
    const unsigned long long n_rows = (unsigned long long)mrad * n_planes * mphi;
    const unsigned long long n_z = (unsigned long long)mrad * n_pairs * mphi;
    const unsigned long long n_b = (unsigned long long)mrad * b_df * nky;
    const unsigned long long n_out = (unsigned long long)b_df * nkx * nky;
    if (n_rows >> 32 || n_a >> 32)
        return xla_ffi::Error(XLA_FFI_Error_Code_INVALID_ARGUMENT, "bracket: grid exceeds 32-bit element indices");

    Key key = {device, b_df, b_pot, mrad, mphi, nkx, nky, em, sizeof(T) == 8};
    std::lock_guard<std::mutex> lock(g_mutex);
    State* s = g_cache[key];
    if (!s) {
        s = new State();
        g_cache[key] = s;
        CHECK_HIP(hipMalloc(&s->a, n_a * sizeof(C)));
        CHECK_HIP(hipMalloc(&s->rows, n_rows * sizeof(C)));
        CHECK_HIP(hipMalloc(&s->z, n_z * sizeof(C)));
        CHECK_HIP(hipMalloc(&s->b, n_b * sizeof(C)));
        CHECK_FFT(make_plan<T>(&s->col_inv, mrad, n_planes * n_cols));
        CHECK_FFT(make_plan<T>(&s->col_fwd, mrad, b_df * nky));
        CHECK_FFT(make_plan<T>(&s->row_inv, mphi, mrad * n_planes));
        CHECK_FFT(make_plan<T>(&s->row_fwd, mphi, mrad * n_pairs));
        std::vector<C> w(mphi);
        for (int m = 0; m < mphi; ++m) {
            const double ang = -2.0 * M_PI * (double)m / (double)mphi;
            w[m] = C{(T)std::cos(ang), (T)std::sin(ang)};
        }
        CHECK_HIP(hipMalloc(&s->w, mphi * sizeof(C)));
        CHECK_HIP(hipMemcpy(s->w, w.data(), mphi * sizeof(C), hipMemcpyHostToDevice));
    }
    for (hipfftHandle p : {s->col_inv, s->row_inv, s->row_fwd, s->col_fwd}) CHECK_FFT(hipfftSetStream(p, stream));

    const V6ColInfo col = {{(const double2*)df.typed_data(), (const double2*)phi.typed_data(),
                            kx.typed_data(), ky.typed_data(), inverse_jind.typed_data(),
                            mrad, mphi, nkx, nky, b_df, b_pot}, n_planes, n_cols};
    const V6StoreInfo store = {(double2*)out->typed_data(), inverse_jind.typed_data(), mrad, b_df, nkx, nky,
                               ixzero, iyzero, fold, out_scale_re, out_scale_im};
    const T scale = (T)(1.0 / ((double)mrad * mphi * (double)mrad * mphi));
    const BracketArgs<T> args = {(const C*)s->rows, (C*)s->z, dum_s.typed_data(), vfac.typed_data(), mrad,
                                 b_df, b_phi, b_sp, b_inner, nv, nspec, mphi, em, scale};

    const dim3 tb(GYX_T, GYX_TR);
    pack_tiled_kernel<T><<<dim3(n_planes * gyx_tiles(n_cols), gyx_tiles(mrad)), tb, 0, stream>>>(
        (C*)s->a, col, gyx_tiles(n_cols));
    CHECK_HIP(hipGetLastError());
    CHECK_FFT(Prec<T>::exec(s->col_inv, (C*)s->a, HIPFFT_BACKWARD));
#ifdef GYX_ROWFUSE
    const RowFuseArgs<T> rf = {(const C*)s->a, (C*)s->b, (const C*)s->w, dum_s.typed_data(), vfac.typed_data(),
                               mrad, n_cols, nky, b_df, b_phi, b_sp, b_inner, nv, nspec, em, 0, 0, scale};
    const bool fused_rows = gyx_rowfuse_launch<T>(rf, mphi, n_pairs, stream);
    CHECK_HIP(hipGetLastError());
#else
    const bool fused_rows = false;
#endif
    if (!fused_rows) {
        expand_tiled_kernel<T><<<dim3(n_planes * gyx_tiles(mrad), gyx_tiles(mphi)), tb, 0, stream>>>(
            (const C*)s->a, (C*)s->rows, n_cols, mrad, mphi, nky, gyx_tiles(mrad));
        CHECK_HIP(hipGetLastError());
        CHECK_FFT(Prec<T>::exec(s->row_inv, (C*)s->rows, HIPFFT_BACKWARD));
        bracket_kernel<T><<<gyx_blocks(n_z), GYX_WG, 0, stream>>>(args, (unsigned)n_z);
        CHECK_HIP(hipGetLastError());
        CHECK_FFT(Prec<T>::exec(s->row_fwd, (C*)s->z, HIPFFT_FORWARD));
        split_tiled_kernel<T><<<dim3(n_pairs * gyx_tiles(nky), gyx_tiles(mrad)), tb, 0, stream>>>(
            (const C*)s->z, (C*)s->b, mrad, mphi, nky, b_df, gyx_tiles(nky));
        CHECK_HIP(hipGetLastError());
    }
    CHECK_FFT(Prec<T>::exec(s->col_fwd, (C*)s->b, HIPFFT_FORWARD));
    unpack_kernel<T><<<gyx_blocks(n_out), GYX_WG, 0, stream>>>((const C*)s->b, store, jind.typed_data(), mrad, (unsigned)n_out);
    CHECK_HIP(hipGetLastError());
    return xla_ffi::Error::Success();
}

}  // namespace

#define BRACKET_BIND                                                              \
    xla_ffi::Ffi::Bind()                                                          \
        .Ctx<xla_ffi::PlatformStream<hipStream_t>>()                              \
        .Arg<xla_ffi::Buffer<xla_ffi::DataType::C128>>()                          \
        .Arg<xla_ffi::Buffer<xla_ffi::DataType::C128>>()                          \
        .Arg<xla_ffi::Buffer<xla_ffi::DataType::F64>>()                           \
        .Arg<xla_ffi::Buffer<xla_ffi::DataType::F64>>()                           \
        .Arg<xla_ffi::Buffer<xla_ffi::DataType::S32>>()                           \
        .Arg<xla_ffi::Buffer<xla_ffi::DataType::S32>>()                           \
        .Arg<xla_ffi::Buffer<xla_ffi::DataType::F64>>()                           \
        .Arg<xla_ffi::Buffer<xla_ffi::DataType::F64>>()                           \
        .Ret<xla_ffi::Buffer<xla_ffi::DataType::C128>>()                          \
        .Attr<int32_t>("batch").Attr<int32_t>("mrad").Attr<int32_t>("mphi")       \
        .Attr<int32_t>("nkx").Attr<int32_t>("nky").Attr<int32_t>("nspec")         \
        .Attr<int32_t>("ixzero").Attr<int32_t>("iyzero")                          \
        .Attr<int32_t>("nsp").Attr<int32_t>("nv").Attr<int32_t>("b_inner").Attr<int32_t>("em")  \
        .Attr<int32_t>("fold").Attr<double>("out_scale_re").Attr<double>("out_scale_im")        \
        .Attr<int32_t>("fused")

XLA_FFI_DEFINE_HANDLER_SYMBOL(cufft_bracket_v6_fp32_ffi, BracketImpl<float>, BRACKET_BIND);
XLA_FFI_DEFINE_HANDLER_SYMBOL(cufft_bracket_v6_fp64_ffi, BracketImpl<double>, BRACKET_BIND);
