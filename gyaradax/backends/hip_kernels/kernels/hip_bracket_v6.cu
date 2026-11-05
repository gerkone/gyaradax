// v6 bracket on hipfft, every pass explicit and every fft contiguous:
//   pack, inverse column fft along mrad, row expand, inverse row fft, bracket into pairs, forward row fft,
//   hermitian split, forward column fft, unpack
// exports the ffi symbols of the cuda v6 bracket, so the python call is shared.

#include <cmath>
#include <map>
#include <mutex>
#include <string>
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
static inline unsigned gyx_blocks(unsigned long long n) { return (unsigned)((n + GYX_WG - 1) / GYX_WG); }

// a: [n_planes][n_cols][mrad]
template <class T>
__global__ void pack_kernel(typename Prec<T>::cplx* a, const V6ColInfo ci, unsigned n) {
    GYX_INDEX(k);
    if (k >= n) return;
    const unsigned mrad = ci.z.mrad;
    const int i = k % mrad;
    const unsigned r = k / mrad;
    const int gb = r / ci.n_cols;
    const int j = v6_col_to_j(r - gb * ci.n_cols, ci.z.nky, ci.z.mphi);
    if constexpr (sizeof(T) == 4) a[k] = v5_z2z_true_fp32_pack_at(gb, i, j, &ci.z);
    else a[k] = v5_z2z_pack_at(gb, i, j, &ci.z);
}

// retained columns -> rows: [n_planes][mrad][mphi], zero in the dealiased band
template <class T>
__global__ void expand_kernel(const typename Prec<T>::cplx* __restrict__ a, typename Prec<T>::cplx* __restrict__ rows,
                              int n_cols, int mrad, int mphi, int nky, unsigned n) {
    using C = typename Prec<T>::cplx;
    GYX_INDEX(k);
    if (k >= n) return;
    const unsigned r = k / mphi;
    const int j = k - r * mphi;
    int jj = -1;
    if (j < nky) jj = j;
    else if (j > mphi - nky) jj = j - (mphi - 2 * nky + 1);
    if (jj < 0) { rows[k] = C{T(0), T(0)}; return; }
    const unsigned gb = r / mrad;
    const unsigned i = r - gb * mrad;
    rows[k] = a[(gb * n_cols + jj) * mrad + i];
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
    a.z[k] = typename Prec<T>::cplx{br0, br1};
}

// X[k] = (Z[k] + conj Z[N-k]) / 2, Y[k] = (Z[k] - conj Z[N-k]) / 2i; b: [b_df][nky][mrad]
template <class T>
__global__ void split_kernel(const typename Prec<T>::cplx* __restrict__ z, typename Prec<T>::cplx* __restrict__ b,
                             int mrad, int mphi, int nky, unsigned n) {
    using C = typename Prec<T>::cplx;
    GYX_INDEX(t);
    if (t >= n) return;
    const unsigned r = t / mrad;
    const unsigned i = t - r * mrad;
    const unsigned gb = r / nky;
    const unsigned q = r - gb * nky;
    const unsigned row = ((gb / 2) * mrad + i) * mphi;
    const C zk = z[row + q], zm = z[row + (mphi - q) % mphi];
    b[t] = (gb & 1) == 0 ? C{T(0.5) * (zk.x + zm.x), T(0.5) * (zk.y - zm.y)}
                         : C{T(0.5) * (zk.y + zm.y), T(0.5) * (zm.x - zk.x)};
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
    si.out_packed[t] = si.fold ? v6_scale_c(si.scale_re, si.scale_im, u.x, u.y) : u;
}

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
    hipfftHandle col_inv = nullptr, row_inv = nullptr, row_fwd = nullptr, col_fwd = nullptr;
    void *a = nullptr, *rows = nullptr, *z = nullptr, *b = nullptr;
    ~State() {
        for (hipfftHandle p : {col_inv, row_inv, row_fwd, col_fwd}) if (p) hipfftDestroy(p);
        for (void* x : {a, rows, z, b}) if (x) hipFree(x);
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
        CHECK_FFT(make_plan<T>(&s->row_inv, mphi, mrad * n_planes));
        CHECK_FFT(make_plan<T>(&s->row_fwd, mphi, mrad * n_pairs));
        CHECK_FFT(make_plan<T>(&s->col_fwd, mrad, b_df * nky));
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

    pack_kernel<T><<<gyx_blocks(n_a), GYX_WG, 0, stream>>>((C*)s->a, col, (unsigned)n_a);
    CHECK_HIP(hipGetLastError());
    CHECK_FFT(Prec<T>::exec(s->col_inv, (C*)s->a, HIPFFT_BACKWARD));
    expand_kernel<T><<<gyx_blocks(n_rows), GYX_WG, 0, stream>>>((const C*)s->a, (C*)s->rows, n_cols, mrad, mphi, nky, (unsigned)n_rows);
    CHECK_HIP(hipGetLastError());
    CHECK_FFT(Prec<T>::exec(s->row_inv, (C*)s->rows, HIPFFT_BACKWARD));
    bracket_kernel<T><<<gyx_blocks(n_z), GYX_WG, 0, stream>>>(args, (unsigned)n_z);
    CHECK_HIP(hipGetLastError());
    CHECK_FFT(Prec<T>::exec(s->row_fwd, (C*)s->z, HIPFFT_FORWARD));
    split_kernel<T><<<gyx_blocks(n_b), GYX_WG, 0, stream>>>((const C*)s->z, (C*)s->b, mrad, mphi, nky, (unsigned)n_b);
    CHECK_HIP(hipGetLastError());
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
