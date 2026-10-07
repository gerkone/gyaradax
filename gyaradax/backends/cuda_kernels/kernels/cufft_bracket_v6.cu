// v6 Poisson bracket: cuFFT column passes along mrad on the retained ky columns only, with the
// inverse row transform, the bracket and the forward real row transform fused in cuFFTDx kernels.
//
//   P1   cuFFT C2C inverse along mrad, A = [mrad][planes][2 nky - 1], packed by a load callback (FP64)
//        or by an explicit pack kernel (FP32)
//   rows cuFFTDx: potential rows -> real space; df row pairs -> inverse, bracket, 2-for-1 forward
//   P3   cuFFT C2C forward along mrad, B = [mrad][b_df][nky], unpacked to [b_df, nkx, nky] by a store
//        callback (FP64) or an explicit unpack kernel (FP32); both forms give bit-identical results
//
// Same FFI signature as the v5 bracket; unsupported sizes run the v5 pipeline.

#include <cmath>
#include <cstdio>
#include <map>
#include <mutex>
#include <string>
#include <utility>
#include "xla/ffi/api/ffi.h"
#include <cuda_runtime.h>
#include <cufft.h>
#include <cufftXt.h>
#include <cufftdx.hpp>

#include "bracket_v5_lto_fatbins_decl.h"
#include "bracket_v6_info.cuh"
#include "gyx_dx_archs.h"

namespace xla_ffi = xla::ffi;

// v5 pipelines, used for configurations v6 does not instantiate
xla_ffi::Error CufftGraphBracketTrueFp32Impl(
    cudaStream_t, xla_ffi::Buffer<xla_ffi::DataType::C128>, xla_ffi::Buffer<xla_ffi::DataType::C128>,
    xla_ffi::Buffer<xla_ffi::DataType::F64>, xla_ffi::Buffer<xla_ffi::DataType::F64>,
    xla_ffi::Buffer<xla_ffi::DataType::S32>, xla_ffi::Buffer<xla_ffi::DataType::S32>,
    xla_ffi::Buffer<xla_ffi::DataType::F64>, xla_ffi::Buffer<xla_ffi::DataType::F64>,
    xla_ffi::Result<xla_ffi::Buffer<xla_ffi::DataType::C128>>,
    int32_t, int32_t, int32_t, int32_t, int32_t, int32_t, int32_t, int32_t, int32_t, int32_t, int32_t, int32_t);
xla_ffi::Error CufftGraphBracketFp64Impl(
    cudaStream_t, xla_ffi::Buffer<xla_ffi::DataType::C128>, xla_ffi::Buffer<xla_ffi::DataType::C128>,
    xla_ffi::Buffer<xla_ffi::DataType::F64>, xla_ffi::Buffer<xla_ffi::DataType::F64>,
    xla_ffi::Buffer<xla_ffi::DataType::S32>, xla_ffi::Buffer<xla_ffi::DataType::S32>,
    xla_ffi::Buffer<xla_ffi::DataType::F64>, xla_ffi::Buffer<xla_ffi::DataType::F64>,
    xla_ffi::Result<xla_ffi::Buffer<xla_ffi::DataType::C128>>,
    int32_t, int32_t, int32_t, int32_t, int32_t, int32_t, int32_t, int32_t, int32_t, int32_t, int32_t, int32_t);

namespace {

template <unsigned... Ns> struct V6List {};
using V6Archs = V6List<GYX_DX_ARCHS>;
using V6Sizes = V6List<16, 32, 36, 40, 64, 96, 128, 144, 192>;

template <class T> struct V6Prec;
template <> struct V6Prec<float> {
    using cplx = float2;
    static constexpr cudaDataType data_type = CUDA_C_32F;
    static constexpr cufftXtCallbackType load_type = CUFFT_CB_LD_COMPLEX;
    static constexpr cufftXtCallbackType store_type = CUFFT_CB_ST_COMPLEX;
    static constexpr const char* load_name = "d_v6_col_true_fp32_load";
    static constexpr const char* store_name = "d_v6_store_fp32_cb";
};
template <> struct V6Prec<double> {
    using cplx = double2;
    static constexpr cudaDataType data_type = CUDA_C_64F;
    static constexpr cufftXtCallbackType load_type = CUFFT_CB_LD_COMPLEX_DOUBLE;
    static constexpr cufftXtCallbackType store_type = CUFFT_CB_ST_COMPLEX_DOUBLE;
    static constexpr const char* load_name = "d_v6_col_fp64_load";
    static constexpr const char* store_name = "d_v6_store_fp64_cb";
};

template <class T>
struct V6RowArgs {
    const typename V6Prec<T>::cplx* a;      // [mrad][n_planes][2 nky - 1]
    typename V6Prec<T>::cplx*       preal;  // [mrad][b_pot][N] real-space potential rows
    typename V6Prec<T>::cplx*       b;      // [mrad][b_df][nky]
    const double* dum_s;
    const double* vfac;
    int mrad, n_planes, b_df, b_pot, b_phi, b_sp, b_inner, nv, nspec, nky, em;
    T scale;
};

template <unsigned Arch, unsigned N, class T>
struct V6Fft {
    using Base = decltype(cufftdx::Size<N>() + cufftdx::Precision<T>() +
                          cufftdx::Type<cufftdx::fft_type::c2c>() + cufftdx::Block() + cufftdx::SM<Arch>());
    using Probe = decltype(Base() + cufftdx::Direction<cufftdx::fft_direction::inverse>() +
                           cufftdx::FFTsPerBlock<1>());
    static constexpr unsigned fpb = Probe::block_dim.x >= 128 ? 1 : 128 / Probe::block_dim.x;
    using Inv = decltype(Base() + cufftdx::Direction<cufftdx::fft_direction::inverse>() +
                         cufftdx::ElementsPerThread<Probe::elements_per_thread>() + cufftdx::FFTsPerBlock<fpb>());
    using Fwd = decltype(Base() + cufftdx::Direction<cufftdx::fft_direction::forward>() +
                         cufftdx::ElementsPerThread<Probe::elements_per_thread>() + cufftdx::FFTsPerBlock<fpb>());
    using C = typename Inv::value_type;
    static constexpr size_t fft_smem =
        Inv::shared_memory_size > Fwd::shared_memory_size ? Inv::shared_memory_size : Fwd::shared_memory_size;
    static constexpr size_t smem_pot = fft_smem;
    static constexpr size_t smem_df  = ((fft_smem + 15) / 16) * 16 + (size_t)fpb * N * sizeof(C);
};

// expand a retained-column row into the cuFFTDx thread layout (zeros in the dealiased ky band)
template <class Inv, unsigned N, class C, class Cplx>
__device__ __forceinline__ void v6_load_row(C* x, const Cplx* src, int nky)
{
    for (unsigned e = 0; e < Inv::elements_per_thread; ++e) {
        const int j = (int)(threadIdx.x + e * Inv::stride);
        int jj = -1;
        if (j < nky) jj = j;
        else if (j > (int)N - nky && j < (int)N) jj = j - ((int)N - 2 * nky + 1);
        const Cplx v = jj >= 0 ? src[jj] : Cplx{0, 0};
        x[e] = C{v.x, v.y};
    }
}

template <unsigned Arch, unsigned N, class T>
__global__ __launch_bounds__(V6Fft<Arch, N, T>::Inv::max_threads_per_block)
void v6_pot_rows_kernel(const V6RowArgs<T> args)
{
#if defined(__CUDA_ARCH__)
    if constexpr (Arch == __CUDA_ARCH__) {
        using F = V6Fft<Arch, N, T>;
        using Inv = typename F::Inv;
        using C = typename F::C;
        using Cplx = typename V6Prec<T>::cplx;
        extern __shared__ __align__(16) unsigned char smem[];
        const int n_rows = args.mrad * args.b_pot;
        int row = (int)(blockIdx.x * Inv::ffts_per_block + threadIdx.y);
        const bool live = row < n_rows;
        if (!live) row = 0;
        const int i = row / args.b_pot, pb = row - i * args.b_pot;
        C x[Inv::storage_size];
        v6_load_row<Inv, N>(x, args.a + ((size_t)i * args.n_planes + args.b_df + pb) * (2 * args.nky - 1),
                            args.nky);
        Inv().execute(x, reinterpret_cast<C*>(smem));
        if (live) {
            Cplx* dst = args.preal + ((size_t)i * args.b_pot + pb) * N;
            for (unsigned e = 0; e < Inv::elements_per_thread; ++e) {
                const unsigned j = threadIdx.x + e * Inv::stride;
                if (j < N) dst[j] = Cplx{x[e].x, x[e].y};
            }
        }
    }
#endif
}

template <unsigned Arch, unsigned N, class T>
__global__ __launch_bounds__(V6Fft<Arch, N, T>::Inv::max_threads_per_block)
void v6_df_rows_kernel(const V6RowArgs<T> args)
{
#if defined(__CUDA_ARCH__)
    if constexpr (Arch == __CUDA_ARCH__) {
        using F = V6Fft<Arch, N, T>;
        using Inv = typename F::Inv;
        using Fwd = typename F::Fwd;
        using C = typename F::C;
        using Cplx = typename V6Prec<T>::cplx;
        extern __shared__ __align__(16) unsigned char smem[];
        C* fft_smem = reinterpret_cast<C*>(smem);
        C* xch = reinterpret_cast<C*>(smem + ((F::fft_smem + 15) / 16) * 16) + threadIdx.y * N;

        const int n_pairs = args.mrad * args.b_df / 2;
        int pair = (int)(blockIdx.x * Inv::ffts_per_block + threadIdx.y);
        const bool live = pair < n_pairs;
        if (!live) pair = 0;
        const int i = (2 * pair) / args.b_df;
        const int gb0 = 2 * pair - i * args.b_df;
        const int n_cols = 2 * args.nky - 1;

        // bracket rows of the two planes gb0, gb0 + 1, packed as the real/imag part of one row
        C z[Inv::storage_size];
        for (int q = 0; q < 2; ++q) {
            const int gb = gb0 + q;
            C x[Inv::storage_size];
            v6_load_row<Inv, N>(x, args.a + ((size_t)i * args.n_planes + gb) * n_cols, args.nky);
            Inv().execute(x, fft_smem);
            const int pidx = (gb / args.b_sp) * args.b_inner + gb % args.b_inner;
            const Cplx* pa = args.preal + ((size_t)i * args.b_pot + pidx) * N;
            const Cplx* pv = pa + (size_t)args.b_phi * N;
            const T w = args.em ? (T)args.vfac[(gb / args.b_sp) * args.nv + (gb % args.b_sp) / args.b_inner] : T(0);
            const T dum = (T)args.dum_s[gb % args.nspec];
            for (unsigned e = 0; e < Inv::elements_per_thread; ++e) {
                const unsigned j = threadIdx.x + e * Inv::stride;
                T br = T(0);
                if (j < N) {
                    Cplx p = pa[j];
                    if (args.em) {
                        const Cplx pq = pv[j];
                        p = Cplx{p.x + w * pq.x, p.y + w * pq.y};
                    }
                    br = args.scale * dum * (p.x * x[e].y - p.y * x[e].x);
                }
                if (q == 0) z[e].x = br; else z[e].y = br;
            }
        }
        Fwd().execute(z, fft_smem);

        // Hermitian split X[k] = (Z[k] + conj Z[N-k]) / 2, Y[k] = (Z[k] - conj Z[N-k]) / 2i
        for (unsigned e = 0; e < Inv::elements_per_thread; ++e) {
            const unsigned j = threadIdx.x + e * Inv::stride;
            if (j < N) xch[j] = z[e];
        }
        __syncthreads();
        if (live) {
            Cplx* out0 = args.b + ((size_t)i * args.b_df + gb0) * args.nky;
            Cplx* out1 = out0 + args.nky;
            for (unsigned e = 0; e < Inv::elements_per_thread; ++e) {
                const int k = (int)(threadIdx.x + e * Inv::stride);
                if (k < args.nky) {
                    const C zk = xch[k];
                    const C zm = xch[(N - k) % N];
                    out0[k] = Cplx{T(0.5) * (zk.x + zm.x), T(0.5) * (zk.y - zm.y)};
                    out1[k] = Cplx{T(0.5) * (zk.y + zm.y), T(0.5) * (zm.x - zk.x)};
                }
            }
        }
    }
#endif
}

template <unsigned Arch, unsigned N, class T>
cudaError_t v6_launch_rows(const V6RowArgs<T>& args, cudaStream_t stream)
{
    using F = V6Fft<Arch, N, T>;
    const unsigned fpb = F::Inv::ffts_per_block;
    cudaError_t err;
    if ((err = cudaFuncSetAttribute(v6_pot_rows_kernel<Arch, N, T>,
                                    cudaFuncAttributeMaxDynamicSharedMemorySize, (int)F::smem_pot)) != cudaSuccess)
        return err;
    if ((err = cudaFuncSetAttribute(v6_df_rows_kernel<Arch, N, T>,
                                    cudaFuncAttributeMaxDynamicSharedMemorySize, (int)F::smem_df)) != cudaSuccess)
        return err;
    const unsigned pot_rows = (unsigned)args.mrad * args.b_pot;
    const unsigned pairs = (unsigned)args.mrad * args.b_df / 2;
    v6_pot_rows_kernel<Arch, N, T><<<(pot_rows + fpb - 1) / fpb, F::Inv::block_dim, F::smem_pot, stream>>>(args);
    v6_df_rows_kernel<Arch, N, T><<<(pairs + fpb - 1) / fpb, F::Inv::block_dim, F::smem_df, stream>>>(args);
    return cudaGetLastError();
}

template <class T, unsigned Arch, unsigned N, unsigned... Ns>
bool v6_dispatch_size(unsigned mphi, const V6RowArgs<T>* args, cudaStream_t stream, cudaError_t* err)
{
    if (mphi == N) {
        if (args) *err = v6_launch_rows<Arch, N, T>(*args, stream);
        return true;
    }
    if constexpr (sizeof...(Ns) > 0) return v6_dispatch_size<T, Arch, Ns...>(mphi, args, stream, err);
    return false;
}

template <class T, unsigned Arch, unsigned... Ns>
bool v6_dispatch_size_list(V6List<Ns...>, unsigned mphi, const V6RowArgs<T>* args, cudaStream_t stream,
                           cudaError_t* err)
{
    return v6_dispatch_size<T, Arch, Ns...>(mphi, args, stream, err);
}

// run (or, with args == nullptr, only check) the row kernels for the device architecture and mphi
template <class T, unsigned A, unsigned... As>
bool v6_dispatch(unsigned arch, unsigned mphi, const V6RowArgs<T>* args, cudaStream_t stream, cudaError_t* err)
{
    if (arch == A) return v6_dispatch_size_list<T, A>(V6Sizes{}, mphi, args, stream, err);
    if constexpr (sizeof...(As) > 0) return v6_dispatch<T, As...>(arch, mphi, args, stream, err);
    return false;
}

template <class T, unsigned... As>
bool v6_dispatch_list(V6List<As...>, unsigned arch, unsigned mphi, const V6RowArgs<T>* args,
                      cudaStream_t stream, cudaError_t* err)
{
    return v6_dispatch<T, As...>(arch, mphi, args, stream, err);
}

template <class T>
__global__ void v6_col_pack_kernel(typename V6Prec<T>::cplx* a, const V6ColInfo ci, unsigned long long n)
{
    for (unsigned long long k = blockIdx.x * (unsigned long long)blockDim.x + threadIdx.x; k < n;
         k += (unsigned long long)gridDim.x * blockDim.x) {
        int gb, i, j;
        v6_col_split(k, &ci, &gb, &i, &j);
        if constexpr (sizeof(T) == 4) {
            const float2 v = v5_z2z_true_fp32_pack_at(gb, i, j, &ci.z);
            a[k] = v;
        } else {
            a[k] = v5_z2z_pack_at(gb, i, j, &ci.z);
        }
    }
}

template <class T>
__global__ void v6_unpack_kernel(const typename V6Prec<T>::cplx* __restrict__ b, const V6StoreInfo si,
                                 const int* __restrict__ jind)
{
    // one thread per packed output element (gb, i_pack, k), k fastest
    const size_t n = (size_t)si.n_df * si.nkx * si.nky;
    for (size_t t = blockIdx.x * (size_t)blockDim.x + threadIdx.x; t < n; t += (size_t)gridDim.x * blockDim.x) {
        const int k = (int)(t % si.nky);
        const size_t r = t / si.nky;
        const int i_pack = (int)(r % si.nkx);
        const int gb = (int)(r / si.nkx);
        const typename V6Prec<T>::cplx v = b[((size_t)__ldg(&jind[i_pack]) * si.n_df + gb) * si.nky + k];
        si.out_packed[t] = (i_pack == si.ixzero && k == si.iyzero) ? make_double2(0.0, 0.0)
                                                                    : make_double2((double)v.x, (double)v.y);
    }
}

struct V6Key {
    int device, b_df, b_pot, mrad, mphi, nkx, nky, em, fp64;
    bool operator<(const V6Key& o) const {
        const int l[9] = {device, b_df, b_pot, mrad, mphi, nkx, nky, em, fp64};
        const int r[9] = {o.device, o.b_df, o.b_pot, o.mrad, o.mphi, o.nkx, o.nky, o.em, o.fp64};
        for (int k = 0; k < 9; ++k)
            if (l[k] != r[k]) return l[k] < r[k];
        return false;
    }
};

struct V6State {
    cufftHandle plan_col = 0, plan_out = 0;
    void *a = nullptr, *preal = nullptr, *b = nullptr;
    V6ColInfo*   d_col = nullptr;   void* d_col_ptr = nullptr;
    V6StoreInfo* d_store = nullptr; void* d_store_ptr = nullptr;
    bool explicit_pack = false;    // P1 packed by v6_col_pack_kernel instead of the load callback
    bool explicit_unpack = false;  // P3 unpacked by v6_unpack_kernel instead of the store callback
    ~V6State() {
        if (plan_col) cufftDestroy(plan_col);
        if (plan_out) cufftDestroy(plan_out);
        if (a) cudaFree(a);
        if (preal) cudaFree(preal);
        if (b) cudaFree(b);
        if (d_col) cudaFree(d_col);
        if (d_store) cudaFree(d_store);
    }
};

static std::map<V6Key, V6State*> g_v6_cache;
static std::mutex g_v6_mutex;

#define V6_CHECK_CUDA(call) do { cudaError_t err_ = (call); \
    if (err_ != cudaSuccess) return xla_ffi::Error::Internal(std::string("CUDA ") + cudaGetErrorString(err_)); \
} while (0)
#define V6_CHECK_CUFFT(call) do { cufftResult res_ = (call); \
    if (res_ != CUFFT_SUCCESS) return xla_ffi::Error::Internal(std::string("cuFFT code ") + std::to_string((int)res_)); \
} while (0)

template <class T>
xla_ffi::Error BracketV6Impl(
    cudaStream_t stream,
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
    int32_t ixzero, int32_t iyzero, int32_t nsp, int32_t nv, int32_t b_inner, int32_t em)
{
    using Cplx = typename V6Prec<T>::cplx;
    constexpr bool fp64 = sizeof(T) == 8;
    int device = 0;
    cudaGetDevice(&device);
    int major = 0, minor = 0;
    cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device);
    cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, device);
    const unsigned arch = (unsigned)(major * 100 + minor * 10);

    const int b_df = batch * nspec;
    size_t phi_elems = 1;
    for (auto d : phi.dimensions()) phi_elems *= d;
    const int b_pot = (int)(phi_elems / ((size_t)nkx * nky));
    const int b_phi = em ? b_pot / 2 : b_pot;
    cudaError_t no_err = cudaSuccess;
    const bool supported = b_df % 2 == 0 && 2 * nky - 1 <= mphi && nsp > 0 && b_inner > 0
        && v6_dispatch_list<T>(V6Archs{}, arch, (unsigned)mphi, nullptr, stream, &no_err);
    if (!supported) {
        if constexpr (fp64)
            return CufftGraphBracketFp64Impl(stream, df, phi, kx, ky, jind, inverse_jind, dum_s, vfac, out,
                batch, mrad, mphi, nkx, nky, nspec, ixzero, iyzero, nsp, nv, b_inner, em);
        else
            return CufftGraphBracketTrueFp32Impl(stream, df, phi, kx, ky, jind, inverse_jind, dum_s, vfac, out,
                batch, mrad, mphi, nkx, nky, nspec, ixzero, iyzero, nsp, nv, b_inner, em);
    }
    if (b_df % nsp != 0 || (b_df / nsp) % b_inner != 0 || b_phi != nsp * b_inner || (em && b_pot != 2 * b_phi))
        return xla_ffi::Error(XLA_FFI_Error_Code_INVALID_ARGUMENT,
            "bracket: potential planes inconsistent with nsp/b_inner");
    const int b_sp = b_df / nsp;
    const int n_planes = b_df + b_pot;
    const int n_cols = 2 * nky - 1;
    const size_t n_a = (size_t)mrad * n_planes * n_cols;

    V6Key key = {device, b_df, b_pot, mrad, mphi, nkx, nky, em, fp64};
    std::lock_guard<std::mutex> lock(g_v6_mutex);
    V6State* s = g_v6_cache[key];

    V6ColInfo h_col = {{(const double2*)df.typed_data(), (const double2*)phi.typed_data(),
                        kx.typed_data(), ky.typed_data(), inverse_jind.typed_data(),
                        mrad, mphi, nkx, nky, b_df, b_pot}, n_planes, n_cols};
    V6StoreInfo h_store = {(double2*)out->typed_data(), inverse_jind.typed_data(),
                           mrad, b_df, nkx, nky, ixzero, iyzero};

    if (!s) {
        s = new V6State();
        g_v6_cache[key] = s;
        V6_CHECK_CUDA(cudaMalloc(&s->a, n_a * sizeof(Cplx)));
        V6_CHECK_CUDA(cudaMalloc(&s->preal, (size_t)mrad * b_pot * mphi * sizeof(Cplx)));
        V6_CHECK_CUDA(cudaMalloc(&s->b, (size_t)mrad * b_df * nky * sizeof(Cplx)));
        V6_CHECK_CUDA(cudaMalloc(&s->d_col, sizeof(V6ColInfo)));     s->d_col_ptr = (void*)s->d_col;
        V6_CHECK_CUDA(cudaMalloc(&s->d_store, sizeof(V6StoreInfo))); s->d_store_ptr = (void*)s->d_store;

        long long n_ll[1] = {mrad};
        long long emb[1] = {mrad};
        size_t ws = 0;
        // FP32: explicit pack/unpack kernels and plain plans; FP64: cuFFT callbacks
        s->explicit_pack = s->explicit_unpack = !fp64;
        const long long col_stride = (long long)n_planes * n_cols;
        V6_CHECK_CUFFT(cufftCreate(&s->plan_col));
        if (!s->explicit_pack)
            V6_CHECK_CUFFT(cufftXtSetJITCallback(s->plan_col, V6Prec<T>::load_name,
                (void*)bracket_v5_z2z_load_cb_fatbin, bracket_v5_z2z_load_cb_fatbin_bytes,
                V6Prec<T>::load_type, &s->d_col_ptr));
        V6_CHECK_CUFFT(cufftXtMakePlanMany(s->plan_col, 1, n_ll, emb, col_stride, 1, V6Prec<T>::data_type,
            emb, col_stride, 1, V6Prec<T>::data_type, col_stride, &ws, V6Prec<T>::data_type));
        V6_CHECK_CUFFT(cufftSetStream(s->plan_col, stream));

        if (!s->explicit_pack) {
            // fall back to the explicit pack if cuFFT skips the load callback (probed on a NaN-filled A)
            V6_CHECK_CUDA(cudaMemcpyAsync(s->d_col, &h_col, sizeof(V6ColInfo), cudaMemcpyHostToDevice, stream));
            V6_CHECK_CUDA(cudaMemsetAsync(s->a, 0xFF, n_a * sizeof(Cplx), stream));
            V6_CHECK_CUFFT(cufftXtExec(s->plan_col, s->a, s->a, CUFFT_INVERSE));
            Cplx probe[2];
            V6_CHECK_CUDA(cudaMemcpyAsync(&probe[0], s->a, sizeof(Cplx), cudaMemcpyDeviceToHost, stream));
            V6_CHECK_CUDA(cudaMemcpyAsync(&probe[1], (Cplx*)s->a + n_a - 1, sizeof(Cplx),
                                          cudaMemcpyDeviceToHost, stream));
            V6_CHECK_CUDA(cudaStreamSynchronize(stream));
            if (!(std::isfinite((double)probe[0].x) && std::isfinite((double)probe[1].x))) {
                V6_CHECK_CUFFT(cufftDestroy(s->plan_col));
                V6_CHECK_CUFFT(cufftCreate(&s->plan_col));
                V6_CHECK_CUFFT(cufftXtMakePlanMany(s->plan_col, 1, n_ll, emb, col_stride, 1, V6Prec<T>::data_type,
                    emb, col_stride, 1, V6Prec<T>::data_type, col_stride, &ws, V6Prec<T>::data_type));
                s->explicit_pack = true;
            }
        }

        const long long out_stride = (long long)b_df * nky;
        V6_CHECK_CUFFT(cufftCreate(&s->plan_out));
        if (!s->explicit_unpack)
            V6_CHECK_CUFFT(cufftXtSetJITCallback(s->plan_out, V6Prec<T>::store_name,
                (void*)bracket_v5_store_cb_fatbin, bracket_v5_store_cb_fatbin_bytes,
                V6Prec<T>::store_type, &s->d_store_ptr));
        V6_CHECK_CUFFT(cufftXtMakePlanMany(s->plan_out, 1, n_ll, emb, out_stride, 1, V6Prec<T>::data_type,
            emb, out_stride, 1, V6Prec<T>::data_type, out_stride, &ws, V6Prec<T>::data_type));
    }

    V6_CHECK_CUFFT(cufftSetStream(s->plan_col, stream));
    V6_CHECK_CUFFT(cufftSetStream(s->plan_out, stream));
    V6_CHECK_CUDA(cudaMemcpyAsync(s->d_col, &h_col, sizeof(V6ColInfo), cudaMemcpyHostToDevice, stream));
    V6_CHECK_CUDA(cudaMemcpyAsync(s->d_store, &h_store, sizeof(V6StoreInfo), cudaMemcpyHostToDevice, stream));

    if (s->explicit_pack) {
        const unsigned long long blocks = (n_a + 255) / 256;
        v6_col_pack_kernel<T><<<(unsigned)(blocks < 65535ull * 64 ? blocks : 65535ull * 64), 256, 0, stream>>>(
            (Cplx*)s->a, h_col, n_a);
        V6_CHECK_CUDA(cudaGetLastError());
    }
    V6_CHECK_CUFFT(cufftXtExec(s->plan_col, s->a, s->a, CUFFT_INVERSE));

    const T scale = (T)(1.0 / ((double)mrad * mphi * (double)mrad * mphi));
    V6RowArgs<T> args = {(const Cplx*)s->a, (Cplx*)s->preal, (Cplx*)s->b, dum_s.typed_data(), vfac.typed_data(),
                         mrad, n_planes, b_df, b_pot, b_phi, b_sp, b_inner, nv, nspec, nky, em, scale};
    cudaError_t err = cudaSuccess;
    v6_dispatch_list<T>(V6Archs{}, arch, (unsigned)mphi, &args, stream, &err);
    V6_CHECK_CUDA(err);

    V6_CHECK_CUFFT(cufftXtExec(s->plan_out, s->b, s->b, CUFFT_FORWARD));
    if (s->explicit_unpack) {
        const size_t n = (size_t)b_df * nkx * nky;
        const unsigned long long blocks = (n + 255) / 256;
        v6_unpack_kernel<T><<<(unsigned)(blocks < 65535ull * 64 ? blocks : 65535ull * 64), 256, 0, stream>>>(
            (const Cplx*)s->b, h_store, jind.typed_data());
        V6_CHECK_CUDA(cudaGetLastError());
    }
    return xla_ffi::Error::Success();
}

}  // namespace

#define V6_BIND                                                                   \
    xla_ffi::Ffi::Bind()                                                          \
        .Ctx<xla_ffi::PlatformStream<cudaStream_t>>()                             \
        .Arg<xla_ffi::Buffer<xla_ffi::DataType::C128>>()                          \
        .Arg<xla_ffi::Buffer<xla_ffi::DataType::C128>>()                          \
        .Arg<xla_ffi::Buffer<xla_ffi::DataType::F64>>()                           \
        .Arg<xla_ffi::Buffer<xla_ffi::DataType::F64>>()                           \
        .Arg<xla_ffi::Buffer<xla_ffi::DataType::S32>>()                           \
        .Arg<xla_ffi::Buffer<xla_ffi::DataType::S32>>()                           \
        .Arg<xla_ffi::Buffer<xla_ffi::DataType::F64>>()                           \
        .Arg<xla_ffi::Buffer<xla_ffi::DataType::F64>>()                           \
        .Ret<xla_ffi::Buffer<xla_ffi::DataType::C128>>()                          \
        .Attr<int32_t>("batch").Attr<int32_t>("mrad").Attr<int32_t>("mphi")      \
        .Attr<int32_t>("nkx").Attr<int32_t>("nky").Attr<int32_t>("nspec")        \
        .Attr<int32_t>("ixzero").Attr<int32_t>("iyzero")                         \
        .Attr<int32_t>("nsp").Attr<int32_t>("nv").Attr<int32_t>("b_inner").Attr<int32_t>("em")

XLA_FFI_DEFINE_HANDLER_SYMBOL(cufft_bracket_v6_fp32_ffi, BracketV6Impl<float>, V6_BIND);
XLA_FFI_DEFINE_HANDLER_SYMBOL(cufft_bracket_v6_fp64_ffi, BracketV6Impl<double>, V6_BIND);
