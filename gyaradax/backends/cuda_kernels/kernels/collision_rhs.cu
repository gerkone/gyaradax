#include "xla/ffi/api/ffi.h"
#include <cuda_runtime.h>
#include <algorithm>

namespace xla_ffi = xla::ffi;

// 9-point (vpar, mu) collision stencil, one thread per output point

// (dv, dmu) of shift k in collisions._SHIFTS order
__host__ __device__ constexpr int coll_dv(int k) { return k == 0 ? 0 : k < 3 ? (k == 1 ? 1 : -1) : k < 5 ? 0 : ((k & 1) ? 1 : -1); }
__host__ __device__ constexpr int coll_dmu(int k) { return k < 3 ? 0 : k < 5 ? (k == 3 ? 1 : -1) : (k < 7 ? 1 : -1); }

constexpr int COLL_VCH = 4;

__global__ __launch_bounds__(256) void collision_rhs_kernel(
    const double2* __restrict__ df,       // (nsp, nv, nmu, ns, nkx, nky)
    const double* __restrict__ stencil,   // (nsp, 9, nv, nmu, ns)
    double2* __restrict__ out,
    int nsp, int nv, int nmu, int ns, int nxy)
{
    // one thread per (sp, mu, vpar chunk, spatial); the chunk's (vpar, mu) neighbourhood is loaded once
    const int nspat = ns * nxy;
    const int spatial = blockIdx.x * blockDim.x + threadIdx.x;
    if (spatial >= nspat) return;
    const int n_vch = (nv + COLL_VCH - 1) / COLL_VCH;
    const int vch = blockIdx.y % n_vch;
    const int mu = (blockIdx.y / n_vch) % nmu;
    const int sp = blockIdx.y / (n_vch * nmu);
    const int s = spatial / nxy;
    const int v0 = vch * COLL_VCH;
    const size_t plane = (size_t)nmu * nspat;
    const double2* f_sp = df + (size_t)sp * nv * plane + spatial;
    const double* st = stencil + (size_t)sp * 9 * nv * nmu * ns + (size_t)mu * ns + s;
    const size_t cstride = (size_t)nv * nmu * ns;
    const double2 z = make_double2(0.0, 0.0);
    double2 w[COLL_VCH + 2][3];
    #pragma unroll
    for (int i = 0; i < COLL_VCH + 2; ++i) {
        const int v = v0 - 1 + i;
        #pragma unroll
        for (int j = 0; j < 3; ++j) {
            const int m = mu - 1 + j;
            w[i][j] = (v >= 0 && v < nv && m >= 0 && m < nmu) ? __ldg(&f_sp[(size_t)v * plane + (size_t)m * nspat]) : z;
        }
    }
    #pragma unroll
    for (int i = 0; i < COLL_VCH; ++i) {
        const int v = v0 + i;
        if (v >= nv) break;
        const double* c = st + (size_t)v * nmu * ns;
        double2 acc = z;
        #pragma unroll
        for (int k = 0; k < 9; ++k) {
            const double2 f = w[i + 1 + coll_dv(k)][1 + coll_dmu(k)];
            const double ck = __ldg(&c[k * cstride]);
            acc.x += ck * f.x;
            acc.y += ck * f.y;
        }
        out[(size_t)sp * nv * plane + (size_t)v * plane + (size_t)mu * nspat + spatial] = acc;
    }
}

xla_ffi::Error CollisionRhsImpl(
    cudaStream_t stream,
    xla_ffi::Buffer<xla_ffi::DataType::C128> df,
    xla_ffi::Buffer<xla_ffi::DataType::F64> stencil,
    xla_ffi::Result<xla_ffi::Buffer<xla_ffi::DataType::C128>> out,
    int32_t nsp, int32_t nv, int32_t nmu, int32_t ns, int32_t nxy)
{
    const int threads = 256;
    const dim3 grid((unsigned)((ns * nxy + threads - 1) / threads),
                    (unsigned)(nsp * nmu * ((nv + COLL_VCH - 1) / COLL_VCH)));
    collision_rhs_kernel<<<grid, threads, 0, stream>>>(
        reinterpret_cast<const double2*>(df.untyped_data()), stencil.typed_data(),
        reinterpret_cast<double2*>(out->untyped_data()), nsp, nv, nmu, ns, nxy);
    const cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
        return xla_ffi::Error(XLA_FFI_Error_Code_INTERNAL, cudaGetErrorString(err));
    return xla_ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    collision_rhs_ffi, CollisionRhsImpl,
    xla_ffi::Ffi::Bind()
        .Ctx<xla_ffi::PlatformStream<cudaStream_t>>()
        .Arg<xla_ffi::Buffer<xla_ffi::DataType::C128>>() // df
        .Arg<xla_ffi::Buffer<xla_ffi::DataType::F64>>()  // stencil
        .Ret<xla_ffi::Buffer<xla_ffi::DataType::C128>>() // out
        .Attr<int32_t>("nsp")
        .Attr<int32_t>("nv")
        .Attr<int32_t>("nmu")
        .Attr<int32_t>("ns")
        .Attr<int32_t>("nxy")
);
