#include "xla/ffi/api/ffi.h"
#include <cuda_runtime.h>

#include "ds_arith.cuh"

namespace xla_ffi = xla::ffi;

// electrostatic linear rhs in double-single arithmetic, one block per (sp, mu, kx, vpar chunk)
struct LinearRhsDsArgs {
    const double2* df;        // (nsp, nv, nmu, ns, nkx, nky)
    const double2* phi;       // (ns, nkx, nky)
    const ds* bessel;         // (nsp, nmu, ns, nkx, nky)
    const ds* s_upar_tab;     // (nsp, nv, ns, n_class, 9)
    const ds* s_t7_tab;       // (nsp, nv, nmu, ns, n_class, 9)
    const int* par_class;     // (ns, nkx, nky)
    const int2* packed_maps;  // (9, ns, nkx, nky)
    const ds* wvp;            // (nsp, nmu, ns, 5)
    const ds* drift_x;        // (nsp, nv, nmu, ns)
    const ds* drift_y;        // (nsp, nv, nmu, ns)
    const ds* drive_a;        // (nsp, nv, nmu, ns, nky)
    const ds* drive_b;        // (nsp, nv, nmu, ns)
    const ds* hyper;          // (nkx, nky)
    const ds* kx_vals;        // (nkx,)
    const ds* ky_vals;        // (nky,)
    const ds* coll;           // (nsp, 9, nv, nmu, ns) collision stencil, when COLL
    double2* rhs_out;
    // rk mode (acc_out set): acc_out = acc_in + coef[0] k, stage_out = prev + coef[1] k, k = rhs + extra
    const double2* prev;
    const double2* acc_in;
    const double2* extra;
    const double* coef;
    double2* acc_out;
    double2* stage_out;
    int nsp, nv, nmu, ns, nkx, nky, n_class;
};

constexpr int TAP_LO = 2;
constexpr int TAP_HI = 7;
constexpr int TAP_OFFKX = 1 << 30;
constexpr int RING = 5;
constexpr int MAX_THREADS = 576;

template <int VCH, bool COLL>
__global__ __launch_bounds__(MAX_THREADS, 2) void linear_rhs_ds_kernel(const LinearRhsDsArgs a) {
    const int ns = a.ns, nky = a.nky, nv = a.nv, nmu = a.nmu, nkx = a.nkx;
    const int nt = ns * nky;

    extern __shared__ unsigned char smem_raw[];
    cds* ring = reinterpret_cast<cds*>(smem_raw);
    cds* sm_gy = ring + RING * nt;

    const int tid = threadIdx.x;
    const int n_vch = nv / VCH;
    const int vch = blockIdx.x % n_vch;
    const int mu = (blockIdx.x / n_vch) % nmu;
    const int kx = (blockIdx.x / (n_vch * nmu)) % nkx;
    const int sp = blockIdx.x / (n_vch * nmu * nkx);
    const int s = tid / nky;
    const int ky = tid % nky;
    const int v0 = vch * VCH;

    const int nspat = ns * nkx * nky;
    const int spatial = s * (nkx * nky) + kx * nky + ky;
    const int cls = __ldg(&a.par_class[spatial]);
    const int plane = nmu * nspat;
    const int base0 = ((sp * nv) * nmu + mu) * nspat + spatial;
    const ds* bessel_mu = a.bessel + (sp * nmu + mu) * nspat;

    // same-kx taps index the shared planes, other-kx taps are flagged global indices
    int tap[TAP_HI - TAP_LO];
    #pragma unroll
    for (int i = 0; i < TAP_HI - TAP_LO; ++i) {
        const int2 m = __ldg(&a.packed_maps[(TAP_LO + i) * nspat + spatial]);
        tap[i] = m.x < 0 ? -1
               : (m.y == kx ? m.x * nky + ky : (TAP_OFFKX | (m.x * (nkx * nky) + m.y * nky + ky)));
    }
    const ds bes = ds_ldg(&bessel_mu[spatial]);
    const cds phi_val = cds_load(a.phi, spatial);
    const cds my_gyro = {ds_mul(bes, phi_val.x), ds_mul(bes, phi_val.y)};
    sm_gy[tid] = my_gyro;
    const ds kx_val = ds_ldg(&a.kx_vals[kx]);
    const ds ky_val = ds_ldg(&a.ky_vals[ky]);
    const ds hyper_val = ds_ldg(&a.hyper[kx * nky + ky]);
    const ds* w = a.wvp + ((sp * nmu + mu) * ns + s) * 5;
    const double c_acc = a.acc_out ? __ldg(&a.coef[0]) : 0.0;
    const double c_stage = a.acc_out ? __ldg(&a.coef[1]) : 0.0;
    const cds z = {{0.0f, 0.0f}, {0.0f, 0.0f}};

    #pragma unroll
    for (int d = -2; d < 2; ++d) {
        const int vv = v0 + d;
        ring[((vv + RING) % RING) * nt + tid] = (vv >= 0 && vv < nv) ? cds_load(a.df, base0 + vv * plane) : z;
    }

    #pragma unroll 1
    for (int v = v0; v < v0 + VCH; ++v) {
        const int vn = v + 2;
        ring[(vn % RING) * nt + tid] = vn < nv ? cds_load(a.df, base0 + vn * plane) : z;
        __syncthreads();
        const cds* cur = ring + (v % RING) * nt;
        const cds my_df = cur[tid];
        const int row_vms = ((sp * nv + v) * nmu + mu) * ns + s;
        const int field = base0 + v * plane;
        const ds* c_upar_row = a.s_upar_tab + (((sp * nv + v) * ns + s) * a.n_class + cls) * 9;
        const ds* c_t7_row = a.s_t7_tab + (row_vms * a.n_class + cls) * 9;

        DsAcc rx, ry;
        rx.init();
        ry.init();
        #pragma unroll
        for (int i = 0; i < TAP_HI - TAP_LO; ++i) {
            const int src = tap[i];
            if (src >= 0) {
                const ds c_upar = ds_ldg(&c_upar_row[TAP_LO + i]);
                const ds c_t7 = ds_ldg(&c_t7_row[TAP_LO + i]);
                cds v_df, v_gy;
                if (!(src & TAP_OFFKX)) {
                    v_df = cur[src];
                    v_gy = sm_gy[src];
                } else {
                    const int sidx = src & (TAP_OFFKX - 1);
                    v_df = cds_load(a.df, field - spatial + sidx);
                    const ds bes_src = ds_ldg(&bessel_mu[sidx]);
                    const cds phi_src = cds_load(a.phi, sidx);
                    v_gy = {ds_mul(bes_src, phi_src.x), ds_mul(bes_src, phi_src.y)};
                }
                rx.fma(v_df.x, c_upar);
                ry.fma(v_df.y, c_upar);
                rx.fma(v_gy.x, c_t7);
                ry.fma(v_gy.y, c_t7);
                rx.renorm();
                ry.renorm();
            }
        }

        // vpar stencil on this thread's own ring entries
        {
            const cds q0 = ring[((v + RING - 2) % RING) * nt + tid];
            const cds q1 = ring[((v + RING - 1) % RING) * nt + tid];
            const cds q3 = ring[((v + 1) % RING) * nt + tid];
            const cds q4 = ring[((v + 2) % RING) * nt + tid];
            ds c;
            c = ds_ldg(&w[0]); rx.fma(q0.x, c); ry.fma(q0.y, c);
            c = ds_ldg(&w[1]); rx.fma(q1.x, c); ry.fma(q1.y, c);
            c = ds_ldg(&w[2]); rx.fma(my_df.x, c); ry.fma(my_df.y, c);
            c = ds_ldg(&w[3]); rx.fma(q3.x, c); ry.fma(q3.y, c);
            c = ds_ldg(&w[4]); rx.fma(q4.x, c); ry.fma(q4.y, c);
            if constexpr (COLL) {
                // shifts (dv, dmu) in collisions._SHIFTS order; mu neighbours come from global memory
                const int cstride = nv * nmu * ns;
                const ds* cs = a.coll + ((sp * 9 * nv + v) * nmu + mu) * ns + s;
                c = ds_ldg(&cs[0]); rx.fma(my_df.x, c); ry.fma(my_df.y, c);
                c = ds_ldg(&cs[cstride]); rx.fma(q3.x, c); ry.fma(q3.y, c);
                c = ds_ldg(&cs[2 * cstride]); rx.fma(q1.x, c); ry.fma(q1.y, c);
                #pragma unroll
                for (int k = 3; k < 9; ++k) {
                    const int dm = k < 5 ? (k == 3 ? 1 : -1) : (k < 7 ? 1 : -1);
                    const int dv = k < 5 ? 0 : ((k & 1) ? 1 : -1);
                    const int vv = v + dv, mm = mu + dm;
                    if (vv >= 0 && vv < nv && mm >= 0 && mm < nmu) {
                        const cds f = cds_load(a.df, ((sp * nv + vv) * nmu + mm) * nspat + spatial);
                        c = ds_ldg(&cs[k * cstride]);
                        rx.fma(f.x, c);
                        ry.fma(f.y, c);
                    }
                }
            }
        }
        rx.renorm();
        ry.renorm();

        const ds kdotvd = ds_fma(ds_ldg(&a.drift_y[row_vms]), ky_val, ds_mul(ds_ldg(&a.drift_x[row_vms]), kx_val));
        const ds drive_c = ds_fma(ds_neg(ds_ldg(&a.drive_b[row_vms])), kdotvd, ds_ldg(&a.drive_a[row_vms * nky + ky]));
        rx.fma(kdotvd, my_df.y);
        ry.fma(ds_neg(kdotvd), my_df.x);
        rx.fma(hyper_val, my_df.x);
        ry.fma(hyper_val, my_df.y);
        rx.fma(ds_neg(drive_c), my_gyro.y);
        ry.fma(drive_c, my_gyro.x);
        double2 kk = make_double2(ds_to_d(rx.fin()), ds_to_d(ry.fin()));
        if (a.acc_out == nullptr) {
            a.rhs_out[field] = kk;
            continue;
        }
        if (a.extra) {
            const double2 e = __ldg(&a.extra[field]);
            kk.x += e.x;
            kk.y += e.y;
        }
        const double2 q = __ldg(&a.acc_in[field]);
        a.acc_out[field] = make_double2(q.x + c_acc * kk.x, q.y + c_acc * kk.y);
        if (a.stage_out) {
            const double2 p = __ldg(&a.prev[field]);
            a.stage_out[field] = make_double2(p.x + c_stage * kk.x, p.y + c_stage * kk.y);
        }
    }
}

template <int VCH, bool COLL>
static cudaError_t launch_ds(const LinearRhsDsArgs& a, cudaStream_t stream) {
    const int nt = a.ns * a.nky;
    const int blocks = a.nsp * a.nmu * a.nkx * (a.nv / VCH);
    const size_t smem = (RING + 1) * (size_t)nt * sizeof(cds);
    cudaError_t e = cudaFuncSetAttribute(linear_rhs_ds_kernel<VCH, COLL>,
                                         cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
    if (e != cudaSuccess) return e;
    linear_rhs_ds_kernel<VCH, COLL><<<blocks, nt, smem, stream>>>(a);
    return cudaGetLastError();
}

template <class T>
static const T* as(const void* p) { return reinterpret_cast<const T*>(p); }

#define DS_TABLE_ARGS                                       \
    xla_ffi::Buffer<xla_ffi::DataType::C128> df,            \
    xla_ffi::Buffer<xla_ffi::DataType::C128> phi,           \
    xla_ffi::Buffer<xla_ffi::DataType::F32>  bessel,        \
    xla_ffi::Buffer<xla_ffi::DataType::F32>  s_upar_tab,    \
    xla_ffi::Buffer<xla_ffi::DataType::F32>  s_t7_tab,      \
    xla_ffi::Buffer<xla_ffi::DataType::S32>  par_class,     \
    xla_ffi::Buffer<xla_ffi::DataType::S32>  packed_maps,   \
    xla_ffi::Buffer<xla_ffi::DataType::F32>  wvp,           \
    xla_ffi::Buffer<xla_ffi::DataType::F32>  drift_x,       \
    xla_ffi::Buffer<xla_ffi::DataType::F32>  drift_y,       \
    xla_ffi::Buffer<xla_ffi::DataType::F32>  drive_a,       \
    xla_ffi::Buffer<xla_ffi::DataType::F32>  drive_b,       \
    xla_ffi::Buffer<xla_ffi::DataType::F32>  hyper,         \
    xla_ffi::Buffer<xla_ffi::DataType::F32>  kx_vals,       \
    xla_ffi::Buffer<xla_ffi::DataType::F32>  ky_vals,       \
    xla_ffi::Buffer<xla_ffi::DataType::F32>  coll

#define DS_TABLE_FILL(a)                                       \
    a.df = as<double2>(df.untyped_data());                     \
    a.phi = as<double2>(phi.untyped_data());                   \
    a.bessel = as<ds>(bessel.untyped_data());                  \
    a.s_upar_tab = as<ds>(s_upar_tab.untyped_data());          \
    a.s_t7_tab = as<ds>(s_t7_tab.untyped_data());              \
    a.par_class = as<int>(par_class.untyped_data());           \
    a.packed_maps = as<int2>(packed_maps.untyped_data());      \
    a.wvp = as<ds>(wvp.untyped_data());                        \
    a.drift_x = as<ds>(drift_x.untyped_data());                \
    a.drift_y = as<ds>(drift_y.untyped_data());                \
    a.drive_a = as<ds>(drive_a.untyped_data());                \
    a.drive_b = as<ds>(drive_b.untyped_data());                \
    a.hyper = as<ds>(hyper.untyped_data());                    \
    a.kx_vals = as<ds>(kx_vals.untyped_data());                \
    a.ky_vals = as<ds>(ky_vals.untyped_data());                \
    a.coll = as<ds>(coll.untyped_data())

static xla_ffi::Error run_ds(LinearRhsDsArgs& a, int nsp, int nv, int nmu, int ns, int nkx, int nky,
                             int n_class, int has_coll, cudaStream_t stream) {
    if (ns * nky > MAX_THREADS)
        return xla_ffi::Error(XLA_FFI_Error_Code_INVALID_ARGUMENT,
                              "linear_rhs_ds: ns * nky must be <= 576");
    if (nv % 8 != 0)
        return xla_ffi::Error(XLA_FFI_Error_Code_INVALID_ARGUMENT,
                              "linear_rhs_ds: nv must be a multiple of 8");
    if ((long long)nsp * nv * nmu * ns * nkx * nky >= (1LL << 31))
        return xla_ffi::Error(XLA_FFI_Error_Code_INVALID_ARGUMENT,
                              "linear_rhs_ds: grid too large for 32-bit indices");
    a.nsp = nsp; a.nv = nv; a.nmu = nmu; a.ns = ns; a.nkx = nkx; a.nky = nky; a.n_class = n_class;
    const cudaError_t err = has_coll ? (nv % 32 == 0 ? launch_ds<32, true>(a, stream)
                                      : nv % 16 == 0 ? launch_ds<16, true>(a, stream)
                                      :                launch_ds<8, true>(a, stream))
                                     : (nv % 32 == 0 ? launch_ds<32, false>(a, stream)
                                      : nv % 16 == 0 ? launch_ds<16, false>(a, stream)
                                      :                launch_ds<8, false>(a, stream));
    if (err != cudaSuccess)
        return xla_ffi::Error(XLA_FFI_Error_Code_INTERNAL, cudaGetErrorString(err));
    return xla_ffi::Error::Success();
}

xla_ffi::Error LinearRhsDsImpl(
    cudaStream_t stream,
    DS_TABLE_ARGS,
    xla_ffi::Result<xla_ffi::Buffer<xla_ffi::DataType::C128>> rhs_out,
    int32_t nsp, int32_t nv, int32_t nmu, int32_t ns, int32_t nkx, int32_t nky, int32_t n_class,
    int32_t has_coll)
{
    LinearRhsDsArgs a{};
    DS_TABLE_FILL(a);
    a.rhs_out = reinterpret_cast<double2*>(rhs_out->untyped_data());
    return run_ds(a, nsp, nv, nmu, ns, nkx, nky, n_class, has_coll, stream);
}

xla_ffi::Error LinearRhsDsRkImpl(
    cudaStream_t stream,
    DS_TABLE_ARGS,
    xla_ffi::Buffer<xla_ffi::DataType::C128> prev,
    xla_ffi::Buffer<xla_ffi::DataType::C128> acc_in,
    xla_ffi::Buffer<xla_ffi::DataType::C128> extra,
    xla_ffi::Buffer<xla_ffi::DataType::F64>  coef,
    xla_ffi::Result<xla_ffi::Buffer<xla_ffi::DataType::C128>> acc_out,
    xla_ffi::Result<xla_ffi::Buffer<xla_ffi::DataType::C128>> stage_out,
    int32_t nsp, int32_t nv, int32_t nmu, int32_t ns, int32_t nkx, int32_t nky, int32_t n_class,
    int32_t has_coll, int32_t has_extra, int32_t has_stage)
{
    LinearRhsDsArgs a{};
    DS_TABLE_FILL(a);
    a.prev = as<double2>(prev.untyped_data());
    a.acc_in = as<double2>(acc_in.untyped_data());
    a.extra = has_extra ? as<double2>(extra.untyped_data()) : nullptr;
    a.coef = coef.typed_data();
    a.acc_out = reinterpret_cast<double2*>(acc_out->untyped_data());
    a.stage_out = has_stage ? reinterpret_cast<double2*>(stage_out->untyped_data()) : nullptr;
    return run_ds(a, nsp, nv, nmu, ns, nkx, nky, n_class, has_coll, stream);
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    linear_rhs_ds_ffi, LinearRhsDsImpl,
    xla_ffi::Ffi::Bind()
        .Ctx<xla_ffi::PlatformStream<cudaStream_t>>()
        .Arg<xla_ffi::Buffer<xla_ffi::DataType::C128>>() // df
        .Arg<xla_ffi::Buffer<xla_ffi::DataType::C128>>() // phi
        .Arg<xla_ffi::Buffer<xla_ffi::DataType::F32>>()  // bessel
        .Arg<xla_ffi::Buffer<xla_ffi::DataType::F32>>()  // s_upar_tab
        .Arg<xla_ffi::Buffer<xla_ffi::DataType::F32>>()  // s_t7_tab
        .Arg<xla_ffi::Buffer<xla_ffi::DataType::S32>>()  // par_class
        .Arg<xla_ffi::Buffer<xla_ffi::DataType::S32>>()  // packed_maps
        .Arg<xla_ffi::Buffer<xla_ffi::DataType::F32>>()  // wvp
        .Arg<xla_ffi::Buffer<xla_ffi::DataType::F32>>()  // drift_x
        .Arg<xla_ffi::Buffer<xla_ffi::DataType::F32>>()  // drift_y
        .Arg<xla_ffi::Buffer<xla_ffi::DataType::F32>>()  // drive_a
        .Arg<xla_ffi::Buffer<xla_ffi::DataType::F32>>()  // drive_b
        .Arg<xla_ffi::Buffer<xla_ffi::DataType::F32>>()  // hyper
        .Arg<xla_ffi::Buffer<xla_ffi::DataType::F32>>()  // kx_vals
        .Arg<xla_ffi::Buffer<xla_ffi::DataType::F32>>()  // ky_vals
        .Arg<xla_ffi::Buffer<xla_ffi::DataType::F32>>()  // coll
        .Ret<xla_ffi::Buffer<xla_ffi::DataType::C128>>() // rhs_out
        .Attr<int32_t>("nsp")
        .Attr<int32_t>("nv")
        .Attr<int32_t>("nmu")
        .Attr<int32_t>("ns")
        .Attr<int32_t>("nkx")
        .Attr<int32_t>("nky")
        .Attr<int32_t>("n_class")
        .Attr<int32_t>("has_coll")
);

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    linear_rhs_ds_rk_ffi, LinearRhsDsRkImpl,
    xla_ffi::Ffi::Bind()
        .Ctx<xla_ffi::PlatformStream<cudaStream_t>>()
        .Arg<xla_ffi::Buffer<xla_ffi::DataType::C128>>() // df
        .Arg<xla_ffi::Buffer<xla_ffi::DataType::C128>>() // phi
        .Arg<xla_ffi::Buffer<xla_ffi::DataType::F32>>()  // bessel
        .Arg<xla_ffi::Buffer<xla_ffi::DataType::F32>>()  // s_upar_tab
        .Arg<xla_ffi::Buffer<xla_ffi::DataType::F32>>()  // s_t7_tab
        .Arg<xla_ffi::Buffer<xla_ffi::DataType::S32>>()  // par_class
        .Arg<xla_ffi::Buffer<xla_ffi::DataType::S32>>()  // packed_maps
        .Arg<xla_ffi::Buffer<xla_ffi::DataType::F32>>()  // wvp
        .Arg<xla_ffi::Buffer<xla_ffi::DataType::F32>>()  // drift_x
        .Arg<xla_ffi::Buffer<xla_ffi::DataType::F32>>()  // drift_y
        .Arg<xla_ffi::Buffer<xla_ffi::DataType::F32>>()  // drive_a
        .Arg<xla_ffi::Buffer<xla_ffi::DataType::F32>>()  // drive_b
        .Arg<xla_ffi::Buffer<xla_ffi::DataType::F32>>()  // hyper
        .Arg<xla_ffi::Buffer<xla_ffi::DataType::F32>>()  // kx_vals
        .Arg<xla_ffi::Buffer<xla_ffi::DataType::F32>>()  // ky_vals
        .Arg<xla_ffi::Buffer<xla_ffi::DataType::F32>>()  // coll
        .Arg<xla_ffi::Buffer<xla_ffi::DataType::C128>>() // prev
        .Arg<xla_ffi::Buffer<xla_ffi::DataType::C128>>() // acc_in
        .Arg<xla_ffi::Buffer<xla_ffi::DataType::C128>>() // extra
        .Arg<xla_ffi::Buffer<xla_ffi::DataType::F64>>()  // coef
        .Ret<xla_ffi::Buffer<xla_ffi::DataType::C128>>() // acc_out
        .Ret<xla_ffi::Buffer<xla_ffi::DataType::C128>>() // stage_out
        .Attr<int32_t>("nsp")
        .Attr<int32_t>("nv")
        .Attr<int32_t>("nmu")
        .Attr<int32_t>("ns")
        .Attr<int32_t>("nkx")
        .Attr<int32_t>("nky")
        .Attr<int32_t>("n_class")
        .Attr<int32_t>("has_coll")
        .Attr<int32_t>("has_extra")
        .Attr<int32_t>("has_stage")
);
