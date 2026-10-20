#include "xla/ffi/api/ffi.h"
#include <cuda_runtime.h>

namespace xla_ffi = xla::ffi;

// electrostatic linear rhs, one block per (sp, mu, kx, vpar chunk) marching along vpar
struct LinearRhsEsArgs {
    const double2* df;        // (nsp, nv, nmu, ns, nkx, nky)
    const double2* phi;       // (ns, nkx, nky)
    const double* bessel;     // (nsp, nmu, ns, nkx, nky)
    const double* s_upar_tab; // (nsp, nv, ns, n_class, 9)
    const double* s_t7_tab;   // (nsp, nv, nmu, ns, n_class, 9)
    const int* par_class;     // (ns, nkx, nky)
    const int2* packed_maps;  // (9, ns, nkx, nky)
    const double* wvp;        // (nsp, nmu, ns, 5)
    const double* drift_x;    // (nsp, nv, nmu, ns)
    const double* drift_y;    // (nsp, nv, nmu, ns)
    const double* drive_a;    // (nsp, nv, nmu, ns, nky)
    const double* drive_b;    // (nsp, nv, nmu, ns)
    const double* hyper;      // (nkx, nky)
    const double* kx_vals;    // (nkx,)
    const double* ky_vals;    // (nky,)
    const double* coll;       // (nsp, 9, nv, nmu, ns) collision stencil, when COLL
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

// (dv, dmu) of collision shift k in collisions._SHIFTS order
__host__ __device__ constexpr int coll_dv(int k) { return k == 0 ? 0 : k < 3 ? (k == 1 ? 1 : -1) : k < 5 ? 0 : ((k & 1) ? 1 : -1); }
__host__ __device__ constexpr int coll_dmu(int k) { return k < 3 ? 0 : k < 5 ? (k == 3 ? 1 : -1) : (k < 7 ? 1 : -1); }

constexpr int ES_TAP_LO = 2;
constexpr int ES_TAP_HI = 7;
constexpr int ES_TAP_OFFKX = 1 << 30;
constexpr int ES_RING = 5;
constexpr int ES_MAX_THREADS = 576;

template <int VCH, bool COLL>
__global__ __launch_bounds__(ES_MAX_THREADS, 2) void linear_rhs_es_kernel(const LinearRhsEsArgs a) {
    const int ns = a.ns, nky = a.nky, nv = a.nv, nmu = a.nmu, nkx = a.nkx;
    const int nt = ns * nky;

    extern __shared__ double2 es_smem[];
    double2* ring = es_smem;
    double2* sm_gy = ring + ES_RING * nt;

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
    const double* bessel_mu = a.bessel + (sp * nmu + mu) * nspat;

    // same-kx taps index the shared planes, other-kx taps are flagged global indices
    int tap[ES_TAP_HI - ES_TAP_LO];
    #pragma unroll
    for (int i = 0; i < ES_TAP_HI - ES_TAP_LO; ++i) {
        const int2 m = __ldg(&a.packed_maps[(ES_TAP_LO + i) * nspat + spatial]);
        tap[i] = m.x < 0 ? -1
               : (m.y == kx ? m.x * nky + ky : (ES_TAP_OFFKX | (m.x * (nkx * nky) + m.y * nky + ky)));
    }
    const double bes = __ldg(&bessel_mu[spatial]);
    const double2 phi_val = __ldg(&a.phi[spatial]);
    const double2 my_gyro = make_double2(bes * phi_val.x, bes * phi_val.y);
    sm_gy[tid] = my_gyro;
    const double kx_val = __ldg(&a.kx_vals[kx]);
    const double ky_val = __ldg(&a.ky_vals[ky]);
    const double hyper_val = __ldg(&a.hyper[kx * nky + ky]);
    const double* w = a.wvp + ((sp * nmu + mu) * ns + s) * 5;
    const double c_acc = a.acc_out ? __ldg(&a.coef[0]) : 0.0;
    const double c_stage = a.acc_out ? __ldg(&a.coef[1]) : 0.0;
    const double2 z = make_double2(0.0, 0.0);

    #pragma unroll
    for (int d = -2; d < 2; ++d) {
        const int vv = v0 + d;
        ring[((vv + ES_RING) % ES_RING) * nt + tid] = (vv >= 0 && vv < nv) ? __ldg(&a.df[base0 + vv * plane]) : z;
    }

    #pragma unroll 1
    for (int v = v0; v < v0 + VCH; ++v) {
        const int vn = v + 2;
        ring[(vn % ES_RING) * nt + tid] = vn < nv ? __ldg(&a.df[base0 + vn * plane]) : z;
        __syncthreads();
        const double2* cur = ring + (v % ES_RING) * nt;
        const double2 my_df = cur[tid];
        const int row_vms = ((sp * nv + v) * nmu + mu) * ns + s;
        const int field = base0 + v * plane;
        const double* c_upar_row = a.s_upar_tab + (((sp * nv + v) * ns + s) * a.n_class + cls) * 9;
        const double* c_t7_row = a.s_t7_tab + (row_vms * a.n_class + cls) * 9;

        double rx = 0.0, ry = 0.0;
        #pragma unroll
        for (int i = 0; i < ES_TAP_HI - ES_TAP_LO; ++i) {
            const int src = tap[i];
            if (src >= 0) {
                const double c_upar = __ldg(&c_upar_row[ES_TAP_LO + i]);
                const double c_t7 = __ldg(&c_t7_row[ES_TAP_LO + i]);
                double2 v_df, v_gy;
                if (!(src & ES_TAP_OFFKX)) {
                    v_df = cur[src];
                    v_gy = sm_gy[src];
                } else {
                    const int sidx = src & (ES_TAP_OFFKX - 1);
                    v_df = __ldg(&a.df[field - spatial + sidx]);
                    const double bes_src = __ldg(&bessel_mu[sidx]);
                    const double2 phi_src = __ldg(&a.phi[sidx]);
                    v_gy = make_double2(bes_src * phi_src.x, bes_src * phi_src.y);
                }
                rx += v_df.x * c_upar;
                ry += v_df.y * c_upar;
                rx += v_gy.x * c_t7;
                ry += v_gy.y * c_t7;
            }
        }

        // vpar stencil on this thread's own ring entries
        {
            const double2 q0 = ring[((v + ES_RING - 2) % ES_RING) * nt + tid];
            const double2 q1 = ring[((v + ES_RING - 1) % ES_RING) * nt + tid];
            const double2 q3 = ring[((v + 1) % ES_RING) * nt + tid];
            const double2 q4 = ring[((v + 2) % ES_RING) * nt + tid];
            double c;
            c = __ldg(&w[0]); rx += q0.x * c; ry += q0.y * c;
            c = __ldg(&w[1]); rx += q1.x * c; ry += q1.y * c;
            c = __ldg(&w[2]); rx += my_df.x * c; ry += my_df.y * c;
            c = __ldg(&w[3]); rx += q3.x * c; ry += q3.y * c;
            c = __ldg(&w[4]); rx += q4.x * c; ry += q4.y * c;
            if constexpr (COLL) {
                // dmu = 0 shifts come from the ring, mu neighbours from global memory
                const int cstride = nv * nmu * ns;
                const double* cs = a.coll + ((sp * 9 * nv + v) * nmu + mu) * ns + s;
                c = __ldg(&cs[0]); rx += my_df.x * c; ry += my_df.y * c;
                c = __ldg(&cs[cstride]); rx += q3.x * c; ry += q3.y * c;
                c = __ldg(&cs[2 * cstride]); rx += q1.x * c; ry += q1.y * c;
                #pragma unroll
                for (int k = 3; k < 9; ++k) {
                    const int vv = v + coll_dv(k), mm = mu + coll_dmu(k);
                    if (vv >= 0 && vv < nv && mm >= 0 && mm < nmu) {
                        const double2 f = __ldg(&a.df[((sp * nv + vv) * nmu + mm) * nspat + spatial]);
                        c = __ldg(&cs[k * cstride]);
                        rx += f.x * c;
                        ry += f.y * c;
                    }
                }
            }
        }

        const double kdotvd = __ldg(&a.drift_x[row_vms]) * kx_val + __ldg(&a.drift_y[row_vms]) * ky_val;
        const double drive_c = __ldg(&a.drive_a[row_vms * nky + ky]) - __ldg(&a.drive_b[row_vms]) * kdotvd;
        rx += kdotvd * my_df.y;
        ry -= kdotvd * my_df.x;
        rx += hyper_val * my_df.x;
        ry += hyper_val * my_df.y;
        rx -= drive_c * my_gyro.y;
        ry += drive_c * my_gyro.x;
        if (a.acc_out == nullptr) {
            a.rhs_out[field] = make_double2(rx, ry);
            continue;
        }
        if (a.extra) {
            const double2 e = __ldg(&a.extra[field]);
            rx += e.x;
            ry += e.y;
        }
        const double2 q = __ldg(&a.acc_in[field]);
        a.acc_out[field] = make_double2(q.x + c_acc * rx, q.y + c_acc * ry);
        if (a.stage_out) {
            const double2 pv = __ldg(&a.prev[field]);
            a.stage_out[field] = make_double2(pv.x + c_stage * rx, pv.y + c_stage * ry);
        }
    }
}

template <int VCH, bool COLL>
static cudaError_t launch_es(const LinearRhsEsArgs& a, cudaStream_t stream) {
    const int nt = a.ns * a.nky;
    const int blocks = a.nsp * a.nmu * a.nkx * (a.nv / VCH);
    const size_t smem = (ES_RING + 1) * (size_t)nt * sizeof(double2);
    cudaError_t e = cudaFuncSetAttribute(linear_rhs_es_kernel<VCH, COLL>,
                                         cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
    if (e != cudaSuccess) return e;
    linear_rhs_es_kernel<VCH, COLL><<<blocks, nt, smem, stream>>>(a);
    return cudaGetLastError();
}

template <class T>
static const T* as(const void* p) { return reinterpret_cast<const T*>(p); }

#define ES_TABLE_ARGS                                       \
    xla_ffi::Buffer<xla_ffi::DataType::C128> df,            \
    xla_ffi::Buffer<xla_ffi::DataType::C128> phi,           \
    xla_ffi::Buffer<xla_ffi::DataType::F64>  bessel,        \
    xla_ffi::Buffer<xla_ffi::DataType::F64>  s_upar_tab,    \
    xla_ffi::Buffer<xla_ffi::DataType::F64>  s_t7_tab,      \
    xla_ffi::Buffer<xla_ffi::DataType::S32>  par_class,     \
    xla_ffi::Buffer<xla_ffi::DataType::S32>  packed_maps,   \
    xla_ffi::Buffer<xla_ffi::DataType::F64>  wvp,           \
    xla_ffi::Buffer<xla_ffi::DataType::F64>  drift_x,       \
    xla_ffi::Buffer<xla_ffi::DataType::F64>  drift_y,       \
    xla_ffi::Buffer<xla_ffi::DataType::F64>  drive_a,       \
    xla_ffi::Buffer<xla_ffi::DataType::F64>  drive_b,       \
    xla_ffi::Buffer<xla_ffi::DataType::F64>  hyper,         \
    xla_ffi::Buffer<xla_ffi::DataType::F64>  kx_vals,       \
    xla_ffi::Buffer<xla_ffi::DataType::F64>  ky_vals,       \
    xla_ffi::Buffer<xla_ffi::DataType::F64>  coll

#define ES_TABLE_FILL(a)                                       \
    a.df = as<double2>(df.untyped_data());                     \
    a.phi = as<double2>(phi.untyped_data());                   \
    a.bessel = as<double>(bessel.untyped_data());              \
    a.s_upar_tab = as<double>(s_upar_tab.untyped_data());      \
    a.s_t7_tab = as<double>(s_t7_tab.untyped_data());          \
    a.par_class = as<int>(par_class.untyped_data());           \
    a.packed_maps = as<int2>(packed_maps.untyped_data());      \
    a.wvp = as<double>(wvp.untyped_data());                    \
    a.drift_x = as<double>(drift_x.untyped_data());            \
    a.drift_y = as<double>(drift_y.untyped_data());            \
    a.drive_a = as<double>(drive_a.untyped_data());            \
    a.drive_b = as<double>(drive_b.untyped_data());            \
    a.hyper = as<double>(hyper.untyped_data());                \
    a.kx_vals = as<double>(kx_vals.untyped_data());            \
    a.ky_vals = as<double>(ky_vals.untyped_data());            \
    a.coll = as<double>(coll.untyped_data())

static xla_ffi::Error run_es(LinearRhsEsArgs& a, int nsp, int nv, int nmu, int ns, int nkx, int nky,
                             int n_class, int has_coll, cudaStream_t stream) {
    if (ns * nky > ES_MAX_THREADS || nv % 8 != 0)
        return xla_ffi::Error(XLA_FFI_Error_Code_INVALID_ARGUMENT,
                              "linear_rhs_es: needs ns * nky <= 576 and nv % 8 == 0");
    if ((long long)nsp * nv * nmu * ns * nkx * nky >= (1LL << 31))
        return xla_ffi::Error(XLA_FFI_Error_Code_INVALID_ARGUMENT,
                              "linear_rhs_es: grid too large for 32-bit indices");
    a.nsp = nsp; a.nv = nv; a.nmu = nmu; a.ns = ns; a.nkx = nkx; a.nky = nky; a.n_class = n_class;
    const cudaError_t err = has_coll ? (nv % 32 == 0 ? launch_es<32, true>(a, stream)
                                      : nv % 16 == 0 ? launch_es<16, true>(a, stream)
                                      :                launch_es<8, true>(a, stream))
                                     : (nv % 32 == 0 ? launch_es<32, false>(a, stream)
                                      : nv % 16 == 0 ? launch_es<16, false>(a, stream)
                                      :                launch_es<8, false>(a, stream));
    if (err != cudaSuccess)
        return xla_ffi::Error(XLA_FFI_Error_Code_INTERNAL, cudaGetErrorString(err));
    return xla_ffi::Error::Success();
}

xla_ffi::Error LinearRhsEsImpl(
    cudaStream_t stream,
    ES_TABLE_ARGS,
    xla_ffi::Result<xla_ffi::Buffer<xla_ffi::DataType::C128>> rhs_out,
    int32_t nsp, int32_t nv, int32_t nmu, int32_t ns, int32_t nkx, int32_t nky, int32_t n_class,
    int32_t has_coll)
{
    LinearRhsEsArgs a{};
    ES_TABLE_FILL(a);
    a.rhs_out = reinterpret_cast<double2*>(rhs_out->untyped_data());
    return run_es(a, nsp, nv, nmu, ns, nkx, nky, n_class, has_coll, stream);
}

xla_ffi::Error LinearRhsEsRkImpl(
    cudaStream_t stream,
    ES_TABLE_ARGS,
    xla_ffi::Buffer<xla_ffi::DataType::C128> prev,
    xla_ffi::Buffer<xla_ffi::DataType::C128> acc_in,
    xla_ffi::Buffer<xla_ffi::DataType::C128> extra,
    xla_ffi::Buffer<xla_ffi::DataType::F64>  coef,
    xla_ffi::Result<xla_ffi::Buffer<xla_ffi::DataType::C128>> acc_out,
    xla_ffi::Result<xla_ffi::Buffer<xla_ffi::DataType::C128>> stage_out,
    int32_t nsp, int32_t nv, int32_t nmu, int32_t ns, int32_t nkx, int32_t nky, int32_t n_class,
    int32_t has_coll, int32_t has_extra, int32_t has_stage)
{
    LinearRhsEsArgs a{};
    ES_TABLE_FILL(a);
    a.prev = as<double2>(prev.untyped_data());
    a.acc_in = as<double2>(acc_in.untyped_data());
    a.extra = has_extra ? as<double2>(extra.untyped_data()) : nullptr;
    a.coef = coef.typed_data();
    a.acc_out = reinterpret_cast<double2*>(acc_out->untyped_data());
    a.stage_out = has_stage ? reinterpret_cast<double2*>(stage_out->untyped_data()) : nullptr;
    return run_es(a, nsp, nv, nmu, ns, nkx, nky, n_class, has_coll, stream);
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    linear_rhs_es_ffi, LinearRhsEsImpl,
    xla_ffi::Ffi::Bind()
        .Ctx<xla_ffi::PlatformStream<cudaStream_t>>()
        .Arg<xla_ffi::Buffer<xla_ffi::DataType::C128>>() // df
        .Arg<xla_ffi::Buffer<xla_ffi::DataType::C128>>() // phi
        .Arg<xla_ffi::Buffer<xla_ffi::DataType::F64>>()  // bessel
        .Arg<xla_ffi::Buffer<xla_ffi::DataType::F64>>()  // s_upar_tab
        .Arg<xla_ffi::Buffer<xla_ffi::DataType::F64>>()  // s_t7_tab
        .Arg<xla_ffi::Buffer<xla_ffi::DataType::S32>>()  // par_class
        .Arg<xla_ffi::Buffer<xla_ffi::DataType::S32>>()  // packed_maps
        .Arg<xla_ffi::Buffer<xla_ffi::DataType::F64>>()  // wvp
        .Arg<xla_ffi::Buffer<xla_ffi::DataType::F64>>()  // drift_x
        .Arg<xla_ffi::Buffer<xla_ffi::DataType::F64>>()  // drift_y
        .Arg<xla_ffi::Buffer<xla_ffi::DataType::F64>>()  // drive_a
        .Arg<xla_ffi::Buffer<xla_ffi::DataType::F64>>()  // drive_b
        .Arg<xla_ffi::Buffer<xla_ffi::DataType::F64>>()  // hyper
        .Arg<xla_ffi::Buffer<xla_ffi::DataType::F64>>()  // kx_vals
        .Arg<xla_ffi::Buffer<xla_ffi::DataType::F64>>()  // ky_vals
        .Arg<xla_ffi::Buffer<xla_ffi::DataType::F64>>()  // coll
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
    linear_rhs_es_rk_ffi, LinearRhsEsRkImpl,
    xla_ffi::Ffi::Bind()
        .Ctx<xla_ffi::PlatformStream<cudaStream_t>>()
        .Arg<xla_ffi::Buffer<xla_ffi::DataType::C128>>() // df
        .Arg<xla_ffi::Buffer<xla_ffi::DataType::C128>>() // phi
        .Arg<xla_ffi::Buffer<xla_ffi::DataType::F64>>()  // bessel
        .Arg<xla_ffi::Buffer<xla_ffi::DataType::F64>>()  // s_upar_tab
        .Arg<xla_ffi::Buffer<xla_ffi::DataType::F64>>()  // s_t7_tab
        .Arg<xla_ffi::Buffer<xla_ffi::DataType::S32>>()  // par_class
        .Arg<xla_ffi::Buffer<xla_ffi::DataType::S32>>()  // packed_maps
        .Arg<xla_ffi::Buffer<xla_ffi::DataType::F64>>()  // wvp
        .Arg<xla_ffi::Buffer<xla_ffi::DataType::F64>>()  // drift_x
        .Arg<xla_ffi::Buffer<xla_ffi::DataType::F64>>()  // drift_y
        .Arg<xla_ffi::Buffer<xla_ffi::DataType::F64>>()  // drive_a
        .Arg<xla_ffi::Buffer<xla_ffi::DataType::F64>>()  // drive_b
        .Arg<xla_ffi::Buffer<xla_ffi::DataType::F64>>()  // hyper
        .Arg<xla_ffi::Buffer<xla_ffi::DataType::F64>>()  // kx_vals
        .Arg<xla_ffi::Buffer<xla_ffi::DataType::F64>>()  // ky_vals
        .Arg<xla_ffi::Buffer<xla_ffi::DataType::F64>>()  // coll
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
