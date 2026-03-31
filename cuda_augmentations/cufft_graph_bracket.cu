#include <cstdio>
#include <string>
#include <vector>
#include <mutex>
#include <map>
#include "xla/ffi/api/ffi.h"
#include <cuda_runtime.h>
#include <cufft.h>

// v5 Graph approach: Unfused kernels + CUDA Graphs
// Goal: Eliminate L1TEX bottlenecks by separating uncoalesced memory mapping from FFT.

namespace {

struct GraphKey {
    int device, batch, mrad, mphi, nkx, nky, nspec;
    bool operator<(const GraphKey& o) const {
        if (device != o.device) return device < o.device;
        if (batch  != o.batch)  return batch  < o.batch;
        if (mrad   != o.mrad)   return mrad   < o.mrad;
        if (mphi   != o.mphi)   return mphi   < o.mphi;
        if (nkx    != o.nkx)    return nkx    < o.nkx;
        if (nky    != o.nky)    return nky    < o.nky;
        return nspec < o.nspec;
    }
};

struct GraphState {
    cudaGraphExec_t instance = nullptr;
    cufftHandle plan_z2d = 0;
    cufftHandle plan_d2z = 0;
    
    // Scratch workspaces
    double2 *ws_phi_y_k = nullptr; // [B*S, mrad, mphi_half]
    double2 *ws_f_x_k   = nullptr;
    double2 *ws_phi_x_k = nullptr;
    double2 *ws_f_y_k   = nullptr;
    
    double  *ws_phi_y_r = nullptr; // [B*S, mrad, mphi]
    double  *ws_f_x_r   = nullptr;
    double  *ws_phi_x_r = nullptr;
    double  *ws_f_y_r   = nullptr;
    double  *ws_nl_r    = nullptr;
    
    double2 *ws_nl_k    = nullptr; // [B*S, mrad, mphi_half]

    ~GraphState() {
        if (instance) cudaGraphExecDestroy(instance);
        if (plan_z2d) cufftDestroy(plan_z2d);
        if (plan_d2z) cufftDestroy(plan_d2z);
        
        if (ws_phi_y_k) cudaFree(ws_phi_y_k);
        if (ws_f_x_k)   cudaFree(ws_f_x_k);
        if (ws_phi_x_k) cudaFree(ws_phi_x_k);
        if (ws_f_y_k)   cudaFree(ws_f_y_k);
        
        if (ws_phi_y_r) cudaFree(ws_phi_y_r);
        if (ws_f_x_r)   cudaFree(ws_f_x_r);
        if (ws_phi_x_r) cudaFree(ws_phi_x_r);
        if (ws_f_y_r)   cudaFree(ws_f_y_r);
        if (ws_nl_r)    cudaFree(ws_nl_r);
        
        if (ws_nl_k)    cudaFree(ws_nl_k);
    }
};

static std::map<GraphKey, GraphState*> g_cache;
static std::mutex g_mutex;

// 1. Pack Kernel: (batch_spec * mrad, mphi_half) 
// Optimized for coalesced access by making sure each row starts a new warp.
__global__ void v5_pack_kernel(
    const double2* __restrict__ df_packed,   // [B * S, nkx, nky]
    const double2* __restrict__ phi_packed,  // [B, nkx, nky]
    const double*  __restrict__ kx,
    const double*  __restrict__ ky,
    const int*     __restrict__ inverse_jind, // [mrad] -> kx_idx
    double2* __restrict__ out_phi_y_k,       // [B, S, mrad, mphi_half]
    double2* __restrict__ out_f_x_k,
    double2* __restrict__ out_phi_x_k,
    double2* __restrict__ out_f_y_k,
    int mrad, int mphi_half, int nkx, int nky, int nspec
) {
    int m_idx = blockIdx.x;
    int bs_idx = blockIdx.y;
    int ky_idx = blockIdx.z * blockDim.x + threadIdx.x;
    if (ky_idx >= mphi_half) return;

    int mbs_idx = bs_idx * mrad + m_idx;
    int batch_idx = bs_idx / nspec;

    int kx_idx = inverse_jind[m_idx];
    double2 val_phi = {0.0, 0.0};
    double2 val_df  = {0.0, 0.0};

    if (kx_idx >= 0 && ky_idx < nky) {
        size_t row_stride = (size_t)nkx * nky;
        size_t packed_idx = (size_t)kx_idx * nky + ky_idx;
        val_df  = df_packed[(size_t)bs_idx * row_stride + packed_idx];
        val_phi = phi_packed[(size_t)batch_idx * row_stride + packed_idx];
    }

    double kx_val = (kx_idx >= 0) ? kx[kx_idx] : 0.0;
    double ky_val = (ky_idx < nky) ? ky[ky_idx] : 0.0;

    size_t out_idx = (size_t)mbs_idx * mphi_half + ky_idx;
    out_phi_y_k[out_idx] = {-ky_val * val_phi.y, ky_val * val_phi.x};
    out_phi_x_k[out_idx] = {-kx_val * val_phi.y, kx_val * val_phi.x};
    out_f_y_k[out_idx]   = {-ky_val * val_df.y,  ky_val * val_df.x};
    out_f_x_k[out_idx]   = {-kx_val * val_df.y,  kx_val * val_df.x};
}

// 2. Real Assembly: (batch_spec * mrad, mphi)
__global__ void v5_assembly_kernel(
    const double* py, const double* fx, const double* px, const double* fy,
    double* nl, const double* dum_s, 
    int mrad_mphi, int nspec, double scale, size_t total_r
) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < total_r) {
        int spec_idx = (int)(i / mrad_mphi) % nspec;
        nl[i] = scale * dum_s[spec_idx] * (py[i] * fx[i] - px[i] * fy[i]);
    }
}

// 3. Unpack Kernel: (batch_spec * nkx, nky)
__global__ void v5_unpack_kernel(
    const double2* __restrict__ nl_dense_k, // [B * S, mrad, mphi_half]
    double2* __restrict__ out_packed,       // [B * S, nkx, nky]
    const int* __restrict__ jind,           // [nkx] -> m_idx
    int mrad, int mphi_half, int nkx, int nky, int nspec,
    int ixzero, int iyzero
) {
    int kx_idx = blockIdx.x;
    int bs_idx = blockIdx.y;
    int ky_idx = blockIdx.z * blockDim.x + threadIdx.x;
    if (ky_idx >= nky) return;

    int nkx_bs_idx = bs_idx * nkx + kx_idx;

    int m_idx = jind[kx_idx];
    double2 val = {0.0, 0.0};

    if (m_idx >= 0) {
        size_t dense_idx = (size_t)bs_idx * (mrad * mphi_half) + (size_t)m_idx * mphi_half + ky_idx;
        val = nl_dense_k[dense_idx];
        if (kx_idx == ixzero && ky_idx == iyzero) val = {0.0, 0.0};
    }

    size_t out_idx = (size_t)nkx_bs_idx * nky + ky_idx;
    out_packed[out_idx] = val;
}

} // namespace

namespace xla_ffi = xla::ffi;

#define CHECK_CUDA(call) { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) return xla::ffi::Error::Internal(std::string("CUDA Error: ") + cudaGetErrorString(err)); \
}
#define CHECK_CUFFT(call) { \
    cufftResult res = (call); \
    if (res != CUFFT_SUCCESS) return xla::ffi::Error::Internal("cuFFT Error"); \
}

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
    GraphKey key = {device, batch, mrad, mphi, nkx, nky, nspec};
    
    std::lock_guard<std::mutex> lock(g_mutex);
    GraphState* s = g_cache[key];
    
    int mphi_half = mphi / 2 + 1;
    size_t c_dist = (size_t)mrad * mphi_half;
    size_t r_dist = (size_t)mrad * mphi;
    size_t total_c = (size_t)batch * nspec * c_dist;
    size_t total_r = (size_t)batch * nspec * r_dist;

    if (!s) {
        s = new GraphState();
        g_cache[key] = s;
        
        size_t total_c_4 = total_c * 4;
        size_t total_r_4 = total_r * 4;
        
        CHECK_CUDA(cudaMalloc(&s->ws_phi_y_k, total_c_4 * 16));
        s->ws_f_x_k   = s->ws_phi_y_k + total_c;
        s->ws_phi_x_k = s->ws_f_x_k   + total_c;
        s->ws_f_y_k   = s->ws_phi_x_k + total_c;
        
        CHECK_CUDA(cudaMalloc(&s->ws_phi_y_r, total_r_4 * 8));
        s->ws_f_x_r   = s->ws_phi_y_r + total_r;
        s->ws_phi_x_r = s->ws_f_x_r   + total_r;
        s->ws_f_y_r   = s->ws_phi_x_r + total_r;
        
        CHECK_CUDA(cudaMalloc(&s->ws_nl_r,    total_r * 8));
        CHECK_CUDA(cudaMalloc(&s->ws_nl_k,    total_c * 16));
        
        CHECK_CUFFT(cufftCreate(&s->plan_z2d));
        int n[2] = {mrad, mphi};
        CHECK_CUFFT(cufftPlanMany(&s->plan_z2d, 2, n, NULL, 1, (int)c_dist, NULL, 1, (int)r_dist, CUFFT_Z2D, batch * nspec * 4));
        
        CHECK_CUFFT(cufftCreate(&s->plan_d2z));
        CHECK_CUFFT(cufftPlanMany(&s->plan_d2z, 2, n, NULL, 1, (int)r_dist, NULL, 1, (int)c_dist, CUFFT_D2Z, batch * nspec));
        
        // Warm-up and capture
        CHECK_CUFFT(cufftSetStream(s->plan_z2d, stream));
        CHECK_CUFFT(cufftExecZ2D(s->plan_z2d, s->ws_phi_y_k, s->ws_phi_y_r));
        cudaStreamSynchronize(stream);
        
        cudaGraph_t graph;
        cudaError_t err = cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);
        if (err != cudaSuccess) return xla::ffi::Error::Internal("BeginCapture failed");

        auto capture_body = [&]() -> cudaError_t {
            dim3 block_p(32, 1, 1);
            dim3 grid_p(mrad, batch * nspec, (mphi_half + 31) / 32);
            v5_pack_kernel<<<grid_p, block_p, 0, stream>>>(
                (const double2*)df.typed_data(), (const double2*)phi.typed_data(),
                kx.typed_data(), ky.typed_data(), inverse_jind.typed_data(),
                s->ws_phi_y_k, s->ws_f_x_k, s->ws_phi_x_k, s->ws_f_y_k,
                mrad, mphi_half, nkx, nky, nspec
            );
            if (cudaPeekAtLastError() != cudaSuccess) return cudaErrorLaunchFailure;
            
            cufftResult res = cufftSetStream(s->plan_z2d, stream);
            if (res != CUFFT_SUCCESS) return cudaErrorUnknown;
            res = cufftExecZ2D(s->plan_z2d, s->ws_phi_y_k, s->ws_phi_y_r);
            if (res != CUFFT_SUCCESS) return cudaErrorUnknown;
            
            double inv_n2 = 1.0 / (double)((size_t)mrad * mphi * mrad * mphi);
            int threads_a = 256;
            int blocks_a = (int)((total_r + threads_a - 1) / threads_a);
            v5_assembly_kernel<<<blocks_a, threads_a, 0, stream>>>(
                s->ws_phi_y_r, s->ws_f_x_r, s->ws_phi_x_r, s->ws_f_y_r,
                s->ws_nl_r, dum_s.typed_data(), mrad * mphi, nspec, inv_n2, total_r
            );
            if (cudaPeekAtLastError() != cudaSuccess) return cudaErrorLaunchFailure;
            
            res = cufftSetStream(s->plan_d2z, stream);
            if (res != CUFFT_SUCCESS) return cudaErrorUnknown;
            res = cufftExecD2Z(s->plan_d2z, s->ws_nl_r, s->ws_nl_k);
            if (res != CUFFT_SUCCESS) return cudaErrorUnknown;
            
            dim3 block_u(32, 1, 1);
            dim3 grid_u(nkx, batch * nspec, (nky + 31) / 32);
            v5_unpack_kernel<<<grid_u, block_u, 0, stream>>>(
                s->ws_nl_k, (double2*)out->typed_data(), jind.typed_data(),
                mrad, mphi_half, nkx, nky, nspec,
                ixzero, iyzero
            );
            if (cudaPeekAtLastError() != cudaSuccess) return cudaErrorLaunchFailure;
            return cudaSuccess;
        };

        cudaError_t capture_err = capture_body();
        cudaError_t end_err = cudaStreamEndCapture(stream, &graph);
        
        if (capture_err != cudaSuccess || end_err != cudaSuccess) {
            if (end_err == cudaSuccess) cudaGraphDestroy(graph);
            return xla::ffi::Error::Internal("Graph capture failed");
        }
        
        CHECK_CUDA(cudaGraphInstantiate(&s->instance, graph, NULL, NULL, 0));
        cudaGraphDestroy(graph);
    }
    
    CHECK_CUDA(cudaGraphLaunch(s->instance, stream));
    return xla_ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    cufft_graph_bracket_ffi, CufftGraphBracketImpl,
    xla_ffi::Ffi::Bind()
        .Ctx<xla_ffi::PlatformStream<cudaStream_t>>()
        .Arg<xla_ffi::Buffer<xla_ffi::DataType::C128>>() // df
        .Arg<xla_ffi::Buffer<xla_ffi::DataType::C128>>() // phi
        .Arg<xla_ffi::Buffer<xla_ffi::DataType::F64>>()  // kx
        .Arg<xla_ffi::Buffer<xla_ffi::DataType::F64>>()  // ky
        .Arg<xla_ffi::Buffer<xla_ffi::DataType::S32>>()  // jind
        .Arg<xla_ffi::Buffer<xla_ffi::DataType::S32>>()  // inverse_jind
        .Arg<xla_ffi::Buffer<xla_ffi::DataType::F64>>()  // dum_s
        .Ret<xla_ffi::Buffer<xla_ffi::DataType::C128>>() // out
        .Attr<int32_t>("batch")
        .Attr<int32_t>("mrad")
        .Attr<int32_t>("mphi")
        .Attr<int32_t>("nkx")
        .Attr<int32_t>("nky")
        .Attr<int32_t>("nspec")
        .Attr<int32_t>("ixzero")
        .Attr<int32_t>("iyzero")
);

