#include <cuda_runtime.h>

namespace {

__global__ void scale_kernel(const double* __restrict__ x,
                             double* __restrict__ y,
                             double alpha,
                             int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        y[idx] = alpha * x[idx];
    }
}

}  // namespace

// Tiny C ABI example for build smoke tests and future ctypes/CFFI experiments.
// This is intentionally not wired into JAX FFI; add a concrete FFI contract only
// when an experiment needs to compare a real custom call against JAX.
extern "C" void gyaradax_experiment_scale(cudaStream_t stream,
                                           const double* x,
                                           double* y,
                                           double alpha,
                                           int n) {
    if (n <= 0) {
        return;
    }
    constexpr int threads = 256;
    int blocks = (n + threads - 1) / threads;
    scale_kernel<<<blocks, threads, 0, stream>>>(x, y, alpha, n);
}
