// SIMT throughput on this GPU for the ops a modular (Ozaki-II) stencil needs:
// int32 multiply-add, int64 multiply-add, fp32 fma, fp64 fma. Dependent chains are
// interleaved 8-wide per thread so issue rate, not latency, is measured.
#include <cstdio>
#include <cstdint>
#define CH 8
#define IT 4096
template <typename T> __global__ void k(T* out, T a, T b) {
    T x[CH];
    #pragma unroll
    for (int c = 0; c < CH; ++c) x[c] = (T)(threadIdx.x + c);
    for (int i = 0; i < IT; ++i) {
        #pragma unroll
        for (int c = 0; c < CH; ++c) x[c] = x[c] * a + b;
    }
    T s = 0;
    #pragma unroll
    for (int c = 0; c < CH; ++c) s += x[c];
    if (s == (T)12345) out[0] = s;          // keep the work live
}
template <typename T> double rate(const char* name) {
    T* d; cudaMalloc(&d, sizeof(T));
    int dev; cudaGetDevice(&dev); cudaDeviceProp p; cudaGetDeviceProperties(&p, dev);
    const int blocks = p.multiProcessorCount * 32, threads = 256;
    k<T><<<blocks, threads>>>(d, (T)3, (T)1); cudaDeviceSynchronize();
    cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
    cudaEventRecord(e0);
    for (int r = 0; r < 5; ++r) k<T><<<blocks, threads>>>(d, (T)3, (T)1);
    cudaEventRecord(e1); cudaEventSynchronize(e1);
    float ms; cudaEventElapsedTime(&ms, e0, e1);
    const double ops = 5.0 * blocks * threads * (double)IT * CH;   // one multiply-add each
    const double r = ops / (ms * 1e-3);
    printf("  %-10s %9.1f T mul-add/s\n", name, r / 1e12);
    cudaFree(d); return r;
}
int main() {
    cudaDeviceProp p; cudaGetDeviceProperties(&p, 0);
    printf("  device %s, %d SMs\n", p.name, p.multiProcessorCount);
    double f64 = rate<double>("fp64 fma");
    double f32 = rate<float>("fp32 fma");
    double i32 = rate<int32_t>("int32 mad");
    double i64 = rate<int64_t>("int64 mad");
    printf("\n  int32 / fp64 = %.0fx    int64 / fp64 = %.1fx    fp32 / fp64 = %.0fx\n",
           i32 / f64, i64 / f64, f32 / f64);
}
