// GEMMul8 Ozaki-II on the gyaradax stencil shape vs native DGEMM, on this GPU.
// Stencil: A (32x32, ns=18 real) @ F (32 x N); N = 64 batches * 73344 columns,
// the same FLOPs and bytes as the 64 per-(sp,vpar) GEMMs. Plus a square control.
#include "../include/gemmul8.hpp"
#include "../test/common/self_hipify.hpp"
#include <cstdio>
#include <cmath>
#include <random>
#include <vector>

static float ms_of(cudaEvent_t a, cudaEvent_t b) { float t; cudaEventElapsedTime(&t, a, b); return t; }

static void run(const char* tag, int m, int n, int k, int real_mk, int reps) {
    cublasHandle_t h; cublasCreate(&h);
    cublasLtHandle_t lt; cublasLtCreate(&lt);
    std::mt19937_64 g(0); std::normal_distribution<double> d;
    std::vector<double> hA((size_t)m*k, 0.0), hB((size_t)k*n, 0.0);
    for (int j = 0; j < real_mk; ++j) for (int i = 0; i < real_mk; ++i) hA[(size_t)j*m+i] = d(g);
    for (size_t j = 0; j < (size_t)n; ++j) for (int i = 0; i < real_mk; ++i) hB[j*k+i] = d(g);
    double *A, *B, *C, *R;
    cudaMalloc(&A, hA.size()*8); cudaMalloc(&B, hB.size()*8);
    cudaMalloc(&C, (size_t)m*n*8); cudaMalloc(&R, (size_t)m*n*8);
    cudaMemcpy(A, hA.data(), hA.size()*8, cudaMemcpyHostToDevice);
    cudaMemcpy(B, hB.data(), hB.size()*8, cudaMemcpyHostToDevice);
    const double one = 1.0, zero = 0.0;
    cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);

    cublasDgemm(h, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &one, A, m, B, k, &zero, R, m);
    cudaEventRecord(e0);
    for (int r = 0; r < reps; ++r)
        cublasDgemm(h, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &one, A, m, B, k, &zero, R, m);
    cudaEventRecord(e1); cudaEventSynchronize(e1);
    const double t64 = ms_of(e0, e1) / reps;
    std::vector<double> hR((size_t)m*n), hC((size_t)m*n);
    cudaMemcpy(hR.data(), R, hR.size()*8, cudaMemcpyDeviceToHost);
    double rmax = 0; for (double v : hR) rmax = std::fmax(rmax, std::fabs(v));
    const double flops = 2.0*m*n*k;
    printf("\n[%s] m=%d n=%d k=%d\n  native DGEMM      %9.3f ms  %8.1f TFLOP/s\n",
           tag, m, n, k, t64, flops/t64/1e9);

    auto ozaki = [&](auto backend_tag, const char* name, int nm) {
        constexpr gemmul8::Backend BK = decltype(backend_tag)::value;
        void* work = nullptr;
        auto res = gemmul8::gemmLt<BK>(lt, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &one, A, m,
                                       B, k, &zero, C, m, nm, false, work);
        cudaMalloc(&work, size_t(res[0]));
        gemmul8::gemmLt<BK>(lt, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &one, A, m, B, k, &zero, C, m, nm, false, work);
        cudaEventRecord(e0);
        for (int r = 0; r < reps; ++r)
            gemmul8::gemmLt<BK>(lt, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &one, A, m, B, k, &zero, C, m, nm, false, work);
        cudaEventRecord(e1); cudaEventSynchronize(e1);
        const double t = ms_of(e0, e1) / reps;
        cudaMemcpy(hC.data(), C, hC.size()*8, cudaMemcpyDeviceToHost);
        double emax = 0; for (size_t i = 0; i < hC.size(); ++i) emax = std::fmax(emax, std::fabs(hC[i]-hR[i]));
        printf("  %-5s moduli=%2d  %9.3f ms  %6.2fx vs DGEMM   max rel err %.1e\n",
               name, nm, t, t64/t, emax/rmax);
        auto ph = gemmul8::gemmLt<BK>(lt, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &one, A, m,
                                      B, k, &zero, C, m, nm, false, work);
        cudaDeviceSynchronize();
        if (ph.size() >= 4) {
            const double tot = (ph[0] + ph[1] + ph[2] + ph[3]) * 1e3;
            printf("        phases: scaling %8.3f ms (%4.1f%%)  mm %8.3f ms (%4.1f%%)  "
                   "mod %8.3f ms (%4.1f%%)  undo/CRT %8.3f ms (%4.1f%%)\n",
                   ph[0]*1e3, 100*ph[0]*1e3/tot, ph[1]*1e3, 100*ph[1]*1e3/tot,
                   ph[2]*1e3, 100*ph[2]*1e3/tot, ph[3]*1e3, 100*ph[3]*1e3/tot);
        }
        cudaFree(work);
    };
    using FP8T  = std::integral_constant<gemmul8::Backend, gemmul8::Backend::FP8>;
    using INT8T = std::integral_constant<gemmul8::Backend, gemmul8::Backend::INT8>;
    for (int nm : {12}) ozaki(FP8T{}, "FP8", nm);
    
    cudaFree(A); cudaFree(B); cudaFree(C); cudaFree(R);
    cublasDestroy(h); cublasLtDestroy(lt);
}

int main() {
    cudaSetDevice(0);
    run("gyaradax stencil", 32, 64 * 73344, 32, 18, 10);
    run("square control",   8192, 8192, 8192, 8192, 3);
    return 0;
}
