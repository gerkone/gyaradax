#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <cufft.h>
#include <cufftXt.h>

#define CHECK_CUDA(call) { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        printf("CUDA Error at %s:%d - %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        return 1; \
    } \
}

#define CHECK_CUFFT(call) { \
    cufftResult res = (call); \
    if (res != CUFFT_SUCCESS) { \
        printf("cuFFT Error at %s:%d - code %d\n", __FILE__, __LINE__, res); \
        return 1; \
    } \
}

#include "callback_fatbin.h"

int main() {
    // 1. Mandatory 16-byte alignment for the fatbin buffer (nvJitLink requirement)
    size_t fatbin_size = sizeof(callback_fatbin);
    void *aligned_fatbin = nullptr;
    if (posix_memalign(&aligned_fatbin, 16, fatbin_size) != 0) {
        printf("Failed to allocate aligned memory\n");
        return 1;
    }
    memcpy(aligned_fatbin, callback_fatbin, fatbin_size);
    void *fatbin_data = aligned_fatbin;

    int n = 1024;
    cufftDoubleComplex *host_in = (cufftDoubleComplex*)malloc(n * sizeof(cufftDoubleComplex));
    for (int i = 0; i < n; i++) {
        host_in[i].x = 7.0;
        host_in[i].y = 10.0;
    }
    
    cufftDoubleComplex *d_in, *d_out;
    CHECK_CUDA(cudaMalloc(&d_in, n * sizeof(cufftDoubleComplex)));
    CHECK_CUDA(cudaMalloc(&d_out, n * sizeof(cufftDoubleComplex)));
    CHECK_CUDA(cudaMemcpy(d_in, host_in, n * sizeof(cufftDoubleComplex), cudaMemcpyHostToDevice));

    cufftHandle plan;
    CHECK_CUFFT(cufftCreate(&plan));
    
    void* d_caller_info = nullptr; 
    CHECK_CUDA(cudaMalloc(&d_caller_info, 8)); // Dummy addr
    
    printf("Registering LTO Callback (name: 'd_load_cb_ptr')...\n");
    CHECK_CUFFT(cufftXtSetJITCallback(
        plan, "d_load_cb_ptr", fatbin_data, fatbin_size, 
        CUFFT_CB_LD_COMPLEX_DOUBLE, &d_caller_info
    ));

    printf("Making plan (JIT Linking size %d)... \n", n);
    long long n_ll = n;
    size_t ws = 0;
    CHECK_CUFFT(cufftXtMakePlanMany(
        plan, 1, &n_ll, NULL, 1, n_ll, CUDA_C_64F, 
        NULL, 1, n_ll, CUDA_C_64F, 1, &ws, CUDA_C_64F
    ));

    printf("Executing FFT...\n");
    CHECK_CUFFT(cufftExecZ2Z(plan, d_in, d_out, CUFFT_FORWARD));

    cufftDoubleComplex host_out_0;
    CHECK_CUDA(cudaMemcpy(&host_out_0, d_out, sizeof(cufftDoubleComplex), cudaMemcpyDeviceToHost));

    printf("========================================\n");
    printf("Expected Result (Callback Re & Im) : {%.2f, %.2f}\n", (double)n * 1.0, (double)n * 2.0);
    printf("Actual Output   (Mode 0)           : {%.2f, %.2f}\n", host_out_0.x, host_out_0.y);
    printf("========================================\n");
    
    if (host_out_0.y == (double)n * 10.0) {
        printf("BUG CONFIRMED: Imaginary part was read from global memory, not the callback!\n");
    } else if (host_out_0.y == (double)n * 2.0) {
        printf("BUG FIXED: The callback returned the imaginary part correctly.\n");
    }

    CHECK_CUFFT(cufftDestroy(plan));
    CHECK_CUDA(cudaFree(d_in)); CHECK_CUDA(cudaFree(d_out)); CHECK_CUDA(cudaFree(d_caller_info));
    free(aligned_fatbin);
    free(host_in);
    return 0;
}