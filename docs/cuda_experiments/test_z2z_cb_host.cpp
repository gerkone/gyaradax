#include <cuda_runtime.h>
#include <cufft.h>
#include <cufftXt.h>
#include <stdio.h>
#include <vector>
#include <iostream>

#include "test_z2z_cb_fatbin.h"

#define CHECK_CUDA(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error in %s at line %d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
}

#define CHECK_CUFFT(call) { \
    cufftResult res = call; \
    if (res != CUFFT_SUCCESS) { \
        fprintf(stderr, "cuFFT error in %s at line %d: %d\n", __FILE__, __LINE__, (int)res); \
        exit(EXIT_FAILURE); \
    } \
}

int main() {
    const int mrad = 4;
    const int mphi = 4;
    const int batch = 1;
    const size_t n_elem = (size_t)batch * mrad * mphi;
    
    cufftHandle plan;
    CHECK_CUFFT(cufftCreate(&plan));
    
    // Set callback
    void* d_cb_ptr = nullptr; // Callback doesn't need callerInfo for this test
    CHECK_CUFFT(cufftXtSetJITCallback(
        plan, 
        "d_test_load_cb", 
        (void*)test_z2z_cb_fatbin, 
        sizeof(test_z2z_cb_fatbin), 
        CUFFT_CB_LD_COMPLEX_DOUBLE, 
        &d_cb_ptr
    ));
    
    long long n_ll[2] = {mrad, mphi};
    size_t workSize = 0;
    CHECK_CUFFT(cufftXtMakePlanMany(
        plan, 2, n_ll, 
        NULL, 1, (long long)mrad*mphi, CUDA_C_64F, 
        NULL, 1, (long long)mrad*mphi, CUDA_C_64F, 
        batch, &workSize, CUDA_C_64F
    ));
    
    // Allocate device memory
    cufftDoubleComplex *d_in, *d_out;
    CHECK_CUDA(cudaMalloc(&d_in, n_elem * sizeof(cufftDoubleComplex)));
    CHECK_CUDA(cudaMalloc(&d_out, n_elem * sizeof(cufftDoubleComplex)));
    CHECK_CUDA(cudaMemset(d_in, 0, n_elem * sizeof(cufftDoubleComplex)));
    
    // Execute inverse Z2Z
    // The load callback should ignore d_in and return (offset, -offset)
    CHECK_CUFFT(cufftExecZ2Z(plan, d_in, d_out, CUFFT_INVERSE));
    
    // Copy back and check
    std::vector<cufftDoubleComplex> h_out(n_elem);
    CHECK_CUDA(cudaMemcpy(h_out.data(), d_out, n_elem * sizeof(cufftDoubleComplex), cudaMemcpyDeviceToHost));
    
    printf("Standalone Z2Z LTO Callback Test Results:\n");
    bool success = true;
    for (int i = 0; i < n_elem; ++i) {
        // If the callback worked, the output should reflect the inverse FFT of the pattern (i, -i)
        // For simplicity, let's just print the values.
        printf("  [%d] re: %10.2f, im: %10.2f\n", i, h_out[i].x, h_out[i].y);
        if (h_out[i].x == 0 && h_out[i].y == 0) {
            success = false;
        }
    }
    
    if (success) {
        printf("\nSUCCESS: Callback appears to have been invoked (non-zero output).\n");
    } else {
        printf("\nFAILURE: Callback may not have been invoked (all-zero output).\n");
    }
    
    cudaFree(d_in);
    cudaFree(d_out);
    cufftDestroy(plan);
    
    return success ? 0 : 1;
}
