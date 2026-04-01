#include <cufftXt.h>

__device__ cufftDoubleComplex d_load_cb_ptr(
    void *dataIn,
    unsigned long long offset,
    void *callerInfo,
    void *sharedPointer)
{
    cufftDoubleComplex val;
    val.x = 1.0;  // Real part
    val.y = 2.0;  // Imag part
    return val;
}

__device__ cufftJITCallbackLoadZ d_load_cb_ptr_addr = d_load_cb_ptr;