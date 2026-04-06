#include <cufft.h>
#include <cufftXt.h>

__device__ cufftDoubleComplex d_test_load_cb(
    void*              dataIn,
    unsigned long long offset,
    void*              callerInfo,
    void*              sharedMem)
{
    cufftDoubleComplex res;
    // Return a distinct pattern based on offset
    res.x = (double)offset + 1.0;
    res.y = -((double)offset + 1.0);
    return res;
}

__device__ cufftJITCallbackLoadZ d_test_load_cb_ptr = d_test_load_cb;
