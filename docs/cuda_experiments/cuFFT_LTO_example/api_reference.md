API reference¶

This section describes the extension to the cuFFT API included in the cuFFT LTO EA.

Note

The non-callback functionalities of cuFFT 11.0.8.X and cuFFT LTO EA 11.1.0.X are unchanged.

cuFFT 11.0.8.X and cuFFT LTO EA 11.1.0.X should have the same functionality and performance for non-callback plans.
Associating LTO callbacks with cuFFT Plan¶
cufftXtSetJITCallback¶

cufftResult cufftXtSetJITCallback(cufftHandle plan, const void *lto_callback_fatbin, size_t lto_callback_fatbin_size, cufftXtCallbackType type, void **caller_info)¶

    cufftXtSetJITCallback associates the specified callback with the plan represented by the handle plan.

    The callback should be compiled into LTO-IR (for example, using the flag -dlto with nvcc or NVRTC) and passed to the function as a pointer to the data containing the compiled device function. The data could be an array containing the fatbin compiled using nvcc, or the result of nvrtcGetLTOIR(…).

    The size of the pointer in bytes should be specified in lto_callback_fatbin_size.

    Note that this function must be called after plan creation (after using cufftCreate to initialize the handle), but before using the planning function (such as cufftMakePlan1D).

    Once associated with a plan, LTO callbacks cannot be unset using cufftXtClearCallback or any other methods. This is a limitation of the cuFFT LTO EA preview, and we are working to lift this restriction.

    Setting the maximum shared memory size for the callbacks with cufftXtSetCallbackSharedSize can be done after using the planning function. The same restriction of non-LTO callbacks of 16 kB applies to LTO callbacks.

    Parameters:

            plan[In] – cufftHandle returned by cufftCreate.

            lto_callback_fatbin[In] – Pointer to the location in host memory where the callback device function is located, after being compiled into LTO-IR with nvcc or NVRTC.

            lto_callback_fatbin_size[In] – Size in bytes of the data pointed at by lto_callback_fatbin.

            type[In] – Type of the callback function, such as CUFFT_CB_LD_COMPLEX, or CUFFT_CB_ST_REAL. See Type definitions for callbacks

            caller_info[In] – Optional array of device pointers to caller specific information, one per GPU. Please note that multi-gpu LTO callbacks are not supported yet.

    Return values:

            CUFFT_SUCCESS – cuFFT successfully associated the plan with the callback device function.

            CUFFT_INVALID_PLAN – The plan is not valid (e.g. the handle was already used to make a plan).

            CUFFT_INVALID_TYPE – The callback type is not valid.

            CUFFT_INVALID_VALUE – The pointer to the callback device function is invalid or the size is 0.

            CUFFT_NOT_SUPPORTED – The functionality is not supported yet (e.g. multi-GPU with LTO callbacks).

            CUFFT_INTERNAL_ERROR – cuFFT encountered an unexpected error. Please contact us with your use case and feedback.

LTO callback signatures¶

Unlike non-LTO callbacks, which are treated as pointers to user functions, LTO callbacks are linked against the cuFFT kernels at runtime. In order to do the linking, the callback signature, including the function name, must match the exactly one signature as listed below. This also means that the cuFFT LTO EA preview is restricted to one callback function of each type per source file / compilation unit.

Note

We are currently working to allow flexible function signatures for LTO callbacks.

Other than the specific function name, the signature of the LTO callbacks matches that of the non-LTO callbacks.

These are the LTO callback kernel signatures currently supported:
Load Single-Precision Complex¶

__device__ cufftComplex cufftJITCallbackLoadComplex(void *dataIn, size_t offset, void *callerInfo, void *sharedPointer)

Load Double-Precision Complex¶

__device__ cufftDoubleComplex cufftJITCallbackLoadDoubleComplex(void *dataIn, size_t offset, void *callerInfo, void *sharedPointer)

Load Single-Precision Real¶

__device__ cufftReal cufftJITCallbackLoadReal(void *dataIn, size_t offset, void *callerInfo, void *sharedPointer)

Load Double-Precision Real¶

__device__ cufftDoubleReal cufftJITCallbackLoadDoubleReal(void *dataIn, size_t offset, void *callerInfo, void *sharedPointer)

Store Single-Precision Complex¶

__device__ void cufftJITCallbackStoreComplex(void *dataOut, size_t offset, cufftComplex element, void *callerInfo, void *sharedPointer)

Store Double-Precision Complex¶

__device__ void cufftJITCallbackStoreDoubleComplex(void *dataOut, size_t offset, cufftDoubleComplex element, void *callerInfo, void *sharedPointer)

Store Single-Precision Real¶

__device__ void cufftJITCallbackStoreReal(void *dataOut, size_t offset, cufftReal element, void *callerInfo, void *sharedPointer)

Store Double-Precision Real¶

__device__ void cufftJITCallbackStoreDoubleReal(void *dataOut, size_t offset, cufftDoubleReal element, void *callerInfo, void *sharedPointer)