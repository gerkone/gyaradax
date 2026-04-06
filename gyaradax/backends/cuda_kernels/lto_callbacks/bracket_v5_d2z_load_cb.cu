// D2Z load callback for v5 layout.
// Fuses the assembly step (bracket computation) into the D2Z FFT:
//   - Reads real-space df and phi from Z2Z/C2C output workspace
//   - Computes Poisson bracket: phi_fy * df_fx - phi_fx * df_fy
//   - Applies dum_s[b%nspec] * scale normalization
//   - Phi broadcast: ws[(b_df + b%b_phi)*plane + loc]
//
// Two variants compiled into one fatbin:
//   d_v5_d2z_mp_load   -- reads float2 from C2C output (mixed precision)
//   d_v5_d2z_fp64_load -- reads double2 from Z2Z output (full FP64)
// Both return double (cufftJITCallbackLoadD).

#include <cufft.h>
#include <cufftXt.h>
#include <cuda_runtime.h>

struct V5D2zMpInfo {
    const float2* ws;     // [(b_df+b_phi), mrad, mphi] -- C2C output workspace
    const double* dum_s;  // [nspec]
    int nspec, mrad, mphi, b_df, b_phi;
    double scale;         // 1/N^2 where N = mrad*mphi
};

struct V5D2zFp32Info {
    const float2* ws;     // [(b_df+b_phi), mrad, mphi] -- C2C output workspace
    const double* dum_s;  // [nspec] - kept as FP64 to match JAX input
    int nspec, mrad, mphi, b_df, b_phi;
    float scale;          // 1/N^2 where N = mrad*mphi
};

struct V5D2zFp64Info {
    const double2* ws;    // [(b_df+b_phi), mrad, mphi] -- Z2Z output workspace
    const double*  dum_s; // [nspec]
    int nspec, mrad, mphi, b_df, b_phi;
    double scale;
};

// Pure FP32: reads float2, computes bracket in FP32, returns float
__device__ float d_v5_d2z_fp32_load(
    void *dataIn, unsigned long long offset,
    void *callerInfo, void *sharedPointer)
{
    const V5D2zFp32Info* ci = (const V5D2zFp32Info*)callerInfo;
    int plane = ci->mrad * ci->mphi;
    int b   = (int)(offset / (unsigned long long)plane);
    int loc = (int)(offset % (unsigned long long)plane);

    float2 d = ci->ws[(size_t)b * plane + loc];
    float2 p = ci->ws[(size_t)(ci->b_df + b % ci->b_phi) * plane + loc];

    // bracket = phi_fy * df_fx - phi_fx * df_fy (all in FP32)
    // dum_s is FP64, cast to FP32 for multiplication
    float dum = (float)ci->dum_s[b % ci->nspec];
    float bracket = p.x * d.y - p.y * d.x;
    return ci->scale * dum * bracket;
}

// Mixed precision: reads float2, promotes to double for bracket
__device__ double d_v5_d2z_mp_load(
    void *dataIn, unsigned long long offset,
    void *callerInfo, void *sharedPointer)
{
    const V5D2zMpInfo* ci = (const V5D2zMpInfo*)callerInfo;
    int plane = ci->mrad * ci->mphi;
    int b   = (int)(offset / (unsigned long long)plane);
    int loc = (int)(offset % (unsigned long long)plane);

    float2 d = ci->ws[(size_t)b * plane + loc];
    float2 p = ci->ws[(size_t)(ci->b_df + b % ci->b_phi) * plane + loc];

    // bracket = phi_fy * df_fx - phi_fx * df_fy
    double bracket = (double)p.x * (double)d.y - (double)p.y * (double)d.x;
    return ci->scale * ci->dum_s[b % ci->nspec] * bracket;
}

// Full FP64: reads double2 directly
__device__ double d_v5_d2z_fp64_load(
    void *dataIn, unsigned long long offset,
    void *callerInfo, void *sharedPointer)
{
    const V5D2zFp64Info* ci = (const V5D2zFp64Info*)callerInfo;
    int plane = ci->mrad * ci->mphi;
    int b   = (int)(offset / (unsigned long long)plane);
    int loc = (int)(offset % (unsigned long long)plane);

    double2 d = ci->ws[(size_t)b * plane + loc];
    double2 p = ci->ws[(size_t)(ci->b_df + b % ci->b_phi) * plane + loc];

    double bracket = p.x * d.y - p.y * d.x;
    return ci->scale * ci->dum_s[b % ci->nspec] * bracket;
}

__device__ cufftJITCallbackLoadR d_v5_d2z_fp32_load_addr = d_v5_d2z_fp32_load;
__device__ cufftJITCallbackLoadD d_v5_d2z_mp_load_addr   = d_v5_d2z_mp_load;
__device__ cufftJITCallbackLoadD d_v5_d2z_fp64_load_addr = d_v5_d2z_fp64_load;
