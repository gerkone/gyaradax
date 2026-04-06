// D2Z store callback for v5 layout with zero-mode masking.
// Scatters D2Z dense output [batch, mrad, mphi_half] to packed [batch, nkx, nky].
// Uses inverse_jind[dense_m] to map dense kx index -> packed index;
// elements with inverse_jind[i_dense] < 0 or j >= nky are dropped.
// Zero-mode masking: (ixzero, iyzero) forced to zero.
//
// Three variants:
//   d_v5_store_cb            -- reads cufftDoubleComplex (FP64 D2Z output)
//   d_v5_store_fp32_cb       -- reads cufftComplex (FP32 R2C output), converts to double2
//   d_v5_z2z_bracket_store_cb -- Z2Z store with fused bracket computation (single-FFT optimization)

#include <cufft.h>
#include <cufftXt.h>
#include <cuda_runtime.h>

struct V5StoreInfo {
    double2*    out_packed;      // [batch, nkx, nky] -- FFI output buffer
    const int*  inverse_jind;    // [mrad] dense -> packed, -1 if absent
    int mrad, mphiw3, nkx, nky;
    int ixzero, iyzero;
};

__device__ void d_v5_store_cb(
    void *dataOut, unsigned long long offset,
    cufftDoubleComplex element,
    void *callerInfo, void *sharedPointer)
{
    const V5StoreInfo* si = (const V5StoreInfo*)callerInfo;
    int batch_idx = (int)(offset / ((unsigned long long)si->mrad * si->mphiw3));
    int i_dense   = (int)((offset / si->mphiw3) % si->mrad);
    int j         = (int)(offset % si->mphiw3);

    if (j >= si->nky) return;

    int i_pack = si->inverse_jind[i_dense];
    if (i_pack < 0) return;

    // Zero-mode masking
    if (i_pack == si->ixzero && j == si->iyzero)
        element = {0.0, 0.0};

    si->out_packed[((unsigned long long)batch_idx * si->nkx + i_pack) * si->nky + j] = element;
}

__device__ void d_v5_store_fp32_cb(
    void *dataOut, unsigned long long offset,
    cufftComplex element,
    void *callerInfo, void *sharedPointer)
{
    const V5StoreInfo* si = (const V5StoreInfo*)callerInfo;
    int batch_idx = (int)(offset / ((unsigned long long)si->mrad * si->mphiw3));
    int i_dense   = (int)((offset / si->mphiw3) % si->mrad);
    int j         = (int)(offset % si->mphiw3);

    if (j >= si->nky) return;

    int i_pack = si->inverse_jind[i_dense];
    if (i_pack < 0) return;

    // Zero-mode masking
    double2 elem_d = {0.0, 0.0};
    if (i_pack == si->ixzero && j == si->iyzero) {
        elem_d = {0.0, 0.0};
    } else {
        elem_d = {(double)element.x, (double)element.y};
    }

    si->out_packed[((unsigned long long)batch_idx * si->nkx + i_pack) * si->nky + j] = elem_d;
}

__device__ cufftJITCallbackStoreZ d_v5_store_cb_addr       = d_v5_store_cb;
__device__ cufftJITCallbackStoreC d_v5_store_fp32_cb_addr  = d_v5_store_fp32_cb;

// ── Z2Z Store Callback with Fused Bracket (Single-FFT Optimization) ─────────
// This callback is used with a single Z2Z inverse transform that processes
// both df and phi in MERGED/INTERLEAVED batches. It computes the Poisson bracket
// and scatters to packed output in one pass, eliminating the D2Z forward FFT.
//
// Workspace layout: ws[b_df * 2, mrad, mphi]
//   ws[2*b]     = df batch b derivatives (fy + i*fx) after IFFT
//   ws[2*b + 1] = phi batch b derivatives (fy + i*fx) after IFFT
//
// The store callback is invoked for ALL batches (both df and phi).
// For df batches (even), it computes bracket and writes output.
// For phi batches (odd), it just stores the transformed data.
//
// For Hermitian-packed input, IFFT output has:
//   element.x = fy (real-space y-derivative)
//   element.y = fx (real-space x-derivative)

struct V5Z2zBracketStoreInfo {
    double2*    out_packed;      // [b_df, nkx, nky] -- final output
    const float2* ws;            // [b_df * 2, mrad, mphi] -- Z2Z workspace
    const double* dum_s;         // [nspec]
    const int*  inverse_jind;    // [mrad] dense->packed, -1 if absent
    int mrad, mphi, mphiw3, nkx, nky;
    int b_df, b_phi, nspec;
    int ixzero, iyzero;
    float scale;                 // 1/N^2 where N = mrad*mphi
};

__device__ void d_v5_z2z_bracket_store_cb(
    void *dataOut, unsigned long long offset,
    cufftComplex element,  // IFFT output at this position
    void *callerInfo, void *sharedPointer)
{
    const V5Z2zBracketStoreInfo* si = (const V5Z2zBracketStoreInfo*)callerInfo;
    
    int plane = si->mrad * si->mphi;
    int gb    = (int)(offset / (unsigned long long)plane);
    int loc   = (int)(offset % (unsigned long long)plane);
    
    // Determine if this is a df batch (even) or phi batch (odd)
    int pair_idx = gb / 2;
    int is_phi_batch = gb % 2;
    
    // Skip phi batches - they're just stored for df to read
    if (is_phi_batch) {
        // Don't write anything for phi batches
        return;
    }
    
    // Only process df batches that exist
    if (pair_idx >= si->b_df) return;
    
    int b = pair_idx;
    
    // Extract df derivatives from IFFT output (current element)
    // For Hermitian IFFT: .x = fy, .y = fx (real-space derivatives)
    float df_fy = element.x;
    float df_fx = element.y;
    
    // Read phi derivatives from the NEXT batch (phi is at 2*b+1)
    int phi_gb = gb + 1;
    float2 phi_elem = si->ws[(size_t)phi_gb * plane + loc];
    float phi_fy = phi_elem.x;
    float phi_fx = phi_elem.y;
    
    // Compute Poisson bracket: {phi, f} = dphi/dy * df/dx - dphi/dx * df/dy
    float bracket = phi_fy * df_fx - phi_fx * df_fy;
    
    // Apply normalization: dum_s[b%nspec] * (1/N^2)
    float dum = (float)si->dum_s[b % si->nspec];
    float result = si->scale * dum * bracket;
    
    // Map back to packed spectrum
    int i_dense = (int)((offset / si->mphi) % si->mrad);
    int j       = (int)(offset % si->mphi);
    
    if (j >= si->nky) return;
    
    int i_pack = si->inverse_jind[i_dense];
    if (i_pack < 0) return;
    
    // Zero-mode masking
    double2 out_elem = {0.0, 0.0};
    if (!(i_pack == si->ixzero && j == si->iyzero)) {
        out_elem = {(double)result, 0.0};
    }
    
    // Scatter to packed output
    si->out_packed[((unsigned long long)b * si->nkx + i_pack) * si->nky + j] = out_elem;
}

__device__ cufftJITCallbackStoreC d_v5_z2z_bracket_store_cb_addr = d_v5_z2z_bracket_store_cb;
