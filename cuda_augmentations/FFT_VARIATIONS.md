# Gyaradax FFT Kernel Variations

This document describes the various FFT-based implementations of the nonlinear Poisson bracket (Term III) in `cuda_augmentations/`.

---

## 1. cuFFT Standard (Non-LTO)
**Handler:** `cufft_bracket_ffi`
**File:** `cufft_bracket.cu`

*   **Inputs:**
    *   `phi_y_k`, `f_x_k`, `phi_x_k`, `f_y_k`: Four complex spectral derivative arrays (pre-computed in JAX). Shape: `[batch, mrad, mphi_half]`.
    *   `dum_s_eff`: Per-species scaling factors.
*   **Outputs:**
    *   `out`: Resulting spectral bracket. Shape: `[batch, mrad, mphi_half]`.
*   **Operations:**
    1.  **4x `cufftExecZ2D`**: Inverse FFTs to real space.
    2.  **`fused_bracket_scale` (Kernel)**: Computes $\{ \phi, f \} = \partial_y \phi \partial_x f - \partial_x \phi \partial_y f$ in real space.
    3.  **1x `cufftExecD2Z`**: Forward FFT back to spectral space.

---

## 2. LTO Baseline (v1 / v2)
**Handlers:** `lto_fft_bracket_v1_ffi`, `lto_fft_bracket_v2_ffi`
**File:** `cufft_lto_bracket.cu`

*   **Inputs:**
    *   `df`, `phi`: Packed spectral fields. Shape: `[batch, nkx, nky]`.
    *   `kx`, `ky`, `jind`, `dum_s`: Geometry and species parameters.
*   **Outputs:**
    *   `out`: Spectral bracket. Shape: `[batch, mrad, mphi_half]`.
*   **Operations (v1):**
    1.  **4x `cufftExecZ2D` (LTO Load CB)**: Gathers packed spectral data and applies $ik$ derivatives during the Load pass of the inverse FFT.
    2.  **`lto_bracket_explicit_kernel`**: Real-space bracket calculation.
    3.  **1x `cufftExecD2Z`**: Forward FFT to spectral space.
*   **Operations (v2 - Production):**
    1.  **4x `cufftExecZ2D` (LTO Load CB)**: Same as v1.
    2.  **1x `cufftExecD2Z` (LTO Load CB)**: Fuses the bracket calculation into the Load pass of the forward FFT, eliminating the explicit real-space kernel.

---

## 3. LTO Sparse Store (v3)
**Handler:** `lto_fft_bracket_v3_ffi`
**File:** `cufft_lto_bracket.cu`

*   **Inputs/Outputs:** Same as v2, but output is packed `[batch, nkx, nky]`.
*   **Operations:**
    1.  **4x `cufftExecZ2D` (LTO Load CB)**: Same as v2.
    2.  **1x `cufftExecD2Z` (LTO Load & Store CB)**:
        *   **Load CB**: Transposes/fuses bracket calculation.
        *   **Store CB**: Scatters the result directly into the packed JAX output format, bypassing the dealiased regions.

---

## 4. LTO Z2Z (2-for-1 Packing)
**Handlers:** `lto_fft_bracket_vz2z_ffi`, `lto_fft_bracket_vz2z_merged_ffi`
**File:** `cufft_lto_bracket.cu`

*   **Technique:** Uses complex-to-complex (Z2Z) transforms to process two real fields in one pass (e.g., $\phi_y + i f_y$).
*   **Operations (vz2z_merged - Peak Performance):**
    1.  **1x `cufftExecZ2Z` (LTO Load CB)**: Batched inverse FFT. Load callback packs 4 spectral derivatives into 2 complex workspaces.
    2.  **1x `cufftExecD2Z` (LTO Load & Store CB)**: Fuses bracket subtraction and sparse scatter.

---

## 5. CUDA Graph LTO (v4)
**Handler:** `lto_fft_bracket_v4_ffi`
**File:** `cufft_lto_bracket.cu`

*   **Technique:** Captures the `vz2z_merged` sequence into a CUDA Graph.
*   **Operations:**
    *   **First Call**: Captures `launch_compute_scale_factors`, `cufftExecZ2Z`, and `cufftExecD2Z`.
    *   **Subsequent Calls**: `cudaGraphLaunch`.
    *   **Pointers**: Monitors input/output pointers and re-captures if they change (JAX cache invalidation).

---

## 6. Unfused CUDA Graph
**Handler:** `cufft_graph_bracket_ffi`
**File:** `cufft_graph_bracket.cu`

*   **Technique:** Uses primitive unfused kernels but eliminates launch overhead via CUDA Graphs. Designed to reduce L1TEX pressure compared to complex LTO callbacks.
*   **Operations:**
    1.  **`v5_pack_kernel`**: Explicitly prepares 4 spectral derivative workspaces from packed inputs.
    2.  **4x `cufftExecZ2D`**: Batch inverse FFTs.
    3.  **`v5_assembly_kernel`**: Real-space bracket.
    4.  **1x `cufftExecD2Z`**: Forward FFT.
    5.  **`v5_unpack_kernel`**: Scatters result to packed output.
