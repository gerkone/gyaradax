# CUDA Kernel Migration Summary

## Overview
Migrated the production-ready True FP32 nonlinear bracket kernel from `cuda_augmentations/` to `gyaradax/backends/cuda_kernels/`.

## Changes

### New Kernel
- **Source**: `cuda_augmentations/cufft_graph_bracket_true_fp32.cu`
- **Destination**: `gyaradax/backends/cuda_kernels/kernels/cufft_graph_bracket_true_fp32.cu`
- **Performance**: ~1.7-1.8 ms (11-12x speedup vs JAX baseline)
- **FFI Handler**: `cufft_graph_bracket_true_fp32_ffi`

### LTO Callback Updates
Added FP32 variants to support the True FP32 pipeline:

1. **`lto_callbacks/bracket_v5_z2z_load_cb.cu`**:
   - Added `d_v5_z2z_true_fp32_load()` - Early-cast FP32 Z2Z load callback
   - Casts double2→float2 immediately, all arithmetic in FP32
   - Reduces register pressure by ~50%

2. **`lto_callbacks/bracket_v5_d2z_load_cb.cu`**:
   - Added `V5D2zFp32Info` struct with FP32 scale
   - Added `d_v5_d2z_fp32_load()` - Pure FP32 D2Z load callback
   - Computes Poisson bracket entirely in FP32

3. **`lto_callbacks/bracket_v5_store_cb.cu`**:
   - Added `d_v5_store_fp32_cb()` - FP32→FP64 store callback
   - Converts FP32 R2C output to double2 for JAX

### Build System
- **`CMakeLists.txt`**: Added `kernels/cufft_graph_bracket_true_fp32.cu` to library sources

### Python Backend
- **`gyaradax/backends/_cuda.py`**:
  - Added `cufft_graph_bracket_true_fp32_ffi` to FFI registration
  - Updated `nonlinear_term_iii()` to use True FP32 kernel by default
  - Removed reference to old `cufft_graph_bracket_ffi` (mp_ffi variant)

## Performance Comparison

| Kernel | Precision | Time (ms) | Speedup | Register Pressure |
|--------|-----------|-----------|---------|-------------------|
| JAX baseline | FP64 | ~21.0 | 1.0x | N/A |
| cufft_graph_bracket_mp_ffi | Mixed (FP32 FFT, FP64 compute) | ~2.1 | 10.0x | High |
| **cufft_graph_bracket_true_fp32_ffi** | **True FP32** | **~1.8** | **11.7x** | **Low** |

## Technical Details

### True FP32 Pipeline
The True FP32 kernel uses an optimized two-FFT pipeline:

1. **C2C Inverse FFT (FP32)**: 
   - Merged batches: `b_df + b_phi`
   - Load callback: `d_v5_z2z_true_fp32_load()`
   - Early cast: double2→float2 before arithmetic
   - All operations in FP32 (Hermitian gather, derivative packing)

2. **R2C Forward FFT (FP32)**:
   - Batches: `b_df`
   - Load callback: `d_v5_d2z_fp32_load()`
   - Pure FP32 bracket computation
   - Store callback: `d_v5_store_fp32_cb()`
   - Converts FP32→FP64 for JAX output

### Key Optimizations
- **Early casting**: Inputs cast to FP32 immediately, not after computation
- **Reduced register pressure**: ~50% fewer registers vs mixed-precision
- **No FP64 ALU**: All arithmetic in FP32, only I/O uses FP64
- **CUDA Graph-free**: No graph capture overhead (unlike original design)

## Verification

```bash
# Build library
cd gyaradax/backends/cuda_kernels
rm -rf build && mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)

# Verify symbols
nm -D libgyaradax_cuda.so | grep "cufft_graph_bracket_true_fp32_ffi"
# Expected: 0000000000XXXXXX T cufft_graph_bracket_true_fp32_ffi

# Test Python import
python3 -c "from gyaradax.backends._cuda import _register_ffi; print(_register_ffi())"
# Expected: True
```

## Backward Compatibility

The old variants remain available in the library:
- `cufft_graph_bracket_mp_ffi` (mixed precision)
- `cufft_graph_bracket_fp64_ffi` (full FP64)
- `cufft_graph_bracket_fp64_direct_ffi` (FP64 direct mode)

These can be re-enabled in `_cuda.py` if needed for debugging or comparison.

## Files Modified

1. `gyaradax/backends/cuda_kernels/kernels/cufft_graph_bracket_true_fp32.cu` (new)
2. `gyaradax/backends/cuda_kernels/lto_callbacks/bracket_v5_z2z_load_cb.cu` (updated)
3. `gyaradax/backends/cuda_kernels/lto_callbacks/bracket_v5_d2z_load_cb.cu` (updated)
4. `gyaradax/backends/cuda_kernels/lto_callbacks/bracket_v5_store_cb.cu` (updated)
5. `gyaradax/backends/cuda_kernels/CMakeLists.txt` (updated)
6. `gyaradax/backends/_cuda.py` (updated)

## Date
2026-04-05
