#pragma once

#include <cuda_runtime.h>
#include <cstddef>

// shard halo slot of point (v, m) outside a local (vpar, mu) block, as a block-uniform shift from its df address; false if none
__device__ __forceinline__ bool halo_shift(
    const double2* df,
    const double2* vp_halo,  // (nsp, 4, nmu, nspat): planes v0 - 2, v0 - 1, v0 + nv, v0 + nv + 1, zero past the grid
    const double2* mu_halo,  // (nsp, nv + 2, 2, nspat): planes mu0 - 1, mu0 + nmu for v0 - 1 .. v0 + nv, zero past the grid
    int sp, int v, int m, int nv, int nmu, int nspat, ptrdiff_t& shift) {
    const ptrdiff_t at = (ptrdiff_t)((sp * nv + v) * nmu + m) * nspat;
    if (m < 0 || m >= nmu) {
        shift = (mu_halo - df) + (ptrdiff_t)((sp * (nv + 2) + v + 1) * 2 + (m >= nmu)) * nspat - at;
        return mu_halo != nullptr && v >= -1 && v <= nv;
    }
    shift = (vp_halo - df) + (ptrdiff_t)((sp * 4 + (v < 0 ? v + 2 : v - nv + 2)) * nmu + m) * nspat - at;
    return vp_halo != nullptr && v >= -2 && v <= nv + 1;
}
