// Callback info structs of the v6 bracket column passes (shared by the LTO callbacks and the host).
#pragma once

#include "bracket_v5_z2z_pack.cuh"

// column-pass input A: [mrad][n_planes][n_cols], n_cols = 2 nky - 1 retained ky columns
struct V6ColInfo {
    V5Z2zInfo z;
    int n_planes, n_cols;
};

// column-pass output B: [mrad][n_df][nky] -> packed [n_df, nkx, nky]
struct V6StoreInfo {
    double2*   out_packed;
    const int* inverse_jind;
    int mrad, n_df, nkx, nky;
    int ixzero, iyzero;
};

// retained column jj -> dense ky index j of the 2-for-1 packed spectrum
__device__ __forceinline__ static int v6_col_to_j(int jj, int nky, int mphi)
{
    return jj < nky ? jj : mphi - (2 * nky - 1 - jj);
}

__device__ __forceinline__ static void v6_col_split(unsigned long long offset, const V6ColInfo* ci,
                                                    int* gb, int* i, int* j)
{
    const unsigned long long row = (unsigned long long)ci->n_planes * ci->n_cols;
    *i = (int)(offset / row);
    const int c = (int)(offset - (unsigned long long)*i * row);
    *gb = c / ci->n_cols;
    *j = v6_col_to_j(c - *gb * ci->n_cols, ci->z.nky, ci->z.mphi);
}
