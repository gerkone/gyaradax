#include <cufft.h>
#include <cufftXt.h>
#include <cuda_runtime.h>

struct CallbackInfoZ2Z {
    const double2* df_packed;
    const double2* phi_packed;
    const double*  kx;
    const double*  ky;
    const int*     inverse_jind;
    int mrad, mphi, nkx, nky, pair_type, n_df_batches, n_phi_batches;
};

__device__ double2 d_get_val_z2z(const CallbackInfoZ2Z* ci, int batch_idx, int i_lookup, int j_lookup, bool df_not_phi) {
    if (i_lookup < 0 || i_lookup >= ci->mrad) return make_double2(0.0, 0.0);
    int i_pack = ci->inverse_jind[i_lookup];
    if (i_pack < 0 || j_lookup >= ci->nky) return make_double2(0.0, 0.0);

    const double2* packed = df_not_phi ? ci->df_packed : ci->phi_packed;
    int b_idx = df_not_phi ? (batch_idx % ci->n_df_batches) : (batch_idx % ci->n_phi_batches);
    
    return packed[((unsigned long long)b_idx * ci->nkx + i_pack) * ci->nky + j_lookup];
}

__device__ cufftDoubleComplex d_z2z_load_cb_ptr(void *dataIn, unsigned long long offset, void *callerInfo, void *sharedPointer) {
    const CallbackInfoZ2Z* ci = (const CallbackInfoZ2Z*)callerInfo;
    int batch_idx = (int)(offset / ((unsigned long long)ci->mrad * ci->mphi));
    int i_dense = (int)((offset / ci->mphi) % ci->mrad);
    int j_dense = (int)(offset % ci->mphi);

    bool mirrored = false;
    int j_lookup = j_dense;
    if (j_dense > ci->mphi / 2) {
        mirrored = true;
        j_lookup = ci->mphi - j_dense;
    }

    int i_lookup = mirrored ? (ci->mrad - i_dense) % ci->mrad : i_dense;
    double2 df = d_get_val_z2z(ci, batch_idx, i_lookup, j_lookup, true);
    double2 phi = d_get_val_z2z(ci, batch_idx, i_lookup, j_lookup, false);

    double kx_val = (ci->inverse_jind[i_lookup] < 0) ? 0.0 : ci->kx[ci->inverse_jind[i_lookup]];
    double ky_val = (j_lookup >= ci->nky) ? 0.0 : ci->ky[j_lookup];

    // gradA = i * kA * valA, gradB = i * kB * valB
    // res = gradA + i * gradB
    double2 valA, valB;
    double kA, kB;
    if (ci->pair_type == 0) { // py + i*fx
        valA = phi; kA = ky_val;
        valB = df;  kB = kx_val;
    } else { // px + i*fy
        valA = phi; kA = kx_val;
        valB = df;  kB = ky_val;
    }

    double2 ga = make_double2(-kA * valA.y, kA * valA.x);
    double2 gb = make_double2(-kB * valB.y, kB * valB.x);

    double2 res;
    if (mirrored) {
        // Correct conjugate formula: Z(-k) = conj(A) + i*conj(B)
        res = make_double2(ga.x + gb.y, -ga.y + gb.x);
    } else {
        // Standard formula: Z(k) = A + i*B
        res = make_double2(ga.x - gb.y, ga.y + gb.x);
    }
    return res;
}

__device__ cufftJITCallbackLoadZ d_z2z_load_cb_ptr_addr = d_z2z_load_cb_ptr;
