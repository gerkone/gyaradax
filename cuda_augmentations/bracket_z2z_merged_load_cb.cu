#include <cufft.h>
#include <cufftXt.h>
#include <cuda_runtime.h>

struct ScaleFactors {
    double alpha0, beta0;
    double alpha1, beta1;
    double inv_a0, inv_b0;
    double inv_a1, inv_b1;
};

struct CallbackInfoZ2Z_Merged {
    const double2* df_packed;
    const double2* phi_packed;
    const double*  kx;
    const double*  ky;
    const int*     inverse_jind;
    const ScaleFactors* sf;
    int mrad, mphi, nkx, nky, field_boundary;
    size_t df_size, phi_size;
};

__device__ double2 d_get_val_z2z_merged(const CallbackInfoZ2Z_Merged* ci, int local_batch, int i_lookup, int j_lookup, bool df_not_phi) {
    if (i_lookup < 0 || i_lookup >= ci->mrad) return make_double2(0.0, 0.0);
    int i_pack = ci->inverse_jind[i_lookup];
    if (i_pack < 0 || j_lookup >= ci->nky) return make_double2(0.0, 0.0);

    const double2* base = df_not_phi ? ci->df_packed : ci->phi_packed;
    size_t actual_batch_size = (df_not_phi ? ci->df_size : ci->phi_size) / (ci->nkx * ci->nky);
    if (actual_batch_size == 0) return make_double2(0.0, 0.0);
    int b_idx = local_batch % actual_batch_size;
    return base[(size_t)b_idx * ci->nkx * ci->nky + (size_t)i_pack * ci->nky + j_lookup];
}

__device__ cufftDoubleComplex d_z2z_merged_load(void *dataIn, unsigned long long offset, void *callerInfo, void *sharedPointer) {
    const CallbackInfoZ2Z_Merged* ci = (const CallbackInfoZ2Z_Merged*)callerInfo;
    int global_batch = (int)(offset / ((unsigned long long)ci->mrad * ci->mphi));
    int pair_type = (global_batch < ci->field_boundary) ? 0 : 1;
    int local_batch = (pair_type == 0) ? global_batch : (global_batch - ci->field_boundary);

    int i_dense = (int)((offset / ci->mphi) % ci->mrad);
    int j_dense = (int)(offset % ci->mphi);

    bool mirrored = false;
    int j_lookup = j_dense;
    if (j_dense > ci->mphi / 2) {
        mirrored = true;
        j_lookup = ci->mphi - j_dense;
    }

    int i_lookup = mirrored ? (ci->mrad - i_dense) % ci->mrad : i_dense;
    double2 df = d_get_val_z2z_merged(ci, local_batch, i_lookup, j_lookup, true);
    double2 phi = d_get_val_z2z_merged(ci, local_batch, i_lookup, j_lookup, false);

    // Symmetrize j=0 column to eliminate imaginary residual leakage
    if (j_lookup == 0) {
        int i_mirror = (ci->mrad - i_lookup) % ci->mrad;
        int ip_mirror = ci->inverse_jind[i_mirror];
        if (ip_mirror >= 0 && i_mirror != i_lookup) {
            double2 df_m = d_get_val_z2z_merged(ci, local_batch, i_mirror, 0, true);
            double2 phi_m = d_get_val_z2z_merged(ci, local_batch, i_mirror, 0, false);
            
            df.x = (df.x + df_m.x) * 0.5;
            df.y = (df.y - df_m.y) * 0.5; // (y - y_m)/2 is the imag part
            
            phi.x = (phi.x + phi_m.x) * 0.5;
            phi.y = (phi.y - phi_m.y) * 0.5;
        }
    }

    double kx_val = (ci->inverse_jind[i_lookup] < 0) ? 0.0 : ci->kx[ci->inverse_jind[i_lookup]];
    double ky_val = (j_lookup >= ci->nky) ? 0.0 : ci->ky[j_lookup];

    double2 valA, valB;
    double kA, kB;
    if (pair_type == 0) { // py + i*fx (phi)
        valA = phi; kA = ky_val;
        valB = df;  kB = kx_val;
    } else { // px + i*fy (df)
        valA = phi; kA = kx_val;
        valB = df;  kB = ky_val;
    }

    double inv_alpha = (pair_type == 0) ? ci->sf->inv_a0 : ci->sf->inv_a1;
    double inv_beta  = (pair_type == 0) ? ci->sf->inv_b0  : ci->sf->inv_b1;

    double2 ga = make_double2(-kA * valA.y * inv_alpha, kA * valA.x * inv_alpha);
    double2 gb = make_double2(-kB * valB.y * inv_beta,  kB * valB.x * inv_beta);

    double2 res;
    if (mirrored) {
        res = make_double2(ga.x + gb.y, -ga.y + gb.x);
    } else {
        res = make_double2(ga.x - gb.y, ga.y + gb.x);
    }
    return res;
}

__device__ cufftJITCallbackLoadZ d_z2z_merged_load_addr = d_z2z_merged_load;
