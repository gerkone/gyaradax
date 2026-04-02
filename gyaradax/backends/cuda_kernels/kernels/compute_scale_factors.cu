#include <cuda_runtime.h>
#include <device_launch_parameters.h>

struct ScaleFactors {
    double alpha0, beta0;     // max magnitudes (for host visibility)
    double alpha1, beta1;
    double inv_a0, inv_b0;    // reciprocals for fast kernel usage
    double inv_a1, inv_b1;
};

// Simple reduction kernel using atomicMax on doubles
// Note: atomicMax for doubles requires a CAS loop on older hardware, 
// but we are on A100 (sm_80) which supports it or we can use the loop.
__device__ double atomicMaxDouble(double* address, double val) {
    unsigned long long int* address_as_ull = (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;
    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed,
                        __double_as_longlong(fmax(val, __longlong_as_double(assumed))));
    } while (assumed != old);
    return __longlong_as_double(old);
}
__global__ void compute_scale_factors_kernel(
    const double2* __restrict__ phi,
    const double2* __restrict__ df,
    const double*  __restrict__ kx,
    const double*  __restrict__ ky,
    int nkx, int nky, int batch,
    size_t phi_size, size_t df_size,
    ScaleFactors* out)
{
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (size_t)nkx * nky) return;

    int i = (int)(idx / nky);
    int j = (int)(idx % nky);

    double kx_val = kx[i];
    double ky_val = ky[j];

    double2 p_val = phi[idx]; 
    double2 d_val = df[idx]; 

    // |i*k*val| = |k| * |val|
    double mag_p = sqrt(p_val.x * p_val.x + p_val.y * p_val.y);
    double mag_d = sqrt(d_val.x * d_val.x + d_val.y * d_val.y);

    double v_alpha0 = fabs(ky_val) * mag_p; // ky * phi
    double v_beta0  = fabs(kx_val) * mag_d; // kx * df
    double v_alpha1 = fabs(kx_val) * mag_p; // kx * phi
    double v_beta1  = fabs(ky_val) * mag_d; // ky * df

    atomicMaxDouble(&out->alpha0, v_alpha0);
    atomicMaxDouble(&out->beta0, v_beta0);
    atomicMaxDouble(&out->alpha1, v_alpha1);
    atomicMaxDouble(&out->beta1, v_beta1);
}

__global__ void compute_reciprocals_kernel(ScaleFactors* sf) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        sf->inv_a0 = 1.0 / sf->alpha0;
        sf->inv_b0 = 1.0 / sf->beta0;
        sf->inv_a1 = 1.0 / sf->alpha1;
        sf->inv_b1 = 1.0 / sf->beta1;
    }
}

extern "C" void launch_compute_scale_factors(
    cudaStream_t stream,
    const double2* phi, const double2* df,
    const double* kx, const double* ky,
    int nkx, int nky, int batch,
    size_t phi_size, size_t df_size,
    ScaleFactors* d_out)
{
    // Initialize out to very small values to ensure we pick up the max magnitude.
    static const ScaleFactors init = {1e-15, 1e-15, 1e-15, 1e-15, 1.0, 1.0, 1.0, 1.0};
    cudaMemcpyAsync(d_out, &init, sizeof(ScaleFactors), cudaMemcpyHostToDevice, stream);

    int threads = 256;
    size_t total = (size_t)nkx * nky;
    int blocks = (int)((total + threads - 1) / threads);
    compute_scale_factors_kernel<<<blocks, threads, 0, stream>>>(phi, df, kx, ky, nkx, nky, batch, phi_size, df_size, d_out);
    compute_reciprocals_kernel<<<1, 1, 0, stream>>>(d_out);
}
