// Path A: fused SIMT Ozaki-II (CRT) for the gyaradax parallel stencil, vs FP64,
// same harness, on this GPU. One block per batch (sp,v,mu,kx); threads (s, ky).
// out[s][ky] = sum_t c[(sp,v)][s][t] * df[s+t-2][ky], zero outside 0..NS-1.
// Coefficients are deconstructed once on the host (persistent, phi < 1); df is
// deconstructed in-kernel with a tile-wide power-of-two scale so all five taps
// reuse it. int32 residues mod primes < 2^14: a 5-term sum of products stays in
// int32, so each modulus needs a single reduction per output (lazy).
#include <cstdio>
#include <cstdint>
#include <cmath>
#include <vector>
#include <random>
#include <algorithm>

constexpr int NS = 18, NKY = 32, T = 5, NSP = 2, NV = 32, NMU = 12, NKX = 191;
constexpr int NB = NSP * NV * NMU * NKX;        // 146688 batches
constexpr int RMAX = 9;
__constant__ int      c_p[RMAX];                // primes
__constant__ uint32_t c_m[RMAX];                // Barrett: floor(2^32 / p)
__constant__ int      c_r27[RMAX];              // 2^27 mod p
__constant__ int      c_inv[RMAX][RMAX];        // inv(p_j) mod p_i, j < i
__constant__ double   c_M[RMAX];                // mixed-radix weights prod_{j<k} p_j
__constant__ uint64_t c_Mhi[RMAX], c_Mlo[RMAX];   // same, exact 128-bit

__device__ __forceinline__ int bmod(uint32_t x, int i) {      // x mod p_i, x < 2^32
    uint32_t q = __umulhi(x, c_m[i]);
    int r = (int)(x - q * (uint32_t)c_p[i]);
    return r >= c_p[i] ? r - c_p[i] : r;
}


// unbiased binary exponent from the raw bits; zeros/denormals map very low
__device__ __forceinline__ int exp_bits(double x) {
    const int e = (int)((__double_as_longlong(x) >> 52) & 0x7FF);
    return e == 0 ? -2000 : e - 1023;
}
// round(x * 2^(B-1-ef)) as a signed integer, from the bits only (no FP64 ops)
template <int B>
__device__ __forceinline__ int64_t to_fixed(double x, int ef) {
    const long long b = __double_as_longlong(x);
    const int ebits = (int)((b >> 52) & 0x7FF);
    if (ebits == 0) return 0;
    const uint64_t m = (uint64_t)(b & 0xFFFFFFFFFFFFFull) | (1ull << 52);
    const int sh = (ebits - 1023 - 52) + (B - 1 - ef);
    uint64_t q;
    if (sh >= 0) q = m << sh;
    else { const int rs = -sh; q = rs >= 64 ? 0 : ((m >> (rs - 1)) + 1) >> 1; }
    return b < 0 ? -(int64_t)q : (int64_t)q;
}
// exact 128-bit unsigned (hi,lo) -> double * 2^scale, built from bits (no FP64 ops)
__device__ __forceinline__ double u128_to_double(uint64_t hi, uint64_t lo, int scale, bool neg) {
    if (hi == 0 && lo == 0) return 0.0;
    int lz = hi ? __clzll(hi) : 64 + __clzll(lo);
    const int top = 127 - lz;                       // index of leading bit
    // take bits [top-52 .. top] (53 bits) with round-to-nearest-even-ish
    uint64_t mant; int drop = top - 52;
    if (drop <= 0) mant = (drop == 0) ? lo : lo << (-drop);
    else {
        // shift (hi,lo) right by drop
        uint64_t sh_lo = drop >= 64 ? hi >> (drop - 64) : (lo >> drop) | (drop ? hi << (64 - drop) : 0);
        const uint64_t half = drop >= 65 ? 0 : 0;   // rounding: inspect bit drop-1
        uint64_t rb;
        const int rbi = drop - 1;
        rb = rbi >= 64 ? (hi >> (rbi - 64)) & 1 : (lo >> rbi) & 1;
        mant = sh_lo + rb;
        (void)half;
        if (mant >> 53) { mant >>= 1; ++drop; }
    }
    const int e = (top - 52 + drop - (top - 52)) + 52 + scale; // unbiased exponent of the value
    const int64_t ex = (int64_t)(top + scale) + 1023;
    if (ex <= 0) return 0.0;
    const uint64_t bits = ((uint64_t)neg << 63) | ((uint64_t)ex << 52) | (mant & 0xFFFFFFFFFFFFFull);
    (void)e;
    return __longlong_as_double((long long)bits);
}

// ---------------- FP64 reference kernel ----------------
__global__ void k_fp64(const double2* __restrict__ df, const double* __restrict__ coef,
                       double2* __restrict__ out) {
    __shared__ double2 t[NS * NKY];
    const int b = blockIdx.x, s = threadIdx.x / NKY, ky = threadIdx.x % NKY;
    const size_t base = (size_t)b * NS * NKY;
    t[threadIdx.x] = df[base + threadIdx.x];
    __syncthreads();
    const double* cr = coef + ((size_t)(b / (NMU * NKX)) * NS + s) * T;
    double2 acc = make_double2(0.0, 0.0);
    #pragma unroll
    for (int tt = 0; tt < T; ++tt) {
        const int ss = s + tt - 2;
        if (ss < 0 || ss >= NS) continue;
        const double2 v = t[ss * NKY + ky]; const double c = cr[tt];
        acc.x = fma(c, v.x, acc.x); acc.y = fma(c, v.y, acc.y);
    }
    out[base + threadIdx.x] = acc;
}

// ---------------- Ozaki-II modular kernel ----------------
// signed integer value (<= 2^53) -> residue mod p_i, using only int32 ops
__device__ __forceinline__ int res_of(int64_t v, int i) {
    const bool neg = v < 0; uint64_t m = neg ? (uint64_t)(-v) : (uint64_t)v;
    const uint32_t hi = (uint32_t)(m >> 27), lo = (uint32_t)(m & ((1u << 27) - 1));
    int r = bmod((uint32_t)bmod(hi, i) * (uint32_t)c_r27[i] + (uint32_t)bmod(lo, i), i);
    return (neg && r) ? c_p[i] - r : r;
}

template <int R, int B>
__global__ void k_ozaki(const double2* __restrict__ df, const int16_t* __restrict__ cres,
                        const int* __restrict__ cexp, double2* __restrict__ out) {
    __shared__ int16_t fr[2][R][NS * NKY];       // residues of re / im
    __shared__ int smax;
    const int b = blockIdx.x, s = threadIdx.x / NKY, ky = threadIdx.x % NKY;
    const size_t base = (size_t)b * NS * NKY;
    const double2 v = df[base + threadIdx.x];
    // tile-wide scale: exponent of max |re|,|im| over the tile, via the exponent bits
    if (threadIdx.x == 0) smax = -2000;
    __syncthreads();
    int e = max(exp_bits(v.x), exp_bits(v.y));
    atomicMax(&smax, e);
    __syncthreads();
    const int ef = smax + 1;                     // |x| < 2^ef for every x in the tile
    // deconstruct: x -> round(x * 2^(B-1-ef)) as int64, then R residues
    #pragma unroll
    for (int part = 0; part < 2; ++part) {
        const double x = part ? v.y : v.x;
        const int64_t q = to_fixed<B>(x, ef);
        #pragma unroll
        for (int i = 0; i < R; ++i) fr[part][i][threadIdx.x] = (int16_t)res_of(q, i);
    }
    __syncthreads();
    const int pv = b / (NMU * NKX);
    const int16_t* cr = cres + ((size_t)pv * NS + s) * T * R;   // [t][i]
    const int ec = cexp[pv * NS + s];
    double2 o;
    #pragma unroll
    for (int part = 0; part < 2; ++part) {
        int res[R];
        #pragma unroll
        for (int i = 0; i < R; ++i) {
            int acc = 0;
            #pragma unroll
            for (int tt = 0; tt < T; ++tt) {
                const int ss = s + tt - 2;
                if (ss < 0 || ss >= NS) continue;
                acc += (int)cr[tt * R + i] * (int)fr[part][i][ss * NKY + ky];
            }
            res[i] = bmod((uint32_t)acc, i);
        }
        // Garner: mixed-radix digits a_k
        int a[R];
        #pragma unroll
        for (int i = 0; i < R; ++i) {
            int x = res[i];
            #pragma unroll
            for (int j = 0; j < i; ++j) {
                int aj = a[j] >= c_p[i] ? a[j] - c_p[i] : a[j];
                int dlt = x - aj; if (dlt < 0) dlt += c_p[i];
                x = bmod((uint32_t)dlt * (uint32_t)c_inv[i][j], i);
            }
            a[i] = x;
        }
        // signed value: S = sum a_k M_k, centred
        const bool neg = 2 * a[R - 1] >= c_p[R - 1];
        uint64_t Shi = 0, Slo = neg ? 1 : 0;          // complement form carries the +1
        #pragma unroll
        for (int k = 0; k < R; ++k) {
            const uint64_t d = (uint64_t)(neg ? c_p[k] - 1 - a[k] : a[k]);
            const uint64_t plo = d * c_Mlo[k];
            const uint64_t phi = __umul64hi(d, c_Mlo[k]) + d * c_Mhi[k];
            Slo += plo; Shi += phi + (Slo < plo);
        }
        const double val = u128_to_double(Shi, Slo, ec + ef - 2 * (B - 1), neg);
        if (part) o.y = val; else o.x = val;
    }
    out[base + threadIdx.x] = o;
}

// ---------------- host ----------------
static std::vector<int> primes(int k) {
    std::vector<int> o; for (int x = (1 << 14) - 1; (int)o.size() < k; --x) {
        bool pr = x > 1; for (int d = 2; d * d <= x; ++d) if (x % d == 0) { pr = false; break; }
        if (pr) o.push_back(x); }
    return o;
}
static long long modinv(long long a, long long m) {
    long long g = m, x = 0, x1 = 1, a1 = a % m; while (a1) { long long q = g / a1;
        long long t = g - q * a1; g = a1; a1 = t; t = x - q * x1; x = x1; x1 = t; }
    return (x % m + m) % m;
}
template <int R, int B>
static void run_ozaki(const char* tag, double2* d_df, const std::vector<double>& coef,
                      double2* d_out, const std::vector<double2>& ref, float t64) {
    auto P = primes(R);
    uint32_t M32[RMAX]; int r27[RMAX]; int inv[RMAX][RMAX] = {}; double Mw[RMAX];
    double acc = 1.0;
    for (int i = 0; i < R; ++i) {
        M32[i] = (uint32_t)((1ull << 32) / P[i]); r27[i] = (int)((1ll << 27) % P[i]);
        for (int j = 0; j < i; ++j) inv[i][j] = (int)modinv(P[j], P[i]);
        Mw[i] = acc; acc *= P[i];
    }
    cudaMemcpyToSymbol(c_p, P.data(), R * sizeof(int)); cudaMemcpyToSymbol(c_m, M32, R * 4);
    cudaMemcpyToSymbol(c_r27, r27, R * 4); cudaMemcpyToSymbol(c_inv, inv, sizeof(inv));
    cudaMemcpyToSymbol(c_M, Mw, R * 8);
    { unsigned __int128 w = 1; uint64_t hi[RMAX], lo[RMAX];
      for (int i = 0; i < R; ++i) { hi[i] = (uint64_t)(w >> 64); lo[i] = (uint64_t)w; w *= (unsigned)P[i]; }
      cudaMemcpyToSymbol(c_Mhi, hi, R * 8); cudaMemcpyToSymbol(c_Mlo, lo, R * 8); }
    // persistent coefficient deconstruction: per (pv, s) row scale
    const int NPV = NSP * NV;
    std::vector<int16_t> cres((size_t)NPV * NS * T * R); std::vector<int> cexp(NPV * NS);
    for (int pv = 0; pv < NPV; ++pv) for (int s = 0; s < NS; ++s) {
        const double* cr = &coef[((size_t)pv * NS + s) * T];
        double mx = 0; for (int t = 0; t < T; ++t) mx = std::max(mx, std::fabs(cr[t]));
        const int ec = mx > 0 ? std::ilogb(mx) + 1 : 0; cexp[pv * NS + s] = ec;
        for (int t = 0; t < T; ++t) {
            const long long q = std::llround(std::ldexp(cr[t], B - 1 - ec));
            for (int i = 0; i < R; ++i)
                cres[(((size_t)pv * NS + s) * T + t) * R + i] = (int16_t)(((q % P[i]) + P[i]) % P[i]);
        }
    }
    int16_t* d_cres; int* d_cexp;
    cudaMalloc(&d_cres, cres.size() * 2); cudaMalloc(&d_cexp, cexp.size() * 4);
    cudaMemcpy(d_cres, cres.data(), cres.size() * 2, cudaMemcpyHostToDevice);
    cudaMemcpy(d_cexp, cexp.data(), cexp.size() * 4, cudaMemcpyHostToDevice);
    k_ozaki<R, B><<<NB, NS * NKY>>>(d_df, d_cres, d_cexp, d_out); cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) { printf("  %s: CUDA error %s\n", tag, cudaGetErrorString(err)); return; }
    cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
    cudaEventRecord(e0); for (int r = 0; r < 10; ++r) k_ozaki<R, B><<<NB, NS * NKY>>>(d_df, d_cres, d_cexp, d_out);
    cudaEventRecord(e1); cudaEventSynchronize(e1); float ms; cudaEventElapsedTime(&ms, e0, e1); ms /= 10;
    std::vector<double2> got(ref.size()); cudaMemcpy(got.data(), d_out, got.size() * 16, cudaMemcpyDeviceToHost);
    double mx = 0, emax = 0; std::vector<double> rel; rel.reserve(got.size() * 2);
    for (size_t i = 0; i < got.size(); ++i) { mx = std::max({mx, std::fabs(ref[i].x), std::fabs(ref[i].y)}); }
    for (size_t i = 0; i < got.size(); ++i) {
        const double a = std::fabs(got[i].x - ref[i].x), c = std::fabs(got[i].y - ref[i].y);
        emax = std::max({emax, a, c});
        if (std::fabs(ref[i].x) > 1e-12 * mx) rel.push_back(a / std::fabs(ref[i].x));
    }
    std::nth_element(rel.begin(), rel.begin() + rel.size() / 2, rel.end());
    printf("  %-22s %8.3f ms  %6.2fx vs FP64   max err/max|ref| %.1e   median rel %.1e\n",
           tag, ms, t64 / ms, emax / mx, rel[rel.size() / 2]);
    cudaFree(d_cres); cudaFree(d_cexp);
}

int main() {
    const size_t N = (size_t)NB * NS * NKY;
    std::mt19937_64 g(0); std::normal_distribution<double> d;
    std::vector<double2> h(N);
    // smooth-in-s physical-like field: neighbours along s are similar in magnitude
    for (size_t b = 0; b < NB; ++b) {
        const double amp = 1e-3 * std::exp(0.5 * d(g));
        for (int s = 0; s < NS; ++s) for (int ky = 0; ky < NKY; ++ky) {
            const double ph = 0.3 * s + 0.1 * ky;
            h[(b * NS + s) * NKY + ky] = make_double2(amp * (std::cos(ph) + 0.05 * d(g)),
                                                      amp * (std::sin(ph) + 0.05 * d(g)));
        }
    }
    std::vector<double> coef((size_t)NSP * NV * NS * T);
    const double row[T] = {1.0 / 12, -8.0 / 12, 0.0, 8.0 / 12, -1.0 / 12};
    for (int pv = 0; pv < NSP * NV; ++pv) {
        const double up = (pv < NV ? 1.0 : 60.6) * (pv % NV - 15.5) / 16.0 * 9.39;  // ion / electron vpar
        for (int s = 0; s < NS; ++s) for (int t = 0; t < T; ++t)
            coef[((size_t)pv * NS + s) * T + t] = up * row[t] / 0.0625 + (t == 2 ? 0.3 * d(g) : 0.0);
    }
    double2 *d_df, *d_out; double* d_coef;
    cudaMalloc(&d_df, N * 16); cudaMalloc(&d_out, N * 16); cudaMalloc(&d_coef, coef.size() * 8);
    cudaMemcpy(d_df, h.data(), N * 16, cudaMemcpyHostToDevice);
    cudaMemcpy(d_coef, coef.data(), coef.size() * 8, cudaMemcpyHostToDevice);
    k_fp64<<<NB, NS * NKY>>>(d_df, d_coef, d_out); cudaDeviceSynchronize();
    cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
    cudaEventRecord(e0); for (int r = 0; r < 10; ++r) k_fp64<<<NB, NS * NKY>>>(d_df, d_coef, d_out);
    cudaEventRecord(e1); cudaEventSynchronize(e1); float t64; cudaEventElapsedTime(&t64, e0, e1); t64 /= 10;
    std::vector<double2> ref(N); cudaMemcpy(ref.data(), d_out, N * 16, cudaMemcpyDeviceToHost);
    cudaDeviceProp p; cudaGetDeviceProperties(&p, 0);
    printf("  %s   %zu outputs, df %.2f GB\n", p.name, N, N * 16 / 1e9);
    printf("  %-22s %8.3f ms   (%.0f GB/s on df in+out)\n", "native FP64", t64, 2 * N * 16 / (t64 * 1e6));
    run_ozaki<5, 31>("Ozaki r=5  B=31 (1e-9)", d_df, coef, d_out, ref, t64);
    run_ozaki<6, 40>("Ozaki r=6  B=40", d_df, coef, d_out, ref, t64);
    run_ozaki<8, 53>("Ozaki r=8  B=53 (FP64)", d_df, coef, d_out, ref, t64);
    run_ozaki<9, 53>("Ozaki r=9  B=53", d_df, coef, d_out, ref, t64);
}
