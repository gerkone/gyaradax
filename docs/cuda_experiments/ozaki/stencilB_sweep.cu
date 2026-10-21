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

constexpr int NS = 18, NKY = 32, NSP = 2, NV = 32, NMU = 12, NKX = 191;
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

// signed integer value (<= 2^53) -> residue mod p_i, using only int32 ops
__device__ __forceinline__ int res_of(int64_t v, int i) {
    const bool neg = v < 0; uint64_t m = neg ? (uint64_t)(-v) : (uint64_t)v;
    const uint32_t hi = (uint32_t)(m >> 27), lo = (uint32_t)(m & ((1u << 27) - 1));
    int r = bmod((uint32_t)bmod(hi, i) * (uint32_t)c_r27[i] + (uint32_t)bmod(lo, i), i);
    return (neg && r) ? c_p[i] - r : r;
}


// tap t of T: neighbour (s + ds, ky + dky), periodic, with ds = t%NS - NS/2, dky = t/NS
__device__ __forceinline__ int nbr(int s, int ky, int t) {
    int ns = s + (t % NS) - NS / 2; if (ns < 0) ns += NS; if (ns >= NS) ns -= NS;
    int nk = ky + t / NS;           if (nk >= NKY) nk -= NKY;
    return ns * NKY + nk;
}

template <int T>
__global__ void k_fp64(const double2* __restrict__ df, const double* __restrict__ coef,
                       double2* __restrict__ out) {
    __shared__ double2 tile[NS * NKY];
    const int b = blockIdx.x, s = threadIdx.x / NKY, ky = threadIdx.x % NKY;
    const size_t base = (size_t)b * NS * NKY;
    tile[threadIdx.x] = df[base + threadIdx.x];
    __syncthreads();
    const double* cr = coef + ((size_t)(b / (NMU * NKX)) * NS + s) * T;
    double2 acc = make_double2(0.0, 0.0);
    #pragma unroll
    for (int t = 0; t < T; ++t) {
        const double2 v = tile[nbr(s, ky, t)]; const double c = cr[t];
        acc.x = fma(c, v.x, acc.x); acc.y = fma(c, v.y, acc.y);
    }
    out[base + threadIdx.x] = acc;
}

template <int T, int R, int B>
__global__ __launch_bounds__(NS * NKY, 1)
void k_ozaki(const double2* __restrict__ df, const int16_t* __restrict__ cres,
             const int* __restrict__ cexp, double2* __restrict__ out) {
    __shared__ int16_t fr[2][R][NS * NKY];
    __shared__ int smax;
    const int b = blockIdx.x, s = threadIdx.x / NKY, ky = threadIdx.x % NKY;
    const size_t base = (size_t)b * NS * NKY;
    const double2 v = df[base + threadIdx.x];
    if (threadIdx.x == 0) smax = -2000;
    __syncthreads();
    atomicMax(&smax, max(exp_bits(v.x), exp_bits(v.y)));
    __syncthreads();
    const int ef = smax + 1;
    #pragma unroll
    for (int part = 0; part < 2; ++part) {
        const int64_t q = to_fixed<B>(part ? v.y : v.x, ef);
        #pragma unroll
        for (int i = 0; i < R; ++i) fr[part][i][threadIdx.x] = (int16_t)res_of(q, i);
    }
    __syncthreads();
    const int pv = b / (NMU * NKX);
    const int16_t* cr = cres + ((size_t)pv * NS + s) * T * R;
    const int ec = cexp[pv * NS + s];
    // tap-outer: each neighbour index computed once, all moduli and both parts updated
    uint32_t acc[2][R]; int red[2][R];
    #pragma unroll
    for (int p = 0; p < 2; ++p)
        #pragma unroll
        for (int i = 0; i < R; ++i) { acc[p][i] = 0; red[p][i] = 0; }
    #pragma unroll 4
    for (int t = 0; t < T; ++t) {
        const int idx = nbr(s, ky, t);
        #pragma unroll
        for (int i = 0; i < R; ++i) {
            const uint32_t c = (uint32_t)cr[t * R + i];
            acc[0][i] += c * (uint32_t)fr[0][i][idx];
            acc[1][i] += c * (uint32_t)fr[1][i][idx];
        }
        if ((t & 15) == 15) {
            #pragma unroll
            for (int p = 0; p < 2; ++p)
                #pragma unroll
                for (int i = 0; i < R; ++i) {
                    red[p][i] = bmod((uint32_t)(red[p][i] + bmod(acc[p][i], i)), i); acc[p][i] = 0;
                }
        }
    }
    double2 o;
    #pragma unroll
    for (int part = 0; part < 2; ++part) {
        int a[R];
        #pragma unroll
        for (int i = 0; i < R; ++i) {
            int x = bmod((uint32_t)(red[part][i] + bmod(acc[part][i], i)), i);
            #pragma unroll
            for (int j = 0; j < i; ++j) {
                int aj = a[j] >= c_p[i] ? a[j] - c_p[i] : a[j];
                int dlt = x - aj; if (dlt < 0) dlt += c_p[i];
                x = bmod((uint32_t)dlt * (uint32_t)c_inv[i][j], i);
            }
            a[i] = x;
        }
        const bool neg = 2 * a[R - 1] >= c_p[R - 1];
        uint64_t Shi = 0, Slo = neg ? 1 : 0;
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

template <int T> static float time_fp64(double2* d_df, double* d_coef, double2* d_out) {
    k_fp64<T><<<NB, NS * NKY>>>(d_df, d_coef, d_out); cudaDeviceSynchronize();
    cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
    cudaEventRecord(e0); for (int r = 0; r < 10; ++r) k_fp64<T><<<NB, NS * NKY>>>(d_df, d_coef, d_out);
    cudaEventRecord(e1); cudaEventSynchronize(e1); float ms; cudaEventElapsedTime(&ms, e0, e1); return ms / 10;
}
template <int T, int R, int B>
static void run(double2* d_df, double2* d_out, const std::vector<double>& coef,
                const std::vector<double2>& ref, float t64) {
    auto P = primes(R);
    uint32_t M32[RMAX]; int r27[RMAX]; int inv[RMAX][RMAX] = {};
    for (int i = 0; i < R; ++i) {
        M32[i] = (uint32_t)((1ull << 32) / P[i]); r27[i] = (int)((1ll << 27) % P[i]);
        for (int j = 0; j < i; ++j) inv[i][j] = (int)modinv(P[j], P[i]);
    }
    cudaMemcpyToSymbol(c_p, P.data(), R * sizeof(int)); cudaMemcpyToSymbol(c_m, M32, R * 4);
    cudaMemcpyToSymbol(c_r27, r27, R * 4); cudaMemcpyToSymbol(c_inv, inv, sizeof(inv));
    { unsigned __int128 w = 1; uint64_t hi[RMAX], lo[RMAX];
      for (int i = 0; i < R; ++i) { hi[i] = (uint64_t)(w >> 64); lo[i] = (uint64_t)w; w *= (unsigned)P[i]; }
      cudaMemcpyToSymbol(c_Mhi, hi, R * 8); cudaMemcpyToSymbol(c_Mlo, lo, R * 8); }
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
    k_ozaki<T, R, B><<<NB, NS * NKY>>>(d_df, d_cres, d_cexp, d_out); cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) { printf("      r=%d: CUDA error %s\n", R, cudaGetErrorString(err)); return; }
    cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
    cudaEventRecord(e0); for (int r = 0; r < 10; ++r) k_ozaki<T, R, B><<<NB, NS * NKY>>>(d_df, d_cres, d_cexp, d_out);
    cudaEventRecord(e1); cudaEventSynchronize(e1); float ms; cudaEventElapsedTime(&ms, e0, e1); ms /= 10;
    std::vector<double2> got(ref.size()); cudaMemcpy(got.data(), d_out, got.size() * 16, cudaMemcpyDeviceToHost);
    double mx = 0, emax = 0;
    for (auto& r : ref) mx = std::max({mx, std::fabs(r.x), std::fabs(r.y)});
    for (size_t i = 0; i < got.size(); ++i)
        emax = std::max({emax, std::fabs(got[i].x - ref[i].x), std::fabs(got[i].y - ref[i].y)});
    printf("      Ozaki r=%d B=%2d   %8.3f ms   %5.2fx vs FP64   err %.1e\n", R, B, ms, t64 / ms, emax / mx);
    cudaFree(d_cres); cudaFree(d_cexp);
}
template <int T>
static void sweep(double2* d_df, double2* d_out, std::mt19937_64& g) {
    std::normal_distribution<double> d;
    std::vector<double> coef((size_t)NSP * NV * NS * T);
    for (auto& c : coef) c = 50.0 * d(g);
    double* d_coef; cudaMalloc(&d_coef, coef.size() * 8);
    cudaMemcpy(d_coef, coef.data(), coef.size() * 8, cudaMemcpyHostToDevice);
    const float t64 = time_fp64<T>(d_df, d_coef, d_out);
    const size_t N = (size_t)NB * NS * NKY;
    std::vector<double2> ref(N); cudaMemcpy(ref.data(), d_out, N * 16, cudaMemcpyDeviceToHost);
    printf("  T=%2d taps  (%3d real FMA/output)   native FP64 %8.3f ms   %.2f TFLOP/s\n",
           T, 2 * T, t64, 4.0 * T * N / (t64 * 1e9));
    run<T, 5, 31>(d_df, d_out, coef, ref, t64);
    run<T, 8, 53>(d_df, d_out, coef, ref, t64);
    cudaFree(d_coef);
}
int main() {
    const size_t N = (size_t)NB * NS * NKY;
    std::mt19937_64 g(0); std::normal_distribution<double> d;
    std::vector<double2> h(N);
    for (size_t b = 0; b < NB; ++b) {
        const double amp = 1e-3 * std::exp(0.5 * d(g));
        for (int s = 0; s < NS; ++s) for (int ky = 0; ky < NKY; ++ky) {
            const double ph = 0.3 * s + 0.1 * ky;
            h[(b * NS + s) * NKY + ky] = make_double2(amp * (std::cos(ph) + 0.05 * d(g)),
                                                      amp * (std::sin(ph) + 0.05 * d(g)));
        }
    }
    double2 *d_df, *d_out; cudaMalloc(&d_df, N * 16); cudaMalloc(&d_out, N * 16);
    cudaMemcpy(d_df, h.data(), N * 16, cudaMemcpyHostToDevice);
    cudaDeviceProp p; cudaGetDeviceProperties(&p, 0);
    printf("  %s, %zu outputs\n\n", p.name, N);
    sweep<5>(d_df, d_out, g);
    sweep<10>(d_df, d_out, g);
    sweep<20>(d_df, d_out, g);
    sweep<30>(d_df, d_out, g);
    sweep<45>(d_df, d_out, g);
    sweep<60>(d_df, d_out, g);
}
