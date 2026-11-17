#pragma once
#include <cuda_runtime.h>
#include <stdint.h>

// double-single arithmetic: a value is hi + lo, two fp32 words (~48 significant bits)

#define DSI __device__ __forceinline__

struct ds { float hi, lo; };
struct cds { ds x, y; };

DSI float fadd(float a, float b) { return __fadd_rn(a, b); }
DSI float fsub(float a, float b) { return __fsub_rn(a, b); }
DSI float fmul(float a, float b) { return __fmul_rn(a, b); }
DSI float ffma(float a, float b, float c) { return __fmaf_rn(a, b, c); }

DSI void two_sum(float a, float b, float& s, float& e) {
    s = fadd(a, b);
    const float bb = fsub(s, a);
    e = fadd(fsub(a, fsub(s, bb)), fsub(b, bb));
}

DSI void fast_two_sum(float a, float b, float& s, float& e) {
    s = fadd(a, b);
    e = fsub(b, fsub(s, a));
}

DSI void two_prod(float a, float b, float& p, float& e) {
    p = fmul(a, b);
    e = ffma(a, b, -p);
}

DSI ds ds_neg(ds a) { return {-a.hi, -a.lo}; }

DSI ds ds_mul(ds a, ds b) {
    float p, e;
    two_prod(a.hi, b.hi, p, e);
    e = ffma(a.hi, b.lo, e);
    e = ffma(a.lo, b.hi, e);
    fast_two_sum(p, e, p, e);
    return {p, e};
}

// acc + a * b
DSI ds ds_fma(ds a, ds b, ds acc) {
    float p, e, s, t;
    two_prod(a.hi, b.hi, p, e);
    two_sum(acc.hi, p, s, t);
    e = ffma(a.hi, b.lo, e);
    e = ffma(a.lo, b.hi, e);
    t = fadd(t, fadd(e, acc.lo));
    fast_two_sum(s, t, s, t);
    return {s, t};
}

DSI ds ds_ldg(const ds* p) {
    const float2 v = __ldg(reinterpret_cast<const float2*>(p));
    return {v.x, v.y};
}

// lazy accumulator: value s + t, t gathers the rounding errors until renorm
struct DsAcc {
    float s, t;
    DSI void init() { s = 0.0f; t = 0.0f; }
    DSI void fma(ds a, ds b) {
        float p, e, r;
        two_prod(a.hi, b.hi, p, e);
        two_sum(s, p, s, r);
        t = ffma(a.hi, b.lo, t);
        t = ffma(a.lo, b.hi, t);
        t = fadd(t, fadd(e, r));
    }
    DSI void renorm() { fast_two_sum(s, t, s, t); }
    DSI ds fin() const { float h, l; fast_two_sum(s, t, h, l); return {h, l}; }
};

// fp64 <-> ds on the integer pipe, valid for 2^-126 <= |d| < 2^115 (anything else maps to 0)

DSI float pow2f(int k) { return __uint_as_float((unsigned)(k + 127) << 23); }

DSI ds ds_from_d(double d) {
    const unsigned long long b = (unsigned long long)__double_as_longlong(d);
    const unsigned bh = (unsigned)(b >> 32);
    const int e = (bh >> 20) & 0x7ff;
    const unsigned sgn = bh & 0x80000000u;
    const bool ok = e >= 897 && e <= 1138;
    float hi = __uint_as_float(sgn | ((unsigned)(e - 896) << 23) | ((unsigned)(b >> 29) & 0x7fffffu));
    float lo = fmul(fmul(__uint2float_rn((unsigned)b & 0x1fffffffu), pow2f(e - 1011)), 0x1p-64f);
    lo = __uint_as_float(__float_as_uint(lo) ^ sgn);
    hi = ok ? hi : 0.0f;
    lo = ok ? lo : 0.0f;
    fast_two_sum(hi, lo, hi, lo);
    return {hi, lo};
}

// offset of lo from hi in units of the fp64 ulp of hi, half step below a power of two
DSI float ulp_scale(unsigned hb, float lo) {
    const unsigned e8 = (hb >> 23) & 0xff;
    float s = __uint_as_float((306u - e8) << 23);
    const bool down = ((hb & 0x7fffffu) == 0) && (((__float_as_uint(lo) ^ hb) & 0x80000000u) != 0);
    return down ? fmul(s, 2.0f) : s;
}

DSI double ds_to_d(ds a) {
    const unsigned hb = __float_as_uint(a.hi);
    const unsigned e8 = (hb >> 23) & 0xff;
    const int n = __float2int_rn(fmul(a.lo, ulp_scale(hb, a.lo)));
    const unsigned long long hd = ((unsigned long long)(hb & 0x80000000u) << 32)
                                | ((unsigned long long)(e8 + 896) << 52)
                                | ((unsigned long long)(hb & 0x7fffffu) << 29);
    const long long db = (long long)hd + ((hb & 0x80000000u) ? -(long long)n : (long long)n);
    return e8 >= 52 ? __longlong_as_double(db) : 0.0;
}

DSI cds cds_load(const double2* p, int i) {
    const double2 v = __ldg(&p[i]);
    return {ds_from_d(v.x), ds_from_d(v.y)};
}
