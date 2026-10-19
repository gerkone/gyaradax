"""Measured matmul throughput by precision on this GPU.
Ozaki stage 2 needs to know whether FP8 or INT8 is the fast emulation substrate."""
import statistics, time
import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

print(f"  device: {jax.devices()[0].device_kind}")
N = 8192
def bench(a, b, pref=None, reps=10):
    f = jax.jit(lambda x, y: jnp.matmul(x, y, preferred_element_type=pref))
    jax.block_until_ready(f(a, b))
    r = []
    for _ in range(5):
        t0 = time.perf_counter()
        for _ in range(reps): o = f(a, b)
        jax.block_until_ready(o); r.append((time.perf_counter()-t0)/reps)
    return 2 * N**3 / statistics.median(r)

rng = np.random.default_rng(0)
x = rng.standard_normal((N, N)).astype(np.float32); y = rng.standard_normal((N, N)).astype(np.float32)
rows = []
for name, dt, pref in (("FP64", jnp.float64, None), ("FP32", jnp.float32, None),
                       ("TF32", jnp.float32, "tf32"), ("BF16", jnp.bfloat16, jnp.float32),
                       ("FP8 e4m3", jnp.float8_e4m3fn, jnp.float32),
                       ("INT8", jnp.int8, jnp.int32)):
    try:
        if name == "TF32":
            with jax.default_matmul_precision("tensorfloat32"):
                a = jnp.asarray(x); b = jnp.asarray(y)
                t = bench(a, b)
        elif name == "INT8":
            a = jnp.asarray(np.clip(x*30, -127, 127).astype(np.int8))
            b = jnp.asarray(np.clip(y*30, -127, 127).astype(np.int8))
            t = bench(a, b, pref)
        elif name in ("FP64", "FP32"):
            with jax.default_matmul_precision("highest"):
                a = jnp.asarray(x).astype(dt); b = jnp.asarray(y).astype(dt)
                assert a.dtype == dt, f"{name}: got {a.dtype} -- x64 not enabled"
                t = bench(a, b, pref)
        else:
            a = jnp.asarray(x).astype(dt); b = jnp.asarray(y).astype(dt)
            t = bench(a, b, pref)
        rows.append((name, t))
        print(f"  {name:9s} {t/1e12:9.1f} T{'OPS' if name=='INT8' else 'FLOP/s'}")
    except Exception as e:
        print(f"  {name:9s} FAILED: {type(e).__name__}: {str(e)[:100]}")
d = dict(rows)
if "FP8 e4m3" in d and "INT8" in d:
    print(f"\n  FP8 : INT8 measured ratio = {d['FP8 e4m3']/d['INT8']:.1f} : 1")
if "FP8 e4m3" in d and "FP64" in d:
    print(f"  FP8 : FP64 measured ratio = {d['FP8 e4m3']/d['FP64']:.0f} : 1")
