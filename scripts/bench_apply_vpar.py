"""Microbenchmark for _apply_vpar variants."""
import time
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np

DEVICE = 1
jax.config.update("jax_default_device", jax.devices()[DEVICE])

NV, NMU, NS, NKX, NKY = 32, 8, 16, 85, 32
RNG = np.random.default_rng(0)

COEFFS_D1 = jnp.array([ 1/12, -2/3,  0.0,  2/3, -1/12], dtype=jnp.float64)
COEFFS_D4 = jnp.array([ 1.0,  -4.0,  6.0, -4.0,  1.0 ], dtype=jnp.float64)


def make_field():
    return jnp.array(
        RNG.standard_normal((NV, NMU, NS, NKX, NKY))
        + 1j * RNG.standard_normal((NV, NMU, NS, NKX, NKY))
    )


# ── v0: baseline (jnp.take + clip + valid mask) ────────────────────────────
def v0(field, coeffs):
    nv = field.shape[0]
    out = jnp.zeros_like(field)
    for c, s in zip(coeffs, (-2, -1, 0, 1, 2)):
        idx = jnp.clip(jnp.arange(nv, dtype=jnp.int32) + s, 0, nv - 1)
        valid = jnp.logical_and(jnp.arange(nv) + s >= 0, jnp.arange(nv) + s < nv)
        shifted = jnp.take(field, idx, axis=0)
        out = out + c * jnp.where(valid[:, None, None, None, None], shifted, 0.0)
    return out


# ── v1: pad + slice (contiguous reads, no Gather HLO) ──────────────────────
def v1(field, coeffs):
    nv = field.shape[0]
    padded = jnp.pad(field, ((2, 2), (0, 0), (0, 0), (0, 0), (0, 0)))
    return (
        coeffs[0] * padded[0:nv]
        + coeffs[1] * padded[1:nv + 1]
        + coeffs[2] * padded[2:nv + 2]
        + coeffs[3] * padded[3:nv + 3]
        + coeffs[4] * padded[4:nv + 4]
    )


# ── v2: conv_general_dilated (real + imag separately) ──────────────────────
def v2(field, coeffs):
    nv = field.shape[0]
    batch = int(np.prod(field.shape[1:]))
    # Layout: (N=batch, H=nv, C=1) for NHC; kernel (O=1, I=1, H=5) for OIH
    f_flat = field.reshape(nv, batch).T.reshape(batch, nv, 1)
    kernel = coeffs[::-1].reshape(1, 1, 5)  # (out_ch, in_ch, width)

    def conv1d_real(x):
        return jax.lax.conv_general_dilated(
            x, kernel.astype(x.dtype),
            window_strides=(1,), padding=[(2, 2)],
            dimension_numbers=('NHC', 'OIH', 'NHC'),
        )

    r = conv1d_real(f_flat.real) + 1j * conv1d_real(f_flat.imag)
    # result: (batch, nv, 1) → (nv, batch) → original shape
    return r.reshape(batch, nv).T.reshape(field.shape)


# ── v3: lax.scan (for reference — expected to be bad in nested context) ─────
def v3(field, coeffs):
    nv = field.shape[0]
    shifts = jnp.array([-2, -1, 0, 1, 2], dtype=jnp.int32)

    def body(out, args):
        c, s = args
        idx = jnp.clip(jnp.arange(nv, dtype=jnp.int32) + s, 0, nv - 1)
        valid = jnp.logical_and(jnp.arange(nv) + s >= 0, jnp.arange(nv) + s < nv)
        shifted = jnp.take(field, idx, axis=0)
        return out + c * jnp.where(valid[:, None, None, None, None], shifted, 0.0), None

    out, _ = jax.lax.scan(body, jnp.zeros_like(field), (coeffs, shifts))
    return out


def benchmark(name, fn_jit, field, coeffs, n_warmup=5, n_trials=20):
    for _ in range(n_warmup):
        fn_jit(field, coeffs).block_until_ready()
    times = []
    for _ in range(n_trials):
        t0 = time.perf_counter()
        fn_jit(field, coeffs).block_until_ready()
        times.append(time.perf_counter() - t0)
    arr = np.array(times) * 1e3
    print(f"  {name:45s}: {arr.mean():.3f} ± {arr.std():.3f} ms")
    return arr.mean()


def main():
    field = make_field()
    variants = [
        ("v0 baseline (take + clip + valid)", v0),
        ("v1 pad + slice (no Gather)",        v1),
        ("v2 conv_general_dilated",           v2),
        ("v3 lax.scan",                       v3),
    ]

    print(f"\nDevice: {jax.devices()[DEVICE]}")
    print(f"Shape: field={field.shape}\n")

    ref = jax.jit(v0)(field, COEFFS_D1)
    for name, fn in variants[1:]:
        out = jax.jit(fn)(field, COEFFS_D1)
        err = jnp.linalg.norm(out - ref) / jnp.linalg.norm(ref)
        print(f"  correctness {name:38s}: {'OK' if err < 1e-10 else f'FAIL {err:.2e}'}")

    print()
    results = {}
    for name, fn in variants:
        results[name] = benchmark(name, jax.jit(fn), field, COEFFS_D1)

    print()
    base = results[variants[0][0]]
    for name, t in results.items():
        print(f"  {name:45s}: {base/t:.3f}x vs baseline")


if __name__ == "__main__":
    main()
