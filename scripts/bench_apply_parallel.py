"""Microbenchmark for _apply_parallel variants — run before committing to solver.py."""
import time
import jax
import jax.numpy as jnp
import numpy as np

DEVICE = 1
jax.config.update("jax_default_device", jax.devices()[DEVICE])

# Match iteration_13 adiabatic shapes (single species slice)
NV, NMU, NS, NKX, NKY = 32, 8, 16, 85, 32
N_STENCIL = 9
RNG = np.random.default_rng(0)


def make_inputs():
    field = jnp.array(RNG.standard_normal((NV, NMU, NS, NKX, NKY))
                      + 1j * RNG.standard_normal((NV, NMU, NS, NKX, NKY)))
    coeffs = jnp.array(RNG.standard_normal((N_STENCIL, NV, NMU, NS, NKX, NKY))
                       + 1j * RNG.standard_normal((N_STENCIL, NV, NMU, NS, NKX, NKY)))
    s_shift = jnp.array(RNG.integers(0, NS, size=(N_STENCIL, NS, NKX, NKY)), dtype=jnp.int32)
    kx_shift = jnp.array(RNG.integers(0, NKX, size=(N_STENCIL, NS, NKX, NKY)), dtype=jnp.int32)
    valid_shift = jnp.ones((N_STENCIL, NS, NKX, NKY), dtype=bool)
    return field, coeffs, s_shift, kx_shift, valid_shift


# ── Variant 0: baseline Python loop ─────────────────────────────────────────
def make_baseline(s_shift, kx_shift, valid_shift):
    def fn(field, coeffs):
        out = jnp.zeros_like(field)
        nky = field.shape[-1]
        ky_idx = jnp.reshape(jnp.arange(nky, dtype=jnp.int32), (1, 1, -1))
        for i in range(N_STENCIL):
            shifted = jnp.where(valid_shift[i][None, None], field[:, :, s_shift[i], kx_shift[i], ky_idx], 0.0)
            out = out + coeffs[i] * shifted
        return out
    return fn


# ── Variant 1: batch gather + moveaxis (first attempt, known regression) ────
def make_v1(s_shift, kx_shift, valid_shift):
    def fn(field, coeffs):
        nky = field.shape[-1]
        ky_idx = jnp.arange(nky, dtype=jnp.int32)
        gathered = field[:, :, s_shift, kx_shift, ky_idx]            # (nv,nmu,9,ns,nkx,nky)
        shifted_stack = jnp.moveaxis(gathered, 2, 0)                  # (9,nv,nmu,ns,nkx,nky)
        shifted_stack = jnp.where(valid_shift[:, None, None], shifted_stack, 0.0)
        return jnp.sum(coeffs * shifted_stack, axis=0)
    return fn


# ── Variant 2: vmap over 9 stencil points (no moveaxis) ─────────────────────
def make_v2(s_shift, kx_shift, valid_shift):
    def fn(field, coeffs):
        nky = field.shape[-1]
        ky_idx = jnp.reshape(jnp.arange(nky, dtype=jnp.int32), (1, 1, -1))

        def gather_one(s_map, kx_map, valid, coeff_i):
            shifted = jnp.where(valid[None, None], field[:, :, s_map, kx_map, ky_idx], 0.0)
            return coeff_i * shifted

        terms = jax.vmap(gather_one)(s_shift, kx_shift, valid_shift, coeffs)  # (9,nv,nmu,ns,nkx,nky)
        return jnp.sum(terms, axis=0)
    return fn


# ── Variant 3: batch gather + einsum (no moveaxis) ──────────────────────────
def make_v3(s_shift, kx_shift, valid_shift):
    def fn(field, coeffs):
        nky = field.shape[-1]
        ky_idx = jnp.arange(nky, dtype=jnp.int32)
        gathered = field[:, :, s_shift, kx_shift, ky_idx]            # (nv,nmu,9,ns,nkx,nky)
        shifted = jnp.where(valid_shift[None, None], gathered, 0.0)  # same shape
        # coeffs: (9,nv,nmu,ns,nkx,nky), shifted: (nv,nmu,9,ns,nkx,nky)
        return jnp.einsum('abicde,iabcde->abcde', shifted, coeffs)
    return fn


# ── Variant 4: lax.scan accumulation ────────────────────────────────────────
def make_v4(s_shift, kx_shift, valid_shift):
    def fn(field, coeffs):
        nky = field.shape[-1]
        ky_idx = jnp.reshape(jnp.arange(nky, dtype=jnp.int32), (1, 1, -1))

        def body(out, args):
            s_map, kx_map, valid, coeff_i = args
            shifted = jnp.where(valid[None, None], field[:, :, s_map, kx_map, ky_idx], 0.0)
            return out + coeff_i * shifted, None

        out, _ = jax.lax.scan(body, jnp.zeros_like(field),
                               (s_shift, kx_shift, valid_shift, coeffs))
        return out
    return fn


def benchmark(name, fn_jit, field, coeffs, n_warmup=5, n_trials=20):
    for _ in range(n_warmup):
        fn_jit(field, coeffs).block_until_ready()
    times = []
    for _ in range(n_trials):
        t0 = time.perf_counter()
        fn_jit(field, coeffs).block_until_ready()
        times.append(time.perf_counter() - t0)
    arr = np.array(times) * 1e3  # ms
    print(f"  {name:45s}: {arr.mean():.3f} ± {arr.std():.3f} ms")
    return arr.mean()


def main():
    field, coeffs, s_shift, kx_shift, valid_shift = make_inputs()

    variants = [
        ("v0 baseline (Python loop)",        make_baseline),
        ("v1 batch-gather + moveaxis",        make_v1),
        ("v2 vmap over 9 stencils",           make_v2),
        ("v3 batch-gather + einsum",          make_v3),
        ("v4 lax.scan accumulation",          make_v4),
    ]

    print(f"\nDevice: {jax.devices()[DEVICE]}")
    print(f"Shape: field={field.shape}, coeffs={coeffs.shape}\n")

    # Check correctness of all variants against baseline
    baseline_fn = jax.jit(make_baseline(s_shift, kx_shift, valid_shift))
    ref = baseline_fn(field, coeffs)
    for name, make_fn in variants[1:]:
        fn = jax.jit(make_fn(s_shift, kx_shift, valid_shift))
        out = fn(field, coeffs)
        err = jnp.linalg.norm(out - ref) / jnp.linalg.norm(ref)
        status = "OK" if err < 1e-10 else f"FAIL rel_l2={err:.2e}"
        print(f"  correctness {name:40s}: {status}")

    print()
    results = {}
    for name, make_fn in variants:
        fn = jax.jit(make_fn(s_shift, kx_shift, valid_shift))
        results[name] = benchmark(name, fn, field, coeffs)

    print()
    base = results[variants[0][0]]
    for name, t in results.items():
        print(f"  {name:45s}: {base/t:.3f}x vs baseline")


if __name__ == "__main__":
    main()
