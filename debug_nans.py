import jax
import jax.numpy as jnp
import numpy as np
import os
from jax_geometry import load_geometry
from jax_integrals import geom_tensors, calculate_phi

# Ensure fp64
jax.config.update("jax_enable_x64", True)

directory = "/restricteddata/ukaea/gyrokinetics/raw/iteration_13"
geom = load_geometry(directory)
gt = geom_tensors(geom)

for k, v in gt.items():
    if jnp.any(jnp.isnan(v)):
        print(f"nan found in {k}")
        # Investigate bessel specifically
        if k == "bessel":
            # Recalculate bessel_arg
            from jax_integrals import j0
            from einops import rearrange
            vthrat = geom["vthrat"]
            if vthrat.ndim > 0: vthrat = vthrat[0]
            vthrat = jnp.reshape(vthrat, (1, 1, 1, 1, 1, 1))
            kxrh = rearrange(geom["kxrh"], "x -> 1 1 1 1 x 1")
            little_g = rearrange(geom["little_g"], "s three -> three 1 1 1 s 1 1")
            krho = rearrange(geom["krho"], "y -> 1 1 1 1 1 y")
            bn = rearrange(geom["bn"], "s -> 1 1 1 s 1 1")
            mugr = rearrange(geom["mugr"], "mu -> 1 1 mu 1 1 1")
            mas = jnp.reshape(geom["mas"][0], (1, 1, 1, 1, 1, 1))
            sz = jnp.reshape(geom["signz"][0], (1, 1, 1, 1, 1, 1))

            krloc_sq = (
                krho ** 2 * little_g[0]
                + 2 * krho * kxrh * little_g[1]
                + kxrh**2 * little_g[2]
            )
            krloc_sq = jnp.where(krloc_sq < 0, 0.0, krloc_sq)
            krloc = jnp.sqrt(krloc_sq)
            mugr_bn = 2.0 * mugr / bn
            mugr_bn = jnp.where(mugr_bn < 0, 0.0, mugr_bn)
            bessel_arg = jnp.sqrt(mugr_bn) / sz
            bessel_arg = mas * vthrat * krloc * bessel_arg
            
            print(f"bessel_arg has nans: {jnp.any(jnp.isnan(bessel_arg))}")
            if jnp.any(jnp.isnan(bessel_arg)):
                # Find where
                nan_indices = jnp.where(jnp.isnan(bessel_arg))
                print(f"Example nan index in bessel_arg: {nan_indices}")
                # check components at nan_indices
                idx = (nan_indices[0][0], nan_indices[1][0], nan_indices[2][0], nan_indices[3][0], nan_indices[4][0], nan_indices[5][0])
                print(f"krloc_sq at idx: {krloc_sq[idx]}")
                print(f"mugr_bn at idx: {mugr_bn[idx]}")
                print(f"sz at idx: {sz[idx]}")
            
            bes = j0(bessel_arg)
            print(f"j0(bessel_arg) has nans: {jnp.any(jnp.isnan(bes))}")
            if jnp.any(jnp.isnan(bes)) and not jnp.any(jnp.isnan(bessel_arg)):
                 print("nans introduced by j0 itself!")

