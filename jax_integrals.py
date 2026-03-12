import jax
import jax.numpy as jnp
from jax.scipy.special import i0, bessel_jn
from einops import rearrange
from typing import Dict, Optional, Tuple, Sequence

# Ensure fp64
jax.config.update("jax_enable_x64", True)

def j0(x):
    # jax.scipy.special.bessel_jn has a bug/feature where x=0 results in nan
    safe_x = jnp.where(jnp.abs(x) < 1e-10, 1.0, x)
    res = bessel_jn(safe_x, v=0)[0]
    return jnp.where(jnp.abs(x) < 1e-10, 1.0, res)

def geom_tensors(geometry: Dict[str, jnp.ndarray]) -> Dict[str, jnp.ndarray]:
    """Expand geometry constants for broadcasting and compute Bessel terms."""
    geom_ = {}
    geom_["krho"] = rearrange(geometry["krho"], "y -> 1 1 1 1 1 y")
    geom_["ints"] = rearrange(geometry["ints"], "s -> 1 1 1 s 1 1")
    geom_["intmu"] = rearrange(geometry["intmu"], "mu -> 1 1 mu 1 1 1")
    geom_["intvp"] = rearrange(geometry["intvp"], "par -> 1 par 1 1 1 1")
    geom_["vpgr"] = rearrange(geometry["vpgr"], "par -> 1 par 1 1 1 1")
    geom_["mugr"] = rearrange(geometry["mugr"], "mu -> 1 1 mu 1 1 1")
    geom_["bn"] = rearrange(geometry["bn"], "s -> 1 1 1 s 1 1")
    geom_["ffun"] = rearrange(geometry["ffun"], "s -> 1 1 1 s 1 1")
    geom_["efun"] = rearrange(geometry["efun"], "s -> 1 1 1 s 1 1")
    geom_["rfun"] = rearrange(geometry["rfun"], "s -> 1 1 1 s 1 1")
    geom_["bt_frac"] = rearrange(geometry["bt_frac"], "s -> 1 1 1 s 1 1")
    geom_["parseval"] = rearrange(geometry["parseval"], "y -> 1 1 1 1 1 y")
    
    for k in ["mas", "tmp", "de", "d2X", "signz", "signB"]:
        val = geometry[k]
        if val.ndim > 0: val = val[0]
        geom_[k] = jnp.reshape(val, (1, 1, 1, 1, 1, 1))

    vthrat = geometry["vthrat"]
    if vthrat.ndim > 0: vthrat = vthrat[0]
    vthrat = jnp.reshape(vthrat, (1, 1, 1, 1, 1, 1))
    
    kxrh = rearrange(geometry["kxrh"], "x -> 1 1 1 1 x 1")
    little_g = rearrange(geometry["little_g"], "s three -> three 1 1 1 s 1 1")
    
    krloc_sq = (
        geom_["krho"] ** 2 * little_g[0]
        + 2 * geom_["krho"] * kxrh * little_g[1]
        + kxrh**2 * little_g[2]
    )
    krloc_sq = jnp.where(krloc_sq < 0, 0.0, krloc_sq)
    krloc = jnp.sqrt(krloc_sq)
    
    mugr_bn = 2.0 * geom_["mugr"] / geom_["bn"]
    mugr_bn = jnp.where(mugr_bn < 0, 0.0, mugr_bn)
    
    sz = jnp.where(jnp.abs(geom_["signz"]) < 1e-15, 1.0, geom_["signz"])
    bessel_arg = jnp.sqrt(mugr_bn) / sz
    bessel_arg = geom_["mas"] * vthrat * krloc * bessel_arg
    
    bessel_arg = jnp.where(jnp.isnan(bessel_arg), 0.0, bessel_arg)
    geom_["bessel"] = j0(bessel_arg)
    
    gamma_arg = geom_["mas"] * vthrat * krloc
    gamma_arg = 0.5 * (gamma_arg / (sz * geom_["bn"])) ** 2
    gamma_arg = jnp.clip(gamma_arg, 0.0, 500.0)
    geom_["gamma"] = i0(gamma_arg) * jnp.exp(-gamma_arg)
    
    return geom_

def calculate_phi(geom: Dict[str, jnp.ndarray], df: jnp.ndarray) -> jnp.ndarray:
    """Computes electrostatic potential integral from the distribution function."""
    de = geom["de"]
    signz, tmp, bn = geom["signz"], geom["tmp"], geom["bn"]
    ints, intvp, intmu = geom["ints"], geom["intvp"], geom["intmu"]
    bessel, gamma = geom["bessel"], geom["gamma"]
    
    # Matching GKW wrap_field_integrals logic
    poisson_int = signz * de * intmu * intvp * bessel * bn
    poisson_int = jnp.where(jnp.abs(intvp) < 1e-9, 0.0, poisson_int)
    
    cfen = 0.0 
    diagz = signz * (gamma - 1.0) * jnp.exp(-cfen) / tmp
    denom = (diagz - jnp.exp(-cfen) / tmp)
    denom = jnp.where(jnp.abs(denom) < 1e-15, 1.0, denom)
    matz = -ints / (signz * de * denom)
    matz = matz.at[..., 1:].set(0.0)

    phi = poisson_int * df 
    phi = jnp.sum(phi, axis=(1, 2), keepdims=True) 
    
    y_mask = jnp.zeros_like(phi)
    y_mask = y_mask.at[..., 0].set(1.0)
    
    bufphi = matz * phi 
    bufphi = jnp.sum(bufphi, axis=3, keepdims=True) 
    
    maty_sum = jnp.sum(-matz * jnp.exp(-cfen), axis=3, keepdims=True)
    maty = tmp / (de * jnp.exp(-cfen)) + maty_sum / jnp.exp(-cfen)
    
    x_mask = jnp.zeros_like(phi)
    x_mask = x_mask.at[..., 0, :].set(1.0)
    maty_val = jnp.where(x_mask > 0, 1.0 + 0j, maty)
    maty_val = jnp.where(jnp.abs(maty_val) < 1e-15, 1.0, maty_val)
    maty_val = 1.0 / maty_val
    phi = phi + maty_val * bufphi * y_mask
    
    poisson_diag = jnp.exp(-cfen) * (signz**2) * de * (gamma - 1.0) / tmp
    norm_mask = jnp.ones_like(phi)
    norm_mask = norm_mask.at[..., 0, 0].set(0.0)
    
    pdiag = poisson_diag * norm_mask - signz * jnp.exp(-cfen) * de / tmp
    pdiag = jnp.where(jnp.abs(pdiag) < 1e-15, -1.0, pdiag)

    phi = phi * (-1.0 / pdiag)
    
    return jnp.squeeze(phi) 

def calculate_fluxes(geom: Dict[str, jnp.ndarray], df: jnp.ndarray, phi: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Computes particle, heat and momentum fluxes."""
    bn, bt_frac, parseval = geom["bn"], geom["bt_frac"], geom["parseval"]
    rfun, efun, d2X, signB = geom["rfun"], geom["efun"], geom["d2X"], geom["signB"]
    ints, intvp, intmu = geom["ints"], geom["intvp"], geom["intmu"]
    vpgr, mugr, krho = geom["vpgr"], geom["mugr"], geom["krho"]
    bessel = geom["bessel"]
    
    phi_expanded = rearrange(phi, "s x y -> 1 1 1 s x y")
    phi_gyro = bessel * phi_expanded
    
    dum = parseval * ints * (efun * krho) * df
    dum1 = dum * jnp.conj(phi_gyro)
    dum2 = dum1 * bn
    
    d3v = ints * d2X * intmu * bn * intvp
    
    pflux = d3v * jnp.imag(dum1)
    eflux = d3v * (vpgr**2 * jnp.imag(dum1) + 2 * mugr * jnp.imag(dum2))
    vflux = d3v * (jnp.imag(dum1) * vpgr * rfun * bt_frac * signB)
    
    return jnp.sum(pflux), jnp.sum(eflux), jnp.sum(vflux)

def get_integrals(df: jnp.ndarray, geometry: Dict[str, jnp.ndarray]) -> Tuple[jnp.ndarray, Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]]:
    """Main interface for flux integrals."""
    geom = geom_tensors(geometry)
    phi = calculate_phi(geom, df)
    fluxes = calculate_fluxes(geom, df, phi)
    return phi, fluxes
