import jax
import jax.numpy as jnp
from typing import Dict, Tuple
from einops import rearrange
from jax_integrals import get_integrals

# Ensure fp64
jax.config.update("jax_enable_x64", True)

def compute_f_maxwellian(geom: Dict[str, jnp.ndarray]) -> jnp.ndarray:
    """Computes the equilibrium Maxwellian distribution function."""
    # fm = (de / (sqrt(tmp*pi)**3)) * exp(-(vpar**2 + 2*bn*mu)/tmp)
    # shapes: vpar(32), mu(8), s(16), x(85), ky(32)
    
    vpgr = rearrange(geom["vpgr"], "par -> par 1 1 1 1")
    mugr = rearrange(geom["mugr"], "mu -> 1 mu 1 1 1")
    bn = rearrange(geom["bn"], "s -> 1 1 s 1 1")
    tmp = geom["tmp"][0] # simplified for first species
    de = geom["de"][0]
    
    exponent = -(vpgr**2 + 2.0 * bn * mugr) / tmp
    norm = de / (jnp.sqrt(tmp * jnp.pi)**3)
    
    return norm * jnp.exp(exponent)

def parallel_streaming(df: jnp.ndarray, geom: Dict[str, jnp.ndarray]) -> jnp.ndarray:
    """Computes -v_par * grad_par * f."""
    vpgr = rearrange(geom["vpgr"], "par -> par 1 1 1 1")
    vthrat = geom["vthrat"][0]
    
    # Gradient in s-direction (centered difference)
    ds = rearrange(geom["ints"], "s -> 1 1 s 1 1")
    df_ds = (jnp.roll(df, -1, axis=2) - jnp.roll(df, 1, axis=2)) / (2.0 * ds)
    
    # Boundary conditions: iteration_13 uses 'open'
    # Simplified: zero-out gradient at boundaries for now
    df_ds = df_ds.at[:, :, 0, :, :].set(0.0)
    df_ds = df_ds.at[:, :, -1, :, :].set(0.0)
    
    return -vthrat * vpgr * df_ds

def magnetic_drift(df: jnp.ndarray, geom: Dict[str, jnp.ndarray]) -> jnp.ndarray:
    """Computes -v_D * grad * f = -1j * (vDx*kx + vDy*ky) * f."""
    kx = rearrange(geom["kxrh"], "x -> 1 1 1 x 1")
    ky = rearrange(geom["krho"], "y -> 1 1 1 1 y")
    
    vpgr = rearrange(geom["vpgr"], "par -> par 1 1 1 1")
    mugr = rearrange(geom["mugr"], "mu -> 1 mu 1 1 1")
    bn = rearrange(geom["bn"], "s -> 1 1 s 1 1")
    tmp = geom["tmp"][0]
    signz = geom["signz"][0]
    
    # Energy term ED = vpar^2 + mu*B
    ED = vpgr**2 + mugr * bn
    
    # Drift components from dfun (nx, ns, 3)
    # dfun[:,:,0] is D_eps (x), dfun[:,:,1] is D_zeta (y)
    dfun = geom["dfun"]
    vDx_geom = rearrange(dfun[:, 0], "s -> 1 1 s 1 1") # simplified if nx=1 or using first index
    vDy_geom = rearrange(dfun[:, 1], "s -> 1 1 s 1 1")
    
    vDx = (tmp / signz) * ED * vDx_geom
    vDy = (tmp / signz) * ED * vDy_geom
    
    return -1j * (vDx * kx + vDy * ky) * df

def electric_drive(phi: jnp.ndarray, geom: Dict[str, jnp.ndarray], fm: jnp.ndarray) -> jnp.ndarray:
    """Computes -v_E * grad * f_m."""
    ky = rearrange(geom["krho"], "y -> 1 1 1 1 y")
    
    # Drive gradients: rlt (temperature), rln (density)
    rlt = geom["rlt"][0]
    rln = geom["rln"][0]
    
    vpgr = rearrange(geom["vpgr"], "par -> par 1 1 1 1")
    mugr = rearrange(geom["mugr"], "mu -> 1 mu 1 1 1")
    bn = rearrange(geom["bn"], "s -> 1 1 s 1 1")
    tmp = geom["tmp"][0]
    
    # Gradient of Maxwellian: fm * [rln + (v^2/T - 1.5)*rlt]
    v_sq_norm = (vpgr**2 + 2.0 * bn * mugr) / tmp
    drive_factor = rln + (v_sq_norm - 1.5) * rlt
    
    phi_expanded = rearrange(phi, "s x y -> 1 1 s x y")
    
    return -1j * ky * phi_expanded * fm * drive_factor

def gksolve_rhs(df: jnp.ndarray, geom: Dict[str, jnp.ndarray], fm: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray, Tuple]:
    """Total RHS of the gyrokinetic equation."""
    phi, fluxes = get_integrals(df, geom)
    
    rhs = parallel_streaming(df, geom)
    rhs += magnetic_drift(df, geom)
    rhs += electric_drive(phi, geom, fm)
    
    return rhs, phi, fluxes

def gksolve(df: jnp.ndarray, geom: Dict[str, jnp.ndarray], dt: float) -> Tuple[jnp.ndarray, Tuple[jnp.ndarray, Tuple]]:
    """RK4 step."""
    fm = compute_f_maxwellian(geom)
    
    def f(y):
        rhs, phi, fluxes = gksolve_rhs(y, geom, fm)
        return rhs, (phi, fluxes)
    
    k1, (phi, fluxes) = f(df)
    k2, _ = f(df + 0.5 * dt * k1)
    k3, _ = f(df + 0.5 * dt * k2)
    k4, _ = f(df + dt * k3)
    
    next_df = df + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
    
    return next_df, (phi, fluxes)
