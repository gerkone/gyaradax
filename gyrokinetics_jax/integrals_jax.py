import jax
import jax.numpy as jnp
from typing import Dict, Optional, Tuple


def bessel_j0_approx(x):
    x = jnp.abs(x)

    def small_x(x):
        y = (x / 3.0) ** 2
        return 1.0 + y * (
            -2.2499997
            + y
            * (
                1.2656208
                + y * (-0.3163866 + y * (0.0444479 + y * (-0.0039444 + y * 0.0002100)))
            )
        )

    def large_x(x):
        y = 3.0 / x
        f0 = 0.79788456 + y * (
            -0.00000077
            + y
            * (
                -0.00552740
                + y
                * (-0.00009512 + y * (0.00137237 + y * (-0.00072805 + y * 0.00014476)))
            )
        )
        theta0 = (
            x
            - 0.78539816
            + y
            * (
                -0.04166397
                + y
                * (
                    -0.00003954
                    + y
                    * (
                        0.00262573
                        + y * (-0.00054125 + y * (-0.00029333 + y * 0.00013558))
                    )
                )
            )
        )
        return f0 * jnp.cos(theta0) / jnp.sqrt(x)

    return jnp.where(x < 3.0, small_x(x), large_x(x))


def get_bessel_gyro(geom: Dict[str, jnp.ndarray]):
    krho = geom["krho"].reshape(1, 1, 1, 1, -1)
    kxrh = geom["kxrh"].reshape(1, 1, 1, -1, 1)
    g0 = geom["little_g"][:, 0].reshape(1, 1, -1, 1, 1)
    g1 = geom["little_g"][:, 1].reshape(1, 1, -1, 1, 1)
    g2 = geom["little_g"][:, 2].reshape(1, 1, -1, 1, 1)
    krloc = jnp.sqrt(krho**2 * g0 + 2 * krho * kxrh * g1 + kxrh**2 * g2)
    mugr = geom["mugr"].reshape(1, -1, 1, 1, 1)
    bn = geom["bn"].reshape(1, 1, -1, 1, 1)
    bessel_arg = jnp.sqrt(2.0 * mugr / bn) / geom["signz"]
    bessel_arg = geom["mas"] * geom["vthrat"] * krloc * bessel_arg
    return bessel_j0_approx(bessel_arg)


def get_gamma_polar(geom: Dict[str, jnp.ndarray]):
    krho = geom["krho"].reshape(1, 1, 1, 1, -1)
    kxrh = geom["kxrh"].reshape(1, 1, 1, -1, 1)
    g0 = geom["little_g"][:, 0].reshape(1, 1, -1, 1, 1)
    g1 = geom["little_g"][:, 1].reshape(1, 1, -1, 1, 1)
    g2 = geom["little_g"][:, 2].reshape(1, 1, -1, 1, 1)
    krloc = jnp.sqrt(krho**2 * g0 + 2 * krho * kxrh * g1 + kxrh**2 * g2)
    bn = geom["bn"].reshape(1, 1, -1, 1, 1)
    gamma_arg = geom["mas"] * geom["vthrat"] * krloc
    gamma_arg = 0.5 * (gamma_arg / (geom["signz"] * bn)) ** 2
    return jax.lax.bessel_i0e(gamma_arg)


def prepare_poisson_operators(geom: Dict[str, jnp.ndarray]):
    """Pre-calculate constant parts of the Poisson solver."""
    signz = geom["signz"]
    tmp = geom["tmp"]
    bn = geom["bn"].reshape(1, 1, -1, 1, 1)
    de = geom["de"]
    intmu = geom["intmu"].reshape(1, -1, 1, 1, 1)
    intvp = geom["intvp"].reshape(-1, 1, 1, 1, 1)
    ints = geom["ints"].reshape(1, 1, -1, 1, 1)
    bessel = get_bessel_gyro(geom)
    gamma = get_gamma_polar(geom)

    cfen = 0.0
    poisson_int = signz * de * intmu * intvp * bessel * bn
    poisson_int = jnp.where(jnp.abs(intvp) < 1e-9, 0.0, poisson_int)

    diagz = signz * (gamma - 1.0) * jnp.exp(-cfen) / tmp
    matz = -ints / (signz * de * (diagz - jnp.exp(-cfen) / tmp))
    matz = matz.at[..., 1:].set(0.0)

    maty_sum = (-matz * jnp.exp(-cfen)).sum(axis=2, keepdims=True)
    maty = tmp / (de * jnp.exp(-cfen)) + maty_sum / jnp.exp(-cfen)
    maty = maty.at[..., 0, :].set(1.0)
    maty = jnp.where(maty == 0, 1.0, maty)
    maty = 1.0 / maty
    maty = maty.at[..., 1:].set(0.0)

    poisson_diag = jnp.exp(-cfen) * (signz**2) * de * (gamma - 1.0) / tmp
    poisson_diag = poisson_diag.at[..., 0, 0].set(0.0)
    poisson_diag = poisson_diag - signz * jnp.exp(-cfen) * de / tmp
    poisson_diag = -1.0 / poisson_diag

    return {
        "poisson_int": poisson_int,
        "matz": matz,
        "maty": maty,
        "poisson_diag": poisson_diag,
        "bessel": bessel,
    }


def solve_phi_fast(operators: Dict[str, jnp.ndarray], df: jnp.ndarray):
    """Optimized Poisson solver using pre-calculated operators."""
    poisson_int = operators["poisson_int"]
    matz = operators["matz"]
    maty = operators["maty"]
    poisson_diag = operators["poisson_diag"]

    phi = (poisson_int * df).sum(axis=(0, 1), keepdims=True)
    bufphi = (matz * phi).sum(axis=(2, 4), keepdims=True)
    phi = (phi + maty * bufphi) * poisson_diag
    return phi.squeeze()


def solve_phi(geom: Dict[str, jnp.ndarray], df: jnp.ndarray):
    """Legacy wrapper for prepare + solve."""
    ops = prepare_poisson_operators(geom)
    return solve_phi_fast(ops, df)


def get_fluxes_fast(
    geom: Dict[str, jnp.ndarray],
    df: jnp.ndarray,
    phi: jnp.ndarray,
    bessel: jnp.ndarray,
    ints_sum: float,
):
    """Optimized flux calculation using pre-calculated Bessel kernel."""
    ns = len(geom["ints"])
    nx, ny = df.shape[3], df.shape[4]

    bn = geom["bn"].reshape(1, 1, ns, 1, 1)
    bt_frac = geom["bt_frac"].reshape(1, 1, ns, 1, 1)
    rfun = geom["rfun"].reshape(1, 1, ns, 1, 1)
    parseval = geom["parseval"].reshape(1, 1, 1, 1, ny)
    ints = geom["ints"].reshape(1, 1, ns, 1, 1)
    efun = geom["efun"].reshape(1, 1, ns, 1, 1)
    krho = geom["krho"].reshape(1, 1, 1, 1, ny)
    d2X = geom["d2X"]
    intmu = geom["intmu"].reshape(1, -1, 1, 1, 1)
    intvp = geom["intvp"].reshape(-1, 1, 1, 1, 1)
    vpgr = geom["vpgr"].reshape(-1, 1, 1, 1, 1)
    mugr = geom["mugr"].reshape(1, -1, 1, 1, 1)
    signB = geom["signB"]

    phi_gyro = bessel * phi.reshape(1, 1, ns, nx, ny)
    dum = parseval * (efun * krho) * df
    dum1 = dum * jnp.conj(phi_gyro)
    dum2 = dum1 * bn

    d3v = ints * d2X * intmu * bn * intvp

    pflux = d3v * jnp.imag(dum1)
    eflux = d3v * (vpgr**2 * jnp.imag(dum1) + 2.0 * mugr * jnp.imag(dum2))
    vflux = d3v * (jnp.imag(dum1) * vpgr * rfun * bt_frac * signB)

    return pflux.sum() / ints_sum, eflux.sum() / ints_sum, vflux.sum() / ints_sum


def get_fluxes(geom: Dict[str, jnp.ndarray], df: jnp.ndarray, phi: jnp.ndarray):
    """Legacy wrapper."""
    bessel = get_bessel_gyro(geom)
    ints_sum = geom["ints"].sum()
    return get_fluxes_fast(geom, df, phi, bessel, float(ints_sum))
