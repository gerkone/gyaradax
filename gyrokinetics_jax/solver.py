import jax
import jax.numpy as jnp
from typing import Dict, Tuple
from .integrals_jax import (
    solve_phi_fast,
    get_fluxes_fast,
    prepare_poisson_operators,
    get_bessel_gyro,
)


def roll_s(f: jnp.ndarray, shift: int, geom: Dict[str, jnp.ndarray]):
    """
    Roll array f along s-axis (axis 2) with Twist-and-Shift boundary conditions.
    """
    ns = f.shape[2]
    nkx = f.shape[3]
    nky = f.shape[4]
    f_rolled = jnp.roll(f, shift, axis=2)
    ikxspace = geom.get("ikxspace", 0.0)
    ky_idx = jnp.arange(nky)
    base_shift = ikxspace * ky_idx

    def apply_kx_shift(val, turn):
        shift_amt = turn * base_shift
        val_fft = jnp.fft.ifftshift(val, axes=2)
        val_x = jnp.fft.ifft(val_fft, axis=2)
        x_idx = jnp.arange(nkx).reshape(1, 1, nkx, 1)
        phase = jnp.exp(
            1j * 2.0 * jnp.pi * x_idx * shift_amt.reshape(1, 1, 1, nky) / nkx
        )
        val_out = jnp.fft.fftshift(jnp.fft.fft(val_x * phase, axis=2), axes=2)
        mask = (x_idx - shift_amt.reshape(1, 1, 1, nky) >= 0) & (
            x_idx - shift_amt.reshape(1, 1, 1, nky) < nkx
        )
        return val_out * mask

    if shift == 1:
        f_rolled = f_rolled.at[:, :, 0, :, :].set(
            apply_kx_shift(f_rolled[:, :, 0, :, :], -1)
        )
    elif shift == -1:
        f_rolled = f_rolled.at[:, :, -1, :, :].set(
            apply_kx_shift(f_rolled[:, :, -1, :, :], 1)
        )
    elif shift == 2:
        f_rolled = f_rolled.at[:, :, 0, :, :].set(
            apply_kx_shift(f_rolled[:, :, 0, :, :], -1)
        )
        f_rolled = f_rolled.at[:, :, 1, :, :].set(
            apply_kx_shift(f_rolled[:, :, 1, :, :], -1)
        )
    elif shift == -2:
        f_rolled = f_rolled.at[:, :, -1, :, :].set(
            apply_kx_shift(f_rolled[:, :, -1, :, :], 1)
        )
        f_rolled = f_rolled.at[:, :, -2, :, :].set(
            apply_kx_shift(f_rolled[:, :, -2, :, :], 1)
        )
    return f_rolled


def roll_vpar(f: jnp.ndarray, shift: int):
    """Roll array f along vpar-axis (axis 0) with zero padding."""
    f_rolled = jnp.roll(f, shift, axis=0)
    if shift > 0:
        f_rolled = f_rolled.at[:shift].set(0.0)
    elif shift < 0:
        f_rolled = f_rolled.at[shift:].set(0.0)
    return f_rolled


def arakawa_bracket(g: jnp.ndarray, H: jnp.ndarray, geom: Dict[str, jnp.ndarray]):
    """4th order Arakawa bracket {H, g} in (s, vpar) space."""

    def J1(g, H):
        res = roll_s(roll_vpar(g, 1), 1, geom) * (roll_s(H, 1, geom) - roll_vpar(H, 1))
        res += roll_s(g, 1, geom) * (
            roll_s(roll_vpar(H, -1), 1, geom)
            - roll_s(roll_vpar(H, 1), 1, geom)
            - roll_vpar(H, 1)
            + roll_vpar(H, -1)
        )
        res += roll_s(roll_vpar(g, -1), 1, geom) * (
            roll_vpar(H, -1) - roll_s(H, 1, geom)
        )
        res += roll_vpar(g, 1) * (
            roll_s(roll_vpar(H, 1), 1, geom)
            + roll_s(H, 1, geom)
            - roll_s(roll_vpar(H, 1), -1, geom)
            - roll_s(H, -1, geom)
        )
        res += roll_vpar(g, -1) * (
            roll_s(H, -1, geom)
            + roll_s(roll_vpar(H, -1), -1, geom)
            - roll_s(H, 1, geom)
            - roll_s(roll_vpar(H, -1), 1, geom)
        )
        res += roll_s(roll_vpar(g, 1), -1, geom) * (
            roll_vpar(H, 1) - roll_s(H, -1, geom)
        )
        res += roll_s(g, -1, geom) * (
            roll_vpar(H, 1)
            + roll_s(roll_vpar(H, 1), -1, geom)
            - roll_vpar(H, -1)
            - roll_s(roll_vpar(H, -1), -1, geom)
        )
        res += roll_s(roll_vpar(g, -1), -1, geom) * (
            roll_s(H, -1, geom) - roll_vpar(H, -1)
        )
        return res / 12.0

    def J2(g, H):
        res = roll_s(g, 2, geom) * (
            roll_s(roll_vpar(H, -1), 1, geom) - roll_s(roll_vpar(H, 1), 1, geom)
        )
        res += roll_s(roll_vpar(g, 1), 1, geom) * (
            roll_s(H, 2, geom)
            + roll_s(roll_vpar(H, -1), 1, geom)
            - roll_vpar(H, 2)
            - roll_s(roll_vpar(H, 1), -1, geom)
        )
        res += roll_s(roll_vpar(g, -1), 1, geom) * (
            roll_s(roll_vpar(H, -1), -1, geom)
            - roll_s(roll_vpar(H, 1), 1, geom)
            - roll_s(H, 2, geom)
            + roll_vpar(H, -2)
        )
        res += roll_vpar(g, 2) * (
            roll_s(roll_vpar(H, 1), 1, geom) - roll_s(roll_vpar(H, 1), -1, geom)
        )
        res += roll_vpar(g, -2) * (
            roll_s(roll_vpar(H, -1), -1, geom) - roll_s(roll_vpar(H, -1), 1, geom)
        )
        res += roll_s(roll_vpar(g, 1), -1, geom) * (
            roll_s(roll_vpar(H, 1), 1, geom)
            - roll_s(roll_vpar(H, -1), -1, geom)
            - roll_s(H, -2, geom)
            + roll_vpar(H, 2)
        )
        res += roll_s(roll_vpar(g, -1), -1, geom) * (
            roll_s(roll_vpar(H, 1), -1, geom)
            - roll_s(roll_vpar(H, -1), 1, geom)
            + roll_s(H, -2, geom)
            - roll_vpar(H, -2)
        )
        res += roll_s(g, -2, geom) * (
            roll_s(roll_vpar(H, 1), -1, geom) - roll_s(roll_vpar(H, -1), -1, geom)
        )
        return res / 24.0

    return 2.0 * J1(g, H) - J2(g, H)


def compute_rhs_fast(
    df: jnp.ndarray,
    phi: jnp.ndarray,
    geom: Dict[str, jnp.ndarray],
    precomputed: Dict[str, jnp.ndarray],
):
    """Optimized RHS with precomputed kernels."""
    vpar = precomputed["vpar"]
    kx, ky = precomputed["kx"], precomputed["ky"]
    fmaxwl = precomputed["fmaxwl"]
    bessel = precomputed["bessel"]
    HH = precomputed["HH"]
    vd_x, vd_y = precomputed["vd_x"], precomputed["vd_y"]
    efun_nl = precomputed["efun_nl"]
    phi_gyro = bessel * phi.reshape(1, 1, phi.shape[0], phi.shape[1], phi.shape[2])

    # Streaming/Trapping (Arakawa)
    rhs_arakawa = (
        geom["vthrat"]
        * geom["ffun"].reshape(1, 1, -1, 1, 1)
        * arakawa_bracket(df, HH, geom)
        / (geom["ds"].mean() * geom["dvp"])
    )

    # Drifts
    rhs_drift = -1j * (vd_x * kx + vd_y * ky) * df

    # Drive (Term V)
    rhs_drive = (
        1j
        * (geom["efun"].reshape(1, 1, -1, 1, 1) * ky)
        * phi_gyro
        * precomputed["drive_factor"]
        * fmaxwl
    )

    # Potential (VII + VIII)
    phi_ga_ds = (
        -roll_s(phi_gyro, -2, geom)
        + 8 * roll_s(phi_gyro, -1, geom)
        - 8 * roll_s(phi_gyro, 1, geom)
        + roll_s(phi_gyro, 2, geom)
    ) / (12.0 * geom["ds"].mean())
    rhs_term7 = (
        -geom["signz"]
        * geom["ffun"].reshape(1, 1, -1, 1, 1)
        * geom["vthrat"]
        * vpar
        * fmaxwl
        * phi_ga_ds
        / geom["tmp"]
    )
    rhs_term8 = -1j * geom["signz"] * (vd_x * kx + vd_y * ky) * fmaxwl * phi_gyro

    # Nonlinear
    nx_real, ny_real = 135, 96
    pad_x = (nx_real - df.shape[3]) // 2

    def to_real(fc):
        fc_shifted = jnp.fft.ifftshift(fc, axes=3)
        space_x = jnp.fft.ifft(fc_shifted, axis=3, norm="forward")
        return jnp.fft.irfft(space_x, n=ny_real, axis=4, norm="forward")

    def pad_spec(arr):
        pw = [(0, 0)] * arr.ndim
        pw[3], pw[4] = (pad_x, pad_x), (0, ny_real // 2 + 1 - df.shape[4])
        return jnp.pad(arr, pw)

    br = to_real(pad_spec(1j * kx * phi_gyro)) * to_real(
        pad_spec(1j * ky * df)
    ) - to_real(pad_spec(1j * ky * phi_gyro)) * to_real(pad_spec(1j * kx * df))
    rhs_nl = jnp.fft.fftshift(
        jnp.fft.fft(
            jnp.fft.rfft(br, n=ny_real, axis=4, norm="forward"), axis=3, norm="forward"
        ),
        axes=3,
    )
    rhs_nl = efun_nl * rhs_nl[..., pad_x : pad_x + df.shape[3], : df.shape[4]]

    # Hyper/Dissipation
    rhs_hyper = precomputed["hyper_mask"] * df
    rhs_par_diss = (
        -geom["ffun"].reshape(1, 1, -1, 1, 1)
        * geom["vthrat"]
        * precomputed["vpgr_rms"]
        * geom.get("disp_par", 1.0)
        * (
            roll_s(df, -2, geom)
            - 4 * roll_s(df, -1, geom)
            + 6 * df
            - 4 * roll_s(df, 1, geom)
            + roll_s(df, 2, geom)
        )
        / (12.0 * geom["ds"].mean())
    )

    return (
        rhs_arakawa
        + rhs_drift
        + rhs_drive
        + rhs_term7
        + rhs_term8
        + rhs_nl
        + rhs_hyper
        + rhs_par_diss
    )


def gksolve_scan(df: jnp.ndarray, geom: Dict[str, jnp.ndarray], dt: float, steps: int):
    # 1. Pre-calculate constants
    vpar = geom["vpgr"].reshape(-1, 1, 1, 1, 1)
    mu = geom["mugr"].reshape(1, -1, 1, 1, 1)
    bn = geom["bn"].reshape(1, 1, -1, 1, 1)
    kx = geom["kxrh"].reshape(1, 1, 1, -1, 1)
    ky = geom["krho"].reshape(1, 1, 1, 1, -1)
    muB = mu * bn
    fmaxwl = jnp.exp(-(vpar**2 + 2.0 * muB)) / (jnp.pi**1.5)
    drive_factor = (
        geom["rln"] + ((vpar**2 + 2.0 * muB) / geom["tmp"] - 1.5) * geom["rlt"]
    )
    vd_x = geom["dfun_x"].reshape(1, 1, -1, 1, 1) * (vpar**2 + muB)
    vd_y = geom["dfun_y"].reshape(1, 1, -1, 1, 1) * (vpar**2 + muB)
    kxm, kym = jnp.max(jnp.abs(kx)), jnp.max(jnp.abs(ky))
    hyper_mask = -(
        geom.get("disp_x", 0.1) * (kx / kxm) ** 4
        + geom.get("disp_y", 0.1) * (ky / kym) ** 4
    )
    poisson_ops = prepare_poisson_operators(geom)
    precomputed = {
        "vpar": vpar,
        "kx": kx,
        "ky": ky,
        "fmaxwl": fmaxwl,
        "drive_factor": drive_factor,
        "vd_x": vd_x,
        "vd_y": vd_y,
        "hyper_mask": hyper_mask,
        "bessel": poisson_ops["bessel"],
        "HH": 0.5 * vpar**2 + muB,
        "efun_nl": geom["efun"].reshape(1, 1, -1, 1, 1),
        "vpgr_rms": jnp.sqrt(jnp.mean(geom["vpgr"] ** 2)),
        "ints_sum": jnp.sum(geom["ints"]),
    }

    def step_fn(carry, _):
        d_curr = carry
        phi = solve_phi_fast(poisson_ops, d_curr)

        def f(y):
            p_in = solve_phi_fast(poisson_ops, y)
            return compute_rhs_fast(y, p_in, geom, precomputed)

        k1 = f(d_curr)
        k2 = f(d_curr + 0.5 * dt * k1)
        k3 = f(d_curr + 0.5 * dt * k2)
        k4 = f(d_curr + dt * k3)
        d_next = d_curr + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        phi_next = solve_phi_fast(poisson_ops, d_next)
        fluxes = get_fluxes_fast(
            geom, d_next, phi_next, precomputed["bessel"], precomputed["ints_sum"]
        )
        return d_next, jnp.array(fluxes)

    return jax.lax.scan(step_fn, df, None, length=steps)


def init_f(geom: Dict[str, jnp.ndarray], amp_ini: float = 0.0001):
    ns, nkx, nky = len(geom["ints"]), len(geom["kxrh"]), len(geom["krho"])
    s_coord = geom.get("s_grid", jnp.linspace(-0.5, 0.5, ns)).reshape(1, 1, -1, 1, 1)
    f_init = amp_ini * (jnp.cos(2.0 * jnp.pi * s_coord) + 1.0)
    f_init = jnp.broadcast_to(
        f_init, (len(geom["vpgr"]), len(geom["intmu"]), ns, nkx, nky)
    )
    f_init = jnp.where((geom["krho"] == 0).reshape(1, 1, 1, 1, -1), 0.0, f_init)
    return f_init.astype(jnp.complex128)


def compute_rhs(df, phi, geom, return_components=False):
    # Legacy wrapper for tests
    nvpar, nmu, ns, nkx, nky = df.shape
    vpar = geom["vpgr"].reshape(-1, 1, 1, 1, 1)
    mu = geom["mugr"].reshape(1, -1, 1, 1, 1)
    bn = geom["bn"].reshape(1, 1, -1, 1, 1)
    muB = mu * bn
    kx = geom["kxrh"].reshape(1, 1, 1, -1, 1)
    ky = geom["krho"].reshape(1, 1, 1, 1, -1)
    fmaxwl = jnp.exp(-(vpar**2 + 2.0 * muB)) / (jnp.pi**1.5)
    vd_x = geom["dfun_x"].reshape(1, 1, -1, 1, 1) * (vpar**2 + muB)
    vd_y = geom["dfun_y"].reshape(1, 1, -1, 1, 1) * (vpar**2 + muB)
    kxm, kym = jnp.max(jnp.abs(kx)), jnp.max(jnp.abs(ky))
    precomputed = {
        "vpar": vpar,
        "kx": kx,
        "ky": ky,
        "fmaxwl": fmaxwl,
        "drive_factor": geom["rln"]
        + ((vpar**2 + 2.0 * muB) / geom["tmp"] - 1.5) * geom["rlt"],
        "vd_x": vd_x,
        "vd_y": vd_y,
        "hyper_mask": -(
            geom.get("disp_x", 0.1) * (kx / kxm) ** 4
            + geom.get("disp_y", 0.1) * (ky / kym) ** 4
        ),
        "bessel": get_bessel_gyro(geom),
        "HH": 0.5 * vpar**2 + muB,
        "efun_nl": geom["efun"].reshape(1, 1, -1, 1, 1),
        "vpgr_rms": jnp.sqrt(jnp.mean(geom["vpgr"] ** 2)),
    }
    return compute_rhs_fast(df, phi, geom, precomputed)


@jax.jit
def gksolve(df: jnp.ndarray, geom: Dict[str, jnp.ndarray], dt: float):
    # Legacy wrapper
    res, _ = gksolve_scan(df, geom, dt, 1)
    phi = solve_phi(geom, res)
    fluxes = get_fluxes(geom, res, phi)
    return res, phi, fluxes
