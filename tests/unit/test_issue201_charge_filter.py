from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np

from scripts.issue201_charge_filter import make_charge_correction


def test_neutral_damping_across_linked_harmonics():
    # Three seven-cell turns form one open chain. Carry two ky values to check
    # that the nonzonal correction cannot enter the zonal field solve.
    ns, nkx, nky = 7, 3, 2
    z = np.arange(ns)[:, None, None] + ns * np.arange(nkx)[None, :, None]
    z = np.broadcast_to(z, (ns, nkx, nky))
    shifted = z + np.arange(-4, 5)[:, None, None, None]
    valid = (shifted >= 0) & (shifted < ns * nkx)
    classes = np.clip(np.minimum(z, 2) + np.maximum(z - (ns * nkx - 3), 0), 0, 4)
    charge_sign = np.array([1., -1.])[:, None, None, None, None, None]
    weight = charge_sign * (1 + z / 100)
    speed = np.array([1., 5.])[:, None, None, None, None, None]
    pre = {
        "par_stencil_class": classes,
        "valid_shift": valid,
        "s_shift": shifted % ns,
        "kx_shift": np.clip(shifted // ns, 0, nkx - 1),
        "phi_weight": jnp.asarray(weight),
        "fmaxwl": jnp.ones_like(weight),
        "bessel": jnp.ones_like(weight),
        "abs_dum2_par": jnp.asarray(speed),
        "sgr_dist": 1.,
    }
    params = SimpleNamespace(
        adiabatic_electrons=False, nlapar=True, nlbpar=False, disp_par_conserve=0,
        signz=(1., -1.), tmp=(1., 1.), disp_par=1.,
    )
    correct = make_charge_correction({"krho": np.array([0., .1])}, params, pre)
    signal = np.cos(2.1 * z)
    neutral = np.broadcast_to(signal, weight.shape).astype(np.complex128)
    update = np.asarray(correct(jnp.asarray(neutral)))
    np.testing.assert_array_equal(update[..., 0], 0)
    rho = (weight * update).sum(axis=(0, 1, 2))
    line = signal[..., 1].T.ravel()
    stencil = -np.diff(line, n=4) / 12
    # Electron damping is five times faster. Its four excess units of charge
    # transfer must be canceled at every center, including both turn seams.
    expected = 4 * (1 + np.arange(2, len(line) - 2) / 100) * stencil
    np.testing.assert_allclose(rho[..., 1].T.ravel()[2:-2], expected)
    np.testing.assert_array_equal(rho[..., 1].T.ravel()[[0, 1, -2, -1]], 0)
    np.testing.assert_allclose(correct(jnp.asarray(charge_sign * neutral)), 0, atol=1e-14)
