"""s-Nyquist banding stop."""

from types import SimpleNamespace

import numpy as np

from gyaradax.diagnostics.banding import EXIT_BANDED, BandingStop, nyquist_shares


def test_nyquist_shares_split_zonal_and_rest():
    s = np.arange(8)
    zonal = np.cos(np.pi * s)[None, :, None] * np.ones((4, 8, 6))
    zs, rest = nyquist_shares(zonal)
    assert np.isclose(zs, 1.0) and rest == 0.0


def test_banding_stop_needs_the_full_window(monkeypatch):
    import gyaradax.io.dataset as ds

    shares = iter([0.5, 0.1, 0.5, 0.5, 0.5])
    monkeypatch.setattr(ds, "phi_to_real", lambda phi: phi)
    monkeypatch.setattr("gyaradax.diagnostics.banding.nyquist_shares", lambda phi: (next(shares), 0.0))
    stop = BandingStop(threshold=0.3, window=3)
    reasons = [stop.observe(SimpleNamespace(time=float(t)), None, None) for t in range(5)]
    assert reasons == [None, None, None, None, "banded"]
    assert "s-Nyquist banding" in stop.message and stop.exit_code == EXIT_BANDED
