"""s-Nyquist banding stop: ends a run whose zonal potential piles up at the parallel Nyquist mode."""

from __future__ import annotations

import logging
from typing import Any, Optional

import numpy as np

log = logging.getLogger(__name__)

# exit code of a run stopped on banding
EXIT_BANDED = 6
THRESHOLD = 0.3
WINDOW = 3


def nyquist_shares(phi_real: Any) -> tuple[float, float]:
    """Share of the s-Nyquist power in the zonal (y-mean) part of a real phi (x, s, y) and in the rest.

    Each share is the power at FFT index ns // 2 along s, summed over x (and y), over the
    total power of that part.
    """
    phi = np.asarray(phi_real, np.float64)
    zonal = phi.mean(axis=2, keepdims=True)

    def share(f: np.ndarray) -> float:
        p = np.abs(np.fft.fft(f, axis=1)) ** 2
        total = float(p.sum())
        return float(p[:, f.shape[1] // 2].sum()) / total if total > 0 else 0.0

    return share(zonal), share(phi - zonal)


class BandingStop:
    """Stops a run once the zonal s-Nyquist share of phi stays above ``threshold`` for ``window`` frames."""

    def __init__(self, threshold: float = THRESHOLD, window: int = WINDOW):
        self.threshold = float(threshold)
        self.window = int(window)
        self.shares: list[float] = []
        self.message = ""
        self.exit_code = EXIT_BANDED

    def observe(self, state: Any, fluxes: Any, phi: Any) -> Optional[str]:
        from gyaradax.io.dataset import phi_to_real

        zonal, _ = nyquist_shares(phi_to_real(phi))
        self.shares = (self.shares + [zonal])[-self.window:]
        if len(self.shares) < self.window or min(self.shares) <= self.threshold:
            return None
        t = float(np.ravel(np.asarray(state.time))[0])
        self.message = (
            f"s-Nyquist banding: zonal phi share {zonal:.2f} above {self.threshold:g} "
            f"for {self.window} frames at t={t:.2f}"
        )
        log.error(self.message)
        return "banded"
