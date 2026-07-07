"""Excitation right-hand sides for the BIE.

Every source fills the BIE right-hand-side vector ``ei`` of shape (2*nn,).
Following the plane-wave convention, only the φ half ``ei[:nn]`` is populated;
the χ half ``ei[nn:]`` stays zero.
"""

import numpy as np

PI = np.pi


# ---------------------------------------------------------------------------
# Incident plane-wave RHS  (subroutine ein)
# ---------------------------------------------------------------------------


def plane_wave_rhs(
    nn: int,
    titad: float,
    lambd: float,
    f: np.ndarray,
    g: np.ndarray,
) -> np.ndarray:
    """Right-hand-side vector for an s-polarised plane wave.

    Fills only the φ half of the solution vector; the χ half stays zero.

    Args:
        nn: Number of boundary points.
        titad: Incidence angle (degrees).
        lambd: Wavelength (nm).
        f: (nn,) boundary x coordinates (nm).
        g: (nn,) boundary z coordinates (nm).

    Returns:
        ei: complex (2*nn,) with ei[:nn] = exp(i k (f sinθ − g cosθ)) and
            ei[nn:] = 0.
    """
    theta = np.deg2rad(titad)
    wnum = 2.0 * PI / lambd
    fb = f * np.sin(theta) - g * np.cos(theta)
    ei = np.zeros(2 * nn, dtype=complex)
    ei[:nn] = np.exp(1j * wnum * fb)
    return ei
