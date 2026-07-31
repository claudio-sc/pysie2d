"""Shared configuration for the physics validation suite.

All physics tests use a circle (the only cross-section with an analytic Mie
solution), refractive-index contrast n_core = 1.5, n_clad = 1.0, radius 200 nm.
"""

import numpy as np
import pytest

RAD = 200.0  # cylinder radius (nm)
N_CORE = 1.5  # particle refractive index
N_CLAD = 1.0  # background refractive index

# QNM work uses a higher contrast than the driven-solver fixture: at
# n_core = 1.5 the modes are Q ≈ 2.4 and overlapping, which cannot anchor a
# mode-resolution test.  At n_core = 3.0 the circle carries isolated modes
# spanning Q = 0.96 … 2289.
QNM_N_CORE = 3.0

# Polarisation code → Mie coefficient tag.  pol=2 (TE, E_y) ↔ b_n ↔ *_TE;
# pol=1 (TM, H_y) ↔ a_n ↔ *_TM.
POL_TAG = {2: "TE", 1: "TM"}


def size_parameter(wavelength: float) -> float:
    """Size parameter x = k·a = 2π·n_clad·a/λ for the shared circle."""
    return 2.0 * np.pi * N_CLAD * RAD / wavelength


@pytest.fixture(scope="module")
def circle():
    """Factory building a circular Geometry at a chosen resolution."""
    from pysie2d import Geometry

    def _make(n_pts: int = 300):
        return Geometry.gielis(rad=RAD, n_pts=n_pts, m=0)

    return _make
