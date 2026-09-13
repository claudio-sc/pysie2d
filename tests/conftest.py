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
    """Size parameter x = k_bg·a = 2π·n_clad·a/λ_vac for the shared circle.

    Mirrors the public :func:`pysie2d.size_parameter` but takes the radius
    from the module fixture, so tests can compute x without building a
    Geometry. ``test_conventions.test_size_parameter_matches_fixture`` pins
    the two together.
    """
    return 2.0 * np.pi * N_CLAD * RAD / wavelength


@pytest.fixture(scope="module")
def circle():
    """Factory building a circular Geometry at a chosen resolution.

    ``m=0`` (the default) is the trivial superformula branch: the Gielis
    ``fact_n2``/``fact_n3`` prefactors are exactly zero there, so a shape
    derivative bug confined to that branch's ``tan``/``cot`` machinery would
    validate against Mie anyway. ``m=4, n1=n2=n3=2`` is the same exact circle
    (``Geometry.is_circle`` holds) routed through the non-trivial branch, and
    is what actually exercises it.
    """
    from pysie2d import Geometry

    def _make(
        n_pts: int = 300,
        *,
        m: int = 0,
        n1: float = 2.0,
        n2: float = 2.0,
        n3: float = 2.0,
    ):
        return Geometry.gielis(rad=RAD, n_pts=n_pts, m=m, n1=n1, n2=n2, n3=n3)

    return _make
