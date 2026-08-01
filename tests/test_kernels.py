import numpy as np
import pytest

from pysie2d.geometry import Geometry
from pysie2d.kernels import (
    assemble_matrix,
    assemble_matrix_dwn,
    assemble_matrix_reference,
)
from pysie2d.material import Material


@pytest.mark.parametrize("pol", [1, 2])
@pytest.mark.parametrize("wnum_bg", [2.0 * np.pi / 600.0, 2.0 * np.pi / 600.0 + 1e-4j])
def test_fast_assembly_matches_reference(pol, wnum_bg):
    # Two functions implement identical arithmetic in a different loop order,
    # so they must agree to rounding. Any looser agreement means the port
    # broke an expression.
    nn = 64
    geom = Geometry.gielis(rad=200.0, n_pts=nn, m=0)
    mat = Material(n_core=1.5, n_clad=1.0, pol=pol)

    args = (
        pol,
        nn,
        geom.f,
        geom.g,
        geom.df,
        geom.dg,
        geom.ddf,
        geom.ddg,
        geom.delt,
        wnum_bg,
        mat.nc,
        mat.eps,
    )
    m_fast = assemble_matrix(*args)
    m_ref = assemble_matrix_reference(*args)

    assert np.allclose(m_fast, m_ref, rtol=1e-12)


# ---------------------------------------------------------------------------
# Analytic dM/dk_bg
#
# Two independent axes beyond the polarisation: a complex wavenumber (the QNM
# case, and the only one the derivative exists to serve) and an absorbing
# particle, which is what makes nc genuinely complex. The M3/M4 derivative
# blocks each carry an explicit factor nc from d k_core/d k_bg, and at
# epsi = 0 that factor is real — a conjugation or a dropped factor there would
# pass unnoticed.
# ---------------------------------------------------------------------------

WNUM_CASES = [2.0 * np.pi / 600.0, 2.0 * np.pi / (600.0 + 40.0j)]


def _assembly_args(pol, nn, geom, mat, wnum_bg):
    """Positional argument tuple shared by both assembly entry points."""
    return (
        pol,
        nn,
        geom.f,
        geom.g,
        geom.df,
        geom.dg,
        geom.ddf,
        geom.ddg,
        geom.delt,
        wnum_bg,
        mat.nc,
        mat.eps,
    )


@pytest.mark.parametrize("pol", [1, 2])
@pytest.mark.parametrize("epsi", [0.0, 0.3])
@pytest.mark.parametrize("wnum_bg", WNUM_CASES)
def test_matrix_derivative_matches_assembly(pol, epsi, wnum_bg):
    # assemble_matrix_dwn duplicates ~50 lines of the assemble_matrix hot path
    # so that the matrix and its derivative can share the four Hankel arrays.
    # Nothing but this test stops the copy drifting, so the requirement is
    # bit-identity, not agreement: np.array_equal, no tolerance. Any rewrite of
    # an expression that changes only the rounding still fails here, which is
    # the intent — the two must stay literally the same arithmetic.
    nn = 64
    geom = Geometry.gielis(rad=200.0, n_pts=nn, m=0)
    mat = Material(n_core=1.5, n_clad=1.0, pol=pol, epsi=epsi)
    args = _assembly_args(pol, nn, geom, mat, wnum_bg)

    m_fused, _ = assemble_matrix_dwn(*args)

    assert np.array_equal(m_fused, assemble_matrix(*args))


@pytest.mark.parametrize("pol", [1, 2])
@pytest.mark.parametrize("epsi", [0.0, 0.3])
@pytest.mark.parametrize("wnum_bg", WNUM_CASES)
def test_matrix_derivative_matches_central_difference(pol, epsi, wnum_bg):
    # The derivative is written term by term from closed-form Hankel
    # identities, so the only independent check is the definition itself. A
    # central difference converges at second order, and observing that order —
    # rather than a single agreement at one step — is what distinguishes a
    # correct derivative from one that is merely close: a wrong constant factor
    # or a missing block leaves an h-independent residual, i.e. order 0.
    nn = 64
    geom = Geometry.gielis(rad=200.0, n_pts=nn, m=0)
    mat = Material(n_core=1.5, n_clad=1.0, pol=pol, epsi=epsi)

    _, dm = assemble_matrix_dwn(*_assembly_args(pol, nn, geom, mat, wnum_bg))
    scale = np.max(np.abs(dm))

    # Steps relative to |k| so the same three values work for both wavenumbers.
    # The window stops at 1e-4: by 1e-5 the round-off floor (~eps·|M|/h) starts
    # to show and the observed order drifts off 2.
    steps = [1e-2, 1e-3, 1e-4]
    errors = []
    for rel_step in steps:
        h = rel_step * abs(wnum_bg)
        fd = (
            assemble_matrix(*_assembly_args(pol, nn, geom, mat, wnum_bg + h))
            - assemble_matrix(*_assembly_args(pol, nn, geom, mat, wnum_bg - h))
        ) / (2.0 * h)
        errors.append(np.max(np.abs(fd - dm)) / scale)

    orders = [np.log10(errors[i] / errors[i + 1]) for i in range(len(errors) - 1)]
    # Measured 2.000 on every one of these eight cases; the ±0.05 band is
    # round-off headroom on the ratio, not a physics tolerance.
    assert np.allclose(orders, 2.0, atol=0.05), orders
    # Measured 6.7e-8 at the finest step. Pinning the magnitude as well as the
    # order stops the test passing on two errors that happen to shrink together.
    assert errors[-1] < 1e-7
