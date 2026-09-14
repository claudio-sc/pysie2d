import numpy as np
import pytest

from pysie2d.geometry import Geometry
from pysie2d.kernels import (
    _kress_log_weights,
    _kress_weights,
    assemble_matrix,
    assemble_matrix_dwn,
    assemble_matrix_reference,
)
from pysie2d.material import Material
from pysie2d.parametrisation import Parametrisation

EPS = np.finfo(float).eps

# Kress's weights: independent anchor
#
# R_d is defined by one property — it integrates ln(4 sin²((t−τ)/2)) exactly
# against every trigonometric polynomial the nodes interpolate — and that
# integral has a closed form: ∫₀^{2π} ln(4 sin²(τ/2)) cos(mτ) dτ = −2π/m for
# m ≥ 1 and 0 for m = 0. Checking the weights against the closed form uses
# nothing else in the package. Both parities, because the Nyquist term exists
# only for even nn and is the place a textbook formula goes wrong.


@pytest.mark.parametrize("nn", [8, 9, 64, 65])
def test_kress_log_weights_integrate_trig_polynomials_exactly(nn):
    weights = _kress_log_weights(nn)
    d = 2.0 * np.pi * np.arange(nn) / nn
    # Round-off of a sum of nn terms each bounded by max|R|: nn·eps·max|R|,
    # measured at that level (8.9e-16 at nn = 8, 1.3e-14 at nn = 65). 4× that
    # is still eleven decades below the O(1/m) value being checked.
    atol = 4.0 * nn * EPS * np.abs(weights).max()
    for m in range(nn // 2 + 1):
        exact = 0.0 if m == 0 else -2.0 * np.pi / m
        assert abs((weights * np.cos(m * d)).sum() - exact) <= atol, m


def test_the_even_only_weight_formula_is_wrong_at_odd_nn():
    # The control for the test above: the handoff's R_j (even nn only) applied
    # at odd nn misses the closed form by O(1/nn), not by round-off. Without
    # this, the parity branch in _kress_log_weights could be deleted and the
    # suite would only notice through a 1 nm QNM shift nobody is looking for.
    nn = 65
    n = nn // 2
    d = 2.0 * np.pi * np.arange(nn) / nn
    p = np.arange(1, n)
    textbook = -(2.0 * np.pi / n) * (
        (np.cos(np.outer(d, p)) / p).sum(1) + np.cos(n * d) / (2 * n)
    )
    worst = max(
        abs((textbook * np.cos(m * d)).sum() + 2.0 * np.pi / m) for m in range(1, n + 1)
    )
    assert worst > 1.0e-3


def test_kress_weights_are_shared_and_read_only():
    # One array per nn for the life of the process, handed to every assembly
    # and to every contour thread: an in-place write would corrupt all later
    # matrices silently, so it must raise instead.
    w = _kress_weights(64)
    assert w is _kress_weights(64)
    assert w[0] == _kress_log_weights(64)[0]
    with pytest.raises(ValueError):
        w[0] = 0.0


# Assembly parity
#
# A circle hides indexing mistakes: every row is a rotation of every other, so
# (i, j) ↔ (j, i) and d ↔ nn − d swaps leave it unchanged. The skewed shape on a
# graded map at odd nn breaks every one of those symmetries at once.
_SKEW = {"a": 1.3, "b": 0.8, "m": 4, "n1": 6.0, "n2": 12.0, "n3": 4.0}
_SKEW_MAP = Parametrisation.gielis(rad=200.0, **_SKEW, n_core=1.5)
GEOMETRIES = {
    "circle-64": lambda: Geometry.gielis(rad=200.0, n_pts=64, m=0),
    "skew-graded-33": lambda: Geometry.gielis(
        rad=200.0, n_pts=33, **_SKEW, parametrisation=_SKEW_MAP
    ),
}


def _assembly_args(pol, geom, mat, wnum_bg):
    """Positional argument tuple shared by every assembly entry point."""
    return (
        pol,
        geom.n_pts,
        geom.f,
        geom.g,
        geom.df,
        geom.dg,
        geom.ddf,
        geom.ddg,
        wnum_bg,
        mat.nc,
        mat.eps,
    )


@pytest.mark.parametrize("shape", sorted(GEOMETRIES))
@pytest.mark.parametrize("pol", [1, 2])
@pytest.mark.parametrize("wnum_bg", [2.0 * np.pi / 600.0, 2.0 * np.pi / 600.0 + 1e-4j])
def test_fast_assembly_matches_reference(shape, pol, wnum_bg):
    # The reference loops over node pairs explicitly, rebuilds W from R and the
    # logarithm inline, and evaluates every Bessel function at a complex
    # argument, so agreement here checks the triu indexing, the cached weights
    # and the real Cephes branch. The arithmetic is otherwise identical, so the
    # bound is rounding: measured 2.6e-16 relative, bound 1e-12.
    geom = GEOMETRIES[shape]()
    mat = Material(n_core=1.5, n_clad=1.0, pol=pol)
    args = _assembly_args(pol, geom, mat, wnum_bg)

    m_fast = assemble_matrix(*args)
    m_ref = assemble_matrix_reference(*args)

    assert np.abs(m_fast - m_ref).max() <= 1.0e-12 * np.abs(m_ref).max()


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


@pytest.mark.parametrize("shape", sorted(GEOMETRIES))
@pytest.mark.parametrize("pol", [1, 2])
@pytest.mark.parametrize("epsi", [0.0, 0.3])
@pytest.mark.parametrize("wnum_bg", WNUM_CASES)
def test_matrix_derivative_matches_assembly(shape, pol, epsi, wnum_bg):
    # assemble_matrix_dwn duplicates the assemble_matrix hot path so that the
    # matrix and its derivative can share the Bessel arrays. Nothing but this
    # test stops the copy drifting, so the requirement is bit-identity, not
    # agreement: np.array_equal, no tolerance. Any rewrite of an expression
    # that changes only the rounding still fails here, which is the intent —
    # the two must stay literally the same arithmetic.
    geom = GEOMETRIES[shape]()
    mat = Material(n_core=1.5, n_clad=1.0, pol=pol, epsi=epsi)
    args = _assembly_args(pol, geom, mat, wnum_bg)

    m_fused, _ = assemble_matrix_dwn(*args)

    assert np.array_equal(m_fused, assemble_matrix(*args))


# Finest-step bound per shape: the O(h²) constant is the third k-derivative of
# M, which is larger on the elongated shape *(measured 6.8e-8 on the circle,
# 2.0e-7 on the skewed shape at nn = 64)*. Each bound is ≤ 2.5× its measurement.
CENTRAL_DIFFERENCE_SHAPES = {
    "circle-64": (lambda: Geometry.gielis(rad=200.0, n_pts=64, m=0), 1.0e-7),
    "skew-graded-64": (
        lambda: Geometry.gielis(
            rad=200.0, n_pts=64, **_SKEW, parametrisation=_SKEW_MAP
        ),
        5.0e-7,
    ),
}


@pytest.mark.parametrize("shape", sorted(CENTRAL_DIFFERENCE_SHAPES))
@pytest.mark.parametrize("pol", [1, 2])
@pytest.mark.parametrize("epsi", [0.0, 0.3])
@pytest.mark.parametrize("wnum_bg", WNUM_CASES)
def test_matrix_derivative_matches_central_difference(shape, pol, epsi, wnum_bg):
    # The derivative is written term by term from closed-form Bessel
    # identities, so the only independent check is the definition itself. A
    # central difference converges at second order, and observing that order —
    # rather than a single agreement at one step — is what distinguishes a
    # correct derivative from one that is merely close: a wrong constant factor
    # or a missing block leaves an h-independent residual, i.e. order 0.
    build, finest_bound = CENTRAL_DIFFERENCE_SHAPES[shape]
    geom = build()
    mat = Material(n_core=1.5, n_clad=1.0, pol=pol, epsi=epsi)

    _, dm = assemble_matrix_dwn(*_assembly_args(pol, geom, mat, wnum_bg))
    scale = np.max(np.abs(dm))

    # Steps relative to |k| so the same three values work for both wavenumbers.
    # The window stops at 1e-4: by 1e-5 the round-off floor (~eps·|M|/h) starts
    # to show and the observed order drifts off 2.
    steps = [1e-2, 1e-3, 1e-4]
    errors = []
    for rel_step in steps:
        h = rel_step * abs(wnum_bg)
        fd = (
            assemble_matrix(*_assembly_args(pol, geom, mat, wnum_bg + h))
            - assemble_matrix(*_assembly_args(pol, geom, mat, wnum_bg - h))
        ) / (2.0 * h)
        errors.append(np.max(np.abs(fd - dm)) / scale)

    orders = [np.log10(errors[i] / errors[i + 1]) for i in range(len(errors) - 1)]
    # Measured 2.000 on every case; the ±0.05 band is round-off headroom on the
    # ratio, not a physics tolerance.
    assert np.allclose(orders, 2.0, atol=0.05), orders
    # Pinning the magnitude as well as the order stops the test passing on two
    # errors that happen to shrink together.
    assert errors[-1] < finest_bound
