"""Validate the Gielis boundary arrays against independent differentiation.

``geometry._boundary_arrays`` computes the θ-derivatives in closed form
(``_rderiv``, ``_rderiv2``) and carries them to the quadrature parameter t by
the chain rule. Each anchor here is a *different* method — a fine-grid,
fourth-order finite difference on ``r(θ)``, or spectral differentiation of the
sampled boundary in t — rather than agreement between two code paths in this
package (non-negotiable 3).
"""

import numpy as np
import pytest

from pysie2d import Parametrisation
from pysie2d.geometry import Geometry, NonClosingBoundaryError, gielis

RAD = 600.0


@pytest.mark.parametrize(
    "label,kwargs",
    [
        ("circle", {"m": 0, "n1": 2, "n2": 2, "n3": 2, "a": 1.0, "b": 1.0}),
        ("ellipse", {"m": 4, "n1": 2, "n2": 2, "n3": 2, "a": 1.3, "b": 0.8}),
        ("superellipse n=4", {"m": 4, "n1": 4, "n2": 4, "n3": 4, "a": 1.0, "b": 1.0}),
        ("superellipse n=8", {"m": 4, "n1": 8, "n2": 8, "n3": 8, "a": 1.0, "b": 1.0}),
    ],
)
def test_ddf_ddg_match_fine_grid_finite_difference(label, kwargs):
    """d²f/dθ², d²g/dθ² from the closed form agree with an independent FD.

    On the default uniform-θ map ``t = θ``, so the stored t-derivatives *are*
    the θ-derivatives and the closed form is compared directly. ``a != b`` is
    honoured on every map since v0.6 (the v0.5 uniform-θ path forced
    ``a = b = 1``).

    A 5-point centred stencil trades truncation error O(h^4) against roundoff
    O(eps/h^2); h=1e-3 is the sweet spot measured for this formula (1e-5 is
    roundoff-dominated at ~4e-4 relative, 1e-3 bottoms out at ~1.6e-8). rtol
    1e-6 leaves two orders of magnitude of margin above that floor without
    masking a real defect.
    """
    geom = Geometry.gielis(RAD, n_pts=37, **kwargs)
    for theta, ddf, ddg in zip(geom.theta, geom.ddf, geom.ddg, strict=True):
        # Skip the exact singular points of tan/cot in the closed form — the
        # comparison there is between two different removable-limit branches,
        # not a test of the formula (see _rderiv2's docstring).
        arg = kwargs["m"] * theta / 4.0
        if np.isclose(np.cos(arg), 0.0, atol=1e-6) or np.isclose(
            np.sin(arg), 0.0, atol=1e-6
        ):
            continue
        f = lambda th: gielis(  # noqa: E731
            np.array([th]),
            RAD,
            kwargs["a"],
            kwargs["b"],
            kwargs["m"],
            kwargs["n1"],
            kwargs["n2"],
            kwargs["n3"],
            0,
            0,
        )[0][0]
        g = lambda th: gielis(  # noqa: E731
            np.array([th]),
            RAD,
            kwargs["a"],
            kwargs["b"],
            kwargs["m"],
            kwargs["n1"],
            kwargs["n2"],
            kwargs["n3"],
            0,
            0,
        )[1][0]
        h = 1e-3
        ddf_fd = (
            -f(theta + 2 * h)
            + 16 * f(theta + h)
            - 30 * f(theta)
            + 16 * f(theta - h)
            - f(theta - 2 * h)
        ) / (12 * h**2)
        ddg_fd = (
            -g(theta + 2 * h)
            + 16 * g(theta + h)
            - 30 * g(theta)
            + 16 * g(theta - h)
            - g(theta - 2 * h)
        ) / (12 * h**2)
        assert ddf == pytest.approx(ddf_fd, rel=1e-6, abs=1e-6 * RAD)
        assert ddg == pytest.approx(ddg_fd, rel=1e-6, abs=1e-6 * RAD)


def _spectral_dt(values, order):
    """Fourier derivative of a smooth periodic sample on an equispaced t grid."""
    n = values.size
    freq = np.fft.fftfreq(n, d=1.0 / n)
    freq[n // 2] = 0.0  # even n: the Nyquist mode has no odd-order derivative
    return np.real(np.fft.ifft(np.fft.fft(values) * (1j * freq) ** order))


@pytest.mark.parametrize("graded", [False, True], ids=["uniform-theta", "arc-length"])
def test_t_derivatives_match_spectral_differentiation(graded):
    """df … ddg are derivatives in t, chain rule included, on a graded map too.

    On a graded map ``ẍ = x''·w'² + x'·w''``: dropping the ``w''`` term, or
    storing θ-derivatives, is an O(1) relative error that no convergence test on
    a circle can see (there ``w'' = 0``). Spectral differentiation of the
    sampled ``f(w(t))`` in t is an independent method that sees it directly.

    Tolerance 1e-9 relative: spectral differentiation of order k amplifies
    round-off by ~(nn/2)^k, i.e. ~4e-12 for k = 2 at nn = 256, and the smooth
    ellipse is resolved far below that *(measured 1.8e-12 uniform θ, 1.2e-11
    arc length)*. A missing ``w''`` term misses by ~0.3.
    """
    shape = {"m": 4, "n1": 2.0, "n2": 2.0, "n3": 2.0, "a": 1.3, "b": 0.8}
    par = Parametrisation.gielis(rad=RAD, **shape, n_core=1.5) if graded else None
    geom = Geometry.gielis(RAD, n_pts=256, **shape, parametrisation=par)

    for values, first, second in (
        (geom.f, geom.df, geom.ddf),
        (geom.g, geom.dg, geom.ddg),
    ):
        assert (
            np.abs(_spectral_dt(values, 1) - first).max() <= 1e-9 * np.abs(first).max()
        )
        assert (
            np.abs(_spectral_dt(values, 2) - second).max()
            <= 1e-9 * np.abs(second).max()
        )


def test_non_closing_boundary_is_rejected():
    """A superformula that does not close after one turn must raise.

    ``r(θ)`` then jumps at θ = 0, and the periodic quadrature integrates the
    jump as if it were boundary — a plausible number with nothing raised. The
    rule: even ``m`` always closes; odd ``m`` only when ``a == b`` and
    ``n2 == n3``. Both halves of the odd-m condition are checked, since the
    v0.5 D5 note named only ``a != b``.
    """
    for bad in ({"m": 3, "b": 1.2}, {"m": 3, "n2": 4.0, "n3": 6.0}, {"m": 2.5}):
        with pytest.raises(NonClosingBoundaryError, match="does not close"):
            Geometry.gielis(RAD, 40, **bad)

    for good in ({"m": 3}, {"m": 4, "b": 1.2, "n2": 4.0, "n3": 6.0}, {"m": 0}):
        geom = Geometry.gielis(RAD, 40, **good)
        assert np.isfinite(geom.ddf).all() and np.isfinite(geom.ddg).all()


def test_m1_asymmetric_star_raises_the_dedicated_error():
    """``m = 1`` has no rotational symmetry to mask a broken closure by accident.

    A caller can catch this without matching on message text.
    """
    with pytest.raises(NonClosingBoundaryError, match="does not close"):
        Geometry.gielis(RAD, 40, m=1, b=1.2)


def test_a_v05_theta_array_is_refused_with_the_migration():
    """Passing angles where a map is expected is the v0.5 call; say so.

    Under Kress the angles alone cannot carry w' and w'', so accepting them
    would assemble a plausible wrong matrix — the failure the API break exists
    to prevent. The v0.5 keyword itself no longer exists (a plain TypeError from
    Python); this covers the array arriving under the new keyword.
    """
    base = Geometry.gielis(RAD, 40, m=4, b=1.2)
    with pytest.raises(TypeError, match="parametrisation=other.parametrisation"):
        Geometry.gielis(RAD, 40, m=4, b=1.2, parametrisation=base.theta)
    with pytest.raises(TypeError):
        Geometry.gielis(RAD, 40, m=4, b=1.2, theta=base.theta)
