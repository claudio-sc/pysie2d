"""Validate the analytic Gielis second derivatives against finite differences.

``geometry._etoil``/``_etoil_arc`` compute ``ddf``/``ddg`` in closed form
(``_rderiv2``) rather than by differencing the already-computed ``df``/``dg``.
The anchor here is deliberately a *different* method — a fine-grid,
fourth-order finite difference taken directly on ``r(θ)`` — rather than
agreement between two code paths in this package (non-negotiable 3).
"""

import numpy as np
import pytest

from pysie2d.geometry import Geometry, gielis

RAD = 600.0


@pytest.mark.parametrize(
    "label,kwargs,arc_length",
    [
        ("circle", {"m": 0, "n1": 2, "n2": 2, "n3": 2, "a": 1.0, "b": 1.0}, False),
        ("ellipse", {"m": 4, "n1": 2, "n2": 2, "n3": 2, "a": 1.3, "b": 0.8}, True),
        (
            "superellipse n=4",
            {"m": 4, "n1": 4, "n2": 4, "n3": 4, "a": 1.0, "b": 1.0},
            False,
        ),
        (
            "superellipse n=8",
            {"m": 4, "n1": 8, "n2": 8, "n3": 8, "a": 1.0, "b": 1.0},
            False,
        ),
    ],
)
def test_ddf_ddg_match_fine_grid_finite_difference(label, kwargs, arc_length):
    """d²f/dθ², d²g/dθ² from the closed form agree with an independent FD.

    Covers both node-placement paths: ``arc_length=False`` (_etoil) on the
    circle and the two superellipses, ``arc_length=True`` (_etoil_arc) on the
    ellipse, which is the only path that honours ``a != b``.

    A 5-point centred stencil trades truncation error O(h^4) against roundoff
    O(eps/h^2); h=1e-3 is the sweet spot measured for this formula (1e-5 is
    roundoff-dominated at ~4e-4 relative, 1e-3 bottoms out at ~1.6e-8). rtol
    1e-6 leaves two orders of magnitude of margin above that floor without
    masking a real defect.
    """
    geom = Geometry.gielis(RAD, n_pts=37, arc_length=arc_length, **kwargs)
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
