"""Adjoint eigenvalue sensitivity dλ/dp against closed forms (gates 1-3).

The adjoint quotient re-extracts no eigenvalue, so nothing inside it is
self-checking: a wrong left null vector, a null space picked from the wrong
side, or a node set silently re-inverted between p₀±h all give a number of the
right order. Every test here is therefore anchored on a derivative known in
closed form, not on agreement between two paths in this package.
"""

import numpy as np
import pytest

from conftest import QNM_N_CORE, RAD
from pysie2d import Geometry, Material, QNMSolver

N_PTS = 200
N_SIDE = 6

# TE n=0 on the shared circle: simple (multiplicity 1) and well separated, so
# the simple-pole quotient is the right branch and the mode cannot be confused
# with a neighbour. Box from test_qnm.py.
BOX_TE_SIMPLE = (520.0 + 15.0j, 545.0 + 40.0j)


@pytest.fixture(scope="module")
def te_simple():
    """A refined simple TE mode of the circle, with its frozen node set."""
    geom = Geometry.gielis(rad=RAD, n_pts=N_PTS, m=0)
    mat = Material(n_core=QNM_N_CORE, n_clad=1.0, pol=2)
    res = QNMSolver(geom, mat).modes(
        z_lo=BOX_TE_SIMPLE[0], z_hi=BOX_TE_SIMPLE[1], n_quad_per_side=N_SIDE
    )
    return res.refine()


def test_gate1_dilation_derivative_is_exact(te_simple):
    """Gate 1, gauge rad: dλ/drad = λ/rad, to the cancellation floor.

    Scale covariance (conventions §9) gives M(s·rad, s·λ) = M(rad, λ) entrywise,
    hence λ(s·rad) = s·λ(rad): λ is **exactly linear** in rad. On the frozen
    node set (§10) that holds for the discretised operator too, so the central
    difference has *no truncation error at all* and the only error left is
    cancellation, ~ε·‖M‖/h ≈ 1e-16/1e-5 = 1e-11.

    The bound is therefore 1e-9 — two decades above that floor, and seven below
    the 1e-2-ish agreement a merely-correct-looking adjoint would give. This is
    the discriminating form of Gate 1: before the node freeze a wrong result
    could hide inside a loose "small" tolerance, and it cannot here.
    """
    lam = te_simple.wavelengths[0]
    theta = te_simple.geometry.theta
    mat = te_simple.material

    def at(delta):
        return Geometry.gielis(rad=RAD + delta, n_pts=N_PTS, m=0, theta=theta), mat

    got = te_simple.sensitivity(at)[0]
    assert got == pytest.approx(lam / RAD, rel=1.0e-9)


def test_gate1_dilation_is_gauge_free(te_simple):
    """Gate 1, second gauge: dλ/d(log rad) = λ, and second order in h.

    Reparametrising rad → rad·exp(δ) must give exactly λ, by the same scale
    covariance. Unlike the linear gauge this one has genuine curvature, so the
    central difference carries an O(h²) truncation term — which makes it the
    better test of the *difference scheme*: the error must fall by ×100 per
    decade in h, and does *(measured: 3.46e-9 at h = 1e-5 against 3.46e-7 at
    h = 1e-4, a ratio of 99.8 against the ideal 100)*.

    Asserting the rate, not just a magnitude, is what conventions §10 requires:
    a single step size cannot tell a converging scheme from one sitting on a
    plateau. The rate window [80, 120] is ±20 % on the ideal 100, wide enough
    for the round-off riding on a 1e-9 number and far too tight for the 2.7
    that a non-frozen node set produces.
    """
    lam = te_simple.wavelengths[0]
    theta = te_simple.geometry.theta
    mat = te_simple.material

    def at(delta):
        geom = Geometry.gielis(rad=RAD * np.exp(delta), n_pts=N_PTS, m=0, theta=theta)
        return geom, mat

    fine = te_simple.sensitivity(at, step=1.0e-5)[0]
    coarse = te_simple.sensitivity(at, step=1.0e-4)[0]

    err_fine = abs(fine - lam) / abs(lam)
    err_coarse = abs(coarse - lam) / abs(lam)

    assert err_fine < 1.0e-8
    assert 80.0 < err_coarse / err_fine < 120.0


def test_sensitivity_rejects_a_re_inverted_node_set(te_simple):
    """A perturbed geometry that re-places its nodes must raise, not compute.

    The failure mode conventions §10 exists to remove is silent: re-inverting
    arc length between p₀−h and p₀+h puts an O(h) term into the quotient that
    *grows* with n_pts, and every number downstream still looks plausible. So
    the precondition is enforced by the library rather than documented.

    This checks the guard, not the size of the term it prevents: on a circle
    re-inversion moves the nodes by only 1.5e-13 rad *(measured)*, because θ is
    uniform in arc length at every rad. The guard is exact equality for that
    reason — there is no threshold at which node motion becomes acceptable.
    """

    def at(delta):
        # No theta= — the arc-length inversion runs again at each delta.
        return (
            Geometry.gielis(rad=RAD + delta, n_pts=N_PTS, m=0),
            te_simple.material,
        )

    with pytest.raises(ValueError, match="base node set"):
        te_simple.sensitivity(at)
