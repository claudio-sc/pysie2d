"""The analytic QNM anchor: complex-x poles of the Mie coefficients.

This is the independent reference that licenses the Beyn contour extractor.
Nothing here touches the BIE solver, so agreement later is evidence about the
solver rather than two paths in this repo sharing an assumption.

The root-finder in ``reference.mie`` seeds from a grid, and a grid can only
miss roots. The two checks that matter therefore do *not* reuse it:

1. The winding number of ``D_n`` around the box, which counts zeros by the
   argument principle and so detects anything the seeder never sampled.
2. The identity ``D^TM_0 == D^TE_1``, which no numerical search can fake and
   which fails loudly if a polarisation code is swapped.
"""

import numpy as np
import pytest

from conftest import QNM_N_CORE, RAD
from pysie2d.reference.mie import (
    qnm_denominator,
    qnm_size_parameters,
    qnm_wavelengths,
)

M = QNM_N_CORE  # n_clad = 1.0 for this fixture, so the relative index is n_core

# The search box for the whole anchor table.  Re x > 0 keeps every Hankel
# argument off the H^(1) branch cut, which is what makes D_n holomorphic here
# and the argument principle applicable.
X_RANGE = (1.0, 3.0)
IM_X_FLOOR = 1.0e-4
IM_X_MAX = 1.5

# Root count per (pol, n) inside X_RANGE, from the winding number.  Most orders
# carry two radial branches; TM n=2 carries three, the third being a very broad
# Q = 0.96 mode.  A rule of thumb ("one root per order") is wrong here.
EXPECTED_ROOT_COUNTS = {
    2: {0: 2, 1: 2, 2: 2, 3: 2, 4: 1, 5: 1, 6: 1},
    1: {0: 2, 1: 2, 2: 3, 3: 2, 4: 1, 5: 1},
}
N_MAX_CHECKED = 14  # ceil(m·x_max) + 5 = 14; roots die out by n = 6

# Phase-3 go/no-go anchors, in nm.  TE n=0 is simple and crowded in Re λ;
# TE n=3 is degenerate and isolated.  Quoted to the digits the root-finder
# reproduces run to run.
ANCHOR_TE_SIMPLE = 530.83214 + 26.37850j
ANCHOR_TE_DEGENERATE = 760.68665 + 7.94771j


def winding_number(n, pol, x_range, im_lo, im_hi, m=M, n_pts=4000):
    """Zeros of ``D_n`` inside the rectangle, counted with multiplicity.

    The argument principle: ``D_n`` is holomorphic on a rectangle with
    ``Re x > 0`` (J and H are analytic away from the H branch cut on the
    negative real axis), so it has no poles to subtract and the change in
    ``arg D_n`` around the boundary counts zeros alone.

    Deliberately independent of ``qnm_size_parameters`` — it evaluates the
    denominator on the boundary and never solves for a root.
    """
    re_lo, re_hi = x_range
    corners = [
        re_lo + 1j * im_lo,
        re_hi + 1j * im_lo,
        re_hi + 1j * im_hi,
        re_lo + 1j * im_hi,
    ]
    t = np.linspace(0.0, 1.0, n_pts, endpoint=False)
    path = np.concatenate(
        [
            a + (b - a) * t
            for a, b in zip(corners, corners[1:] + corners[:1], strict=True)
        ]
    )
    with np.errstate(all="ignore"):
        d = qnm_denominator(n, path, m, pol)
    assert np.all(np.isfinite(d)) and not np.any(d == 0), (
        "D_n hit a zero or an overflow on the contour itself; the winding "
        "number is undefined there and the box must be moved"
    )
    # Close the loop before unwrapping so the final phase step is included.
    phase = np.unwrap(np.angle(np.concatenate([d, d[:1]])))
    return (phase[-1] - phase[0]) / (2.0 * np.pi)


def test_tm0_and_te1_denominators_are_identical():
    """D^TM_0 == D^TE_1 — the 2-D TE01/TM11 degeneracy, as an identity.

    Both reduce to ``m J_0(mx) H_0'(x) - J_0'(mx) H_0(x)`` with the m-weight on
    opposite terms of the same pair, so equality is algebraic, not numerical.
    A swapped ``pol`` code in the port breaks it immediately.
    """
    rng = np.random.default_rng(0)
    z = rng.uniform(0.5, 4.0, 200) - 1j * rng.uniform(1.0e-4, 1.5, 200)
    d_tm0 = qnm_denominator(0, z, M, 1)
    d_te1 = qnm_denominator(1, z, M, 2)

    rel = np.abs(d_tm0 - d_te1) / (np.abs(d_tm0) + np.abs(d_te1))
    # Machine epsilon accumulated over four special-function calls; measured
    # max is 1.4e-15, so 1e-13 is the identity holding, not a loose gate.
    assert rel.max() < 1.0e-13


@pytest.mark.parametrize("pol", [2, 1])
def test_root_count_matches_winding_number(pol):
    """The seeder finds every root the argument principle says is there.

    This is the completeness check.  A high-Q mode sits closer to the real axis
    than the seeding grid's ``im_x_floor`` and would be missed silently; only
    an independent count can reveal that, and a claim the seeder makes about
    its own coverage is not a check.
    """
    for n in range(N_MAX_CHECKED + 1):
        w = winding_number(n, pol, X_RANGE, -IM_X_MAX, -1.0e-5)
        found = qnm_size_parameters(n, M, pol, X_RANGE, IM_X_FLOOR, IM_X_MAX)
        expected = EXPECTED_ROOT_COUNTS[pol].get(n, 0)

        # The winding number is an integer by construction; 1e-3 only absorbs
        # the discretisation of the boundary integral.
        assert abs(w - expected) < 1.0e-3, f"pol={pol} n={n}: winding {w}"
        assert found.size == expected, f"pol={pol} n={n}: found {found.size}"


@pytest.mark.parametrize("pol", [2, 1])
def test_roots_are_decaying_modes_with_vanishing_denominator(pol):
    """Every root is an actual zero of D and an actual decaying mode."""
    orders, lams, mults = qnm_wavelengths(
        RAD, M, pol, X_RANGE, im_x_floor=IM_X_FLOOR, im_x_max=IM_X_MAX
    )
    assert orders.size == sum(EXPECTED_ROOT_COUNTS[pol].values())

    x = 2.0 * np.pi * RAD / lams
    for n, xi in zip(orders, x, strict=True):
        # Measured residuals run 8e-17 … 6e-15; 1e-12 is that floor with room
        # for platform variation in the Amos routines, not a tuned tolerance.
        assert abs(qnm_denominator(int(n), xi, M, pol)) < 1.0e-12

    # exp(-iωt) with Im ω < 0 for a decaying mode gives Im x < 0, hence Im λ > 0.
    # A root in the lower half λ-plane would be a growing mode: unphysical here.
    assert np.all(lams.imag > 0.0)
    assert np.all(lams.real > 0.0)
    assert np.array_equal(mults, np.where(orders == 0, 1, 2))


def test_anchor_wavelengths_reproduce():
    """The two Phase-3 anchors, to the precision the extractor will be judged by.

    TE n=0 at 530.83 + 26.38j is simple but crowded in Re λ (TE n=5 sits 25 nm
    below it); TE n=3 at 760.69 + 7.95j is degenerate and isolated.  Both must
    be exact to well beyond the discretisation error of any BIE result, or the
    anchor cannot distinguish an extractor bug from a mesh effect.
    """
    _, lams, _ = qnm_wavelengths(
        RAD, M, 2, X_RANGE, im_x_floor=IM_X_FLOOR, im_x_max=IM_X_MAX
    )

    for anchor in (ANCHOR_TE_SIMPLE, ANCHOR_TE_DEGENERATE):
        # 1e-5 nm: the table is quoted to 5 decimals, and Newton converges to
        # |D| ~ 1e-15, so the only error here is the quoted rounding.
        assert np.min(np.abs(lams - anchor)) < 1.0e-5


def test_quality_factors_span_the_useful_range():
    """The fixture carries both a broad and a very high-Q mode.

    This is what n_core = 3.0 buys over the driven fixture's Q ≈ 2.4, and it is
    the property that makes the extractor's rank and residual thresholds
    testable rather than assumed.
    """
    q_all = []
    for pol in (2, 1):
        _, lams, _ = qnm_wavelengths(
            RAD, M, pol, X_RANGE, im_x_floor=IM_X_FLOOR, im_x_max=IM_X_MAX
        )
        q_all.extend(lams.real / (2.0 * lams.imag))

    assert min(q_all) < 1.0  # the broad TM n=2 mode, Q = 0.96
    assert max(q_all) > 2.0e3  # the TE n=6 mode, Q = 2289


def test_truncation_order_is_asserted_not_assumed():
    """A root surviving at n_max means modes are being discarded: raise, don't trim.

    Silently returning a truncated set is the failure that a downstream
    completeness claim cannot recover from.
    """
    with pytest.raises(ValueError, match="n_max is too low"):
        qnm_wavelengths(RAD, M, 2, X_RANGE, n_max=3)


def test_search_box_must_avoid_the_branch_cut():
    """Re x <= 0 puts Hankel arguments on the H^(1) cut and breaks holomorphy."""
    with pytest.raises(ValueError, match="branch cut"):
        qnm_size_parameters(0, M, 2, (-1.0, 3.0))
