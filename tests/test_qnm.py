"""QNM extraction on the circle, against the analytic Mie roots.

This is the go/no-go for the whole feature. Phase 1 established the analytic
poles independently (``test_mie_qnm.py``, winding-number complete) and Phase 2
established that the contour algorithm is correct on pencils with known spectra
(``test_beyn.py``). What is left to test is the composition: that the BIE
operator's singularities *are* the Mie poles, to within its discretisation
error and nothing more.

Because the two halves were validated separately, a failure here is physics or
convention — not arithmetic.

Cost note: every test in this file assembles a complex 2·n_pts matrix at each
of ``4·n_quad_per_side`` contour points. ``n_quad_per_side = 6`` is used
throughout instead of the default 12, on the measurement that the two give
identical modes to 1e-8 nm while the discretisation error is 0.38 nm — the
contour integral is nowhere near the accuracy bottleneck here.
"""

import numpy as np
import pytest

from conftest import QNM_N_CORE, RAD
from pysie2d import Geometry, Material, QNMSolver

# Anchors from the Phase-1 table in test_mie_qnm.py, vacuum nm.
# TE n=0: simple (multiplicity 1), but crowded in Re λ — TE n=5 sits at
# 505.68+0.42j and TE n=2 at 550.47+20.24j, only ~20 nm away. What separates
# them is Im λ, so the box must be tight in Im and not generous.
ANCHOR_TE_SIMPLE = 530.83214 + 26.37850j
BOX_TE_SIMPLE = (520.0 + 15.0j, 545.0 + 40.0j)

# TE n=3: doubly degenerate and isolated (nearest TE neighbours 690.51 and
# 1035.09), so the box placement is forgiving and the rank is the point.
ANCHOR_TE_DEGENERATE = 760.68665 + 7.94771j
BOX_TE_DEGENERATE = (745.0 + 2.0j, 775.0 + 15.0j)

# TM n=0: simple, and the polarisation cross-check. A swapped pol code would
# put a TE mode here and miss by tens of nm.
ANCHOR_TM_SIMPLE = 690.51371 + 40.64175j
BOX_TM_SIMPLE = (675.0 + 30.0j, 705.0 + 50.0j)

N_PTS = 200
N_SIDE = 6

# Convergence is first order in n_pts, in *both* Re λ and Im λ
# *(measured: order 1.00-1.03 for each over n_pts = 100 → 400)*. At n_pts = 200
# the errors are Re −0.377 nm and Im −0.240 nm, so these bounds sit ~30 % above
# what the discretisation delivers. They are error budgets for a first-order
# method, not fitted numbers, and must not be widened to rescue a failure.
ATOL_RE_NM = 0.5
ATOL_IM_NM = 0.32


def qnm_solver(pol, n_pts=N_PTS):
    """QNMSolver on the D2 fixture: circle, rad 200 nm, n_core 3.0, n_clad 1.0."""
    return QNMSolver(
        Geometry.gielis(rad=RAD, n_pts=n_pts, m=0),
        Material(n_core=QNM_N_CORE, n_clad=1.0, pol=pol),
    )


@pytest.fixture(scope="module")
def te_simple():
    """The simple TE anchor, extracted once and shared: ~11 s to compute."""
    return qnm_solver(pol=2).modes(*BOX_TE_SIMPLE, n_quad_per_side=N_SIDE)


@pytest.fixture(scope="module")
def te_degenerate():
    """The degenerate TE anchor, extracted once and shared: ~11 s to compute."""
    return qnm_solver(pol=2).modes(*BOX_TE_DEGENERATE, n_quad_per_side=N_SIDE)


def test_beyn_matches_analytic_pole_te(te_simple):
    """The BIE operator is singular at the Mie pole — the go/no-go itself.

    Re and Im are asserted separately because they carry different absolute
    error at the same convergence order, and because a sign or convention
    error shows up in Im long before it shows up in Re.
    """
    assert te_simple.n_modes == 1
    lam = te_simple.wavelengths[0]

    assert abs(lam.real - ANCHOR_TE_SIMPLE.real) < ATOL_RE_NM
    assert abs(lam.imag - ANCHOR_TE_SIMPLE.imag) < ATOL_IM_NM


def test_beyn_matches_analytic_pole_tm():
    """Same, for TM — the polarisation mapping is part of the claim.

    pol=1 ↔ a_n ↔ TM. Swapping the code would land this box on a TE mode tens
    of nm away, which no tolerance here could absorb.
    """
    res = qnm_solver(pol=1).modes(*BOX_TM_SIMPLE, n_quad_per_side=N_SIDE)

    assert res.n_modes == 1
    lam = res.wavelengths[0]
    assert abs(lam.real - ANCHOR_TM_SIMPLE.real) < ATOL_RE_NM
    assert abs(lam.imag - ANCHOR_TM_SIMPLE.imag) < ATOL_IM_NM


def test_pole_error_is_first_order_in_resolution():
    """The tolerances above are an error budget; this is what justifies them.

    A pole that did not converge — or converged to the wrong thing — would
    still pass a fixed tolerance if that tolerance were loose enough. Measuring
    the order instead makes the tolerance falsifiable: first order in n_pts is
    the same rate the near-field quantities obey (see test_convergence.py).
    """
    errors = []
    for n_pts in (100, 200):
        res = qnm_solver(pol=2, n_pts=n_pts).modes(
            *BOX_TE_SIMPLE, n_quad_per_side=N_SIDE
        )
        assert res.n_modes == 1
        errors.append(abs(res.wavelengths[0] - ANCHOR_TE_SIMPLE))

    # Doubling n_pts halves a first-order error. Measured ratio 2.02 over this
    # interval; the window admits order 0.8-1.3 and excludes both stagnation
    # (ratio 1) and the second order that would mean the anchor is not the
    # limit being approached.
    ratio = errors[0] / errors[1]
    assert 1.74 < ratio < 2.46


def test_degenerate_pair_has_rank_two(te_degenerate):
    """The ±n pair is two modes, and the extractor must report it as two.

    Every n ≥ 1 mode of a circle is doubly degenerate through exp(±inθ). The
    pair is reported as two numerically equal entries with multiplicity 2
    rather than collapsed — merging them would discard an independent mode
    vector and make the count disagree with the analytic table.
    """
    assert te_degenerate.n_modes == 2
    assert np.array_equal(te_degenerate.multiplicity, [2, 2])

    lam = te_degenerate.wavelengths
    # The two partners are the same eigenvalue to working precision
    # *(measured: 2.4e-13 nm apart)*, so they are a degeneracy and not two
    # nearby distinct modes.
    assert abs(lam[0] - lam[1]) < 1.0e-9

    assert abs(lam[0].real - ANCHOR_TE_DEGENERATE.real) < ATOL_RE_NM
    assert abs(lam[0].imag - ANCHOR_TE_DEGENERATE.imag) < ATOL_IM_NM


def test_simple_pole_has_multiplicity_one(te_simple):
    """n = 0 is the one order that is *not* degenerate; the count must show it."""
    assert np.array_equal(te_simple.multiplicity, [1])


def test_poles_are_upper_half_plane(te_simple, te_degenerate):
    """Im λ > 0 is the decaying half-plane; a mode below it would be unphysical.

    Under exp(-iωt) a decaying mode has Im ω < 0, hence Im k < 0, hence
    Im λ > 0. This is the convention the whole search region rests on.
    """
    for res in (te_simple, te_degenerate):
        assert np.all(res.wavelengths.imag > 0.0)
        assert np.all(res.wavelengths.real > 0.0)
        assert np.all(res.quality_factors > 0.0)


def test_quality_factor_matches_the_analytic_mode(te_simple):
    """Q = Re λ / (2 Im λ) — the derived quantity users actually quote.

    Q inherits the error of Im λ, which is relatively the worse of the two
    (0.24 nm on 26 nm), so its tolerance is looser than that of Re λ and is
    the one that matters for a resonance claim.
    """
    q_analytic = ANCHOR_TE_SIMPLE.real / (2.0 * ANCHOR_TE_SIMPLE.imag)

    # *(measured 0.87 % relative error at n_pts = 200)*: first order in n_pts,
    # dominated by Im λ. 2 % is that with margin, not a fitted bound.
    assert abs(te_simple.quality_factors[0] / q_analytic - 1.0) < 0.02


def test_no_conjugate_pair_symmetry(te_simple):
    """The mirror partner is at −λ̄, not λ̄; the conjugate is not a mode.

    A negative test, because assuming conjugate pairs is the natural mistake
    for anyone carrying real-eigenvalue intuition into a non-Hermitian problem.
    The reality condition λ → −λ̄ puts partners at negative Re λ, outside the
    physical region entirely.
    """
    lam = te_simple.wavelengths[0]
    solver = qnm_solver(pol=2)

    # *(measured: sigma_ratio 4.3e-4 at the analytic pole against 7.2e-3 at its
    # conjugate — the conjugate is no more singular than a generic point.)*
    assert solver._sigma_ratio(lam.conjugate()) > 1.0e-3


def test_search_region_rejects_lower_half_plane():
    """A box reaching Im λ ≤ 0 is searching for growing modes: refuse it."""
    solver = qnm_solver(pol=2)
    with pytest.raises(ValueError, match="growing modes"):
        solver.modes(520.0 - 5.0j, 545.0 + 40.0j)


def test_search_region_rejects_negative_real_part():
    """Re λ ≤ 0 puts Hankel arguments on the branch cut and breaks holomorphy."""
    solver = qnm_solver(pol=2)
    with pytest.raises(ValueError, match="branch cut"):
        solver.modes(-10.0 + 15.0j, 545.0 + 40.0j)


def test_search_region_rejects_degenerate_rectangle():
    """A zero-area box encloses nothing; returning "no modes" would be a lie."""
    solver = qnm_solver(pol=2)
    with pytest.raises(ValueError, match="strictly above and right"):
        solver.modes(520.0 + 15.0j, 520.0 + 40.0j)


def test_modes_are_singular_and_generic_points_are_not(te_simple):
    """sigma_ratio separates a mode from an arbitrary point by six decades.

    The ratio, never the absolute σ_min: σ_min carries the scale of the
    operator, so an absolute threshold is dimensionally meaningless *(absolute
    σ_min at the exact analytic pole is 8.2e-4, so the research code's
    sigma_threshold=1e-12 could never have fired)*.

    *(measured at n_pts = 200: 8.2e-9 at the mode extracted with this test's 24
    contour nodes, 1.5e-14 with the default 48, against 2.4e-3 to 4.6e-3 at
    generic points.)* Note the analytic pole itself scores 4.3e-4 — no better
    than a generic point, because it is not a pole of the *discrete* operator.
    That gap is the discretisation error, and it is why this test compares
    against generic points rather than against the analytic value.
    """
    assert te_simple.sigma_ratio[0] < 1.0e-6

    solver = qnm_solver(pol=2)
    for generic in (532.5 + 27.5j, ANCHOR_TE_SIMPLE + 5.0):
        assert solver._sigma_ratio(generic) > 1.0e-4


def test_rank_may_exceed_mode_count_from_outside_leakage(te_simple):
    """A pole just outside the box leaks a rank direction in; that is expected.

    The spec assumed the rank gap would be clean because pysie2d assembles
    exactly. It is not: the degenerate TE n=2 pair at 550.47+20.24j sits 5 nm
    outside this box and leaks two rank directions through imperfect quadrature
    cancellation, whose eigenvalues then land near 550.40 and are discarded by
    the in-contour filter. Rank 3, one mode — and the filter is doing real
    work, so this test pins that behaviour rather than the rank number alone.
    """
    assert te_simple.rank > te_simple.n_modes
    # Leakage is quadrature error, so it falls geometrically with contour nodes
    # *(measured sv_ratio[1] at n_quad_per_side = 6, 8, 12, 16, 24: 6.2e-4,
    # 8.9e-5, 1.8e-6, 3.8e-8, 1.6e-11 — and by 24 it drops below rank_tol and
    # the rank becomes 1, the correct value)*. So the spec's "clean gap" is true
    # asymptotically, just not at the default resolution.
    assert te_simple.sv_ratio[1] < 1.0e-3
    # It is still three decades below the genuine direction, so nothing reading
    # sv_ratio could mistake it for a mode.
    assert te_simple.sv_ratio[1] < 1.0e-3 * te_simple.sv_ratio[0]


def test_edge_margin_reports_a_comfortable_box(te_simple):
    """The diagnostic that catches a pole being clipped by its own contour.

    Near zero means the box is too tight and the value is not to be trusted.
    Here the mode sits 42 % of the shorter side from the nearest edge.
    """
    assert np.all(te_simple.edge_margin > 0.1)


def test_empty_box_finds_nothing_and_says_why(te_simple):
    """A box in a gap of the spectrum returns no modes, and says so by cancelling.

    The nearest TE modes are 505.68+0.42j and 530.83+26.38j; this box sits
    between them in Im λ, where the analytic table says there is nothing.

    Note it does *not* report rank 0 — the neighbours leak a rank direction in,
    whose eigenvalue then lands outside and is filtered. What separates empty
    from populated is the cancellation, and it separates them by four decades
    *(measured 5.2e-7 here against 3.1e-2 for the box holding a mode)*. The two
    scale differently: leakage is quadrature error and vanishes with more nodes,
    while a residue inside the contour does not.
    """
    res = qnm_solver(pol=2).modes(515.0 + 5.0j, 525.0 + 12.0j, n_quad_per_side=N_SIDE)

    assert res.n_modes == 0
    assert res.cancellation < 1.0e-4 < te_simple.cancellation


def test_modes_are_seed_independent(te_simple):
    """The probe matrix is random; the physics is not."""
    other = qnm_solver(pol=2).modes(*BOX_TE_SIMPLE, n_quad_per_side=N_SIDE, rng_seed=99)

    assert other.n_modes == te_simple.n_modes
    assert np.allclose(other.wavelengths, te_simple.wavelengths, atol=1.0e-9)
