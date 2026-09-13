"""QNM extraction on the circle, against the analytic Mie roots.

This is the go/no-go for the whole feature. Phase 1 established the analytic
poles independently (``test_mie_qnm.py``, winding-number complete) and Phase 2
established that the contour algorithm is correct on pencils with known spectra
(``test_beyn.py``). What is left to test is the composition: that the BIE
operator's singularities *are* the Mie poles, to within its discretisation
error and nothing more.

Because the two halves were validated separately, a failure here is physics or
convention — not arithmetic.

Resolution note: under Kress–Martensen quadrature the discretisation error at
``n_pts = 40`` is 2.8e-14 nm on the TE n=0 anchor, so the contour integral is
the accuracy floor: 7.0e-6 nm at 6 nodes per side, 1.1e-11 nm at the default
12 *(measured)*. The extraction fixtures therefore use 12, and the refinement
tests deliberately use 6, where there is a contour error for refine() to
remove.
"""

import numpy as np
import pytest

from conftest import QNM_N_CORE, RAD
from pysie2d import BIESolver, Geometry, Material, QNMSolver
from pysie2d.qnm import _null_vectors

# Anchors: Newton on reference.mie.qnm_denominator, full precision, vacuum nm.
# Five decimals (the Phase-1 table) is a fixed 1.3e-6 nm error, which a
# spectral solver resolves, so the digits matter here.
# TE n=0: simple (multiplicity 1), but crowded in Re λ — TE n=5 sits at
# 505.68+0.42j and TE n=2 at 550.47+20.24j, only ~20 nm away. What separates
# them is Im λ, so the box must be tight in Im and not generous.
ANCHOR_TE_SIMPLE = 530.8321407288238 + 26.37849897377869j
BOX_TE_SIMPLE = (520.0 + 15.0j, 545.0 + 40.0j)

# TE n=3: doubly degenerate and isolated (nearest TE neighbours 690.51 and
# 1035.09), so the box placement is forgiving and the rank is the point.
ANCHOR_TE_DEGENERATE = 760.6866483138609 + 7.94771058718579j
BOX_TE_DEGENERATE = (745.0 + 2.0j, 775.0 + 15.0j)

# TM n=0: simple, and the polarisation cross-check. A swapped pol code would
# put a TE mode here and miss by tens of nm.
ANCHOR_TM_SIMPLE = 690.5137112394168 + 40.64175329927141j
BOX_TM_SIMPLE = (675.0 + 30.0j, 705.0 + 50.0j)

N_PTS = 40
N_SIDE = 12
# Contour-limited extraction for the refinement tests: see the module docstring.
N_SIDE_CONTOUR_LIMITED = 6

# Error budget for an extracted pole against its analytic anchor at N_PTS and
# N_SIDE. *(measured |Δλ|: TE n=0 1.1e-11 nm — the contour floor — TE n=3
# 5.7e-13, TM n=0 7.1e-15.)* 1e-9 nm is ~90× the worst; a convention or
# polarisation error misses by tens of nm and a first-order discretisation by
# tenths, so neither can hide in it. Re and Im are held to the same bound.
ATOL_RE_NM = 1.0e-9
ATOL_IM_NM = 1.0e-9


def qnm_solver(pol, n_pts=N_PTS):
    """QNMSolver on the D2 fixture: circle, rad 200 nm, n_core 3.0, n_clad 1.0."""
    return QNMSolver(
        Geometry.gielis(rad=RAD, n_pts=n_pts, m=0),
        Material(n_core=QNM_N_CORE, n_clad=1.0, pol=pol),
    )


@pytest.fixture(scope="module")
def te_simple():
    """The simple TE anchor, extracted once and shared."""
    return qnm_solver(pol=2).modes(*BOX_TE_SIMPLE, n_quad_per_side=N_SIDE)


@pytest.fixture(scope="module")
def te_degenerate():
    """The degenerate TE anchor, extracted once and shared."""
    return qnm_solver(pol=2).modes(*BOX_TE_DEGENERATE, n_quad_per_side=N_SIDE)


# A deliberately badly drawn box for the refinement tests: the left edge is
# pushed to 530 nm, ~0.8 nm from the pole, so edge_margin falls to 0.033 — the
# regime the diagnostic exists to warn about. It also swallows TE n=2 at
# 550.47+20.24j, so it returns three modes and the anchor one must be picked
# out by proximity.
BAD_BOX_TE_SIMPLE = (530.0 + 15.0j, 560.0 + 40.0j)
N_SIDE_COARSE = 4


@pytest.fixture(scope="module")
def te_simple_contour_limited():
    """The simple anchor on a coarse contour: 7.0e-6 nm out, all of it contour."""
    return qnm_solver(pol=2).modes(
        *BOX_TE_SIMPLE, n_quad_per_side=N_SIDE_CONTOUR_LIMITED
    )


@pytest.fixture(scope="module")
def te_simple_refined(te_simple_contour_limited):
    """The contour-limited anchor after one refinement pass."""
    return te_simple_contour_limited.refine()


@pytest.fixture(scope="module")
def te_badly_placed():
    """Extraction from BAD_BOX_TE_SIMPLE, and the same refined."""
    res = qnm_solver(pol=2).modes(*BAD_BOX_TE_SIMPLE, n_quad_per_side=N_SIDE_COARSE)
    return res, res.refine()


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


def test_pole_error_is_spectral_in_resolution():
    """The tolerances above are an error budget; this is what justifies them.

    A pole that did not converge — or converged to the wrong thing — would
    still pass a fixed tolerance if that tolerance were loose enough. Measuring
    the rate instead makes the tolerance falsifiable. Under Kress–Martensen
    quadrature the error falls geometrically, faster at each step: *(measured
    at 24 contour nodes per side, so the contour floor of ~1e-13 nm is far
    below every rung: 1.20e-3, 8.66e-6, 3.30e-8 nm at n_pts = 20, 24, 28 —
    ratios 139 and 262)*. The bar is ×50 per step; a first-order scheme gives
    ×1.2 over the same steps.
    """
    errors = []
    for n_pts in (20, 24, 28):
        res = qnm_solver(pol=2, n_pts=n_pts).modes(*BOX_TE_SIMPLE, n_quad_per_side=24)
        assert res.n_modes == 1
        errors.append(abs(res.wavelengths[0] - ANCHOR_TE_SIMPLE))

    for coarse, fine in zip(errors[:-1], errors[1:], strict=True):
        assert fine < coarse / 50.0
    # ~3× the measured 3.3e-8, so the last rung is pinned in magnitude too.
    assert errors[-1] < 1.0e-7


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
    # *(measured: 6.8e-13 nm apart)*, so they are a degeneracy and not two
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
    (26 nm against 531), so its tolerance follows from the λ budget rather than
    being chosen: |ΔQ|/Q ≤ |ΔRe λ|/Re λ + |ΔIm λ|/Im λ ≤ 1e-9/531 + 1e-9/26
    ≈ 4e-11.
    """
    q_analytic = ANCHOR_TE_SIMPLE.real / (2.0 * ANCHOR_TE_SIMPLE.imag)

    # *(measured 3.8e-13)*; the bound is the derived 4e-11 rounded up.
    assert abs(te_simple.quality_factors[0] / q_analytic - 1.0) < 1.0e-10


def test_no_conjugate_pair_symmetry(te_simple):
    """The mirror partner is at −λ̄, not λ̄; the conjugate is not a mode.

    A negative test, because assuming conjugate pairs is the natural mistake
    for anyone carrying real-eigenvalue intuition into a non-Hermitian problem.
    The reality condition λ → −λ̄ puts partners at negative Re λ, outside the
    physical region entirely.
    """
    lam = te_simple.wavelengths[0]
    solver = qnm_solver(pol=2)

    # *(measured: sigma_ratio 1.9e-16 at the analytic pole against 3.5e-2 at its
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

    *(measured at n_pts = 40: 1.1e-14 at the mode extracted with 12 contour
    nodes per side, 6.8e-9 with 6, against 2.0e-3 and 5.2e-3 at the generic
    points below.)* The analytic pole itself now scores 1.9e-16: the
    discretised operator is singular there to round-off. v0.5's 0.38 nm error
    made it score like a generic point, which is why this test was written
    against generic points; that comparison still holds and still does not
    depend on the discretisation error.
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
    # *(measured sv_ratio[1] at n_pts = 40: 3.8e-4 at 6 nodes per side, 1.1e-6
    # at 12)*. So the spec's "clean gap" is true asymptotically, just not at the
    # default resolution.
    assert te_simple.sv_ratio[1] < 1.0e-3
    # It is still three decades below the genuine direction, so nothing reading
    # sv_ratio could mistake it for a mode.
    assert te_simple.sv_ratio[1] < 1.0e-3 * te_simple.sv_ratio[0]


def test_edge_margin_reports_a_comfortable_box(te_simple):
    """The diagnostic that catches a pole being clipped by its own contour.

    Near zero means the box is too tight and the value is not to be trusted.
    Here the mode sits 43 % of the shorter side from the nearest edge.
    """
    assert np.all(te_simple.edge_margin > 0.1)


def test_empty_box_finds_nothing_and_says_why(te_simple):
    """A box in a gap of the spectrum returns no modes, and says so by cancelling.

    The nearest TE modes are 505.68+0.42j and 530.83+26.38j; this box sits
    between them in Im λ, where the analytic table says there is nothing.

    Note it does *not* report rank 0 — the neighbours leak a rank direction in,
    whose eigenvalue then lands outside and is filtered. What separates empty
    from populated is the cancellation, and it separates them by ten decades
    *(measured 5.7e-11 here against 0.42 for the box holding a mode)*. The two
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


# ---------------------------------------------------------------------------
# Refinement (spec §6.2)
#
# What refinement does here is converge onto the singularity of the
# *discretised* operator. Under spectral boundary quadrature that operator is
# already at the analytic pole to 2.8e-14 nm at n_pts = 40, so what refinement
# removes is the contour-quadrature error — 7.0e-6 nm at 6 nodes per side. The
# tests below use that contour-limited extraction on purpose; see
# test_refine_removes_the_contour_error, which pins it.
# ---------------------------------------------------------------------------


def test_fresh_result_reports_no_refinement_attempted(te_simple, te_degenerate):
    """all-False plus all-NaN is the documented reading of "not tried".

    Neither field is a verdict on the modes: every mode in te_simple is good.
    The pairing matters — False alone would read as failure, and NaN keeps a
    user's ``cond_jacobian > DEGENERATE_COND`` test from classifying an
    unrefined mode as degenerate, since every comparison against NaN is False.
    """
    for res in (te_simple, te_degenerate):
        assert not res.converged.any()
        assert np.isnan(res.cond_jacobian).all()
        assert res.converged.shape == res.cond_jacobian.shape == res.wavelengths.shape


def test_refine_is_idempotent_at_convergence(te_simple, te_simple_refined):
    """A second pass has nothing left to do — the fixed point is a fixed point.

    The first pass is not a no-op: it moves λ by 7.0e-6 nm *(measured)*, the
    distance from the contour estimate to the true pole of the discretised
    operator. It is the *second* pass that must not move *(measured 1.2e-13
    nm)*; 1e-12 nm means the iteration recognised its own fixed point rather
    than orbiting it.
    """
    assert te_simple_refined.converged.all()
    again = te_simple_refined.refine()

    assert abs(again.wavelengths[0] - te_simple_refined.wavelengths[0]) < 1.0e-12
    assert again.converged.all()
    # cond(J) on a simple pole is ~1e3 *(measured 1.04e3)*, twelve orders below
    # DEGENERATE_COND. Asserting it here is what makes the degenerate test's
    # threshold a separation rather than a cutoff.
    assert te_simple_refined.cond_jacobian[0] < 1.0e6


def test_refine_drives_the_operator_to_singularity(te_badly_placed):
    """Refinement's actual claim, stated without a reference wavelength.

    A QNM *is* a wavelength where M(λ) is singular, so σ_min/σ_max is the
    property being converged on, and unlike a distance to some other
    computation it is not circular — nothing in this assertion was produced by
    the code under test.

    The contour here is badly drawn on purpose (edge_margin 0.033, the pole
    ~0.8 nm from the left edge, 4 nodes per side). It still yields a usable
    estimate: σ_min/σ_max falls from 2.5e-7 to 7.3e-17 *(measured)*, the floor
    set by the conditioning of an 80×80 complex matrix, while λ moves by
    2.6e-4 nm — onto the analytic pole, to 1.3e-13 nm. 1e-12 is five decades
    below the "before" and four above the "after", so it separates the two
    without pinning either.
    """
    coarse, refined = te_badly_placed
    k = int(np.argmin(np.abs(coarse.wavelengths - ANCHOR_TE_SIMPLE)))

    assert coarse.edge_margin[k] < 0.05  # the box really is badly placed
    assert coarse.sigma_ratio[k] > 1.0e-12
    assert refined.sigma_ratio[k] < 1.0e-12
    assert refined.converged[k]
    # The mode did not move far while becoming far more singular. 1e-3 nm is
    # ~4× the measured move and four decades below the 20 nm to TE n=2, the
    # neighbour this box also holds, so it cannot hide a jump to it.
    assert abs(refined.wavelengths[k] - coarse.wavelengths[k]) < 1.0e-3


def test_refine_removes_the_contour_error(te_simple_contour_limited, te_simple_refined):
    """The claim the documentation has to make, as a test.

    Under spectral boundary quadrature the discretised operator's pole sits on
    the analytic one to 2.8e-14 nm at n_pts = 40, so a coarse contour is the
    whole error — and refine() removes it: 7.0e-6 nm before, 1.3e-13 nm after
    *(measured)*. The "before" bound is a sanity check that the fixture really
    is contour-limited; the "after" bound sits ~800× above the measurement and
    four decades below the "before", so neither half can pass by accident.
    """
    err_before = abs(te_simple_contour_limited.wavelengths[0] - ANCHOR_TE_SIMPLE)
    err_after = abs(te_simple_refined.wavelengths[0] - ANCHOR_TE_SIMPLE)

    assert err_before > 1.0e-6
    assert err_after < 1.0e-10


def test_refine_flags_degenerate_pole(te_degenerate):
    """Every n ≥ 1 circle mode is degenerate, so this is the common case.

    Bordered Newton assumes a one-dimensional null space; the ±n pair gives it
    two, and the Jacobian is singular in exact arithmetic (cond 1.5e15 and
    2.9e15 here, *measured*). The requirement is that refine() notices, keeps
    the contour estimate untouched, and returns normally — a raised exception
    would make refine() unusable on any circle, and a silently "refined"
    wavelength would be worse.
    """
    refined = te_degenerate.refine()

    assert te_degenerate.n_modes == 2
    assert (te_degenerate.multiplicity == 2).all()
    assert (refined.cond_jacobian > 1.0e12).all()
    assert not refined.converged.any()
    # Bit-identical, not merely close: the Newton step was discarded, not taken
    # and found small. A "close" assertion would pass on a step that was taken.
    assert np.array_equal(refined.wavelengths, te_degenerate.wavelengths)
    # The vectors are discarded with the wavelength. Polishing them would be
    # worse than useless here: with a two-dimensional null space the pair is
    # only defined up to a rotation within it, so a "polished" partner could
    # silently duplicate the other one.
    assert np.array_equal(refined.vectors, te_degenerate.vectors)


def test_refine_polishes_the_mode_vector(te_simple_contour_limited, te_simple_refined):
    """A refined result must not disagree with itself.

    Before this, refine() moved λ onto the singularity and left ``vectors``
    where the contour put them, so ``sigma_ratio`` reported 1e-16 — the
    operator *is* singular there — while the mode vector could only
    demonstrate 1e-8. Polishing the pair together closes that: the residual
    falls from 4.8e-8 to 4.4e-16 on the contour-limited extraction
    *(measured)*.

    Columns stay unit-norm, and stay in the gauge the contour set: the Newton
    anchor conj(v0)·v = 1 fixes the phase against the input vector, so
    renormalising by a positive real cannot rotate it. Without that, columns
    from modes() and refine() would not be comparable.
    """
    coarse = te_simple_contour_limited
    bie = BIESolver(coarse.geometry, coarse.material)

    def residual(lam, v):
        m = bie.assemble(lam)
        return np.linalg.norm(m @ v) / (np.linalg.norm(m, 2) * np.linalg.norm(v))

    v_before = coarse.vectors[:, 0]
    v_after = te_simple_refined.vectors[:, 0]

    assert residual(coarse.wavelengths[0], v_before) > 1.0e-9
    assert residual(te_simple_refined.wavelengths[0], v_after) < 1.0e-13

    assert np.linalg.norm(v_after) == pytest.approx(1.0)
    # <v_before, v_after> = 1, not merely |<v_before, v_after>| = 1. The
    # modulus alone is phase-blind and would pass on any gauge; requiring the
    # inner product itself to be +1 is what pins the phase. A vector recomputed
    # from an SVD of M(λ) would satisfy every assertion above and fail this
    # one, which is the whole reason the Newton iterate is the one kept.
    assert abs(np.vdot(v_before, v_after) - 1.0) < 1.0e-6


def test_left_null_vector_is_not_the_right_one(te_simple_refined):
    """u from _null_vectors annihilates M from the left, and u != conj(v).

    Two claims, and the second is the one that matters. The adjoint quotient
    dλ/dp = −uᴴ(∂M/∂p)v / uᴴ(∂M/∂λ)v needs a genuine left null vector; a
    complex-symmetric M would give u = conj(v) for free, and this operator is
    not complex-symmetric *(measured here: ‖M − Mᵀ‖/‖M‖ = 1.16)*. Substituting
    conj(v) would leave the quotient wrong by an O(1) factor while every
    residual still looked right — |<u, conj(v)>| = 0.32 at this pole, so the
    error would be a factor of order three, not a rounding effect.

    The residual bound is not a tolerance to be tuned: from M = U Σ Vᴴ the left
    residual ‖uᴴM‖ is σ_min *exactly*, so ‖uᴴM‖/‖M‖₂ must equal σ_min/σ_max to
    floating-point round-off on the SVD. That round-off is ~eps/(σ_min/σ_max)
    relative, so the check needs a λ where M is *not* singular to round-off —
    which at the pole itself it now is (σ_min/σ_max = 1.3e-16 after refining).
    It is taken 0.5 nm off the pole, where σ_min/σ_max = 4.9e-4 and the two
    agree to 7.8e-15 *(measured)*: 1e-12 relative is ~100× above that floor and
    ~9 orders below the value itself. |⟨u, conj v⟩| = 0.32 there and at the
    pole alike — the non-symmetry is a property of the operator, not of λ.
    """
    bie = BIESolver(te_simple_refined.geometry, te_simple_refined.material)
    lam = ANCHOR_TE_SIMPLE + 0.5
    m = bie.assemble(lam)
    sigma = np.linalg.svd(m, compute_uv=False)
    ratio = sigma[-1] / sigma[0]

    u, v = _null_vectors(bie, lam)

    assert np.linalg.norm(u) == pytest.approx(1.0)
    assert np.linalg.norm(v) == pytest.approx(1.0)
    assert np.linalg.norm(u.conj() @ m) / sigma[0] == pytest.approx(ratio, rel=1.0e-12)
    assert np.linalg.norm(m @ v) / sigma[0] == pytest.approx(ratio, rel=1.0e-12)

    # The operator is not complex-symmetric, so conj(v) is not a left null
    # vector and the two directions are far from parallel.
    assert np.linalg.norm(m - m.T) / np.linalg.norm(m) > 1.0
    assert abs(np.vdot(u, v.conj())) < 0.5
