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
from pysie2d import Geometry, Material, Parametrisation, QNMSolver, richardson_limit

# As test_qnm.py: at n_pts = 40 the discretisation is at 2.8e-14 nm on the TE
# n=0 anchor, and 12 contour nodes per side put extraction at 1.1e-11 nm.
N_PTS = 40
N_SIDE = 12

# TE n=0 on the shared circle: simple (multiplicity 1) and well separated, so
# the simple-pole quotient is the right branch and the mode cannot be confused
# with a neighbour. Box from test_qnm.py.
BOX_TE_SIMPLE = (520.0 + 15.0j, 545.0 + 40.0j)


@pytest.fixture(scope="module")
def te_simple():
    """A refined simple TE mode of the circle, with its frozen map."""
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
    map (§10) that holds for the discretised operator too, so the central
    difference has *no truncation error at all* and the only error left is
    cancellation. Each entry of ∂M/∂rad is ~M/rad and is differenced over
    h = 1e-5 nm, so that floor is ε·rad/h ≈ 2.2e-16·200/1e-5 ≈ 4e-9 relative
    *(measured 5.8e-10)*.

    The bound is therefore 1e-8 — 2.5× the derived floor, and six decades below
    the 1e-2-ish agreement a merely-correct-looking adjoint would give. v0.5
    quoted this floor as ε/h = 1e-11, which omitted the factor rad.
    """
    lam = te_simple.wavelengths[0]
    par = te_simple.geometry.parametrisation
    mat = te_simple.material

    def at(delta):
        return (
            Geometry.gielis(rad=RAD + delta, n_pts=N_PTS, m=0, parametrisation=par),
            mat,
        )

    got = te_simple.sensitivity(at)[0]
    assert got == pytest.approx(lam / RAD, rel=1.0e-8)


def test_gate1_dilation_is_gauge_free(te_simple):
    """Gate 1, second gauge: dλ/d(log rad) = λ, and second order in h.

    Reparametrising rad → rad·exp(δ) must give exactly λ, by the same scale
    covariance. Unlike the linear gauge this one has genuine curvature, so the
    central difference carries an O(h²) truncation term — which makes it the
    better test of the *difference scheme*: the error must fall by ×100 per
    decade in h, and does *(measured at n_pts = 40: 3.4e-9 at h = 1e-5)*.

    Asserting the rate, not just a magnitude, is what conventions §10 requires:
    a single step size cannot tell a converging scheme from one sitting on a
    plateau. The rate window [80, 120] is ±20 % on the ideal 100, wide enough
    for the round-off riding on a 1e-9 number and far too tight for an O(h)
    stall.
    """
    lam = te_simple.wavelengths[0]
    par = te_simple.geometry.parametrisation
    mat = te_simple.material

    def at(delta):
        geom = Geometry.gielis(
            rad=RAD * np.exp(delta), n_pts=N_PTS, m=0, parametrisation=par
        )
        return geom, mat

    fine = te_simple.sensitivity(at, step=1.0e-5)[0]
    coarse = te_simple.sensitivity(at, step=1.0e-4)[0]

    err_fine = abs(fine - lam) / abs(lam)
    err_coarse = abs(coarse - lam) / abs(lam)

    assert err_fine < 1.0e-8
    assert 80.0 < err_coarse / err_fine < 120.0


def test_sensitivity_rejects_a_rebuilt_map(te_simple):
    """A perturbed geometry on a different map must raise, not compute.

    Rebuilding an arc-length map at p₀ ± h moves the nodes with the shape, so
    ∂M/∂p differentiates the node gauge too *(measured: 20 % of ‖∂M/∂b‖,
    test_conventions.py)* and the adjoint quotient would combine matrices from
    different discretisations, with every number downstream still plausible.
    So the precondition is enforced by the library rather than documented.

    An ellipse on its own arc-length map, because the default uniform-θ map is
    shape-independent and *cannot* be rebuilt differently — on a circle, or on
    uniform θ, this test would have nothing to catch. Exact equality is the
    guard: the check runs before any matrix is built, so the box need not even
    hold a mode.
    """
    shape = {"a": 1.0, "b": 1.2, "m": 4, "n1": 2.0, "n2": 2.0, "n3": 2.0}
    mat = te_simple.material

    def arc_map(b):
        return Parametrisation.gielis(rad=RAD, **{**shape, "b": b}, n_core=QNM_N_CORE)

    base = QNMSolver(
        Geometry.gielis(RAD, N_PTS, **shape, parametrisation=arc_map(1.2)), mat
    ).modes(543.0 + 18.0j, 560.0 + 32.0j, n_quad_per_side=N_SIDE)

    def at(delta):
        b = 1.2 + delta
        return (
            Geometry.gielis(
                RAD, N_PTS, **{**shape, "b": b}, parametrisation=arc_map(b)
            ),
            mat,
        )

    with pytest.raises(ValueError, match="base node set"):
        base.sensitivity(at)


# TE n=3 on the shared circle: doubly degenerate through exp(±3iθ) and
# isolated (nearest TE neighbours 690.51 and 1035.09 nm), so the box placement
# is forgiving and the multiplicity is what the test is about. Box from
# test_qnm.py. Not refined: bordered Newton assumes a one-dimensional null
# space and QNMResult.refine deliberately keeps the contour estimate here.
BOX_TE_DEGENERATE = (745.0 + 2.0j, 775.0 + 15.0j)


@pytest.fixture(scope="module")
def te_degenerate():
    """The doubly degenerate TE n=3 pair of the circle."""
    geom = Geometry.gielis(rad=RAD, n_pts=N_PTS, m=0)
    mat = Material(n_core=QNM_N_CORE, n_clad=1.0, pol=2)
    return QNMSolver(geom, mat).modes(
        z_lo=BOX_TE_DEGENERATE[0], z_hi=BOX_TE_DEGENERATE[1], n_quad_per_side=N_SIDE
    )


def test_gate1_degenerate_pair_does_not_split_under_dilation(te_degenerate):
    """Gate 1, degenerate half: dilation moves the pair without splitting it.

    A dilation cannot lift the ±n degeneracy of a circle — it maps the circle
    to a circle — so the 2×2 secular matrix must be a multiple of the identity
    and both eigenvalues must equal λ/rad. This is the test the whole degenerate
    branch stands on: the scalar quotient applied to a two-dimensional null
    space returns a plausible number for an arbitrary vector out of that space,
    and only the identity structure of the secular matrix exposes it.

    Assertions are on the *splitting* rather than on smallness alone, since a
    symmetry-breaking perturbation would split at first order, i.e. by O(1)
    relative — nine decades above the bound here.

    Tolerance 1e-8: λ is exactly linear in rad (§9), so the difference quotient
    has no truncation error and the floor is cancellation, ε·rad/h ≈ 4e-9 as in
    gate 1 *(measured at n_pts = 40: splitting 3.1e-10, both eigenvalues within
    7.8e-10 of λ/rad)*. The bound sits an order of magnitude above what was
    measured and cannot be reached by a broken branch.
    """
    assert tuple(te_degenerate.multiplicity) == (2, 2)

    lam = te_degenerate.wavelengths[0]
    par = te_degenerate.geometry.parametrisation
    mat = te_degenerate.material

    def at(delta):
        return (
            Geometry.gielis(rad=RAD + delta, n_pts=N_PTS, m=0, parametrisation=par),
            mat,
        )

    got = te_degenerate.sensitivity(at)

    assert got.shape == (2,)
    assert abs(got[0] - got[1]) / abs(got[0]) < 1.0e-8
    assert got[0] == pytest.approx(lam / RAD, rel=1.0e-8)
    assert got[1] == pytest.approx(lam / RAD, rel=1.0e-8)


# Gate 2 lives on an ellipse rather than a circle: m=4, n1=n2=n3=2 makes the
# Gielis boundary an exact ellipse with semi-axes b·rad along x and a·rad
# along z (D14), so a≠b puts the derivative through the real Gielis shape path
# — the same finite-difference code the catalogue depends on — instead of
# through the ∂M/∂rad = −(λ/rad)·∂M/∂λ identity, which can pass while the shape
# derivative is broken.
ELLIPSE = {"m": 4, "n1": 2.0, "n2": 2.0, "n3": 2.0, "a": 1.0, "b": 1.2, "n_pts": N_PTS}
BOX_ELLIPSE = (520.0 + 5.0j, 620.0 + 45.0j)


def _ellipse(parametrisation=None, **overrides):
    """The Gate-2 ellipse, optionally on a frozen map."""
    kw = {**ELLIPSE, "rad": RAD}
    kw.update(overrides)
    return Geometry.gielis(**kw, parametrisation=parametrisation)


def test_gate2_ab_dilation_is_degenerate_with_rad(te_simple):
    """Gate 2: at n2 = n3, (log a + log b) and log rad move λ identically.

    D15: at n2 = n3 the map (a, b) → (ta, tb) scales the radius by exactly
    t^(n2/n1) at every θ — a pure dilation, degenerate with rad. So the
    Jacobian of λ over (log a + log b, log rad) has rank one, and

        δ(log a + log b) − δ(log rad)

    is an **exact null direction**. That is the statement asserted here, on two
    modes at once, and it is the strongest form available: not "the derivative
    is small" but "two independently computed derivatives agree", with the
    common value λ itself pinned by scale covariance.

    Tolerance 1e-9 on the ratio: both derivatives run through the same central
    difference with no truncation error (λ is linear in each of these dilations)
    so the floor is cancellation *(measured: ratio − 1 is 1.3e-12 at n_pts =
    40)*. A shape derivative that were merely approximately right would miss
    this by orders of magnitude.

    ``te_simple`` is requested only so the module fixture ordering keeps the
    expensive circle solve shared; this test builds its own ellipse.
    """
    mat = te_simple.material
    geom = _ellipse()
    par = geom.parametrisation
    # Refined: the quotient is evaluated *at* λ, so the identity dλ/dlog rad = λ
    # can only hold as tightly as λ itself sits on the pole. An unrefined
    # contour estimate is off by ~3e-9 relative here, which is what the second
    # assertion would otherwise be measuring.
    res = QNMSolver(geom, mat).modes(
        z_lo=BOX_ELLIPSE[0], z_hi=BOX_ELLIPSE[1], n_quad_per_side=N_SIDE
    )
    assert res.n_modes == 2
    res = res.refine()
    assert res.converged.all()

    d_ab = res.sensitivity(
        lambda d: (
            _ellipse(
                parametrisation=par,
                a=ELLIPSE["a"] * np.exp(d),
                b=ELLIPSE["b"] * np.exp(d),
            ),
            mat,
        )
    )
    d_rad = res.sensitivity(
        lambda d: (_ellipse(parametrisation=par, rad=RAD * np.exp(d)), mat)
    )

    assert np.allclose(d_ab / d_rad, 1.0, rtol=1.0e-9, atol=1.0e-9)
    # ...and the common value is λ, which is what makes this a null *direction*
    # of a rank-one Jacobian rather than two derivatives that merely match.
    # Looser than the ratio by a decade, and for a reason: the exponential
    # gauge has curvature, so each derivative carries the O(h²) truncation term
    # measured in test_gate1_dilation_is_gauge_free (3.4e-9 at h = 1e-5). It
    # is common to both gauges and cancels in the ratio above, which is why
    # that assertion can be four decades tighter than this one.
    assert np.allclose(d_ab, res.wavelengths, rtol=1.0e-8)


def test_gate2_boundary_control_ab_dilation_is_only_pure_at_n2_eq_n3():
    """The D15 premise, measured on the boundary, with its n2 ≠ n3 control.

    (a, b) → (ta, tb) is a *pure* dilation only at n2 = n3; away from it the
    same map deforms the shape. Without this control the test above could pass
    on a Jacobian that is rank-one for the wrong reason. No electromagnetics
    here — it is the geometric premise on its own.

    Exact equality to round-off is the bar at n2 = n3 (the radius ratio is
    t^(n2/n1) = 1.3 at every node), so 1e-12 is ~4 decades above the measured
    8.9e-16 spread; the n2 ≠ n3 control comes out at 0.39, i.e. 15 orders away.
    """
    t = 1.3
    for n3, bound in ((2.0, 1.0e-12), (4.0, None)):
        base = _ellipse(n3=n3)
        r0 = np.hypot(base.f, base.g)
        grown = _ellipse(
            parametrisation=base.parametrisation,
            n3=n3,
            a=ELLIPSE["a"] * t,
            b=ELLIPSE["b"] * t,
        )
        r1 = np.hypot(grown.f, grown.g)
        spread = (r1 / r0).max() - (r1 / r0).min()
        if bound is None:
            assert spread > 0.1  # control: the map is not a dilation here
        else:
            assert spread < bound
            assert (r1 / r0).mean() == pytest.approx(
                t ** (ELLIPSE["n2"] / ELLIPSE["n1"])
            )


def test_gate3_adjoint_matches_re_extracted_poles(te_simple):
    """Gate 3: adjoint dλ/db against central differences of re-extracted poles.

    The independent anchor for the whole API. Everything in ``sensitivity()``
    — the left null vector, the analytic ∂M/∂λ, the frozen-node ∂M/∂p — is
    bypassed on the reference side: λ(b ± Δ) is obtained by running Beyn's
    contour method again on the perturbed geometry and polishing it, which
    shares no code path with the quotient beyond the assembly itself.

    Linearity in the step is the pass criterion, not a single Δ: a single step
    size cannot distinguish a converging scheme from one that has stalled, and
    the O(h) stall of the un-frozen node set (conventions §10) is exactly the
    failure this must be able to see. A central difference of a smooth function
    must fall ×100 per decade in Δ, and does *(measured on the ellipse's lower
    mode at n_pts = 40: 4.495e-4 → 4.489e-6 → 4.493e-8, ratios 100.1 and 99.9;
    a further decade gives only 76.7 as cancellation in the re-extracted poles
    takes over, which is why the ladder stops at Δ = 1e-4)*.

    The rate window [80, 120] is ±20 % on the ideal, and the absolute bound
    1e-7 at the finest step is ~3× above what was measured. Neither can be
    reached by a broken adjoint: a wrong left null vector misses by an O(1)
    factor.

    One mode, in a box tight enough that the perturbed pole cannot be confused
    with a neighbour: dλ/db ≈ 52 nm per unit b, so Δ = 1e-2 moves the pole by
    0.5 nm inside a 17 × 14 nm box.
    """
    mat = te_simple.material
    geom = _ellipse()
    par = geom.parametrisation
    box = (543.0 + 18.0j, 560.0 + 32.0j)

    def solve(b):
        res = QNMSolver(_ellipse(parametrisation=par, b=b), mat).modes(
            z_lo=box[0], z_hi=box[1], n_quad_per_side=N_SIDE
        )
        assert res.n_modes == 1
        return res.refine()

    base = solve(ELLIPSE["b"])
    adjoint = base.sensitivity(
        lambda d: (_ellipse(parametrisation=par, b=ELLIPSE["b"] + d), mat)
    )[0]

    errors = []
    for delta in (1.0e-2, 1.0e-3, 1.0e-4):
        fd = (
            solve(ELLIPSE["b"] + delta).wavelengths[0]
            - solve(ELLIPSE["b"] - delta).wavelengths[0]
        ) / (2.0 * delta)
        errors.append(abs(fd - adjoint) / abs(adjoint))

    assert errors[-1] < 1.0e-7
    for coarse, fine in zip(errors[:-1], errors[1:], strict=True):
        assert 80.0 < coarse / fine < 120.0


# Gate 10 design point: the Gate-2/3 ellipse (m=4, n1=n2=n3=2 is an exact
# ellipse, D14) and its single mode, so the resolution ladder runs through the
# real Gielis shape-derivative path rather than the circle's dilation identity.
GATE10_ELLIPSE = {"m": 4, "n1": 2.0, "n2": 2.0, "n3": 2.0, "a": 1.0, "b": 1.2}
GATE10_BOX = (543.0 + 18.0j, 560.0 + 32.0j)


def _dlam_db(n_pts):
    """dλ/db at the Gate-10 design point, at one resolution."""
    mat = Material(n_core=3.0, n_clad=1.0, pol=2)
    geom = Geometry.gielis(rad=RAD, n_pts=n_pts, **GATE10_ELLIPSE)
    res = QNMSolver(geom, mat).modes(
        z_lo=GATE10_BOX[0], z_hi=GATE10_BOX[1], n_quad_per_side=N_SIDE
    )
    assert res.n_modes == 1, f"n_pts={n_pts}: box holds {res.n_modes} modes"
    res = res.refine()
    par = res.geometry.parametrisation

    def at(delta):
        moved = Geometry.gielis(
            rad=RAD,
            n_pts=n_pts,
            parametrisation=par,
            **{**GATE10_ELLIPSE, "b": GATE10_ELLIPSE["b"] + delta},
        )
        return moved, mat

    return res.sensitivity(at)[0]


def test_gate10_jacobian_converges_spectrally():
    """Gate 10: J = dλ/db is converged by resolution, no extrapolation needed.

    v0.5 met this gate by two-rung Richardson because J converged at first
    order (conventions §12). Under Kress–Martensen quadrature it converges
    spectrally, like λ, and the premise that made extrapolation necessary is
    gone *(measured: J at n_pts = 40, 60, 80 is 51.3180585297, 51.3180585126,
    51.3180585145 nm per unit b in Re; |J(40) − J(80)|/|J(80)| = 3.0e-10 and
    |J(60) − J(80)|/|J(80)| = 4e-11, the latter at the contour and
    central-difference floor)*.

    Bounds 1e-8 and 1e-9: each ~25× its measurement. A first-order J at these
    resolutions — v0.5 sat 1.5 % from its limit at R = 50, n_pts = 378 — misses
    both by six decades.
    """
    j_40, j_60, j_80 = (_dlam_db(n) for n in (40, 60, 80))

    assert abs(j_40 - j_80) / abs(j_80) < 1.0e-8
    assert abs(j_60 - j_80) / abs(j_80) < 1.0e-9


def test_richardson_limit_is_exact_on_first_order_data_and_refuses_swapped_rungs():
    """The estimator itself, now that no pysie2d quantity exercises it.

    On q(n) = q* + C/n two rungs determine q* exactly, so agreement is to
    round-off: 1e-13 relative is ~500 ulp on a value of order 1. Swapped rungs
    must raise — the formula is antisymmetric in them and would extrapolate the
    wrong way with nothing to show for it.
    """
    q_star, c = 51.3 - 3.2j, 7.0 + 2.0j
    coarse, fine = q_star + c / 115, q_star + c / 228

    with pytest.warns(DeprecationWarning, match="no pysie2d quantity"):
        result = richardson_limit(coarse, fine, 115, 228)
    assert abs(result - q_star) < 1.0e-13 * abs(q_star)
    with (
        pytest.warns(DeprecationWarning),
        pytest.raises(ValueError, match="must exceed"),
    ):
        richardson_limit(fine, coarse, 228, 115)


def test_richardson_limit_warns_that_it_is_deprecated():
    """F7: no pysie2d quantity is first order in n_pts any more under Kress."""
    with pytest.warns(DeprecationWarning, match="deprecated"):
        richardson_limit(1.0, 2.0, 40, 80)
