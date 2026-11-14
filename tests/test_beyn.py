"""Beyn's contour eigensolver against pencils whose spectra are known exactly.

The point of testing the algorithm here rather than on the BIE matrix is
separation of blame. On a circle, a wrong pole can mean a broken contour
integral, a mis-detected rank, a sign convention, or a discretisation error,
and the four are hard to tell apart. On a pencil built from a diagonal of known
roots there is nothing to get wrong but the algorithm, so a failure in this
file localises immediately — and a pass licenses reading a later circle
disagreement as physics.

Every pencil is M(λ) = X · diag(fᵢ(λ)) · Y with X, Y unitary, so M(λ) is
singular exactly where some fᵢ vanishes and its condition number is that of the
diagonal alone.
"""

import itertools

import numpy as np
import pytest

from pysie2d.beyn import (
    EMPTY_CANCELLATION,
    INTERP_PEAK_OVER_SAMPLES,
    RANK_GAP_FLOOR,
    _EllipseInterpolant,
    beyn_modes,
    beyn_modes_adaptive,
    beyn_poles,
    contour_moments,
    ellipse_contour_quad,
    ellipse_faber_basis,
    newton_refine,
    probe_matrix,
    rect_contour_quad,
)

N_DIM = 40
Z_LO, Z_HI = 1.0 - 1.0j, 3.0 + 1.0j

# Roots deliberately placed outside the rectangle: they must not appear in any
# result, which is what tests the in-contour filter rather than assuming it.
OUTSIDE_ROOTS = [10.0 + 5.0j, -8.0 - 3.0j, 20.0 + 1.0j]


def make_pencil(eigs, scales=None, nonlinear=False, seed=1):
    """Build M(λ) and dM/dλ with eigenvalues exactly at ``eigs``.

    Args:
        eigs: Roots to place inside the rectangle. Repeat one for a degenerate
            eigenvalue.
        scales: Per-root multiplier on fᵢ. Scaling fᵢ up by c scales the
            residue of M⁻¹ there by 1/c, which is how a weak pole is built.
        nonlinear: Use fᵢ(λ) = exp(λ − rᵢ) − 1 instead of λ − rᵢ, so the
            problem is genuinely nonlinear in λ and not a disguised matrix
            eigenvalue problem.
        seed: Seed for the unitary factors.

    Returns:
        m_builder, dm_builder: callables λ → (N, N).
    """
    rng = np.random.default_rng(seed)
    roots = np.array(list(eigs) + OUTSIDE_ROOTS, dtype=complex)
    sc = np.ones(roots.size)
    if scales is not None:
        sc[: len(scales)] = scales

    # The remaining diagonal is constant and O(1). Padding with far-away roots
    # instead would make exp(λ − r) enormous and let the pencil's conditioning,
    # not the algorithm, decide the answer.
    pad = 1.0 + 0.5 * np.arange(1, N_DIM - roots.size + 1)

    x_mat, _ = np.linalg.qr(
        rng.standard_normal((N_DIM, N_DIM)) + 1j * rng.standard_normal((N_DIM, N_DIM))
    )
    y_mat, _ = np.linalg.qr(
        rng.standard_normal((N_DIM, N_DIM)) + 1j * rng.standard_normal((N_DIM, N_DIM))
    )

    def m_builder(lam):
        head = sc * (np.exp(lam - roots) - 1.0) if nonlinear else sc * (lam - roots)
        return x_mat @ np.diag(np.concatenate([head, pad])) @ y_mat

    def dm_builder(lam):
        head = sc * np.exp(lam - roots) if nonlinear else sc * np.ones(roots.size)
        return x_mat @ np.diag(np.concatenate([head, np.zeros(pad.size)])) @ y_mat

    return m_builder, dm_builder


# Test points kept ≥ 0.6 from every edge. The distance matters: Gauss-Legendre
# on 1/(λ−z) converges at ρ^(−2n) with ρ set by the Bernstein ellipse through
# z, so a point near an edge converges arbitrarily slowly. That is the same
# effect the QNM façade will report as edge_margin.
CAUCHY_INSIDE = (2.0 + 0.0j, 1.6 - 0.4j, 2.4 + 0.4j)
CAUCHY_OUTSIDE = (6.0 + 0.0j, -2.0 + 0.0j, 2.0 + 4.0j)


def test_contour_weights_reproduce_cauchy():
    """The quadrature reproduces Cauchy's integral formula, normalisation included.

    Three properties: Σw = 0 (the integral of an analytic function),
    Σ w/(λⱼ−z) = 1 for z inside (the residue), and 0 for z outside. The middle
    one validates the 1/(2πi) folded into the weights — a missing 2π would
    leave the other two intact and show up only here.
    """
    pts, wts = rect_contour_quad(Z_LO, Z_HI, 24)

    # Σw integrates the constant 1, which Gauss-Legendre does exactly, so this
    # is round-off alone *(measured: 0.0 to double precision)*.
    assert abs(wts.sum()) < 1.0e-15

    # *(measured 1.3e-13 at 24 nodes/side for these points)* — quadrature
    # error, not round-off, hence the separate and looser bound.
    for z_in in CAUCHY_INSIDE:
        assert abs(np.sum(wts / (pts - z_in)) - 1.0) < 1.0e-12

    # Outside points have no singularity inside the Bernstein ellipse of any
    # edge, so these are exact to round-off *(measured 2.8e-17)*.
    for z_out in CAUCHY_OUTSIDE:
        assert abs(np.sum(wts / (pts - z_out))) < 1.0e-13


def test_contour_quadrature_converges_geometrically():
    """Doubling the nodes must buy orders of magnitude, not a constant factor.

    A stronger statement than any single tolerance: it is what distinguishes a
    correct analytic-integrand quadrature from one that happens to be close.
    It also fixes the cost model — *(measured: 4.1e-7 at the default 12
    nodes/side against 1.3e-13 at 24, a factor of 3e6)* — so the default is a
    deliberate accuracy choice, not an assumption of exactness.
    """
    errors = []
    for n_side in (12, 24):
        pts, wts = rect_contour_quad(Z_LO, Z_HI, n_side)
        errors.append(max(abs(np.sum(wts / (pts - z)) - 1.0) for z in CAUCHY_INSIDE))

    assert errors[0] / errors[1] > 1.0e4


def test_contour_is_counter_clockwise():
    """Orientation sets the sign of every residue; a clockwise loop inverts it."""
    pts, _ = rect_contour_quad(Z_LO, Z_HI, 8)
    centre = 0.5 * (Z_LO + Z_HI)
    turning = np.sum(np.diff(np.unwrap(np.angle(pts - centre))))
    assert turning > 0.0


def test_degenerate_rectangle_is_rejected():
    """A zero-area rectangle encloses nothing; silently returning is worse."""
    with pytest.raises(ValueError, match="strictly above and right"):
        rect_contour_quad(1.0 + 0.0j, 1.0 + 1.0j, 8)


@pytest.mark.parametrize("nonlinear", [False, True], ids=["linear", "nonlinear"])
def test_beyn_recovers_synthetic_eigenvalues(nonlinear):
    """Both eigenvalues are found, to a precision the contour alone determines.

    Run for a nonlinear pencil too: a linear pencil would pass even if the
    method silently degenerated into a matrix eigenvalue solve, which is the
    one thing Beyn is used here *not* to be.
    """
    want = np.array([1.5 - 0.3j, 2.5 + 0.4j])
    m_builder, _ = make_pencil(want, nonlinear=nonlinear)

    res = beyn_modes(m_builder, Z_LO, Z_HI)

    assert res.n_modes == 2
    # 1e-10: Gauss-Legendre on an analytic integrand converges geometrically,
    # so at 48 nodes the error is round-off amplified by the pencil's condition
    # number, not quadrature error.
    assert np.allclose(
        np.sort_complex(res.eigenvalues), np.sort_complex(want), atol=1.0e-10
    )


def test_recovered_vectors_are_null_vectors():
    """The eigenvectors are the physical output; recovering λ alone is not enough.

    ‖M(λ)v‖ small relative to ‖M(λ)‖ is the statement that v spans the null
    space — the quantity the QNM façade will report as sigma_ratio.
    """
    want = [1.5 - 0.3j, 2.5 + 0.4j]
    m_builder, _ = make_pencil(want)

    res = beyn_modes(m_builder, Z_LO, Z_HI)

    for lam, vec in zip(res.eigenvalues, res.vectors.T, strict=True):
        mat = m_builder(lam)
        assert np.linalg.norm(mat @ vec) / np.linalg.norm(mat) < 1.0e-10


def test_eigenvalues_outside_the_contour_are_filtered():
    """Roots exist outside the box; the contour must not report them."""
    m_builder, _ = make_pencil([2.0 + 0.1j])

    res = beyn_modes(m_builder, Z_LO, Z_HI)

    assert res.n_modes == 1
    for root in OUTSIDE_ROOTS:
        assert np.min(np.abs(res.eigenvalues - root)) > 1.0


def test_beyn_empty_contour_returns_no_modes():
    """An analytic integrand integrates to zero: no rank, no modes, no error."""
    m_builder, _ = make_pencil([])

    res = beyn_modes(m_builder, Z_LO, Z_HI)

    assert res.n_modes == 0
    assert res.rank == 0
    # The spectrum is a slow decay, not a cliff — this is the diagnostic that
    # tells a user the box is empty rather than the tolerance being wrong.
    assert res.max_gap < RANK_GAP_FLOOR
    # And A₀ cancelled to nothing, which is what makes "empty" distinguishable
    # from "probe saturated": both leave the spectrum flat.
    assert res.cancellation < EMPTY_CANCELLATION


def test_beyn_is_seed_independent():
    """The probe is random; the spectrum is not.

    Beyn's rank argument holds for almost every probe, so a result that moves
    with the seed means the rank rule is reading noise.
    """
    want = [1.5 - 0.3j, 2.5 + 0.4j]
    m_builder, _ = make_pencil(want)

    results = [beyn_modes(m_builder, Z_LO, Z_HI, rng_seed=s) for s in (0, 1, 7, 12345)]

    for res in results:
        assert res.rank == 2
        assert np.allclose(res.eigenvalues, results[0].eigenvalues, atol=1.0e-10)


def test_beyn_raises_when_rank_saturates_probe():
    """Five eigenvalues, three probe columns: no gap can form, so refuse to guess.

    This is the failure that must not be silent. A saturated probe produces the
    same flat spectrum as an empty box, so returning "no modes" would be a
    confident wrong answer rather than an error.
    """
    m_builder, _ = make_pencil(
        [1.2 - 0.5j, 1.7 + 0.2j, 2.1 - 0.1j, 2.5 + 0.4j, 2.8 - 0.6j]
    )

    with pytest.raises(ValueError, match="saturated"):
        beyn_modes(m_builder, Z_LO, Z_HI, n_probe=3)

    # The same problem resolves cleanly once the probe can span the eigenspace.
    assert beyn_modes(m_builder, Z_LO, Z_HI, n_probe=12).n_modes == 5


def test_degenerate_eigenvalue_has_rank_two():
    """A repeated root contributes twice to the rank and appears twice.

    Collapsing the pair would discard an independent null vector — for the
    circle these are the ±n partners, real physics rather than a duplicate.
    """
    m_builder, _ = make_pencil([2.0 + 0.1j, 2.0 + 0.1j])

    res = beyn_modes(m_builder, Z_LO, Z_HI)

    assert res.rank == 2
    assert res.n_modes == 2
    assert np.allclose(res.eigenvalues, 2.0 + 0.1j, atol=1.0e-8)


def test_rank_survives_disparate_residues():
    """A pole 10⁶ weaker than its neighbour is still a pole.

    This is why the rank is read off the *last* cliff and not the largest one:
    here the drop between the two poles (1.1e6) exceeds the drop to the noise
    floor (2.1e5), so argmax over the gaps returns rank 1 and loses the weak
    mode entirely.
    """
    want = [1.5 - 0.3j, 2.5 + 0.4j]
    m_builder, _ = make_pencil(want, scales=[1.0, 1.0e6])

    res = beyn_modes(m_builder, Z_LO, Z_HI)

    assert res.rank == 2
    assert np.allclose(
        np.sort_complex(res.eigenvalues), np.sort_complex(want), atol=1.0e-8
    )


def test_finer_contour_does_not_cost_accuracy():
    """Refining the contour must not make the eigenvalues worse.

    Four poles sit just outside the box. Their weight in A₀ falls as the
    contour is refined, and on the way down it passes through every value
    between O(1) and round-off. A rank rule with a fixed floor drops such a
    pole while it still weighs far more than round-off, and because this pencil
    is not normal — its eigenvectors are not orthogonal — the dropped pole
    contaminates the ones inside at about its own weight *(measured with the
    former 1e-8 floor: 7.3e-14 at 16 nodes per side, then 2.1e-10 at 24 and
    1.5e-8 at 32)*. The pencils of ``make_pencil`` cannot show this: their
    factors are unitary, so a dropped pole is orthogonal to the rest.
    """
    rng = np.random.default_rng(1)
    n_dim = 80
    inside = np.array([0.3 + 0.2j, -0.5 + 0.1j, 0.1 - 0.3j, 0.85 + 0.0j, -0.2 + 0.42j])
    near = np.array([1.12 + 0.1j, -0.3 + 0.62j, 0.4 - 0.6j, -1.2 - 0.2j])
    n_far = n_dim - inside.size - near.size
    far = 3.0 * np.exp(2j * np.pi * rng.random(n_far)) * (1.0 + rng.random(n_far))
    roots = np.concatenate([inside, near, far])
    q = rng.standard_normal((n_dim, n_dim)) + 1j * rng.standard_normal((n_dim, n_dim))
    q_inv = np.linalg.inv(q)

    def m_builder(lam):
        return (q * (roots - lam)) @ q_inv

    for n_side in (16, 24, 32, 64):
        res = beyn_modes(m_builder, -1.0 - 0.5j, 1.0 + 0.5j, n_quad_per_side=n_side)
        assert res.n_modes == inside.size
        worst = max(np.min(np.abs(res.eigenvalues - root)) for root in inside)
        # 1e-12: *(measured 7.3e-14, 5.2e-14, 2.7e-14 and 1.9e-14)*, round-off
        # through a pencil whose eigenvector matrix has condition ~1e2. The
        # failure this guards against is four to six decades above the bound.
        assert worst < 1.0e-12


def test_linear_pencil_is_quadrature_exact():
    """A simple pole of a linear pencil is recovered exactly at any node count.

    Not a curiosity — it is why the refinement test below must use the
    nonlinear pencil. For a rank-1 residue, A₁'s integrand is
    λ/(λ−λ₀) = 1 + λ₀/(λ−λ₀), and Gauss-Legendre integrates the constant
    exactly, so A₁ = λ₀·A₀ *identically* and the quadrature error cancels out
    of the ratio. A linear pencil therefore cannot produce a coarse estimate to
    refine *(measured: 2.3e-15 at two nodes per side)*.
    """
    exact = 2.0 + 0.1j
    m_builder, _ = make_pencil([exact])

    for n_side in (2, 12):
        res = beyn_modes(m_builder, Z_LO, Z_HI, n_quad_per_side=n_side)
        assert abs(res.eigenvalues[0] - exact) < 1.0e-13


def test_newton_refine_converges_on_synthetic():
    """Refinement earns its place by fixing a contour too coarse to be accurate.

    The nonlinear pencil, where the cancellation above does not apply: three
    nodes per side leaves the estimate wrong in the seventh digit *(measured
    1.4e-7)*, and closing that gap is the behaviour that justifies shipping the
    bordered Newton at all.
    """
    exact = 2.0 + 0.1j
    m_builder, dm_builder = make_pencil([exact], nonlinear=True)

    coarse = beyn_modes(m_builder, Z_LO, Z_HI, n_quad_per_side=3)
    assert coarse.n_modes == 1
    err_before = abs(coarse.eigenvalues[0] - exact)
    assert err_before > 1.0e-9  # the premise: this estimate is genuinely coarse

    refined = newton_refine(
        m_builder,
        dm_builder,
        coarse.eigenvalues[0],
        coarse.vectors[:, 0],
        Z_LO,
        Z_HI,
    )

    assert refined.converged
    assert abs(refined.eigenvalue - exact) < 1.0e-12
    assert abs(refined.eigenvalue - exact) < err_before


def test_newton_refine_is_idempotent_at_convergence():
    """Started at a converged estimate, Newton must not move it."""
    exact = 2.0 + 0.1j
    m_builder, dm_builder = make_pencil([exact])
    res = beyn_modes(m_builder, Z_LO, Z_HI)

    refined = newton_refine(
        m_builder, dm_builder, res.eigenvalues[0], res.vectors[:, 0], Z_LO, Z_HI
    )

    assert refined.converged
    assert refined.step < 1.0e-9
    assert abs(refined.eigenvalue - res.eigenvalues[0]) < 1.0e-11


def test_newton_refine_flags_degenerate_eigenvalue():
    """The bordered system assumes a simple eigenvalue; say so instead of lying.

    With a two-dimensional null space the bordered Jacobian is singular, so the
    step is meaningless. It must be detectable from the result — a large
    condition number — rather than returned as a refined value.
    """
    m_builder, dm_builder = make_pencil([2.0 + 0.1j, 2.0 + 0.1j])
    res = beyn_modes(m_builder, Z_LO, Z_HI)

    refined = newton_refine(
        m_builder, dm_builder, res.eigenvalues[0], res.vectors[:, 0], Z_LO, Z_HI
    )

    assert refined.cond_jacobian > 1.0e12
    assert np.isfinite(refined.eigenvalue)


def test_beyn_poles_requires_the_cancellation_diagnostic():
    """The saturation check cannot be bypassed by calling the lower level.

    ``beyn_poles`` takes ``cancellation`` positionally and without a default so
    that a caller assembling its own moments still gets the empty-versus-
    saturated distinction rather than a plausible "no modes found".
    """
    m_builder, _ = make_pencil(
        [1.2 - 0.5j, 1.7 + 0.2j, 2.1 - 0.1j, 2.5 + 0.4j, 2.8 - 0.6j]
    )
    pts, wts = rect_contour_quad(Z_LO, Z_HI, 12)
    a0, a1, cancellation = contour_moments(
        m_builder, pts, wts, probe_matrix(N_DIM, 3, 0)
    )

    assert cancellation > EMPTY_CANCELLATION
    with pytest.raises(ValueError, match="saturated"):
        beyn_poles(a0, a1, cancellation, Z_LO, Z_HI)


def test_newton_refine_polishes_the_vector_with_the_eigenvalue():
    """The bordered system solves for (v, λ) jointly, so both must come back.

    Returning λ alone would hand the caller a vector one Newton step staler
    than the wavelength beside it — an internally inconsistent pair, where the
    eigenvalue claims a singularity the vector cannot demonstrate. On the
    coarse nonlinear pencil the stale vector leaves a residual of 3.8e-10 at
    the refined λ while the polished one reaches 3.4e-17 *(measured)*, seven
    orders apart.

    The residual is normalised by ‖M‖ and ‖v‖ so it measures direction, not
    the arbitrary scale of either.
    """
    exact = 2.0 + 0.1j
    m_builder, dm_builder = make_pencil([exact], nonlinear=True)
    # rank_tol = 1e-8 cuts the subspace to the one pole inside, which is what
    # leaves the vector coarse. The default rank rule also keeps the two
    # quadrature-error directions at 1e-8 and 1e-9, and on this pencil — whose
    # null vector does not depend on λ — that returns the vector at round-off
    # *(measured 4.8e-17)* with nothing left to polish.
    coarse = beyn_modes(m_builder, Z_LO, Z_HI, n_quad_per_side=3, rank_tol=1.0e-8)
    v0 = coarse.vectors[:, 0]

    refined = newton_refine(
        m_builder, dm_builder, coarse.eigenvalues[0], v0, Z_LO, Z_HI
    )

    def residual(lam, v):
        m = m_builder(lam)
        return np.linalg.norm(m @ v) / (np.linalg.norm(m, 2) * np.linalg.norm(v))

    stale = residual(refined.eigenvalue, v0)
    polished = residual(refined.eigenvalue, refined.vector)
    # A factor of 1e3 is four decades inside the seven measured, so this fails
    # on a vector that was returned unpolished but not on ordinary variation.
    assert polished < stale / 1.0e3

    # The documented normalisation: anchored, conj(v0)·v = 1, not unit length.
    # A caller renormalising to unit columns depends on this being the
    # constraint that is enforced *(measured exactly 1.0 here)*.
    assert abs(np.vdot(v0, refined.vector) - 1.0) < 1.0e-9

    # The two normalisations coincide only because beyn_modes hands out unit
    # columns. Feeding a scaled vector separates them — the anchor still holds
    # while the length does not — which is why a caller wanting unit columns
    # must divide rather than assume. Without this case the division in
    # QNMResult.refine would be untestable through any public path.
    scaled = newton_refine(
        m_builder, dm_builder, coarse.eigenvalues[0], 3.0 * v0, Z_LO, Z_HI
    )
    assert abs(np.vdot(3.0 * v0, scaled.vector) - 1.0) < 1.0e-9
    assert abs(np.linalg.norm(scaled.vector) - 1.0) > 0.1


def test_threading_reproduces_the_serial_moments_bit_for_bit():
    """Threading the contour must change no number, not merely no digit.

    ``np.array_equal``, not ``allclose``: the workers run the same calls on the
    same nodes, and the parent accumulates in *contour* order rather than in
    completion order, so the sum is the identical sequence of floating-point
    additions. A tolerance here would hide exactly the bug this guards — an
    accumulation whose order depends on thread timing, which would make A₀
    irreproducible in its last digits between two identical calls.
    """
    m_builder, _ = make_pencil([1.4 + 0.3j, 2.2 - 0.4j, 2.6 + 0.5j])
    pts, wts = rect_contour_quad(Z_LO, Z_HI, 12)
    v_probe = probe_matrix(N_DIM, 6, 0)

    serial = contour_moments(m_builder, pts, wts, v_probe, max_workers=1)
    threaded = contour_moments(m_builder, pts, wts, v_probe, max_workers=8)

    assert np.array_equal(threaded[0], serial[0])
    assert np.array_equal(threaded[1], serial[1])
    assert threaded[2] == serial[2]


def test_threaded_node_failure_still_warns_in_the_caller():
    """A per-node LU failure survives the thread boundary.

    The two traps of performance.md §3.1 in one test. ``warnings.warn`` issued
    inside a worker does not reliably reach the caller's ``catch_warnings``
    block, and a failure counter mutated from several threads is a race that
    loses counts — so the worker *returns* its status and the parent both
    counts and warns. Failing every third node makes the count deterministic
    (16 of 48), which is what turns "a warning appeared" into a check that the
    joining is correct.
    """
    m_builder, _ = make_pencil([1.4 + 0.3j, 2.2 - 0.4j])
    pts, wts = rect_contour_quad(Z_LO, Z_HI, 12)
    v_probe = probe_matrix(N_DIM, 6, 0)

    calls = itertools.count()

    def failing_builder(lam):
        if next(calls) % 3 == 0:
            raise np.linalg.LinAlgError("injected singular factorisation")
        return m_builder(lam)

    with pytest.warns(RuntimeWarning, match="failed to factorise") as record:
        contour_moments(failing_builder, pts, wts, v_probe, max_workers=8)

    assert f"{pts.size // 3} of {pts.size}" in str(record[0].message)


def test_ellipse_weights_reproduce_cauchy():
    """The trapezoid rule on the ellipse reproduces Cauchy's formula.

    The same three properties as the rectangle, on the ellipse inscribed in the
    same box (centre 2, semi-axes 1 and 1). Geometric convergence is the fourth:
    doubling the nodes squares the error, which is what a trapezoid rule does on
    an analytic periodic integrand and nothing else does.
    """
    errors = []
    for n_nodes in (16, 32):
        pts, wts = ellipse_contour_quad(Z_LO, Z_HI, n_nodes)
        # The trapezoid rule integrates the constant exactly: Σ dλ/dt over a
        # full period is a sum of roots of unity *(measured 5.7e-17)*.
        assert abs(wts.sum()) < 1.0e-15
        errors.append(max(abs(np.sum(wts / (pts - z)) - 1.0) for z in CAUCHY_INSIDE))

    # *(measured 1.1e-4 at 16 nodes and 1.2e-8 at 32, the square to within the
    # constant)*; the points 0.57 from the centre set the rate.
    assert errors[1] < 1.0e-7
    assert errors[0] / errors[1] > 1.0e3

    # No singularity anywhere near: round-off *(measured 1.8e-17 at 32)*.
    for z_out in CAUCHY_OUTSIDE:
        assert abs(np.sum(wts / (pts - z_out))) < 1.0e-13


def test_ellipse_rule_nests_under_doubling():
    """The nodes at n are every other node at 2n, at half the weight.

    This is the whole basis of the adaptive contour: a rule that did not nest
    would have to discard its solves at every refinement. Exact equality, not
    closeness — the shared nodes are the same floating-point numbers, so the
    matrices assembled at them are reused as they are.
    """
    pts, wts = ellipse_contour_quad(Z_LO, Z_HI, 16)
    pts2, wts2 = ellipse_contour_quad(Z_LO, Z_HI, 32)

    assert np.array_equal(pts2[::2], pts)
    assert np.array_equal(wts2[::2], 0.5 * wts)


# Inside the rectangle and outside its inscribed ellipse: (0.8)² + (0.8)² > 1
# in units of the semi-axes.
CORNER_ROOT = 2.8 + 0.8j


def test_adaptive_contour_converges_and_reuses_every_solve():
    """Doubling stops when the eigenvalues do, and no node is assembled twice.

    Three claims. The corner root is not reported: the contour is the ellipse.
    The reported change bounds the true error from above, because it is the
    error of the previous level and the levels converge geometrically
    *(measured: change 3.3e-12, true error 1.3e-13)*. And the builder is called
    once per node of the final contour plus once to size the probe — 33 calls
    for 32 nodes — which is the statement that nesting was used and not merely
    available.
    """
    want = np.array([1.5 - 0.3j, 2.5 + 0.4j])
    m_builder, _ = make_pencil([*want, CORNER_ROOT], nonlinear=True)
    calls = itertools.count()

    def counted(lam):
        next(calls)
        return m_builder(lam)

    res = beyn_modes_adaptive(counted, Z_LO, Z_HI, tol=1.0e-9)

    assert res.converged
    assert res.modes.n_modes == 2
    assert next(calls) == res.n_nodes + 1

    errors = np.array([np.min(np.abs(res.modes.eigenvalues - w)) for w in want])
    # 1e-9 is the tolerance asked for; the measured error is four decades
    # inside it because the stop is tested on the previous, coarser level.
    assert errors.max() < 1.0e-9
    assert res.change.max() <= 1.0e-9
    assert errors.max() <= res.change.max()


def test_adaptive_contour_matches_the_same_contour_from_scratch():
    """Halving the old sums and adding the new nodes is the same integral.

    Not bit-identical — the nodes are summed in a different order — but equal
    to round-off *(measured 1.4e-15 on the eigenvalues, 1.1e-16 on the
    cancellation)*. A wrong weight on either half would show at O(1).
    """
    m_builder, _ = make_pencil([1.5 - 0.3j, 2.5 + 0.4j], nonlinear=True)

    res = beyn_modes_adaptive(m_builder, Z_LO, Z_HI, tol=1.0e-9)

    pts, wts = ellipse_contour_quad(Z_LO, Z_HI, res.n_nodes)
    a0, a1, cancellation = contour_moments(
        m_builder, pts, wts, probe_matrix(N_DIM, 12, 0)
    )
    ref = beyn_poles(a0, a1, cancellation, Z_LO, Z_HI, contour="ellipse")

    assert np.allclose(res.modes.eigenvalues, ref.eigenvalues, atol=1.0e-13)
    assert abs(res.modes.cancellation - ref.cancellation) < 1.0e-13


def test_adaptive_contour_reports_an_empty_ellipse_as_converged():
    """Nothing inside, twice in a row, is an answer and not a failure."""
    m_builder, _ = make_pencil([CORNER_ROOT])

    res = beyn_modes_adaptive(m_builder, Z_LO, Z_HI)

    assert res.converged
    assert res.modes.n_modes == 0
    assert res.change.size == 0


def test_adaptive_contour_says_when_it_ran_out_of_nodes():
    """Hitting ``n_max`` returns the best result with ``converged`` False.

    A tolerance below round-off can never be met, so this is the path taken;
    the result is still the finest contour, and the flag is what tells the
    caller not to read ``change`` as a bound that was satisfied.
    """
    m_builder, _ = make_pencil([1.5 - 0.3j, 2.5 + 0.4j], nonlinear=True)

    res = beyn_modes_adaptive(m_builder, Z_LO, Z_HI, tol=1.0e-30, n_max=64)

    assert not res.converged
    assert res.n_nodes == 64
    assert res.modes.n_modes == 2


def test_adaptive_contour_still_raises_on_a_saturated_probe():
    """A level that cannot be reduced is skipped, but the last one is not.

    Coarse levels often show no rank gap and are passed over. A probe with
    fewer columns than there are eigenvalues shows none at any level, and that
    must surface as the same error the fixed contour raises.
    """
    m_builder, _ = make_pencil(
        [1.4 - 0.3j, 1.7 + 0.2j, 2.1 - 0.1j, 2.5 + 0.4j, 2.6 - 0.3j]
    )

    with pytest.raises(ValueError, match="saturated"):
        beyn_modes_adaptive(m_builder, Z_LO, Z_HI, n_probe=3, n_max=64)


def test_unknown_contour_is_rejected():
    """A misspelt contour must not fall through to the rectangle filter."""
    a0 = np.eye(4, 2, dtype=complex)
    with pytest.raises(ValueError, match="contour must be"):
        beyn_poles(a0, a0, 1.0, Z_LO, Z_HI, contour="circle")


# ── Interpolation route ─────────────────────────────────────────────────────


def test_faber_basis_is_what_it_claims_on_the_ellipse():
    """On λ(t) the basis is ``e^{ikt} + qᵏ e^{−ikt}``, and on a circle ``(u/a)ᵏ``.

    The closed form is derived from the Chebyshev–Joukowski identity, not from
    the recurrence, so agreement checks the recurrence and its k = 2 start,
    where ψ₀ enters as 2. A wrong start shifts every even degree at O(q).
    """
    z_lo, z_hi = 500.0 + 1.0j, 1100.0 + 50.0j
    a, b = 300.0, 24.5
    q = (a - b) / (a + b)
    t = 2.0 * np.pi * np.arange(32) / 32
    pts, _ = ellipse_contour_quad(z_lo, z_hi, 32)
    k = np.arange(20)[:, None]
    want = np.exp(1j * k * t) + q**k * np.exp(-1j * k * t)
    got = ellipse_faber_basis(pts, z_lo, z_hi, 20)
    # Degree 0 is 1 by convention; the closed form gives 2 there.
    assert np.array_equal(got[0], np.ones(32))
    # Round-off on O(1) values through 20 steps of a stable recurrence.
    assert np.allclose(got[1:], want[1:], atol=1.0e-12)

    # The circle: q = 0, monomials in units of the radius, no special case.
    lam = np.array([2.3 + 0.1j, 1.6 - 0.7j])
    assert np.allclose(
        ellipse_faber_basis(lam, Z_LO, Z_HI, 6),
        (lam - 2.0)[None, :] ** k[:6],
        atol=1e-15,
    )


def test_interpolation_matrix_is_well_conditioned_on_an_elongated_box():
    """A 600 × 49 box, the shape S5 gained on, at 16 and 64 nodes.

    *(measured: condition 2.55 and 2.28)*. A monomial basis in ``λ − centre``
    on the same nodes is beyond 1e16 by 64; 10 is four times the measured
    value and still says "a Fourier matrix", which is the claim.
    """
    z_lo, z_hi = 500.0 + 1.0j, 1100.0 + 50.0j
    for n in (16, 64):
        pts, _ = ellipse_contour_quad(z_lo, z_hi, n)
        assert np.linalg.cond(ellipse_faber_basis(pts, z_lo, z_hi, n).T) < 10.0


def test_interpolation_is_exact_on_a_polynomial_pencil():
    """A pencil linear in λ is fitted exactly, so only round-off is left.

    *(measured: fit error 2.0e-16 at the first doubling, eigenvalue error
    1.9e-15)*. The bounds are a decade above, and five decades below what a
    wrong basis or a mis-merged sample stack would leave.
    """
    want = np.array([1.5 - 0.3j, 2.5 + 0.4j])
    m_builder, _ = make_pencil(want)

    res = beyn_modes_adaptive(m_builder, Z_LO, Z_HI, interpolate=True)

    assert res.converged
    assert res.fit_error < 1.0e-14
    errors = [np.min(np.abs(res.modes.eigenvalues - w)) for w in want]
    assert max(errors) < 1.0e-13


def test_interpolated_route_converges_and_assembles_each_node_once():
    """The genuinely nonlinear pencil, through the polynomial.

    The builder is called exactly ``n_nodes`` times: the samples size the probe,
    so there is no extra call, unlike the direct route's ``n_nodes + 1``. The
    corner root stays out, as on the direct ellipse. *(measured: 32 nodes, error
    3.3e-13, change 7.3e-13, fit error 2.4e-13)*.
    """
    want = np.array([1.5 - 0.3j, 2.5 + 0.4j])
    m_builder, _ = make_pencil([*want, CORNER_ROOT], nonlinear=True)
    calls = itertools.count()

    def counted(lam):
        next(calls)
        return m_builder(lam)

    res = beyn_modes_adaptive(counted, Z_LO, Z_HI, tol=1.0e-9, interpolate=True)

    assert res.converged
    assert res.modes.n_modes == 2
    assert next(calls) == res.n_nodes
    errors = np.array([np.min(np.abs(res.modes.eigenvalues - w)) for w in want])
    # The tolerance asked for; measured four decades inside it.
    assert errors.max() < 1.0e-9
    assert np.isfinite(res.fit_error)


def test_interpolated_route_rejects_a_noise_eigenvalue_by_the_stopping_test():
    """A pole 0.05 outside the ellipse, and a contour on the polynomial too fine.

    At 1024 nodes the pole's weight on the contour is ``1.05^−1024 ≈ 2e-22``,
    below round-off, but a direction at 5e-13 stays in A₀. On M it lies
    within PLATEAU_GAP of the plateau and is dropped. On the 32-node
    polynomial the plateau is lower (2e-14), the direction clears the gap, and
    the level reports a third eigenvalue at 2.9995+0.0072j *(measured)*. The
    first half shows that trap is real; the second shows the adaptive driver
    does not fall into it, because the noise eigenvalue moves between levels
    (2.9795, 2.9995, 2.9973 at 768, 1024, 2048 fine nodes) and two levels must
    agree.
    """
    want = np.array([1.5 - 0.3j, 2.5 + 0.4j])
    m_builder, _ = make_pencil([*want, 3.05 + 0.0j], nonlinear=True)

    pts, _ = ellipse_contour_quad(Z_LO, Z_HI, 32)
    poly = _EllipseInterpolant(pts, np.array([m_builder(z) for z in pts]), Z_LO, Z_HI)
    fine, fine_w = ellipse_contour_quad(Z_LO, Z_HI, 1024)
    a0, a1, canc = contour_moments(poly, fine, fine_w, probe_matrix(N_DIM, 12, 0))
    one_level = beyn_poles(a0, a1, canc, Z_LO, Z_HI, contour="ellipse")
    assert one_level.n_modes == 3

    res = beyn_modes_adaptive(
        m_builder, Z_LO, Z_HI, tol=1.0e-9, interpolate=True, n_max=512, n_fine=1024
    )
    assert res.converged
    assert res.modes.n_modes == 2
    errors = [np.min(np.abs(res.modes.eigenvalues - w)) for w in want]
    assert max(errors) < 1.0e-9


def test_interpolation_memory_cap_refuses_a_first_level_that_does_not_fit():
    """Too little memory for ``n_start`` samples is an error, not a silent cut."""
    m_builder, _ = make_pencil([1.5 - 0.3j])
    with pytest.raises(ValueError, match="interp_memory"):
        beyn_modes_adaptive(m_builder, Z_LO, Z_HI, interpolate=True, interp_memory=1e3)


def test_interpolation_memory_cap_ends_the_doubling():
    """A budget that holds 32 samples and not 64 makes 32 the last level.

    The budget is exactly the peak at 32, so the boundary itself is tested:
    ``≤`` admits it, and 64 is refused. ``tol = 1e-30`` cannot be met, so the
    result must come back unconverged, at 32, rather than as an error.
    """
    m_builder, _ = make_pencil([1.5 - 0.3j, 2.5 + 0.4j], nonlinear=True)
    budget = INTERP_PEAK_OVER_SAMPLES * 16.0 * 32 * N_DIM**2

    res = beyn_modes_adaptive(
        m_builder, Z_LO, Z_HI, tol=1.0e-30, interpolate=True, interp_memory=budget
    )

    assert res.n_nodes == 32
    assert not res.converged
    assert res.modes.n_modes == 2


def test_interpolation_needs_a_fine_contour_at_least_as_fine_as_the_sampling():
    """A polynomial integrated more coarsely than it was sampled wastes the fit."""
    m_builder, _ = make_pencil([1.5 - 0.3j])
    with pytest.raises(ValueError, match="n_fine"):
        beyn_modes_adaptive(m_builder, Z_LO, Z_HI, interpolate=True, n_fine=128)


def test_fixed_ellipse_is_the_adaptive_contour_at_one_level():
    """``beyn_modes(contour="ellipse")`` integrates ``4·n_quad_per_side`` nodes.

    Same nodes, same weights, same probe as the adaptive driver's level at that
    count, so the eigenvalues agree to round-off *(measured 1.4e-15 for the
    nested sum against the direct one)*; and the corner root is out.
    """
    want = np.array([1.5 - 0.3j, 2.5 + 0.4j])
    m_builder, _ = make_pencil([*want, CORNER_ROOT], nonlinear=True)

    fixed = beyn_modes(m_builder, Z_LO, Z_HI, n_quad_per_side=8, contour="ellipse")
    pts, wts = ellipse_contour_quad(Z_LO, Z_HI, 32)
    a0, a1, canc = contour_moments(m_builder, pts, wts, probe_matrix(N_DIM, 12, 0))
    ref = beyn_poles(a0, a1, canc, Z_LO, Z_HI, contour="ellipse")

    assert fixed.n_modes == 2
    assert np.allclose(fixed.eigenvalues, ref.eigenvalues, atol=1.0e-13)
    with pytest.raises(ValueError, match="contour must be"):
        beyn_modes(m_builder, Z_LO, Z_HI, contour="circle")
