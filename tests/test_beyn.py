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

import numpy as np
import pytest

from pysie2d.beyn import (
    EMPTY_CANCELLATION,
    RANK_GAP_FLOOR,
    beyn_modes,
    beyn_poles,
    contour_moments,
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
