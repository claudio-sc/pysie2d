"""Beyn's contour-integral method for nonlinear eigenvalue problems.

Given a matrix-valued function M(λ) that is holomorphic on and inside a closed
contour, the eigenvalues inside — the λ where M(λ) is singular — are recovered
from two contour integrals of M(λ)⁻¹ against a random probe matrix:

    A₀ = (1/2πi) ∮ M(λ)⁻¹ V dλ,    A₁ = (1/2πi) ∮ λ M(λ)⁻¹ V dλ.

A₀ has rank equal to the number of eigenvalues inside (with multiplicity), and
its SVD reduces A₁ to a small standard eigenvalue problem whose spectrum *is*
that set. No initial guess, no scan, no root bracketing: the contour decides
what is found.

Reference: W.-J. Beyn, "An integral method for solving nonlinear eigenvalue
problems", Linear Algebra Appl. 436, 3839-3863 (2012), Algorithm 3.1.

This module is deliberately free of electromagnetics. It sees only a callable
``λ → M(λ)`` and a rectangle, which is what makes it testable against pencils
with known spectra — a bug here cannot hide behind a convention error in the
BIE assembly, and vice versa.
"""

import warnings
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.linalg import lu_factor, lu_solve

# Minimum drop in the singular-value spectrum that counts as the rank gap.
# Below this the spectrum is a slow decay rather than a cliff — see _detect_rank.
RANK_GAP_FLOOR = 1.0e3

# Below this, A₀ has cancelled to nothing and the contour is empty; above it a
# flat spectrum means the probe is saturated instead.
#
# The two sides scale differently, which is what makes a threshold possible at
# all: over an empty contour the surviving A₀ is pure quadrature error and falls
# geometrically with the node count, while a pole inside contributes a genuine
# residue that does not move.  *(measured on the m=3 circle: empty boxes give
# 6.0e-15 … 5.2e-7 — the worst being a coarse 24-node contour with poles just
# outside — against 3.09e-2 and 4.94e-2 for boxes containing one and two modes,
# both unchanged between 24 and 48 nodes.  Synthetic pencils cancel far harder
# still, to 2.9e-16.)*  1e-4 sits ~200x above the worst empty case and ~300x
# below the weakest populated one.  A tighter threshold is not safer: at 1e-10 a
# coarse contour over an empty box raises a spurious saturation error.
EMPTY_CANCELLATION = 1.0e-4


@dataclass(frozen=True)
class BeynModes:
    """Eigenvalues of M(λ) inside a contour, with their rank diagnostics.

    Attributes:
        eigenvalues: complex (K,) eigenvalues inside the contour, sorted by
            real part then imaginary part.
        vectors: complex (N, K) approximate null-space vectors of M(λ_k), unit
            columns; column k belongs to ``eigenvalues[k]``.
        sv_ratio: float (n_probe,) singular values of A₀ divided by the
            largest. The rank gap is visible here; a spectrum with no cliff
            means an empty contour.
        max_gap: float largest ratio between consecutive singular values.
        rank: int detected rank of A₀ — the eigenvalue count before the
            in-contour filter is applied.
        cancellation: float ``‖A₀‖ / Σⱼ|wⱼ|‖xⱼ‖``, how much the contour
            integral cancelled. Near machine epsilon means the integrand was
            analytic inside, i.e. an empty contour.
    """

    eigenvalues: np.ndarray
    vectors: np.ndarray
    sv_ratio: np.ndarray
    max_gap: float
    rank: int
    cancellation: float

    @property
    def n_modes(self) -> int:
        """Number of eigenvalues retained inside the contour."""
        return int(self.eigenvalues.size)


def rect_contour_quad(
    z_lo: complex,
    z_hi: complex,
    n_per_side: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Gauss-Legendre quadrature on a counter-clockwise rectangular contour.

    The weights returned already carry the parametrisation Jacobian and the
    ``1/(2πi)`` of Cauchy's formula, so a moment matrix is a bare sum:
    ``A₀ = Σⱼ w[j]·x[j]``. Folding the normalisation in here is what makes the
    Cauchy test in the suite a test of *this* function rather than of its
    caller.

    Gauss-Legendre rather than the trapezoid: the integrand is analytic on each
    straight edge but not periodic along it, so the trapezoid rule's spectral
    accuracy does not apply while Gauss-Legendre's does.

    Args:
        z_lo: Bottom-left corner of the rectangle.
        z_hi: Top-right corner of the rectangle.
        n_per_side: Nodes per edge; the contour carries ``4·n_per_side``.

    Returns:
        pts: complex (4·n_per_side,) quadrature nodes on the contour.
        wts: complex (4·n_per_side,) weights, including ``dλ/dt`` and 1/(2πi).

    Raises:
        ValueError: If the rectangle is degenerate (zero width or height).
    """
    if z_hi.real <= z_lo.real or z_hi.imag <= z_lo.imag:
        raise ValueError(
            f"z_hi must be strictly above and right of z_lo; "
            f"got z_lo={z_lo}, z_hi={z_hi}"
        )

    t, w = leggauss(n_per_side)

    # Counter-clockwise, so the winding number of an enclosed point is +1 and
    # the Cauchy normalisation comes out with the sign the formula assumes.
    corners = [
        z_lo,
        complex(z_hi.real, z_lo.imag),
        z_hi,
        complex(z_lo.real, z_hi.imag),
    ]
    edges = list(zip(corners, corners[1:] + corners[:1], strict=True))

    pts_list, wts_list = [], []
    for z_s, z_e in edges:
        dz = z_e - z_s
        # λ(t) = z_s + dz·(t+1)/2 on t ∈ [-1, 1], so dλ/dt = dz/2.
        pts_list.append(z_s + dz * 0.5 * (t + 1.0))
        wts_list.append(w * (0.5 * dz) / (2.0j * np.pi))

    return np.concatenate(pts_list), np.concatenate(wts_list)


def probe_matrix(n_dim: int, n_probe: int, rng_seed: int) -> np.ndarray:
    """Complex Gaussian probe matrix V, the random right-hand sides.

    Beyn's rank argument holds for almost every V; the seed is exposed so that
    seed-independence is a property the suite can check, not a knob to tune.

    Args:
        n_dim: Rows — the dimension of M(λ).
        n_probe: Columns; must exceed the number of eigenvalues sought.
        rng_seed: Seed for the generator.

    Returns:
        complex (n_dim, n_probe) probe matrix with unit-variance entries.
    """
    rng = np.random.default_rng(rng_seed)
    real = rng.standard_normal((n_dim, n_probe))
    imag = rng.standard_normal((n_dim, n_probe))
    return (real + 1j * imag) / np.sqrt(2.0)


def contour_moments(
    m_builder: Callable[[complex], np.ndarray],
    pts: np.ndarray,
    wts: np.ndarray,
    v_probe: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Accumulate the moment matrices A₀ and A₁ along the contour.

    One LU factorisation per node serves both moments and all probe columns,
    which is the whole cost of the method: ``4·n_per_side`` assemblies and
    factorisations, independent of how many eigenvalues are inside.

    A node whose factorisation fails is skipped with a warning rather than
    aborting: it means the contour passed essentially through an eigenvalue,
    and the remaining nodes still integrate — but the result is then suspect,
    hence the warning is not silenced.

    Args:
        m_builder: ``λ → complex (N, N)`` matrix function.
        pts: complex (n_quad,) contour nodes.
        wts: complex (n_quad,) contour weights from ``rect_contour_quad``.
        v_probe: complex (N, p) probe matrix.

    Returns:
        a0: complex (N, p) zeroth moment Σⱼ wⱼ M(λⱼ)⁻¹V.
        a1: complex (N, p) first moment Σⱼ wⱼ λⱼ M(λⱼ)⁻¹V.
        cancellation: ``‖A₀‖ / Σⱼ|wⱼ|‖xⱼ‖``, the fraction of the integrand
            that survived the integration. The triangle inequality bounds it by
            1; an analytic integrand cancels it to machine epsilon. This is the
            only thing that distinguishes an empty contour from one whose poles
            outnumber the probe columns — both give a flat singular spectrum.
    """
    a0 = np.zeros(v_probe.shape, dtype=complex)
    a1 = np.zeros(v_probe.shape, dtype=complex)
    integrand_scale = 0.0

    n_failed = 0
    for lam, wt in zip(pts, wts, strict=True):
        try:
            lu, piv = lu_factor(m_builder(lam))
            x_sol = lu_solve((lu, piv), v_probe)
        except (np.linalg.LinAlgError, ValueError):
            n_failed += 1
            continue
        a0 += wt * x_sol
        a1 += wt * lam * x_sol
        integrand_scale += abs(wt) * float(np.linalg.norm(x_sol))

    if n_failed:
        warnings.warn(
            f"{n_failed} of {pts.size} contour points failed to factorise; the "
            "contour may pass through an eigenvalue. Move or resize it.",
            RuntimeWarning,
            stacklevel=2,
        )

    cancellation = (
        float(np.linalg.norm(a0)) / integrand_scale if integrand_scale > 0 else 0.0
    )
    return a0, a1, cancellation


def _detect_rank(sv_ratio: np.ndarray, rank_tol: float) -> tuple[int, float]:
    """Rank of A₀ from the cliff in its singular-value spectrum.

    Exactly, A₀ has rank K = the number of eigenvalues inside the contour;
    quadrature error lifts the remaining singular values off zero into a noise
    floor. The rank is therefore read off a *gap*, not off an absolute
    threshold, and specifically off the **last** gap clearing
    ``RANK_GAP_FLOOR`` whose retained value is still above ``rank_tol``.

    Last, not largest. Two poles whose residues differ by 10⁶ produce two
    cliffs — one between the poles and one down to the noise floor — and the
    first can be the taller of the two *(measured: gaps of 1.1e6 then 2.1e5 for
    a 10⁶ residue ratio, where argmax returns rank 1 and loses the weak pole)*.
    Everything past the last cliff is flat noise, which is the definition of
    the floor; a drop between two genuine poles never is.

    A spectrum with no qualifying gap has no cliff, hence no meromorphic
    content: rank 0, an empty contour. In a homogeneous background that is a
    statement about the contour, not about analyticity — M(λ) is holomorphic on
    any admissible rectangle there, so a flat spectrum cannot be blamed on a
    branch cut.

    Args:
        sv_ratio: float (p,) singular values divided by the largest.
        rank_tol: Relative floor below which a singular value is noise.

    Returns:
        rank: Detected rank, possibly 0.
        max_gap: The largest consecutive ratio, reported as a diagnostic.
    """
    # Guard the log against an exactly-zero tail; the ratio only has to be
    # comparable against RANK_GAP_FLOOR, and a zero means "infinite gap".
    safe = np.maximum(sv_ratio, 1.0e-300)
    gaps = safe[:-1] / safe[1:]
    if gaps.size == 0:
        return (1, 1.0) if sv_ratio[0] > 0 else (0, 1.0)

    max_gap = float(gaps.max())
    # A gap qualifies only if the value it retains is above the noise floor;
    # that is what stops a chance drop *inside* the floor from being read as
    # the cliff, and it is why the last qualifying gap is safe to take.
    qualifies = (gaps >= RANK_GAP_FLOOR) & (sv_ratio[:-1] > rank_tol)
    if not qualifies.any():
        return 0, max_gap
    return int(np.flatnonzero(qualifies)[-1]) + 1, max_gap


def beyn_poles(
    a0: np.ndarray,
    a1: np.ndarray,
    cancellation: float,
    z_lo: complex,
    z_hi: complex,
    rank_tol: float = 1.0e-8,
) -> BeynModes:
    """Reduce the moment matrices to the eigenvalues inside the contour.

    Steps 4-6 of Beyn 2012 Alg. 3.1: SVD of A₀, project A₁ onto the numerical
    range, and solve the resulting K×K standard eigenvalue problem.

    Args:
        a0: complex (N, p) zeroth moment.
        a1: complex (N, p) first moment.
        cancellation: The ratio returned by ``contour_moments``; required, not
            optional, because without it a saturated probe is indistinguishable
            from an empty contour and would be reported as "no modes found".
        z_lo: Bottom-left corner of the contour rectangle.
        z_hi: Top-right corner.
        rank_tol: Relative singular-value floor for rank detection.

    Returns:
        The eigenvalues inside the rectangle and their rank diagnostics.

    Raises:
        ValueError: If the singular spectrum has no cliff while A₀ is large —
            the eigenvalues inside outnumber the probe columns, so the probe
            cannot span the eigenspace and the result would be silently
            incomplete.
    """
    u, s, vh = np.linalg.svd(a0, full_matrices=False)
    sv_ratio = s / s[0] if s[0] > 0 else s
    k_rank, max_gap = _detect_rank(sv_ratio, rank_tol)

    if k_rank == 0:
        # No cliff. Either nothing is inside, or everything is: a probe with
        # fewer columns than there are eigenvalues has no small singular values
        # left to form a gap against.
        if cancellation > EMPTY_CANCELLATION:
            # ASCII only: an exception message can be printed to a cp1252
            # console, where a subscript or a lambda raises UnicodeEncodeError
            # and buries the real error. Docstrings and comments stay Unicode.
            raise ValueError(
                f"singular spectrum has no rank gap but A0 did not cancel "
                f"(cancellation {cancellation:.2e} > {EMPTY_CANCELLATION:.0e}): "
                f"the {a0.shape[1]}-column probe is saturated and eigenvalues "
                "are being lost. Increase n_probe or shrink the contour."
            )
        return BeynModes(
            eigenvalues=np.array([], dtype=complex),
            vectors=np.zeros((a0.shape[0], 0), dtype=complex),
            sv_ratio=sv_ratio,
            max_gap=max_gap,
            rank=0,
            cancellation=cancellation,
        )

    u1 = u[:, :k_rank]
    s1 = s[:k_rank]
    v1 = vh[:k_rank, :].conj().T

    # B = U₁ᴴ A₁ V₁ Σ₁⁻¹ — the inverse post-multiplies, i.e. column j is
    # divided by s₁[j] (Beyn 2012 Alg. 3.1, step 5). Pre-multiplying gives a
    # matrix similar through Σ₁, so the eigenvalues survive the mistake but
    # the eigenvectors do not.
    b_eig = (u1.conj().T @ a1 @ v1) / s1[np.newaxis, :]
    eigvals, w = np.linalg.eig(b_eig)

    inside = (
        (eigvals.real > z_lo.real)
        & (eigvals.real < z_hi.real)
        & (eigvals.imag > z_lo.imag)
        & (eigvals.imag < z_hi.imag)
    )
    idx = np.where(inside)[0]
    order = np.lexsort((eigvals[idx].imag, eigvals[idx].real))
    idx = idx[order]

    # Back to the full space: the reduced eigenvectors lift through U₁.
    vecs = u1 @ w[:, idx]
    norms = np.linalg.norm(vecs, axis=0, keepdims=True)
    vecs = vecs / np.where(norms > 0, norms, 1.0)

    return BeynModes(
        eigenvalues=eigvals[idx],
        vectors=vecs,
        sv_ratio=sv_ratio,
        max_gap=max_gap,
        rank=k_rank,
        cancellation=cancellation,
    )


def beyn_modes(
    m_builder: Callable[[complex], np.ndarray],
    z_lo: complex,
    z_hi: complex,
    *,
    n_quad_per_side: int = 12,
    n_probe: int = 12,
    rank_tol: float = 1.0e-8,
    rng_seed: int = 0,
) -> BeynModes:
    """All eigenvalues of M(λ) inside a rectangle, by Beyn's method.

    Args:
        m_builder: ``λ → complex (N, N)``, holomorphic on and inside the
            rectangle. Poles of M itself invalidate the rank argument.
        z_lo: Bottom-left corner of the search rectangle.
        z_hi: Top-right corner.
        n_quad_per_side: Gauss-Legendre nodes per edge, so ``4×`` this many
            matrix assemblies and factorisations.
        n_probe: Probe columns; must exceed the eigenvalue count inside.
        rank_tol: Relative singular-value floor for rank detection.
        rng_seed: Probe-matrix seed. Results must not depend on it.

    Returns:
        The eigenvalues inside the rectangle and their rank diagnostics.

    Raises:
        ValueError: If the eigenvalues inside outnumber ``n_probe``.
    """
    pts, wts = rect_contour_quad(z_lo, z_hi, n_quad_per_side)
    n_dim = m_builder(pts[0]).shape[0]
    v_probe = probe_matrix(n_dim, n_probe, rng_seed)
    a0, a1, cancellation = contour_moments(m_builder, pts, wts, v_probe)
    return beyn_poles(a0, a1, cancellation, z_lo, z_hi, rank_tol)


@dataclass(frozen=True)
class RefinedMode:
    """One eigenvalue after bordered-Newton polishing.

    Attributes:
        eigenvalue: The refined λ, or the unrefined input when the iteration
            made no accepted step.
        converged: Whether the final step fell below the tolerance.
        step: Size of the last accepted step, ``inf`` if none was taken.
        cond_jacobian: Condition number of the first bordered Jacobian. A
            degenerate eigenvalue makes it singular — the bordered system
            assumes a one-dimensional null space — so a value above ~1e12
            means the result is not to be trusted, not that it failed loudly.
    """

    eigenvalue: complex
    converged: bool
    step: float
    cond_jacobian: float


def newton_refine(
    m_builder: Callable[[complex], np.ndarray],
    dm_builder: Callable[[complex], np.ndarray],
    lam0: complex,
    v0: np.ndarray,
    z_lo: complex,
    z_hi: complex,
    *,
    tol: float = 1.0e-9,
    max_iter: int = 30,
) -> RefinedMode:
    """Polish one eigenpair (λ, v) with the bordered-system Newton method.

    Each iteration solves the (N+1)×(N+1) system

        | M(λ)   M'(λ)v | |Δv|      | M(λ)v     |
        | cᴴ     0      | |Δλ|  = − | cᴴv − 1   |

    for the pair jointly. The anchor ``c = conj(v0)`` is the incoming vector
    and stays fixed for the whole iteration: it pins the normalisation and, by
    being different for every eigenvalue, stops a nearby one from capturing the
    iterate.

    This is insurance rather than accuracy. On a well-resolved simple pole the
    Beyn estimate is already converged and this makes one no-op step; its value
    is recovering from a contour too coarse to place the pole accurately.

    Args:
        m_builder: ``λ → complex (N, N)``.
        dm_builder: ``λ → complex (N, N)``, the derivative dM/dλ.
        lam0: Initial eigenvalue estimate.
        v0: complex (N,) initial null-space vector.
        z_lo: Bottom-left corner of the contour the estimate came from.
        z_hi: Top-right corner; both bound the escape check.
        tol: Convergence threshold on the step size, in units of λ.
        max_iter: Iteration cap.

    Returns:
        The refined eigenvalue and its convergence diagnostics.
    """
    c_anchor = v0.conj()
    scale = c_anchor @ v0  # ‖v0‖² for a unit-norm input, so real and positive
    v = v0 / scale if abs(scale) > 0 else v0.copy()
    lam = lam0
    n_dim = v.size

    def _residual(lam_c: complex, v_c: np.ndarray) -> float:
        mv = m_builder(lam_c) @ v_c
        anchor = c_anchor @ v_c - 1.0
        return float(np.sqrt(np.linalg.norm(mv) ** 2 + abs(anchor) ** 2))

    # 20 % of the average half-side. An iterate this far outside the rectangle
    # is chasing something the contour never claimed to contain.
    escape_tol = 0.1 * ((z_hi.real - z_lo.real) + (z_hi.imag - z_lo.imag))

    res_cur = _residual(lam, v)
    cond_j = float("inf")
    step = float("inf")
    converged = False

    for it in range(max_iter):
        m_k = m_builder(lam)
        mpv = dm_builder(lam) @ v

        jac = np.empty((n_dim + 1, n_dim + 1), dtype=complex)
        jac[:n_dim, :n_dim] = m_k
        jac[:n_dim, n_dim] = mpv
        jac[n_dim, :n_dim] = c_anchor
        jac[n_dim, n_dim] = 0.0

        if it == 0:
            # Reported, not acted on: a degenerate eigenvalue must be flagged
            # rather than silently returned as if it had been refined.
            cond_j = float(np.linalg.cond(jac))

        rhs = np.empty(n_dim + 1, dtype=complex)
        rhs[:n_dim] = -(m_k @ v)
        rhs[n_dim] = -(c_anchor @ v - 1.0)

        try:
            delta = np.linalg.solve(jac, rhs)
        except np.linalg.LinAlgError:
            break

        dv, dlam = delta[:n_dim], complex(delta[n_dim])

        # Backtracking line search on ‖F‖: the bordered system is exact only to
        # first order, and a full step can overshoot into a neighbouring basin.
        alpha = 1.0
        decreased = False
        for _ in range(12):
            lam_try = lam + alpha * dlam
            v_try = v + alpha * dv
            res_try = _residual(lam_try, v_try)
            if res_try < res_cur:
                decreased = True
                break
            alpha *= 0.5
        if not decreased:
            break

        lam, v, res_cur = lam_try, v_try, res_try
        step = float(abs(alpha * dlam))
        if step < tol:
            converged = True
            break

        dist_re = max(z_lo.real - lam.real, 0.0, lam.real - z_hi.real)
        dist_im = max(z_lo.imag - lam.imag, 0.0, lam.imag - z_hi.imag)
        if max(dist_re, dist_im) > escape_tol:
            break

    return RefinedMode(
        eigenvalue=lam, converged=converged, step=step, cond_jacobian=cond_j
    )
