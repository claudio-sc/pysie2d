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

This module deliberately knows nothing about where M comes from. It sees only
a callable ``λ → M(λ)`` and a rectangle, which is what makes it testable against
pencils with known spectra — a bug here cannot hide behind an error in the code
that assembles M, and vice versa.

The measured numbers quoted below were taken with this package's own
boundary-integral matrix as the workload; they say how the method behaves on
one dense, expensive-to-assemble M, not on every M.
"""

import os
import warnings
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.linalg import lu_factor, lu_solve

# Minimum drop in the singular-value spectrum that counts as the rank gap.
# Below this the spectrum is a slow decay rather than a cliff — see _detect_rank.
RANK_GAP_FLOOR = 1.0e3

# Largest ratio between consecutive singular values of A₀ that still counts as
# "the same level" when the round-off plateau is traced up from the bottom.
#
# Round-off leaves A₀ with a tail of singular values that carry nothing. Its
# height is not known in advance — the conditioning of M multiplies it — and it
# is not flat either, so it is found by its shape: starting from the smallest
# value, the plateau extends for as long as each value is within PLATEAU_GAP of
# the one below it.  *(measured, docs/design/studies/beyn S1, 534 runs over
# seven problems, ten contour sizes, five probe seeds and two probe counts:
# false eigenvalues inside the box appear at a gap of 5 (2 runs) and at none of
# 10, 20, 30, 50, 100; the runs that lose a digit against the best rule number
# 2, 13, 15, 23, 28 at those five.  The former fixed 1e-8 floor loses a digit
# in 142.)*  20 is four times the gap at which a round-off direction was first
# mistaken for a pole. A false mode is the worse error of the two, hence the
# choice on that side.
PLATEAU_GAP = 20.0

# A singular value below this many units of ε/cancellation is round-off whatever
# its spacing from its neighbours. ε/cancellation is the rounding of the contour
# sum relative to what survives it *(same study: the smallest singular value
# sits at 0.25–0.73 units on this package's boundary-integral matrix and at
# 10–17 on synthetic pencils with a wide spread of entries)*. This is the
# backstop for a tail that is exactly zero, which has no shape to trace.
ROUNDOFF_MARGIN = 10.0

# Below this, A₀ has cancelled to nothing and the contour is empty; above it a
# flat spectrum means the probe is saturated instead.
#
# The two sides scale differently, which is what makes a threshold possible at
# all: over an empty contour the surviving A₀ is pure quadrature error and falls
# geometrically with the node count, while a pole inside contributes a genuine
# residue that does not move.  *(measured on the test suite's circle, m=3: empty
# boxes give 6.0e-15 … 5.2e-7 — the worst being a coarse 24-node contour with
# poles just outside — against 3.09e-2 and 4.94e-2 for boxes containing one and
# two eigenvalues,
# both unchanged between 24 and 48 nodes.  Synthetic pencils cancel far harder
# still, to 2.9e-16.)*  1e-4 sits ~200x above the worst empty case and ~300x
# below the weakest populated one.  A tighter threshold is not safer: at 1e-10 a
# coarse contour over an empty box raises a spurious saturation error.
EMPTY_CANCELLATION = 1.0e-4

# Peak memory one contour node holds, as a multiple of the M(λ) it assembles.
# *(measured, docs/design/performance.md §5.1: serial peak is 3.56× the matrix —
# the assembly's temporary arrays plus `lu_factor`'s non-overwriting copy —
# and each added worker costs the same again, 8 workers reaching 28.2×.)*
# Threading is a memory-for-time trade, so the worker count
# is derived from a budget rather than fixed: at nn = 4000 a hard-coded 8 turns
# a working run into a 29 GB OOM (performance.md §5.5 C).
NODE_PEAK_OVER_MATRIX = 3.6

# The budget that cap is taken against. Deliberately a fixed conservative
# number and not a query of free memory: a worker count that changed with what
# else the machine happened to be running would make the LU-failure count, and
# hence the warning, irreproducible between two identical calls.  Callers who
# know their machine pass ``max_workers``.
THREAD_MEMORY_BUDGET = 2.0e9

# A **memory** cap, not a scaling one. When the builder spends its time in
# compiled code that releases the GIL, the scaling is real well past here
# *(measured on 10 cores, performance.md §3.1: 5.73× at 8 workers and still
# gaining at 6.40× on 10)*, but each worker costs another NODE_PEAK_OVER_MATRIX
# of memory and §5.1 puts the 8-thread ceiling at nn ≈ 2500. Buying the last
# 11 % would move that ceiling down again.
MAX_CONTOUR_WORKERS = 8

# Bytes the interpolation route may hold for M's samples and the polynomial
# through them, against which its node count is capped. Fixed and conservative
# for the same reason as THREAD_MEMORY_BUDGET: a cap that moved with free memory
# would make the node count, and hence the answer, depend on what else ran.
INTERPOLATION_MEMORY_BUDGET = 2.0e9

# Copies of the sample stack the fit holds at its peak: the samples, the
# coefficients, and the Fortran-ordered copy LAPACK makes of the right-hand side
# *(measured with tracemalloc at N = 100, n = 32: peak 3.0× the stack)*.
INTERP_PEAK_OVER_SAMPLES = 3.0


def _worker_count(n_dim: int, n_quad: int, max_workers: int | None) -> int:
    """Threads to run the contour on, capped by the memory budget.

    Args:
        n_dim: Dimension N of M(λ); the matrix is ``16·N²`` bytes complex128.
        n_quad: Contour nodes — never spawn more workers than there is work.
        max_workers: Caller's override, used as given (still floored at 1).

    Returns:
        The number of worker threads; 1 means run serially in this thread.
    """
    if max_workers is not None:
        return max(1, min(int(max_workers), n_quad))

    matrix_bytes = 16.0 * float(n_dim) ** 2
    budget = int(THREAD_MEMORY_BUDGET / (NODE_PEAK_OVER_MATRIX * matrix_bytes))
    return max(1, min(budget, MAX_CONTOUR_WORKERS, os.cpu_count() or 1, n_quad))


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
        rank: int number of singular values of A₀ kept — the eigenvalue count
            before the in-contour filter is applied. It counts the poles
            inside, the poles outside whose weight on this contour is still
            above round-off, and, on a coarse contour, directions that carry
            quadrature error; it is therefore not a mode count.
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


def ellipse_contour_quad(
    z_lo: complex,
    z_hi: complex,
    n_nodes: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Trapezoid rule on the counter-clockwise ellipse inscribed in a rectangle.

    The ellipse touches the four sides, so it stays inside whatever region the
    rectangle was checked against, and it leaves the corners out: an eigenvalue
    in a corner is outside this contour.

    The trapezoid rule, because the integrand is analytic *and periodic* along
    a closed smooth curve, which is the case where it converges geometrically.
    Two things follow that the rectangle does not have. There are no corners,
    and at equal node count the error is about that of the rectangle with twice
    the nodes *(measured, docs/design/studies/beyn S2, eleven modes in a
    600 × 49 box: worst mode 1.1e-1 against 4.9e-6 at 128 nodes; median 5.8e-11
    against 3.0e-12 at 64)*. And the rule **nests**: with the first node fixed
    at angle zero, the nodes at ``n`` are every other node at ``2n`` with half
    the weight, so a contour can be refined without discarding a single solve
    (:func:`beyn_modes_adaptive`).

    Args:
        z_lo: Bottom-left corner of the bounding rectangle.
        z_hi: Top-right corner of the bounding rectangle.
        n_nodes: Nodes on the ellipse.

    Returns:
        pts: complex (n_nodes,) quadrature nodes, the first at the rightmost
            point of the ellipse.
        wts: complex (n_nodes,) weights, including ``dλ/dt`` and 1/(2πi).

    Raises:
        ValueError: If the rectangle is degenerate (zero width or height).
    """
    if z_hi.real <= z_lo.real or z_hi.imag <= z_lo.imag:
        raise ValueError(
            f"z_hi must be strictly above and right of z_lo; "
            f"got z_lo={z_lo}, z_hi={z_hi}"
        )

    centre = 0.5 * (z_lo + z_hi)
    half = 0.5 * (z_hi - z_lo)
    t = 2.0 * np.pi * np.arange(n_nodes) / n_nodes

    # λ(t) = centre + a·cos t + i·b·sin t, counter-clockwise. The trapezoid
    # weight 2π/n times 1/(2πi) leaves 1/(i·n) on dλ/dt.
    pts = centre + half.real * np.cos(t) + 1j * half.imag * np.sin(t)
    dlam = -half.real * np.sin(t) + 1j * half.imag * np.cos(t)
    return pts, dlam / (1j * n_nodes)


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
    *,
    max_workers: int | None = None,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Accumulate the moment matrices A₀ and A₁ along the contour.

    One LU factorisation per node serves both moments and all probe columns,
    which is the whole cost of the method: ``4·n_per_side`` assemblies and
    factorisations, independent of how many eigenvalues are inside.

    The nodes are independent, so they run on a thread pool. This pays off
    when ``m_builder`` spends its time in compiled code that **releases the
    GIL**: on this package's assembly, where one special-function call is
    97.9 % of the cost at complex λ, 8 workers gave 5.36× *(measured,
    docs/design/performance.md §5.1)*. A pure-Python builder gains nothing. Threads,
    not processes: the arrays are shared and nothing is pickled, and
    `multiprocessing` measured strictly worse. This changes **no number** —
    the same calls, accumulated in the same contour order — which is what makes
    it the lowest-risk speedup available on this code.

    A node whose factorisation fails is skipped with a warning rather than
    aborting: it means the contour passed essentially through an eigenvalue,
    and the remaining nodes still integrate — but the result is then suspect,
    hence the warning is not silenced.

    Args:
        m_builder: ``λ → complex (N, N)`` matrix function.
        pts: complex (n_quad,) contour nodes.
        wts: complex (n_quad,) contour weights from ``rect_contour_quad``.
        v_probe: complex (N, p) probe matrix.
        max_workers: Threads to run the nodes on. ``None`` derives the count
            from ``THREAD_MEMORY_BUDGET``, because each worker holds another
            ~3.6 matrices: a fixed 8 turns a working ``nn = 4000`` run into a
            29 GB OOM. 1 runs serially in the calling thread.

    Returns:
        a0: complex (N, p) zeroth moment Σⱼ wⱼ M(λⱼ)⁻¹V.
        a1: complex (N, p) first moment Σⱼ wⱼ λⱼ M(λⱼ)⁻¹V.
        cancellation: ``‖A₀‖ / Σⱼ|wⱼ|‖xⱼ‖``, the fraction of the integrand
            that survived the integration. The triangle inequality bounds it by
            1; an analytic integrand cancels it to machine epsilon. This is the
            only thing that distinguishes an empty contour from one whose poles
            outnumber the probe columns — both give a flat singular spectrum.
    """
    a0, a1, integrand_scale = _moment_sums(
        m_builder, pts, wts, v_probe, max_workers=max_workers
    )
    return a0, a1, _cancellation(a0, integrand_scale)


def _cancellation(a0: np.ndarray, integrand_scale: float) -> float:
    """``‖A₀‖ / Σⱼ|wⱼ|‖xⱼ‖``, or 0 when nothing was integrated."""
    return float(np.linalg.norm(a0)) / integrand_scale if integrand_scale > 0 else 0.0


def _moment_sums(
    m_builder: Callable[[complex], np.ndarray],
    pts: np.ndarray,
    wts: np.ndarray,
    v_probe: np.ndarray,
    *,
    max_workers: int | None,
) -> tuple[np.ndarray, np.ndarray, float]:
    """The sums behind :func:`contour_moments`, with the scale left unreduced.

    Returned as ``(A₀, A₁, Σⱼ|wⱼ|‖xⱼ‖)`` rather than with the cancellation
    ratio, because all three are sums over nodes: a nested rule adds the new
    nodes' sums to half of the old ones, which a ratio does not allow.
    """
    workers = _worker_count(v_probe.shape[0], int(pts.size), max_workers)

    def _node(job: tuple[complex, complex]) -> tuple[np.ndarray | None, float]:
        """Solve one node. Returns ``(None, 0.0)`` when the factorisation fails.

        The failure is *returned*, not counted into an enclosing variable and
        not warned about here: a shared counter mutated from several threads is
        a race, and ``warnings.warn`` raised in a worker does not reliably reach
        the caller's ``catch_warnings`` block. Both are re-joined in the parent.
        """
        lam, wt = job
        try:
            lu, piv = lu_factor(m_builder(lam))
            x_sol = lu_solve((lu, piv), v_probe)
        except (np.linalg.LinAlgError, ValueError):
            return None, 0.0
        return x_sol, abs(wt) * float(np.linalg.norm(x_sol))

    jobs = list(zip(pts, wts, strict=True))
    if workers == 1:
        results = map(_node, jobs)
        pool = None
    else:
        pool = ThreadPoolExecutor(max_workers=workers)
        # ``map`` yields in *submission* order, so the accumulation below runs
        # over the nodes in contour order however the threads happen to finish.
        # Floating-point addition is not associative, which is why this matters:
        # summing in completion order would make A₀ depend on thread timing and
        # the result irreproducible in its last digits. Bit-for-bit agreement
        # with the serial path is a test in the suite.
        results = pool.map(_node, jobs)

    a0 = np.zeros(v_probe.shape, dtype=complex)
    a1 = np.zeros(v_probe.shape, dtype=complex)
    integrand_scale = 0.0
    n_failed = 0
    try:
        for (lam, wt), (x_sol, scale) in zip(jobs, results, strict=True):
            if x_sol is None:
                n_failed += 1
                continue
            a0 += wt * x_sol
            a1 += wt * lam * x_sol
            integrand_scale += scale
    finally:
        if pool is not None:
            pool.shutdown()

    if n_failed:
        warnings.warn(
            f"{n_failed} of {pts.size} contour points failed to factorise; the "
            "contour may pass through an eigenvalue. Move or resize it.",
            RuntimeWarning,
            stacklevel=3,
        )

    return a0, a1, integrand_scale


def _detect_rank(sv_ratio: np.ndarray, floor: float) -> tuple[int, float]:
    """Rank of A₀: every singular value above the round-off plateau.

    Exactly, A₀ has rank K = the number of eigenvalues inside the contour. A
    quadrature rule changes that in two ways. Every pole *outside* the contour
    enters with a weight that falls geometrically as the rule is refined, and
    round-off adds a plateau that carries nothing. The first kind must be kept
    for as long as it stands above the second: a pole that is in A₀ but not in
    the retained subspace contaminates the eigenvalues inside at about its own
    weight, so a floor fixed well above round-off makes the answer *worse* as
    the contour is refined *(measured, docs/design/studies/beyn S1: with a
    fixed 1e-8 floor a five-pole pencil went 7.3e-14 → 2.1e-10 → 1.5e-8 at
    16 → 24 → 32 nodes per side; keeping down to the plateau holds 5e-14)*.

    Keeping such a pole costs nothing: A₁ carries it with the same weight times
    its own position, so it comes out as an eigenvalue at its true place,
    outside the contour, and the in-contour filter removes it.

    Two steps. A **cliff** — a drop of ``RANK_GAP_FLOOR`` between consecutive
    values, the upper one above ``floor`` — must exist somewhere; that is what
    says the spectrum has structure at all. Then the plateau is traced up from
    the smallest value (see ``PLATEAU_GAP``) and everything above it is kept.
    Not only what lies above the last cliff: the descent into the plateau is
    often a staircase of small drops, and the steps of that staircase are the
    weak outside poles this rule exists to keep.

    The smallest value is always assigned to the plateau. With a probe column
    to spare that is what it is; without one the probe is saturated and the
    weakest direction is the one that can be given up.

    A spectrum with no cliff has no meromorphic content that stands out:
    rank 0, an empty contour or a saturated probe. In a homogeneous background
    that is a statement about the contour, not about analyticity — M(λ) is
    holomorphic on any admissible rectangle there, so a flat spectrum cannot be
    blamed on a branch cut.

    Args:
        sv_ratio: float (p,) singular values divided by the largest.
        floor: Relative level below which a singular value is round-off
            whatever its spacing.

    Returns:
        rank: Number of singular values kept, possibly 0.
        max_gap: The largest consecutive ratio, reported as a diagnostic.
    """
    # Guard the log against an exactly-zero tail; the ratio only has to be
    # comparable against RANK_GAP_FLOOR, and a zero means "infinite gap".
    safe = np.maximum(sv_ratio, 1.0e-300)
    gaps = safe[:-1] / safe[1:]
    if gaps.size == 0:
        return (1, 1.0) if sv_ratio[0] > 0 else (0, 1.0)

    max_gap = float(gaps.max())
    # A gap qualifies only if the value it retains is above the floor; that is
    # what stops a chance drop *inside* the plateau from being read as a cliff.
    qualifies = (gaps >= RANK_GAP_FLOOR) & (sv_ratio[:-1] > floor)
    if not qualifies.any():
        return 0, max_gap

    rank = min(int(np.count_nonzero(sv_ratio > floor)), sv_ratio.size - 1)
    while rank >= 1 and gaps[rank - 1] < PLATEAU_GAP:
        rank -= 1
    return rank, max_gap


def beyn_poles(
    a0: np.ndarray,
    a1: np.ndarray,
    cancellation: float,
    z_lo: complex,
    z_hi: complex,
    rank_tol: float = 0.0,
    *,
    contour: str = "rectangle",
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
        rank_tol: Extra relative floor on the singular values kept. The
            default 0 leaves the decision to the round-off plateau traced from
            the spectrum (``PLATEAU_GAP``); raise it to drop directions that
            rule keeps.
        contour: ``"rectangle"``, or ``"ellipse"`` for moments integrated on
            the ellipse inscribed in it (:func:`ellipse_contour_quad`). It
            decides which eigenvalues count as inside.

    Returns:
        The eigenvalues inside the contour and their rank diagnostics.

    Raises:
        ValueError: If the singular spectrum has no cliff while A₀ is large —
            the eigenvalues inside outnumber the probe columns, so the probe
            cannot span the eigenspace and the result would be silently
            incomplete. Also for an unknown ``contour``.
    """
    if contour not in ("rectangle", "ellipse"):
        raise ValueError(f"contour must be 'rectangle' or 'ellipse'; got {contour!r}")

    u, s, vh = np.linalg.svd(a0, full_matrices=False)
    sv_ratio = s / s[0] if s[0] > 0 else s
    roundoff = np.finfo(float).eps / cancellation if cancellation > 0 else np.inf
    k_rank, max_gap = _detect_rank(sv_ratio, max(rank_tol, ROUNDOFF_MARGIN * roundoff))

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

    if contour == "ellipse":
        centre = 0.5 * (z_lo + z_hi)
        half = 0.5 * (z_hi - z_lo)
        inside = ((eigvals.real - centre.real) / half.real) ** 2 + (
            (eigvals.imag - centre.imag) / half.imag
        ) ** 2 < 1.0
    else:
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
    rank_tol: float = 0.0,
    rng_seed: int = 0,
    max_workers: int | None = None,
    contour: str = "rectangle",
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
        rank_tol: Extra relative floor on the singular values kept; see
            :func:`beyn_poles`.
        rng_seed: Probe-matrix seed. Results must not depend on it.
        max_workers: Threads for the contour loop; see
            :func:`contour_moments`. Results must not depend on it either.
        contour: ``"rectangle"``, or ``"ellipse"`` for the ellipse inscribed
            in it with the trapezoid rule on the same ``4·n_quad_per_side``
            nodes (:func:`ellipse_contour_quad`); eigenvalues in the corners
            are then outside.

    Returns:
        The eigenvalues inside the contour and their rank diagnostics.

    Raises:
        ValueError: If the eigenvalues inside outnumber ``n_probe``, or for an
            unknown ``contour``.
    """
    if contour == "ellipse":
        pts, wts = ellipse_contour_quad(z_lo, z_hi, 4 * n_quad_per_side)
    elif contour == "rectangle":
        pts, wts = rect_contour_quad(z_lo, z_hi, n_quad_per_side)
    else:
        raise ValueError(f"contour must be 'rectangle' or 'ellipse'; got {contour!r}")
    n_dim = m_builder(pts[0]).shape[0]
    v_probe = probe_matrix(n_dim, n_probe, rng_seed)
    a0, a1, cancellation = contour_moments(
        m_builder, pts, wts, v_probe, max_workers=max_workers
    )
    return beyn_poles(a0, a1, cancellation, z_lo, z_hi, rank_tol, contour=contour)


@dataclass(frozen=True)
class AdaptiveBeynModes:
    """Eigenvalues from a contour refined until they stopped moving.

    Attributes:
        modes: The result at the finest contour used.
        n_nodes: Nodes on that contour, which is also the number of matrices
            assembled: every coarser level is a subset of it. With
            interpolation, the nodes M was sampled at; the contour itself ran
            on ``n_fine`` nodes of the polynomial.
        change: float (K,) distance from each eigenvalue to the nearest one at
            the previous level; ``nan`` when no previous level gave a result.
            This is the error of the *previous* level. The error falls
            geometrically with the node count, so the current level is better
            than this by about the factor the last doubling bought.
        converged: Whether the last doubling kept the eigenvalue count and
            moved none of them by more than the tolerance.
        fit_error: With interpolation, the largest relative Frobenius error
            of the previous level's polynomial at this level's new nodes —
            the points halfway between its own, where an interpolant is
            worst. ``nan`` without interpolation or at the first level.
    """

    modes: BeynModes
    n_nodes: int
    change: np.ndarray
    converged: bool
    fit_error: float = float("nan")


def ellipse_faber_basis(
    lam: complex | np.ndarray,
    z_lo: complex,
    z_hi: complex,
    count: int,
) -> np.ndarray:
    """Faber polynomials ψ₀ … ψ_{count−1} of the ellipse inscribed in a rectangle.

    With ``u = λ − centre`` and semi-axes ``a``, ``b``:

        ψ₀ = 1,  ψ₁ = 2u/(a+b),  ψₖ = (2u/(a+b))·ψₖ₋₁ − q·ψₖ₋₂ (ψ₀ read as 2 for k = 2),

    with ``q = (a−b)/(a+b)``. These are the Chebyshev polynomials of
    ``u/f``, ``±f`` the foci, rescaled by ``2·(f/(a+b))ᵏ``. On the ellipse
    ``λ(t)`` they equal ``e^{ikt} + qᵏ e^{−ikt}`` for ``k ≥ 1``, so they stay
    of order one there and the interpolation matrix at the trapezoid nodes is a
    discrete Fourier matrix plus an aliased term weighted by ``|q|ᵏ < 1``. That
    is why the fit stays well conditioned on an elongated box *(condition 2.3
    to 2.6 on a 600 × 49 box at 16 and 64 nodes)*. The recurrence
    never divides by ``f``: a circle (``q = 0``) gives the monomials
    ``(u/a)ᵏ`` with no special case, and ``b > a`` (vertical foci) needs none.

    Args:
        lam: Point or points to evaluate at.
        z_lo: Bottom-left corner of the bounding rectangle.
        z_hi: Top-right corner.
        count: Number of polynomials.

    Returns:
        complex (count, len(lam)) values, one row per degree.
    """
    centre = 0.5 * (z_lo + z_hi)
    a = 0.5 * (z_hi.real - z_lo.real)
    b = 0.5 * (z_hi.imag - z_lo.imag)
    q = (a - b) / (a + b)
    w = 2.0 * (np.atleast_1d(np.asarray(lam, dtype=complex)) - centre) / (a + b)

    out = np.empty((count, w.size), dtype=complex)
    out[0] = 1.0
    if count > 1:
        out[1] = w
    if count > 2:
        out[2] = w * out[1] - 2.0 * q
    for k in range(3, count):
        out[k] = w * out[k - 1] - q * out[k - 2]
    return out


class _EllipseInterpolant:
    """The polynomial in λ through M at the trapezoid nodes of an ellipse.

    M is holomorphic on and inside the ellipse, and so is ``P − M``. By the
    maximum principle, the fit error measured on the ellipse therefore bounds
    it everywhere inside. That is what makes a contour on ``P`` a contour on M
    to within that error.
    """

    def __init__(
        self, pts: np.ndarray, samples: np.ndarray, z_lo: complex, z_hi: complex
    ) -> None:
        self.z_lo, self.z_hi = z_lo, z_hi
        self.count = int(pts.size)
        vander = ellipse_faber_basis(pts, z_lo, z_hi, self.count).T
        self.coeffs = np.linalg.solve(vander, samples.reshape(self.count, -1)).reshape(
            samples.shape
        )

    def __call__(self, lam: complex) -> np.ndarray:
        """``P(λ)``, one N×N matrix."""
        t = ellipse_faber_basis(lam, self.z_lo, self.z_hi, self.count)[:, 0]
        return np.tensordot(t, self.coeffs, axes=1)

    def rel_error(self, pts: np.ndarray, samples: np.ndarray) -> float:
        """Largest ``‖P − M‖_F / ‖M‖_F`` over the given nodes and samples."""
        return max(
            float(np.linalg.norm(self(z) - m) / np.linalg.norm(m))
            for z, m in zip(pts, samples, strict=True)
        )


def beyn_modes_adaptive(
    m_builder: Callable[[complex], np.ndarray],
    z_lo: complex,
    z_hi: complex,
    *,
    tol: float = 1.0e-9,
    n_start: int = 16,
    n_max: int = 256,
    n_probe: int = 12,
    rank_tol: float = 0.0,
    rng_seed: int = 0,
    max_workers: int | None = None,
    interpolate: bool = False,
    n_fine: int = 512,
    interp_memory: float = INTERPOLATION_MEMORY_BUDGET,
) -> AdaptiveBeynModes:
    """All eigenvalues of M(λ) inside an ellipse, to a requested accuracy.

    The contour is the ellipse inscribed in the rectangle, with the trapezoid
    rule. The node count is doubled until two consecutive levels agree, and
    because the rule nests (:func:`ellipse_contour_quad`) each doubling
    assembles only the new nodes: reaching ``n`` nodes this way costs ``n``
    assemblies, the same as asking for ``n`` at the outset.

    What this buys over :func:`beyn_modes` is that the node count stops being a
    guess. A fixed contour is either wasteful or, silently, too coarse; here
    the stopping test is the measured movement of the eigenvalues.

    **Interpolation.** The node count a contour needs is set by how close the
    poles of M⁻¹ come to it, while M itself is smooth across the region. With
    ``interpolate=True`` the ``n`` matrices assembled at a level are not
    integrated directly: a polynomial in λ is fitted through them
    (:func:`ellipse_faber_basis`) and the contour runs on ``n_fine`` nodes
    of the polynomial, which costs a factorisation per node and no assembly.
    The doubling and the stopping test are unchanged, so ``n_nodes`` counts the
    assemblies *(measured, docs/design/studies/beyn S6, ``tol = 1e-9``, 600 ×
    49 boxes on a circle: 64 assemblies against 512 for eleven TE modes, worst
    2.0e-11 against 4.8e-12; 64 against 128 for nine TM modes; 64 against 512
    with an absorbing core, both 6.0e-10. S5 measured 5.3× less wall time on
    one thread at N = 400)*. The gain needs a box where the poles, not M, set
    the node count; on a box with one well-separated mode both routes stop at
    the same level. Every level stores ``n`` matrices, so the node count is
    capped by ``interp_memory``: at N = 4000 not even 16 nodes fit the default.

    A contour on the polynomial much finer than the poles need can let a
    direction left by the fit clear the round-off plateau and come out as a
    noise eigenvalue at one level. It moves between levels, so the stopping
    test rejects it (``tests/test_beyn.py``); this route is offered only
    behind that test for this reason.

    A level that cannot be reduced — a coarse contour often shows no rank gap
    yet — is skipped, not fatal, unless it is the last one allowed.

    Args:
        m_builder: ``λ → complex (N, N)``, holomorphic on and inside the
            ellipse.
        z_lo: Bottom-left corner of the rectangle bounding the ellipse.
        z_hi: Top-right corner.
        tol: Largest movement of any eigenvalue over the last doubling, in
            units of λ, at which to stop.
        n_start: Nodes at the first level.
        n_max: Largest node count allowed; the last level is the largest
            ``n_start·2ᵏ`` that does not exceed it.
        n_probe: Probe columns; must exceed the eigenvalue count inside,
            counting the poles outside that the contour still sees.
        rank_tol: Extra relative floor on the singular values kept; see
            :func:`beyn_poles`.
        rng_seed: Probe-matrix seed. Results must not depend on it.
        max_workers: Threads for the contour loop; see
            :func:`contour_moments`. With interpolation they run the contour
            on the polynomial; the assemblies are serial.
        interpolate: Integrate a polynomial through the assembled matrices
            instead of the matrices themselves (see above).
        n_fine: Contour nodes on the polynomial; with interpolation, at least
            ``n_max``. Ignored without interpolation.
        interp_memory: Bytes the samples and the polynomial may take at their
            peak (``INTERP_PEAK_OVER_SAMPLES`` copies of ``n`` complex N×N
            matrices). A level that would exceed it is not run, and the level
            before it is the last. Ignored without interpolation.

    Returns:
        The eigenvalues at the finest level used, with the node count, the
        movement over the last doubling and whether it met ``tol``.

    Raises:
        ValueError: If the finest level allowed still cannot be reduced: the
            eigenvalues inside outnumber ``n_probe``. With interpolation, also
            if ``n_fine < n_max``, or if ``n_start`` nodes already exceed
            ``interp_memory``.
    """
    if interpolate and n_fine < n_max:
        raise ValueError(
            f"n_fine ({n_fine}) must be at least n_max ({n_max}): the contour on "
            "the polynomial would be coarser than the sampling"
        )
    pts, wts = ellipse_contour_quad(z_lo, z_hi, n_start)

    if not interpolate:
        v_probe = probe_matrix(m_builder(pts[0]).shape[0], n_probe, rng_seed)
        a0, a1, scale = _moment_sums(
            m_builder, pts, wts, v_probe, max_workers=max_workers
        )
    else:
        # Assembled serially: the samples are kept, so threads would multiply
        # the stack's peak rather than a per-node temporary.
        samples = np.array([m_builder(z) for z in pts])
        n_dim = samples.shape[1]

        def _fits(n: int) -> bool:
            return INTERP_PEAK_OVER_SAMPLES * 16.0 * n * n_dim**2 <= interp_memory

        if not _fits(n_start):
            raise ValueError(
                f"{n_start} interpolation nodes at N = {n_dim} need "
                f"{INTERP_PEAK_OVER_SAMPLES * 16.0 * n_start * n_dim**2:.2e} bytes, "
                f"over interp_memory = {interp_memory:.2e}; use the direct "
                "contour or raise interp_memory"
            )
        v_probe = probe_matrix(n_dim, n_probe, rng_seed)
        fine_pts, fine_wts = ellipse_contour_quad(z_lo, z_hi, n_fine)
        interp = _EllipseInterpolant(pts, samples, z_lo, z_hi)
        fit_error = float("nan")

    n_nodes = n_start
    previous: BeynModes | None = None

    while True:
        is_last = 2 * n_nodes > n_max or (interpolate and not _fits(2 * n_nodes))
        if interpolate:
            a0, a1, scale = _moment_sums(
                interp, fine_pts, fine_wts, v_probe, max_workers=max_workers
            )
        try:
            modes: BeynModes | None = beyn_poles(
                a0,
                a1,
                _cancellation(a0, scale),
                z_lo,
                z_hi,
                rank_tol,
                contour="ellipse",
            )
        except ValueError:
            if is_last:
                raise
            modes = None

        if modes is not None:
            change = np.full(modes.n_modes, np.nan)
            converged = False
            if previous is not None:
                if previous.n_modes and modes.n_modes:
                    # Nearest neighbour, not index against index: the sort
                    # order of two close eigenvalues can flip between levels.
                    change = np.abs(
                        modes.eigenvalues[:, None] - previous.eigenvalues[None, :]
                    ).min(axis=1)
                converged = previous.n_modes == modes.n_modes and bool(
                    np.all(change <= tol)
                )
            if converged or is_last:
                return AdaptiveBeynModes(
                    modes=modes,
                    n_nodes=n_nodes,
                    change=change,
                    converged=converged,
                    fit_error=fit_error if interpolate else float("nan"),
                )
            previous = modes

        # The odd-numbered nodes of the doubled rule are the new ones; the
        # even-numbered are the nodes already summed, at half their old weight.
        n_nodes *= 2
        pts, wts = ellipse_contour_quad(z_lo, z_hi, n_nodes)
        if interpolate:
            new = np.array([m_builder(z) for z in pts[1::2]])
            # The new nodes sit halfway between the old ones, which is where
            # the old polynomial is worst: its error there costs no assembly.
            fit_error = interp.rel_error(pts[1::2], new)
            merged = np.empty((n_nodes, *samples.shape[1:]), dtype=complex)
            merged[0::2] = samples
            merged[1::2] = new
            # Released before the next fit so the peak is one stack's worth
            # of coefficients, not two.
            del interp, samples, new
            samples = merged
            interp = _EllipseInterpolant(pts, samples, z_lo, z_hi)
            continue
        b0, b1, b_scale = _moment_sums(
            m_builder, pts[1::2], wts[1::2], v_probe, max_workers=max_workers
        )
        a0 = 0.5 * a0 + b0
        a1 = 0.5 * a1 + b1
        scale = 0.5 * scale + b_scale


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
        vector: complex (N,) the null-space vector polished alongside
            ``eigenvalue``, or the unrefined input when no step was accepted.
            The bordered system solves for the pair jointly, so returning only
            the eigenvalue would leave a caller holding a vector one Newton
            step staler than the λ beside it.

            Normalised to the **anchor**, ``conj(v0)·vector = 1``, which is the
            constraint the iteration enforces — *not* to unit length. A caller
            wanting unit columns must renormalise; the anchor is what keeps the
            phase tied to ``v0`` rather than free, so renormalising by
            ``‖vector‖`` preserves the gauge while rescaling does not disturb
            it.
    """

    eigenvalue: complex
    converged: bool
    step: float
    cond_jacobian: float
    vector: np.ndarray


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
        The refined eigenvalue, its null-space vector, and the convergence
        diagnostics.
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
        eigenvalue=lam,
        converged=converged,
        step=step,
        cond_jacobian=cond_j,
        vector=v,
    )
