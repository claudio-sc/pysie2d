"""Shared pieces of the Beyn numerical-performance study.

Every script here measures the extraction step alone: a matrix function with a
known spectrum goes in, and the error of what the contour returns comes out.
Two families of problems are used.

Synthetic pencils have their eigenvalues placed by hand, so the reference is
exact and a solve costs microseconds. The circle problems use this package's
boundary-integral matrix, with the analytic Mie poles as the reference; at
``n_pts = 64`` the boundary discretisation is at round-off, so what is measured
is the contour.

Run any script from the repository root:
    uv run python docs/design/studies/beyn/<script>.py
"""

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from scipy.optimize import linear_sum_assignment

from pysie2d import Geometry, Material, QNMSolver
from pysie2d.reference import mie

EPS = float(np.finfo(float).eps)


@dataclass(frozen=True)
class Problem:
    """A matrix function, a search rectangle, and the spectrum inside it.

    Attributes:
        name: Label used in the printed tables.
        m_builder: ``λ → complex (N, N)``.
        n_dim: Dimension N.
        z_lo: Bottom-left corner of the rectangle.
        z_hi: Top-right corner.
        poles: complex (P,) every known eigenvalue in and around the rectangle,
            repeated by multiplicity.
        n_probe: Probe columns used for this problem.
    """

    name: str
    m_builder: Callable[[complex], np.ndarray]
    n_dim: int
    z_lo: complex
    z_hi: complex
    poles: np.ndarray
    n_probe: int

    @property
    def inside(self) -> np.ndarray:
        """The exact eigenvalues inside the rectangle."""
        return self.poles[in_box(self.poles, self.z_lo, self.z_hi)]


def in_box(z: np.ndarray, z_lo: complex, z_hi: complex) -> np.ndarray:
    """Mask of the points strictly inside the rectangle."""
    return (
        (z.real > z_lo.real)
        & (z.real < z_hi.real)
        & (z.imag > z_lo.imag)
        & (z.imag < z_hi.imag)
    )


def in_ellipse(
    z: np.ndarray, z_lo: complex, z_hi: complex, scale: float = 1.0
) -> np.ndarray:
    """Mask of the points inside the ellipse inscribed in the rectangle.

    ``scale`` multiplies both semi-axes; ``√2`` gives the ellipse through the
    four corners.
    """
    centre = 0.5 * (z_lo + z_hi)
    half = 0.5 * (z_hi - z_lo) * scale
    return ((z.real - centre.real) / half.real) ** 2 + (
        (z.imag - centre.imag) / half.imag
    ) ** 2 < 1.0


def ellipse_quad(
    z_lo: complex, z_hi: complex, n_nodes: int, scale: float = 1.0
) -> tuple[np.ndarray, np.ndarray]:
    """Trapezoid rule on the ellipse inscribed in the rectangle, times ``scale``.

    The weights carry ``dλ/dt`` and ``1/(2πi)``, as ``rect_contour_quad`` does.
    """
    centre = 0.5 * (z_lo + z_hi)
    half = 0.5 * (z_hi - z_lo) * scale
    t = 2.0 * np.pi * np.arange(n_nodes) / n_nodes
    pts = centre + half.real * np.cos(t) + 1j * half.imag * np.sin(t)
    dz = -half.real * np.sin(t) + 1j * half.imag * np.cos(t)
    return pts, dz / (1j * n_nodes)


def synthetic(nonlinear: bool, seed: int = 1) -> Problem:
    """Non-normal pencil with poles inside, just outside, and far from the box.

    ``M(λ) = Q·diag(f_i(λ))·Q⁻¹`` with a random, non-unitary Q, so the pencil
    is not normal and its conditioning is that of a generic dense matrix.
    ``f_i = λ − r_i`` (linear) or ``exp(λ − r_i) − 1`` (nonlinear; the further
    roots ``r_i + 2πik`` all lie outside the box, whose height is 1).

    The four poles just outside the box are the point: they leak into the
    moment matrix with a weight that falls as the contour is refined, which is
    the regime where the rank decision is hard.
    """
    rng = np.random.default_rng(seed)
    n_dim = 80
    inside = np.array([0.3 + 0.2j, -0.5 + 0.1j, 0.1 - 0.3j, 0.85 + 0.0j, -0.2 + 0.42j])
    near = np.array([1.12 + 0.1j, -0.3 + 0.62j, 0.4 - 0.6j, -1.2 - 0.2j])
    n_far = n_dim - inside.size - near.size
    far = 3.0 * np.exp(2j * np.pi * rng.random(n_far)) * (1.0 + rng.random(n_far))
    roots = np.concatenate([inside, near, far])
    q = rng.standard_normal((n_dim, n_dim)) + 1j * rng.standard_normal((n_dim, n_dim))
    q_inv = np.linalg.inv(q)

    def m_builder(lam: complex) -> np.ndarray:
        f = np.exp(lam - roots) - 1.0 if nonlinear else roots - lam
        return (q * f) @ q_inv

    return Problem(
        name="synthetic, nonlinear" if nonlinear else "synthetic, linear",
        m_builder=m_builder,
        n_dim=n_dim,
        z_lo=-1.0 - 0.5j,
        z_hi=1.0 + 0.5j,
        poles=roots,
        n_probe=12,
    )


def unitary_pencil() -> Problem:
    """Nonlinear pencil with unitary factors and a wide spread of entries.

    ``M(λ) = X·diag(exp(λ − r_i) − 1, pad)·Y`` with X, Y unitary and three roots
    far outside the box, the pencil of ``tests/test_beyn.py``. The far roots
    make the diagonal span ``e⁻¹⁸ … e¹¹``, so round-off is far from uniform
    across directions: the plateau of A₀ is the widest of the set.
    """
    rng = np.random.default_rng(1)
    n_dim = 40
    inside = np.array([1.5 - 0.3j, 2.5 + 0.4j])
    roots = np.concatenate([inside, [10.0 + 5.0j, -8.0 - 3.0j, 20.0 + 1.0j]])
    pad = 1.0 + 0.5 * np.arange(1, n_dim - roots.size + 1)
    shape = (n_dim, n_dim)
    x_mat, _ = np.linalg.qr(
        rng.standard_normal(shape) + 1j * rng.standard_normal(shape)
    )
    y_mat, _ = np.linalg.qr(
        rng.standard_normal(shape) + 1j * rng.standard_normal(shape)
    )

    def m_builder(lam: complex) -> np.ndarray:
        return (x_mat * np.concatenate([np.exp(lam - roots) - 1.0, pad])) @ y_mat

    return Problem(
        name="unitary factors, nonlinear",
        m_builder=m_builder,
        n_dim=n_dim,
        z_lo=1.0 - 1.0j,
        z_hi=3.0 + 1.0j,
        poles=roots,
        n_probe=12,
    )


def circle(
    name: str,
    z_lo: complex,
    z_hi: complex,
    *,
    pol: int = 2,
    n_core: complex = 3.0,
    n_probe: int = 20,
    n_pts: int = 64,
) -> Problem:
    """Circular cylinder, radius 200 nm, with its analytic Mie poles.

    Orders ``n ≥ 1`` are doubly degenerate (the ±n partners), so each appears
    twice in ``inside``.
    """
    rad = 200.0
    geom = Geometry.gielis(rad=rad, n_pts=n_pts, m=0)
    mat = Material(n_core=n_core, n_clad=1.0, pol=pol)
    # Size-parameter window wide enough to cover the rectangle with margin.
    x_range = (2 * np.pi * rad / (1.5 * z_hi.real), 2 * np.pi * rad / (0.7 * z_lo.real))
    orders, lams, _ = mie.qnm_wavelengths(
        rad=rad, m=n_core, pol=pol, x_range=x_range, n_clad=1.0
    )
    poles = np.repeat(lams, np.where(orders == 0, 1, 2))
    return Problem(
        name=name,
        m_builder=QNMSolver(geom, mat)._bie.assemble,
        n_dim=2 * n_pts,
        z_lo=z_lo,
        z_hi=z_hi,
        poles=poles,
        n_probe=n_probe,
    )


def problems() -> list[Problem]:
    """The fixed problem set every script runs."""
    return [
        synthetic(nonlinear=False),
        synthetic(nonlinear=True),
        unitary_pencil(),
        circle("circle TE, one mode", 520 + 15j, 545 + 40j, n_probe=12),
        circle("circle TE, wide box", 500 + 1j, 1100 + 50j),
        circle("circle TM, wide box", 500 + 1j, 1100 + 50j, pol=1),
        circle("circle TE, lossy core", 500 + 1j, 1100 + 50j, n_core=3.0 + 0.05j),
    ]


def reduce_moments(
    a0: np.ndarray, a1: np.ndarray, rank: int, shift: complex = 0.0
) -> np.ndarray:
    """Eigenvalues of the rank-``rank`` reduction of the two moments."""
    u, s, vh = np.linalg.svd(a0, full_matrices=False)
    u1, v1 = u[:, :rank], vh[:rank].conj().T
    return np.linalg.eigvals((u1.conj().T @ a1 @ v1) / s[:rank]) + shift


@dataclass(frozen=True)
class Score:
    """How a set of computed eigenvalues compares with the exact ones.

    Attributes:
        worst: Largest matched distance.
        median: Median matched distance.
        missing: Exact eigenvalues left without a partner.
        spurious: Computed eigenvalues left without a partner.
    """

    worst: float
    median: float
    missing: int
    spurious: int

    def __str__(self) -> str:
        """One table cell."""
        if np.isnan(self.worst):
            return f"{'—':>8} {'—':>8} -{self.missing} +{self.spurious}"
        return f"{self.worst:8.1e} {self.median:8.1e} -{self.missing} +{self.spurious}"


def score(
    computed: np.ndarray,
    problem: Problem,
    region: Callable[..., np.ndarray] = in_box,
    **region_args: float,
) -> Score:
    """Match the computed eigenvalues inside the region one-to-one with the exact.

    The assignment minimises the total distance, so a degenerate pair needs two
    computed eigenvalues to count as found. ``region`` is ``in_box`` or
    ``in_ellipse``, applied to the computed and to the exact eigenvalues alike.
    """
    got = computed[region(computed, problem.z_lo, problem.z_hi, **region_args)]
    want = problem.poles[
        region(problem.poles, problem.z_lo, problem.z_hi, **region_args)
    ]
    if got.size == 0 or want.size == 0:
        return Score(np.nan, np.nan, want.size, got.size)
    dist = np.abs(got[:, None] - want[None, :])
    rows, cols = linear_sum_assignment(dist)
    matched = dist[rows, cols]
    return Score(
        worst=float(matched.max()),
        median=float(np.median(matched)),
        missing=want.size - rows.size,
        spurious=got.size - rows.size,
    )
