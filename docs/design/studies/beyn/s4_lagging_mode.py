"""S4 — why one mode of the wide box lags, and what removes the lag.

On the eleven-mode box the ``n = 0`` mode converges far more slowly than the
other ten. This script tests one explanation: the eigenvectors of a circle are
confined, order by order, to a two-dimensional subspace (one angular harmonic
in each of the two boundary unknowns). Two moments, A₀ and A₁, can separate at
most as many poles as that subspace has dimensions. A third pole of the same
order, wherever it is, cannot be given a direction of its own, so its weight
on the contour lands on its neighbours as an error.

If that is right, the error is not a quadrature error of the mode itself, and
more probe columns cannot help. What helps is more moments: with A₀ … A₃
arranged as block Hankel matrices (Beyn 2012, section 5) a subspace of
dimension two can hold four poles. The solves are the same ones; only the sums
differ.

Columns: moments used (2 is the shipped method), rank kept, error of the
``n = 0`` mode, worst error among the other ten, eigenvalues found inside.
"""

import numpy as np
from common import EPS, in_box, problems
from scipy.linalg import lu_factor, lu_solve

from pysie2d.beyn import (
    ROUNDOFF_MARGIN,
    _detect_rank,
    probe_matrix,
    rect_contour_quad,
)

N0_POLE = 530.8321407288238 + 26.37849897377869j
NODES_PER_SIDE = (8, 12, 16, 24, 32)


def moments(prob, pts, wts, v_probe, count: int) -> tuple[list[np.ndarray], float]:
    """Moments of ``((λ − c)/ρ)ᵖ M⁻¹V`` for p < count, and the cancellation.

    The variable is centred and scaled to the box so that high powers stay
    O(1); raw powers of a wavelength near 800 would not.
    """
    centre = 0.5 * (prob.z_lo + prob.z_hi)
    radius = 0.5 * abs(prob.z_hi - prob.z_lo)
    sums = [np.zeros(v_probe.shape, dtype=complex) for _ in range(count)]
    scale = 0.0
    for lam, wt in zip(pts, wts, strict=True):
        x_sol = lu_solve(lu_factor(prob.m_builder(lam)), v_probe)
        scale += abs(wt) * float(np.linalg.norm(x_sol))
        for power in range(count):
            sums[power] += wt * ((lam - centre) / radius) ** power * x_sol
    return sums, float(np.linalg.norm(sums[0])) / scale


def hankel_eigenvalues(prob, sums, canc: float, blocks: int) -> tuple[np.ndarray, int]:
    """Eigenvalues from ``blocks`` × ``blocks`` block Hankel matrices."""
    centre = 0.5 * (prob.z_lo + prob.z_hi)
    radius = 0.5 * abs(prob.z_hi - prob.z_lo)
    b0 = np.block([[sums[i + j] for j in range(blocks)] for i in range(blocks)])
    b1 = np.block([[sums[i + j + 1] for j in range(blocks)] for i in range(blocks)])
    u, s, vh = np.linalg.svd(b0, full_matrices=False)
    rank, _ = _detect_rank(s / s[0], ROUNDOFF_MARGIN * EPS / canc)
    reduced = (u[:, :rank].conj().T @ b1 @ vh[:rank].conj().T) / s[:rank]
    return np.linalg.eigvals(reduced) * radius + centre, rank


def main() -> None:
    """Print the ladder for the TE wide box."""
    prob = next(p for p in problems() if p.name == "circle TE, wide box")
    v_probe = probe_matrix(prob.n_dim, prob.n_probe, 0)
    others = prob.inside[np.abs(prob.inside - N0_POLE) > 1.0]
    print(f"## {prob.name}  ({prob.inside.size} inside, {prob.n_probe} probes)")
    print(
        f"{'nodes':>5} {'moments':>7} {'rank':>4} {'n=0 error':>10} {'others':>9} found"
    )
    for n_side in NODES_PER_SIDE:
        pts, wts = rect_contour_quad(prob.z_lo, prob.z_hi, n_side)
        sums, canc = moments(prob, pts, wts, v_probe, 6)
        for blocks in (1, 2, 3):
            got, rank = hankel_eigenvalues(prob, sums, canc, blocks)
            got = got[in_box(got, prob.z_lo, prob.z_hi)]
            if rank == 0 or got.size == 0:
                print(f"{4 * n_side:5d} {2 * blocks:7d} {'no rank':>25}")
                continue
            print(
                f"{4 * n_side:5d} {2 * blocks:7d} {rank:4d} "
                f"{np.abs(got - N0_POLE).min():10.1e} "
                f"{max(np.abs(got - w).min() for w in others):9.1e} {got.size:5d}"
            )


if __name__ == "__main__":
    main()
