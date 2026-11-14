"""S3 — the adaptive stop: what it costs and whether its estimate can be trusted.

The trapezoid rule on an ellipse nests: the nodes at n are every other node at
2n, so doubling reuses every solve. ``beyn_modes_adaptive`` doubles until the
eigenvalues stop moving, and reports how far each moved in the last doubling.

Two questions, per problem and tolerance:

  - Is the reported change an upper bound on the true error? (``est/err`` ≥ 1)
  - How many nodes does it use, against the smallest fixed rectangle from
    S2's ladder that reaches the same tolerance on its own region?

The rectangle column is generous to the rectangle: it assumes the right node
count was known in advance. The two regions differ (the ellipse leaves the
corners out), so the comparison is of cost per region, not per eigenvalue.
"""

import numpy as np
from common import in_ellipse, problems, score

from pysie2d.beyn import beyn_modes, beyn_modes_adaptive

TOLS = (1.0e-6, 1.0e-9)
RECT_LADDER = (4, 6, 8, 12, 16, 24, 32, 48, 64)
N_MAX = 512


def rectangle_nodes(prob, tol: float) -> str:
    """Fewest nodes on the ladder at which the fixed rectangle meets ``tol``."""
    for n_side in RECT_LADDER:
        try:
            res = beyn_modes(
                prob.m_builder,
                prob.z_lo,
                prob.z_hi,
                n_quad_per_side=n_side,
                n_probe=prob.n_probe,
            )
        except ValueError:
            continue
        s = score(res.eigenvalues, prob)
        if s.missing == 0 and s.spurious == 0 and s.worst < tol:
            return f"{4 * n_side:5d}"
    return f"{'>256':>5}"


def main() -> None:
    """Print one row per problem and tolerance."""
    print(
        f"{'problem':<28} {'tol':>7} | {'nodes':>5} {'conv':>5} {'estimate':>9} "
        f"{'true err':>9} {'est/err':>8} {'-miss +spur':>11} | rectangle nodes"
    )
    for prob in problems():
        for tol in TOLS:
            res = beyn_modes_adaptive(
                prob.m_builder,
                prob.z_lo,
                prob.z_hi,
                tol=tol,
                n_probe=prob.n_probe,
                n_max=N_MAX,
            )
            s = score(res.modes.eigenvalues, prob, in_ellipse)
            est = float(res.change.max()) if res.change.size else 0.0
            ratio = est / s.worst if s.worst > 0 else np.inf
            print(
                f"{prob.name:<28} {tol:7.0e} | {res.n_nodes:5d} {res.converged!s:>5} "
                f"{est:9.1e} {s.worst:9.1e} {ratio:8.1e} "
                f"{f'-{s.missing} +{s.spurious}':>11} | {rectangle_nodes(prob, tol)}"
            )


if __name__ == "__main__":
    main()
