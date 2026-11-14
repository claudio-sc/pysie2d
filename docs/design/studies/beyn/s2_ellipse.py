"""S2 — rectangle with Gauss-Legendre against an ellipse with the trapezoid rule.

Three contours carry the same number of nodes:

  rectangle    Gauss-Legendre on each side, nodes/4 per side (the shipped rule)
  inscribed    trapezoid rule on the ellipse touching the four sides
  through      trapezoid rule on the ellipse through the four corners

Each is scored on the region it encloses, so the three columns do not count the
same eigenvalues: the inscribed ellipse leaves the corners out, the one through
the corners takes in more than the rectangle. The rank rule is the package's.

Each cell is: rank kept, worst error, median error, -missing +spurious.
"""

import numpy as np
from common import (
    EPS,
    ellipse_quad,
    in_box,
    in_ellipse,
    problems,
    reduce_moments,
    score,
)

from pysie2d.beyn import (
    ROUNDOFF_MARGIN,
    _detect_rank,
    contour_moments,
    probe_matrix,
    rect_contour_quad,
)

NODES = (16, 24, 32, 48, 64, 96, 128, 192, 256)
SQRT2 = float(np.sqrt(2.0))


def cell(prob, pts, wts, v_probe, region, **region_args) -> str:
    """Run one contour and format its score."""
    a0, a1, canc = contour_moments(prob.m_builder, pts, wts, v_probe)
    sv = np.linalg.svd(a0, compute_uv=False)
    rank, _ = _detect_rank(sv / sv[0], ROUNDOFF_MARGIN * EPS / canc)
    if rank == 0:
        return f"{'no rank':^29}"
    got = reduce_moments(a0, a1, rank)
    return f"{rank:2d} {score(got, prob, region, **region_args)}"


def main() -> None:
    """Print one table per problem."""
    for prob in problems():
        v_probe = probe_matrix(prob.n_dim, prob.n_probe, 0)
        counts = (
            prob.inside.size,
            int(in_ellipse(prob.poles, prob.z_lo, prob.z_hi).sum()),
            int(in_ellipse(prob.poles, prob.z_lo, prob.z_hi, SQRT2).sum()),
        )
        print(f"\n## {prob.name}  ({prob.n_probe} probes)")
        print(
            f"{'nodes':>5} | "
            + " | ".join(
                f"{f'{name} ({k} inside)':^29}"
                for name, k in zip(
                    ("rectangle", "inscribed", "through"), counts, strict=True
                )
            )
        )
        for n in NODES:
            rect = rect_contour_quad(prob.z_lo, prob.z_hi, n // 4)
            small = ellipse_quad(prob.z_lo, prob.z_hi, n)
            large = ellipse_quad(prob.z_lo, prob.z_hi, n, SQRT2)
            print(
                f"{n:5d} | "
                + " | ".join(
                    (
                        cell(prob, *rect, v_probe, in_box),
                        cell(prob, *small, v_probe, in_ellipse),
                        cell(prob, *large, v_probe, in_ellipse, scale=SQRT2),
                    )
                )
            )


if __name__ == "__main__":
    main()
