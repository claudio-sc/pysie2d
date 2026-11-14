"""S1 — which singular values of A₀ to keep.

Two rules are applied to the same moment matrices. Both return no rank unless
the spectrum has a cliff, a drop of 1e3 or more between consecutive singular
values (the empty-or-saturated branch, which this study does not touch).

  cliff 1e-8   shipped up to v1.3.0: keep down to the last cliff whose upper
               value is above 1e-8 of the largest
  plateau G    walk up from the smallest singular value while each one is
               within a factor G of the one below it; that run is the round-off
               plateau, and everything above it is kept. Values under
               10·ε/cancellation belong to the plateau whatever their spacing.

ε/cancellation is the rounding of the contour sum relative to what survives
it, the "unit" of the last column.

Each cell is: rank kept, worst error, median error, -missing +spurious.
The tally at the end scans G over probe seeds and two probe counts.
"""

import numpy as np
from common import EPS, problems, reduce_moments, score

from pysie2d.beyn import contour_moments, probe_matrix, rect_contour_quad

CLIFF = 1.0e3
ROUNDOFF_MARGIN = 10.0
NODES_PER_SIDE = (3, 4, 6, 8, 12, 16, 24, 32, 48, 64)
SEEDS = range(5)
GAPS = (5.0, 10.0, 20.0, 30.0, 50.0, 100.0)
SHOWN = 20.0


def cliff(sv: np.ndarray, floor: float) -> int:
    """Rank at the last drop of ``CLIFF`` or more whose upper value clears ``floor``."""
    safe = np.maximum(sv, 1.0e-300)
    ok = (safe[:-1] / safe[1:] >= CLIFF) & (sv[:-1] > floor)
    return int(np.flatnonzero(ok)[-1]) + 1 if ok.any() else 0


def plateau(sv: np.ndarray, canc: float, gap: float) -> int:
    """Rank above the round-off plateau, or 0 when the spectrum has no cliff."""
    level = ROUNDOFF_MARGIN * EPS / canc
    if not cliff(sv, level):
        return 0
    rank = min(int((sv > level).sum()), sv.size - 1)
    while rank >= 1 and sv[rank - 1] < gap * sv[rank]:
        rank -= 1
    return rank


def main() -> None:
    """Print one table per problem at seed 0, then the scan of G."""
    rules = ["cliff 1e-8"] + [f"plateau {g:g}" for g in GAPS]
    tally = {
        r: {"runs": 0, "spurious": 0, "missing": 0, "lost a digit": 0} for r in rules
    }
    shown = rules.index(f"plateau {SHOWN:g}")
    for prob in problems():
        print(f"\n## {prob.name}  ({prob.inside.size} inside, {prob.n_probe} probes)")
        print(
            f"{'nodes':>5} {'cancel':>8} | {'cliff 1e-8':^29} | "
            f"{rules[shown]:^29} | smallest sv, units"
        )
        for n_side in NODES_PER_SIDE:
            pts, wts = rect_contour_quad(prob.z_lo, prob.z_hi, n_side)
            for seed in SEEDS:
                for n_probe in (prob.n_probe, prob.n_probe + 8):
                    v_probe = probe_matrix(prob.n_dim, n_probe, seed)
                    a0, a1, canc = contour_moments(prob.m_builder, pts, wts, v_probe)
                    sv = np.linalg.svd(a0, compute_uv=False)
                    sv = sv / sv[0]
                    ks = [cliff(sv, 1.0e-8)] + [plateau(sv, canc, g) for g in GAPS]
                    scores = [
                        score(reduce_moments(a0, a1, k), prob) if k else None
                        for k in ks
                    ]
                    found = [s for s in scores if s]
                    best = min((s.worst for s in found), default=np.nan)
                    fewest = min((s.spurious for s in found), default=0)
                    for rule, s in zip(rules, scores, strict=True):
                        if s is None:
                            continue
                        t = tally[rule]
                        t["runs"] += 1
                        t["spurious"] += s.spurious > fewest
                        t["missing"] += s.missing > 0
                        t["lost a digit"] += s.worst > 10.0 * best
                    if seed or n_probe != prob.n_probe:
                        continue
                    cells = [
                        f"{ks[i]:2d} {scores[i]}" if scores[i] else f"{'no rank':^29}"
                        for i in (0, shown)
                    ]
                    print(
                        f"{4 * n_side:5d} {canc:8.1e} | "
                        + " | ".join(cells)
                        + f" | {sv[-1] * canc / EPS:8.1e}"
                    )

    print(f"\n## Scan of G: {len(SEEDS)} probe seeds, n_probe and n_probe + 8")
    print("spurious:     more false eigenvalues inside the box than the best rule")
    print("lost a digit: worst error more than 10x the best of the rules")
    for rule, t in tally.items():
        print(f"{rule:>11}: " + ", ".join(f"{k} {v}" for k, v in t.items()))


if __name__ == "__main__":
    main()
