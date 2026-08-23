"""A17: does the §12 two-rung Richardson still hold 6.4e-4 at aspect 3?

Gate 10 closed on a measurement at **one** design point, the aspect-1.2 ellipse
(`gate10_richardson.py`). The premise underneath it — that J converges at first
order in ``n_pts`` — is a property of the discretisation and there is no reason
for it to fail as the boundary elongates. But that is an argument, and the
backlog recorded it as one. This is the check.

Elongation is the direction that matters, because A13 measured it as the whole
cost story of the shape family: aspect 3 costs 4.51× the circle for the
``R = 15 + 30`` pair, while ``n1`` over its whole declared range spans only
1.10–1.31×. If the extrapolation survives at ``b/a = 3`` it survives everywhere
the Level 1–3 region goes (``b ≤ 1.4``), with room to spare.

**Design point.** ``b = 3.0``, everything else as the reference ellipse. The
pole box moves with it: at this aspect the mode the reference box holds is gone
and the one used here sits at 549.81 + 18.34j, a single mode in
``540 + 10j … 580 + 30j``, verified isolated at n_pts = 600.

**The contour settings move too, and that is a result in itself.** The
reference study's ``n_quad_per_side = 6`` fails Beyn's cancellation check at
*every* rung here — R = 15, 30 and 50 alike — so it is the contour that is
under-resolved at this aspect, not the boundary. Beyn raises rather than
returning a wrong pole, which is the behaviour that makes this safe to discover
at run time. Raised to 12 (and ``n_probe`` to 20) for the whole ladder, so all
five rungs share one setting and none of the comparison is between settings.

**Same estimator, same bar, same scoring.** Two-rung Richardson with the
exponent pinned at ``p = 1``, scored against an ``R = 75/100`` limit built from
rungs no candidate uses — not against the finest rung a candidate extrapolated
from, which would be circular. Pass criterion fixed before the run and
unchanged from the gate: **1 % relative on every J component**, with the
recorded 6.4e-4 of the reference design point as the value to beat.

Run: ``uv run python docs/design/studies/a17_richardson_elongated.py``
"""

from gate10_jacobian_convergence import N_CORE, jacobian, n_pts_for
from gate10_richardson import LADDER_R, richardson

from pysie2d import Material

# b/a = 3. `a` stays 1: D15 makes (a, b) → (ta, tb) a pure dilation, so the
# aspect ratio is the only thing `b` alone moves.
DESIGN = {"b": 3.0}
BOX = (540.0 + 10.0j, 580.0 + 30.0j)


def main():
    """The gate10_richardson ladder, re-run at the elongated design point."""
    material = Material(n_core=N_CORE, n_clad=1.0, pol=2)
    probe_lam = 549.8  # sizes n_pts only; each rung recomputes λ

    rungs = []
    for target in LADDER_R:
        n_pts = n_pts_for(target, material, probe_lam, **DESIGN)
        lam, r, j = jacobian(
            n_pts, material, box=BOX, n_quad_per_side=12, n_probe=20, **DESIGN
        )
        rungs.append((target, n_pts, r, j))
        print(f"R target {target:4.0f}  n_pts {n_pts:5d}  R {r:6.2f}  lam {lam:.6f}")

    keys = list(rungs[-1][3])
    (_, n_c, _, j_c), (_, n_f, _, j_f) = rungs[3], rungs[4]
    ref = {k: richardson(j_c[k], j_f[k], n_c, n_f) for k in keys}

    n_50 = rungs[2][1]

    def report(label, est, n_used):
        cost = sum((n / n_50) ** 2 for n in n_used)
        cells = "  ".join(f"{abs(est[k] - ref[k]) / abs(ref[k]):10.3e}" for k in keys)
        print(f"{label:>22s}  {cost:5.2f}  {cells}")

    print("\nrelative error against the R = 75/100 limit; cost in R = 50 assemblies")
    print(f"{'estimator':>22s}  {'cost':>5s}  " + "  ".join(f"{k:>10s}" for k in keys))
    for target, n_pts, _, j in rungs[:3]:
        report(f"raw rung R = {target:.0f}", j, [n_pts])
    for lo, hi in ((0, 1), (0, 2), (1, 2)):
        (t_c, n_c, _, j_c), (t_f, n_f, _, j_f) = rungs[lo], rungs[hi]
        est = {k: richardson(j_c[k], j_f[k], n_c, n_f) for k in keys}
        report(f"Richardson {t_c:.0f}+{t_f:.0f}", est, [n_c, n_f])

    print("\nreference (R = 75/100 Richardson limit)")
    for k in keys:
        print(f"  {k:>12s} = {ref[k]!r}")


if __name__ == "__main__":
    main()
