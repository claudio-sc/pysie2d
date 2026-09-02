"""Gate 10, route 1: does two-rung Richardson extrapolation of J reach 1 %?

Route 2 is closed off (`gate10_jacobian_differences.py`): differencing J between
nearby designs keeps the error and loses the signal. What is left is to use the
one thing three independent measurements now agree on — that J converges at
**first order** in ``n_pts``, `p` = 0.98…1.02 — and extrapolate it away.

**The estimator, and why p is fixed rather than fitted.** From two rungs,

    J* = J_f + (J_f − J_c) / ((n_f / n_c) − 1)

with the exponent pinned at 1. Fitting `p` needs a third rung, costs 1.7× more,
and pays for it with a fitted exponent whose conditioning is poor when the two
differences are close — while the thing being fitted is already known to 2 %
from three separate ladders. A wrong-by-2 % exponent moves J* by far less than
a third of the assembly budget buys elsewhere.

**The check.** Extrapolating and then testing against the finest rung one
extrapolated *from* would be circular. So this ladder carries two rungs above
the working band, R = 75 and R = 100, that no scored candidate uses, and every
candidate is scored against the R = 75/100 Richardson limit — the best estimate
available, and built from rungs strictly finer than any candidate's.

Pass criterion, fixed before the run: **1 % relative on every J component**, the
tolerance already stated for the gate (the OED objective is quadratic in J, so
1 % on J is 2 % on the Fisher matrix, below what the noise model and the D6a
acceptance fraction contribute to a design ranking). Cost is reported in
assembly units alongside, since the whole point of the route is that it is
cheaper than the `n_pts ≈ 580` a raw rung would need.

Run: ``uv run python docs/design/studies/gate10_richardson.py``
"""

from gate10_jacobian_convergence import N_CORE, jacobian, n_pts_for

from pysie2d import Material

LADDER_R = (15.0, 30.0, 50.0, 75.0, 100.0)


def richardson(j_coarse, j_fine, n_coarse, n_fine):
    """First-order Richardson limit of two rungs. Exponent pinned at p = 1."""
    return j_fine + (j_fine - j_coarse) / ((n_fine / n_coarse) - 1.0)


def main():
    """Score every two-rung candidate against the finest pair's limit."""
    material = Material(n_core=N_CORE, n_clad=1.0, pol=2)
    probe_lam = 551.4  # only to size n_pts; each rung recomputes λ

    rungs = []
    for target in LADDER_R:
        n_pts = n_pts_for(target, material, probe_lam)
        _, r, j = jacobian(n_pts, material)
        rungs.append((target, n_pts, r, j))
        print(f"R target {target:4.0f}  n_pts {n_pts:4d}  R {r:6.2f}")

    keys = list(rungs[-1][3])
    # Reference: the two finest rungs, which no scored candidate uses.
    (_, n_c, _, j_c), (_, n_f, _, j_f) = rungs[3], rungs[4]
    ref = {k: richardson(j_c[k], j_f[k], n_c, n_f) for k in keys}

    # Cost in assembly units: assembly is O(n_pts²), so a rung costs (n/n_50)².
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
