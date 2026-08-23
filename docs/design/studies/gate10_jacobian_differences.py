"""Gate 10, route 2: does ΔJ between nearby designs converge faster than J?

``gate10_jacobian_convergence.py`` measured the thing the gate asks for and got
a negative answer: J converges at first order in ``n_pts``, so the 1 % bar is
met at no rung that fits the compute budget. Two routes out were recorded, and
this script measures the second one.

**The claim under test.** A design ranking never reads J at one point; it reads
how J *changes* between nearby designs. ``docs/conventions.md`` §9 argues the
fixed discretisation error is smooth in the shape parameter, so if that holds,
the error is common-mode between two designs a small δ apart and cancels a
second time in ``ΔJ = J(p₀ + δ) − J(p₀)``. Then the criterion could be restated
on ΔJ and the cheap rungs would serve.

**Why it has to be measured rather than assumed.** That is the same §9 argument
whose *order* prediction the first script has just refuted. It survived as a
statement about the constant — ``dλ/drad`` is three decades better resolved
than ``dλ/db`` at the same R — and a second cancellation of the same kind would
also show up in the constant. The question is whether it shows up in the order,
and only a ladder answers that.

**Pass criterion, fixed before the run** — route 2 is viable if ΔJ either
converges at an order above 1, or lands its residual at R = 50 at least 10×
below J's own 1.5e-2. Anything less and restating the criterion on ΔJ buys
nothing the first script did not already have.

δ = 0.02 in ``b``: large enough that ΔJ is not itself a cancellation artefact
(it moves λ by ~1 nm, a hundred times the R = 50 pole error), small enough to
be the scale at which two catalogue designs are compared.

Run: ``uv run python docs/design/studies/gate10_jacobian_differences.py``
"""

from gate10_jacobian_convergence import (
    ELLIPSE,
    N_CORE,
    TARGET_R,
    _observed_order,
    jacobian,
    n_pts_for,
)

from pysie2d import Material

DELTA_B = 0.02


def _order_and_residual(seq, n_of):
    """Observed order and distance of the finest rung from its Richardson limit."""
    j_coarse, j_mid, j_fine = seq
    p = _observed_order(abs(j_coarse - j_mid) / abs(j_mid - j_fine), n_of)
    limit = j_fine + (j_fine - j_mid) / ((n_of[1] / n_of[2]) ** -p - 1.0)
    return p, abs(j_fine - limit) / abs(limit)


def main():
    """Ladder J and ΔJ together, and compare how each converges."""
    material = Material(n_core=N_CORE, n_clad=1.0, pol=2)
    probe_lam = 551.4  # only to size n_pts; each rung recomputes λ

    base, moved, n_of = [], [], []
    for target in TARGET_R:
        # Both designs at the *same* n_pts. Sizing each from its own R target
        # would let the two differ by a point or two, and that difference is
        # itself of the order of the error being chased — it would decide the
        # measurement by the sizing rule rather than by the physics. δb = 0.02
        # moves R by under 1 %, so one n_pts serves the pair.
        n_pts = n_pts_for(target, material, probe_lam)
        n_of.append(n_pts)
        _, r, j0 = jacobian(n_pts, material)
        _, _, j1 = jacobian(n_pts, material, b=ELLIPSE["b"] + DELTA_B)
        base.append(j0)
        moved.append(j1)
        print(f"R target {target:4.0f}  n_pts {n_pts:4d}  R {r:6.2f}")

    print(f"\norder and residual at R = 50, J against dJ (db = {DELTA_B})")
    header = f"{'component':>12s}  {'p(J)':>6s} {'res(J)':>10s}"
    print(header + f"   {'p(dJ)':>6s} {'res(dJ)':>10s}")
    for k in base[0]:
        pj, rj = _order_and_residual([b[k] for b in base], n_of)
        pd, rd = _order_and_residual(
            [m[k] - b[k] for m, b in zip(moved, base, strict=True)], n_of
        )
        print(f"{k:>12s}  {pj:6.2f} {rj:10.3e}   {pd:6.2f} {rd:10.3e}")


if __name__ == "__main__":
    main()
