"""Gate 10: does the sensitivity Jacobian converge in the discretisation?

The whole compute budget of a shape catalogue rests on how finely the boundary
has to be sampled before ``J = ∂λ/∂p`` stops moving. Assembly is O(n_pts²) in
memory and worse in time, so the difference between "R = 15 is enough" and
"R = 50 is needed" is roughly an order of magnitude in the cost of every
design point.

**Stated in R, not in n_pts** (``docs/conventions.md``, D17). ``n_pts`` is not
comparable across shapes: the same 200 points give R = 37.1 on a circle and
17.5 on an aspect-3 ellipse, so a ladder in ``n_pts`` would be measuring
different resolutions at different points of a catalogue.
``pysie2d.wavelength_over_ds`` is the shape-independent reading, and the ladder
here is taken at **R = 15 / 30 / 50** — the cheap band, the accurate band, and
one in between.

**What is compared, and against what.** The pole λ itself converges only at
first order in n_pts, and at R = 30 it is still ~0.4 nm from the analytic Mie
value on the circle — so an absolute test of λ is not the question. The
question is whether the *derivative* converges faster, which
``docs/conventions.md`` §9 argues on the grounds that the fixed discretisation
error is smooth in the shape parameter and largely cancels in a ratio. That is
an argument; this is the measurement. The reference is the finest rung, and
what is reported is the relative movement of each J component between rungs.

Run: ``uv run python docs/design/studies/gate10_jacobian_convergence.py``
"""

from pysie2d import Geometry, Material, QNMSolver, wavelength_over_ds

# The Gate-2/3 design point: m=4, n1=n2=n3=2 is an exact ellipse (D14) with
# semi-axes b·rad along x and a·rad along z, so a ≠ b exercises the real Gielis
# shape-derivative path rather than the circle's dilation identity.
RAD = 200.0
ELLIPSE = {"m": 4, "n1": 2.0, "n2": 2.0, "n3": 2.0, "a": 1.0, "b": 1.2}
N_CORE = 3.0
BOX = (543.0 + 18.0j, 560.0 + 32.0j)
TARGET_R = (15.0, 30.0, 50.0)


def geometry(n_pts, theta=None, **overrides):
    """The design-point ellipse at a chosen resolution, optionally frozen."""
    kw = {**ELLIPSE, "rad": RAD, "n_pts": n_pts}
    kw.update(overrides)
    if theta is not None:
        kw["theta"] = theta
    return Geometry.gielis(**kw)


def n_pts_for(target_r, material, wavelength):
    """Smallest even n_pts whose R reaches ``target_r``.

    R is very nearly proportional to n_pts (the boundary is fixed; only the
    chord length changes), so one scaling step from a probe lands within a
    point or two and a short walk finishes it. Solved rather than tabulated
    because the map depends on the shape, which is the entire reason D17
    states the requirement in R.
    """
    probe = 200
    r_probe = wavelength_over_ds(geometry(probe), material, wavelength)
    n = max(50, int(probe * target_r / r_probe))
    while wavelength_over_ds(geometry(n), material, wavelength) < target_r:
        n += 2
    return n


def jacobian(n_pts, material, **design):
    """dλ/dp at the design point for p in (b, a, rad, n_core), plus λ and R.

    ``design`` overrides Gielis parameters, so a nearby design can be laddered
    on exactly this code path (route 2, ``gate10_jacobian_differences.py``).
    """
    at_design = {**ELLIPSE, **design}
    geom = geometry(n_pts, **design)
    res = QNMSolver(geom, material).modes(z_lo=BOX[0], z_hi=BOX[1], n_quad_per_side=6)
    if res.n_modes != 1:
        raise RuntimeError(f"n_pts={n_pts}: box holds {res.n_modes} modes, expected 1")
    res = res.refine()
    theta = res.geometry.theta
    lam = res.wavelengths[0]

    def shape(name):
        def at(delta):
            moved = geometry(
                n_pts, theta=theta, **{**design, name: at_design[name] + delta}
            )
            return moved, material

        return res.sensitivity(at)[0]

    def scale(delta):
        return geometry(n_pts, theta=theta, rad=RAD + delta, **design), material

    def core(delta):
        return geometry(n_pts, theta=theta, **design), Material(
            n_core=N_CORE + delta, n_clad=1.0, pol=2
        )

    return (
        lam,
        wavelength_over_ds(geom, material, lam),
        {
            "dl/db": shape("b"),
            "dl/da": shape("a"),
            "dl/drad": res.sensitivity(scale)[0],
            "dl/dn_core": res.sensitivity(core)[0],
        },
    )


def _observed_order(ratio, n_of, lo=0.1, hi=6.0):
    """Convergence order p implied by three rungs at unequal n_pts.

    Solves ratio = (n0^-p − n1^-p) / (n1^-p − n2^-p) by bisection. The usual
    log-of-a-ratio formula assumes equally spaced rungs, which these are not:
    R is the quantity held on a ladder, and the n_pts it implies are 115, 228
    and 378. The function is monotone in p over this bracket, so bisection is
    both sufficient and immune to the bad conditioning of a fitted exponent.
    """

    def f(p):
        a, b, c = (float(n) ** -p for n in n_of)
        return (a - b) / (b - c) - ratio

    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if f(lo) * f(mid) <= 0.0:
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi)


def main():
    """Run the R ladder and print the movement of J between rungs."""
    material = Material(n_core=N_CORE, n_clad=1.0, pol=2)
    probe_lam = 551.4  # only to size n_pts; the ladder recomputes λ at each rung

    rows = []
    for target in TARGET_R:
        n_pts = n_pts_for(target, material, probe_lam)
        lam, r, j = jacobian(n_pts, material)
        rows.append((target, n_pts, r, lam, j))
        print(f"R target {target:4.0f}  n_pts {n_pts:4d}  R {r:6.2f}  lam {lam:.6f}")

    finest = rows[-1][4]
    print("\nrelative movement of each J component against the R = 50 rung")
    keys = list(finest)
    print("R target  " + "  ".join(f"{k:>12s}" for k in keys))
    for target, _, _, _, j in rows[:-1]:
        cells = [abs(j[k] - finest[k]) / abs(finest[k]) for k in keys]
        print(f"{target:8.0f}  " + "  ".join(f"{c:12.3e}" for c in cells))

    # The movement above is a *lower bound* on the error: the R = 50 rung is
    # itself converging. With three rungs the observed order and a Richardson
    # limit are available, and the distance from the finest rung to that limit
    # is the honest error estimate.
    n_of = [row[1] for row in rows]
    print("\nobserved order in n_pts, and distance of the R = 50 rung from the limit")
    for k in keys:
        j_coarse, j_mid, j_fine = (row[4][k] for row in rows)
        p = _observed_order(abs(j_coarse - j_mid) / abs(j_mid - j_fine), n_of)
        limit = j_fine + (j_fine - j_mid) / ((n_of[1] / n_of[2]) ** -p - 1.0)
        print(
            f"  {k:>12s}  p = {p:5.2f}   |J50 - limit|/|limit| = "
            f"{abs(j_fine - limit) / abs(limit):.3e}"
        )

    print("\nJ at the finest rung")
    for k in keys:
        print(f"  {k:>12s} = {finest[k]!r}")


if __name__ == "__main__":
    main()
