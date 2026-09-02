"""How far does the R → n_pts map move across the shape family (D17, A13)?

`conventions.md` §12 places Jacobian rungs in `R = wavelength_over_ds`, not in
raw `n_pts`, because `n_pts` is not comparable across shapes. That settles
*what* to hold fixed; it leaves open what it costs to hold it fixed, which is
the number a catalogue budget is built from. A shape needing 3× the points of a
circle for the same R costs 9× the assembly.

This measures the map on a spanning set of Gielis shapes rather than on a
sampling region: the numeric bounds of the catalogue's region are a
catalogue-side decision (backlog B5) and are not fixed yet. What is fixed, and
solver-side, is that `R` is the shape-independent reading and `n_pts_for` is
how a design point is sized. The spread reported here is what that policy
costs.

Cheap by construction — `wavelength_over_ds` is a boundary quantity, no Beyn
box and no assembly, so the whole table is sub-second.

Run: ``uv run python docs/design/studies/r_band_policy.py``
"""

from pysie2d import Geometry, Material, wavelength_over_ds

RAD = 200.0
LAM = 551.4  # the Gate-2/3 design point's mode, representative of the band
N_CORE = 3.0

# A spanning set, not a sample: the circle (the reference every n_pts intuition
# was built on), elongation at fixed n1, and the two directions n1 moves the
# boundary away from an ellipse — n1 < 2 towards the cusped regime D6 excludes,
# n1 > 2 towards a rounded rectangle.
SHAPES = {
    "circle": {"m": 0},
    "ellipse b/a = 1.2": {"m": 4, "n1": 2.0, "n2": 2.0, "n3": 2.0, "a": 1.0, "b": 1.2},
    "ellipse b/a = 2": {"m": 4, "n1": 2.0, "n2": 2.0, "n3": 2.0, "a": 1.0, "b": 2.0},
    "ellipse b/a = 3": {"m": 4, "n1": 2.0, "n2": 2.0, "n3": 2.0, "a": 1.0, "b": 3.0},
    "n1 = 1.5, b/a = 1.2": {
        "m": 4,
        "n1": 1.5,
        "n2": 2.0,
        "n3": 2.0,
        "a": 1.0,
        "b": 1.2,
    },
    "n1 = 4, b/a = 1.2": {"m": 4, "n1": 4.0, "n2": 2.0, "n3": 2.0, "a": 1.0, "b": 1.2},
}
TARGET_R = (15.0, 30.0)  # the §12 production pair


def n_pts_for(shape, target_r, material):
    """Smallest even n_pts whose R reaches ``target_r`` for this shape."""

    def r_at(n):
        return wavelength_over_ds(
            Geometry.gielis(rad=RAD, n_pts=n, **shape), material, LAM
        )

    probe = 200
    n = max(50, int(probe * target_r / r_at(probe)))
    n += n % 2
    while r_at(n) < target_r:
        n += 2
    return n


def main():
    """Tabulate the R = 15 and R = 30 rungs, and their cost against the circle."""
    material = Material(n_core=N_CORE, n_clad=1.0, pol=2)

    print(f"{'shape':>22s}  {'R@200':>7s}  {'n(15)':>6s} {'n(30)':>6s}  {'cost':>6s}")
    circle_cost = None
    for label, shape in SHAPES.items():
        r200 = wavelength_over_ds(
            Geometry.gielis(rad=RAD, n_pts=200, **shape), material, LAM
        )
        n15, n30 = (n_pts_for(shape, t, material) for t in TARGET_R)
        # Assembly is O(n_pts²) and the pair costs both rungs.
        cost = n15**2 + n30**2
        circle_cost = circle_cost or cost
        print(
            f"{label:>22s}  {r200:7.2f}  {n15:6d} {n30:6d}  {cost / circle_cost:6.2f}x"
        )


if __name__ == "__main__":
    main()
