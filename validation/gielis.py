"""The gate-4 star: one Gielis contour, defined once for both external tools.

Neither external driver may import pysie2d — a reference that calls the code
under test is not a reference. But the two tools must see the *same* boundary
to the last digit, or a disagreement on the star is a disagreement about
geometry and is unattributable, which is exactly what the gates exist to
prevent. So the superformula lives here, in numpy and the standard library
only, and the sampled contour is committed beside it as
``validation/data/gielis-star.npz``: the file the drivers read is the shape
that was simulated, auditable in the repo without running anything.

The shape is the README star — the one ``examples/nearfield_map.py`` and
``examples/purcell_map.py`` already draw — so the figure a visitor sees and
the geometry that carries the validation claim are one shape.

Conventions this file is pinned to (conventions §2, and ``geometry.gielis``):

* ``x = r·sin θ``, ``z = r·cos θ``. Not the textbook ``(cos, sin)``: pysie2d's
  invariant axis is **y**, and the incident wave at ``angle = 0`` runs along
  **−z**. Swapping the two rotates the star by 90°, which on a six-fold shape
  is a different scatterer and *not* a relabelling.
* Lengths in nm, and ``a = b = 1``.

Equality with the package is asserted by ``tests/test_gielis_contour.py``,
which evaluates ``pysie2d.geometry.gielis`` at the same θ and demands exact
agreement — not a tolerance. Both sides run the same closed form in double
precision; anything but bit-for-bit equality means the formula drifted.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

DATA = Path(__file__).resolve().parent / "data"

STAR = {
    "rad": 200.0,
    "m": 6,
    "n1": 6.0,
    "n2": 12.0,
    "n3": 12.0,
}
"""The frozen shape. ``n_core = 2.0``, ``n_clad = 1.0`` live with the cases."""

SKEW = {**STAR, "n2": 8.0}
"""A deliberately non-centrosymmetric shape. Not frozen, and not an anchor.

It exists to hold one measurement, and the measurement came out the other way.
The incident direction is pinned to −z in both drivers and is unobservable on
a circle, so the plan was to falsify it on the star. The star cannot: at
``m = 6`` a shift of θ by π shifts the superformula's argument ``mθ/4`` by
1.5π, exchanging the ``|cos|^n2`` and ``|sin|^n3`` terms, and with ``n2 = n3``
they are the same term, so ``r(θ+π) = r(θ)`` to 7e-13 nm. ``n2 = 8`` breaks
that exchange and lifts the inversion defect to 33 nm.

**It changed nothing.** Reversing the incidence on the skew shape moves
``C_ext`` by 3e-13 in dolfinx and 2.5e-05 in MEEP — its own noise — exactly as
on the star. The cause is not the geometry but reciprocity: the forward
scattering amplitude for ``k̂`` equals the one for ``−k̂`` for any reciprocal
scatterer, so ``C_ext`` is direction-invariant whatever the shape, and
losslessness carries ``C_sca`` with it. No total cross-section can pin this
convention. The check that can is the far-field *pattern*, which is what the
skew shape is now kept for.
"""

SHAPES = {"star": STAR, "skew": SKEW}

N_CONTOUR = 1440
"""Contour samples handed to gmsh and to MEEP.

Not a convergence knob for the *physics* — it is how finely the analytic curve
is resolved before either mesher sees it, and it must be far finer than any
mesh either tool will build on top of it, so that the geometry error each tool
carries is its own and not this sampling. 1440 is 240 points per lobe at
``m = 6``; the sharpest feature of this star (``n1 = 6``, ``n2 = n3 = 12``) is
resolved to ~0.1 nm of chord at ``rad = 200 nm``.
"""


def contour(
    n_pts: int = N_CONTOUR,
    rad: float = STAR["rad"],
    m: int = STAR["m"],
    n1: float = STAR["n1"],
    n2: float = STAR["n2"],
    n3: float = STAR["n3"],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sample the Gielis boundary at uniform θ.

    Args:
        n_pts: Number of samples. The endpoint θ = 2π is excluded, so the
            contour is open and the consumer closes it — gmsh and MEEP both
            reject a repeated first point.
        rad: Scale radius (nm).
        m: Rotational symmetry order.
        n1: Gielis shape exponent.
        n2: Gielis shape exponent.
        n3: Gielis shape exponent.

    Returns:
        theta, x, z — all (n_pts,), lengths in nm.
    """
    theta = np.linspace(0.0, 2.0 * np.pi, n_pts, endpoint=False)
    arg = m * theta / 4.0
    co = np.abs(np.cos(arg)) ** n2
    se = np.abs(np.sin(arg)) ** n3
    r = rad * (co + se) ** (-1.0 / n1)
    return theta, r * np.sin(theta), r * np.cos(theta)


def load(shape: str = "star") -> dict[str, np.ndarray]:
    """Read one committed contour. The drivers call this and nothing else.

    Args:
        shape: ``star`` (the frozen anchor) or ``skew`` (the direction check).
    """
    with np.load(DATA / f"gielis-{shape}.npz") as d:
        return {k: d[k] for k in d.files}


def main() -> None:
    """Write the committed contour file."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-pts", type=int, default=N_CONTOUR)
    args = parser.parse_args()

    DATA.mkdir(exist_ok=True)
    for name, shape in SHAPES.items():
        theta, x, z = contour(args.n_pts, **shape)
        out = DATA / f"gielis-{name}.npz"
        np.savez(
            out, theta=theta, x=x, z=z, **{k: np.asarray(v) for k, v in shape.items()}
        )
        r = np.hypot(x, z)
        defect = np.max(np.abs(r - np.roll(r, r.size // 2)))
        print(
            f"{out.name}: {theta.size} points, "
            f"r ∈ [{r.min():.3f}, {r.max():.3f}] nm, "
            f"inversion defect {defect:.3g} nm"
        )


if __name__ == "__main__":
    main()
