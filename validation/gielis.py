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


def load() -> dict[str, np.ndarray]:
    """Read the committed contour. The drivers call this and nothing else."""
    with np.load(DATA / "gielis-star.npz") as d:
        return {k: d[k] for k in d.files}


def main() -> None:
    """Write the committed contour file."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-pts", type=int, default=N_CONTOUR)
    args = parser.parse_args()

    theta, x, z = contour(args.n_pts)
    DATA.mkdir(exist_ok=True)
    out = DATA / "gielis-star.npz"
    np.savez(out, theta=theta, x=x, z=z, **{k: np.asarray(v) for k, v in STAR.items()})
    r = np.hypot(x, z)
    print(f"{out}: {theta.size} points, r ∈ [{r.min():.3f}, {r.max():.3f}] nm")


if __name__ == "__main__":
    main()
