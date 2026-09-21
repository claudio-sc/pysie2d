"""The external drivers' star is the package's star, bit for bit.

Gate 4 compares pysie2d against MEEP and dolfinx on a non-circular shape. The
drivers must not import pysie2d, so ``validation/gielis.py`` carries its own
copy of the superformula and commits the sampled contour the meshers read. A
copy that drifts turns a physics disagreement into a geometry disagreement
that looks exactly like one — this is the only thing standing between those
two readings.

Exactness, not a tolerance, is the right assertion: both sides evaluate the
same closed form on the same θ in double precision, so equal inputs give
identical bits. A tolerance here would quietly accept a reordered expression
that is no longer the same curve at the sharp lobe tips.
"""

import sys
from pathlib import Path

import numpy as np

from pysie2d.geometry import gielis

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "validation"))

import gielis as validation_gielis  # noqa: E402


def test_contour_matches_the_package_exactly():
    star = validation_gielis.STAR
    theta, x, z = validation_gielis.contour()
    f, g, *_ = gielis(
        theta,
        rad=star["rad"],
        a=1.0,
        b=1.0,
        m=star["m"],
        n1=star["n1"],
        n2=star["n2"],
        n3=star["n3"],
        x0=0.0,
        z0=0.0,
    )
    # f is x and g is z: x = r·sin θ, z = r·cos θ. Asserting the pair in this
    # order is also what pins the orientation — a (cos, sin) contour would
    # still pass a radius-only check while presenting a 90°-rotated star to
    # the −z incident wave.
    np.testing.assert_array_equal(x, f)
    np.testing.assert_array_equal(z, g)


def test_committed_contour_is_what_the_module_generates():
    # The drivers read the .npz, not the function. If the two ever disagree,
    # the shape that was simulated is not the shape in the repo.
    stored = validation_gielis.load()
    theta, x, z = validation_gielis.contour(stored["theta"].size)
    np.testing.assert_array_equal(stored["theta"], theta)
    np.testing.assert_array_equal(stored["x"], x)
    np.testing.assert_array_equal(stored["z"], z)
    for key, value in validation_gielis.STAR.items():
        assert stored[key] == value


def test_the_star_is_six_fold_and_not_a_circle():
    # Cheap, but it is what makes the file above a *star*: a bug that
    # collapsed the exponents would leave every test above passing against an
    # equally collapsed reference. r/rad spans 1.00 to 1.78 here.
    _, x, z = validation_gielis.contour()
    r = np.hypot(x, z)
    assert r.max() / r.min() > 1.5
    # Six-fold symmetry: rotating the index by a sixth of the contour leaves
    # the radius profile unchanged.
    np.testing.assert_allclose(r, np.roll(r, r.size // 6), rtol=1e-12)
