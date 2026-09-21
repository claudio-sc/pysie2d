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
import pytest

from pysie2d.geometry import gielis

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "validation"))

import gielis as validation_gielis  # noqa: E402


@pytest.fixture(params=sorted(validation_gielis.SHAPES))
def shape_name(request):
    return request.param


def test_contour_matches_the_package_exactly(shape_name):
    star = validation_gielis.SHAPES[shape_name]
    theta, x, z = validation_gielis.contour(**star)
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


def test_committed_contour_is_what_the_module_generates(shape_name):
    # The drivers read the .npz, not the function. If the two ever disagree,
    # the shape that was simulated is not the shape in the repo.
    shape = validation_gielis.SHAPES[shape_name]
    stored = validation_gielis.load(shape_name)
    theta, x, z = validation_gielis.contour(stored["theta"].size, **shape)
    np.testing.assert_array_equal(stored["theta"], theta)
    np.testing.assert_array_equal(stored["x"], x)
    np.testing.assert_array_equal(stored["z"], z)
    for key, value in shape.items():
        assert stored[key] == value


def test_the_star_is_centrosymmetric_and_the_skew_shape_is_not():
    # This is why there are two shapes. The incident direction is pinned to
    # -z in both external drivers and is unobservable on a circle, so it was
    # to be falsified on the star -- but at m = 6 a shift of θ by π shifts
    # mθ/4 by 1.5π, exchanging the |cos|^n2 and |sin|^n3 terms, and with
    # n2 = n3 they are the same term. Both incidence directions then give
    # identical cross-sections (measured in dolfinx: 1.3e-12 relative on
    # C_ext), so the check cannot fail on the frozen shape. SKEW breaks the
    # exchange with n2 != n3 and is what the two-angle check runs on.
    for name, expected in (("star", True), ("skew", False)):
        _, x, z = validation_gielis.contour(**validation_gielis.SHAPES[name])
        r = np.hypot(x, z)
        defect = np.max(np.abs(r - np.roll(r, r.size // 2)))
        # Round-off against tens of nm: no tolerance is being tuned here.
        assert bool(defect < 1e-9) is expected, f"{name}: inversion defect {defect}"


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
