"""Fast cluster solves at one wavelength, and cluster eigenpermittivities.

`ClusterEigenEps` solves the dense coupled system of conventions §14 by compressing
the neighbour coupling, so for driven solves the dense `ClusterBIESolver` is the
right reference: this is linear algebra, not new physics. Tolerances cite the
measured values in docs/design/studies/cluster-reduction/README.md.
"""

import numpy as np
import pytest

from pysie2d import (
    Cluster,
    ClusterBIESolver,
    Geometry,
    HalfSpace,
    Material,
    Stack,
)
from pysie2d.kernels import _interior_rows, assemble_matrix

NC = 1.444
LAM = 1550.0
BACKGROUNDS = {
    "free": None,
    "glass": HalfSpace(2.25, z_int=0.0),
    "pec": HalfSpace.pec(),
    "sin": Stack(((4.0, 400.0),), NC**2),
}


def _mixed() -> list[Geometry]:
    """Four particles differing in shape, size, height and resolution."""
    return [
        Geometry.gielis(rad=70.0, n_pts=64, m=0, x0=-420.0, z0=170.0),
        Geometry.gielis(rad=90.0, n_pts=120, m=3, n1=4.0, n2=6.0, n3=6.0, x0=-80.0, z0=200.0),
        Geometry.gielis(rad=60.0, n_pts=90, m=4, n1=8.0, n2=8.0, n3=8.0, x0=240.0, z0=140.0),
        Geometry.gielis(rad=110.0, n_pts=140, m=0, x0=640.0, z0=230.0, a=1.0, b=0.8),
    ]


def _mixed_materials(pol: int) -> list[Material]:
    return [
        Material.from_eps((-2.0 + 0.2j) * NC**2, n_clad=NC, pol=pol),
        Material(n_core=2.4, n_clad=NC, pol=pol),
        Material.from_eps((4.0 + 0.3j) * NC**2, n_clad=NC, pol=pol),
        Material(n_core=1.8, n_clad=NC, pol=pol),
    ]


@pytest.mark.parametrize("pol", [1, 2])
@pytest.mark.parametrize(
    ("ri", "kd", "k_bg"),
    [
        (2.0, 4.0, 0.01),  # real ε > 0, real k: the Cephes branch of _real_if_real
        (np.sqrt(-2.0 + 0.1j), -2.0 + 0.1j, 0.01),  # absorbing: complex k_core only
        (2.0, 4.0, 0.01 + 0.0005j),  # complex k_bg: the QNM path
    ],
)
def test_interior_rows_are_bit_identical_to_assemble_matrix(pol, ri, kd, k_bg):
    """The deliberate copy in kernels agrees exactly with what it copies.

    `array_equal`, not `allclose`: this test is the only guard on the duplicate, and
    a drift in one expression must fail here however small. All three `_real_if_real`
    routes are covered, because each takes different Bessel code.
    """
    g = Geometry.gielis(rad=90.0, n_pts=48, m=3, n1=4.0, n2=6.0, n3=6.0)
    args = (pol, 48, g.f, g.g, g.df, g.dg, g.ddf, g.ddg, k_bg, ri, kd)
    assert np.array_equal(_interior_rows(*args), assemble_matrix(*args)[48:])


@pytest.mark.parametrize("pol", [1, 2])
def test_exterior_half_of_the_matrix_does_not_see_the_material(pol):
    """The premise of caching exterior rows, exact to the bit.

    `_exterior_rows` builds the self block with ri = kd = 1. That is only legitimate if
    the upper half of `assemble_matrix` truly ignores both, so compare across materials
    that take different arithmetic paths in the interior half.
    """
    g = Geometry.gielis(rad=90.0, n_pts=48, m=3, n1=4.0, n2=6.0, n3=6.0)
    base = (pol, 48, g.f, g.g, g.df, g.dg, g.ddf, g.ddg, 0.01)
    reference = assemble_matrix(*base, 1.0, 1.0)[:48]
    for ri, kd in ((2.0, 4.0), (np.sqrt(-2.0 + 0.1j), -2.0 + 0.1j), (3.5 + 0.1j, 12.0)):
        assert np.array_equal(assemble_matrix(*base, ri, kd)[:48], reference)


@pytest.mark.parametrize("name", list(BACKGROUNDS))
@pytest.mark.parametrize("pol", [1, 2])
def test_exterior_rows_are_bit_identical_to_the_dense_assembly(name, pol):
    """Building rows one particle at a time changes nothing, on every background.

    Ragged `nn` and different heights are used because a wrong offset, or a reflection
    block added at the wrong particle, cannot then cancel by symmetry.
    """
    cl = Cluster(_mixed())
    solver = ClusterBIESolver(cl, _mixed_materials(pol), pol, BACKGROUNDS[name])
    dense = solver._assemble(LAM)
    refl = solver._reflection(LAM)
    for q, g in enumerate(cl.geometries):
        start = cl.offsets[q]
        rows = solver._exterior_rows(q, LAM, refl)
        assert np.array_equal(rows, dense[start : start + g.n_pts])
