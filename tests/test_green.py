import numpy as np
import pytest
from scipy.linalg import lu_factor, lu_solve

from conftest import N_CLAD, N_CORE, RAD, size_parameter
from pysie2d import (
    BIESolver,
    Geometry,
    Material,
    ScatterResult,
    line_dipole_rhs,
    relative_ldos,
    relative_ldos_map,
    self_green,
)
from pysie2d.reference.mie import self_green_cylinder

WAVELENGTH = 600.0


def _cross_scattered(
    solver: BIESolver, x_s: float, z_s: float, x_o: float, z_o: float
) -> complex:
    """Scattered field at (x_o, z_o) from a line dipole at (x_s, z_s)."""
    result = solver.scatter_dipole(WAVELENGTH, x_s, z_s)
    return complex(result.eval_field(np.array([x_o]), np.array([z_o]))[0])


def test_reciprocity():
    # Reciprocity is exact for the continuous problem and holds to
    # discretisation accuracy here: the scattered field at r2 from a source at
    # r1 equals the scattered field at r1 from a source at r2. This exercises
    # the RHS, the solve, and eval_field in two independent combinations, so it
    # cannot pass by accident.
    solver = BIESolver(Geometry.gielis(rad=RAD, n_pts=300, m=0), Material(N_CORE))
    r1 = (1.5 * RAD, 0.0)
    r2 = (0.0, 1.8 * RAD)
    s12 = _cross_scattered(solver, *r1, *r2)
    s21 = _cross_scattered(solver, *r2, *r1)
    assert s12 == pytest.approx(s21, rel=1e-6)


ANCHOR_NN = 1000


@pytest.fixture(scope="module", params=[2, 1], ids=["TE", "TM"])
def factorized_solver(request):
    # nn is raised above the 300 used for the far-field efficiency tests because
    # the near-field self-Green converges only at first order in nn; ~1e3 points
    # are needed for 1 % agreement at the closest distance d = 1.2a. The matrix
    # is identical across d at fixed pol, so it is LU-factorised once per
    # polarisation (module scope) and reused across the distance sweep.
    pol = request.param
    geom = Geometry.gielis(rad=RAD, n_pts=ANCHOR_NN, m=0)
    mat = Material(N_CORE, N_CLAD, pol=pol)
    lu = lu_factor(BIESolver(geom, mat).assemble(WAVELENGTH))
    return pol, geom, mat, lu


@pytest.mark.parametrize("d_over_a", [1.2, 1.5, 2.0, 3.0])
def test_self_green_vs_analytic_cylinder(factorized_solver, d_over_a):
    # Strong anchor: for a circular cylinder the self-Green function has a
    # closed form via Graf's addition theorem (reference.mie.self_green_cylinder).
    pol, geom, mat, lu = factorized_solver
    k = 2.0 * np.pi / WAVELENGTH
    x = size_parameter(WAVELENGTH)
    m = N_CORE / N_CLAD

    d = d_over_a * RAD
    rhs = line_dipole_rhs(ANCHOR_NN, WAVELENGTH, geom.f, geom.g, d, 0.0)
    ei = lu_solve(lu, rhs)
    s_bie = complex(
        ScatterResult(ei, geom, mat, WAVELENGTH).eval_field(
            np.array([d]), np.array([0.0])
        )[0]
    )
    s_ref = self_green_cylinder(x, m, k, d, pol)

    assert s_bie.real == pytest.approx(s_ref.real, rel=1e-2)
    assert s_bie.imag == pytest.approx(s_ref.imag, rel=1e-2)


def test_free_space_limit():
    # Far from the particle the environment does nothing: S → 0 and the LDOS
    # returns to its free-space value of 1.
    solver = BIESolver(Geometry.gielis(rad=RAD, n_pts=300, m=0), Material(N_CORE))
    d = 30.0 * RAD
    s = self_green(solver, WAVELENGTH, d, 0.0)
    assert abs(s) < 1e-2
    assert relative_ldos(solver, WAVELENGTH, d, 0.0) == pytest.approx(1.0, abs=1e-2)


def test_dipole_source_guards():
    # The RHS refuses sources inside the particle or within five boundary
    # spacings of the surface, where the incident field is near-singular.
    geom = Geometry.gielis(rad=RAD, n_pts=300, m=0)
    with pytest.raises(ValueError, match="inside"):
        line_dipole_rhs(geom.n_pts, WAVELENGTH, geom.f, geom.g, 0.0, 0.0)
    with pytest.raises(ValueError, match="5 boundary spacings"):
        line_dipole_rhs(geom.n_pts, WAVELENGTH, geom.f, geom.g, RAD + 1.0, 0.0)


def test_ldos_is_positive():
    # The relative LDOS is positive everywhere (a mathematical property); a
    # negative value would signal a convention or normalisation bug. Sweep 20
    # positions at varying distance and angle around the particle.
    solver = BIESolver(Geometry.gielis(rad=RAD, n_pts=300, m=0), Material(N_CORE))
    angles = np.linspace(0.0, 2.0 * np.pi, 20, endpoint=False)
    dists = np.linspace(1.3 * RAD, 2.5 * RAD, 20)
    for angle, d in zip(angles, dists, strict=True):
        x_s = d * np.cos(angle)
        z_s = d * np.sin(angle)
        assert relative_ldos(solver, WAVELENGTH, x_s, z_s) > 0.0


def test_ldos_map_matches_pointwise(circle):
    """The batched LDOS map agrees with per-point relative_ldos, NaNs included."""
    geom = circle(n_pts=120)
    mat = Material(n_core=1.5, n_clad=1.0, pol=2)
    solver = BIESolver(geom, mat)
    axis = np.linspace(-500.0, 500.0, 9)
    xx, zz = np.meshgrid(axis, axis)

    mapped = relative_ldos_map(solver, 600.0, xx, zz)

    for i in range(xx.shape[0]):
        for j in range(xx.shape[1]):
            try:
                expected = relative_ldos(solver, 600.0, xx[i, j], zz[i, j])
            except ValueError:
                assert np.isnan(mapped[i, j])
            else:
                assert mapped[i, j] == pytest.approx(expected, rel=1e-9)
