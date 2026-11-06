"""Fast cluster solves at one wavelength, and cluster eigenpermittivities.

`ClusterEigenEps` solves the dense coupled system of conventions §14 by compressing
the neighbour coupling, so for driven solves the dense `ClusterBIESolver` is the
right reference: this is linear algebra, not new physics. Tolerances cite the
measured values in docs/design/studies/cluster-reduction/README.md.
"""

import time

import numpy as np
import pytest

from pysie2d import (
    BIESolver,
    Cluster,
    ClusterBIESolver,
    ClusterCancellationWarning,
    ClusterEigenEps,
    Geometry,
    HalfSpace,
    Material,
    Stack,
)
from pysie2d.beyn import beyn_modes
from pysie2d.kernels import _interior_rows, assemble_matrix
from pysie2d.reference import two_cylinder

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
        Geometry.gielis(
            rad=90.0, n_pts=120, m=3, n1=4.0, n2=6.0, n3=6.0, x0=-80.0, z0=200.0
        ),
        Geometry.gielis(
            rad=60.0, n_pts=90, m=4, n1=8.0, n2=8.0, n3=8.0, x0=240.0, z0=140.0
        ),
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


def _rel(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b) / np.linalg.norm(b))


@pytest.mark.parametrize("name", list(BACKGROUNDS))
@pytest.mark.parametrize("pol", [1, 2])
def test_fast_solve_equals_dense_solve_on_every_background(name, pol):
    """Mixed shapes, `nn` and materials: the same solution, and the same far field.

    The dense solve is the right reference because this is the same linear system
    solved another way. Measured worst `x` 1.4e-13 and far field 1.6e-15 (study s5 H);
    both are round-off in a matrix whose condition number is near 1e4. 1e-12 and 1e-14
    leave ~7× headroom. A wrong rank, offset or dropped substrate block misses by
    orders of magnitude, so this cannot pass by accident.
    """
    cl = Cluster(_mixed())
    mats = _mixed_materials(pol)
    solver = ClusterBIESolver(cl, mats, pol, BACKGROUNDS[name])
    dense = solver.scatter(LAM, 13.0)
    fast = solver.at_wavelength(LAM).scatter(mats, 13.0)

    assert _rel(fast.ei, dense.ei) < 1e-12
    amp_fast, _ = fast.far_field(181)
    amp_dense, _ = dense.far_field(181)
    assert np.max(np.abs(amp_fast - amp_dense)) / np.max(np.abs(amp_dense)) < 1e-14


def test_materials_change_only_the_interior_rows():
    """One prepared object answers for several material sets, each like a fresh solve.

    This is the feature's whole point: if any state from one set leaked into the next,
    or if the materials given at construction were used, the later solves would differ
    from their dense references.
    """
    cl = Cluster(_mixed())
    sets = [_mixed_materials(1), [Material(n_core=1.5, n_clad=NC, pol=1)] * 4]
    prepared = ClusterBIESolver(cl, sets[0], 1).at_wavelength(LAM)
    for mats in sets + sets[:1]:
        dense = ClusterBIESolver(cl, mats, 1).scatter(LAM)
        # Round-off as in the test above: measured x <= 1.4e-13.
        assert _rel(prepared.scatter(mats).ei, dense.ei) < 1e-12


def test_dipole_and_guided_mode_excitations_equal_the_dense_solve():
    """The other two excitations share the solve; their right-hand sides must agree."""
    cl = Cluster(_mixed())
    mats = _mixed_materials(1)
    solver = ClusterBIESolver(cl, mats, 1, BACKGROUNDS["sin"])
    prepared = solver.at_wavelength(LAM)
    # Round-off as above: measured x <= 1.4e-13.
    assert (
        _rel(
            prepared.scatter_dipole(mats, 100.0, 500.0).ei,
            solver.scatter_dipole(LAM, 100.0, 500.0).ei,
        )
        < 1e-12
    )
    assert (
        _rel(
            prepared.scatter_dipole(mats, 100.0, 500.0, moment=(1.0, 0.5j)).ei,
            solver.scatter_dipole(LAM, 100.0, 500.0, moment=(1.0, 0.5j)).ei,
        )
        < 1e-12
    )
    fast = prepared.scatter_mode(mats, 0, 1, x_ref=0.0)
    dense = solver.scatter_mode(LAM, 0, 1, x_ref=0.0)
    assert _rel(fast.ei, dense.ei) < 1e-12
    assert fast.mode.beta == dense.mode.beta


@pytest.mark.parametrize(
    ("pitch", "nn", "name", "cond"),
    [
        (300.0, 100, "free", 4.4e2),
        (250.0, 200, "sin", 9.3e2),
        (240.0, 400, "free", 1.9e3),
        (234.0, 800, "free", 3.8e3),
    ],
)
def test_coupling_compression_holds_as_the_gap_closes(pitch, nn, name, cond):
    """Down to gap/a = 0.04, with `nn` from the gap envelope.

    A tight gap is where the coupling is least compressible (rank 170 of 800 here), so
    it is where a truncation error would first show: it would not scale with the
    conditioning. The bound does. Measured difference 2.6e-13 to 2.3e-12, a steady
    2.3–2.8 × cond(M)·eps, backward error 2e-16. So 5·cond·eps is ~2× headroom over
    round-off at each rung. `cond` is measured, not computed here: an SVD at N = 6400
    costs more than the test.
    """
    gs = [
        Geometry.gielis(
            rad=90.0,
            n_pts=nn,
            m=3,
            n1=4.0,
            n2=6.0,
            n3=6.0,
            x0=pitch * (p - 1.5),
            z0=140.0,
        )
        for p in range(4)
    ]
    mats = [Material.from_eps((-2.0 + 0.1j) * NC**2, n_clad=NC, pol=1)] * 4
    solver = ClusterBIESolver(Cluster(gs), mats, 1, BACKGROUNDS[name])
    fast = solver.at_wavelength(LAM).scatter(mats).ei
    assert _rel(fast, solver.scatter(LAM).ei) < 5.0 * cond * np.finfo(float).eps


@pytest.mark.parametrize("pol", [1, 2])
def test_fast_solve_matches_the_two_cylinder_addition_theorem(pol):
    """The independent anchor, reached through the new path.

    Same dimer and same bound as test_two_cylinder.py: measured 3.8e-15 there, a
    round-off floor, and this path adds only round-off.
    """
    radii, core = (180.0, 110.0), (2.0, 1.6)
    centres = np.array([[-450.0, 0.0], [520.0, 0.0]])
    gs = [
        Geometry.gielis(r, 60, m=0, x0=c[0])
        for r, c in zip(radii, centres, strict=True)
    ]
    mats = [Material(n, 1.0, pol=pol) for n in core]
    res = (
        ClusterBIESolver(Cluster(gs), mats, pol)
        .at_wavelength(633.0)
        .scatter(mats, 37.0)
    )
    amp, theta = res.far_field(361)
    ref = two_cylinder.scattering_amplitude(
        pol, radii, centres, [m.nc for m in mats], res.wnum_bg, 37.0, theta
    )
    assert np.max(np.abs(amp - ref)) / np.max(np.abs(ref)) < 2e-14


@pytest.mark.parametrize("pol", [1, 2])
def test_one_particle_has_no_coupling_and_equals_the_single_solver(pol):
    """The edge case with an empty small system: nothing may break or change."""
    g = Geometry.gielis(rad=90.0, n_pts=64, m=3, n1=4.0, n2=6.0, n3=6.0)
    mat = Material(n_core=2.0, pol=pol)
    prepared = ClusterBIESolver(Cluster([g]), [mat], pol).at_wavelength(600.0)
    assert prepared.coupling_ranks == (0,)
    # Same matrix, same right-hand side, and LU on it twice: round-off, measured below.
    assert _rel(prepared.scatter([mat]).ei, BIESolver(g, mat).scatter(600.0).ei) < 1e-13


def test_complex_wavelength_equals_the_dense_solve():
    """Non-negotiable 1: the fast path keeps complex wavenumbers working."""
    cl = Cluster(_mixed())
    mats = _mixed_materials(2)
    solver = ClusterBIESolver(cl, mats, 2)
    wavelength = 1550.0 + 40.0j
    # Round-off as above: measured x <= 1.4e-13 at real λ.
    assert (
        _rel(
            solver.at_wavelength(wavelength).scatter(mats).ei,
            solver.scatter(wavelength).ei,
        )
        < 1e-12
    )


def test_foreign_materials_are_refused():
    """A wrong count, polarisation or background index must not solve quietly."""
    cl = Cluster(_mixed())
    mats = _mixed_materials(1)
    prepared = ClusterBIESolver(cl, mats, 1).at_wavelength(LAM)
    with pytest.raises(ValueError, match="materials for"):
        prepared.scatter(mats[:2])
    with pytest.raises(ValueError, match="pol"):
        prepared.scatter(_mixed_materials(2))
    with pytest.raises(ValueError, match="built for n_clad"):
        prepared.scatter([Material(n_core=2.0, n_clad=1.0, pol=1)] * 4)


def _slab_quartet() -> tuple[ClusterEigenEps, complex, list[Geometry]]:
    """Four triangles over SiN, and an isolated-particle mode (a pole of D, not M)."""
    gs = [
        Geometry.gielis(
            rad=90.0,
            n_pts=100,
            m=3,
            n1=4.0,
            n2=6.0,
            n3=6.0,
            x0=600.0 * (p - 1.5),
            z0=140.0,
        )
        for p in range(4)
    ]
    mat = Material(n_core=1.5, n_clad=NC, pol=1)
    prepared = ClusterBIESolver(
        Cluster(gs), [mat] * 4, 1, BACKGROUNDS["sin"]
    ).at_wavelength(LAM)

    def isolated(e: complex) -> np.ndarray:
        # Principal root, as Material.nc takes: the same mode the driven path sees.
        return np.vstack([prepared._top[0], prepared._interior(0, np.sqrt(e), e)])

    bm = beyn_modes(
        isolated, -1.6 - 0.2j, -1.1 - 0.002j, n_quad_per_side=32, n_probe=12
    )
    return prepared, bm.eigenvalues[np.argmin(np.abs(bm.eigenvalues.imag))], gs


@pytest.mark.parametrize(
    ("distance", "unrefined_at_least"), [(1e-6, 1e-12), (1e-10, 1e-8), (1e-12, 1e-6)]
)
def test_refinement_rescues_the_cancellation_near_an_isolated_pole(
    monkeypatch, distance, unrefined_at_least
):
    """Near an isolated resonance x = a − R·y cancels; refining must restore round-off.

    The second half disables refinement and requires the error to come back. Without
    it this test could pass on a problem that never cancelled at all. It uses a plane
    wave on purpose: this one barely drives the resonance, so ‖a‖/‖x‖ stays at 0.26 and
    an a-priori guard on that ratio would never fire, as the first design did not.
    Measured with refinement 3.8e-13 to 4.3e-13, without it 4.4e-11, 5.4e-7, 5e-5. The
    dense matrix's condition number is near 2e4, so cond·eps is 4.4e-12 and 2e-12 is ~5×
    headroom over what round-off produced.
    """
    prepared, pole, gs = _slab_quartet()
    eps = pole + distance * np.exp(0.7j)
    mats = [Material.from_eps(eps * NC**2, n_clad=NC, pol=1)] * 4
    dense = ClusterBIESolver(Cluster(gs), mats, 1, BACKGROUNDS["sin"]).scatter(LAM)

    assert _rel(prepared.scatter(mats).ei, dense.ei) < 2e-12

    monkeypatch.setattr("pysie2d.cluster.MAX_REFINE_STEPS", 0)
    with pytest.warns(ClusterCancellationWarning):
        unrefined = prepared.scatter(mats).ei
    assert _rel(unrefined, dense.ei) > unrefined_at_least


def test_a_solve_that_cannot_be_rescued_warns():
    """Exactly on an isolated pole the small problems are singular: it must say so."""
    prepared, pole, _ = _slab_quartet()
    mats = [Material.from_eps(pole * NC**2, n_clad=NC, pol=1)] * 4
    with pytest.warns(ClusterCancellationWarning):
        prepared.scatter(mats)


def test_cost_per_material_beats_the_dense_solve_with_cached_rows():
    """Distinct particles, the fair case: no shared factorisation to flatter the ratio.

    Against the dense solve with exterior rows already cached, not `scatter`, so the
    assembly saving does not count. Measured 8× at 16 particles (study s1 B); asserting
    2× leaves margin for a loaded CI machine. Best of three, since timing is noisy.
    """
    gs = [
        Geometry.gielis(
            rad=90.0 + (p % 5),
            n_pts=100,
            m=3,
            n1=4.0,
            n2=6.0,
            n3=6.0,
            x0=600.0 * (p - 7.5),
            z0=140.0,
        )
        for p in range(16)
    ]
    cl = Cluster(gs)
    mats = [Material.from_eps((-2.0 + 0.1j) * NC**2, n_clad=NC, pol=1)] * 16
    solver = ClusterBIESolver(cl, mats, 1)
    prepared = solver.at_wavelength(LAM)
    dense = solver._assemble(LAM)
    rhs = solver._rhs_plane_wave(LAM, 0.0)

    def dense_solve():
        m = dense.copy()
        for p, g in enumerate(gs):
            sl = cl.slice(p)
            m[sl.start + g.n_pts : sl.stop, sl] = prepared._interior(
                p, mats[p].nc, mats[p].eps
            )
        return np.linalg.solve(m, rhs)

    def best(fn):
        times = []
        for _ in range(3):
            t0 = time.perf_counter()
            fn()
            times.append(time.perf_counter() - t0)
        return min(times)

    assert best(dense_solve) > 2.0 * best(lambda: prepared.scatter(mats))
