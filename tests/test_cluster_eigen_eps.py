"""Fast cluster solves at one wavelength, and cluster eigenpermittivities.

`ClusterEigenEps` solves the dense coupled system of conventions §14 by compressing
the neighbour coupling, so for driven solves the dense `ClusterBIESolver` is the
right reference: this is linear algebra, not new physics. Tolerances cite the
measured values in docs/design/studies/cluster-reduction/README.md.
"""

import time

import numpy as np
import pytest
from scipy.optimize import newton
from scipy.special import h1vp, hankel1, jn_zeros, jv, jvp

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
from pysie2d.beyn import beyn_modes, newton_refine
from pysie2d.kernels import _interior_rows, assemble_matrix
from pysie2d.reference import two_cylinder
from pysie2d.reference.mie import an, bn

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
    solved another way. Measured worst `x` 1.4e-13 and far field 1.8e-15, both
    round-off. 1e-12 and 1e-14 leave 5–7× headroom. A wrong rank, offset or dropped
    substrate block misses by orders of magnitude, so this cannot pass by accident.
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
        # Round-off, as in the test above: measured x <= 1.4e-13, so ~7× headroom.
        assert _rel(prepared.scatter(mats).ei, dense.ei) < 1e-12


def test_dipole_and_guided_mode_excitations_equal_the_dense_solve():
    """The other two excitations share the solve; their right-hand sides must agree."""
    cl = Cluster(_mixed())
    mats = _mixed_materials(1)
    solver = ClusterBIESolver(cl, mats, 1, BACKGROUNDS["sin"])
    prepared = solver.at_wavelength(LAM)
    # Round-off: measured 7.5e-14, 6.9e-14 and 5.3e-14 below, so ~7× headroom.
    bound = 5e-13
    assert (
        _rel(
            prepared.scatter_dipole(mats, 100.0, 500.0).ei,
            solver.scatter_dipole(LAM, 100.0, 500.0).ei,
        )
        < bound
    )
    assert (
        _rel(
            prepared.scatter_dipole(mats, 100.0, 500.0, moment=(1.0, 0.5j)).ei,
            solver.scatter_dipole(LAM, 100.0, 500.0, moment=(1.0, 0.5j)).ei,
        )
        < bound
    )
    fast = prepared.scatter_mode(mats, 0, 1, x_ref=0.0)
    dense = solver.scatter_mode(LAM, 0, 1, x_ref=0.0)
    assert _rel(fast.ei, dense.ei) < bound
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

    Same dimer and same bound as test_two_cylinder.py. Measured 3.5e-15 and 3.7e-15,
    the same floor as the dense path's 3.4e-15 and 3.8e-15: this path adds nothing.
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
    # The same matrix and right-hand side. Measured 3.6e-15 and 5.7e-15, ~9× headroom.
    assert _rel(prepared.scatter([mat]).ei, BIESolver(g, mat).scatter(600.0).ei) < 5e-14


def test_complex_wavelength_equals_the_dense_solve():
    """Non-negotiable 1: the fast path keeps complex wavenumbers working."""
    cl = Cluster(_mixed())
    mats = _mixed_materials(2)
    solver = ClusterBIESolver(cl, mats, 2)
    wavelength = 1550.0 + 40.0j
    # Round-off: measured 6.1e-14, so ~8× headroom.
    fast = solver.at_wavelength(wavelength).scatter(mats).ei
    assert _rel(fast, solver.scatter(wavelength).ei) < 5e-13


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


# --- Eigenpermittivities ------------------------------------------------------------


RAD, LAM_M = 100.0, 600.0
X = 2 * np.pi * RAD / LAM_M
BOX = {1: (-2.0 - 1.5j, -0.2 - 0.25j), 2: (11.0 - 1.4j, 13.8 - 0.1j)}


def _mie_denominator(eps: complex, n: int, x: float, pol: int) -> complex:
    """Zeros are a circle's modes (B&H denominators); parity makes √ε's sign moot."""
    m = np.sqrt(complex(eps))
    if pol == 1:
        return jvp(n, m * x) * hankel1(n, x) - m * jv(n, m * x) * h1vp(n, x)
    return m * jvp(n, m * x) * hankel1(n, x) - jv(n, m * x) * h1vp(n, x)


def _winding(
    n: int, x: float, pol: int, lo: complex, hi: complex, m: int = 4000
) -> int:
    """Argument principle: the denominator has no poles, so this counts its zeros."""
    t = np.linspace(0, 1, m, endpoint=False)
    w, h = hi.real - lo.real, hi.imag - lo.imag
    path = np.concatenate(
        [lo + w * t, lo + w + 1j * h * t, hi - w * t, lo + 1j * h - 1j * h * t]
    )
    d = np.array([_mie_denominator(e, n, x, pol) for e in path])
    phase = np.unwrap(np.angle(np.append(d, d[0])))
    return round((phase[-1] - phase[0]) / (2 * np.pi))


def _circles(d: float = 0.0, nn: int = 128) -> list[Geometry]:
    if d == 0.0:
        return [Geometry.gielis(rad=RAD, n_pts=nn, m=0)]
    return [Geometry.gielis(rad=RAD, n_pts=nn, m=0, x0=s * d / 2) for s in (-1, 1)]


def _prepared(gs, pol, wavelength=LAM_M, n_clad=1.0, background=None):
    mats = [Material(n_core=1.5, n_clad=n_clad, pol=pol)] * len(gs)
    solver = ClusterBIESolver(Cluster(gs), mats, pol, background)
    return solver.at_wavelength(wavelength)


def _graf(eps: complex, pol: int, d: float, n_max: int = 12) -> np.ndarray:
    """Two equal circles: I − s·T from Mie coefficients and Graf, with no BIE in it."""
    n = np.arange(-n_max, n_max + 1)
    s = -(an if pol == 1 else bn)(np.abs(n), X, 1j * np.sqrt(-complex(eps)))
    mat = np.eye(2 * n.size, dtype=complex)
    k = 2 * np.pi / LAM_M
    for el, j, phi in ((0, 1, np.pi), (1, 0, 0.0)):
        nm = n[None, :] - n[:, None]
        mat[el * n.size : (el + 1) * n.size, j * n.size : (j + 1) * n.size] -= s[
            :, None
        ] * (hankel1(nm, k * d) * np.exp(1j * nm * phi))
    return mat


def _polish(f, eigs, lo, hi) -> np.ndarray:
    out = []
    for e in eigs:
        v = np.linalg.svd(f(e))[2][-1].conj()

        def df(z, f=f, h=1e-6):
            return (f(z + h) - f(z - h)) / (2 * h)

        out.append(newton_refine(f, df, e, v, lo, hi, tol=1e-13).eigenvalue)
    return np.sort_complex(np.array(out))


def _dense_operator(prepared: ClusterEigenEps):
    """M(ε) from `_assemble`, so the reduced path cannot agree by sharing Woodbury."""
    solver, cl = prepared.solver, prepared.solver.cluster
    full = solver._assemble(prepared.wavelength)

    def op(e):
        m = full.copy()
        for p, g in enumerate(cl.geometries):
            sl = cl.slice(p)
            m[sl.start + g.n_pts : sl.stop, sl] = prepared._interior(
                p, 1j * np.sqrt(-complex(e)), e
            )
        return m

    return op


def _distance(a: np.ndarray, b: np.ndarray) -> float:
    return max(float(np.min(np.abs(b - x))) for x in a)


@pytest.mark.parametrize("pol", [1, 2])
def test_one_circle_reproduces_the_analytic_mie_modes_and_their_count(pol):
    """The closed form, with the number of modes settled by the argument principle.

    Winding the denominators counts zeros independently of any contour eigensolver, so a
    missed or spurious mode fails the count even if the found ones are perfect. |n| ≥ 1
    orders are doubly degenerate. Measured 8.7e-16 (TM) and 7.8e-15 (TE), round-off;
    5e-14 is ~6× headroom.
    """
    lo, hi = BOX[pol]
    expected = []
    for n in range(13):
        count = _winding(n, X, pol, lo, hi)
        if count:
            grid = (
                lo.real
                + (hi.real - lo.real) * np.linspace(0, 1, 200)[:, None]
                + 1j * (lo.imag + (hi.imag - lo.imag) * np.linspace(0, 1, 60))[None, :]
            )
            values = np.abs(np.vectorize(_mie_denominator)(grid, n, X, pol))
            roots: list[complex] = []
            for idx in np.argsort(values, axis=None)[:400]:
                start = grid[np.unravel_index(idx, grid.shape)]
                r = complex(
                    newton(
                        lambda e, n=n: _mie_denominator(e, n, X, pol),
                        start,
                        tol=1e-14,
                        maxiter=200,
                    )
                )
                inside = lo.real < r.real < hi.real and lo.imag < r.imag < hi.imag
                if inside and all(abs(r - q) > 1e-6 for q in roots):
                    roots.append(r)
                if len(roots) == count:
                    break
            assert len(roots) == count
            expected += [r for r in roots for _ in range(2 if n else 1)]
    expected = np.array(expected)

    res = _prepared(_circles(nn=64), pol).eigenpermittivities(
        lo, hi, n_quad_per_side=48
    )

    assert res.eps.size == expected.size
    assert _distance(res.eps, expected) < 5e-14
    assert _distance(expected, res.eps) < 5e-14


@pytest.mark.parametrize(
    ("pol", "d", "bound"),
    [(2, 1000.0, 1e-13), (1, 1000.0, 1e-13), (2, 400.0, 2e-13), (1, 400.0, 2e-13)],
)
def test_two_circles_match_the_graf_addition_theorem(pol, d, bound):
    """An independent answer for coupled modes: Mie coefficients and Graf, no BIE.

    Measured 1.3e-14 (TE) and 9.4e-16 (TM) at d = 1000, 2.5e-14 and 2.9e-14 at d = 400.
    The floor is Graf's: truncation at order 10 against 12 differs by 1.3e-14 and 6e-14
    in TE and 4.9e-12 in TM at d = 400, which is why 12 is used. At d = 300 it reaches
    1e-8 and says nothing about the BIE, so that gap is not tested. Bounds are ~7×.
    """
    lo, hi = BOX[pol]
    res = _prepared(_circles(d), pol).eigenpermittivities(lo, hi, n_quad_per_side=48)
    graf = beyn_modes(
        lambda e: _graf(e, pol, d),
        lo,
        hi,
        n_quad_per_side=48,
        n_probe=24,
        max_workers=1,
    )
    reference = _polish(lambda e: _graf(e, pol, d), graf.eigenvalues, lo, hi)

    assert res.eps.size == reference.size > 0
    assert _distance(res.eps, reference) < bound
    assert _distance(reference, res.eps) < bound


@pytest.mark.parametrize(("pol", "h"), [(2, 200.0), (2, 500.0), (1, 200.0), (1, 500.0)])
def test_modes_over_pec_are_modes_of_the_mirrored_pair(pol, h):
    """A perfect mirror is its own image: an anchor on the reflected blocks in ε.

    Every mode over PEC must be a mode of the circle and its image in free space, and
    fewer of them, since the mirror selects one parity. Measured ≤ 3.6e-15, round-off,
    so 1e-14 is ~3× headroom. A wrong image sign selects the other parity, whose modes
    are measured 0.095 to 0.56 away, so this cannot pass by accident.
    """
    lo, hi = BOX[pol]
    circle = Geometry.gielis(rad=RAD, n_pts=128, m=0, z0=h)
    image = Geometry.gielis(rad=RAD, n_pts=128, m=0, z0=-h)
    over_pec = _prepared([circle], pol, background=HalfSpace.pec())
    pec = over_pec.eigenpermittivities(lo, hi, n_quad_per_side=40, n_probe=20)
    pair = _prepared([circle, image], pol).eigenpermittivities(
        lo, hi, n_quad_per_side=40, n_probe=20
    )

    assert 0 < pec.eps.size < pair.eps.size
    assert _distance(pec.eps, pair.eps) < 1e-14


def _rotated_triangle() -> Geometry:
    g = Geometry.gielis(rad=90.0, n_pts=160, m=3, n1=4.0, n2=6.0, n3=6.0)
    c, s = np.cos(np.deg2rad(17.0)), np.sin(np.deg2rad(17.0))

    def rot(x, z):
        return c * x - s * z, s * x + c * z

    f, z = rot(g.f, g.g)
    df, dz = rot(g.df, g.dg)
    ddf, ddz = rot(g.ddf, g.ddg)
    return Geometry(
        f=f, g=z + 140.0, df=df, dg=dz, ddf=ddf, ddg=ddz, rad=g.rad, z0=140.0
    )


def test_mode_over_a_slab_is_a_pole_of_the_driven_response():
    """The one check over a Stack: not external, and the docs say so (spec §7).

    Passive driven solves near the real axis, with no contour and no branch choice, are
    fitted to one pole plus a smooth remainder: ε·f = p·f + Σ b_k·ε^k is linear in p.
    That driven path is the one v1.3 compared with MEEP. Measured 5.9e-7 at Im ε = 1e-4
    with 21 solves; the floor is the fit's model error, moving with the window (1.5e-6
    at twice the width), so 5e-6 is ~8× headroom, far below the 8.6e-3 by which a mode
    on the real axis would miss.
    """
    geom = _rotated_triangle()
    stack = BACKGROUNDS["sin"]
    prepared = _prepared([geom], 1, LAM, NC, stack)
    res = prepared.eigenpermittivities(
        (-1.6 - 0.2j) * NC**2, (-1.1 - 0.002j) * NC**2, n_probe=12
    )
    target = res.eps[np.argmin(np.abs(res.eps.imag))] / NC**2

    def guided(e: complex) -> complex:
        mat = Material.from_eps(e * NC**2, n_clad=NC, pol=1)
        driven = BIESolver(geom, mat, background=stack).scatter(LAM, 0.0)
        return driven.mode_amplitudes(x_ref=0.0).s_plus[0]

    half = 0.03
    es = target.real + np.linspace(-half, half, 21) + 1e-4j
    fs = np.array([guided(e) for e in es])
    t = (es - target.real) / half
    design = np.column_stack([fs] + [t**k for k in range(5)])
    pole = np.linalg.lstsq(design, es * fs, rcond=None)[0][0]

    assert abs(pole - target) < 5e-6


def test_mode_over_a_dielectric_half_space_is_a_pole_of_the_driven_response():
    """The same check as over a slab, on a near field: no guided mode to read here.

    The window must exclude the neighbouring mode at Re ε = −0.989. At ±0.015 the fit
    measures 4.7e-7, and at ±0.03 it degrades to 6.2e-6 as that neighbour enters, which
    is the model error setting the floor. 5e-6 is ~10× headroom on it. A pole fit needs
    a mode near the passive axis: at Im ε = −0.69 it misses by 0.07 and says nothing.
    """
    geom = _rotated_triangle()
    glass = BACKGROUNDS["glass"]
    res = _prepared([geom], 1, LAM, NC, glass).eigenpermittivities(
        (-0.99 - 0.1j) * NC**2, (-0.8 - 0.002j) * NC**2
    )
    target = res.eps[np.argmax(res.eps.real)] / NC**2

    def field(e: complex) -> complex:
        mat = Material.from_eps(e * NC**2, n_clad=NC, pol=1)
        driven = BIESolver(geom, mat, background=glass).scatter(LAM, 13.0)
        return driven.eval_field(np.array([400.0]), np.array([500.0]))[0]

    half = 0.015
    es = target.real + np.linspace(-half, half, 21) + 1e-4j
    fs = np.array([field(e) for e in es])
    t = (es - target.real) / half
    design = np.column_stack([fs] + [t**k for k in range(5)])
    pole = np.linalg.lstsq(design, es * fs, rcond=None)[0][0]

    assert abs(pole - target) < 5e-6


@pytest.mark.parametrize("pol", [1, 2])
def test_reduced_contour_agrees_with_the_dense_operator(pol):
    """G against M on one box: same count, same polished values.

    Both are built from the same rows, so this checks the Woodbury reduction of the
    eigenproblem, not the physics. Measured 5.0e-16 (TM) and 3.6e-15 (TE), round-off
    that scales with |ε|, so 2e-14 is ~5× headroom.
    """
    lo, hi = BOX[pol]
    prepared = _prepared(_circles(400.0), pol)
    res = prepared.eigenpermittivities(lo, hi, n_quad_per_side=48)
    op = _dense_operator(prepared)
    dense = beyn_modes(op, lo, hi, n_quad_per_side=48, n_probe=24, max_workers=1)
    reference = _polish(op, dense.eigenvalues, lo, hi)

    assert res.eps.size == reference.size > 0
    assert _distance(res.eps, reference) < 2e-14


def test_converged_mode_is_a_null_vector_of_the_dense_matrix():
    """The returned vector really solves M·x = 0, in the conventions §14 layout."""
    lo, hi = BOX[2]
    prepared = _prepared(_circles(400.0), 2)
    res = prepared.eigenpermittivities(lo, hi, n_quad_per_side=48)
    op = _dense_operator(prepared)

    assert res.converged.all()
    for k, eps in enumerate(res.eps):
        m = op(eps)
        # ‖M x‖/‖M‖ for unit x. Measured 1.1e-16, round-off, so ~9× headroom. A wrong
        # sign on one particle's sub-vector misses by order 1.
        assert np.linalg.norm(m @ res.vectors[:, k]) / np.linalg.norm(m) < 1e-15


def test_high_q_mode_is_not_hidden_by_the_reduction():
    """The worst case for G: a resonance so narrow that pole and zero nearly cancel.

    A whispering-gallery pair of two circles at Im ε_iso ≈ 5.6e-13, Q ≈ 7e12. G⁻¹'s
    residue is proportional to the mode's field at its neighbour, which is what becomes
    small here, so if truncation at 1e-13 could lose a mode it would be this one: it
    fails at a rank tolerance of 1e-7. Both operators must find all four. Agreement
    measured 8e-15, against an Im ε of 5.6e-13, so 5e-14 is ~6× headroom and stays
    meaningful.
    """
    x = 2 * np.pi * 1000.0 / LAM_M
    n = 24

    def denom(e):
        return _mie_denominator(e, n, x, 2) / hankel1(n, x)

    iso = complex(newton(denom, complex((jn_zeros(n, 1)[0] / x) ** 2), tol=1e-15))
    assert -1e-11 < iso.imag < 0  # otherwise this tests nothing

    w = 8 * abs(iso.imag)
    lo, hi = iso - w - 1j * w, complex(iso.real + w, iso.imag / 2)
    nn = int(np.ceil(12 * np.sqrt(iso.real) * x))
    gs = [Geometry.gielis(rad=1000.0, n_pts=nn, m=0, x0=s * 3000.0) for s in (-1, 1)]
    prepared = _prepared(gs, 2)
    reduced = beyn_modes(prepared._reduced_at, lo, hi, n_quad_per_side=24, n_probe=10)
    dense = beyn_modes(
        _dense_operator(prepared), lo, hi, n_quad_per_side=24, n_probe=10
    )

    assert reduced.eigenvalues.size == dense.eigenvalues.size == 4
    assert _distance(reduced.eigenvalues, dense.eigenvalues) < 5e-14


def test_contour_inputs_that_cannot_work_are_refused():
    """A box on the branch cut, on ε = 0, or at a complex wavelength must not run."""
    prepared = _prepared(_circles(nn=48), 2)
    with pytest.raises(ValueError, match="branch cut"):
        prepared.eigenpermittivities(1.0 - 1.0j, 4.0 + 1.0j)
    with pytest.raises(ValueError, match="branch cut"):
        prepared.eigenpermittivities(-1.0 - 1.0j, 1.0 + 1.0j)
    with pytest.raises(ValueError, match="real wavelength"):
        _prepared(_circles(nn=48), 2, 600.0 + 10.0j).eigenpermittivities(
            1.0 - 1.0j, 4.0 - 0.1j
        )


def test_public_permittivity_is_absolute_and_feeds_back_into_a_material():
    """D9's one conversion point: results must not be background-relative.

    At n_clad = 1.444 the same physical mode appears at ε·n_clad². The returned value,
    given to `Material.from_eps`, must make that material's own interior rows singular
    with the exterior rows. A missing or doubled n_clad² would miss by a factor of 2.
    Compared with a vacuum computation at the same k_bg, because the BIE sees only
    background-relative quantities (conventions §2). Measured 5.3e-15, round-off, so
    5e-14 is ~9× headroom.
    """
    lo, hi = BOX[2]
    vacuum = _prepared(_circles(nn=64), 2).eigenpermittivities(
        lo, hi, n_quad_per_side=48
    )
    clad = _prepared(_circles(nn=64), 2, LAM_M * NC, NC).eigenpermittivities(
        lo * NC**2, hi * NC**2, n_quad_per_side=48
    )

    assert clad.eps.size == vacuum.eps.size > 0
    assert _distance(clad.eps / NC**2, vacuum.eps) < 5e-14

    mat = Material.from_eps(clad.eps[0], n_clad=NC, pol=2)
    assert mat.eps == pytest.approx(vacuum.eps[0], abs=5e-14)


def test_a_box_too_close_to_a_mode_raises_rather_than_answering():
    """The failure is loud. Top edge at Im ε = −0.15 meets a TM mode at d = 400."""
    prepared = _prepared(_circles(400.0), 1)
    with pytest.raises(ValueError, match="rank gap"):
        prepared.eigenpermittivities(
            -2.0 - 1.5j, -0.2 - 0.15j, n_quad_per_side=48, n_probe=24
        )
