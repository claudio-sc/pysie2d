import numpy as np
import pytest

from conftest import N_CLAD, N_CORE, POL_TAG, size_parameter
from pysie2d import BIESolver, Geometry, Material
from pysie2d.reference import mie

# Tolerances. Under Kress–Martensen quadrature the boundary solve is at
# round-off by nn = 30 on this circle, so nn = 300 measures the observables,
# not the discretisation.
#
# qsca sums |amp|² over a uniform periodic angular grid — spectral on a smooth
# far field — and reaches round-off: measured ≤ 2.7e-15 over 500/600/800 nm,
# both polarisations and both circle branches. 1e-12 leaves ~400× for BLAS and
# libm differences and is still nine decades below the first-order v0.5 error.
RTOL_QSCA = 1e-12
# qext evaluates the amplitude at the exact forward angle, off the grid. It used
# to read a grid sample: one step off under int() (4.5e-6), and off forward for
# any incidence not on the grid (1.5e-7 at 37°). Measured ≤ 2.4e-15 over
# 500/600/800 nm, both polarisations, both circle branches. Same 1e-12 bound as
# qsca, so either grid read cannot pass.
RTOL_QEXT = 1e-12
# qabs = qext − qsca on the lossy particle: measured ≤ 5.6e-16. Bound 1e-12.
RTOL_QABS = 1e-12
# Lossless energy conservation |qext − qsca|/qext: measured ≤ 1.0e-15. Bound 1e-12.
RTOL_ENERGY = 1e-12


#  m=0 is the trivial superformula branch (fact_n2 = fact_n3 = 0, so the
# analytic ddf/ddg's tan/cot terms never fire); m=4, n1=n2=n3=2 is the same
# exact circle routed through the non-trivial branch. Both must match Mie, or
# a bug confined to the non-trivial branch would pass silently.
CIRCLE_BRANCHES = [
    {"m": 0},
    {"m": 4, "n1": 2.0, "n2": 2.0, "n3": 2.0},
]


@pytest.mark.parametrize("wavelength", [500.0, 600.0, 800.0])
@pytest.mark.parametrize("pol", [1, 2])
@pytest.mark.parametrize("branch", CIRCLE_BRANCHES, ids=["m0", "m4-n2"])
def test_efficiencies_match_mie(circle, branch, wavelength, pol):
    geom = circle(300, **branch)
    mat = Material(n_core=N_CORE, n_clad=N_CLAD, pol=pol)
    result = BIESolver(geom, mat).scatter(wavelength=wavelength)
    eff = result.efficiencies()

    x = size_parameter(wavelength)
    # mat.nc is already the *relative* index n_core/n_clad — the m of Mie
    # theory. Dividing by n_clad again double-counts the background; it was
    # invisible only because this fixture runs at n_clad = 1.
    m = complex(mat.nc)
    ref = mie.efficiencies(x, m)
    tag = POL_TAG[pol]

    assert eff["qsca"] == pytest.approx(ref[f"Q_sca_{tag}"], rel=RTOL_QSCA)
    assert eff["qext"] == pytest.approx(ref[f"Q_ext_{tag}"], rel=RTOL_QEXT)


@pytest.mark.parametrize("pol", [1, 2])
def test_absorbing_particle(circle, pol):
    # Lossy particle: qabs = qext - qsca > 0 and must match the analytic
    # absorption efficiency.  qext (forward amplitude) and qsca (angular
    # integral) are computed independently, so their difference matching the
    # analytic absorption is a strong check of the complex-permittivity path.
    wavelength = 600.0
    geom = circle(300)
    mat = Material(n_core=N_CORE, n_clad=N_CLAD, pol=pol, epsi=0.5)
    result = BIESolver(geom, mat).scatter(wavelength=wavelength)
    eff = result.efficiencies()

    x = size_parameter(wavelength)
    # mat.nc is already the *relative* index n_core/n_clad — the m of Mie
    # theory. Dividing by n_clad again double-counts the background; it was
    # invisible only because this fixture runs at n_clad = 1.
    m = complex(mat.nc)
    ref = mie.efficiencies(x, m)
    tag = POL_TAG[pol]

    assert eff["qabs"] > 0.0
    assert eff["qabs"] == pytest.approx(ref[f"Q_abs_{tag}"], rel=RTOL_QABS)


@pytest.mark.parametrize("pol", [1, 2])
def test_energy_conservation_lossless(circle, pol):
    wavelength = 600.0
    geom = circle(300)
    mat = Material(n_core=N_CORE, n_clad=N_CLAD, pol=pol, epsi=0.0)
    result = BIESolver(geom, mat).scatter(wavelength=wavelength)
    eff = result.efficiencies()

    assert abs(eff["qext"] - eff["qsca"]) / eff["qext"] < RTOL_ENERGY


@pytest.mark.parametrize("pol", [1, 2])
def test_qsca_is_independent_of_incidence_angle_on_the_circle(pol):
    """A20's discriminating check: qsca must not care where the grid closes.

    A circle is isotropic, so qsca cannot depend on the incidence angle no
    matter where the far-field angular grid's duplicated endpoint falls
    relative to the forward peak. Before A20 it did — rotating incidence by
    90 degrees moved qsca by 1.13e-3 at n_angles=3000 (the ~1.8e-3 floor this
    file's tolerances used to carry). Bar 1e-10: what remains after dropping
    the duplicate is round-off in a uniform-grid quadrature of a smooth
    periodic integrand, which for the circle's few-mode far field is many
    orders below the discretisation floor above — measured 3.4e-16.
    """
    geom = Geometry.gielis(rad=200.0, n_pts=300, m=0)
    mat = Material(n_core=N_CORE, n_clad=N_CLAD, pol=pol)
    qscas = [
        BIESolver(geom, mat)
        .scatter(wavelength=600.0, angle=angle)
        .efficiencies()["qsca"]
        for angle in (0.0, 37.0, 90.0, 123.456)
    ]
    qscas = np.array(qscas)
    assert (qscas.max() - qscas.min()) / qscas.mean() < 1.0e-10


@pytest.mark.parametrize("pol", [1, 2])
def test_qext_is_exact_at_off_grid_incidence(pol):
    """qext must read the forward amplitude, not the nearest grid sample to it.

    A circle's Q_ext is isotropic, so Mie is the reference at any incidence.
    37° and 123.456° fall between far-field grid angles at every n_angles
    used: reading the nearest sample was 1.5e-7 off at n_angles = 3000 and
    8.3e-6 at 500, so the 1e-12 bar (RTOL_QEXT) cannot pass on a grid sample.
    n_angles = 17 makes the grid step 22.5°, so qext is checked independent of it.
    """
    geom = Geometry.gielis(rad=200.0, n_pts=300, m=0)
    mat = Material(n_core=N_CORE, n_clad=N_CLAD, pol=pol)
    ref = mie.efficiencies(size_parameter(600.0), complex(mat.nc))
    for angle in (37.0, 123.456):
        result = BIESolver(geom, mat).scatter(wavelength=600.0, angle=angle)
        for n_angles in (17, 500):
            assert result.efficiencies(n_angles)["qext"] == pytest.approx(
                ref[f"Q_ext_{POL_TAG[pol]}"], rel=RTOL_QEXT
            )


@pytest.mark.parametrize("pol", [1, 2])
def test_default_angular_grid_resolves_a_high_multipole_particle(pol):
    """The default n_angles must integrate a strongly multipolar far field.

    rad = 2000 nm at 500 nm, n_core = 2: size parameter 25, so |amp|² carries
    many angular harmonics. qsca against Mie at the default grid measured
    1.1e-15 / 1.3e-15 (TE/TM) at nn = 400; 1e-12 is RTOL_QSCA. The test can
    fail: TE measured 6.1e-1 at 33 angles and 4.1e-6 at 65, round-off from 129
    — so a default shrunk toward the grid this far field needs fails here.
    """
    geom = Geometry.gielis(rad=2000.0, n_pts=400, m=0)
    mat = Material(n_core=2.0, n_clad=N_CLAD, pol=pol)
    result = BIESolver(geom, mat).scatter(wavelength=500.0)
    ref = mie.efficiencies(result.size_parameter, complex(mat.nc))
    assert result.efficiencies()["qsca"] == pytest.approx(
        ref[f"Q_sca_{POL_TAG[pol]}"], rel=RTOL_QSCA
    )


@pytest.mark.parametrize("pol", [1, 2])
def test_cross_sections_and_efficiencies_agree(circle, pol):
    """C = Q · 2·rad exactly, pinning the two entry points to one quadrature.

    `cross_sections()` and `efficiencies()` are separate computations of the
    same far-field integrals; the identity C = Q · 2·rad is algebraic, so the
    only discrepancy admissible is round-off in re-summing the same terms —
    hence 1e-14, not a convergence-order bound. A lossy particle exercises all
    three of sca/ext/abs with none of them incidentally zero.
    """
    geom = circle(300)
    mat = Material(n_core=N_CORE, n_clad=N_CLAD, pol=pol, epsi=0.5)
    result = BIESolver(geom, mat).scatter(wavelength=600.0, angle=37.0)

    eff = result.efficiencies()
    cs = result.cross_sections()
    width = 2.0 * geom.rad
    for q_key, c_key in (("qsca", "c_sca"), ("qext", "c_ext"), ("qabs", "c_abs")):
        assert np.isclose(cs[c_key], eff[q_key] * width, rtol=1e-14, atol=0.0)
