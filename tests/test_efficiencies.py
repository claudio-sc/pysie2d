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
# qext reads one sample, amp[nforw]. The index is an integer computed in floating
# point, and at the default n_angles = 3000 it lands just below one: under int()
# the sample was one grid step (0.12°) off forward, a fixed 4.5e-6 error at every
# nn. With round() qext is at round-off — measured ≤ 2.4e-15 over 500/600/800 nm,
# both polarisations, both circle branches, n_angles 1000/2001/3000. Same 1e-12
# bound as qsca, so the truncated index (4.5e-6) cannot pass.
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
