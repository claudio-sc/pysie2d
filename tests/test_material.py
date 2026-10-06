import numpy as np
import pytest

from conftest import POL_TAG
from pysie2d import BIESolver, Geometry, Material
from pysie2d.reference import mie


@pytest.mark.parametrize("n_clad", [1.0, 1.33])
@pytest.mark.parametrize("n_core", [1.5, 2.0])
def test_from_eps_of_n_squared_equals_n_core(n_core, n_clad):
    # The two constructors must produce the same operator inputs when ε = n².
    # Round-off only: nc is a square root of the same complex number and eps
    # differs by one rounding of n·n against n²/n_clad² (measured ≤ 2.2e-16).
    a = Material(n_core=n_core, n_clad=n_clad)
    b = Material.from_eps(n_core**2, n_clad=n_clad)
    assert abs(a.nc - b.nc) < 1e-15
    assert abs(a.eps - b.eps) < 1e-15


def test_from_eps_expresses_negative_real_part():
    # n_core is real, so n_core² ≥ 0 and Material(...) can never give Re ε < 0.
    # The test cannot pass by accident: a from_eps that dropped epsr_abs would
    # return Re ε = |ε| > 0.
    mat = Material.from_eps(-18.3 + 0.48j, n_clad=1.33)
    assert mat.eps == pytest.approx((-18.3 + 0.48j) / 1.33**2, rel=1e-15)


# Silver at 633 nm (ε = −18.3 + 0.48i), rad = 50 nm, nn = 100 on the circle.
# Measured against Mie: qsca/qext ≤ 1.4e-15 in both polarisations, so 1e-14
# (10×). qabs = qext − qsca is a difference of near-equal numbers: measured
# ≤ 7.3e-14 relative, so 1e-12 (10×, rounded up).
RTOL_Q = 1e-14
RTOL_QABS = 1e-12


@pytest.mark.parametrize("pol", [1, 2])
def test_silver_cylinder_matches_mie(pol):
    mat = Material.from_eps(-18.3 + 0.48j, pol=pol)
    geom = Geometry.gielis(rad=50.0, n_pts=100, m=0)
    eff = BIESolver(geom, mat).scatter(wavelength=633.0).efficiencies()
    ref = mie.efficiencies(2 * np.pi * 50.0 / 633.0, complex(mat.nc))
    tag = POL_TAG[pol]
    assert eff["qsca"] == pytest.approx(ref[f"Q_sca_{tag}"], rel=RTOL_Q)
    assert eff["qext"] == pytest.approx(ref[f"Q_ext_{tag}"], rel=RTOL_Q)
    assert eff["qabs"] == pytest.approx(ref[f"Q_abs_{tag}"], rel=RTOL_QABS)
