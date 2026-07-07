from conftest import N_CLAD, N_CORE, RAD, size_parameter
from pysie2d import BIESolver, Geometry, Material
from pysie2d.reference import mie


def test_convergence():
    # λ = 600 nm, TE. Relative error of qsca vs Mie must shrink with nn.
    # The exact convergence order depends on the quadrature, so we assert the
    # robust shape only: an order-of-magnitude drop from the coarsest to the
    # finest grid, and monotone non-increase after the first point.
    wavelength = 600.0
    x = size_parameter(wavelength)
    ref = mie.efficiencies(x, N_CORE / N_CLAD)["Q_sca_TE"]

    n_values = [40, 80, 160, 320]
    errors = []
    for nn in n_values:
        geom = Geometry.gielis(rad=RAD, n_pts=nn, m=0)
        mat = Material(n_core=N_CORE, n_clad=N_CLAD, pol=2)
        eff = BIESolver(geom, mat).scatter(wavelength=wavelength).efficiencies()
        errors.append(abs(eff["qsca"] - ref) / ref)

    assert errors[-1] < errors[0] / 10.0
    for prev, nxt in zip(errors[1:], errors[2:], strict=False):
        assert nxt <= prev
