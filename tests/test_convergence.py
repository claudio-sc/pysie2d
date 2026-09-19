from conftest import N_CLAD, N_CORE, RAD, size_parameter
from pysie2d import BIESolver, Geometry, Material
from pysie2d.reference import mie


def test_convergence():
    # λ = 600 nm, TE. qsca against Mie must converge *spectrally* in nn: under
    # Kress–Martensen quadrature with analytic ddf/ddg the error falls
    # geometrically, faster at each step, until round-off (handoff §5.3).
    # Measured 2.9e-4 → 1.0e-6 → 1.2e-9 → 6.3e-13 over nn = 12, 16, 20, 24:
    # ratios 290, 830, 1900. The bar is ×100 per step, and a first-order scheme
    # manages ×1.33 over the same steps, so no algebraic rate can pass by
    # accident. The final bound 1e-11 is ~16× the measured 6.3e-13 and still
    # above the ~1e-15 floor, so the last rung is a real measurement.
    wavelength = 600.0
    x = size_parameter(wavelength)
    # Material.nc is the relative index; never divide it by n_clad again.
    ref = mie.efficiencies(x, complex(Material(N_CORE, N_CLAD).nc))["Q_sca_TE"]

    errors = []
    for nn in (12, 16, 20, 24):
        geom = Geometry.gielis(rad=RAD, n_pts=nn, m=0)
        mat = Material(n_core=N_CORE, n_clad=N_CLAD, pol=2)
        eff = BIESolver(geom, mat).scatter(wavelength=wavelength).efficiencies()
        errors.append(abs(eff["qsca"] - ref) / ref)

    for coarse, fine in zip(errors[:-1], errors[1:], strict=True):
        assert fine < coarse / 100.0
    assert errors[-1] < 1e-11
