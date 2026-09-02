from conftest import N_CLAD, N_CORE, RAD, size_parameter
from pysie2d import BIESolver, Geometry, Material
from pysie2d.reference import mie


def test_convergence():
    # λ = 600 nm, TE. Relative error of qsca vs Mie must shrink with nn.
    # Post-A20, qsca converges cleanly at first order in nn (measured ratio
    # ~2.00 per doubling here) rather than flooring on the far-field grid's
    # double-counted endpoint. A first-order quantity cannot drop by more
    # than the refinement factor itself — 8x from nn=40 to nn=320 — so the
    # bar is 7x, not the pre-A20 10x (which only passed because the floor
    # partially cancelled the nn-truncation error at nn=320 by coincidence,
    # not because the quantity actually dropped by an order of magnitude).
    # Measured 7.92x.
    wavelength = 600.0
    x = size_parameter(wavelength)
    # Material.nc is the relative index; never divide it by n_clad again.
    ref = mie.efficiencies(x, complex(Material(N_CORE, N_CLAD).nc))["Q_sca_TE"]

    n_values = [40, 80, 160, 320]
    errors = []
    for nn in n_values:
        geom = Geometry.gielis(rad=RAD, n_pts=nn, m=0)
        mat = Material(n_core=N_CORE, n_clad=N_CLAD, pol=2)
        eff = BIESolver(geom, mat).scatter(wavelength=wavelength).efficiencies()
        errors.append(abs(eff["qsca"] - ref) / ref)

    assert errors[-1] < errors[0] / 7.0
    for prev, nxt in zip(errors[1:], errors[2:], strict=False):
        assert nxt <= prev
