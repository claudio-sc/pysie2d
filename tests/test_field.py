import numpy as np

from conftest import N_CLAD, N_CORE, RAD
from pysie2d import BIESolver, Geometry, Material


def test_scattered_field_far_consistency():
    # A 2-D outgoing scattered field decays like 1/sqrt(k·r). Comparing the
    # RMS amplitude on two large circles, the ratio must match sqrt(r1/r2).
    wavelength = 600.0
    geom = Geometry.gielis(rad=RAD, n_pts=300, m=0)
    result = BIESolver(geom, Material(n_core=N_CORE, n_clad=N_CLAD, pol=2)).scatter(
        wavelength=wavelength
    )

    angles = np.linspace(0.0, 2.0 * np.pi, 64, endpoint=False)

    def rms_amplitude(radius: float) -> float:
        x = radius * np.cos(angles)
        z = radius * np.sin(angles)
        return float(np.sqrt(np.mean(np.abs(result.eval_field(x, z)) ** 2)))

    r1 = 50.0 * RAD
    r2 = 100.0 * RAD
    ratio = rms_amplitude(r2) / rms_amplitude(r1)
    expected = np.sqrt(r1 / r2)

    assert abs(ratio / expected - 1.0) < 0.05
