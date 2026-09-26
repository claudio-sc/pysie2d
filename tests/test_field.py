import numpy as np
import pytest

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


def _mie_interior(mat: Material, wavelength: float, rad: float, x, z):
    """Analytic interior field of a circle under incidence exp(−i k_bg z).

    `c_n = (−i)ⁿ [J_n(x)H_n'(x) − J_n'(x)H_n(x)] / [J_n(mx)H_n'(x) − μ J_n'(mx)H_n(x)]`
    with μ = m in TE and 1/m in TM — continuity of ψ and of (1/ε^p)·∂ψ/∂n,
    solved in closed form, independent of the BIE.
    """
    from scipy.special import h1vp, hankel1, jv, jvp

    k = mat.wnum_bg(wavelength)
    m = mat.nc
    xs = k * rad
    mu = m if mat.pol == 2 else 1.0 / m
    n = np.arange(-40, 41)[:, None]
    c = (
        (-1j) ** n
        * (jv(n, xs) * h1vp(n, xs) - jvp(n, xs) * hankel1(n, xs))
        / (jv(n, m * xs) * h1vp(n, xs) - mu * jvp(n, m * xs) * hankel1(n, xs))
    )
    r = np.hypot(x, z)
    th = np.arctan2(x, z)
    return np.sum(c * jv(n, m * k * r) * np.exp(1j * n * th), axis=0)


# Interior points ≥ 5 node spacings (2π·300/100 ≈ 19 nm) inside the boundary.
_R_IN = np.array([0.0, 50.0, 120.0, 180.0, 200.0])
_TH_IN = np.array([0.0, 0.7, 2.0, 3.0, 4.5])


@pytest.mark.parametrize("pol", [2, 1])
@pytest.mark.parametrize("epsi", [0.0, 0.8])
def test_interior_field_matches_mie(pol, epsi):
    """The interior field equals the analytic Mie interior field.

    The interior representation carries the opposite sign to the exterior one
    and, in TM, the interior-side derivative eps·χ. Dropping either leaves an
    O(1) error — measured 1–2 relative before the fix — so this cannot pass by
    accident. With epsi > 0, eps is complex and the TM weighting is exercised
    in phase as well as magnitude.
    """
    rad, wavelength = 300.0, 633.0
    mat = Material(n_core=2.0, n_clad=1.45, pol=pol, epsi=epsi)
    geom = Geometry.gielis(rad=rad, n_pts=100, m=0)
    res = BIESolver(geom, mat).scatter(wavelength=wavelength)
    x, z = _R_IN * np.sin(_TH_IN), _R_IN * np.cos(_TH_IN)

    ref = _mie_interior(mat, wavelength, rad, x, z)
    err = np.max(np.abs(res.eval_field(x, z) - ref)) / np.max(np.abs(ref))
    # Kress is spectral on the circle: measured ≤ 1.4e-15 at nn = 100. The
    # bound is round-off × cond(M) (≤ 94 here) ≈ 2e-14, with margin.
    assert err < 1e-13


@pytest.mark.parametrize("pol", [2, 1])
def test_interior_and_exterior_fields_satisfy_the_interface_conditions(pol):
    """ψ is continuous and (1/ε^p)·∂ψ/∂n is continuous across a non-circular boundary.

    Each side is extrapolated to the boundary along the normal, from 11 points
    at 5–15 node spacings, by a degree-8 fit: the total field outside
    (scattered + incident) and the interior field inside. The ψ values must
    agree, and the interior normal derivative must be eps times the exterior
    one in TM and equal to it in TE. This is Maxwell's interface condition,
    not a second path through the solver, and it holds on a shape with no
    closed form. In TM, leaving out the eps factor gives 0.50 relative error
    in the derivative. A lossy, complex eps rules out a real rescaling.
    """
    wavelength, alpha, nn = 633.0, np.deg2rad(23.0), 600
    mat = Material(n_core=2.0, n_clad=1.45, pol=pol, epsi=0.8)
    geom = Geometry.gielis(rad=300.0, n_pts=nn, m=4, n1=4.0)
    res = BIESolver(geom, mat).scatter(wavelength=wavelength, angle=23.0)
    k = mat.wnum_bg(wavelength)

    j = np.array([nn // 8, nn // 3, 3 * nn // 5])
    speed = np.hypot(geom.df[j], geom.dg[j])
    nx, nz = -geom.dg[j] / speed, geom.df[j] / speed  # outward normal
    d = np.linspace(5.0, 15.0, 11)[:, None] * speed * geom.delt

    def boundary_value_and_derivative(side):
        x, z = geom.f[j] + side * d * nx, geom.g[j] + side * d * nz
        v = res.eval_field(x.ravel(), z.ravel()).reshape(x.shape)
        if side > 0:
            v = v + np.exp(1j * k * (x * np.sin(alpha) - z * np.cos(alpha)))
        fits = [np.polyfit(side * d[:, c], v[:, c], 8) for c in range(len(j))]
        return np.array([p[-1] for p in fits]), np.array([p[-2] for p in fits])

    v_out, d_out = boundary_value_and_derivative(+1)
    v_in, d_in = boundary_value_and_derivative(-1)
    eta = mat.eps if pol == 1 else 1.0

    # The floor is the one-sided extrapolation's truncation, not the solver:
    # it falls 50× from nn = 400 to 600 at fixed degree. Measured 2.4e-8
    # (value) and 1.1e-6 (derivative) at nn = 600. The bounds allow about 20×
    # margin and sit five orders below the 0.50 of a wrong TM factor.
    assert np.max(np.abs(v_in - v_out)) / np.max(np.abs(v_out)) < 5e-7
    assert np.max(np.abs(d_in - eta * d_out)) / np.max(np.abs(d_in)) < 2e-5
