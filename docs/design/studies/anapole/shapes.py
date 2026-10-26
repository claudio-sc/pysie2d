"""Mirror-symmetric Fourier-exponential shapes and the dark-state objective.

r(θ) = R·exp(Σ_{m=1..M} a_m cos mθ), θ from +z toward +x (the Geometry.gielis
sense). Cosines only: the shape is symmetric about the propagation axis x = 0, so
a plane wave at angle 0 (along z) excites only the x-even class, and the x-odd
class needs no darkness. The exponential keeps r > 0, so the curve is star-shaped
and cannot self-intersect. R is set by the target area, not optimised.
"""

import numpy as np

from pysie2d import BIESolver, Geometry, Material

LAM0 = 1550.0  # nm, vacuum
N_SI = 3.476  # Si at 1550 nm, lossless (Im ε ~ 0 below the band gap)
POL = 2  # pysie2d TE, E_y along the axis — the anapole note's TM (conventions §1)
L_MAX = 10.0 * LAM0 / N_SI  # perimeter cap: 10 interior wavelengths, 4459 nm


def _profile(a, theta):
    m = np.arange(1, len(a) + 1)
    c, s = np.cos(np.outer(theta, m)), np.sin(np.outer(theta, m))
    sv = c @ a
    ds = -(s * m) @ a
    dds = -(c * m**2) @ a
    return sv, ds, dds


def radius_scale(a, area, n_fine=4096):
    th = 2 * np.pi * (np.arange(n_fine) + 0.5) / n_fine
    sv, _, _ = _profile(a, th)
    # A = ½∮r² dθ; the periodic trapezoid rule is spectral here.
    return np.sqrt(2 * area / (np.exp(2 * sv).mean() * 2 * np.pi))


def perimeter(a, area, n_fine=4096):
    th = 2 * np.pi * (np.arange(n_fine) + 0.5) / n_fine
    sv, ds, _ = _profile(a, th)
    r = radius_scale(a, area, n_fine) * np.exp(sv)
    return float(np.sqrt(r**2 + (r * ds) ** 2).mean() * 2 * np.pi)


def geometry(a, area, nn, delta=0.0):
    th = 2 * np.pi * (np.arange(nn) + 0.5) / nn
    sv, ds, dds = _profile(np.asarray(a, float), th)
    R = radius_scale(a, area)
    r = R * np.exp(sv)
    dr = r * ds
    ddr = r * (dds + ds**2)
    st, ct = np.sin(th), np.cos(th)
    f, g = r * st, r * ct
    df = dr * st + r * ct
    dg = dr * ct - r * st
    ddf = ddr * st + 2 * dr * ct - r * st
    ddg = ddr * ct - 2 * dr * st - r * ct
    if delta:
        return _offset(f, g, df, dg, delta, float(np.sqrt(area / np.pi)))
    return Geometry(f, g, df, dg, ddf, ddg, rad=float(np.sqrt(area / np.pi)))


def _offset(f, g, df, dg, delta, rad):
    """Boundary moved by delta (nm) along the outward normal: a uniform fabrication bias.

    The traversal is clockwise in (x, z), so the outward normal is (−ż, ẋ)/|ṗ|. The offset
    curve is not a Fourier-exponential shape, so its t-derivatives are taken spectrally on
    the same uniform nodes; it stays smooth while |delta| < the 100 nm curvature radius.
    """
    sp = np.hypot(df, dg)
    fo, go = f - delta * dg / sp, g + delta * df / sp
    n = len(f)
    ik = 1j * np.fft.fftfreq(n, 1.0 / n)  # d/dt on t ∈ [0, 2π)
    ik[n // 2] = 0.0  # Nyquist mode carries no derivative for an even n
    d1 = lambda u: np.fft.ifft(ik * np.fft.fft(u)).real  # noqa: E731
    d2 = lambda u: np.fft.ifft(ik**2 * np.fft.fft(u)).real  # noqa: E731
    return Geometry(fo, go, d1(fo), d1(go), d2(fo), d2(go), rad=rad)


def c_sca(a, area, nn, wavelength=LAM0, n_core=N_SI, delta=0.0):
    s = BIESolver(geometry(a, area, nn, delta), Material(n_core=n_core, n_clad=1.0, pol=POL))
    return s.scatter(wavelength, angle=0.0).cross_sections()["c_sca"]
