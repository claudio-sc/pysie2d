"""Independent real-axis reference: scipy.integrate.quad with the legacy
trigonometric/hyperbolic substitutions (cancels 1/alpha), real k only.
Branch of alpha2 chosen by a DIFFERENT rule from hs.alpha: principal sqrt,
then flip to Im >= 0 (tie Re >= 0) -- the radiation condition on the real axis."""
import numpy as np
from scipy.integrate import quad

PI = np.pi


def a_phys(k, q):
    a = np.sqrt(complex(k * k - q * q))
    if a.imag < 0 or (a.imag == 0 and a.real < 0):
        a = -a
    return a


def R_ref(q, k1, k2, eps1, eps2, pol):
    a1 = a_phys(k1, q); a2 = a_phys(k2, q)
    if pol == 2:
        return (a1 - a2) / (a1 + a2)
    return (a1 / eps1 - a2 / eps2) / (a1 / eps1 + a2 / eps2)


def g_ref(X, Z, k0, eps1, eps2, pol, which=0, s_max=None):
    """which: 0 G, 1 dG/dx', 2 dG/dz'. Real k0 only."""
    k1 = k0 * np.sqrt(eps1)
    k2 = k0 * np.sqrt(complex(eps2))
    assert np.isreal(k1)
    k1 = float(np.real(k1))

    def fac(q, a):
        return [np.cos(q * X), q * np.sin(q * X), 1j * a * np.cos(q * X)][which]

    def f1(th):  # q = k1 sin th, alpha1 = k1 cos th ; (i/2pi) R fac e^{i a Z} dth
        q = k1 * np.sin(th); a = k1 * np.cos(th)
        return 1j / (2 * PI) * R_ref(q, k1, k2, eps1, eps2, pol) * fac(q, a) * np.exp(1j * a * Z)

    def f2(s):   # q = k1 cosh s, alpha1 = i k1 sinh s ; (1/2pi) R fac e^{-k1 sinh s Z} ds
        q = k1 * np.cosh(s); a = 1j * k1 * np.sinh(s)
        return 1.0 / (2 * PI) * R_ref(q, k1, k2, eps1, eps2, pol) * fac(q, a) * np.exp(-k1 * np.sinh(s) * Z)

    pts1, pts2 = [], []
    feats = [k2.real]
    if pol == 1 and np.real(eps2) < 0:
        feats.append((k1 * np.sqrt(eps2 / (eps1 + eps2))).real)
    for f in feats:
        if 0 < f < k1:
            pts1.append(np.arcsin(f / k1))
        elif f > k1:
            pts2.append(np.arccosh(f / k1))
    if s_max is None:
        s_max = np.arcsinh(40.0 / (k1 * Z))
    out = 0j
    kw = dict(epsabs=1e-17, epsrel=1e-14, limit=4000)
    for fn, a, b, pts in [(f1, 0, PI / 2, pts1), (f2, 0, s_max, pts2)]:
        pts = [p for p in pts if a < p < b]
        re = quad(lambda t: fn(t).real, a, b, points=pts or None, **kw)[0]
        im = quad(lambda t: fn(t).imag, a, b, points=pts or None, **kw)[0]
        out += re + 1j * im
    return out
