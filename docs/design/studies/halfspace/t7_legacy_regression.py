"""Regression anchor (6): the legacy real-axis integral with the per-point
Im(alpha) >= 0 rule, continued naively to Im k < 0, computes an INCOMING wave.
R == 1:  (i/4pi) int_R exp(i q X + i a Z)/a dq  with a = sqrt(k^2-q^2), Im a >= 0."""
import numpy as np
from scipy.integrate import quad
from scipy.special import hankel1, hankel2
from hs import *
PI = np.pi
def legacy(X, Z, k):
    def f(q):
        a = np.sqrt(k*k - q*q + 0j)
        if a.imag < 0: a = -a
        return 1j/(2*PI) * np.cos(q*X) * np.exp(1j*a*Z) / a      # folded +-q
    kw = dict(limit=4000, epsabs=1e-16, epsrel=1e-13, points=[k.real])
    qmax = k.real + 45.0/Z
    re = quad(lambda q: f(q).real, 0, qmax, **kw)[0]; im = quad(lambda q: f(q).imag, 0, qmax, **kw)[0]
    return re + 1j*im
for lam in [800*(1+1j/20), 800*(1+1j/6)]:
    k = 2*PI/lam
    for X, Z in [(0., 50.), (200., 100.), (-400., 300.)]:
        rho = np.hypot(X, Z)
        L = legacy(X, Z, k)
        q, wq, _ = make_path(k, 1.0, 'pec', 1, Z, max(abs(X), 1.))
        D = g_ind(X, Z, k, 1.0, 'pec', 1, q, wq)[0]
        h1 = 0.25j*hankel1(0, k*rho); h2m = -0.25j*hankel2(0, k*rho)
        print(f"lam={lam:.0f} X={X:5.0f} Z={Z:4.0f} | legacy vs -(i/4)H0^(2): {abs(L-h2m)/abs(h2m):.1e}  legacy vs +(i/4)H0^(1): {abs(L-h1)/abs(h1):.1e} | deformed vs +(i/4)H0^(1): {abs(D-h1)/abs(h1):.1e}")
