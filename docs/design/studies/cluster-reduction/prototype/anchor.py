"""Analytic eigenpermittivities of a circular cylinder at fixed size parameter x.

TM (pol=1, H_y): (1/m) J_n'(m x) H_n(x) - J_n(m x) H_n'(x) = 0
TE (pol=2, E_y):  m   J_n'(m x) H_n(x) - J_n(m x) H_n'(x) = 0
Zeros depend on eps = m^2 only (parity in m), so the branch of sqrt is irrelevant here.
"""
import numpy as np
from scipy.special import jv, jvp, hankel1, h1vp
from scipy.optimize import newton

def D(eps, n, x, pol):
    m = np.sqrt(complex(eps))
    if pol == 1:
        return jvp(n, m*x)*hankel1(n, x) - m*jv(n, m*x)*h1vp(n, x)   # x m to clear 1/m
    return m*jvp(n, m*x)*hankel1(n, x) - jv(n, m*x)*h1vp(n, x)

def root(eps0, n, x, pol):
    return complex(newton(lambda e: D(e, n, x, pol), complex(eps0), tol=1e-14, maxiter=200))

def winding(n, x, pol, lo, hi, m=4000):
    # argument-principle count of zeros of D in the rectangle (D has no poles there)
    t = np.linspace(0, 1, m, endpoint=False)
    z = np.concatenate([lo + (hi.real-lo.real)*t, hi.real+1j*lo.imag + 1j*(hi.imag-lo.imag)*t,
                        hi - (hi.real-lo.real)*t, lo.real+1j*hi.imag - 1j*(hi.imag-lo.imag)*t])
    d = np.array([D(e, n, x, pol) for e in z])
    ph = np.unwrap(np.angle(np.append(d, d[0])))
    return int(round((ph[-1]-ph[0])/(2*np.pi)))

if __name__ == "__main__":
    x = 2*np.pi*100/600
    print("x =", x)
    print("TM plasmonic family (start from quasistatic -1):")
    for n in range(1, 9):
        r = root(-1.0 - 0.01j, n, x, 1)
        print(f"  n={n}: eps_m = {r.real:+.10f} {r.imag:+.3e}i   |D|={abs(D(r,n,x,1)):.1e}")
