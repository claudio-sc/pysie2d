"""(d) complex k: (i) independent -- Chebyshev interpolation in k of the REAL-k quad
reference, evaluated at complex k (G(k) analytic in Re k>0; nearest singularity k=0);
(ii) consistency -- path independence under hump depth."""
import numpy as np, warnings
from numpy.polynomial import chebyshev as C
from hs import *
from refquad import g_ref
warnings.simplefilter("ignore")
cases = [("glass", 800., 2.25, 2), ("glass", 800., 2.25, 1), ("Si", 1600., 12.25, 1),
         ("Ag", 633., -18.3 + 0.48j, 1), ("Au", 800., -24.1 + 1.5j, 1), ("Au", 800., -24.1 + 1.5j, 2)]
pts = [(0., 10.), (150., 40.), (-300., 210.)]
N = 32
for name, lam, e2, pol in cases:
    k0r = 2 * PI / lam
    a, b = 0.6 * k0r, 1.4 * k0r
    xs = np.cos(PI * (np.arange(N) + 0.5) / N)
    ks = 0.5 * (b - a) * xs + 0.5 * (a + b)
    for X, Z in pts:
        vals = np.array([g_ref(X, Z, k, 1.0, e2, pol) for k in ks])
        cr = C.chebfit(xs, vals.real, N - 1); ci = C.chebfit(xs, vals.imag, N - 1)
        tail = np.max(np.abs(cr[-4:]) + np.abs(ci[-4:])) / np.max(np.abs(cr) + np.abs(ci))
        out = []
        for Q in (10., 3.):
            lamc = lam * (1 + 1j / (2 * Q))
            k0 = 2 * PI / lamc
            u = (k0 - 0.5 * (a + b)) / (0.5 * (b - a))
            ext = C.chebval(u, cr) + 1j * C.chebval(u, ci)
            q, wq, info = make_path(k0, 1.0, e2, pol, Z, max(abs(X), 1.0))
            num = g_ind(X, Z, k0, 1.0, e2, pol, q, wq)[0]
            q2, wq2, _ = make_path(k0, 1.0, e2, pol, Z, max(abs(X), 1.0), delta_frac=0.8)
            num2 = g_ind(X, Z, k0, 1.0, e2, pol, q2, wq2)[0]
            out.append(f"Q={Q:2.0f}: cheb {abs(num-ext)/abs(ext):.1e} depth {abs(num-num2)/abs(num):.1e}")
        print(f"{name:5s} {lam:4.0f} pol={pol} X={X:5.0f} Z={Z:4.0f} chebtail={tail:.0e} | " + " | ".join(out))
