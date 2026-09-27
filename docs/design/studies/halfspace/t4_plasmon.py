"""Plasmon pole: location q_sp = k1 sqrt(e1 e2/(e1+e2)), is it a zero of the TM
denominator ON OUR BRANCH (proper sheet), and does the hump pass below it
(clearance) over a complex-lambda box 600-900 nm, Q >= 3."""
import numpy as np
from hs import *
for name, e2 in [("Ag633", -18.3+0.48j), ("Au800", -24.1+1.5j), ("Ag900", -38.0+0.5j), ("Au600(weak)", -8.0+1.6j)]:
    worst = (np.inf, None)
    for lr in np.linspace(600, 900, 13):
        for Q in [np.inf, 30, 10, 5, 3]:
            k0 = 2*PI/(lr*(1+1j/(2*Q)))
            k1 = k0
            qsp = k1*np.sqrt(e2/(1+e2))
            den = alpha(qsp,k1)/1.0 + alpha(qsp,k0*np.sqrt(e2+0j))/e2
            q, wq, info = make_path(k0, 1.0, e2, 1, 10., 1000.)
            T, d = info['T'], info['delta']
            yq = -d*np.sin(PI*qsp.real/T)       # path Im at Re q_sp
            clear = (qsp.imag - yq)/abs(k1)    # >0: path below pole
            dist = np.min(np.abs(q - qsp))/abs(k1)
            if clear < worst[0]: worst = (clear, (lr, Q, qsp/k1, abs(den)/abs(alpha(qsp,k1)), dist))
    c, (lr, Q, r, den, dist) = worst
    print(f"{name:12s} worst clearance (Im q_sp - Im path)/|k1| = {c:.3f} at lam={lr:.0f} Q={Q}; q_sp/k1={r:.4f}; |den|/|a1|={den:.1e}; min node dist/|k1|={dist:.3f}")
