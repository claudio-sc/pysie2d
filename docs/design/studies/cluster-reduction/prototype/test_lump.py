import numpy as np
import os
HERE = os.path.dirname(os.path.abspath(__file__))
d = np.load(os.path.join(HERE, "expand2.npz")); eps, exact, poles = d["eps"], d["exact"], d["poles"]
scale = np.max(np.abs(exact)); re = eps.real
near = (re > -1.3) & (re < -0.7)
def fit(basis, idx):
    A = np.column_stack([f(eps) for f in basis])
    c, *_ = np.linalg.lstsq(A[idx], (exact - poles)[idx], rcond=None)
    model = poles + A @ c
    return np.max(np.abs(model-exact))/scale, np.max(np.abs(model-exact)[~near])/scale, np.max(np.abs(model-exact)[near])/scale
poly = [lambda e: 1+0*e, lambda e: e, lambda e: e**2]
lump = poly + [lambda e: 1/(e + 1.0)]
lump2 = poly + [lambda e: 1/(e + 1.0), lambda e: 1/(e + 1.0)**2]
for name, basis in (("quadratic", poly), ("quadratic + c/(eps+1)", lump), ("quadratic + c1/(eps+1) + c2/(eps+1)^2", lump2)):
    k = len(basis)
    idx = np.round(np.linspace(0, eps.size-1, k)).astype(int)        # k exact solves, evenly spread
    g, off, on = fit(basis, idx)
    print(f"{name:40s} {k} solves: max err whole sweep {g:.1e} | away from -1 {off:.1e} | near -1 {on:.1e}")
