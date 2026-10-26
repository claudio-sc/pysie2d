"""Plan step 3: maximise P_abs under a ladder of P_scatt caps (E3, E5, E6) over the SiN slab.

R_eq = 500 nm, gap 100 nm (E7), CD 200 nm rules in the loop (E8, campaign_500.CONS),
Si real index with Im ε = 0.3 (the circle-baseline optimum, wg_baseline.py; E2 sweeps it
afterwards). TE0, pol 2, 1550 nm. The gap is held at 100 nm from the lowest boundary
point, so a shape change cannot buy coupling by moving closer to the slab.

Run:  uv run python docs/design/studies/anapole/wg_opt.py   (→ results/wg_opt.json)
"""

import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import numpy as np  # noqa: E402
from scipy.optimize import minimize  # noqa: E402

from campaign_500 import AREA, CONS, M, feasible  # noqa: E402
from campaign_cd import width_gap_ok  # noqa: E402
from shapes import N_SI, geometry  # noqa: E402
from pysie2d import BIESolver, Geometry, Material, Stack  # noqa: E402

LAM, GAP, EPSI, NN = 1550.0, 100.0, 0.3, 192
STACK = Stack(layers=((2.0**2, 400.0),), eps_bot=1.444**2, z_top=0.0)
CAPS = (None, 0.10, 0.05, 0.02, 0.01)
N_RANDOM = 2
# P_abs is O(0.1–1) and balances to 1e-15, so a 1e-5 central step keeps the FD
# truncation (~h²) and round-off (~1e-15/h) errors both below 1e-9.
FD_H = 1e-5


def solve(a, nn=NN):
    g0 = geometry(np.asarray(a, float), AREA, nn)
    dz = GAP - g0.g.min()
    g = Geometry(g0.f, g0.g + dz, g0.df, g0.dg, g0.ddf, g0.ddg, rad=g0.rad, z0=dz)
    s = BIESolver(g, Material(n_core=N_SI, n_clad=1.0, pol=2, epsi=EPSI), background=STACK)
    ma = s.scatter_mode(LAM, 0, +1, x_ref=0.0).mode_amplitudes(x_ref=0.0)
    return {"abs": float(ma.p_abs), "scatt": float(ma.p_up + ma.p_down), "T": float(np.sum(ma.p_plus)),
            "R": float(np.sum(ma.p_minus)), "balance": float(ma.balance())}


class Cached:
    """One solve serves both P_abs and P_scatt (and their FD stencils)."""

    def __init__(self):
        self.memo = {}

    def __call__(self, a):
        key = np.asarray(a, float).tobytes()
        if key not in self.memo:
            self.memo[key] = solve(a)
        return self.memo[key]

    def grad(self, a, q):
        a = np.asarray(a, float)
        out = np.empty(M)
        for k in range(M):
            e = np.zeros(M)
            e[k] = FD_H
            out[k] = (self(a + e)[q] - self(a - e)[q]) / (2 * FD_H)
        return out


def run(job):
    start, a0, cap = job
    a0 = np.array(a0, float)
    while not feasible(a0):
        a0 *= 0.85
    c = Cached()
    cons = list(CONS)
    if cap is not None:
        cons.append({"type": "ineq", "fun": lambda a: (cap - c(a)["scatt"]) / cap,
                     "jac": lambda a: -c.grad(a, "scatt") / cap})
    t = time.time()
    res = minimize(lambda a: -c(a)["abs"], a0, jac=lambda a: -c.grad(a, "abs"), method="SLSQP",
                   bounds=[(-0.6, 0.6)] * M, constraints=cons, options={"maxiter": 60, "ftol": 1e-8})
    a = res.x
    r192, r384 = c(a), solve(a, 384)
    return {"start": start, "cap": cap, "a": a.tolist(), **r192, "abs_384": r384["abs"],
            "scatt_384": r384["scatt"], "feasible": feasible(a),
            "cap_ok": cap is None or r192["scatt"] <= cap * (1 + 1e-6),
            "width_gap_ok": width_gap_ok(a, AREA), "message": res.message, "nit": res.nit,
            "seconds": time.time() - t}


def main():
    t0 = time.time()
    c5 = json.loads((HERE / "results/campaign_500.json").read_text())
    dark = min((o for o in c5["phase_a"] if o["feasible"]), key=lambda o: o["f"])["a"]
    rng = np.random.default_rng(1550)
    starts = [("circle", np.zeros(M)), ("free-space dark", np.array(dark))]
    starts += [(f"random {k}", rng.normal(0, 0.15, M) / np.arange(1, M + 1) ** 1.5) for k in range(N_RANDOM)]
    for name, a in starts[:2]:
        print(f"{name}: {solve(a)}", flush=True)
    jobs = [(name, a.tolist(), cap) for cap in CAPS for name, a in starts]
    with ProcessPoolExecutor(4) as ex:  # 4 performance cores
        out = []
        for o in ex.map(run, jobs):
            out.append(o)
            print(f"cap {o['cap']}  {o['start']:16s}: abs {o['abs']:.4f} (nn384 {o['abs_384']:.4f}) scatt {o['scatt']:.4f}"
                  f" T {o['T']:.3f} R {o['R']:.3f} bal {o['balance']:.0e} feas {o['feasible']} cap {o['cap_ok']}"
                  f" w/g {o['width_gap_ok']} nit {o['nit']} {o['seconds']:.0f}s", flush=True)
    print(f"total {time.time() - t0:.0f}s")
    (HERE / "results/wg_opt.json").write_text(json.dumps({"meta": {"lam": LAM, "gap": GAP, "epsi": EPSI, "nn": NN,
                                                                     "caps": CAPS}, "runs": out}, indent=1))


if __name__ == "__main__":
    main()
