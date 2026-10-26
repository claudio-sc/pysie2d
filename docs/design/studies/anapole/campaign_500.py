"""R_eq = 500 nm: manufacturable dark states, width/gap in the loop, then fabrication tolerance.

Phase A — multi-start (circle, the campaign_cd.py optimum, random smooth starts) minimising
log10(c_sca/A) at 1550 nm under: radius of curvature ≥ CD/2 (campaign_cd.py), perimeter
cap, and a smooth neck/gap constraint in the loop: any two boundary points more than
π·CD/2 apart along the curve must be ≥ CD apart in space. On a curve with ρ ≥ CD/2, two such
points closer than CD can only be the two sides of a neck or of a gap. The morphological
check of campaign_cd.py stays as an independent post-check.

Phase B — the best distinct Phase A designs re-optimised for the worst case over a uniform
boundary bias δ ∈ {−10, −5, 0, +5, +10} nm (shapes._offset), as an epigraph problem:
minimise t subject to t ≥ log10(c_sca(δ)/A) for every δ. The normalisation keeps the
nominal area A, so a bias that adds material is not rewarded for it.

Run:  uv run python docs/design/studies/anapole/campaign_500.py   (→ results/campaign_500.json)
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

from campaign_cd import CD, RHO_MIN, curvature, width_gap_ok  # noqa: E402
from shapes import L_MAX, c_sca, geometry, perimeter  # noqa: E402

R_EQ = 500.0
AREA = np.pi * R_EQ**2
M = 10
NN = 192  # converged to 1e-15 on the preview optimum and on its ±10 nm offsets
N_STARTS = 32
DELTAS = (-10.0, -5.0, 0.0, 5.0, 10.0)
N_ROBUST = 4
FD_H = 1e-6
S0 = np.pi * CD / 2  # arc separation beyond which two points belong to different sides


def neck(a, n=256):
    """(min distance to any point ≥ S0 away along the curve)/CD − 1, per boundary point."""
    g = geometry(a, AREA, n)
    sp = np.hypot(g.df, g.dg) * 2 * np.pi / n
    s = np.cumsum(sp) - sp[0]
    L = sp.sum()
    sep = np.abs(s[:, None] - s[None, :])
    sep = np.minimum(sep, L - sep)
    d = np.hypot(g.f[:, None] - g.f[None, :], g.g[:, None] - g.g[None, :])
    d[sep < S0] = np.inf
    return d.min(axis=1) / CD - 1


CONS = [
    {"type": "ineq", "fun": lambda a: 1 - RHO_MIN * np.abs(curvature(a, AREA))},
    {"type": "ineq", "fun": lambda a: (L_MAX - perimeter(a, AREA)) / L_MAX},
    {"type": "ineq", "fun": neck},
]


def feasible(a):
    return all(np.all(c["fun"](a) >= -1e-9) for c in CONS)


def f_nom(a, delta=0.0):
    return np.log10(c_sca(a, AREA, NN, delta=delta) / AREA)


def fd(fun, a):
    g = np.empty_like(a)
    for k in range(len(a)):
        e = np.zeros_like(a)
        e[k] = FD_H
        g[k] = (fun(a + e) - fun(a - e)) / (2 * FD_H)
    return g


def phase_a(job):
    k, a_seed = job
    rng = np.random.default_rng(500 + k)
    if k == 0:
        a0 = np.zeros(M)
    elif k == 1:
        a0 = np.array(a_seed)
    else:
        a0 = rng.normal(0, rng.uniform(0.08, 0.25), M) / np.arange(1, M + 1) ** 1.5
    while not feasible(a0):
        a0 *= 0.85
    t = time.time()
    res = minimize(f_nom, a0, jac=lambda a: fd(f_nom, a), method="SLSQP", bounds=[(-0.6, 0.6)] * M,
                   constraints=CONS, options={"maxiter": 150, "ftol": 1e-10})
    a = res.x
    return {"start": k, "a": a.tolist(), "f": float(f_nom(a)), "feasible": feasible(a),
            "width_gap_ok": width_gap_ok(a, AREA), "message": res.message, "seconds": time.time() - t}


def worst(a):
    return max(f_nom(a, d) for d in DELTAS)


def phase_b(job):
    a0 = np.asarray(job["a"])
    x0 = np.r_[a0, worst(a0) + 1e-3]
    cons = [{"type": "ineq", "fun": lambda x, d=d: x[-1] - f_nom(x[:-1], d),
             "jac": lambda x, d=d: np.r_[-fd(lambda a: f_nom(a, d), x[:-1]), 1.0]} for d in DELTAS]
    cons += [{"type": "ineq", "fun": (lambda x, c=c: c["fun"](x[:-1]))} for c in CONS]
    t = time.time()
    res = minimize(lambda x: x[-1], x0, jac=lambda x: np.r_[np.zeros(M), 1.0], method="SLSQP",
                   bounds=[(-0.6, 0.6)] * M + [(None, None)], constraints=cons,
                   options={"maxiter": 150, "ftol": 1e-10})
    a = res.x[:-1]
    return {"from_start": job["start"], "a": a.tolist(), "f_nominal": float(f_nom(a)),
            "f_by_delta": {str(d): float(f_nom(a, d)) for d in DELTAS}, "worst": float(worst(a)),
            "seed_by_delta": {str(d): float(f_nom(a0, d)) for d in DELTAS},
            "feasible": feasible(a), "width_gap_ok": width_gap_ok(a, AREA), "message": res.message,
            "seconds": time.time() - t}


def main():
    t0 = time.time()
    seed = next(b["a"] for b in json.loads((HERE / "results/campaign_cd.json").read_text())["best"] if b["r_eq"] == R_EQ)
    c_circ = c_sca(np.zeros(M), AREA, NN)
    print(f"circle c/A {c_circ / AREA:.3e}; preview seed c/A {10 ** f_nom(np.array(seed)):.3e}, "
          f"worst over ±10 nm {10 ** worst(np.array(seed)):.3e}", flush=True)
    with ProcessPoolExecutor(4) as ex:
        A = list(ex.map(phase_a, [(k, seed) for k in range(N_STARTS)]))
        for o in sorted(A, key=lambda o: o["f"]):
            print(f"A start {o['start']:2d}: c/A {10 ** o['f']:.3e} ×{c_circ / AREA / 10 ** o['f']:7.1f}  feasible {o['feasible']}"
                  f"  width/gap {o['width_gap_ok']}  {o['seconds']:.0f}s", flush=True)
        print(f"phase A done at {time.time() - t0:.0f}s", flush=True)
        ok = sorted([o for o in A if o["feasible"] and o["width_gap_ok"]], key=lambda o: o["f"])
        picks = []
        for o in ok:  # distinct designs only: coefficient distance > 0.02
            if all(np.linalg.norm(np.subtract(o["a"], p["a"])) > 0.02 for p in picks):
                picks.append(o)
            if len(picks) == N_ROBUST:
                break
        B = list(ex.map(phase_b, picks))
    for o in B:
        print(f"B from {o['from_start']:2d}: worst c/A {10 ** o['worst']:.3e} (seed worst {10 ** max(o['seed_by_delta'].values()):.3e})"
              f"  nominal {10 ** o['f_nominal']:.3e}  feasible {o['feasible']}  width/gap {o['width_gap_ok']}  {o['seconds']:.0f}s", flush=True)
    print(f"total {time.time() - t0:.0f}s")
    out = {"meta": {"r_eq": R_EQ, "cd": CD, "deltas": DELTAS, "nn": NN, "c_sca_circle": c_circ},
           "phase_a": A, "phase_b": B}
    (HERE / "results/campaign_500.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
