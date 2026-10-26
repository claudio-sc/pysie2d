"""Manufacturable dark states: campaign.py under a 200 nm critical dimension (CD).

Constraint: |κ| ≤ 1/ρ_min with ρ_min = CD/2 = 100 nm, on convex and concave boundary alike
(a feature or gap of width CD cannot have a tip or corner sharper than CD/2). Curvature does
not stop two lobes pinching a neck or gap, so every result also gets a morphological check:
opening and closing with a disk of radius CD/2 on a 2 nm raster must leave the shape
unchanged (min width ≥ CD and min gap ≥ CD).

Starts per size: the circle, the unconstrained optimum of campaign.py shrunk toward the
circle until it meets the curvature bound, and random smooth starts. A global wall-clock
deadline stops every run and keeps its best feasible point (preview batches).

Run:  uv run python docs/design/studies/anapole/campaign_cd.py [budget_s]   (→ results/campaign_cd.json)
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
from scipy import ndimage  # noqa: E402
from scipy.optimize import minimize  # noqa: E402

from shapes import L_MAX, _profile, c_sca, perimeter, radius_scale  # noqa: E402

CD = 200.0
RHO_MIN = CD / 2
M = 10
NN = 192  # ρ ≥ 100 nm shapes are smooth; checked against 768 at the end
N_CURV = 512  # curvature samples; 256 let ρ slip to 99.2 nm between samples
FD_H = 1e-6
R_EQ = (300.0, 350.0, 400.0, 450.0, 500.0, 550.0, 600.0, 650.0, 700.0)
STARTS = ("circle", "shrunk", "random1", "random2")


class Deadline(Exception):
    pass


def curvature(a, area, n=N_CURV):
    th = 2 * np.pi * (np.arange(n) + 0.5) / n
    sv, ds, dds = _profile(np.asarray(a, float), th)
    r = radius_scale(a, area) * np.exp(sv)
    # r = R e^s: κ = (1 + s'² − s'') / (r (1 + s'²)^{3/2}), > 0 where convex
    return (1 + ds**2 - dds) / (r * (1 + ds**2) ** 1.5)


def curv_ok(a, area):
    return np.abs(curvature(a, area)).max() * RHO_MIN <= 1.0


def width_gap_ok(a, area, px=2.0):
    """Opening (min width) and closing (min gap) with a disk of radius CD/2 change nothing."""
    th = np.linspace(0, 2 * np.pi, 4096, endpoint=False)
    sv, _, _ = _profile(np.asarray(a, float), th)
    r = radius_scale(a, area) * np.exp(sv)
    ext = r.max() + CD
    xs = np.arange(-ext, ext + px, px)
    X, Z = np.meshgrid(xs, xs)
    rr, tt = np.hypot(X, Z), np.arctan2(X, Z) % (2 * np.pi)
    img = rr < np.interp(tt, th, r, period=2 * np.pi)
    k = int(round(RHO_MIN / px))
    yy, xx = np.mgrid[-k:k + 1, -k:k + 1]
    disk = xx**2 + yy**2 <= k**2
    tol = 0.002 * img.sum()  # raster edge noise: 0.2 % of the area
    opened = ndimage.binary_opening(img, disk)
    closed = ndimage.binary_closing(img, disk)
    return bool((img & ~opened).sum() <= tol and (closed & ~img).sum() <= tol)


def run(job):
    r_eq, start, deadline, a_prev = job
    area = np.pi * r_eq**2
    rng = np.random.default_rng(int(r_eq) * 10 + STARTS.index(start))
    if start == "circle":
        a0 = np.zeros(M)
    elif start == "shrunk":
        a0 = np.array(a_prev)
    else:
        a0 = rng.normal(0, 0.15, M) / np.arange(1, M + 1) ** 1.5
    while not (curv_ok(a0, area) and perimeter(a0, area) <= L_MAX):
        a0 *= 0.85
    best = {"f": np.inf, "a": a0.copy()}

    def obj(a):
        if time.time() > deadline:
            raise Deadline
        f = np.log10(c_sca(a, area, NN) / area)
        if f < best["f"] and curv_ok(a, area) and perimeter(a, area) <= L_MAX * (1 + 1e-9):
            best.update(f=f, a=a.copy())
        return f

    def grad(a):
        g = np.empty_like(a)
        for k in range(len(a)):
            e = np.zeros_like(a)
            e[k] = FD_H
            g[k] = (obj(a + e) - obj(a - e)) / (2 * FD_H)
        return g

    t = time.time()
    msg = "converged"
    try:
        res = minimize(obj, a0, jac=grad, method="SLSQP", bounds=[(-0.6, 0.6)] * M,
                       constraints=[{"type": "ineq", "fun": lambda a: 1 - RHO_MIN * np.abs(curvature(a, area))},
                                    {"type": "ineq", "fun": lambda a: (L_MAX - perimeter(a, area)) / L_MAX}],
                       options={"maxiter": 150, "ftol": 1e-10})
        msg = res.message
    except Deadline:
        msg = "stopped at deadline"
    a = best["a"]
    return {"r_eq": r_eq, "start": start, "a": a.tolist(), "log10_csca_per_area": float(best["f"]),
            "min_curv_radius": float(1 / np.abs(curvature(a, area, 2048)).max()),
            "perimeter_frac": perimeter(a, area) / L_MAX, "width_gap_ok": width_gap_ok(a, area),
            "message": msg, "seconds": time.time() - t}


def main():
    budget = float(sys.argv[1]) if len(sys.argv) > 1 else 280.0
    deadline = time.time() + budget
    prev = {b["r_eq"]: b["a"] for b in json.loads((HERE / "results/campaign.json").read_text())["best"]}
    near = lambda r: prev[min(prev, key=lambda k: abs(k - r))]  # §2 ladder steps by 20 nm
    jobs = [(r, s, deadline, near(r)) for r in R_EQ for s in STARTS]
    out = []
    with ProcessPoolExecutor(4) as ex:
        for o in ex.map(run, jobs):
            out.append(o)
            print(f"R_eq {o['r_eq']:.0f} {o['start']:8s}: log10(c/A) {o['log10_csca_per_area']:+.3f}  ρ_min {o['min_curv_radius']:6.1f} nm"
                  f"  width/gap {'ok' if o['width_gap_ok'] else 'FAIL'}  {o['message'][:22]}  {o['seconds']:.0f}s", flush=True)
    summary = []
    prev_camp = {b["r_eq"]: b for b in json.loads((HERE / "results/campaign.json").read_text())["best"]}
    for r_eq in R_EQ:
        area = np.pi * r_eq**2
        ok = [o for o in out if o["r_eq"] == r_eq and o["width_gap_ok"] and np.isfinite(o["log10_csca_per_area"])]
        if not ok:
            continue
        b = min(ok, key=lambda o: o["log10_csca_per_area"])
        a = np.array(b["a"])
        c192, c768, cc = c_sca(a, area, NN), c_sca(a, area, 768), c_sca(np.zeros(M), area, NN)
        summary.append({**b, "c_sca": c768, "nn_rel_diff": abs(c192 / c768 - 1), "c_sca_circle": cc, "gain_vs_circle": cc / c768,
                        "gain_unconstrained": prev_camp[min(prev_camp, key=lambda k: abs(k - r_eq))]["gain_vs_circle"]})
        s = summary[-1]
        print(f"R_eq {r_eq:.0f}: best c/A {c768 / area:.2e}  ×{s['gain_vs_circle']:.1f} vs circle"
              f"  (unconstrained ×{s['gain_unconstrained']:.0f})  ρ_min {s['min_curv_radius']:.0f} nm  from {s['start']}  nn rel {s['nn_rel_diff']:.0e}")
    meta = {"cd": CD, "rho_min": RHO_MIN, "M": M, "nn": NN, "budget_s": budget}
    (HERE / "results/campaign_cd.json").write_text(json.dumps({"meta": meta, "best": summary, "all": out}, indent=1))


if __name__ == "__main__":
    main()
