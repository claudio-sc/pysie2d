"""Cross-check of method (C) (wg_reactance.py) by the interior problem alone.

Anapole note §5: an anapole is a real ε where the interior DtN maps of core (√ε·X) and
background (X) share boundary data, A(ε) = DtN(ε) − DtN(1) singular. eigen-eps
pysie2d_crosscheck.dtn builds those maps from pysie2d's interior rows only: no outgoing
wave, no exterior field, no S. Its eigenvalues fall with ε between poles, so a root is where
the count of negative eigenvalues rises by one. Shared with (C): the boundary discretisation;
independent: the physics (interior vs. scattering).

Mirror x → −x splits even/odd classes as in (C). Unit-radius units: lengths / 500 nm,
X = k0·500 nm. ε ∈ [9, 14] holds the roots of interest (11.85, 13.20) and two disk anchors.

Run:  uv run --with mpmath python docs/design/studies/anapole/wg_interior_check.py  (mpmath: eigen-eps import only)
"""
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, os.environ.get("EIGEN_EPS_SCRIPTS", str(HERE.parents[4] / "eigen-eps/manuscripts/anapole-design-rules/scripts")))
import numpy as np  # noqa: E402
from scipy.optimize import brentq  # noqa: E402

from pysie2d_crosscheck import class_basis, dtn  # noqa: E402
from shapes import geometry  # noqa: E402
from wg_dark import AREA, LAM  # noqa: E402
from pysie2d import Geometry  # noqa: E402

R0, NN = 500.0, 256
X = 2 * np.pi / LAM * R0
CLS = {"even": {lambda f, g: (-f, g): +1}, "odd": {lambda f, g: (-f, g): -1}}


def unit_geo(a):
    g = geometry(np.asarray(a, float), AREA, NN)
    return Geometry(g.f / R0, g.g / R0, g.df / R0, g.dg / R0, g.ddf / R0, g.ddg / R0, rad=g.rad / R0)


def roots(geo, cls, lo=9.0, hi=14.0, step=0.01):
    B = class_basis(geo, CLS[cls])
    L_bg = B.T @ dtn(geo, 2, X, 1.0) @ B
    eig = lambda e: np.linalg.eigvals(B.T @ dtn(geo, 2, X, e) @ B - L_bg)  # noqa: E731
    es = np.arange(lo, hi + step / 2, step)
    lam = [eig(e) for e in es]
    neg = [int(np.sum(v.real < 0)) for v in lam]
    out = []
    for k in range(len(es) - 1):
        if neg[k + 1] > neg[k]:
            near = lambda e: (lambda v: v[np.argmin(abs(v))].real)(eig(e))  # noqa: E731
            out.append(float(brentq(near, es[k], es[k + 1], xtol=1e-10)))
    return out


c5 = json.loads((HERE / "results/campaign_500.json").read_text())
dk = json.loads((HERE / "results/wg_dark.json").read_text())
rc = json.loads((HERE / "results/wg_reactance.json").read_text())
shapes = {"circle": np.zeros(10),
          "anapole": np.array(min((o for o in dk["phase1"] if o["feasible"] and o["width_gap_ok"]), key=lambda o: o["scatt"])["a"]),
          "fs-dark": np.array(min((o for o in c5["phase_a"] if o["feasible"]), key=lambda o: o["f"])["a"])}
# ---- strong channels only (2nd pass) ----------------------------------------------------
# The first pass compared every root in [9, 14]. The mismatched ones belong to weakly coupled
# channels (dominant |n| = 5, |δ| ≈ 1e-3 over the whole sweep): their zero crossing is real
# but ill-conditioned in both methods and physically irrelevant. Compare instead the roots of
# channels whose tracked δ exceeds STRONG somewhere in the sweep, over the full ε range.
STRONG = 0.3  # rad


def strong_s_roots(name, cls):
    eps = np.array(rc["eps"])
    D = np.array(rc["shapes"][name]["delta"][cls])
    out = []
    for j in np.flatnonzero(np.abs(D).max(axis=0) > STRONG):
        for k in range(len(eps) - 1):
            x, y = D[k, j], D[k + 1, j]
            if x * y < 0 and abs(x) < 0.15 and abs(y) < 0.15:  # a strong channel moves ≤ 0.15 per step at a true zero
                out.append(float(eps[k] - x * (eps[k + 1] - eps[k]) / (y - x)))
    return sorted(out)


def _full_roots(job):
    name, cls = job
    return roots(unit_geo(shapes[name]), cls, lo=1.5, hi=25.0, step=0.01)


if __name__ == "__main__":
    from concurrent.futures import ProcessPoolExecutor
    jobs = [(n, c) for n in shapes for c in ("even", "odd")]
    with ProcessPoolExecutor(4) as ex:  # owner: 4 cores
        full = list(ex.map(_full_roots, jobs))
    print('first pass, all roots in [9, 14]')
    for name, a in shapes.items():
        geo = unit_geo(a)
        for cls in ("even", "odd"):
            r_int = roots(geo, cls)
            r_s = [r for r in rc["shapes"][name]["roots"][cls] if 9.0 <= r <= 14.0]
            pairs = "  ".join(f"{x:.4f}/{min(r_s, key=lambda y: abs(y - x)):.4f} (Δ {x - min(r_s, key=lambda y: abs(y - x)):+.1e})"
                              for x in r_int) if r_s else str([round(x, 4) for x in r_int])
            print(f"{name:8s} {cls:4s}: interior {len(r_int)} roots, S-matrix {len(r_s)} roots;  interior/S: {pairs}", flush=True)
    print("\nstrong channels (max |δ| > 0.3 rad), S-matrix root → nearest interior root")
    for (name, cls), r_int in zip(jobs, full):
        r_s = strong_s_roots(name, cls)
        print(f"{name:8s} {cls:4s}: " + "  ".join(f"{x:.3f}→{min(r_int, key=lambda y: abs(y - x)):.3f} (Δ {x - min(r_int, key=lambda y: abs(y - x)):+.0e})" for x in r_s), flush=True)
