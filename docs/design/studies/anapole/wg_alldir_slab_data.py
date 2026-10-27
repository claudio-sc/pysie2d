"""All-direction optimum (wg_alldir.py) above the SiN slab: T/R/P_scatt/P_abs spectra and the
complex near field at its dark point (minimum lossless P_scatt), for the page's animation.

Stack, gap and mode as wg_dark.py (TE0, gap 100 nm). Im ε = 0 (no absorption) and 0.3 (the
loss of wg_opt.py). Run after wg_alldir_page_data.py; adds "slab" to results/wg_alldir_page.json.
Run:  uv run python docs/design/studies/anapole/wg_alldir_slab_data.py
"""
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import numpy as np  # noqa: E402

from shapes import N_SI, geometry  # noqa: E402
from wg_dark import AREA, GAP, STACK  # noqa: E402
from pysie2d import BIESolver, Geometry, Material  # noqa: E402

LAMS = np.linspace(1300, 1800, 201)
EPSIS = (0.0, 0.3)


def build(a, nn):
    g0 = geometry(np.asarray(a, float), AREA, nn)
    dz = GAP - g0.g.min()
    return Geometry(g0.f, g0.g + dz, g0.df, g0.dg, g0.ddf, g0.ddg, rad=g0.rad, z0=dz)


def point(job):
    a, epsi, lam = job
    s = BIESolver(build(a, 256), Material(n_core=N_SI, n_clad=1.0, pol=2, epsi=epsi), background=STACK)
    ma = s.scatter_mode(lam, 0, +1, x_ref=0.0).mode_amplitudes(x_ref=0.0)
    return [float(np.sum(ma.p_plus)), float(np.sum(ma.p_minus)), float(ma.p_up + ma.p_down), float(ma.p_abs), float(ma.balance())]


def main():
    P = HERE / "results/wg_alldir_page.json"
    page = json.loads(P.read_text())
    best = next(d for d in page["designs"] if d["key"] == "alldir")
    a = best["a"]
    spec = {}
    with ProcessPoolExecutor(4) as ex:  # owner: 4 cores
        for e in EPSIS:
            rows = list(ex.map(point, [(a, e, l) for l in LAMS], chunksize=8))
            spec[str(e)] = {q: [r[i] for r in rows] for i, q in enumerate(("T", "R", "scatt", "abs"))}
            print(f"Im ε {e}: max |balance| {max(abs(r[4]) for r in rows):.1e}", flush=True)
    k = int(np.argmin(spec["0.0"]["scatt"]))
    lam_d = float(LAMS[k])
    g = build(a, 384)
    r = BIESolver(g, Material(n_core=N_SI, n_clad=1.0, pol=2), background=STACK).scatter_mode(lam_d, 0, +1, x_ref=0.0)
    xs, zs = np.linspace(-2000, 2000, 201), np.linspace(-1000, 1800, 141) + 5.0  # off the interface planes
    X, Z = np.meshgrid(xs, zs)
    u = (r.eval_field(X.ravel(), Z.ravel()) + r.incident_field(X.ravel(), Z.ravel())).reshape(X.shape)
    ma = r.mode_amplitudes(x_ref=0.0)
    page["slab"] = {"lam": LAMS.tolist(), "spec": spec, "lam_dark": lam_d,
                    "at_dark": {"T": float(np.sum(ma.p_plus)), "R": float(np.sum(ma.p_minus)), "scatt": float(ma.p_up + ma.p_down)},
                    "x": xs.tolist(), "z": zs.tolist(), "re": np.round(u.real, 4).tolist(), "im": np.round(u.imag, 4).tolist(),
                    "outline": [np.r_[g.f, g.f[0]].round(1).tolist(), np.r_[g.g, g.g[0]].round(1).tolist()],
                    "slab": [0.0, -400.0]}
    P.write_text(json.dumps(page, separators=(",", ":")))
    print(f"dark point λ {lam_d:.1f} nm: {page['slab']['at_dark']}; |u| max {np.abs(u).max():.2f}")


if __name__ == "__main__":
    main()
