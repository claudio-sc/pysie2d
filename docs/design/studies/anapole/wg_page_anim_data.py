"""Complex near field of the guided-drive anapole at its dark point, for the page animation.

Dark point: minimum of the lossless P_scatt spectrum in results/wg_page.json (1550 nm).
Adds "anim" to results/wg_page.json. Run after wg_page_data.py:
    uv run python docs/design/studies/anapole/wg_page_anim_data.py
"""
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from shapes import N_SI, geometry  # noqa: E402
from wg_dark import AREA, GAP, STACK  # noqa: E402
from pysie2d import BIESolver, Geometry, Material  # noqa: E402


def build(a, nn):  # as wg_page_data.build; that module computes on import
    g0 = geometry(np.asarray(a, float), AREA, nn)
    dz = GAP - g0.g.min()
    return Geometry(g0.f, g0.g + dz, g0.df, g0.dg, g0.ddf, g0.ddg, rad=g0.rad, z0=dz)


P = HERE / "results/wg_page.json"
page = json.loads(P.read_text())
d = next(x for x in page["designs"] if x["key"] == "anapole")
lam = float(page["lam"][int(np.argmin(d["spec"]["scatt"]))])
g = build(d["a"], 384)
r = BIESolver(g, Material(n_core=N_SI, n_clad=1.0, pol=2), background=STACK).scatter_mode(lam, 0, +1, x_ref=0.0)
ma = r.mode_amplitudes(x_ref=0.0)
xs, zs = np.linspace(-2000, 2000, 201), np.linspace(-1000, 1800, 141) + 5.0  # off the interface planes
X, Z = np.meshgrid(xs, zs)
u = (r.eval_field(X.ravel(), Z.ravel()) + r.incident_field(X.ravel(), Z.ravel())).reshape(X.shape)
page["anim"] = {"lam": lam, "T": float(np.sum(ma.p_plus)), "R": float(np.sum(ma.p_minus)), "scatt": float(ma.p_up + ma.p_down),
                "x": xs.tolist(), "z": zs.tolist(), "re": np.round(u.real, 4).tolist(), "im": np.round(u.imag, 4).tolist(),
                "outline": [np.r_[g.f, g.f[0]].round(1).tolist(), np.r_[g.g, g.g[0]].round(1).tolist()], "slab": [0.0, -400.0]}
P.write_text(json.dumps(page, separators=(",", ":")))
print(f"λ {lam} nm: T {page['anim']['T']:.4f} R {page['anim']['R']:.1e} P_scatt {page['anim']['scatt']:.2e}, |u| max {np.abs(u).max():.2f}")
