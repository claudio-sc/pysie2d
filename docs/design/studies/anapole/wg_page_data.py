"""Data for the guided-drive anapole page: near fields, own-radiation multipoles, spectra.

Multipoles over a stack are not a package feature. Here pysie2d.multipole.decompose is
applied to the stack-solved densities: the free-space part of the representation, i.e.
the particle's own outgoing radiation before the slab acts on it. With no stack this is
exactly ScatterResult.multipoles().

Run:  uv run python docs/design/studies/anapole/wg_page_data.py   (→ results/wg_page.json)
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
from pysie2d.multipole import decompose  # noqa: E402

MMAX, NN = 8, 384
c5 = json.loads((HERE / "results/campaign_500.json").read_text())
dk = json.loads((HERE / "results/wg_dark.json").read_text())
fs_dark = min((o for o in c5["phase_a"] if o["feasible"]), key=lambda o: o["f"])["a"]
anap = min((o for o in dk["phase1"] if o["feasible"] and o["width_gap_ok"]), key=lambda o: o["scatt"])
absb = max((o for o in dk["phase3"] if o["cap"] == 0.05 and o["feasible"] and o["cap_ok"]), key=lambda o: o["abs"])
DESIGNS = [("circle", "Circle", np.zeros(10), 0.0), ("anapole", "Guided-drive anapole", np.array(anap["a"]), 0.0),
           ("fsdark", "Free-space dark design", np.array(fs_dark), 0.0),
           ("absorber", "Best absorber, P_scatt ≤ 0.05", np.array(absb["a"]), absb["epsi"])]


def build(a, nn=NN):
    g0 = geometry(np.asarray(a, float), AREA, nn)
    dz = GAP - g0.g.min()
    return Geometry(g0.f, g0.g + dz, g0.df, g0.dg, g0.ddf, g0.ddg, rad=g0.rad, z0=dz)


def run(a, epsi, lam, nn=NN, eps_re=N_SI**2):
    g = build(a, nn)
    s = BIESolver(g, Material(n_core=float(np.sqrt(eps_re)), n_clad=1.0, pol=2, epsi=float(epsi)), background=STACK)
    r = s.scatter_mode(lam, 0, +1, x_ref=0.0)
    ma = r.mode_amplitudes(x_ref=0.0)
    rc = float(np.hypot(g.f - g.x0, g.g - g.z0).max())
    mp = decompose(r.ei, g.n_pts, g.f, g.df, g.g, g.dg, g.delt, r.wnum_bg, 3 * rc, mmax=MMAX, x0=g.x0, z0=g.z0)
    c2 = np.abs(mp.c) ** 2
    m = np.arange(-MMAX, MMAX + 1)
    pw = {"T": float(np.sum(ma.p_plus)), "R": float(np.sum(ma.p_minus)), "scatt": float(ma.p_up + ma.p_down),
          "abs": float(ma.p_abs)}
    return g, r, pw, [float(c2[abs(m) == k].sum()) for k in range(MMAX + 1)]


out = {"designs": [], "x": None, "z": None, "lam": [], "eps_sweep": {}, "pareto": [], "loss": [],
       "circle_anapoles": dk["circle_anapoles"], "slab": {"top": 0.0, "bottom": -400.0}}
xs, zs = np.linspace(-1600, 1600, 161), np.linspace(-1000, 1600, 131) + 5.0  # off the interface planes
X, Z = np.meshgrid(xs, zs)
out["x"], out["z"] = xs.tolist(), zs.tolist()
lams = np.linspace(1300, 1800, 101)
out["lam"] = lams.tolist()
for key, label, a, epsi in DESIGNS:
    g, r, pw, mpl = run(a, epsi, 1550.0, nn=512)
    u = (r.eval_field(X.ravel(), Z.ravel()) + r.incident_field(X.ravel(), Z.ravel())).reshape(X.shape)
    spec = [run(a, epsi, lam, nn=256) for lam in lams]
    out["designs"].append({
        "key": key, "label": label, "epsi": epsi, "a": a.tolist(), **pw, "mp": mpl,
        "outline": [np.r_[g.f, g.f[0]].round(1).tolist(), np.r_[g.g, g.g[0]].round(1).tolist()],
        "re": np.round(u.real, 4).tolist(), "abs_f": np.round(np.abs(u), 4).tolist(),
        "spec": {q: [s[2][q] for s in spec] for q in ("T", "R", "scatt", "abs")},
        "spec_mp": [[s[3][k] for s in spec] for k in range(MMAX + 1)],
    })
    print(key, pw, flush=True)
for e in dk["eps_sweep"]:
    out["eps_sweep"][e["name"]] = {"eps": e["eps"], "scatt": [r["scatt"] for r in e["rows"]]}
for r in json.loads((HERE / "results/wg_opt.json").read_text())["runs"]:
    if r["feasible"] and r["width_gap_ok"] and (r["cap"] is None or r["scatt"] <= r["cap"] * 1.01):
        out["pareto"].append({"scatt": r["scatt"], "abs": r["abs"], "epsi": 0.3, "set": "Im ε = 0.3"})
for r in dk["phase3"]:
    if r["feasible"] and r["width_gap_ok"] and r["cap_ok"]:
        out["pareto"].append({"scatt": r["scatt"], "abs": r["abs"], "epsi": r["epsi"], "set": "Im ε free"})
p2 = dk["phase2"][0]
out["loss"] = {"epsi": p2["epsi"], "abs": [r["abs"] for r in p2["rows"]], "scatt": [r["scatt"] for r in p2["rows"]]}
(HERE / "results/wg_page.json").write_text(json.dumps(out, separators=(",", ":")))
print("written", (HERE / "results/wg_page.json").stat().st_size // 1024, "kB")
