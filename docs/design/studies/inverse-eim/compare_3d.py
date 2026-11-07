"""2.5D (inverse EIM + pysie2d) against the 3D MEEP reference, by the fixed rule.

The rule is in README.md (*3D reference and the comparison rule*, fixed
2026-11-04 before this script ran); this script implements it and nothing
else. Reads the MEEP ladder from ``validation/_study/v14/`` and MPB's band
n_eff from ``band_neff.json``; writes ``compare_3d.json``::

    uv run python docs/design/studies/inverse-eim/compare_3d.py
"""

import json
from pathlib import Path

import numpy as np
from scipy.optimize import brentq

from pysie2d import Cluster, ClusterBIESolver, Geometry, Material, Stack

HERE = Path(__file__).resolve().parent
V = HERE.parents[3] / "validation" / "_study" / "v14"
W_NM, R_NM, GAP_NM = 500.0, 250.0, 60.0
X_PLANE = 2000.0
"""MEEP's production monitor plane, nm (±2.0 µm)."""
POL = 1
"""Top view of the 3D TE-like mode (E in the chip plane)."""
CONVENTIONS = {"air|air": (1.0, 1.0), "ox|ox": (1.44, 1.44), "air|ox": (1.0, 1.44)}
"""(cover, bottom): the cover is the pillar's side, as particles must be."""


def _meep() -> dict:
    """Rule steps 1–2: X_M by p = 2 Richardson on res 40/50, and U."""
    obs = {}
    for r in (40, 50):
        a = np.load(V / f"norm-r{r}.npz")
        b = np.load(V / f"pillar-r{r}.npz")
        al, be = a["alpha"], b["alpha"]  # [plane (−2, 2, −2.5, 2.5), freq, (+, −)]
        t = be[1, :, 0] / al[1, :, 0]
        rr = be[0, :, 1] / al[0, :, 0]
        obs[r] = {"t": t, "t2": np.abs(t) ** 2, "r2": np.abs(rr) ** 2,
                  "p_rad": 1 - np.abs(t) ** 2 - np.abs(rr) ** 2}
        if r == 50:
            f_norm = np.max(np.abs(np.abs(al[1, :, 0] / al[0, :, 0]) ** 2 - 1))
            f_plane = np.max(np.abs(t - be[3, :, 0] / al[3, :, 0]))
            lam = 1000 / a["freqs"]
    s = 50 / 40
    out = {"lam_nm": lam, "f_norm": f_norm, "f_plane": f_plane}
    for k in ("t", "t2", "r2", "p_rad"):
        x40, x50 = obs[40][k], obs[50][k]
        p2 = x50 + (x50 - x40) / (s**2 - 1)
        p1 = x50 + (x50 - x40) / (s - 1)
        out[k] = p2
        out[f"U_{k}"] = float(np.max(np.abs(p2 - p1)) + f_norm + f_plane)
    return out


def _n2d(n_eff: float, lam: float, cover: float, bot: float) -> float:
    def f(n: float) -> float:
        return Stack(((n**2, W_NM),), bot**2).guided_modes(lam, POL, cover)[0].n_eff - n_eff

    return brentq(f, max(n_eff, cover, bot) + 1e-9, 3.47, xtol=1e-14)


def _t_25d(lam: float, n2d: float, cover: float, bot: float, n_pts: int) -> complex:
    stack = Stack(((n2d**2, W_NM),), bot**2)
    pillar = Geometry.gielis(R_NM, n_pts, m=0, x0=0.0, z0=GAP_NM + R_NM)
    mat = Material(n_core=n2d, n_clad=cover, pol=POL)
    sol = ClusterBIESolver(Cluster([pillar]), [mat], pol=POL, background=stack)
    a = sol.scatter_mode(lam, 0, +1, x_ref=-X_PLANE).mode_amplitudes(-X_PLANE)
    return a.s_plus[0], a.s_minus[0]


def main() -> None:
    m = _meep()
    band = json.loads((HERE / "band_neff.json").read_text())
    assert np.allclose([b["lam_nm"] for b in band], m["lam_nm"]), "λ grids differ"
    res = {"meep": {k: (v.tolist() if isinstance(v, np.ndarray) and not np.iscomplexobj(v)
                        else [[c.real, c.imag] for c in v] if isinstance(v, np.ndarray) else v)
                    for k, v in m.items()}}
    print(f"MEEP: U(|t|²) {m['U_t2']:.1e}  U(P_rad) {m['U_p_rad']:.1e}  U(t) {m['U_t']:.1e}  "
          f"U(|r|²) {m['U_r2']:.1e}  (f_norm {m['f_norm']:.1e}, f_plane {m['f_plane']:.1e})")
    print(f"MEEP: P_rad {m['p_rad'].min():.4f}–{m['p_rad'].max():.4f}, |r|² max {m['r2'].max():.1e}")
    for name, (cover, bot) in CONVENTIONS.items():
        rows = []
        for b in band:
            lam = b["lam_nm"]
            n2d = _n2d(b["n_eff"], lam, cover, bot)
            t, r = _t_25d(lam, n2d, cover, bot, 100)
            t2, _ = _t_25d(lam, n2d, cover, bot, 200)
            rows.append((n2d, t, r, abs(t2 - t)))
        n2d, t, r, dn = (np.array(c) for c in zip(*rows))
        X = {"t": t, "t2": np.abs(t) ** 2, "r2": np.abs(r) ** 2,
             "p_rad": 1 - np.abs(t) ** 2 - np.abs(r) ** 2}
        line = f"{name:8s} n_2D {n2d.min():.4f}–{n2d.max():.4f} (nodes Δ|t| {dn.max():.0e})"
        conv = {"n_2d": n2d.tolist(), "node_check": float(dn.max())}
        for k in ("t2", "p_rad", "t", "r2"):
            E = X[k] - m[k]
            ratio = float(np.max(np.abs(E)) / m[f"U_{k}"])
            judged = k != "r2" or m["U_r2"] <= 0.5 * np.max(m["r2"])
            conv[k] = {"E_max": float(np.max(np.abs(E))), "E_over_U": ratio,
                       "agrees": bool(judged and ratio <= 3), "judged": bool(judged)}
            if k == "p_rad":
                conv[k]["rel"] = float(np.max(np.abs(E) / m["p_rad"]))
                conv[k]["value_25d"] = [float(X[k].min()), float(X[k].max())]
            line += f"  {k}: |E| {np.max(np.abs(E)):.1e} = {ratio:.1f} U"
        print(line + f"  P_rad 2.5D {X['p_rad'].min():.4f}–{X['p_rad'].max():.4f}"
              f" (rel err ≤ {conv['p_rad']['rel']:.0%})")
        res[name] = conv
    (HERE / "compare_3d.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
