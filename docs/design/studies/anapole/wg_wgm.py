"""Option (a), step 0: do interior resonances couple to the guide better than to radiation?

Coupled-mode theory for one resonance with decay into the guide γ_wg, into radiation γ_r and
into absorption γ_a: at critical coupling (γ_a = γ_wg + γ_r) P_abs = η/2 for a standing wave
and η for a degenerate travelling pair (the disk's ±m), η = γ_wg/(γ_wg + γ_r). A dark absorber
is η → 1. The guided drive weights order m by ≈ 3.1^|m| over free-space coupling
(wg_evanescent.py), so high-m whispering-gallery modes should have η rising with m.

Circle, 1550 nm, gap 100 nm, Si real part. Lossless radius scan → resonances (P_scatt peaks)
and their order (dominant own-radiation multipole) → Im ε sweep at each.
Diagnostic: R ≠ 500 nm on purpose (E7 fixes 500 for the design work). One core.

Run:  uv run python docs/design/studies/anapole/wg_wgm.py   (→ results/wg_wgm.json)
"""
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import numpy as np  # noqa: E402
from scipy.optimize import minimize_scalar  # noqa: E402

from shapes import N_SI  # noqa: E402
from wg_dark import GAP, LAM, STACK  # noqa: E402
from pysie2d import BIESolver, Geometry, Material  # noqa: E402
from pysie2d.multipole import decompose  # noqa: E402

MMAX = 14


def solve(rad, epsi, mp=False):
    nn = int(max(256, 0.6 * rad))  # ≥ ~12 nodes per interior wavelength, spacing ≤ 0.25·2·gap
    th = 2 * np.pi * (np.arange(nn) + 0.5) / nn
    zc = GAP + rad
    g = Geometry(rad * np.sin(th), zc + rad * np.cos(th), rad * np.cos(th), -rad * np.sin(th),
                 -rad * np.sin(th), -rad * np.cos(th), rad=rad, z0=zc)
    r = BIESolver(g, Material(n_core=N_SI, n_clad=1.0, pol=2, epsi=epsi), background=STACK).scatter_mode(LAM, 0, +1, x_ref=0.0)
    ma = r.mode_amplitudes(x_ref=0.0)
    out = {"T": float(np.sum(ma.p_plus)), "R": float(np.sum(ma.p_minus)), "scatt": float(ma.p_up + ma.p_down), "abs": float(ma.p_abs)}
    if mp:
        c = decompose(r.ei, nn, g.f, g.df, g.g, g.dg, g.delt, r.wnum_bg, 3 * rad, mmax=MMAX, x0=0.0, z0=zc).c
        m = np.arange(-MMAX, MMAX + 1)
        out["mp"] = [float(np.sum(abs(c[abs(m) == k]) ** 2)) for k in range(MMAX + 1)]
    return out


radii = np.arange(400.0, 1000.01, 1.0)
scan = [solve(r, 0.0) for r in radii]
sc = np.array([s["scatt"] for s in scan]) + np.array([s["R"] for s in scan])
ext = 1 - np.array([s["T"] for s in scan])
peaks = [k for k in range(1, len(radii) - 1) if ext[k] > ext[k - 1] and ext[k] >= ext[k + 1] and ext[k] > 0.02]
res = []
for k in peaks:
    # refine the lossless extinction peak to 0.01 nm
    f = minimize_scalar(lambda r: -(1 - solve(r, 0.0)["T"]), bounds=(radii[k] - 1, radii[k] + 1), method="bounded",
                        options={"xatol": 1e-2})
    r0 = float(f.x)
    lossless = solve(r0, 0.0, mp=True)
    m_dom = int(np.argmax(lossless["mp"]))
    e = np.logspace(-4, 0.5, 46)
    sw = [solve(r0, x) for x in e]
    ab = np.array([s["abs"] for s in sw])
    j = int(ab.argmax())
    row = {"rad": r0, "m": m_dom, "ext0": float(1 - lossless["T"]), "R0": lossless["R"], "scatt0": lossless["scatt"],
           "mp": lossless["mp"], "epsi_opt": float(e[j]), "abs_max": float(ab[j]), "scatt_at": sw[j]["scatt"],
           "R_at": sw[j]["R"], "T_at": sw[j]["T"], "sweep": {"epsi": e.tolist(), "rows": sw}}
    res.append(row)
    print(f"R {r0:7.2f}  |m| {m_dom:2d}  ext0 {row['ext0']:.3f} (R {row['R0']:.3f})  P_abs,max {row['abs_max']:.3f} at Im ε {e[j]:.2e}"
          f"  P_scatt {row['scatt_at']:.3f}  R {row['R_at']:.3f}  scatt/abs {row['scatt_at']/row['abs_max']:.3f}", flush=True)
(HERE / "results/wg_wgm.json").write_text(json.dumps({"radii": radii.tolist(), "scan": scan, "res": res}))
