"""Figure: c_sca/area vs wavelength for selected campaign designs, against the same-area circle.

Run after sweep.py:  uv run python docs/design/studies/anapole/plot_sweep.py  (→ results/sweep.png)
"""

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from shapes import geometry  # noqa: E402

PICK = (260, 300, 320, 440, 620, 700)

camp = {b["r_eq"]: b for b in json.loads((HERE / "results/campaign.json").read_text())["best"]}
sw = json.loads((HERE / "results/sweep.json").read_text())
lams = np.array(sw["lams"])
des = {d["r_eq"]: d for d in sw["designs"]}

fig, axes = plt.subplots(len(PICK), 2, figsize=(10, 2.3 * len(PICK)), gridspec_kw={"width_ratios": [4, 1]})
for row, r_eq in zip(axes, PICK):
    d, area = des[r_eq], np.pi * r_eq**2
    ax, axs = row
    ax.semilogy(lams, np.array(d["c_circle"]) / area, color="0.6", lw=1.2, label="circle, same area")
    ax.semilogy(lams, np.array(d["c"]) / area, color="#eb6834", lw=1.4, label="optimised")
    for p in d["dips"]:
        if p["depth"] < 0.05:
            ax.plot(p["lam"], p["c_min"] / area, "v", color="#eb6834", ms=5)
    ax.axvline(1550, color="0.3", lw=0.6, ls=":")
    ax.set_ylabel("c_sca / area (nm⁻¹)")
    ax.set_title(f"R_eq = {r_eq} nm   (c/A at 1550: {camp[r_eq]['c_sca'] / area:.1e},"
                 f" ×{camp[r_eq]['gain_vs_circle']:.0f} below circle)", fontsize=9, loc="left")
    ax.set_xlim(lams[0], lams[-1])
    g = geometry(np.array(camp[r_eq]["a"]), area, 1024)
    axs.fill(np.r_[g.f, g.f[0]], np.r_[g.g, g.g[0]], color="#eb6834", alpha=0.35, lw=0)
    axs.plot(np.r_[g.f, g.f[0]], np.r_[g.g, g.g[0]], color="#eb6834", lw=0.8)
    t = np.linspace(0, 2 * np.pi, 200)
    axs.plot(r_eq * np.sin(t), r_eq * np.cos(t), color="0.6", lw=0.8, ls="--")
    axs.annotate("", xy=(0, -1.15 * abs(g.g).max()), xytext=(0, -0.7 * abs(g.g).max() - 0.6 * abs(g.g).max()),
                 arrowprops={"arrowstyle": "<-", "color": "0.3", "lw": 0.6})
    axs.set_aspect("equal")
    axs.axis("off")
axes[0][0].legend(fontsize=8, loc="lower right")
axes[-1][0].set_xlabel("vacuum wavelength (nm)")
fig.suptitle("Si (n = 3.476) in air, pol = 2 (E_y), plane wave along z; optimised at 1550 nm", fontsize=10)
fig.tight_layout()
fig.savefig(HERE / "results/sweep.png", dpi=130)
