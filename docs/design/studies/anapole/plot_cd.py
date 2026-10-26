"""Preview figure for campaign_cd.py: best manufacturable shape per size, and the spectrum of the darkest."""
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from shapes import c_sca, geometry  # noqa: E402

best = json.loads((HERE / "results/campaign_cd.json").read_text())["best"]
fig = plt.figure(figsize=(12, 6.2))
for k, b in enumerate(best):
    ax = fig.add_subplot(2, len(best), k + 1)
    area = np.pi * b["r_eq"] ** 2
    g = geometry(np.array(b["a"]), area, 1024)
    ax.fill(np.r_[g.f, g.f[0]], np.r_[g.g, g.g[0]], color="#eb6834", alpha=0.35, lw=0)
    ax.plot(np.r_[g.f, g.f[0]], np.r_[g.g, g.g[0]], color="#eb6834", lw=1)
    t = np.linspace(0, 2 * np.pi, 200)
    ax.plot(b["r_eq"] * np.sin(t), b["r_eq"] * np.cos(t), color="0.6", lw=0.7, ls="--")
    ax.plot([-b["r_eq"], -b["r_eq"] + 200], [-1.35 * b["r_eq"]] * 2, color="0.2", lw=1.5)
    ax.set_xlim(-1.45 * b["r_eq"], 1.45 * b["r_eq"]); ax.set_ylim(-1.45 * b["r_eq"], 1.45 * b["r_eq"])
    ax.set_aspect("equal"); ax.axis("off")
    ax.set_title(f"{b['r_eq']:.0f} nm\n×{b['gain_vs_circle']:.0f}", fontsize=9)
top = max(best, key=lambda b: b["gain_vs_circle"])
area = np.pi * top["r_eq"] ** 2
lams = np.arange(1300.0, 1800.1, 1.0)
c = [c_sca(np.array(top["a"]), area, 192, wavelength=lam) / area for lam in lams]
cc = [c_sca(np.zeros(10), area, 192, wavelength=lam) / area for lam in lams]
ax = fig.add_subplot(2, 1, 2)
ax.semilogy(lams, cc, color="0.6", lw=1.2, label="circle, same area")
ax.semilogy(lams, c, color="#eb6834", lw=1.5, label=f"CD-compliant, R_eq {top['r_eq']:.0f} nm")
ax.axvline(1550, color="0.3", lw=0.6, ls=":")
ax.set_xlabel("vacuum wavelength (nm)"); ax.set_ylabel("c_sca / area (nm⁻¹)"); ax.legend(fontsize=8)
fig.suptitle("Best shape per size with radius of curvature ≥ 100 nm and width/gap ≥ 200 nm (×: darker than circle at 1550 nm)", fontsize=10)
fig.tight_layout()
fig.savefig(HERE / "results/campaign_cd.png", dpi=120)
i = int(np.argmin(c)); print("min", lams[i], c[i])
lo = lams[(np.array(c) < 10 * min(c))]; print("width x10", lo.max() - lo.min())
