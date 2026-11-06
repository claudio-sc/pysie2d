import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch
from pysie2d import Geometry
import os
HERE = os.path.dirname(os.path.abspath(__file__))

plt.rcParams.update({"font.size": 9, "font.family": "serif", "mathtext.fontset": "cm",
                     "axes.spines.top": False, "axes.spines.right": False})
INK, MUTED, ACC, ACC2 = "#222222", "#888888", "#C0392B", "#1F6FB2"

# ---------- figure 1: what lives where ----------
fig, (ax, bx) = plt.subplots(1, 2, figsize=(6.6, 2.55), gridspec_kw={"width_ratios": [1.25, 1]})
g = Geometry.gielis(rad=90.0, n_pts=36, m=3, n1=4.0, n2=6.0, n3=6.0)
f, z = np.append(g.f, g.f[0]), np.append(g.g, g.g[0]) + 150
ax.fill(f, z, color="#F4D9D5", zorder=1)
ax.plot(f, z, color=ACC, lw=1.2, zorder=2)
ax.plot(g.f, g.g + 150, "o", ms=2.2, color=ACC, zorder=3)
ax.add_patch(Rectangle((-330, -110), 660, 70, facecolor="#E6E6E6", edgecolor=MUTED, lw=0.6, hatch="////", zorder=0))
ax.text(325, -30, "slab", ha="right", va="bottom", color=MUTED)
j = int(np.argmax(g.f + 0.3*g.g))
nx, nz = g.dg[j], -g.df[j]; s = np.hypot(nx, nz); nx, nz = nx/s, nz/s
p0 = np.array([g.f[j], g.g[j] + 150])
if (p0[0]-0)*nx + (p0[1]-150)*nz < 0: nx, nz = -nx, -nz
ax.add_patch(FancyArrowPatch(p0, p0 + 70*np.array([nx, nz]), arrowstyle="-|>", mutation_scale=8, color=INK, lw=0.8))
ax.text(*(p0 + 74*np.array([nx, nz]) + np.array([4, -22])), "normal", fontsize=7.5, color=MUTED)
ax.text(105, 292, r"on the boundary:" + "\n" + r"$\varphi=\psi,\ \chi=\partial\psi/\partial n$", va="center", fontsize=8.5)
ax.text(0, 150, "inside:\n" + r"$k\sqrt{\varepsilon_i}$", ha="center", va="center", color=ACC, fontsize=8.5)
ax.text(-325, 300, "outside: background\n" + r"Green function $g_b$" + "\n(slab included)", va="top", fontsize=8.5)
ax.add_patch(FancyArrowPatch((-300, 75), (-160, 75), arrowstyle="-|>", mutation_scale=8, color=ACC2, lw=1))
ax.text(-300, 92, r"incident $\psi_0$", color=ACC2, fontsize=8.5)
ax.set_xlim(-330, 330); ax.set_ylim(-115, 320); ax.set_aspect("equal"); ax.axis("off")
ax.set_title("(a) unknowns live on the boundary", fontsize=9, loc="left")

bx.axis("off"); bx.set_xlim(0, 10); bx.set_ylim(0, 6.2)
bx.set_title(r"(b) $M(\varepsilon_i)\,x=\mathrm{right\ side}$", fontsize=9, loc="left")
cells = [((0.3, 3.0), "$A$", "#EEEEEE", INK), ((2.5, 3.0), "$B$", "#EEEEEE", INK),
         ((0.3, 0.6), r"$C(\varepsilon_i)$", "#F4D9D5", ACC), ((2.5, 0.6), r"$\eta D(\varepsilon_i)$", "#F4D9D5", ACC)]
for (x0, y0), lab, fc, tc in cells:
    bx.add_patch(Rectangle((x0, y0), 2.1, 2.3, facecolor=fc, edgecolor="white", lw=2))
    bx.text(x0 + 1.05, y0 + 1.15, lab, ha="center", va="center", color=tc, fontsize=10)
bx.text(5.05, 4.15, r"$\varphi$", ha="center", va="center", fontsize=10)
bx.text(5.05, 1.75, r"$\chi$", ha="center", va="center", fontsize=10)
bx.text(5.85, 2.95, "=", ha="center", va="center", fontsize=11)
bx.text(6.65, 4.15, "$b$", ha="center", va="center", fontsize=10, color=ACC2)
bx.text(6.65, 1.75, "$0$", ha="center", va="center", fontsize=10)
for xx in (4.7, 6.3):
    bx.plot([xx, xx], [0.6, 5.3], color=INK, lw=0.6); bx.plot([xx+0.7, xx+0.7], [0.6, 5.3], color=INK, lw=0.6)
bx.text(7.6, 4.15, "outside rows:\nfixed", va="center", fontsize=8, color=MUTED)
bx.text(7.6, 1.75, "inside rows:\n" + r"change with $\varepsilon_i$", va="center", fontsize=8, color=ACC)
fig.tight_layout(); fig.savefig(os.path.join(HERE, "fig1.pdf")); plt.close(fig)

# ---------- figure 2: the slab test ----------
d = np.load(os.path.join(HERE, "..", "expand2.npz")); eps, exact, poles, modes = d["eps"], d["exact"], d["poles"], d["modes"]
re = eps.real; fitted = poles.copy()
for msk in ((re <= -1.3), (re > -1.3) & (re < -0.7), (re >= -0.7)):
    idx = np.flatnonzero(msk)[[0, msk.sum()//2, -1]]
    c = np.linalg.solve(np.vander(eps[idx], 3, increasing=True), exact[idx]-poles[idx])
    fitted[msk] = poles[msk] + np.vander(eps[msk], 3, increasing=True) @ c
fig, ax = plt.subplots(figsize=(6.6, 2.15))
ax.axvspan(-1.3, -0.7, color="#F2F2F2", zorder=0)
ax.text(-1.0, 0.07, "modes crowd\nnear $\\varepsilon_i=-\\varepsilon_b$", ha="center", va="bottom", fontsize=8, color=MUTED)
ax.plot(re, np.abs(exact), color=INK, lw=1.6, label="exact solve at every $\\varepsilon_i$")
ax.plot(re, np.abs(poles), color=ACC2, lw=1.1, ls="--", label="six mode terms only")
ax.plot(re, np.abs(fitted), color=ACC, lw=1.1, label="mode terms + smooth leftover")
for m in modes:
    if -3 < m.real < -0.2: ax.plot([m.real]*2, [0, 0.035], color=INK, lw=1.0)
ax.text(-2.98, 0.045, r"ticks: Re $\varepsilon_m$", fontsize=7.5, color=MUTED)
ax.set_xlim(-3, -0.2); ax.set_ylim(0, 0.9)
ax.set_xlabel(r"Re $\varepsilon_i/\varepsilon_b$   (Im $\varepsilon_i/\varepsilon_b = 0.02$)")
ax.set_ylabel("guided wave amplitude")
ax.legend(frameon=False, fontsize=7.5, loc="upper left")
fig.tight_layout(); fig.savefig(os.path.join(HERE, "fig2.pdf")); plt.close(fig)
print("ok")
