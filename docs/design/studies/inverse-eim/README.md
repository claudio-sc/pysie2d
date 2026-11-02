# Inverse EIM — reproducing Høvik & Aksnes (2024)

Reproduction of J. Høvik and A. Aksnes, "Inverse Effective Index Method for
Two-Dimensional Simulations of Photonic Components", IEEE Photon. J. 16,
6600106 (2024), the method v1.4 adopts (roadmap R10, revised 2026-11-02).
Scope (owner, 2026-11-02): Figs. 2–5. Fig. 6 (a 5 µm ring) is out: pysie2d
particles cannot contain a hole.

Paper materials throughout: Si core 3.47 on SiO₂ 1.44 under air 1.0,
λ = 1550 nm. The 3D n_eff comes from MPB (`mpb_neff.py`, MEEP conda env); every
2D quantity comes from `Stack.guided_modes`.

| Step | Paper | Here | Status |
|---|---|---|---|
| R1 — Fig. 2, plain-EIM side view, 220 nm slab, TE | n_eff = 2.825 | 2.8247 | done |
| R2 — Fig. 3, inverse-EIM n_2D, 500 × 220 nm, TE | 2.763 | **2.7637** | done |
| R3 — Figs. 4–5, β error maps over the single-mode domain | ≤ 17 % EIM, ~100 % none, < 0.01 % inverse | — | planned |

## R2

MPB, 500 × 220 nm, 4 × 4 µm cell. Second-order convergence from res 64
(steps 0.0131 → 0.0033 in TE); the oxide-clad strip at the same settings
extrapolates to 2.438, the commonly quoted ≈ 2.44.

| res (px/µm) | TE n_eff | TM n_eff |
|---|---|---|
| 32 | 2.35982 | 1.56492 |
| 64 | 2.37293 | 1.57239 |
| 128 | 2.37618 | 1.57395 |
| → ∞ (2nd order) | **2.3773** | **1.5745** |

The top-view slab is 500 nm wide; the 3D TE-like mode is pysie2d `pol = 1`.
n_2D depends on what the slab's two lateral sides are:

| Lateral claddings of the top-view slab | n_2D (TE) | 3D n_eff that gives 2.763 |
|---|---|---|
| air \| air (the physical sides of the core) | 2.7920 | 2.3450 |
| oxide \| oxide | 2.7371 | 2.4037 |
| **air \| oxide** (the side view's claddings) | **2.7637** | **2.3766** |

**Finding.** The paper's Fig. 3 uses its eq. (2) with the side view's two
claddings, n_cl = 1 and n_sub = 1.44, on the two *lateral* sides of the
top-view slab. Only that reading gives 2.763, to the published digits, and it
implies a 3D n_eff within 0.03 % of MPB's. Inverse EIM reproduces β whatever
the lateral claddings are, by construction. The choice changes the 2D mode's
lateral tails, so it changes the coupling to a scatterer, which is what the
v1.4 3D MEEP reference measures. It is a convention v1.4 has to pin.

## Reproduce

    ~/miniforge3/envs/pysie2d-meep/bin/python mpb_neff.py 500 220 128
