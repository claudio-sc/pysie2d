# ε-Beyn prototype on pysie2d main (a14e407, 2026-11-02; package version 1.2.1)

Run any script from a pysie2d checkout root, e.g.
`uv run python /path/to/eps_beyn_prototype/test_circle_tm.py`.
Scripts find their sibling modules and data files next to themselves, whatever the working directory.
`note/figs.py` also needs matplotlib: `uv run --with matplotlib python .../note/figs.py`.

## Single particle

| file | what it measures |
|---|---|
| epsbeyn.py | `EpsOperator` (exterior rows once per λ, interior rows per ε, branch choice) + `solve_box` (beyn_modes + newton_refine) |
| anchor.py | analytic circle eigenpermittivities from the Mie denominators at fixed x, winding-number counts |
| test_circle_tm.py | TM plasmonic box vs analytic, both branches (upper: 1e-15 at nn=32; principal: breaks at nn=64/96) |
| test_spurious.py, test_sigma.py | M(ε) exactly singular at H_n^(1)(√ε x)=0 on the principal branch; left null vector 100 % interior rows |
| test_side.py | same spurious set in the λ-QNM operator (Q≈0.17); exterior irregular frequency k_bg a = j01 singular for all ε |
| test_te_passive.py | upper branch == BIESolver.assemble bit-for-bit on Im ε>0; long TE box failure |
| test_te_split.py, test_te_sv.py, test_te_missing.py | split boxes recover 8/8 TE dielectric modes; diagnosis of the long-box failure |
| test_zero.py | TM: ε=0 is an exact eigenvalue of M (σ_min ∝ |ε|) |
| test_reciprocity.py | boundary form of the paper's eq. 34: w_m^T b(r') = κ_m F_m(r') to 1e-14 |

## Particle over the SiN slab

| file | what it measures |
|---|---|
| test_stack.py | rotated Gielis triangle 140 nm above 400 nm SiN in SiO2, 1550 nm TM: modes, nn-convergence, timings (importable: definitions only) |
| test_expand.py, test_expand2.py | mode expansion of the guided amplitude vs exact driven solves; test_expand2.py writes expand2.npz |
| test_lump.py, chart_data.py | remainder models (quadratic, lumped at −ε_b); chart data. Read expand2.npz |
| test_channels.py | power account of each eigenvector (balance to 1e-14) and guided fraction |

## Performance

| file | what it measures |
|---|---|
| perf.py | per-ε cost vs nn: full assembly (free / slab), interior rows only, factorisation, solve; mode-expansion evaluation cost. Defines `interior_rows` used by the cluster scripts |
| perf2.py | smallest contour that finds the slab modes; cluster per-ε cost vs number of particles |

## Clusters (re-expansion)

| file | what it measures |
|---|---|
| cluster_reexp.py [pitch_nm] | 4 identical triangles over the slab: full cluster vs route 1 (exact reduction by compressed neighbour coupling), route 2 (isolated modes + quadratic leftover), route 3 (pure mode hybridization) |
| cluster_reexp_bg.py [pitch_nm] [free] | same, with `free` for no slab |
| cluster_scale.py | route 1 vs full: cost and memory per contour point for 4 / 8 / 16 particles |
| driven_reduction.py [P] | route 1 for ordinary driven solves: matches the full cluster solve over 40 permittivities, cost per permittivity |

## Note

| file | what it is |
|---|---|
| note/boundary_expansion.tex | source of boundary_expansion.pdf (paper's eqs. 21 and 26 carried to the boundary form) |
| note/figs.py | its two figures; reads ../expand2.npz |

Data: expand2.npz (the slab sweep used by test_lump.py, chart_data.py, test_channels.py, note/figs.py) is included; regenerate with test_expand2.py.
