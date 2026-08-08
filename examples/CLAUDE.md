# Figures

The three `examples/` scripts generate the README figures, and they are part of
the package's public face. **Load the `dataviz` skill before writing or changing
any plotting code** — take its color formula, accessibility checks, and mark
specs; ignore the dashboard/KPI material, which does not apply here.

Conventions specific to these figures:

- `purcell_map` is **diverging around a physically meaningful midpoint**
  (`relative_ldos = 1`, the free-space value). Enhancement and suppression must
  be visually symmetric about it, and the midpoint must be pinned — an
  auto-scaled diverging colormap that centres on the data mean is wrong here.
- `nearfield_map` is sequential magnitude data; use a perceptually uniform
  colormap, not `jet`.
- `convergence_study` is log-log error vs `nn` for both polarisations. If a
  convergence *order* is being claimed, show the reference slope.
- Masked regions (the NaNs `relative_ldos_map` returns inside and near the
  particle) must read as "no data", visually distinct from a low value.
- Axes carry units (nm). Both polarisations should be distinguishable without
  relying on colour alone.
- `MPLBACKEND=Agg` in CI; figures must render headless.
