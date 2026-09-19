# Figures

The `examples/` scripts generate the README figures, and they are part of
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
- `convergence_study` is semi-log error vs a linear `nn` axis for both
  polarisations: convergence is spectral since v0.6, which is a straight line
  there and not on log-log. Mark the round-off floor. If an algebraic order is
  ever claimed instead, go log-log and show the reference slope.
- `convergence_map` is log error over `nn` and log size parameter, jet by owner
  choice, smooth shading. Contours bracket the steep front (1e-3, 1e-12 dotted;
  1e-6 solid) and are black, since white vanishes in jet's yellow–cyan band;
  the resolution line is white. The 6-points-per-interior-wavelength line and
  the printed 4.7 median are measurements of this figure — rerun and update both
  together.
- Masked regions (the NaNs `relative_ldos_map` returns inside and near the
  particle) must read as "no data", visually distinct from a low value.
- Axes carry units (nm). Both polarisations should be distinguishable without
  relying on colour alone.
- `MPLBACKEND=Agg` in CI; figures must render headless.
