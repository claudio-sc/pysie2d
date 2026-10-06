# Parked — v1.0 external validation

**Parked 6 Oct 2026** while v0.9 (half-space) and the `Material.nc` gain fix
go first. PR #20 is back in draft. Nothing here is lost: the branch is pushed,
and the raw run outputs live in `validation/_study/`, which is gitignored and
on the maintainer machine only.

## Where it stands (`docs/design/v1.0-external-validation.md` §5)

- Gates 1–5 are done. dolfinx and MEEP reproduce Mie on the circle (gate 1).
  The convergence study (gate 2) and the per-tool null tests (gate 3) are run.
  The gate-4 star and the skew shape are compared, with the incidence
  direction pinned via MEEP's far-field pattern (`direction_check.py`). The
  spectra are frozen with measured tolerances (gate 5).
- Open: **gate 6**, re-running MEEP's `pml` / `box` / `decay` isolation groups
  at resolution 50. Until that runs, MEEP's tolerance is worded as a bounded,
  unlocalised systematic.
- Open: the absorption check for lossy dielectrics and metals is scoped
  (`a9ed659`) but not run.

## When resuming

1. Merge `main` in. The branch last merged it at v0.8.1, and v0.9 will have
   landed by then.
2. Run gate 6, which needs the conda MEEP environment (`validation/meep/`).
3. Add the v0.9 item that belongs here: MEEP as the independent anchor for
   QNMs over a dielectric substrate (layered-spec G9, owner decision 1).
4. Merging this is the 1.0 release, so ask before merging.
