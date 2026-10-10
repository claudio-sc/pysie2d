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
- **Gate 6 done 2026-10-10**: MEEP's systematics at resolution 50 are
  attributed (design doc, *Gate 6, as measured*).
- Open: the absorption check for lossy dielectrics and metals is scoped
  (`a9ed659`) but not run.

## When resuming

1. ~~Merge `main` in~~ — done at v0.9.0 (`ee30335`).
2. Add the v0.9 item that belongs here: MEEP as the independent anchor for
   QNMs over a dielectric substrate (layered-spec G9, owner decision 1).
3. Merging this is the 1.0 release, so ask before merging.
