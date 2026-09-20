# External validation — maintainer operations

This directory is **not installed and not imported by the package**. It holds
the drivers that produce the frozen reference spectra in `tests/data/`, and
the recipes for the environments they need. Nothing here runs in CI, and a
user who clones the repo and runs `uv sync --frozen && uv run pytest` never
touches it — see `docs/design/v1.0-external-validation.md` §6 for why that
separation is deliberate.

## Layout

| path | what it is |
|---|---|
| `spectrum.py` | The frozen-file format: schema, writer, reader. One definition, shared by the drivers here and by `tests/test_external_validation.py`. |
| `mie_bootstrap.py` | Freezes the repo's own analytic Mie series. **Not an external anchor** — it exists to prove the format and its consumer. |
| `dolfinx/` | FEM frequency-domain driver. Not yet written. |
| `meep/` | FDTD time-domain driver. Not yet written. |

## The three layers

1. **Frozen data** (`tests/data/*.json`) — committed, human-readable, carrying
   the geometry, material, polarisation mapping, tool version, converged
   settings, gate-2 convergence evidence and the tolerance justification.
2. **The drivers** (here) — runnable in an environment you build yourself,
   readable as the specification of what was actually simulated.
3. **The repo's test** (`tests/test_external_validation.py`) — numpy only,
   runs for everyone, fails if pysie2d's answers move.

## Regenerating

From the repo root, in the uv environment (no external tools needed):

```
uv run python validation/mie_bootstrap.py
```

It rewrites `tests/data/mie-circle-*.json` and prints the agreement achieved,
which is where the committed tolerances come from. Regeneration is deliberate:
the diff of a data file is itself the reviewable evidence that the physics
being compared changed.

The dolfinx and MEEP drivers will each carry their own environment recipe
beside them. Those environments are conda-first and are documented honestly as
such — there is no `[project.optional-dependencies]` group for them, because an
extras group most readers cannot install produces a skipped test that looks
like a defect.

## The polarisation trap

pysie2d's invariant axis is **y** (`E_y` for `pol = 2`/TE, `H_y` for
`pol = 1`/TM; conventions §2). Every external 2-D solver is **z**-invariant and
names its polarisations accordingly, so the mapping inverts the TE/TM labels.
Independently, pysie2d uses `exp(-iωt)`; a tool using `exp(+iωt)` returns the
conjugate field and wants the opposite sign of `Im ε` to absorb. Each frozen
file records the mapping it was written under in its `pol_mapping` field, and
`C_abs` is the observable that catches getting it wrong.
