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
| `gielis.py` | Gate 4's star: the superformula in numpy alone, and the sampled contour both drivers read from `data/gielis-star.npz`. Neither driver may import pysie2d, so this is the one place the shape is written down; `tests/test_gielis_contour.py` asserts it equals the package's own `gielis()` bit for bit. |
| `mie_bootstrap.py` | Freezes the repo's own analytic Mie series. **Not an external anchor** — it exists to prove the format and its consumer. |
| `dolfinx/` | FEM frequency-domain driver. Carries the lossy cases: it represents a constant `Im ε` exactly. |
| `meep/` | FDTD time-domain driver. Lossless only — MEEP has no frequency-independent `Im ε`, so a broadband run would make the material dispersive. |

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

## The incidence trap

Both drivers pin the incident wave to pysie2d's `angle = 0`, travelling along
**−z** (their own −y). On a circle that choice is unobservable: reversing it
leaves every cross-section unchanged, so the convention has never been
falsifiable. The six-fold star is not symmetric under the flip, so `--angle-deg
180` — offered by both drivers, and not a production setting — makes it fail
loudly: the two spectra must differ, and only one can match pysie2d.

## The polarisation trap

pysie2d's invariant axis is **y** (`E_y` for `pol = 2`/TE, `H_y` for
`pol = 1`/TM; conventions §2). Every external 2-D solver is **z**-invariant and
names its polarisations accordingly, so the mapping inverts the TE/TM labels.
Independently, pysie2d uses `exp(-iωt)`; a tool using `exp(+iωt)` returns the
conjugate field and wants the opposite sign of `Im ε` to absorb. Each frozen
file records the mapping it was written under in its `pol_mapping` field, and
`C_abs` is the observable that catches getting it wrong.
