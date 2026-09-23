# How pysie2d is validated

One page, one question: *why should anyone believe the numbers this package
produces?*

The short answer is that every observable pysie2d exposes is checked against
something that is **not pysie2d** — a closed form where one exists, and two
independent Maxwell solvers where one does not. All of that evidence is
committed to this repository as data, and the test that reads it runs for
anyone who clones the repo, with no simulation software installed.

```
git clone … && cd sie
uv sync --frozen
uv run pytest
```

That is the whole reproduction path. No conda, no MEEP, no dolfinx, no
downloads. If it passes, every claim below has been re-checked on your machine.

---

## 1. The three layers

**Layer 1 — closed forms.** Where the physics has an analytic answer, the
package is checked against it directly, in the test suite, every run.

| what is checked | against | where |
|---|---|---|
| Cross-sections on a circle | analytic Mie series (`reference/mie.py`) | `tests/test_efficiencies.py`, `tests/test_convergence.py` |
| Quasinormal modes | complex-`x` poles of the Mie coefficients | `tests/test_mie_qnm.py` |
| The contour eigensolver itself | matrix pencils with exactly known spectra | `tests/test_beyn.py` |
| Coupled scattering from two cylinders | the two-cylinder addition theorem | `tests/test_two_cylinder.py` |
| Multipole decomposition | analytic Mie coefficients | `tests/test_multipole.py` |
| Eigenvalue sensitivity `dλ/dp` | closed-form derivatives | `tests/test_sensitivity.py` |
| Scale covariance `λ(s·rad) = s·λ(rad)` | an exact structural identity | `tests/test_scale_covariance.py` |

These are strong but they share one limitation: **they are all circular.** The
superformula reduces to a circle at `m = 0`, and that is the only boundary for
which any closed form exists. A Gielis star has no analytic solution.

**Layer 2 — external solvers, frozen.** For the non-circular case the reference
has to be another solver. Two were used, and their spectra are committed to
`tests/data/` as JSON, each carrying its full provenance.

**Layer 3 — the test that runs for everyone.** `tests/test_external_validation.py`
reads those files with the standard library and compares with numpy. The
external tools produced the reference once; they are not needed to check it.

---

## 2. Why two external solvers

Neither tool is asked to do the regime it is bad at.

| | MEEP | FEniCSx / dolfinx |
|---|---|---|
| Method | FDTD, time domain, Yee grid | FEM, frequency domain, curved elements |
| Shares with pysie2d | nothing — not the formulation, discretisation or linear algebra | a volume formulation with PML |
| Carries | the most independent check available | the sharp, quantitative claim |
| Limitation here | first-order on a cusped boundary; lossless only | less independent — still a PDE discretisation |

MEEP is the more valuable check precisely because nothing in a time-stepped
Yee grid resembles a boundary-integral equation. It is also the coarser one:
its agreement on the star is one to two significant figures (median 0.8%, worst 4.8%). dolfinx agrees to parts in 10⁶.
Both are frozen, because a loose check from a maximally independent method and
a tight check from a less independent one answer different questions.

## 3. What was actually simulated

Two shapes, both at `rad = 200 nm`, on a 51-point vacuum-wavelength grid from
400 to 900 nm, in both polarisations.

- **Circle** — `n_core = 1.5`. Lossless, and lossy (`n_clad = 1.33`,
  `epsi = 0.5`) under dolfinx.
- **Star** — the Gielis shape the README figures draw: `m = 6`, `n1 = 6`,
  `n2 = n3 = 12`, `n_core = 2.0`. Lossless.

The star's boundary is defined once, in `validation/gielis.py`, in numpy and
the standard library alone. Neither external driver imports pysie2d, and
`tests/test_gielis_contour.py` asserts that this contour equals the package's
own `gielis()` **bit for bit** — not to a tolerance. That is what keeps a
disagreement about physics from being readable as a disagreement about
geometry.

## 4. The committed tolerances

| file | shape | pol | material | `rtol` | `atol` (nm) | tool's own drift |
|---|---|---|---|---|---|---|
| `dolfinx-circle-lossless-te` | circle | TE | lossless | 2e-06 | 1e-09 | 1.82e-07 |
| `dolfinx-circle-lossless-tm` | circle | TM | lossless | 3e-06 | 1e-09 | 2.39e-07 |
| `dolfinx-circle-lossy-te` | circle | TE | lossy | 1e-05 | 2e-05 | 1.04e-07 |
| `dolfinx-circle-lossy-tm` | circle | TM | lossy | 1e-05 | 2e-05 | 1.12e-07 |
| `dolfinx-star-lossless-te` | star | TE | lossless | 2e-06 | 5e-05 | 3.94e-06 |
| `dolfinx-star-lossless-tm` | star | TM | lossless | 2e-06 | 2e-05 | 2.49e-06 |
| `meep-circle-lossless-te` | circle | TE | lossless | 5e-04 | 5e-01 | 3.29e-04 |
| `meep-circle-lossless-tm` | circle | TM | lossless | 5e-04 | 5e-01 | 6.75e-04 |
| `meep-star-lossless-te` | star | TE | lossless | 5e-02 | 5e-01 | 1.03e-02 |
| `meep-star-lossless-tm` | star | TM | lossless | 2e-02 | 1e+00 | 2.32e-03 |
| `mie-circle-*` (4 files) | circle | both | both | 1e-12 | 1e-10 | — |

**Every one of these is computed, not chosen.** `validation/freeze.py` measures
the worst pointwise deviation between pysie2d and the external spectrum and
rounds it up to the next clean number; the justification string inside each
file states that measurement, names the observable that set it, and gives the
external tool's own convergence drift beside it. Nobody can widen one to make a
test pass without editing the measurement it was derived from, which is
reviewable in a diff.

The last column is the point. In every row the tool's own level-to-level drift
is the *same order* as its disagreement with pysie2d — which is what it means
to say the tolerance bounds the external solver's discretisation error rather
than ours.

### Why the star files say `n_pts = 400`

Each frozen file records the boundary discretisation pysie2d should use when
reproducing it. The circle files say 200; the star files say 400, and the
difference is load-bearing. Measured on the frozen star spectrum:

| `n_pts` | worst deviation | verdict |
|---|---|---|
| 100 | 9.35e-03 | fails the 2e-06 tolerance |
| 200 | 1.34e-04 | fails |
| 400 | 1.55e-06 | passes |
| 800 | 1.55e-06 | passes — identical |

At 400 and 800 the deviation is the *same number*, because it has saturated on
dolfinx's error and pysie2d has stopped contributing. At 200 the test would
have been measuring pysie2d's own discretisation and attributing it to dolfinx.

### These tests cannot pass by accident

Each file pins a full 51-point spectrum, not a scalar. A polarisation swap on
the frozen star case misses by a factor of **897,000**. A sign error, a lost
factor of `n_clad`, or a wrong incidence convention moves every point at once.

## 5. What is *not* validated

Stated plainly, because a validation page that only lists successes is not
evidence.

- **One non-circular shape.** Everything non-circular rests on a single Gielis
  star. It is a demanding one — six near-cusps — but it is one.
- **Absorption on a non-circular shape has no external anchor.** Every star run
  is lossless. `C_abs` off a circle is checked by nothing.
- **Absorption anywhere rests on a single external tool.** MEEP has no
  frequency-independent `Im ε`, so the lossy cases are dolfinx alone, against
  the closed-form lossy Mie series. The two externals do not cross-check each
  other there.
- **Metals are out of scope entirely.** `Material` cannot express `Re ε < 0`:
  `epsr = (n_core/n_clad)²` is non-negative by construction. Nothing with a
  negative real permittivity has been run or checked.
- **MEEP's systematics are bounded, not localised.** Its absorber, flux-contour
  and transform-truncation sensitivities were measured at resolution 25 and the
  frozen spectra are at 50 (circle) and 32 (star). They bound those effects
  where they were measured; they are documented as *a bounded systematic we did
  not eliminate*, never as a converged floor.
- **MEEP on the star is first order, not second.** Six near-cusps on a Yee grid,
  which subpixel averaging cannot rescue. Measured separately, the cusps cost
  9.6× and the `n_core = 2.0` contrast 4.2×; their product is the observed 40×
  gap against MEEP's own circle.
- **Cross-sections only.** Near field, LDOS and Purcell factor have analytic
  circular anchors but no external non-circular check.

## 6. Reproducing the external runs

Only needed if you want to regenerate the references rather than check them.
Both drivers live in `validation/`, run in their own conda environments, and
are readable as the specification of what was simulated.

```
uv run python validation/study.py --dry-run     # the matrix and its cost
uv run python validation/study.py --tool all    # the convergence study
uv run python validation/study.py --analyse     # the floors
uv run python validation/compare.py --tool dolfinx --name final-star-L4 \
    --case lossless-te --nn 800                 # one comparison
uv run python validation/freeze.py --dry-run    # what would be frozen
```

`validation/README.md` documents the layout, the environments, and the two
convention traps (incidence direction and polarisation naming) that cost the
most to get right.
