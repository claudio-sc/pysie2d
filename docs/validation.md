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
| Transmitted Green function below a half-space (GT1) | `(i/4)·H₀` at `ε_rel = 1`, real and complex k: 7.2e-14 | `tests/test_transmitted.py` |
| Transmitted Green function (GT2) | a real-axis QUADPACK integral on the radiation-condition branch (`reference/sommerfeld.gt_ref`), glass, lossy, silver, low-n: 4.9e-11, the reference's own limit | `tests/test_transmitted.py` |
| Transmitted Green function at complex k (GT3) | itself on a path 1.5× deeper: 8.8e-15 (consistency; GT1 is the complex-k anchor) | `tests/test_transmitted.py` |
| Continuity across the interface (GT4) | `G_free + G_ind` above: 1.1e-13; the scattered field above, by Richardson in δ | `tests/test_transmitted.py` |
| Downward far field (GT5) | the free-space far field at `ε_rel = 1`: 5.1e-14; the Richardson limit of the near field at 40/80 µm: 3.2e-6 | `tests/test_transmitted.py` |
| Interface flux into the substrate (GT6) | the far-field integral over a lossless substrate: 2.6e-14 | `tests/test_halfspace_power.py` |
| Dipole power balance with a substrate (GT7) | energy conservation `P_up + P_down + P_abs = P_in`, five substrates: 8.7e-14 | `tests/test_halfspace_power.py` |
| Guards (GT8) | `NoFarFieldError` conditions, 0 below PEC, `c_sca_down = 0` exactly | `tests/test_transmitted.py`, `tests/test_halfspace_power.py` |
| `incident_field` (GT9) | the solver's own right-hand side on the boundary: 1.1e-15; continuity across the interface | `tests/test_transmitted.py` |
| Scale covariance below the interface (GT10) | an exact structural identity at s = 1.7: ≤ 3.7e-14 | `tests/test_scale_covariance.py` |
| v0.9 outputs above the interface (GT11) | themselves, bit for bit, with substrate points added | `tests/test_transmitted.py` |
| Banded transmitted GEMM (GT12) | the pointwise Green-function sum: 2.8e-16 | `tests/test_transmitted.py` |
| Stack: HalfSpace and free outputs unchanged (GW1) | themselves, bit for bit; the feature-list path with v0.9's features | the full suite, `tests/test_stack.py` |
| Stack reductions (GW2) | `HalfSpace` / no background, bit for bit at the solve level | `tests/test_stack.py` |
| Stack recursion (GW3) | a scaled transfer matrix: 4.3e-14 absolute; energy 2.2e-15; finite where the naive form is NaN | `tests/test_stack.py` |
| Reflected G over a stack (GW4) | real-axis QUADPACK with PV + iπ·Res (`g_ref_stack`): 2.6e-10; near cutoff (γ-reference) 6.6e-10; lossy 6.1e-14 | `tests/test_stack.py` |
| Transmitted G below a stack (GW5) | upward shooting (`gt_ref_stack`): 1.8e-11 | `tests/test_stack.py` |
| Guided modes (GW6) | the three-layer closed form: 2.8e-16; three counts on a double core; `NearDegenerateModesError` | `tests/test_stack.py` |
| Mode normalisation (GW7) | `Res_β R = γ₁/(βN)`: 4.2e-14 | `tests/test_stack.py` |
| Complex k and loss (GW8) | continued poles are zeros of D: 3.7e-16; path depth independence 4.3e-13 (consistency) | `tests/test_stack.py` |
| Dipole power over a bare stack (GW9) | energy conservation, scalar and in-plane: 7.0e-15 | `tests/test_stack.py` |
| Far field along the stack (GW10) | the closed-form guided sum, both directions: ≤ 4.9e-2 at 40 µm, decaying | `tests/test_stack.py` |
| Layer field (GW11) | interface conditions: 2.6e-13; cover match 6e-16; core field at 30 µm vs the guided sum 1.6e-2 | `tests/test_stack.py` |
| Dipole + particle (GW12) | energy conservation, scalar 4.0e-15, in-plane 1.3e-15 | `tests/test_stack.py` |
| Mode incidence (GW13) | `Σ|S|² + P_rad + P_abs = 1`: 2.4e-15 | `tests/test_stack.py` |
| Reciprocity (GW14) | `t_{n←m}(L→R) = t_{m←n}(R→L)`, lossy asymmetric particle: 1.1e-16 | `tests/test_stack.py` |
| Born limit (GW15) | first Born with the stack G: 5.0e-4 at a = 2 nm, ratio 3.4–3.7 per halving | `tests/test_stack.py` |
| In-plane dipole (GW16) | the electric-dipole pattern: 2.4e-15; ∇u_s vs a 5-point FD: 3e-10 (FD floor) | `tests/test_stack.py` |
| Mode sensitivity (GW17) | the exact translation law `i(β_m ∓ β_n)S`: 3.2e-9 | `tests/test_stack.py` |
| Scale covariance over a stack (GW18) | an exact structural identity at s = 1.7: 1.4e-15 | `tests/test_scale_covariance.py` |

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

| file | shape | pol | material | `rtol` C_ext | `rtol` C_sca | `rtol` C_abs | `atol` C_abs (nm) | null bound (nm) |
|---|---|---|---|---|---|---|---|---|
| `dolfinx-circle-lossless-te` | circle | TE | lossless | 5e-08 | 2e-06 | — | 1e-09 | 3e-12 |
| `dolfinx-circle-lossless-tm` | circle | TM | lossless | 1e-07 | 3e-06 | — | 1e-09 | 5e-12 |
| `dolfinx-circle-lossy-te` | circle | TE | lossy | 3e-08 | 1e-05 | 3e-08 | 2e-05 | — |
| `dolfinx-circle-lossy-tm` | circle | TM | lossy | 3e-08 | 1e-05 | 3e-08 | 2e-05 | — |
| `dolfinx-star-lossless-te` | star | TE | lossless | 1e-06 | 2e-06 | — | 5e-05 | 5e-05 |
| `dolfinx-star-lossless-tm` | star | TM | lossless | 1e-06 | 2e-06 | — | 2e-05 | 2e-05 |
| `dolfinx-star-lossy-te` | star | TE | lossy | 2e-07 | 1e-05 | 2e-07 | 5e-04 | — |
| `dolfinx-star-lossy-tm` | star | TM | lossy | 3e-07 | 1e-05 | 3e-07 | 5e-04 | — |
| `meep-circle-lossless-te` | circle | TE | lossless | 5e-04 | 5e-04 | — | 5e-01 | 3e-12 |
| `meep-circle-lossless-tm` | circle | TM | lossless | 5e-04 | 2e-04 | — | 5e-01 | 5e-12 |
| `meep-star-lossless-te` | star | TE | lossless | 5e-02 | 5e-02 | — | 5e-01 | 5e-05 |
| `meep-star-lossless-tm` | star | TM | lossless | 2e-02 | 2e-02 | — | 1e+00 | 2e-05 |
| `mie-circle-*` (4 files) | circle | both | both | 1e-12 | 1e-12 | 1e-12 | 1e-10 | 1e-10 |

One external check is not a spectrum: the **substrate QNMs** (layered-spec
G9). `tests/data/qnm/meep-substrate-qnm.json` freezes MEEP harminv poles of a
split whispering-gallery pair over glass, and
`tests/test_external_substrate_qnm.py` compares the substrate-induced shift at
1e-2 nm against shifts of 0.4–1.4 nm (measured worst 4.7e-3). Details in the
design doc, *Substrate QNMs against MEEP, as measured*.

**Every one of these is computed, not chosen.** `validation/freeze.py` measures
the worst pointwise deviation between pysie2d and the external spectrum *per
observable* and rounds it up to the next clean number; the justification
string inside each file gives each measurement beside the external tool's own
convergence drift **on the same observable**, and words the attribution from
the ratio rather than asserting it. Nobody can widen one to make a test pass
without editing the measurement it was derived from, which is reviewable in a
diff.

In every row each observable's deviation is within 3× of the tool's own drift
on that observable — which is what it means to say the tolerance bounds the
external solver's discretisation error rather than ours. Per observable
matters: dolfinx's `C_sca` is a flux of `∇u_s` over a facet set and converges
~60× slower than its volume-integral `C_ext`, so one shared `rtol` set by
`C_sca` used to give `C_ext` 40–230× slack, and quoting the `C_ext` drift beside
a `C_sca`-limited deviation made the gap look unexplained.

`atol` applies to `C_abs` only — never as a floor under `C_sca`/`C_ext`, where
for MEEP it would be the 0.5–1 nm absorption leak and would widen the stated
`rtol` up to 4×. The null bound is pysie2d's own lossless `C_abs` floor (×3),
not the reference tool's leak.

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
- **Absorption on a non-circular shape has one external anchor.** The lossy
  star (`n_core = 2.0`, `epsi = 0.5`, `n_clad = 1.33`) is checked against
  dolfinx alone: worst `C_abs` deviation 8e-08 of peak `C_ext` at L4, falling
  ~5× from L3, so it is a bound and not a floor.
- **Absorption anywhere rests on a single external tool.** MEEP has no
  frequency-independent `Im ε`, so the lossy cases are dolfinx alone, against
  the closed-form lossy Mie series. The two externals do not cross-check each
  other there.
- **Metals have no external anchor.** `Material.from_eps` (v0.9) expresses
  `Re ε < 0`, and a silver cylinder is checked against the analytic Mie series
  (`test_metal_absorption_matches_mie` in `tests/test_efficiencies.py`). Metal
  substrates are checked by closed forms and energy conservation only. Neither
  MEEP nor dolfinx has been run on a metal. The scope was
  [docs/design/absorption-scope.md](design/absorption-scope.md).
- **MEEP's systematics are attributed, not eliminated.** At its production
  resolution 50: transform truncation is negligible (≤ 7e-06); the flux
  contour's placement on the grid costs ~1.4e-04 in both polarisations and is
  what flattens the order near 50; in TE the sheet source's ends in the PML
  make the illumination non-uniform (~0.3 % across the box), worth ~3.5e-04.
  The last two are bounded, not removed, and never read as a converged floor
  (design doc, *Gate 6, as measured*).
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
