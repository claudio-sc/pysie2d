# Multiple particles — code-spec (draft 1)

**Status:** draft 1, 20 Sep 2026. Not implemented. Supersedes the decision
sections of [multiparticle-handoff.md](multiparticle-handoff.md), which remains
the guide to *why*; where the two disagree, **this document wins**.

**Who it is for:** the worker who lands multiple particles. You do not need to
derive anything, you do not need to read the legacy research code, and you do
not need to invent a single constant. §4 gives the implementation as code that
has been run. §6 gives every test with the number it must reach. Where this
document says "measured", the measurement is reproducible from §9.

**Measured on this machine** against `9d6e288` (branch `v0.7-multipole`,
pysie2d 0.6.0): the convention bridge (§3.7), the bit-identical `Np = 1`
reduction (§6.1), the G2 decay slope (§6.2), the spectral convergence and
truncation window of the G4 anchor (§3.8, §6.4), the reciprocity residual
(§6.5), the near-gap envelope pilot (§5.2), and the `Np` timing pilot (§5.1).

**Two numbers are deliverables of this work, not inputs to it:** the final
`Np` ceiling (M1) and the envelope constant `GAP_ENVELOPE_C` (M2). §5 gives the
pilot data, the method, and exactly what remains to be swept. Everything else is
fixed here.

**Decisions it works within:** [CLAUDE.md](../../CLAUDE.md) non-negotiables 1–4,
[docs/conventions.md](../conventions.md) §1–§6, §9, §13. This work **adds**
conventions §14 (the cluster degree-of-freedom layout) and changes nothing
above it.

---

## 0. How to execute this spec

Read §1, §2, §3 and §10 before writing anything. §4 tells you *what* to write;
§3 tells you *why*, which is what you need when a number does not match.

### 0.1 The procedure

1. `git switch main && git pull`. Confirm the v0.7 multipole work has merged:
   `uv run python -c "import pysie2d; pysie2d.Multipoles"` must succeed. If it
   does not, **stop and report** — you are on the wrong commit.
2. `git switch -c v0.8-multiparticle`.
3. Do commit 1 of §7 (the roadmap renumber) **first and alone**. It touches no
   code.
4. Write the code of §4 and the tests of §6, in the commit order of §7.
5. Run the measurement steps of §5 where §7 places them. M2 produces a constant
   that §4.9's guard needs; until M2 is done, that guard is not written.
6. Run the acceptance checklist, §8. Every item must pass as written.
7. `git push -u origin v0.8-multiparticle` and open a **draft PR as soon as
   commit 2 exists** — CI runs only on PRs and on `main`, so a branch with no PR
   gets no CI. Short body, no signature.
8. Report back with the §8 checklist ticked and the CI link.

**Do not merge to `main`.** Every `feat:` here bumps the version and publishes to
PyPI; that is the owner's decision (CLAUDE.md, *Git workflow*).

### 0.2 Rules that are not negotiable

- **Never widen a tolerance to make a test pass** (non-negotiable 4). Every
  tolerance in §6 is a measured value with headroom already applied. If an
  assertion fails, or a measured value comes out more than 3× worse than the
  number quoted, **stop and report both numbers**. Do not re-measure and update
  the constant.
- **Complex `k` survives everywhere** (non-negotiable 1). The cross-block kernel
  is written once and must accept a complex `wnum_bg` unchanged. Do not add a
  real-only fast path the complex case cannot take.
- **Do not edit anything listed out of scope in §1.2.**
- **Do not copy the legacy file in.** `~/Documents/sie-legacy-0726/sie/bie.py`
  §540–1000 is a different solver's conventions (pre-Kress, pre-`Parametrisation`,
  pre-vacuum-wavelength) and carries the four assumptions of handoff §3.1 inside
  its *indexing*. Write §4 from this document.
- **Do not reformat or "tidy" code this spec does not touch.**
  `uv run ruff format --check` must be clean when you are done.
- **If something here does not work, stop and report it.** Do not improvise a
  fix to a spec'd formula. The formulas in §3 and the code in §4 have been run.

### 0.3 The one thing most likely to go wrong

**The cross-block quadrature degrades silently as a gap closes.** Nothing raises,
nothing looks wrong, and the answer is confidently incorrect. Measured (§5.2,
TE, two circles of radius 180 and 110 nm, λ = 633 nm): at a gap of 0.016 λ,
`nn = 30` is wrong by 7.4e-4 and `nn = 60` by 5.2e-6, while at a gap of 1.4 λ
both sit at round-off. The guard of §4.9 is the substance of this feature, not
decoration.

The second most likely is an **indexing error across particles**. G1
(bit-identical `Np = 1`) will not catch it on a symmetric configuration; G5
(reciprocity) will. Both are cheap. Write them first.

---

## 1. Scope

### 1.1 In

- A finite cluster of **arbitrary** particles: different shapes, different sizes,
  different materials, different `n_pts` per particle. All four legacy
  assumptions of handoff §3.1 are removed.
- Coupled assembly, dense solve, plane-wave and line-dipole excitation.
- Observables: total far-field amplitude, absolute cross-sections
  `C_sca`/`C_ext`/`C_abs` in nm, near fields at arbitrary points.
- `ScatterResult.cross_sections()` on the **single-particle** path, so both
  paths report the same quantity (D5).
- A two-cylinder addition-theorem reference anchor,
  `src/pysie2d/reference/two_cylinder.py`.
- Validation gates G1–G5 of handoff §5, plus the studies M1 and M2.

### 1.2 Out — do not touch

- `kernels.assemble_matrix`, `assemble_matrix_reference`, `assemble_matrix_dwn`
  and every existing test of them. The self-blocks **are** `assemble_matrix`,
  called verbatim. If you find yourself editing it, you have taken a wrong turn.
- `fields.eval_field`, `fields.far_field`, `fields._far_field_at`,
  `sources.plane_wave_rhs`, `sources.line_dipole_rhs`. All reused unchanged.
  §4.5 adds a *new* private helper beside `eval_field`; it does not refactor it.
- `ScatterResult.efficiencies()`, `BIESolver`, `QNMSolver`, `beyn.py`,
  `multipole.py`, `green.py`, `geometry.py`, `parametrisation.py`,
  `material.py`, `reference/mie.py`.
- `docs/conventions.md` §1–§13. You **add** §14 and change nothing above it.

### 1.3 What this feature does not cover

Each is a named future milestone, not a limit to be quietly worked around:

- **Periodic or infinite arrays.** A finite cluster is this feature.
- **Particles in contact or overlapping.** Excluded by the formulation itself
  (§3.1): the argument that the M3/M4 cross-blocks vanish needs the interior
  domains disjoint. §4.2 raises.
- **A substrate or layered background.** Separate roadmap item.
- **QNMs of a cluster.** Beyn's method does not care about the internal
  structure of `M(λ)`, so §4.6's assembly makes it nearly free — but the
  half-plane argument (conventions §8) and degeneracy handling would both need
  re-examining. Named as future work; **not** attempted here, which is why
  `ClusterBIESolver` exposes no public `assemble` (D12).
- **LDOS inside a cluster.** `line_dipole_rhs` stacks (§3.3) so the *excitation*
  ships, but `relative_ldos` and the self-Green sign convention (conventions §7)
  are not extended. Separate milestone.
- **Cluster multipoles.** The v0.7 expansion is about *a* particle centre; a
  cluster needs an output-side translation theorem that `multipole.py`
  explicitly does not implement.

---

## 2. Decisions

Owner's answers, 20 Sep 2026. Each supersedes the corresponding handoff section.

| | Decision | Reasoning |
|---|---|---|
| **D1** | **Additive.** New `Cluster`, `ClusterBIESolver`, `ClusterScatterResult` alongside `BIESolver`/`ScatterResult`, whose signatures do not change. | Handoff §4.1. The single-particle path stays the simple thing; most users have one particle and making them build a one-element cluster taxes the common case. The cost — a duplicated result façade — is confined to the façade, because every primitive it calls is already per-boundary and additive. |
| **D2** | **`Cluster` holds geometries only; materials live on the solver.** `Cluster(geometries)`, `ClusterBIESolver(cluster, materials, pol=2)`. | Answers handoff §8's first open question. An arrangement is a geometric fact; it can be swept over material sets without rebuilding, and moving a particle never touches a material. |
| **D3** | **`pol` is an explicit argument of `ClusterBIESolver`**, and every `Material.pol` is validated equal to it and to each other, naming the offending index. `Material` is **not** changed. | Handoff §4.2 option 1, with the polarisation visible in the signature. `pol` is a property of the problem, not of a particle (conventions §1), so `Np` copies must at minimum be asserted to agree. Splitting `pol` out of `Material` is cleaner and still right long-term, but breaks a published constructor and every example, figure and test. |
| **D3b** | **`n_clad` is validated equal across all materials**, with its own error. | **Not in the handoff; found while writing §3.1.** `Material.wnum_bg` is `2π·n_clad/λ_vac`, and a cluster has exactly one background. With differing `n_clad` the background wavenumber is ambiguous, every cross-block is meaningless, and nothing else would notice. `nc` and `eps` are background-relative (conventions §2) and may differ freely — that is the per-particle material freedom D1 promises. |
| **D4** | **Per-particle contiguous DOF ordering:** `[φ_0, χ_0, φ_1, χ_1, …]`. Recorded as conventions **§14**. | Handoff §4.3. With ragged `nn_p` the legacy `[φ_0…φ_{Np−1}, χ_0…χ_{Np−1}]` layout needs two offset tables and makes no block contiguous. Contiguous means each particle's `2·nn_p` sub-vector goes straight into the existing single-particle primitives — which is why §6.1 comes out **bit-identical** rather than merely close. It does **not** extend conventions §4; §14 must say so, so nobody later "fixes" it back. |
| **D5** | **Absolute cross-sections `C_sca`/`C_ext`/`C_abs` in nm.** `ScatterResult` gains `cross_sections()`; `efficiencies()` untouched. No `efficiencies()` on the cluster. | Handoff §4.6. `efficiencies()` normalises by `2·rad`, approximate for a non-circular shape and undefined for a cluster. The external-validation comparison contract (v1.0, then drafted as v0.9) already fixed absolute cross-sections for the same reason. |
| **D6** | **Overlap raises; a gap below the measured envelope warns.** | Handoff §4.5. The failure mode here is a silent wrong answer, so it must be loud. Raising on a sub-envelope gap was rejected: the envelope is circle-measured and the user may legitimately be exploring. |
| **D7** | **The user picks `nn_p`; the solver warns on imbalance.** | Handoff §4.4. Per-particle resolution is the point of ragged `nn`, but the coupled system is only as accurate as its worst block and that is not locally visible. Resampling the user's geometries was rejected — it collides with the frozen-map convention (§10). |
| **D8** | **Serial, with the threading seam left in.** | Handoff §4.7. `hankel1` releases the GIL and `contour_moments` proves a 5.02× threaded win (performance.md §3.1), so threading is the right *next* move — but writing it before M1 motivates it is optimising ahead of evidence. The seam costs one list comprehension. |
| **D9** | **The envelope is measured on circles only.** For a pair where either boundary is non-circular the same check runs, but the warning says the rule is circle-measured and unvalidated there. | Handoff §8's fourth open question stays open, honestly. Diagnosing rather than claiming is the honest middle. |
| **D10** | **`src/pysie2d/reference/two_cylinder.py`** — installed, alongside `reference/mie.py`. | Same role as `mie.py`: a deliberate second implementation kept as a validation anchor, and therefore off-limits to `/simplify` (CLAUDE.md, *Review*). A cold visitor can run it from an install. |
| **D11** | **Ships as v0.8.0.** External validation renumbers to **v0.9**. | This work is finished and additive; external validation is conda-first and CI will never run it. Roadmap numbers follow ship order. This is the *second* renumber of that milestone — §7.1 lists every file, so do it once, completely. **Superseded 20 Sep 2026**: the owner decided external validation ships as **v1.0** itself rather than a v0.9 step before it — see CLAUDE.md's roadmap and [v1.0-external-validation.md](v1.0-external-validation.md). §7.1's renumber below is historical; it renumbered to v0.9, which a later commit renumbered again to v1.0. |
| **D12** | **No public `assemble` on the cluster solver;** it is `_assemble`. | The only caller for a public one would be a cluster `QNMSolver`, which §1.3 excludes. Exporting an unvalidated entry into the eigenvalue machinery invites exactly the use this milestone has not validated. |

---

## 3. The mathematics

### 3.1 The coupled system

`N = 2·Σ_p nn_p`. In the D4 layout particle `p` occupies `[o_p : o_p + 2·nn_p]`
with `o_p = Σ_{q<p} 2·nn_q`; the first `nn_p` entries of that slice are `φ_p`,
the next `nn_p` are `χ_p`.

The block at (field particle `q`, source particle `p`) is `2·nn_q × 2·nn_p`.

**Diagonal (`q = p`)** is the existing single-particle matrix, unchanged, at that
particle's own `nn_p`, `nc_p`, `eps_p`, and the **common** `k_bg` (that is D3b).
The whole v0.6 Kress–Martensen validation carries over with nothing to re-derive.

**Off-diagonal (`q ≠ p`)**:

```
[ M1_cross   M2_cross ]     rows 0 … nn_q−1       (the exterior equation)
[    0           0    ]     rows nn_q … 2nn_q−1   (the interior equation)
```

The lower half is **exactly zero**, and the whole feature rests on it: the
interior Green's function of particle `p` is confined to particle `p`'s own
volume, so coupling enters only through the exterior background Green's
function. The argument needs the interior domains disjoint, which is why overlap
is excluded by the formulation and not merely by an implementation limit.

### 3.2 The cross-block kernels

The two boundaries never touch, so the integrand has **no singularity**, the
Kress correction `W_d` is identically zero off-diagonal, and what remains is the
plain periodic trapezoid rule — spectrally accurate on analytic boundaries,
exactly like the Kress scheme beside it. No new singular quadrature anywhere.

With field node `i` on `q`, source node `j` on `p`, and `h_p = 2π/nn_p` the
**source** particle's quadrature step:

```
dx   = f_q[i] − f_p[j]
dz   = g_q[i] − g_p[j]
z    = k_bg · √(dx² + dz²)
c_ij = dx·dg_p[j] − dz·df_p[j]

M1_cross[i, j] = k_bg² · 0.25j · h_p · H₁⁽¹⁾(z)/z · c_ij
M2_cross[i, j] = 0.25j · h_p · H₀⁽¹⁾(z)
```

These are `assemble_matrix`'s `double_bg` and `single_bg` with `w_tri = 0`.
Three things about them:

- **`h_p`, not `h_q`.** The integral is over particle `p`'s boundary. Invisible
  at equal `nn`; a wrong answer that raises nothing the moment `nn` varies.
- **`df_p`, `dg_p`, not `df_q`.** The normal belongs to the source.
- **No `eta`.** `eta = eps` for `pol = 1` multiplies M4 only, and M4 cross is 0.

### 3.3 The right-hand side

Because of D4 the single-particle RHS builders drop in **verbatim**: each returns
a `(2·nn_p,)` vector in exactly `[φ_p, χ_p]` order, which is exactly the slice it
occupies. `line_dipole_rhs`'s inside/too-close guards then run once per particle,
which is precisely the cluster condition (the source must lie outside *every*
particle). No new source code is written.

### 3.4 The far field

Additive over boundaries, and `f_p`, `g_p` are **absolute** coordinates, so the
geometric phases are already inside `exp(−i k(f sinθ + g cosθ))`. **There is no
extra phase factor to apply** — a tempting and wrong addition.

### 3.5 The near field

Outside every particle: the sum of the representation integral over all
boundaries at `k_bg`. Inside particle `p`: that boundary alone at
`k_core,p = nc_p · k_bg`. Overlap being excluded, a point is inside at most one.

Classification uses `sources._point_inside` (ray-casting), **not**
`fields._is_outside` (nearest-point normal). The latter is documented as
unreliable for concave superformula shapes, and a cluster runs it `Np` times, so
a misclassification is `Np` times more likely. Deliberate divergence from
`ScatterResult.eval_field`, which is out of scope to change.

### 3.6 Absolute cross-sections

```
dσ/dθ = |amp(θ)|² / (8π·k_bg)             (nm per radian)
C_sca = Σ_{i<nff−1} dσ/dθ|_i · Δθ,        Δθ = 2π/(nff−1)
C_ext = Im[amp(π − α)] / k_bg
C_abs = C_ext − C_sca
```

`C = Q · 2·rad` exactly, which §6.6 asserts rather than assumes. Two carried-over
details, both already commented in `efficiencies()`: the far-field grid spans
`[−π, π]` **inclusive**, so index 0 and `nff−1` are the same direction and the
sum drops one of them; and `π − α` lands on that grid only by accident, so it is
evaluated exactly with `_far_field_at`.

`C_ext` presumes a unit-amplitude incident **plane wave**, so `cross_sections()`
is meaningful only for plane-wave excitation (§4.8, §11).

### 3.7 The convention bridge — verified to 1.8e-15

G4 compares the solver against an independent closed form across three
conventions. Getting any wrong gives a confidently wrong answer of the right
magnitude. All three were verified numerically (§9, `bridge.py`), **1.2e-15 to
1.8e-15 in both polarisations at normal and oblique incidence**. Take them as
given; do not re-derive.

Work the addition theorem entirely in the **standard** polar convention
`(x = ρ cos φ, z = ρ sin φ)` inside `two_cylinder.py`, converting only at the two
boundaries. Forcing the solver's θ-from-`+z` convention through the translation
algebra buys nothing and risks a silent rotation by `e^{imπ/2}`.

**Angle mapping.** The solver's observation direction is `r̂ = (sin θ, cos θ)`
and its incidence direction is `k̂_i = (sin α, −cos α)` for `angle = α` degrees:

```
φ   = π/2 − θ            (observation)
α_i = deg2rad(α) − π/2   (incidence)
```

**Mie coefficient sign.** With
`ψ = Σ_n i^n e^{−inα_i}[J_n(kρ) + s_n H_n⁽¹⁾(kρ)]e^{inφ}`,

```
s_n = − c_n ,   c_n = mie.an(|n|, x, m)  for pol = 1 (TM)
                c_n = mie.bn(|n|, x, m)  for pol = 2 (TE)
```

with `x = k_bg·a`, `m = Material.nc`. The sign is forced, not chosen: for a
lossless scatterer unitarity of `S_n = 1 + 2s_n` requires `Re s_n = −|s_n|²`,
while `mie.efficiencies`' `Q_ext = Q_sca` at zero loss requires
`Re c_n = +|c_n|²`. `c_{−n} = c_n`, so negative orders need no separate
evaluation.

**Far-field normalisation.** With
`S(φ) = Σ_j e^{−i k R_j·r̂} Σ_n A_n^{(j)} (−i)^n e^{inφ}`,

```
amp_solver(θ) = −4i · S(π/2 − θ)
```

exactly — both sides then give `dσ/dφ = |amp|²/(8πk)`, which is the cross-check
that the constant is right. G4 asserts this identity directly, so there is no
free constant to fit.

### 3.8 The two-cylinder addition theorem

Incident coefficients about centre `j`, unit-amplitude plane wave:

```
I_n^{(j)} = e^{i k R_j·k̂_i} · i^n · e^{−i n α_i}
```

Graf translation of cylinder `j`'s field re-expanded about centre `l`:

```
T^{(j→l)}[m, n] = H_{n−m}⁽¹⁾(k·d_jl) · e^{i(n−m)·φ_jl}
```

where `(d_jl, φ_jl)` are the polar coordinates of the vector **from centre `j` to
centre `l`**, i.e. of `R_l − R_j`. (Check it at `ρ_l → 0`: only `m = 0` survives
and `H_n(kd)e^{inφ_jl}` must reproduce the left-hand side, which fixes the
direction.) Validity needs `a_l < d_jl`, which non-overlap gives.

The coupled system, truncated at `|n| ≤ M`, is `J·(2M+1)` square:

```
A_m^{(l)} − s_m^{(l)} · Σ_{j≠l} Σ_n T^{(j→l)}[m, n] · A_n^{(j)}  =  s_m^{(l)} · I_m^{(l)}
```

**Truncation is bounded on both sides, and more orders is not safer.**
`H_n⁽¹⁾(kd)` grows superexponentially past `n ≈ kd` while `s_n` decays
superexponentially past `n ≈ x`, so a generously truncated system is
ill-conditioned and returns garbage. Measured (§9, `conv2.py`; two circles,
`x₁ = 1.79`, `k d = 9.63`), relative error against a converged BIE solution:

| `M` | 4 | 6 | 8 | **10** | **12** | **16** | **19** | **22** | 25 | 30 | 40 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| rel err | 1e-5 | 2.4e-9 | 2.7e-13 | **1.2e-15** | **1.4e-15** | **1.2e-15** | **3.0e-15** | **6.8e-15** | 2.3e-12 | 2.0e-10 | 3.2e-6 |
| `\|H_M(kd)\|` | 0.27 | 0.29 | 0.33 | 0.46 | 0.97 | 26 | 8.4e2 | 4.9e4 | 4.6e6 | 2.1e10 | 5.9e18 |

So `two_cylinder.py` picks `M` from the Wiscombe criterion on `max_j |x_j|` and
**raises** when that `M` would put `|H_M(k·d_min)|` above `1e6` — the last value
in the table that is still at round-off is `M = 22` at `4.9e4`, and the first
that is not is `M = 25` at `4.6e6`, so `1e6` sits between the two measured
points. Do not silently pad `M`.

---

## 4. The implementation

**Every code block below was executed verbatim before this document was
written** (§9, `spec_check.py` and `g1field.py`), not sketched: §4.8's module as
written reproduces the solver to 4.2e-15 at its *default* truncation, its
conditioning guard fires as intended, and §4.3's helper is **bit-identical** to
`eval_field` on both interior and exterior points. If your transcription does
not reproduce those, it is a transcription error. Transcribe, do not improvise.

Docstrings are abbreviated here to `"""…"""` where the content is obvious from
the spec text — write real Google-style ones with `Args`/`Returns`/`Raises`
(CLAUDE.md, *Style*).

### 4.1 `kernels.assemble_cross_block` — new public primitive

Append to `kernels.py`, after `assemble_matrix_dwn`. Raw arrays only, no
`Geometry` — that is this module's contract.

```python
def assemble_cross_block(
    wnum_bg: complex,
    nn_p: int,
    f_p: np.ndarray,
    g_p: np.ndarray,
    df_p: np.ndarray,
    dg_p: np.ndarray,
    f_q: np.ndarray,
    g_q: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """M1 and M2 coupling blocks between two non-overlapping boundaries.

    No Kress correction appears here: the two boundaries never touch, so the
    integrand has no singularity and the plain periodic trapezoid rule is
    already spectral on an analytic boundary (docs/design/multiparticle-spec.md
    §3.2). These are assemble_matrix's `single_bg` and `double_bg` with the
    circulant weight W_d set to zero.

    ``nn_p``, ``df_p`` and ``dg_p`` belong to the **source** particle: the
    integral is over p's boundary. Using the field particle's step is invisible
    at equal nn and a wrong answer the moment nn varies.
    """
    h_p = 2.0 * PI / nn_p
    dx = f_q[:, None] - f_p[None, :]
    dz = g_q[:, None] - g_p[None, :]
    z = wnum_bg * np.sqrt(dx**2 + dz**2)
    c = dx * dg_p[None, :] - dz * df_p[None, :]
    m1 = wnum_bg**2 * (0.25j * h_p * hank1(z) / z) * c
    m2 = 0.25j * h_p * hank0(z)
    return m1, m2
```

`hank0`/`hank1` already live in this module. **Do not call
`scipy.special.hankel1` directly** — that loses the Cephes fast path for real
`k`, which is 99 % of this package's runtime (conventions §6). Do **not** call
`_real_if_real` here; the caller does it once for the whole assembly.

### 4.2 `src/pysie2d/cluster.py` — `Cluster` and the geometric guards

New file. Module docstring states: the D4 layout with a pointer to conventions
§14; that `k_bg` is common to the cluster (D3b); and that overlap is excluded by
the formulation, not by an implementation limit.

```python
import warnings
from collections.abc import Sequence

import numpy as np

from .fields import _cross_sections, _far_field_at, _representation_at
from .geometry import Geometry
from .kernels import _real_if_real, assemble_cross_block, assemble_matrix
from .material import Material
from .solver import wavelength_over_ds
from .sources import _point_inside, line_dipole_rhs, plane_wave_rhs

PI = np.pi

# A resolution spread wider than this between the best- and worst-resolved
# particle warns. The coupled system is only as accurate as its worst block,
# and that is not visible from any one particle. Four is two halvings of nn,
# which in the M2 pilot is the difference between round-off and 1e-6 at a
# fixed gap (docs/design/studies/cluster-gap-envelope.md).
RESOLUTION_SPREAD_WARN = 4.0


class ClusterOverlapError(ValueError):
    """Two particle boundaries intersect; the formulation does not apply."""


class ClusterGapWarning(UserWarning):
    """A gap is too small for the quadrature resolution in use."""


class ClusterResolutionWarning(UserWarning):
    """One particle is far more coarsely resolved than its neighbours."""


class Cluster:
    """An arrangement of non-overlapping particle boundaries."""

    def __init__(self, geometries: Sequence[Geometry]) -> None:
        if len(geometries) == 0:
            raise ValueError("a Cluster needs at least one geometry")
        self.geometries = tuple(geometries)
        sizes = [2 * g.n_pts for g in self.geometries]
        self.offsets = tuple(np.cumsum([0] + sizes).tolist())
        self.n_dof = self.offsets[-1]
        self._check_overlap()
        self.min_gap = self._min_gap()

    def __len__(self) -> int:
        return len(self.geometries)

    def slice(self, p: int) -> slice:
        """The 2·nn_p slice of particle p (conventions §14)."""
        return slice(self.offsets[p], self.offsets[p + 1])
```

**Radii helpers.** `r_circ` is `max |r − (x0, z0)|` over the nodes — **not**
`Geometry.rad`, which on a rounded square is 2.29× smaller (the same trap
`ScatterResult.multipoles` guards against). `r_insc` is the `min`.

```python
    def _radii(self, p: int) -> tuple[float, float]:
        g = self.geometries[p]
        r = np.hypot(g.f - g.x0, g.g - g.z0)
        return float(r.min()), float(r.max())
```

**The three-band overlap test**, per pair. Band 1 accepts, band 2 raises, band 3
samples:

```python
    def _check_overlap(self) -> None:
        for p in range(len(self)):
            gp = self.geometries[p]
            rp_in, rp_out = self._radii(p)
            for q in range(p + 1, len(self)):
                gq = self.geometries[q]
                rq_in, rq_out = self._radii(q)
                d = float(np.hypot(gq.x0 - gp.x0, gq.z0 - gp.z0))
                if rp_out + rq_out < d:
                    continue                       # certainly disjoint
                if rp_in + rq_in > d:              # certainly overlapping
                    raise ClusterOverlapError(
                        f"particles {p} and {q} overlap: their inscribed "
                        f"circles ({rp_in:.4g} + {rq_in:.4g} nm) already "
                        f"exceed the centre separation {d:.4g} nm"
                    )
                hit = any(_point_inside(x, z, gp.f, gp.g)
                          for x, z in zip(gq.f, gq.g)) or \
                      any(_point_inside(x, z, gq.f, gq.g)
                          for x, z in zip(gp.f, gp.g))
                if hit:
                    raise ClusterOverlapError(
                        f"particles {p} and {q} overlap: a boundary node of "
                        f"one lies inside the other"
                    )
```

The docstring must state the residual failure mode rather than claim the test is
decisive: **two boundaries can cross in a lens-shaped sliver containing no node
of either, which band 3 misses.** The `min_gap` reported is then small and the
§4.9 envelope warning fires, which is the backstop.

```python
    def _min_gap(self) -> float:
        """Smallest node-to-node distance between any two particles (nm)."""
        best = np.inf
        for p in range(len(self)):
            gp = self.geometries[p]
            for q in range(p + 1, len(self)):
                gq = self.geometries[q]
                d = np.hypot(gq.f[:, None] - gp.f[None, :],
                             gq.g[:, None] - gp.g[None, :])
                best = min(best, float(d.min()))
        return best
```

`O((Σ nn)²)` — 9e6 distances at `Np = 10`, `nn = 300`, milliseconds, far below
one assembly. Computed once in `__init__`. It slightly **overestimates** the true
gap (the closest approach may fall between nodes); say so in the docstring.

### 4.3 `fields._representation_at` — new private helper

Append to `fields.py`. The representation integral at all `M` points for one
boundary at one wavenumber, **with no inside/outside classification** — that is
what lets the cluster sum it over boundaries.

```python
def _representation_at(
    ei_p: np.ndarray,
    nn: int,
    f: np.ndarray,
    df: np.ndarray,
    g: np.ndarray,
    dg: np.ndarray,
    delt: float,
    wnum: complex,
    x_pts: np.ndarray,
    z_pts: np.ndarray,
) -> np.ndarray:
    """BIE representation integral over one boundary, at every point.

    The vectorised twin of the body of :func:`eval_field`'s point loop, with
    the inside/outside test removed so a caller can sum it over several
    boundaries (docs/design/multiparticle-spec.md §3.5). ``eval_field`` is left
    as it is: its point loop keeps its memory at O(nn), and the two are pinned
    together by the Np = 1 cluster test.
    """
    xmf = x_pts[:, None] - f[None, :]
    zmg = z_pts[:, None] - g[None, :]
    arg1 = wnum * np.sqrt(xmf**2 + zmg**2)
    arg2 = -dg[None, :] * xmf + df[None, :] * zmg
    integrand = (
        wnum**2 * arg2 * hank1(arg1) / arg1 * ei_p[None, :nn]
        - hank0(arg1) * ei_p[None, nn:]
    )
    return (1j / 4.0) * np.sum(integrand * delt, axis=1)
```

`eval_field` is **not** refactored to use it (§1.2). The duplication is one
four-line formula and G1 pins it: if the two ever drift, a test fails.
**Measured: bit-identical** to `eval_field` at exterior and interior points
alike, which is why G1 asserts `np.array_equal` on the near field too.

### 4.4 `fields._cross_sections` — new private helper

```python
def _cross_sections(
    amp: np.ndarray, amp_fwd: complex, wnum_bg: complex
) -> dict[str, float]:
    """C_sca, C_ext, C_abs in nm from a far-field amplitude (§3.6)."""
    nff = len(amp)
    delthe = 2.0 * PI / (nff - 1.0)
    # amp spans [-pi, pi] inclusive, so index 0 and nff-1 are the same physical
    # direction; summing both double-counts it. Dropping the duplicate (not
    # halving both) is what makes C_sca independent of where that one grid
    # angle falls relative to the forward peak.
    c_sca = float(np.sum(np.abs(amp[:-1]) ** 2) / (8.0 * PI * wnum_bg) * delthe)
    c_ext = float(amp_fwd.imag / wnum_bg)
    return {"c_sca": c_sca, "c_ext": c_ext, "c_abs": c_ext - c_sca}
```

Shared by both result classes so the two paths cannot disagree.

### 4.5 `ScatterResult.cross_sections` — `solver.py`

```python
    def cross_sections(self, n_angles: int = 500) -> dict[str, float]:
        """Absolute scattering, extinction and absorption cross-sections (nm).

        The multiparticle observable of docs/design/multiparticle-spec.md §3.6,
        provided here too so the single- and multi-particle paths report the
        same quantity. Equal to :meth:`efficiencies` scaled by the geometric
        width ``2·rad`` — which is why this is the quantity to compare across
        shapes, since that normalisation is only approximate off a circle.

        Plane-wave excitation only: ``C_ext`` is defined against a
        unit-amplitude incident plane wave.
        """
        g = self.geometry
        wnum_bg = self.wnum_bg
        amp, _ = self.far_field(n_angles)
        forward = np.array([PI - np.deg2rad(self.angle)])
        amp_fwd = _far_field_at(
            forward, g.n_pts, wnum_bg, g.f, g.g, g.df, g.dg, g.delt, self.ei
        )[0]
        return _cross_sections(amp, amp_fwd, wnum_bg)
```

`efficiencies()` is left exactly as it is; §6.6 asserts the two agree, which pins
them together without editing the older one.

### 4.6 `ClusterBIESolver` — `cluster.py`

```python
class ClusterBIESolver:
    """Coupled BIE solver for a finite cluster of particles."""

    def __init__(
        self,
        cluster: Cluster,
        materials: Sequence[Material],
        pol: int = 2,
    ) -> None:
        if len(materials) != len(cluster):
            raise ValueError(
                f"{len(materials)} materials for {len(cluster)} particles"
            )
        for i, mat in enumerate(materials):
            if mat.pol != pol:
                raise ValueError(
                    f"material {i} has pol = {mat.pol}, but the cluster is "
                    f"being solved at pol = {pol}. Polarisation is a property "
                    f"of the problem, not of a particle (conventions §1)."
                )
            if mat.n_clad != materials[0].n_clad:
                raise ValueError(
                    f"material {i} has n_clad = {mat.n_clad}, material 0 has "
                    f"{materials[0].n_clad}. A cluster has one background, and "
                    f"k_bg = 2π·n_clad/λ_vac must be unambiguous. Per-particle "
                    f"optical properties go in n_core and epsi, which enter as "
                    f"the background-relative nc and eps (conventions §2)."
                )
        self.cluster = cluster
        self.materials = tuple(materials)
        self.pol = pol
```

**The assembly.** The task list is the D8 seam:

```python
    def _assemble(self, wavelength: float | complex) -> np.ndarray:
        cl = self.cluster
        k_bg = _real_if_real(self.materials[0].wnum_bg(wavelength))
        me = np.zeros((cl.n_dof, cl.n_dof), dtype=complex)

        # Each task writes a disjoint block, so this loop is embarrassingly
        # parallel, and hank0/hank1 both release the GIL — threading it is the
        # proven 5.02x pattern of contour_moments (docs/design/performance.md
        # §3.1). Serial until M1 says the ceiling demands otherwise (D8).
        tasks = [("self", p, p) for p in range(len(cl))]
        tasks += [("cross", q, p) for q in range(len(cl))
                  for p in range(len(cl)) if q != p]

        for kind, q, p in tasks:
            gp = cl.geometries[p]
            if kind == "self":
                mat = self.materials[p]
                me[cl.slice(p), cl.slice(p)] = assemble_matrix(
                    self.pol, gp.n_pts, gp.f, gp.g, gp.df, gp.dg,
                    gp.ddf, gp.ddg, k_bg, mat.nc, mat.eps,
                )
                continue
            gq = cl.geometries[q]
            m1, m2 = assemble_cross_block(
                k_bg, gp.n_pts, gp.f, gp.g, gp.df, gp.dg, gq.f, gq.g
            )
            r0 = cl.offsets[q]
            c0 = cl.offsets[p]
            me[r0:r0 + gq.n_pts, c0:c0 + gp.n_pts] = m1
            me[r0:r0 + gq.n_pts, c0 + gp.n_pts:c0 + 2 * gp.n_pts] = m2
        return me
```

Note the lower half of each cross-block is left at its initialised zero — that
is §3.1's M3/M4 result, and it must be a comment in the code, not an accident.

**The solve**, with the two D6/D7 checks:

```python
    def scatter(self, wavelength: float, angle: float = 0.0
                ) -> "ClusterScatterResult":
        self._check_resolution(wavelength)
        self._check_gap(wavelength)
        cl = self.cluster
        k_bg = self.materials[0].wnum_bg(wavelength)
        rhs = np.zeros(cl.n_dof, dtype=complex)
        for p, gp in enumerate(cl.geometries):
            rhs[cl.slice(p)] = plane_wave_rhs(gp.n_pts, angle, k_bg, gp.f, gp.g)
        ei = np.linalg.solve(self._assemble(wavelength), rhs)
        return ClusterScatterResult(ei, cl, self.materials, self.pol,
                                    wavelength, angle, "plane_wave")

    def scatter_dipole(self, wavelength: float, x_s: float, z_s: float
                       ) -> "ClusterScatterResult":
        self._check_resolution(wavelength)
        self._check_gap(wavelength)
        cl = self.cluster
        k_bg = self.materials[0].wnum_bg(wavelength)
        rhs = np.zeros(cl.n_dof, dtype=complex)
        for p, gp in enumerate(cl.geometries):
            # line_dipole_rhs's own guards run once per particle, which is
            # exactly the cluster condition: the source must lie outside every
            # particle. No extra check is needed and none should be added.
            rhs[cl.slice(p)] = line_dipole_rhs(
                gp.n_pts, k_bg, gp.f, gp.g, x_s, z_s
            )
        ei = np.linalg.solve(self._assemble(wavelength), rhs)
        return ClusterScatterResult(ei, cl, self.materials, self.pol,
                                    wavelength, 0.0, "dipole")
```

`_check_resolution` computes `wavelength_over_ds` per particle and warns when
`max/min > RESOLUTION_SPREAD_WARN`, naming the worst particle and both values.
`_check_gap` is written in commit 10, after M2 — §4.9.

### 4.7 `ClusterScatterResult` — `cluster.py`

```python
class ClusterScatterResult:
    """Result of a single-wavelength coupled cluster solve.

    There is deliberately no ``efficiencies()``: it normalises by ``2·rad``,
    and a cluster has no ``rad`` (D5). There is deliberately no
    ``multipoles()``: the v0.7 expansion is about one particle centre and a
    cluster needs an output-side translation theorem that ``multipole.py``
    does not implement (§1.3). Both absences are decisions, not omissions.
    """

    def __init__(self, ei, cluster, materials, pol, wavelength, angle,
                 excitation) -> None:
        self.ei = ei
        self.cluster = cluster
        self.materials = tuple(materials)
        self.pol = pol
        self.wavelength = wavelength
        self.angle = angle
        self._excitation = excitation

    @property
    def wnum_bg(self) -> complex:
        return self.materials[0].wnum_bg(self.wavelength)

    def ei_particle(self, p: int) -> np.ndarray:
        """Particle p's own (2·nn_p,) sub-vector, in conventions §4 order."""
        return self.ei[self.cluster.slice(p)]

    def far_field(self, n_angles: int = 3000):
        angles = -PI + np.arange(n_angles) * 2.0 * PI / (n_angles - 1.0)
        return self._amp_at(angles), angles

    def _amp_at(self, angles: np.ndarray) -> np.ndarray:
        # Absolute coordinates already carry each particle's geometric phase;
        # adding one here would double it (§3.4).
        k = self.wnum_bg
        total = np.zeros(len(angles), dtype=complex)
        for p, gp in enumerate(self.cluster.geometries):
            total += _far_field_at(angles, gp.n_pts, k, gp.f, gp.g,
                                   gp.df, gp.dg, gp.delt, self.ei_particle(p))
        return total

    def cross_sections(self, n_angles: int = 500) -> dict[str, float]:
        if self._excitation != "plane_wave":
            raise ValueError(
                "cross_sections() is defined against a unit-amplitude "
                "incident plane wave; this result came from "
                f"{self._excitation} excitation"
            )
        amp, _ = self.far_field(n_angles)
        amp_fwd = self._amp_at(np.array([PI - np.deg2rad(self.angle)]))[0]
        return _cross_sections(amp, amp_fwd, self.wnum_bg)

    def eval_field(self, x, z) -> np.ndarray:
        x = np.asarray(x, dtype=float).ravel()
        z = np.asarray(z, dtype=float).ravel()
        k = _real_if_real(self.wnum_bg)
        inside = np.full(len(x), -1, dtype=int)
        for p, gp in enumerate(self.cluster.geometries):
            for j in range(len(x)):
                if inside[j] < 0 and _point_inside(x[j], z[j], gp.f, gp.g):
                    inside[j] = p
        field = np.zeros(len(x), dtype=complex)
        out = inside < 0
        if out.any():
            for p, gp in enumerate(self.cluster.geometries):
                field[out] += _representation_at(
                    self.ei_particle(p), gp.n_pts, gp.f, gp.df, gp.g, gp.dg,
                    gp.delt, k, x[out], z[out],
                )
        for p, gp in enumerate(self.cluster.geometries):
            sel = inside == p
            if sel.any():
                k_core = _real_if_real(self.materials[p].nc * self.wnum_bg)
                field[sel] = _representation_at(
                    self.ei_particle(p), gp.n_pts, gp.f, gp.df, gp.g, gp.dg,
                    gp.delt, k_core, x[sel], z[sel],
                )
        return field

    def resolution(self) -> tuple[float, ...]:
        """Points per interior wavelength, per particle (D7)."""
        return tuple(
            wavelength_over_ds(g, m, self.wavelength)
            for g, m in zip(self.cluster.geometries, self.materials)
        )
```

### 4.8 `src/pysie2d/reference/two_cylinder.py` — new file

Implements §3.7–§3.8. This code has been run and reproduces the solver to
1.4e-15 (§6.4). Written for `J` cylinders, not hard-coded to two: the algebra is
identical and the loop bounds are the only difference, so `J = 3` costs nothing
and is what G6 will want.

```python
"""Analytic multiple scattering from parallel circular cylinders.

The closed-form anchor for the coupled BIE: each cylinder's single-cylinder Mie
coefficients (:mod:`pysie2d.reference.mie`) are coupled by Graf's addition
theorem, giving an independent answer for a cluster of circles — not a second
path through this repository (CLAUDE.md non-negotiable 3).

**Conventions.** The algebra here runs in the *standard* polar convention
``x = ρ cos φ``, ``z = ρ sin φ``, converting to the solver's θ-from-+z
convention only at the two boundaries (``angle`` in and ``theta`` in). Forcing
the solver's convention through the translation algebra buys nothing and risks
a silent rotation by ``e^{imπ/2}``. The bridge is
docs/design/multiparticle-spec.md §3.7, verified to 1.8e-15:

    φ   = π/2 − θ                 (observation)
    α_i = deg2rad(angle) − π/2    (incidence)
    s_n = −c_n                    (c_n = mie.an for TM, mie.bn for TE)
    amp_solver(θ) = −4i · S(π/2 − θ)
"""

import numpy as np
from scipy.special import hankel1

from .mie import _nmax, an, bn

PI = np.pi

# Above this the Graf translation matrix is ill-conditioned: H_M(k·d) grows
# superexponentially past M ~ k·d while s_n decays superexponentially past
# n ~ x, so a generously truncated system returns garbage rather than a more
# accurate answer. Measured (spec §3.8): the error is at round-off through
# |H_M(kd)| = 4.9e4 (M = 22) and has degraded to 2.3e-12 by 4.6e6 (M = 25).
CONDITION_LIMIT = 1.0e6


def scattering_amplitude(
    pol: int,
    radii,
    centres,
    m_rel,
    wnum_bg: complex,
    angle: float,
    theta: np.ndarray,
    n_max: int | None = None,
) -> np.ndarray:
    """Far-field amplitude of a cluster of circles, in the solver's units.

    Args:
        pol: 1 = TM (a_n), 2 = TE (b_n), matching the solver's codes.
        radii: (J,) cylinder radii (nm).
        centres: (J, 2) centres as (x, z) in nm — the solver's coordinates.
        m_rel: (J,) relative refractive indices (``Material.nc``).
        wnum_bg: Background wavenumber k_bg = 2π·n_clad/λ_vac (rad/nm). This is
            a primitive and takes no wavelength (conventions §2). May be
            complex.
        angle: Plane-wave incidence angle (degrees), the solver's ``angle``.
        theta: (nff,) observation angles (rad), the solver's far-field angles.
        n_max: Truncation order M. Default: the Wiscombe criterion on the
            largest size parameter.

    Returns:
        complex (nff,) amplitude, directly comparable to
        ``ClusterScatterResult.far_field``.

    Raises:
        ValueError: If the truncation order is too high for the closest centre
            separation; see ``CONDITION_LIMIT``.
    """
    k = wnum_bg
    R = np.asarray(centres, dtype=float)
    radii = np.asarray(radii, dtype=float)
    n_cyl = len(radii)
    alpha_i = np.deg2rad(angle) - PI / 2.0
    khat = np.array([np.cos(alpha_i), np.sin(alpha_i)])

    x = k * radii
    if n_max is None:
        n_max = _nmax(np.max(np.abs(x)))
    n = np.arange(-n_max, n_max + 1)
    nd = len(n)

    d_min = min(
        float(np.hypot(*(R[j] - R[i])))
        for i in range(n_cyl) for j in range(i + 1, n_cyl)
    )
    h_max = float(np.abs(hankel1(n_max, np.abs(k) * d_min)))
    if h_max > CONDITION_LIMIT:
        raise ValueError(
            f"truncation order {n_max} gives |H_M(k·d_min)| = {h_max:.3g}, "
            f"above the conditioning limit {CONDITION_LIMIT:.0e}: these "
            f"cylinders are too close for the addition theorem at the order "
            f"their size parameters demand"
        )

    # s_n = -c_n, forced by unitarity against mie.efficiencies (§3.7).
    coeff = an if pol == 1 else bn
    s = [-coeff(np.abs(n), x[j], m_rel[j]) for j in range(n_cyl)]
    inc = [
        np.exp(1j * k * (R[j] @ khat)) * (1j**n) * np.exp(-1j * n * alpha_i)
        for j in range(n_cyl)
    ]

    mat = np.zeros((n_cyl * nd, n_cyl * nd), dtype=complex)
    rhs = np.zeros(n_cyl * nd, dtype=complex)
    for l in range(n_cyl):
        blk = slice(l * nd, (l + 1) * nd)
        mat[blk, blk] += np.eye(nd)
        rhs[blk] = s[l] * inc[l]
        for j in range(n_cyl):
            if j == l:
                continue
            # (d, φ) are the polar coordinates of the vector FROM centre j TO
            # centre l. Check it at ρ_l → 0: only m = 0 survives, and
            # H_n(kd)·e^{inφ} must reproduce the left-hand side (§3.8).
            dvec = R[l] - R[j]
            d = float(np.hypot(*dvec))
            phi_jl = float(np.arctan2(dvec[1], dvec[0]))
            nm = n[None, :] - n[:, None]        # nm[m, n] = n − m
            t = hankel1(nm, k * d) * np.exp(1j * nm * phi_jl)
            mat[blk, j * nd:(j + 1) * nd] -= s[l][:, None] * t
    amp_coef = np.linalg.solve(mat, rhs)

    phi = PI / 2.0 - np.asarray(theta, dtype=float)
    rhat = np.stack([np.cos(phi), np.sin(phi)])
    total = np.zeros(len(phi), dtype=complex)
    for j in range(n_cyl):
        a_j = amp_coef[j * nd:(j + 1) * nd]
        phase = np.exp(-1j * k * (R[j][:, None] * rhat).sum(0))
        total += phase * (
            a_j[:, None] * ((-1j) ** n)[:, None]
            * np.exp(1j * n[:, None] * phi[None, :])
        ).sum(0)
    return -4j * total
```

Add to `[tool.ruff.lint.per-file-ignores]` in `pyproject.toml`, extending the
existing `mie.py` entry rather than renaming the physics:

```toml
"src/pysie2d/reference/two_cylinder.py" = ["N806"]
```

### 4.9 The envelope guard — commit 10, after M2

```python
    def _check_gap(self, wavelength: float) -> None:
        cl = self.cluster
        if len(cl) < 2:
            return
        gap = cl.min_gap
        a = max(float(np.hypot(g.f - g.x0, g.g - g.z0).max())
                for g in cl.geometries)
        nn_needed = GAP_ENVELOPE_C * np.sqrt(a / gap)
        nn_worst = min(g.n_pts for g in cl.geometries)
        if nn_worst >= nn_needed:
            return
        circular = all(g.is_circle for g in cl.geometries)
        detail = "" if circular else (
            " This envelope was measured on circles and is not validated for "
            "non-circular facing boundaries (spec D9); treat it as a "
            "diagnostic, not a bound."
        )
        warnings.warn(
            f"minimum gap {gap:.4g} nm needs about n_pts = {nn_needed:.0f} "
            f"per particle at this configuration, but the coarsest particle "
            f"has {nn_worst}. The cross-block quadrature degrades silently "
            f"below the envelope.{detail}",
            ClusterGapWarning,
            stacklevel=3,
        )
```

`GAP_ENVELOPE_C` is the M2 deliverable, defined beside `RESOLUTION_SPREAD_WARN`
with the measurement in its comment, in the style of
`multipole.SPECTRUM_TAIL_WARN`:

```python
# Nodes needed before the near-gap integrand's analyticity strip stops being
# resolved. For a circle of radius a, a near-singularity at distance `gap`
# sits at conformal half-width ~ sqrt(2·gap/a), so the trapezoid rule's rate
# exp(-nn·sqrt(2·gap/a)) fixes the form nn >= C·sqrt(a/gap). Measured in
# docs/design/studies/cluster-gap-envelope.md; the M2 pilot gave
# nn·sqrt(gap/a) in 39-57 over four decades of gap at x = 1.79.
GAP_ENVELOPE_C = <the value M2 measures>
```

### 4.10 Exports and conventions

`__init__.py` gains `Cluster`, `ClusterBIESolver`, `ClusterScatterResult`,
`ClusterOverlapError`, `ClusterGapWarning`, `ClusterResolutionWarning`,
`assemble_cross_block` — in `__all__` **and** in the module docstring's API
list. `reference.two_cylinder` is **not** exported at top level; neither is
`reference.mie`.

Regenerate `tests/api_baseline.txt` per its own recipe test. **Additions only.**
If a line is removed or changed, stop — this work is additive and a removal means
something in §1.2 was touched.

`docs/conventions.md` gains **§14, the cluster degree-of-freedom layout**, in the
same commit as §4.6 (a new convention is recorded in the change that pins it,
CLAUDE.md *Read first*). It must state: the layout; that `o_p` is cumulative over
`2·nn_q`; that each particle's slice is directly consumable by the
single-particle primitives; and — explicitly — that it **does not** extend the
single-particle layout of §4, so nobody later "fixes" it into the legacy form.

---

## 5. The measurement steps

Two numbers the spec deliberately does not contain. Pilot data is given for both
so you know what a right answer looks like.

### M1 — the `Np` ceiling (commit 9)

Handoff §8's third open question. Sweep `Np` at fixed `nn` and `nn` at fixed
`Np`, recording assembly time, solve time, peak matrix memory, and the assembly
fraction. Report the `Np` at which the dense solve stops being the ~2 % of
runtime that performance.md records for one particle — that is where D8's
"assembly dominates" premise expires.

Pilot, already run (§9, `g2np.py`; identical circles, `rad` 200 nm, `nn` 200,
pitch 1400 nm, TE, λ = 633 nm):

| `Np` | `N` | assemble | solve | matrix | assembly % |
|---|---|---|---|---|---|
| 2 | 800 | 0.043 s | 0.007 s | 9.8 MiB | 85.5 % |
| 3 | 1200 | 0.115 s | 0.024 s | 22.0 MiB | 82.7 % |
| 5 | 2000 | 0.329 s | 0.100 s | 61.0 MiB | 76.7 % |
| 8 | 3200 | 0.825 s | 0.346 s | 156.2 MiB | 70.4 % |
| 12 | 4800 | 1.779 s | 1.005 s | 351.6 MiB | 63.9 % |
| 16 | 6400 | 3.061 s | 2.413 s | 625.0 MiB | 55.9 % |

**Read this pilot with one correction, and state it in your write-up.** The
prototype that produced it called `scipy.special.hankel1` directly in the
cross-blocks rather than `kernels.hank0`/`hank1`, so for a real `k` it took the
Amos path where the shipped code takes Cephes — roughly 11× faster on that part
(conventions §6). **The shipped assembly is therefore substantially cheaper than
this table shows, and the solve overtakes it at a smaller `Np` than 16.** Re-run
the sweep against the real `assemble_cross_block`; that corrected crossover is
the number M1 delivers.

Deliverable: a section in [performance.md](performance.md) with the corrected
table, the machine (**4 performance cores** — size any future threading to
those, not to the logical core count), and an explicit statement of the `Np`
above which dense is the wrong tool. Do **not** implement an iterative solver,
FMM or block preconditioner: all three are speculative until this measurement
says dense is not enough (handoff §4.7).

### M2 — the near-gap envelope (commits 10 and 11)

Handoff §5's G3, and the one measurement this feature cannot ship without.

**The method matters, and the obvious method is wrong.** Do **not** measure the
envelope against the addition theorem: it degrades as the gap closes too, at a
different rate, so it reports a false floor and hides the real degradation.
Measured (§9, `g3.py`): at gap = 0.016 λ the BIE at `nn = 240` is converged to
2.1e-15, while judging that same solution against the addition theorem reads
5.2e-9 — there, the reference is the limit, not the solver.

Measure **BIE self-convergence**: for each gap, take a high-`nn` solution as
truth and find the smallest `nn` reaching a stated accuracy.

Pilot, already run (TE, two circles `rad` 180 and 110 nm, λ = 633 nm,
`angle = 37°`, truth at `nn = 1024`, max relative far-field error over 361
angles):

| gap/λ | `nn` = 30 | 60 | 120 | 240 |
|---|---|---|---|---|
| 1.438 | 1.6e-15 | 1.7e-15 | 1.4e-15 | 1.5e-15 |
| 0.490 | 3.1e-15 | 1.7e-15 | 2.2e-15 | 1.7e-15 |
| 0.237 | 1.3e-11 | 1.6e-15 | 2.0e-15 | 2.7e-15 |
| 0.142 | 1.2e-09 | 1.8e-15 | 2.2e-15 | 2.3e-15 |
| 0.063 | 1.0e-06 | 4.8e-11 | 2.5e-15 | 1.8e-15 |
| 0.032 | 5.2e-05 | 8.9e-08 | 7.1e-13 | 2.1e-15 |
| 0.016 | 7.4e-04 | 5.2e-06 | 2.6e-09 | 2.1e-15 |
| 0.009 | 3.4e-03 | 5.3e-05 | 3.4e-07 | 3.6e-11 |

Each doubling of `nn` buys roughly a quartering of the sustainable gap — the
`nn ∝ (gap/a)^{−1/2}` law the analyticity-strip argument predicts (§4.9).
`nn·√(gap/a)` over the pilot's four decades lands in **39–57**, so
`GAP_ENVELOPE_C` will come out in that region; a fitted value far outside it
means the sweep is wrong, not the law.

**What the pilot does not fix, and you must:** it is one `x` (1.79), one λ, one
polarisation, one incidence. Extend over `x` and both polarisations before fixing
`GAP_ENVELOPE_C`. The `a` in `√(a/gap)` is the *larger* of the two radii here,
and a two-radius sweep is what decides whether that is the right choice — §11
records it as open.

Deliverable: `docs/design/studies/cluster-gap-envelope.md` with the sweep, the
fitted constant, and the rule as implemented — in the style of the `eval_field`
near-boundary rule and, unlike it, actually measured. Per D9 the rule is
circle-measured; the study must say so and the warning repeats it for
non-circular pairs.

---

## 6. Tests

`tests/test_cluster.py` unless stated. Names and comments state what physical
property is checked and why it cannot pass by accident (CLAUDE.md, *Style*).

### 6.1 G1 — `test_one_particle_cluster_is_bit_identical_to_BIESolver`

**Measured: exactly bit-identical**, matrix and solution vector both. Assert with
`np.array_equal`, not `allclose`.

```python
geom = Geometry.gielis(rad=200.0, n_pts=180, m=5, n1=4.0, x0=13.0, z0=-7.0)
mat = Material(n_core=2.1, n_clad=1.3, pol=1, epsi=0.4)
single = BIESolver(geom, mat).scatter(wavelength=700.0, angle=23.0)
cl = ClusterBIESolver(Cluster([geom]), [mat], pol=1)
multi = cl.scatter(wavelength=700.0, angle=23.0)
assert np.array_equal(cl._assemble(700.0), BIESolver(geom, mat).assemble(700.0))
assert np.array_equal(multi.ei, single.ei)
assert np.array_equal(multi.far_field(401)[0], single.far_field(401)[0])
```

The awkward particle is deliberate: off-centre, `n_clad ≠ 1`, lossy, `pol = 1`,
oblique. `eval_field` is **also** bit-identical — measured across a mixed set of
interior and exterior points — so assert it with `np.array_equal` as well. It is
a different code path (§4.3), and this is what pins the two against drift.

Run it on a circle and on a star, in both polarisations (handoff §5).

Bit-identity is available **only** because of D4. If it degrades to "close", the
DOF layout has been changed and conventions §14 is no longer true — that, not
the number, is what this test is really guarding.

### 6.2 G2 — `test_coupling_decays_at_the_two_dimensional_Green_function_rate`

The far-separated limit, asserted as a **rate**, not a tolerance at one gap. The
first-order multiple-scattering correction is proportional to the other
particle's field at this one, which in 2-D decays as `H₀⁽¹⁾(kd) ~ d^{−1/2}`.

The "independent" comparison is `Σ_p amp_p` from separate `BIESolver` runs on the
same `Geometry` objects, with **no phase factor applied** — absolute coordinates
already carry it (§3.4). A test that needs a hand-added phase has a bug.

Measured (§9, `g2np.py`; the §6.4 dimer, `nn = 200`, TE, λ = 633 nm, gaps from
2.7 λ to 75 λ, relative deviation from 8.8e-2 down to 2.0e-2):

```
log-log slope over the full range   −0.4478
sliding 4-point windows             −0.427, −0.430, −0.393, −0.444, −0.506, −0.521
```

The rate approaches `−1/2` from above as the gap grows, which is the expected
behaviour: the higher-order terms that survive at short range are what bend the
slope. **Fit over the four widest gaps and assert the slope in
`(−0.60, −0.40)`.** That band brackets the measured `−0.52` with headroom on
both sides while still excluding `−1` and `0`, which is the point — a wrong
cross-block normalisation shows up as a wrong power, not a wrong constant.

### 6.3 G3 — `test_gap_below_the_envelope_warns`

The envelope *study* is M2; what is tested here is the guard. Three cases:
a configuration comfortably inside the envelope is silent
(`warnings.catch_warnings()` with `simplefilter("error")`); one below it raises
`ClusterGapWarning` whose message names a sufficient `n_pts`; and a non-circular
pair below it warns with the "not validated" wording of D9. Assert on the
message text, so a guard firing for the wrong reason fails.

### 6.4 G4 — `tests/test_two_cylinder.py`

Assert `amp_cluster(θ) == two_cylinder.scattering_amplitude(...)` pointwise
(§3.7), on a **deliberately asymmetric** dimer — radii 180 / 110 nm, indices
2.0 / 1.6, centres at `x = ∓450` and `+520` nm, λ = 633 nm, `angle = 37°` — in
**both** polarisations. The asymmetry is what makes a mirror flip `θ → −θ`
detectable; a symmetric dimer at normal incidence would pass with the angle
mapping reversed.

Measured, reference at `M = 12`, max relative error over 361 angles:

| `nn` | 20 | 25 | **30** | 40 | 60 | 300 |
|---|---|---|---|---|---|---|
| TE | 1.9e-08 | 1.3e-11 | **1.7e-15** | 1.1e-15 | 1.3e-15 | 1.4e-15 |
| TM | 1.0e-08 | 9.1e-12 | **1.5e-15** | 1.6e-15 | 1.3e-15 | 1.6e-15 |

Spectral, at round-off by `nn = 30` — exactly the circle-anchor behaviour
CLAUDE.md describes.

That table pins `M = 12`. The shipped module of §4.8 defaults `n_max` to the
Wiscombe order, **19** for this dimer, whose floor is slightly higher: rerunning
the module exactly as written gives **2.5e-15 (TE) to 4.2e-15 (TM)** at
`nn = 60`, across `angle` 0° and 37°. **Run the test at `nn = 60` with
`rtol = 2e-14`** — the measured 4.2e-15 floor with ~5× headroom, cited as a
measured value against a round-off floor and not as a convergence order
(non-negotiable 4). Do not pass an explicit `n_max` to buy back the last factor
of three; the default is what users get and the default is what must be tested.

Add `test_addition_theorem_truncation_has_a_conditioning_window`: assert the
error is flat for `M` in 10…22 and **worse** at `M = 40` (measured 3.2e-6). That
is what stops a later reader from "improving" the truncation, and it is the only
thing that would catch it. The `CONDITION_LIMIT` guard itself is verified to
fire: at `M = 60` on a 300 nm centre separation it reports
`|H_M(k·d_min)| = 1.95e+69` and raises.

### 6.5 G5 — `test_far_field_reciprocity_on_an_asymmetric_cluster`

Colton & Kress Thm 3.13, `u_∞(x̂, d̂) = u_∞(−d̂, −x̂)`. In the solver's angles,
with incidence `α` and observation `θ` both in degrees:

```
amp(θ = θ₁ ; angle = α₁)  ==  amp(θ = −α₁ ; angle = −θ₁)
```

**Measured: 8.8e-16** at `α₁ = 20°`, `θ₁ = 75°`, `nn = 300`, on the §6.4 dimer
with one particle made lossy (`epsi = 0.3`). Assert `rtol = 5e-15` — a round-off
floor with headroom.

This is a property of the operator, holds for any number of particles of any
shape, and is sensitive to exactly the block-transposition and index errors G1
cannot see on a symmetric configuration. **The cluster must be asymmetric or the
test is vacuous** — that is the sentence to put in the test's docstring.

### 6.6 `test_cross_sections_and_efficiencies_agree` — `tests/test_efficiencies.py`

`ScatterResult.cross_sections()` equals `efficiencies()` scaled by `2·rad` to
round-off. Pins D5's two entry points together without editing the older one.

### 6.7 Guards — one test per guard

`pytest.raises` / `pytest.warns`, each asserting on the message: band-2 overlap;
band-3 overlap; band-1 asserted **silent**; mismatched `pol`; mismatched
`n_clad`; `len(materials) != len(cluster)`; empty cluster; `cross_sections()`
after `scatter_dipole`; a dipole source inside one particle of a cluster;
`two_cylinder` above `CONDITION_LIMIT`; the resolution-imbalance warning.

### 6.8 `test_complex_wavelength_reaches_the_cluster_assembly`

Non-negotiable 1. `_assemble(600 - 20j)` returns a finite complex matrix of the
right shape. No cluster QNM ships, but the path must not be quietly real-only —
that is exactly how the complex branch dies.

### 6.9 `tests/test_public_api.py`

Regenerated `api_baseline.txt`, additions only (§4.10).

---

## 7. Commits

Conventional-commit form, in this order. **Merge, do not squash** — the changelog
reads the individual subjects.

1. `docs: renumber multiparticle to v0.8 and external validation to v0.9`
2. `feat: absolute cross-sections on ScatterResult`
3. `feat: Cluster container with overlap guards`
4. `feat: coupled multiparticle BIE assembly and solve` — `assemble_cross_block`,
   `ClusterBIESolver`, `ClusterScatterResult`, the exports, and **conventions
   §14**
5. `test: bit-identical Np=1 reduction and far-field reciprocity` (G1, G5, G8)
6. `feat: two-cylinder addition-theorem reference anchor`
7. `test: two-cylinder anchor for the coupled far field` (G4)
8. `test: coupling decays at the 2-D Green function rate` (G2)
9. `docs: Np ceiling of the dense cluster solve` (M1 → performance.md)
10. `feat: warn when a cluster gap is below the quadrature envelope` (M2's
    constant and its guard land with the measurement that fixes them; G3)
11. `docs: near-gap validity envelope study` (M2 → `docs/design/studies/`)
12. `docs: document the cluster API in the README`

### 7.1 The renumber — do it once, completely

Commit 1 only. `v0.8` currently means external validation; after this it means
multiparticle.

- `git mv docs/design/v0.8-external-validation.md docs/design/v0.9-external-validation.md`,
  then fix its title, its §7 heading, and its two other internal `v0.8` mentions.
- `docs/design/README.md`: the table row's link and text, plus the renumber note
  — it already records the 20 Sep 2026 renumber from v0.7; **add** this one
  rather than replacing it. Add a row for this spec.
- `CLAUDE.md` *Roadmap*: mark v0.7 multipole shipped; add a **v0.8
  multiparticle** entry pointing at this spec and the handoff; renumber the
  external-validation entry to v0.9 and fix its link; and rewrite the closing
  paragraph, which currently says multiparticle gates v1.0 alongside v0.8 —
  under D1 it is additive and no longer does. Say that explicitly.
- `docs/design/multiparticle-handoff.md`: four links to
  `v0.8-external-validation.md` (§4.6, §5 G6, §6, §6 closing). Fix the links;
  **do not** rewrite its prose — it is a historical guide, and §2 above records
  where this spec supersedes it.
- `docs/design/multipole-spec.md` §0.1 and §6.1, and `tests/test_multipole.py:152`
  mention v0.8 as the external milestone. Update all three.

No `feat:`/`fix:` in this commit, so it publishes nothing on its own.

---

## 8. Acceptance checklist

- [ ] `uv run pytest` green, no new warnings in the summary.
- [ ] `uv run ruff check` and `uv run ruff format --check` clean.
- [ ] G1 asserts `np.array_equal` — matrix, solution vector, far field **and**
      near field — and passes. Bit-identical, not close.
- [ ] G4 passes at `nn = 60`, `rtol = 2e-14`, **both** polarisations, on the
      asymmetric dimer at oblique incidence, with the **default** `n_max`.
- [ ] G4's truncation-window test passes: flat for `M` in 10…22, worse at 40.
- [ ] G5 passes at `rtol = 5e-15` on an asymmetric, partly lossy cluster.
- [ ] G2's fitted slope lands in `(−0.60, −0.40)`.
- [ ] Every `rtol`/`atol` added carries a comment with the measured number or
      floor that justifies it.
- [ ] A complex `wavelength` reaches `_assemble` and returns a finite complex
      matrix (§6.8).
- [ ] `docs/conventions.md` §14 exists and says the layout does **not** extend §4.
- [ ] `api_baseline.txt` regenerated with **added lines only**.
- [ ] M1's corrected table is in performance.md — corrected for the Cephes fast
      path, per §5.1.
- [ ] M2's study is in `docs/design/studies/`, and `GAP_ENVELOPE_C` cites it in
      its comment and lands in 39–57.
- [ ] `grep -rn "v0\.8" --include="*.md" --include="*.py" .` shows no stale
      external-validation references (§7.1).
- [ ] README documents the cluster API and states the near-gap envelope.
- [ ] `uv.lock` untouched — no runtime dependency was added. If you think one is
      needed, **stop and raise it**; it is a scope decision (CLAUDE.md).

---

## 9. Reproducing the measurements

Every number came from four scripts run against `9d6e288` with pysie2d 0.6.0.
They are short enough to rewrite from §3 and §4; what follows is what each
established, so you can tell a transcription error from a real discrepancy.

- **`bridge.py`** — single cylinder, `rad` 200 nm, λ = 633 nm, `n_core = 2`,
  `nn = 400`. Asserts `amp_solver == −4i·S` with `s_n = −c_n`, `φ = π/2 − θ`,
  `α_i = α − π/2`. Result **1.2e-15 … 1.8e-15**, both polarisations, `angle` 0°
  and 37°; ratio mean `1.000000 + 0.000000j`, std 2.8e-15. This is all of §3.7.
- **`g4.py`** — the coupled dimer of §3.1–§3.4 against §3.8. At `nn = 500`,
  `M = 30`: 2.0e-10 (TE) / 2.9e-09 (TM) — **the reference's truncation, not the
  solver's error**, which `conv2.py` then showed.
- **`conv2.py`** — the `nn` table of §6.4 (round-off at `nn = 30`) and the `M`
  window of §3.8 (flat 10…22, degrading outside).
- **`g3.py`** — the §5.2 pilot: BIE self-convergence against `nn = 1024`
  alongside the same runs judged against the addition theorem, which is what
  establishes M2's method.
- **`g2np.py`** — the §6.2 decay slopes and the §5.1 `Np` pilot.
- **`spec_check.py`** — §4.8's module and §4.3's helper transcribed *verbatim
  from this document* and re-run: 2.5e-15 … 4.2e-15 against the coupled BIE at
  `nn = 60` with the default `n_max`, the conditioning guard raising at
  `|H_M| = 1.95e+69`, and the helper bit-identical to `eval_field`.
- **`g1field.py`** — the full §4.7 `eval_field` path (classification, masking,
  accumulation) bit-identical to `ScatterResult.eval_field` at `Np = 1` over a
  mixed interior/exterior point set. This is what licenses §6.1's
  `np.array_equal`.

---

## 10. Traps, in the order you will hit them

1. **`h_p` versus `h_q` in the cross-block** (§3.2). Invisible at equal `nn`; a
   wrong answer that raises nothing the moment `nn` varies. The legacy code has
   exactly this latent bug in `plane_wave_rhs_multi` and `far_field_multi`, which
   read `nn = boundaries[0][0]` and then index with `nn_p`.
2. **Differing `n_clad`** (D3b). Not guarded anywhere in the package today.
3. **Adding a geometric phase to the far-field sum** (§3.4). Absolute
   coordinates already carry it; doubling it looks plausible and is wrong.
4. **Measuring G3 against the addition theorem** (§5.2). Reports a floor that is
   the reference's, not the solver's, and hides the degradation you are looking
   for by three decades.
5. **Padding the addition theorem's truncation "to be safe"** (§3.8). `M = 40`
   is 3.2e-6 where `M = 12` is 1.4e-15.
6. **`Geometry.rad` versus the circumscribing radius** in the overlap test
   (§4.2). They differ by 2.29× on a rounded square.
7. **Calling `scipy.special.hankel1` directly** instead of `kernels.hank0/hank1`
   (§4.1) — loses the Cephes fast path on 99 % of the runtime, and it is what
   makes §5.1's pilot table pessimistic.
8. **Using `fields._is_outside` to classify near-field points** (§3.5). It is
   documented as unreliable for concave shapes, and a cluster runs it `Np` times.
9. **Treating the M1 pilot table as the answer** (§5.1). It was produced by the
   slow-Hankel prototype; the shipped crossover is at a smaller `Np`.

---

## 11. Open items — not blocking, do not resolve silently

- **`efficiencies()` after `scatter_dipole` returns a meaningless number**
  today, silently: `ScatterResult` carries `angle = 0.0` from the dipole path and
  the forward-direction extinction is computed anyway. Pre-existing, out of scope
  (§1.2), mentioned rather than fixed per CLAUDE.md. §4.7 avoids adding a second
  instance by guarding the *cluster* path. Worth its own `fix:`.
- **Does the envelope depend on the facing shapes?** Handoff §8's fourth
  question, left open by D9. Two flat facing sides and two facing tips are
  plausibly different problems, and the answer decides whether the envelope is
  one constant or a family.
- **Is `a` in `√(a/gap)` the larger radius, the smaller, or a harmonic mean?**
  The M2 pilot used one radius pair and cannot tell. Design the `x` sweep to
  answer it.
- **Cluster QNMs.** Free, per §1.3 — §4.6's assembly is all Beyn needs. The most
  likely next milestone, explicitly not attempted here.
