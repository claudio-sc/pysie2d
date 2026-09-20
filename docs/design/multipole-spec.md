# Multipole decomposition of the scattered field — code-spec (draft 2)

**Status:** draft 2, 20 Sep 2026. Not implemented. Ported from
`~/Documents/sie-legacy-0726/sie/multipole.py` (262 lines, written against the
pre-v0.6 solver). Every number in this document was measured on this machine
against `main` at `9224127` with pysie2d 0.6.0 — the measurement scripts are
described in §9 so you can reproduce any of them.

**Draft 2 change:** implementation of draft 1 found that D2's original default
`r0 = 1.5·r_circ`, combined with D3's own default `ntheta` rule (not a fine
grid), under-resolves spiky non-circular shapes at the default `mmax = 8` —
measured coefficient error 5.9e-10 against a 4× finer grid on the star, six
decades worse than draft 1's claimed round-off, and a `RuntimeWarning` on
every default call on that shape. D2's default is revised to `3.0·r_circ`,
the smallest multiplier where the star reaches round-off *and* clears D5's
warning screen at the default `ntheta`. See D2 and D3 below for the full
measurement. §5.4's tolerance needs re-measurement against the new radii
(flagged inline) before this draft is executed.

**Who it is for:** the worker who lands the multipole decomposition. You do not
need to derive anything and you do not need to read the legacy code. Every
formula is stated, every constant has the measurement that fixed it, every test
has the number it must reach, and §4 gives the implementation in full. Where
this document and the legacy file disagree, **this document wins**: the legacy
file has two silent failure modes (§2, D2 and D4) and was written for a solver
whose conventions have since changed twice.

**Decisions it works within:** [CLAUDE.md](../../CLAUDE.md) non-negotiables 1–4,
[docs/conventions.md](../conventions.md) §1–§6 and §9. Nothing here changes a
convention, so `conventions.md` is **not** edited by this work.

**The code in §4 was executed before this document was written**, not sketched:
`_hankel_orders` against direct negative-order scipy at `mmax = 0 … 25` and real
and complex argument, the `c_plus`/`c_minus` slicing against an explicit loop at
`mmax = 0, 1, 2, 8`, and the full projection and reconstruction end to end
against Mie in both polarisations. Reproduced there: `ntheta = 40` from the
default rule, `spectrum_tail` 6.0e-17, `max|Δc⁺|` 2.8e-15. If your transcription
does not reproduce those, it is a transcription error — the snippets are known
to run as written.

---

## 0. How to execute this spec

Read §1, §2 and §8 before writing anything. §4 tells you *what* to write; §2
tells you *why*, which is what you need when a number does not match.

### 0.1 The procedure

1. `git switch main && git pull`. Confirm `git log -1` is `9224127`
   (`docs: scope v0.7 external validation`) or a descendant. If `src/` has
   changed since `9224127`, **stop and report**: every measured number below was
   produced against that tree.
2. `git switch -c v0.7-multipole`.
3. Write the code of §4 and the tests of §5.
4. Run the acceptance checklist, §7. Every item must pass as written.
5. Commit in the three commits of §6, **with exactly the messages given there**.
6. `git push -u origin v0.7-multipole` and open a draft PR (CI runs only on PRs
   and on `main` — a branch with no PR gets no CI).
7. Report back with the §7 checklist ticked and the CI link.

**The roadmap renumbering of §6.1 has already been done** (20 Sep 2026), so it
is not one of your commits. Confirm before you start that `CLAUDE.md`'s roadmap
carries a v0.7 multipole entry and a v0.8 external-validation entry, and that
`docs/design/v0.8-external-validation.md` exists. If it does not, you are on the
wrong commit — **stop and report**.

### 0.2 Rules that are not negotiable

- **Never widen a tolerance to make a test pass** (CLAUDE.md non-negotiable 4).
  Every tolerance in §5 is a measured value with headroom already applied. If an
  assertion fails, or a measured value comes out more than 3× worse than the
  number quoted, **stop and report both numbers**. Do not re-measure and update
  the constant.
- **Do not edit anything listed out of scope in §1.2**, and do not edit
  `docs/conventions.md` at all.
- **Do not reformat or "tidy" code this spec does not touch.** `ruff format`
  must report the tree already formatted when you are done.
- **Do not merge to `main`.** Merging releases to PyPI and is the owner's
  decision (CLAUDE.md, Git workflow).
- **Do not copy the legacy file in.** It is a different solver's conventions and
  it carries the two defects of §2. Write §4 from this document.

### 0.3 The one thing most likely to go wrong

The decomposition **fails silently** when the evaluation circle is too small or
the angular grid too coarse — not by a few percent, but by returning a
confidently wrong number of the right shape and magnitude. Measured: at
`r0 = 0.9·r_circ` the reconstructed field is wrong by 58 % of its own amplitude
and nothing raises (§2, D2). The guards of §4.3 are the substance of this port,
not decoration. If you are tempted to drop one because "the default never trips
it" — the default is exactly why it must stay, because the first user who passes
an explicit `r0` is the one who needs it.

---

## 1. Scope

### 1.1 In

| # | Change | Where |
|---|---|---|
| 1 | New module: `decompose`, `reconstruct`, `Multipoles` | `src/pysie2d/multipole.py` (new) |
| 2 | `ScatterResult.multipoles()` façade | `src/pysie2d/solver.py` |
| 3 | Exports `multipole_decompose`, `multipole_reconstruct`, `Multipoles` | `src/pysie2d/__init__.py` |
| 4 | Regenerated API baseline | `tests/api_baseline.txt` |
| 5 | New test module (§5) | `tests/test_multipole.py` (new) |
| 6 | README section documenting the feature | `README.md` |
| 7 | Status line for this spec flipped to implemented | `docs/design/README.md` |

### 1.2 Out — do not touch

| Out | Why | Owner |
|---|---|---|
| `docs/conventions.md` | This port pins no new convention. The ones it relies on (§1, §2, §4, §5, §6, §9) are already written. | — |
| Multipole decomposition of the **incident/source** field | Does not exist in the legacy tree either; it is a new derivation (regular `J_m` in place of `H_m^{(1)}`). Explicitly deferred. | owner |
| Per-order scattering cross-sections `C_sca,m` | New physics, needs its own normalisation pinned. Not a port. | owner |
| Multiple particles | Separate milestone — [multiparticle-handoff.md](multiparticle-handoff.md) | multiparticle |
| `fields.eval_field`'s Python loop over observation points | Pre-existing; 5.9 ms at `ntheta = 512`, and the default `ntheta` is ~40–100, so it is 0.8 ms. Not this port's problem. **Mention it, do not fix it.** | — |
| `fields.eval_field`'s "keep 5 spacings away" docstring | Measured stale under Kress: at `n_pts = 100, rad = 200` the field is at round-off 0.8 spacings out (§2, D2 table). Pre-existing doc defect. **Report it, do not fix it.** | — |
| Anything under `docs/design/studies/` | Historical | — |

### 1.3 What this feature does *not* cover

State this in the README section (item 7) and in the module docstring. It is the
CROSS-REFERENCES discipline applied inside one repo: an entry names what its
evidence does not reach.

- The anchor is **Mie, on a circle**. For a non-circular shape the only check
  available is internal self-consistency (§5.4) — decompose on one circle,
  reconstruct on another. That proves the expansion is consistent with the
  solver's own field; it does **not** independently validate that field. The
  external anchor for non-circular shapes is the v0.8 milestone.
- The coefficients describe the field **outside the circumscribing circle
  only**. Nothing about the field between that circle and the boundary.
- The expansion centre is the **particle centre** `(x0, z0)`. Coefficients about
  any other centre are a different set of numbers, related by a translation
  addition theorem that is not implemented.

---

## 2. Decisions

Each row gives the evidence that settled it. Where the evidence is a measured
number, §9 says how to reproduce it.

| # | Decision | Evidence / reason |
|---|---|---|
| **D1** | **Signed-order `c_m` is the primary output**; `c_plus`/`c_minus` are derived properties. | Owner's answer, 20 Sep 2026. `c_m` maps onto the Mie coefficients with no reindexing (§3.4), and the ± split is a two-line transform of it (§3.3). The legacy file returned only the ± pair, from which `c_m` is not recoverable without redoing the algebra. The two routes were checked head to head and agree to **5.7e-16**, at real and at complex wavenumber. |
| **D2** | **`r0` defaults to `3.0·r_circ`**, where `r_circ = max hypot(f − x0, g − z0)` is the **circumscribing radius** — *not* a multiple of `Geometry.rad`. `r0 ≤ r_circ` raises. | `Geometry.rad` is a scale parameter, not a radius (conventions §2.4). Measured `r_circ/rad`: circle 1.000, star (`m=5, n1=3, n2=n3=6`) **1.587**, rounded square (`m=4, n1=20, n2=n3=50`) **2.289**, ellipse (`b=2`) 2.000. The legacy caller's "typically 1.5 × rad_ref is sufficient" is therefore **wrong for every non-circular shape**: on the star, `r0 = 1.5·rad = 0.945·r_circ` gives a reconstruction error of **8.7e-1** against a field of amplitude 1.4 — 60 % wrong, silently. Convergence in `r0/r_circ`, star, `mmax = 20` (fine `ntheta`, isolating truncation): `0.90 → 5.8e-1`, `1.00 → 3.1e-2`, `1.02 → 1.4e-5`, `1.05 → 6.1e-7`, `1.10 → 4.1e-10`, `1.20 → 1.0e-13`, `1.50 → 5.2e-15`, `3.00 → 5.1e-15`. **Revised from an original `1.5` (draft 1) after implementation surfaced a second constraint D3 alone does not resolve**: at the default `mmax = 8` and D3's own `ntheta` rule (not a fine grid), the star's angular spectrum decays too slowly at `1.5·r_circ` for that `ntheta` to resolve — measured coefficient error **5.9e-10** against a 4× finer grid (spec claims round-off), and `spectrum_tail` **4.7e-7**, both because `|k·r0|` in the D3 rule does not see how much of a spiky boundary still sits near the evaluation circle. `r0 = 2.0·r_circ` fixes the coefficient error (max Δc `5.68e-15`, at round-off) but not D5's screen: `spectrum_tail = 2.31e-10`, just over `SPECTRUM_TAIL_WARN`, so a default call would still warn despite being correct. Only `3.0·r_circ` clears both: max Δc `3.33e-16`, tail `2.15e-15` — consistent with the `mmax = 20` fine-grid sweep above, which was already at round-off there. 3.0 is the smallest tested multiplier where the default (`mmax = 8`, D3's own `ntheta`) reaches round-off *and* stays below the warning screen on the worst-case shape measured. |
| **D3** | **`ntheta` defaults to `2·(mmax + ceil|k_bg·r0|) + 16`.** | The angular grid must resolve *the field*, whose Fourier content runs to roughly `|k·r0|`, **and** separate the `2·mmax + 1` retained orders. A fixed default (the legacy's 512) resolves neither claim: it is 10× waste on a small particle and silently insufficient once `|k·r0| ≳ 250`. Measured at this default, **at the D2 default `r0 = 3.0·r_circ`**, on four cases spanning `|k·r0| = 3.1 … 10.8` and `mmax = 8 … 30`: self-consistency against a 4× finer grid is **2.5e-16 … 6.6e-16** throughout, at `ntheta = 40 … 96`. **`|k·r0|` alone does not see shape**: at the smaller `r0 = 1.5·r_circ` originally proposed for D2, this same rule gave `5.9e-10` on the star at `mmax = 8` — D2's revision, not this rule, is what fixes it; see D2. |
| **D4** | **`ntheta < 2·mmax + 1` raises.** | Below it the retained basis functions are not orthogonal on the grid and the result is catastrophically, silently wrong. Measured at `mmax = 8`: `ntheta = 16` → coefficient error **2.4e+12**; `ntheta = 17` → **7.1e-12**; `ntheta = 24` → **2.1e-15**. **The spectrum-tail diagnostic of D5 does not catch this case** (its tail at `ntheta = 16` is a healthy 1.2e-6), so the explicit inequality is the only thing standing between the user and a 12-decade error. |
| **D5** | **A `spectrum_tail` diagnostic is computed and returned**; above `1e-10` it emits `RuntimeWarning`. It **warns, it does not raise**. | The tail is the largest of the three angular-DFT coefficients nearest Nyquist, relative to the largest overall — i.e. "did the grid exhaust the field's angular content". It orders the failures correctly: tail `1e-16` → error 5e-15; tail `1.1e-4` → error 1.1e-12; tail `8.5e-1` → error 1.2. It is a reliable **screen**, not a calibrated error bar (the middle and last rows differ by 4 decades in tail and 12 in error), which is why it warns rather than raising, and why the number is put on the result for a test to assert on. At the D2/D3 defaults the tail is round-off on a circle and ~2e-15 on the star (the worst shape measured), both well below the screen, so a default call never warns — this is what forced D2's `r0` up from `1.5·r_circ` to `3.0·r_circ`: at `1.5`, the star's tail was `4.7e-7`, and it stayed above the screen even at `2.0·r_circ` (`2.31e-10`). |
| **D6** | **Non-finite `H_m^{(1)}(k·r0)` raises**, checked once before the projection. | `scipy.special.hankel1` overflows to `inf` at high order and small argument: first non-finite order is **m = 133** at `k·r0 = 0.5`, **170** at 2.0, **217** at 6.28, **295** at 20.0. Beyond it every coefficient silently becomes `0` or `nan`. Not reachable at sane `mmax`, cheap to exclude, and the message can say the useful thing (raise `r0`, or lower `mmax`). |
| **D7** | **Call `fields.eval_field` with `ri=None`**, and trip on any observation point that comes back exactly `0+0j`. | `ri=None` makes a point that `_is_outside` misclassifies as interior return exactly `0+0j` rather than an interior field value, converting a wrong number into a detectable one. `_is_outside` is documented as "unreliable for extreme concave superformula shapes". Measured: **0 misclassifications out of 720 angles** on each of three shapes (the `m=5` star, a cusped `m=6, n1=0.5` star, the rounded square) at `r0/r_circ ∈ {1.05, 1.2, 1.5}` — so the tripwire is not expected ever to fire. It costs one `np.any` and it removes the last way this function can return a plausible wrong answer. |
| **D8** | Evaluate `hankel1` only at orders `0 … mmax`; get negative orders from `H_{−m}^{(1)} = (−1)^m H_m^{(1)}`. | Halves the special-function work — which is 99 % of the runtime of anything in this package (CLAUDE.md, *Performance shape*) — and avoids depending on scipy's negative-order handling. |
| **D9** | **Do not** apply `kernels._real_if_real` to `wnum_bg` here. | It exists to reach the Cephes order-0/1 fast path inside `hank0`/`hank1`. There is no general-order fast path, `scipy.special.hankel1` takes the same branch either way, and applying it would imply a speed-up that does not exist. Complex `wnum_bg` works unchanged — verified against Mie at `λ = 600 + 5j` and `600 + 30j`, **8.7e-16** (non-negotiable 1). |
| **D10** | Module functions are `decompose` and `reconstruct`; the package exports them **aliased** as `multipole_decompose` and `multipole_reconstruct`. | `pysie2d.decompose` is too generic a name for a package whose surface is flat. The alias is one line in one place, it is what the legacy package did (`sie/__init__.py:77`), and it reads correctly at both call sites: `multipole.decompose(...)` inside, `from pysie2d import multipole_decompose` outside. |
| **D11** | `decompose` returns a frozen dataclass `Multipoles`, not a tuple. | `reconstruct` needs `c`, `wnum_bg` and `mmax` together; returning them as a tuple makes every caller carry four variables in the right order. The dataclass also carries `spectrum_tail` (D5), which a bare tuple has nowhere to put. |
| **D12** | Ships as **v0.7.0**; the external-validation milestone is renumbered **v0.8**. | Owner's answer, 20 Sep 2026. This work is purely additive and its gate is one already-measured test; the external validation is slow and depends on conda environments that CI will never run. Blocking a finished feature behind it is backwards. A roadmap number is cheap to change, a published PyPI version is not. |

---

## 3. The mathematics

Notation: `k` is the background wavenumber `k_bg = 2π·n_clad/λ_vac` (conventions
§2.2); `(x0, z0)` the particle centre; `r0` the evaluation radius about that
centre; `θ` the polar angle **in the repo's convention**, measured from `+z`
toward `+x`, so that a point is `x = x0 + r sin θ`, `z = z0 + r cos θ`. That is
the same `θ` that `geometry.gielis` uses (`f = r sin θ + x0`,
`g = r cos θ + z0`, [geometry.py:61-62](../../src/pysie2d/geometry.py#L61-L62))
and the same one `far_field` reports angles in. **It is not the usual
`x = r cos θ`.** Getting this wrong rotates every coefficient by a
phase and the Mie test of §5.2 will fail.

### 3.1 The expansion

Outside the circumscribing circle the scattered field is

    ψ_sc(r, θ) = Σ_{m=−∞}^{∞} c_m · i^m · H_m^{(1)}(k·r) · e^{imθ}

with `H_m^{(1)}` the Hankel function of the first kind — outgoing under the
`exp(−iωt)` convention (conventions §3). The `i^m` is part of the basis, kept
from the source formulation so that the `c_m` are the numbers the reference
paper tabulates.

For negative orders use

    H_{−m}^{(1)}(z) = (−1)^m · H_m^{(1)}(z)

### 3.2 The projection

Sample the field at midpoint nodes on the circle,

    θ_j = 2π(j + ½)/N,    j = 0 … N−1,    N = ntheta

    ψ_j = ψ_sc(x0 + r0 sin θ_j,  z0 + r0 cos θ_j)

then

    P_m = (1/N) · Σ_j ψ_j · e^{−imθ_j}                (angular DFT)

    c_m = P_m / ( i^m · H_m^{(1)}(k·r0) )

for `m = −mmax … mmax`. This is exact for a field whose angular content is
resolved by the grid; D3 and D4 are what make that true.

The `i^m` and `H_m` both cancel out of any ratio, so the projection is exact at
**complex** `k` as well — the orthogonality lives in `θ`, which is always real.
Verified in §5.3.

### 3.3 The symmetric / antisymmetric basis

The source formulation reorganises the expansion into

    ψ_m^+(r, θ) = i^m H_m^{(1)}(k r) cos(mθ)          [even]
    ψ_m^-(r, θ) = i^m H_m^{(1)}(k r) · i sin(mθ)      [odd]

which works because `i^{−m} H_{−m}^{(1)} = i^m H_m^{(1)}`, so the `+m` and `−m`
terms share a factor. Hence

    c_m^+ = c_m + c_{−m}   for m ≥ 1,   c_0^+ = c_0
    c_m^- = c_m − c_{−m}   for m ≥ 1

`c_0^+ = c_0` and **not** `2c_0`: the `m = 0` term appears once in the sum, not
twice. This is the single most likely off-by-a-factor-of-two in the port, and
§5.2 catches it.

For `pol = 1` (TM, `H_y`) the leading terms read physically as
`m = 0` → magnetic dipole, `m = 1` → electric dipole, `m = 2` → electric
quadrupole. Put that in the module docstring; it is the reason the ± basis is
kept at all.

### 3.4 The Mie relation — the anchor

For a **circular** particle of radius `a` centred at the origin, under the
plane wave `plane_wave_rhs` produces at incidence angle `α` (degrees, `α_r` in
radians):

    c_m = −(−1)^m · χ_m · e^{i m α_r}

where

    χ_m = b_m   for pol = 2 (TE, E_y)
    χ_m = a_m   for pol = 1 (TM, H_y)

are the Bohren & Huffman cylinder coefficients as
`pysie2d.reference.mie.mie_coefficients(x, m_rel)` returns them, at size
parameter `x = ScatterResult.size_parameter` and `m_rel = Material.nc`
(conventions §1, §2.4). In the ± basis:

    c_m^+ = −2·(−1)^m · χ_m · cos(m·α_r)    for m ≥ 1
    c_0^+ = −χ_0
    c_m^- = −2i·(−1)^m · χ_m · sin(m·α_r)   for m ≥ 1

**This relation was measured, not derived from a textbook**, precisely so that
no sign or phase in it is a guess. Agreement, both polarisations, `α = 0°` and
`α = 30°`, `rad = 200 nm`, `λ = 600 nm`, `n_core = 1.5`, `mmax = 8`:

| pol | α | max&#124;Δc⁺&#124; | max&#124;Δc⁻&#124; |
|---|---|---|---|
| 2 (TE) | 0° | 2.795e-15 | 4.347e-16 |
| 2 (TE) | 30° | 2.763e-15 | 1.319e-15 |
| 1 (TM) | 0° | 2.264e-15 | 3.056e-16 |
| 1 (TM) | 30° | 2.493e-15 | 1.056e-15 |

against coefficients of magnitude up to 1.86.

> **At normal incidence `c_m^- ≡ 0` identically** (`sin(0) = 0`), and `c_{−m} =
> c_m`. An implementation with a broken antisymmetric half passes every
> normal-incidence test. **The oblique case is not optional** — it is the only
> thing that exercises half the output.

### 3.5 Two invariances the coefficients must have

Both are free tests, and both catch a whole class of centring and scaling bugs
that the Mie test does not.

**Translation.** Moving the particle to `(x0, z0)` multiplies every coefficient
by the phase the incident plane wave acquires at the new centre:

    c_m(x0, z0) = c_m(0, 0) · exp( i·k·(x0·sin α_r − z0·cos α_r) )

because `plane_wave_rhs` phases the incident field to the **origin**
(`exp(i k (f sin α − g cos α))`,
[sources.py:44-46](../../src/pysie2d/sources.py#L44-L46)) while the expansion is
about the **particle centre**. Measured on the star at `(500, 0)`, `(0, −300)`
and `(500, −300)`: **2.2e-15, 2.1e-15, 1.4e-15**.

**Scale covariance** (conventions §9). `(rad, λ) → (s·rad, s·λ)` leaves every
`c_m` unchanged, exactly, because the whole problem depends only on `k·rad` and
`c_m` is dimensionless. Measured on the star: **0.0** at `s = 0.5` and `s = 2`
(exact, both powers of two), **3.2e-15** at `s = 7`.

---

## 4. The implementation

### 4.1 `src/pysie2d/multipole.py` — new file

Module docstring: state the expansion (§3.1), the `θ` convention (§3, the
`x = r sin θ` warning), the ± reorganisation and its MD/ED/EQ reading (§3.3),
that the expansion is valid **outside the circumscribing circle only**, and that
the anchor is Mie on a circle (§1.3).

```python
"""Cylindrical-harmonic multipole decomposition of the BIE scattered field.

...docstring per the paragraph above...
"""

import warnings
from dataclasses import dataclass

import numpy as np
from scipy.special import hankel1

from .fields import eval_field

PI = np.pi

# Above this, the angular grid has not exhausted the field's angular content
# and the coefficients are suspect. A screen, not an error bar: at a tail of
# 1.1e-4 the measured coefficient error was 1.1e-12, at 8.5e-1 it was 1.2.
# The default ntheta (`_default_ntheta`) at the D2 default r0 lands at
# round-off on a circle and ~2e-15 on the worst shape measured (the star) —
# both several decades below this screen.
SPECTRUM_TAIL_WARN = 1e-10


@dataclass(frozen=True)
class Multipoles:
    """Cylindrical-harmonic coefficients of a scattered field.

    Attributes:
        c: complex (2*mmax+1,) coefficients c_m for m = −mmax … mmax, with
            ``c[m + mmax]`` the coefficient of order m. The expansion is
            ψ_sc = Σ_m c_m · i^m · H_m^(1)(k_bg·r) · e^{imθ}, valid outside
            the circumscribing circle of the particle only, with θ measured
            from +z toward +x about ``(x0, z0)``.
        mmax: Highest order retained.
        r0: Radius of the evaluation circle (nm).
        wnum_bg: Background wavenumber the field was evaluated at (rad/nm).
        x0, z0: Expansion centre — the particle centre (nm).
        ntheta: Number of angular samples used.
        spectrum_tail: Angular-DFT content nearest Nyquist, relative to the
            largest coefficient. A resolution screen: ≲1e-12 means the grid
            exhausted the field's angular content. See ``SPECTRUM_TAIL_WARN``.
    """

    c: np.ndarray
    mmax: int
    r0: float
    wnum_bg: complex
    x0: float
    z0: float
    ntheta: int
    spectrum_tail: float

    @property
    def orders(self) -> np.ndarray:
        """(2*mmax+1,) integer orders m = −mmax … mmax, aligned with ``c``."""
        return np.arange(-self.mmax, self.mmax + 1)

    @property
    def c_plus(self) -> np.ndarray:
        """(mmax+1,) symmetric coefficients c_m^+ = c_m + c_{−m}, m = 0 … mmax.

        ``c_0^+ = c_0``, not ``2·c_0``: the m = 0 term appears once in the sum.
        """
        m = self.mmax
        out = self.c[m:] + self.c[m::-1]
        out[0] = self.c[m]
        return out

    @property
    def c_minus(self) -> np.ndarray:
        """(mmax,) antisymmetric coefficients c_m^- = c_m − c_{−m}, m = 1 … mmax.

        Identically zero at normal incidence on a boundary symmetric about the
        incidence axis — which is why a test of this half must be run oblique.
        """
        m = self.mmax
        return self.c[m + 1 :] - self.c[m - 1 :: -1]

    def reconstruct(self, r: float, theta: np.ndarray) -> np.ndarray:
        """Field on a circle of radius ``r`` about the expansion centre.

        Args:
            r: Radius (nm), measured from ``(x0, z0)``. Must exceed the
                circumscribing radius; nothing here can check that.
            theta: (M,) angles (rad), from +z toward +x.

        Returns:
            complex (M,) reconstructed scattered field.
        """
        return reconstruct(self.c, self.wnum_bg, r, theta)
```

### 4.2 Hankel helper and defaults

```python
def _hankel_orders(mmax: int, z: complex) -> np.ndarray:
    """H_m^(1)(z) for m = −mmax … mmax, from the orders 0 … mmax only.

    Uses H_{−m}^(1) = (−1)^m H_m^(1) rather than calling scipy at negative
    order: the special functions are 99 % of this package's runtime, so
    evaluating each order once is the whole optimisation available here.
    """
    h_pos = hankel1(np.arange(mmax + 1), z)
    signs = (-1.0) ** np.arange(mmax, 0, -1)
    return np.concatenate([signs * h_pos[:0:-1], h_pos])


def _default_ntheta(mmax: int, wnum_bg: complex, r0: float) -> int:
    """Angular samples that resolve both the field and the retained orders.

    The field's angular content runs to roughly |k·r0|; the 2·mmax+1 retained
    orders must additionally be separated. Measured at this rule: agreement
    with a 4× finer grid of 2.5e-16 … 6.6e-16 over |k·r0| = 3.1 … 10.8 and
    mmax = 8 … 30 (docs/design/multipole-spec.md §2, D3).
    """
    return int(2 * (mmax + np.ceil(abs(wnum_bg * r0))) + 16)
```

### 4.3 `decompose`

The guards are in the order given. Each one must raise **before** any expensive
work, and each message must name the offending value *and* what to do about it.

```python
def decompose(
    ei: np.ndarray,
    nn: int,
    f: np.ndarray,
    df: np.ndarray,
    g: np.ndarray,
    dg: np.ndarray,
    delt: float,
    wnum_bg: complex,
    r0: float,
    *,
    mmax: int = 8,
    ntheta: int | None = None,
    x0: float = 0.0,
    z0: float = 0.0,
) -> Multipoles:
    """Project the scattered field onto cylindrical harmonics.

    ...Google-style Args/Returns/Raises, per CLAUDE.md Style...

    Raises:
        ValueError: If ``mmax`` is negative; if ``ntheta`` is below
            ``2·mmax + 1``; or if ``H_mmax^(1)(k_bg·r0)`` overflows.
    """
```

1. `mmax < 0` → `ValueError`.
2. `ntheta`: default from `_default_ntheta`; then
   **`if ntheta < 2 * mmax + 1: raise ValueError(...)`** — message must state
   that below this the retained orders are not orthogonal on the grid and quote
   the minimum. (D4.)
3. `h_all = _hankel_orders(mmax, wnum_bg * r0)`;
   **`if not np.all(np.isfinite(h_all)): raise ValueError(...)`** — message must
   say "increase r0 or reduce mmax". (D6.)
4. Build nodes and evaluate:
   ```python
   theta = (np.arange(ntheta) + 0.5) * 2.0 * PI / ntheta
   x_pts = x0 + r0 * np.sin(theta)
   z_pts = z0 + r0 * np.cos(theta)
   psi = eval_field(ei, nn, f, df, g, dg, delt, wnum_bg, x_pts, z_pts, ri=None)
   ```
   `ri=None` is load-bearing (D7): it makes a misclassified point return exactly
   `0+0j` instead of an interior field value.
5. **Tripwire** (D7):
   ```python
   if np.any(psi == 0.0):
       raise ValueError(
           f"eval_field returned exactly zero at "
           f"{int(np.count_nonzero(psi == 0.0))} of {ntheta} points on the "
           f"r0 = {r0:.6g} nm circle, which means its inside/outside test "
           f"placed them inside the particle. r0 must exceed the "
           f"circumscribing radius of the boundary."
       )
   ```
6. Projection:
   ```python
   orders = np.arange(-mmax, mmax + 1)
   spectrum = (np.exp(-1j * np.outer(orders, theta)) @ psi) / ntheta
   c = spectrum / (1j**orders) / h_all
   ```
7. Diagnostic (D5), computed from the **full** grid, not the retained orders:
   ```python
   full = np.fft.fft(psi) / ntheta
   nyq = ntheta // 2
   peak = np.abs(full).max()
   tail = float(np.abs(full[nyq - 1 : nyq + 2]).max() / peak) if peak > 0.0 else 0.0
   if tail > SPECTRUM_TAIL_WARN:
       warnings.warn(..., RuntimeWarning, stacklevel=2)
   ```
   The warning text must quote the tail, and say that `ntheta` is too small for
   this `k_bg·r0`.
8. Return the `Multipoles`.

> **`r0 ≤ r_circ` is not checkable here.** The primitive receives boundary
> arrays, so it *could* compute `r_circ` — but `(x0, z0)` are parameters, and a
> caller who passes the wrong centre would get a wrong `r_circ` and a guard that
> lies. The check belongs where the centre is known to be right: the
> `ScatterResult` façade (§4.5). The tripwire of step 5 is the primitive's own
> backstop and it fires on exactly the same mistake.

### 4.4 `reconstruct`

```python
def reconstruct(
    c: np.ndarray,
    wnum_bg: complex,
    r: float,
    theta: np.ndarray,
) -> np.ndarray:
    """Rebuild the scattered field from signed-order coefficients."""
    theta = np.asarray(theta, dtype=float)
    mmax = (len(c) - 1) // 2
    orders = np.arange(-mmax, mmax + 1)
    radial = c * (1j**orders) * _hankel_orders(mmax, wnum_bg * r)
    return radial @ np.exp(1j * np.outer(orders, theta))
```

`len(c)` must be odd; raise `ValueError` if it is not — it is the one way a
caller can pass a `c_plus` array by mistake and get a plausible wrong answer.

### 4.5 `ScatterResult.multipoles` — `solver.py`

```python
    def multipoles(
        self,
        mmax: int = 8,
        *,
        r0: float | None = None,
        ntheta: int | None = None,
    ) -> Multipoles:
        """Cylindrical-harmonic decomposition of the scattered field.

        Args:
            mmax: Highest order retained. The default of 8 resolves the
                electric quadrupole and two orders beyond it.
            r0: Radius of the evaluation circle (nm), about the particle
                centre. Default ``3.0 × the circumscribing radius`` — which is
                **not** 3.0 × ``Geometry.rad``: on a rounded square the
                circumscribing radius is 2.29 × ``rad``.
            ntheta: Angular samples. Default resolves both the field and the
                retained orders; see ``pysie2d.multipole``.

        Returns:
            Multipoles, about the particle centre ``(geometry.x0,
            geometry.z0)``.

        Raises:
            ValueError: If ``r0`` does not exceed the circumscribing radius of
                the boundary — inside it the expansion does not converge and
                the coefficients are meaningless (measured: 58 % error at
                0.9 × the circumscribing radius, with nothing else to warn you).
        """
        geo = self.geometry
        r_circ = float(np.hypot(geo.f - geo.x0, geo.g - geo.z0).max())
        if r0 is None:
            r0 = 3.0 * r_circ
        elif r0 <= r_circ:
            raise ValueError(
                f"r0 = {r0:.6g} nm does not exceed the circumscribing radius "
                f"{r_circ:.6g} nm of this boundary; the multipole expansion "
                f"does not converge inside it. Note that the circumscribing "
                f"radius is not Geometry.rad = {geo.rad:.6g} nm."
            )
        return decompose(
            self.ei, geo.n_pts, geo.f, geo.df, geo.g, geo.dg, geo.delt,
            self.wnum_bg, r0, mmax=mmax, ntheta=ntheta, x0=geo.x0, z0=geo.z0,
        )
```

### 4.6 Exports — `__init__.py`

```python
from .multipole import Multipoles
from .multipole import decompose as multipole_decompose
from .multipole import reconstruct as multipole_reconstruct
```

Add `"Multipoles"`, `"multipole_decompose"`, `"multipole_reconstruct"` to
`__all__` (it is sorted), and **add all three to the `Public API:` list in the
module docstring** — `test_every_export_is_documented_in_the_package_docstring`
fails otherwise. Then regenerate the baseline with the command in
[test_public_api.py:91](../../tests/test_public_api.py#L91):

```
uv run python -c "from tests.test_public_api import _surface; print('\n'.join(_surface()))" > tests/api_baseline.txt
```

---

## 5. Tests — `tests/test_multipole.py`

Every tolerance below is the measured value with headroom; §2 and §3 carry the
measurements. Each test's docstring must state **what physical property is being
checked and why it cannot pass by accident** (CLAUDE.md, Style).

Fixtures: `rad = 200.0`, `λ = 600.0`, `n_core = 1.5`, `n_clad = 1.0`,
`n_pts = 100`. The star is `Geometry.gielis(200.0, 200, m=5, n1=3.0, n2=6.0,
n3=6.0)`, whose circumscribing radius is `1.587 × rad`.

**§5.1–§5.3 evaluate at `r0 = 1.5·r_circ`, not the D2 default.** The Mie
relation is independent of the evaluation radius — the radial factor
`i^m·H_m^{(1)}(k·r0)` is divided out — but the residual is a handful of ulp of a
coefficient of magnitude ~1.9 and it jitters across the whole
`2.4e-15 … 3.1e-15` band with a **one-ulp** change of `r0` (measured: at
`r0 = 600.0` exactly, 2.58e-15; at the next float up, 3.12e-15). The `3e-15`
below sits inside that band, so it is pinned to the radius the number was
measured at — at the draft-2 default of `3.0·r_circ` the worst case is
`3.117e-15` (pol = 1, `α = 0`) and the assertion fails by 4 %. Widening the
constant was declined (non-negotiable 4); **that `3e-15` is a ~10-ulp bound with
no headroom for a different BLAS or libm is an open item for the owner**, not a
defect in the implementation. The default radius is exercised by §5.4, §5.7 and
§5.9.

### 5.1 `test_mie_coefficients_TE_and_TM_normal_incidence`
Both polarisations, `α = 0`. Assert `c_plus` against §3.4 and that `c_minus` is
identically zero. **tol: `atol = 3e-15`** (measured worst 2.795e-15 against
coefficients of magnitude 1.86 — a round-off floor, not a convergence order:
under Kress the circle anchors reach round-off by `nn ≈ 30–40` and this runs at
100). Docstring must say that the `c_minus` half of this test is trivially
satisfied and that §5.2 is what actually checks it.

### 5.2 `test_mie_coefficients_TE_and_TM_oblique_incidence`
Both polarisations, `α = 30°`. Assert `c_plus` **and** `c_minus` against §3.4.
**tol: `atol = 3e-15`** (measured worst 2.763e-15). Also assert
`max|c_minus| > 0.7`, so the test fails loudly if a refactor ever makes the
antisymmetric half vanish. Docstring: this is the only test that exercises
`c_minus`, because at normal incidence it is zero by symmetry.

### 5.3 `test_mie_relation_holds_at_complex_wavelength`
`λ = 600 + 30j`, `pol = 2`, `α = 30°`, Mie evaluated at the complex size
parameter. **tol: `atol = 3e-15`** (measured 8.671e-16). Docstring must cite
non-negotiable 1: the complex path is what makes QNM work and it is never to be
simplified away.

### 5.4 `test_reconstruction_matches_the_solver_field_on_a_second_circle`
The star, `mmax = 20`, decompose at `3.0·r_circ` (the D2 default), reconstruct
at `6.0·r_circ`, compare against `ScatterResult.eval_field` there. **tol:
`atol = 1e-13`** (re-measured against the draft-2 radii: **1.644e-15** at
`mmax = 20`, against a field of amplitude 0.90 — 78× *better* than draft 1's
`1.28e-13`, because doubling the reconstruction radius with the decompose
radius leaves far less truncated content there). The convergence in `mmax` at
these radii: `8 → 3.7e-06`, `12 → 1.3e-09`, `16 → 1.3e-13`, `20 → 1.6e-15`,
`24 → 1.1e-15`, so `mmax = 20` has just reached the round-off floor and the
`1e-13` bound still fails if four retained orders are dropped. Docstring must be
explicit that this is a **self-consistency** check, not an independent
validation — it shows the expansion reproduces the solver's own field, and
nothing more. Note in the docstring that the residual here is set by
**truncation** (`mmax`), whereas §5.7 is set by **quadrature** (`ntheta`): they
are different knobs.

### 5.5 `test_expansion_centre_follows_the_particle`
Star at `(0,0)` vs at `(500, −300)`, `α = 30°`. Assert the phase relation of
§3.5. **tol: `atol = 3e-15`** (measured 1.419e-15). Docstring: this is what
catches an expansion built about the origin instead of the particle centre — a
bug the Mie tests, which centre the particle at the origin, cannot see.

### 5.6 `test_coefficients_are_scale_covariant`
Star, `(rad, λ) → (2·rad, 2·λ)` and `(7·rad, 7·λ)`. **tol: `atol = 1e-14`**
(measured exactly 0.0 at `s = 2`, 3.189e-15 at `s = 7`). Cite conventions §9.

### 5.7 `test_default_angular_grid_is_converged`
For the circle and the star: decompose at the default `ntheta`, and again at
`4 × ntheta`; assert agreement. **tol: `atol = 1e-14`** (measured 2.5e-16 …
6.6e-16). Also assert `spectrum_tail < 1e-12` at the default.

### 5.8 `test_symmetric_basis_matches_an_independent_projection`
Implement the legacy ± projection **independently inside the test** — build
`ψ_m^±` explicitly, project with `np.dot(field, psi.conj()) / np.sum(|psi|²)` —
and assert it reproduces `c_plus`/`c_minus`. **tol: `atol = 1e-15`** (measured
5.661e-16, at real and complex `k`). Docstring must explain why a second
implementation earns its place here: it is the same argument that keeps
`assemble_matrix_reference` (CLAUDE.md, `/simplify` note), and it is what proves
the `c_0^+ = c_0` special case (§3.3) is right rather than merely consistent.

### 5.9 `test_guards` — one test per guard, all `pytest.raises`/`pytest.warns`
- `r0` equal to and just below `r_circ` → `ValueError` naming the circumscribing
  radius. Include the **star at `r0 = 1.5·rad`**, which is `0.945·r_circ` — the
  exact mistake the legacy docstring's "1.5 × rad_ref" invites.
- `ntheta = 2·mmax` → `ValueError`. Docstring must quote D4's measured 2.4e+12,
  so nobody later "simplifies" the guard away.
- `mmax` large enough to overflow `hankel1` (e.g. `mmax = 400` at the default
  `r0`) → `ValueError`.
- `ntheta` small but legal (`= 2·mmax + 1`) on a large `k·r0` case →
  `pytest.warns(RuntimeWarning)`.
- `reconstruct` with an even-length `c` → `ValueError`.
- Negative `mmax` → `ValueError`.

### 5.10 `test_public_api.py`
No new test. The baseline regeneration of §4.6 is the change.

---

## 6. Commits

Three, in this order, with exactly these messages.

```
feat: multipole decomposition of the scattered field

Cylindrical-harmonic expansion of the exterior scattered field about the
particle centre, ported from the pre-v0.6 research code. Returns signed-order
coefficients c_m with the symmetric/antisymmetric pair derived; validated
against analytic Mie cylinder coefficients in both polarisations, at normal and
oblique incidence, at real and complex wavelength, to 2.8e-15.

The port adds the three guards the original lacked: the evaluation circle must
exceed the circumscribing radius (not Geometry.rad — 2.29x rad on a rounded
square), the angular grid must separate the retained orders, and the angular
spectrum is screened for under-resolution. Each had a silent failure mode
behind it; the first was 58 % error with nothing raised.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
```

```
test: Mie, invariance and guard tests for the multipole decomposition

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
```

```
docs: document the multipole decomposition in the README

Also flips this feature's roadmap and design-index rows from specced to
implemented.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
```

### 6.1 The roadmap renumbering — already done, do not redo it

Landed 20 Sep 2026, before this spec was handed to you:

- `docs/design/v0.7-external-validation.md` → `v0.8-external-validation.md`,
  with its title and its two internal `v0.7` references renumbered.
- `CLAUDE.md` *Roadmap*: a v0.7 entry for this work, a v0.8 entry for the
  external validation, and multiple particles named as the remaining
  API-breaking item and therefore the gate on v1.0.
- `docs/design/README.md`: rows added for this spec, the renamed validation
  doc, and the multiparticle handoff.
- `docs/design/v0.6-architecture.md`: its forward reference "item 4 moves to
  v0.7" now reads "to a later release" — that number referred to *the next
  release*, which is no longer the external-validation one.

Two things it deliberately did **not** do, both the owner's call:

- **The branch `v0.7-validation` was not renamed**, locally or on the remote.
  Renaming a pushed branch is the one git operation that can strand work that
  is not yours. It currently holds the v0.8 doc under a v0.7 name; say so in
  your PR body and leave it alone.
- `CLAUDE.md` says the v0.7 roadmap entry is *specced, not implemented*. The
  third commit below is what makes that line true, so **the fourth bullet of
  §1.1 item 7 is yours**: flip this spec's row in `docs/design/README.md` from
  *"Not implemented"* to *"Historical. The authority on intent."*

---

## 7. Acceptance checklist

Run in order. Every line must pass as written.

- [ ] `uv sync --frozen` clean. **No dependency was added** — `hankel1` comes
      from scipy, already a dependency (CLAUDE.md, *Dependencies*).
- [ ] `uv run pytest tests/test_multipole.py -q` — all pass.
- [ ] `uv run pytest -q` — the whole suite passes, no new warnings, no test
      slower than it was.
- [ ] `uv run ruff check src tests` and `uv run ruff format --check src tests`
      both clean.
- [ ] `uv run mypy src` clean (the package is typed and ships `py.typed`).
- [ ] `uv run pytest tests/test_public_api.py -q` passes **with the regenerated
      baseline**, and the three new names appear in both `__all__` and the
      module docstring.
- [ ] Every tolerance in `tests/test_multipole.py` carries a comment with the
      measured number behind it (non-negotiable 4).
- [ ] `git log --oneline` shows exactly the three commits of §6, in order.
- [ ] `grep -rn 'v0\.7' docs/ CLAUDE.md README.md` returns only the roadmap
      entry for this feature and this spec's own references — no stale
      reference to the external-validation milestone under its old number.
- [ ] PR opened against `main`, CI green.

---

## 8. Traps, in the order you will hit them

1. **`θ` is measured from `+z`, not `+x`.** `x = x0 + r sin θ`,
   `z = z0 + r cos θ`. A `cos`/`sin` swap rotates every coefficient by `e^{imπ/2}`
   and §5.1 fails with errors of order 1.
2. **`c_0^+ = c_0`, not `2·c_0`.** §3.3.
3. **`Geometry.rad` is not a radius.** §2, D2. If you write `1.5 * geo.rad`
   anywhere, you have reintroduced the legacy bug.
4. **Normal incidence proves nothing about `c_minus`.** §3.4.
5. **The spectrum-tail diagnostic does not catch `ntheta < 2·mmax + 1`.** Its
   tail at that failure is a healthy 1.2e-6 while the error is 2.4e+12. Two
   separate guards, both needed. §2, D4.
6. **`ri=None` in the `eval_field` call is deliberate.** Passing
   `self.material.nc` would silently return interior-field values for any
   misclassified point instead of the detectable `0+0j`. §2, D7.
7. **Do not "optimise" `eval_field`'s point loop.** It is 0.8 ms at the default
   `ntheta`, dense linear algebra is ~2 % of this package's runtime, and the
   special functions are 99 % (CLAUDE.md, *Performance shape*). The only
   optimisation taken here is D8, evaluating each Hankel order once.
8. **`hankel1` at negative order is never called.** D8. If you find yourself
   writing `hankel1(-m, z)`, you have diverged from §4.2.

---

## 9. Reproducing the measurements

Every number in §2 and §3 came from a short script against `main` at `9224127`
with `.venv/bin/python`. They are not committed — they are throwaway
measurements, and this document is their record. To re-derive any of them:

| Number | How |
|---|---|
| Mie relation and its sign (§3.4) | Solve a circle, project `ψ_sc` on a circle of radius `3·rad` onto `e^{imθ}`, divide by `hankel1(m, k·r0)`, and take the ratio to `mie.mie_coefficients`. The ratio is `−(−i)^m` exactly; that is where `c_m = −(−1)^m χ_m` comes from. |
| `r0` convergence (D2) | Star, `mmax = 20`; sweep `r0/r_circ ∈ [0.9, 3.0]`; at each, decompose and reconstruct at `6·rad`, compare to `eval_field`. |
| `ntheta` failure (D4) | `mmax = 8`, sweep `ntheta ∈ {8, 16, 17, 18, 24, 32}` against the Mie reference. |
| Tail vs error (D5) | Same sweep, recording `max|DFT near Nyquist| / max|DFT|`. |
| Overflow orders (D6) | `hankel1(m, z)` for increasing `m` at fixed `z`, first non-finite. |
| Misclassification count (D7) | `fields._is_outside` at 720 angles on three shapes, `r0/r_circ ∈ {1.05, 1.2, 1.5}`. |
| `r_circ/rad` table (D2) | `np.hypot(g.f − g.x0, g.g − g.z0).max() / g.rad` at `n_pts = 400`. |
| Translation and scale invariance (§3.5) | §5.5 and §5.6 are these measurements, promoted to tests. |

## 10. Open items — not blocking, do not resolve silently

- **[?]** The incident/source-field decomposition (regular `J_m` harmonics) is
  the natural companion and does not exist in the legacy tree either. It is a
  new derivation, out of scope here (§1.2), and worth doing next if the
  application needs the excitation's multipole content rather than the
  response's.
- **[?]** Per-order scattering cross-sections `C_sca,m` would make the
  decomposition sum to a measurable total and give it a second Mie-checkable
  anchor. Deferred with the same reasoning.
- **[D]** `fields.eval_field`'s "keep ~5 boundary-point spacings away" docstring
  predates Kress quadrature. Measured here: at `n_pts = 100`, `rad = 200 nm`,
  the field is at round-off **0.8 spacings** out (`r0 = 1.05·rad`, error
  2.0e-15) and only degrades to 3e-7 at 0.08 spacings. The docstring is
  conservative by roughly a factor of six. **Not fixed by this work** — report
  it, it belongs to whoever owns `fields.py`.
