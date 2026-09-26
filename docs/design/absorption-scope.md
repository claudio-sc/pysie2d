# Scope: an absorption check for lossy dielectrics and metals

**Status: scoped, not implemented.** This document decides *what* the check is
and *what it needs*; no code, no runs. Written 2026-09-23, after the v1.0
freeze.

The one-line summary: the lossy-dielectric half is largely covered already and
the real gap is **metals**, which the package cannot currently express at all.

---

## 1. What exists today

Stating this first so the check does not re-buy evidence already in the repo.

| material | anchor | where |
|---|---|---|
| Lossy dielectric, circle | analytic Mie series (bootstrap) | `tests/data/mie-circle-lossy-{te,tm}.json` |
| Lossy dielectric, circle | dolfinx, external | `tests/data/dolfinx-circle-lossy-{te,tm}.json` |
| Lossy dielectric, circle | optical theorem, energy balance | `tests/test_efficiencies.py` |
| Lossy dielectric, **non-circular** | *nothing* | — |
| **Metal**, any shape | *nothing* | — |

So absorption in a weakly lossy dielectric on a circle is genuinely
well-anchored: a closed form and an independent PDE solver agree with the
package to 1e-05. What is untested is (a) absorption off a circle and (b) any
material with `Re ε < 0`.

This document scopes (b). Absorption on a non-circular shape is a *simulation*
problem — it needs a dolfinx star lossy run, roughly two machine-hours — and is
tracked separately; it needs no design decision, only the run.

## 2. The check

**One test, on a circle, against the analytic Mie series.**

`reference/mie.py` already accepts a complex relative index `m`, so a metal
cylinder has a closed-form reference available today with no external tool, no
conda environment, and no frozen data file. That makes it the strongest and
cheapest anchor available: it runs in CI for every user, on every commit.

Shape: circle, `rad = 200 nm`. Material points, all non-dispersive:

| point | `ε` (absolute) | `|m|` | purpose |
|---|---|---|---|
| lossy dielectric | `2.25 + 0.5i` | 1.52 | the regime already frozen; guards the new path against regression |
| metal, moderate | `-16.0 + 0.5i` | 4.00 | `Re ε < 0`, silver-like at ~600 nm |
| metal, strong | `-40.0 + 1.5i` | 6.33 | pushes `|m|·x` to the edge of the series truncation |

Both polarisations. The assertion is on `C_abs` specifically — not only on the
optical-theorem balance, which can be satisfied by two compensating errors.

**Not a frozen data file.** A plain test against `reference/mie.py`, like
`tests/test_efficiencies.py`. This matters: the frozen-file `material` block is
`{n_core, n_clad, epsi}` and cannot express `Re ε < 0` either, so routing the
metal check through `tests/data/` would force a `spectrum.py` schema bump to
`pysie2d-external-validation/2`. Going direct avoids that entirely.

## 3. The blocker: `Material` cannot express a metal

```python
@property
def epsr(self) -> float:
    return (self.n_core / self.n_clad) ** 2      # non-negative by construction
```

`n_core` is typed `float`, so `Re ε` is a square and can never be negative.
This is the *only* blocker. Everything downstream is already general:

- `Material.nc` reconstructs the index from `er` and `ei` with
  `√(½(−er + |ε|))` for the imaginary part, which is correct for `er < 0`;
- the kernels take `Material.nc` and `Material.eps` and never see `n_core`;
- `reference/mie.py` takes complex `m` throughout;
- complex wavenumbers already work everywhere (non-negotiable §1).

So this is a constructor problem, not a physics problem.

## 4. Recommended API

`Material` gains an optional absolute complex permittivity, and a classmethod
over it:

```python
@dataclass
class Material:
    n_core: float | None = None
    n_clad: float = 1.0
    pol: int = 2
    epsi: float = 0.0
    eps_core: complex | None = None    # absolute; wins over n_core when given

    @classmethod
    def from_eps(cls, eps: complex, n_clad: float = 1.0, pol: int = 2) -> "Material":
        ...
```

**`eps` is absolute**, matching `n_core` and `epsi`, which are both absolute
(conventions §2); `epsr` and `epsi_rel` keep dividing by `n_clad²`. Getting
this backwards is the single most likely way to implement it wrongly, so the
convention must be recorded in `docs/conventions.md` §2 in the same change
(CLAUDE.md's rule on pinning a new convention).

Two alternatives were considered and rejected:

- **Let `n_core` accept complex.** It reinterprets a field pinned in
  `tests/api_baseline.txt`, and `epsi` would then double-count against
  `Im(n_core²)`. Breaking, and it makes the absolute/relative convention harder
  to state rather than easier.
- **A separate `MetalMaterial` type.** More public surface, and every consumer
  — `BIESolver`, `QNMSolver`, `ClusterBIESolver`, `ScatterResult` — would need
  to accept both.

Cost: two new lines in `tests/api_baseline.txt`, one new field with a default,
no change to any existing signature. It is additive in the same sense the
v0.8 cluster work was.

## 5. Risks, measured rather than assumed

Two were checked while writing this, at `rad = 200 nm`, `λ = 600 nm`
(`x = 2.09`):

**Series truncation is adequate — checked.** `_nmax(x)` is the Wiscombe rule on
the *exterior* size parameter and carries no `|m|` dependence, which is a real
hazard when the interior argument is `m·x`. Measured: doubling and quadrupling
`n_max` changes `Q_abs` by **exactly zero** at all three material points above,
up to `|m|·x = 13.3` — the rule's `+10` buffer covers it. This holds *at these
parameters*; the rule's lack of an `|m|` term means it must be re-checked if
`rad/λ` or `|ε|` grows materially.

**No conditioning cliff at the plasmon condition — checked.** The quasi-static
cylinder plasmon sits at `ε = −ε_bg`. Scanning `Re ε` from −0.5 to −30 at
`Im ε = 0.5`, `Q_abs` varies smoothly with a gentle maximum near −1 and no
pole: at `x = 2.09` the resonance is off the quasi-static limit and damped by
the loss. A grid spanning `Re ε ∈ [−1, −30]` is therefore safe. **At smaller
`x`, or with much smaller `Im ε`, it would not be** — the operator becomes
near-singular there, and a tolerance must then reflect conditioning rather than
be widened to accommodate it.

**One risk left to measure at implementation time.** The boundary unknowns
oscillate at the interior wavelength `λ/|m|`, so `n_pts` must grow roughly
linearly with `|m|`: at `|m| = 6.33` the interior wavelength is ~95 nm against
a ~1257 nm perimeter. The required `n_pts` must be **measured the way the
star's 400 was** — by refining until the deviation saturates on the reference —
and never assumed from the circle's 200.

## 6. Out of scope

- Dispersion. A real metal is dispersive; these are fixed permittivities at a
  nominal wavelength. Nothing here is a physical silver cylinder, and
  `docs/conventions.md` §9 scale covariance depends on non-dispersion.
- Gain (`Im ε < 0`). Already documented as outside the validated scope in
  `Material.nc`.
- Metals in clusters, QNM extraction on metals, plasmonic mode tracking.
- Metals under the external solvers. Both drivers compute `eps_p` from a real
  `N_CORE` and share this constructor's limitation.
- Absorption on a non-circular shape (§1) — a run, not a design.

## 7. Acceptance criteria

1. `Material.from_eps` exists, is exported, and appears in
   `tests/api_baseline.txt`; every existing signature is unchanged.
2. `docs/conventions.md` §2 records that `from_eps` takes an **absolute**
   permittivity.
3. One test asserts `C_abs` against `reference/mie.py` at all three material
   points, both polarisations, with `n_pts` justified by a measured saturation
   and each tolerance citing that measurement (non-negotiable §4).
4. The test demonstrably fails if the sign of `Re ε` is flipped.

Estimated size: one commit for the API plus conventions, one for the test.
