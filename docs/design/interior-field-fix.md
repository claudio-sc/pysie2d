# Interior-field evaluation: required fix

**Status:** applied — shipped in v0.8.1 (PR #21, `b07168f`). Kept as the
record of the defect. Found 24 Sep 2026 while mapping near
fields of elongated ellipses (the slab-membrane study,
[slab-scoping/](slab-scoping/)).

## The defect

Every path that evaluates the field **inside** a particle applies the exterior
representation formula with the wavenumber changed to `k_core = nc·k_bg`, and
changes nothing else:

- `fields.eval_field` (the `ri=` branch), and through it `ScatterResult.eval_field`;
- `ClusterScatterResult.eval_field` (the `_representation_at` call at `k_core`).

That is wrong in two ways:

1. **Sign.** The boundary normal points *out* of the interior region. The
   representation integral over the boundary of the interior region carries the
   opposite sign to the exterior one:

       ψ_out(r) = ψ_inc(r) + (i/4)∮[ φ ∂H₀(k_bg R)/∂n' − H₀(k_bg R) χ ] ds'
       ψ_in(r)  =          − (i/4)∮[ φ ∂H₀(k_core R)/∂n' − H₀(k_core R) χ_in ] ds'

2. **TM normal derivative.** `χ` in the solution vector is the exterior-side
   normal derivative (conventions §4). The interface condition is continuity of
   `(1/ε^p)·∂ψ/∂n`, where p = 0 for TE and p = 1 for TM. The interior-side
   derivative is therefore

       χ_in = χ          (TE, pol = 2)
       χ_in = eps · χ    (TM, pol = 1), eps = Material.eps (background-relative)

   This is the same factor `eta = eps` that `assemble_matrix` already puts on
   M4.

The returned interior values are not a rescaling of the true field: for TM,
the φ and χ terms are weighted wrongly relative to each other. The sign alone
is enough to break continuity across the boundary.

## Evidence

A circle with rad = 300 nm, n_core = 2.0, n_clad = 1.45, λ = 633 nm, nn = 200,
and plane-wave incidence. Split the interior representation into its φ part
`a` and its χ part `b`, both evaluated at `k_core`. Then least-squares fit the
exact Mie interior field `Σ c_n J_n(k_core r) e^{inθ}` at three interior points
as `α·a + β·b`:

| pol | α | β | residual |
|---|---|---|---|
| 2 (TE) | −1.0000 | −1.0000 | 1.6e-31 |
| 1 (TM) | −1.0000 | −1.9025 (= −eps) | 3.2e-31 |

The current code amounts to α = β = +1. The fitted coefficients show that the
correct values are α = −1 and β = −1 (TE) or β = −eps (TM), with round-off
residual.

## Why no test caught it

No test checks an interior value. `test_cluster.py` asserts only that interior
values are finite and non-zero. Every other `eval_field` test is exterior:
multipoles, self-Green, and the field RMS.

## Fix

In `fields.eval_field`, inside the point loop:

```python
k = wnum_bg if outside else wnum_core
chi = ei[nn:] if outside else eta_in * ei[nn:]
sign = 1.0 if outside else -1.0
...
field[j] = sign * (1j / 4.0) * sum_h   # with chi in place of ei[nn:]
```

`eta_in` is 1 for TE and `eps` for TM. The primitive does not know `pol`
today, so it needs a new keyword, for example `eta_in: complex = 1.0`. It
belongs next to `ri`: pass both, or neither. `ScatterResult.eval_field` passes
`eta_in=mat.eps if mat.pol == 1 else 1.0`. `ClusterScatterResult.eval_field`
does the same per particle: it negates the `_representation_at` result and
scales the χ half of `ei_particle(p)`. Adding a keyword is additive for the API
baseline.

Record in `conventions.md` §4 that χ is the exterior-side derivative. The
interior side is `eps·χ` in TM.

## Validation anchor

Analytic Mie interior field, independent of the BIE:

- `c_n^TE = (−i)^n [J_n(x)H_n'(x) − J_n'(x)H_n(x)] / [J_n(mx)H_n'(x) − m J_n'(mx)H_n(x)]`
- TM: the same with `m J_n'(mx)` replaced by `J_n'(mx)/m`

Here `x = k_bg·rad` and `m = nc`, for incidence `exp(−i k_bg z)`, the
direction `plane_wave_rhs` uses at angle 0.

Tests:

1. Interior field vs Mie, both polarisations, at points ≥ 5 spacings from
   the boundary. Tolerance at round-off × the matrix condition number, since
   Kress is spectral on the circle (CLAUDE.md, non-negotiable 4).
2. Continuity across the boundary: total field just outside vs just inside, at
   ±5 spacings, for a lossy particle in TM. A sign-only fix would still fail this
   test in TM, and it cannot pass by accident.
3. The cluster variant: one particle far from the other, compared with test 1.

## Knock-on

- `examples/nearfield_map.py` (a README figure) fills the interior through
  `ri`, so the published interior is wrong. Regenerate it after the fix. Note
  also that `ScatterResult.eval_field` returns the **scattered** field outside
  and the **total** field inside. Either document that mixed quantity for the
  map, or add the incident field outside.
- The fix changes numbers, so it is a `fix:` commit and a release.
