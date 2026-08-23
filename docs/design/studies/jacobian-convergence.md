# Gate 10: convergence of the sensitivity Jacobian in R

Script: `docs/design/studies/gate10_jacobian_convergence.py`. Design point: the
Gate-2/3 ellipse — `m = 4`, `n1 = n2 = n3 = 2`, `a = 1.0`, `b = 1.2`,
`rad = 200 nm`, `n_core = 3.0`, `n_clad = 1.0`, TE — and its single mode in the
box `543+18j … 560+32j`. Ladder in `R = wavelength_over_ds` (D17), not in
`n_pts`.

## The rungs

| `R` target | `n_pts` | `R` achieved | λ (nm) |
|---|---|---|---|
| 15 | 115 | 15.25 | 551.136315+24.743434j |
| 30 | 228 | 30.26 | 551.440442+24.663061j |
| 50 | 378 | 50.17 | 551.561402+24.629257j |

## Movement of each J component against the `R = 50` rung

| `R` | `dλ/db` | `dλ/da` | `dλ/drad` | `dλ/dn_core` |
|---|---|---|---|---|
| 15 | 3.364e-2 | 4.871e-3 | 7.972e-4 | 1.239e-2 |
| 30 | 9.749e-3 | 1.409e-3 | 2.275e-4 | 3.576e-3 |

## Observed order, and the distance still to run

Three rungs at unequal `n_pts` give the order by bisection on
`(n₀^-p − n₁^-p)/(n₁^-p − n₂^-p)`, and then a Richardson limit from the two
finest:

| component | `p` | \|J(R=50) − limit\|/\|limit\| |
|---|---|---|
| `dλ/db` | 0.98 | 1.531e-2 |
| `dλ/da` | 0.99 | 2.172e-3 |
| `dλ/drad` | 1.02 | 3.376e-4 |
| `dλ/dn_core` | 0.99 | 5.461e-3 |

**J converges at first order in `n_pts`, exactly like λ itself.** That is the
result, and it is a negative one: `conventions.md` §9 argues that the fixed
`x_disc(n_pts) ≠ x_Mie` error is smooth in the shape parameter and therefore
largely cancels in a ratio. The cancellation is real in *magnitude* — `dλ/drad`
is three decades better resolved than `dλ/db` at the same `R` — but it does not
raise the *order*. Nothing here converges faster than 1/`n_pts`.

## Verdict against the stated tolerance

Tolerance set in advance: **1 % relative on every J component.** The reason: the
OED objective is quadratic in J, so 1 % on J is 2 % on the Fisher matrix, which
is below the uncertainty the noise model and the D6a acceptance fraction
contribute to any design ranking.

**Not met at any rung measured.** The worst component, `dλ/db`, is 1.5 % from
its Richardson limit at `R = 50` (`n_pts = 378`). At first order, 1 % needs
`n_pts ≈ 580` (`R ≈ 77`) and 0.5 % needs `n_pts ≈ 1160`; assembly cost goes as
`n_pts²`, so that is 2.3× and 9.4× the `R = 50` point respectively.

**Gate 10 stays open**, with two ways forward that are a decision, not a
measurement:

1. **Richardson-extrapolate J** from two rungs. The order is measured at 1.00
   to within 2 %, so the extrapolation is well founded, and two rungs at
   `R = 30` and `R = 50` cost 1.4× the fine rung alone while cutting the
   residual by roughly the rung ratio. Cheapest route to the stated 1 %.
2. **Restate the criterion on the quantity the OED actually uses.** Design
   ranking depends on J *differences between nearby designs*, where the
   common-mode part of this first-order error cancels a second time. That
   cancellation is plausible on the same §9 grounds — and §9's argument is
   exactly what the order measurement above has just shown to be weaker than
   assumed, so it would have to be measured, not asserted.

Either way the compute-budget consequence is recorded: the earlier hope that
`n_pts = 200` would serve does not survive contact with a 1 % requirement on J.

## Route 2 measured: ΔJ between nearby designs does **not** converge faster

Script: `docs/design/studies/gate10_jacobian_differences.py`, same design point
and same ladder, with a second design at `b = 1.22` (`δb = 0.02` — it moves λ by
~1 nm, a hundred times the `R = 50` pole error, so `ΔJ` is not itself a
cancellation artefact, while still being the scale at which two catalogue
designs get compared). Both designs are laddered at the **same** `n_pts`: sizing
each from its own `R` target would let the two differ by a point or two, and
that difference is of the order of the error under test, so the sizing rule
rather than the physics would decide the answer.

Pass criterion fixed before the run: route 2 is viable if `ΔJ` converges at an
order above 1, **or** its residual at `R = 50` lands at least 10× below `J`'s
own 1.5e-2.

| component | `p(J)` | res(`J`) | `p(ΔJ)` | res(`ΔJ`) |
|---|---|---|---|---|
| `dλ/db` | 0.98 | 1.531e-2 | 0.94 | 4.611e-2 |
| `dλ/da` | 0.99 | 2.172e-3 | 0.94 | 4.611e-2 |
| `dλ/drad` | 1.02 | 3.376e-4 | 0.97 | 1.425e-2 |
| `dλ/dn_core` | 0.99 | 5.461e-3 | 0.94 | 6.352e-1 |

**Neither half is met, and the miss is not marginal.** The order is 0.94–0.97 —
first order, like `J` and like λ — and every residual is *worse* than `J`'s own,
by 3× on `dλ/db` and by 116× on `dλ/dn_core`. Differencing subtracts two
quantities whose errors are the same size and only partly common-mode, so the
difference keeps the error while losing the signal: exactly the behaviour of a
cancellation that acts on the constant and not on the order, which is what the
first script already found. §9's smoothness argument does not buy a second
cancellation any more than it bought the first.

`dλ/da` and `dλ/db` return the same relative residual to four digits because
their `ΔJ` sequences are proportional: `ΔJ(a)/ΔJ(b) = −1.209 ≈ −b/a` at both
rungs. That is D15 showing through — at `n2 = n3` the combination
`a ∂_a + b ∂_b` is a pure dilation, so a perturbation in `b` moves the two
components along that one direction. It is a corroboration of Gate 2 through an
unrelated quantity, not a coincidence and not a bug.

**Route 2 is therefore closed off.** Gate 10 has one route left, and it is
Richardson extrapolation of `J` from two rungs (route 1), whose premise — the
order — is measured at 1.00 ± 2 % here for a third time.
