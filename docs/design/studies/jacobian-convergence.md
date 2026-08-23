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
