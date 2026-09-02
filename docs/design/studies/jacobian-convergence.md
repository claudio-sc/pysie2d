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

## Route 1 measured: two-rung Richardson reaches 1 % at 0.46 assemblies

Script: `docs/design/studies/gate10_richardson.py`. The estimator is

    J* = J_f + (J_f − J_c) / ((n_f / n_c) − 1)

with the exponent **pinned at `p = 1`, not fitted**. Fitting needs a third rung,
costs 1.7× more, and buys a fitted exponent whose conditioning is poor when the
two differences are close — while the exponent itself is now measured at
1.00 ± 2 % by three independent ladders.

Scoring against the finest rung a candidate was built from would be circular, so
the ladder carries two rungs above the working band, `R = 75` and `R = 100`
(`n_pts` 567 and 755), which no scored candidate uses. Every candidate is scored
against the `R = 75/100` Richardson limit. The raw ladder underneath is clean
first order — `dλ/db` = 53.3036, 52.3359, 51.9360, 51.7314, 51.6290 at
`n_pts` = 115, 228, 378, 567, 755, successive differences −0.968, −0.400,
−0.205, −0.102, halving exactly as `1/n` requires.

Relative error against that limit; cost in units of one `R = 50` assembly
(assembly is O(`n_pts`²), so a rung costs `(n/n₅₀)²`):

| estimator | cost | `dλ/db` | `dλ/da` | `dλ/drad` | `dλ/dn_core` |
|---|---|---|---|---|---|
| raw rung `R = 15` | 0.09 | 4.910e-2 | 7.009e-3 | 1.140e-3 | 1.777e-2 |
| raw rung `R = 30` | 0.36 | 2.494e-2 | 3.554e-3 | 5.709e-4 | 8.989e-3 |
| raw rung `R = 50` | 1.00 | 1.507e-2 | 2.147e-3 | 3.435e-4 | 5.429e-3 |
| **Richardson 15+30** | **0.46** | **6.381e-4** | 7.242e-5 | 1.370e-5 | 2.360e-4 |
| Richardson 15+50 | 1.09 | 3.708e-4 | 4.217e-5 | 7.722e-6 | 1.366e-4 |
| Richardson 30+50 | 1.36 | 1.694e-4 | 1.939e-5 | 3.223e-6 | 6.178e-5 |

**Richardson from `R = 15 + 30` meets the 1 % bar with 15× margin, at 0.46× the
cost of a single `R = 50` rung** — which does not meet it at all, missing by
1.5 %. Extrapolation buys about two decades of accuracy for less than half the
price of the rung it replaces, because the cost is dominated by the finer of the
two rungs and `R = 30` is 2.8× cheaper than `R = 50`. `R = 30 + 50` is available
for another 4× of margin at 3× the cost, and is the fallback if a catalogue
region turns out to be less well behaved than this design point.

**A trap worth naming.** The first run of this script had the Richardson
denominator inverted, `(n_c/n_f) − 1` for `(n_f/n_c) − 1`, and the tell was not
a crash but a *non-monotone score column*: raw `R = 30` scored 50× better than
raw `R = 50` against the reference. A reference built by the same broken
estimator moves the target rather than raising an error. Any score table like
this one must be read down the raw-rung rows first — they have to improve
monotonically with `R`, and if they do not, the reference is wrong, not the
rungs.

## A17 — the §12 extrapolation at the elongated end of the family

`a17_richardson_elongated.py`. Gate 10 closed on a measurement at one design
point, the aspect-1.2 ellipse; the backlog recorded the extension to the rest of
the shape family as an argument, not a check. This is the check, run at
`b/a = 3` — the direction A13 measured as the whole cost story (aspect 3 costs
4.51× the circle, while `n1` over its full declared range spans 1.10–1.31×).

Design point `b = 3.0`, `a = 1.0`, everything else as the reference ellipse.
Single pole at 549.871 + 18.299j, box `540 + 10j … 580 + 30j`. Same estimator,
same scoring against an `R = 75/100` limit built from rungs no candidate uses,
same pre-stated **1 % on every J component**.

| estimator | cost (R=50 assemblies) | `dλ/db` | `dλ/da` | `dλ/drad` | `dλ/dn_core` |
|---|---|---|---|---|---|
| raw rung `R = 15` | 0.09 | 1.065e-2 | 1.732e-3 | 6.183e-4 | 6.353e-3 |
| raw rung `R = 30` | 0.36 | 5.245e-3 | 8.528e-4 | 3.053e-4 | 3.157e-3 |
| raw rung `R = 50` | 1.00 | 3.130e-3 | 5.090e-4 | 1.824e-4 | 1.890e-3 |
| **Richardson `15 + 30`** | **0.45** | **2.256e-4** | 3.282e-5 | 9.096e-6 | 8.734e-5 |
| Richardson `30 + 50` | 1.36 | 5.857e-5 | 8.617e-6 | 2.437e-6 | 2.192e-5 |

**A17 passes, and by more margin than the design point the gate closed on**:
worst component **2.256e-4** against the 6.381e-4 recorded at aspect 1.2. The
raw rungs read down monotonically and halve — 1.065e-2 → 5.245e-3 → 3.130e-3 —
which is the first-order premise holding at aspect 3 and, per the S05 trap, the
first column to read before trusting any score in the table.

Two things worth carrying forward.

**Holding `R` fixed does what D17 said it would.** At aspect 3 the *raw*
`R = 50` rung is 3.1e-3, five times better than the 1.5e-2 the same rung gives
at aspect 1.2, and it meets the 1 % bar unaided. Elongation costs assemblies
(A13) but does not cost accuracy at fixed `R`; stating the criterion in `R`
rather than in `n_pts` is what makes that true.

**The contour settings do not transfer, and this is the one to watch.** The
reference study's `n_quad_per_side = 6` fails Beyn's cancellation check at
**every** rung here — R = 15, 30, 50, 75 and 100 alike, so it is the contour
that is under-resolved at this aspect and not the boundary. Raised to 12 with
`n_probe = 20` for the whole ladder, so no comparison in the table is between
settings. It **raised** rather than returning a wrong pole
(`cancellation 2.57e-2 > 1e-4`, naming the saturated probe and what to do), which
is why this was safe to meet at run time. A catalogue sweep over elongated
shapes must expect to size the contour per design, not once.
