# What holding `R` fixed costs across the shape family (D17, A13)

Script: `docs/design/studies/r_band_policy.py`. `conventions.md` §12 places
Jacobian rungs in `R = wavelength_over_ds` rather than raw `n_pts`, and §12's
production pair is `R = 15 + 30`. That settles what to hold fixed. This measures
what holding it fixed costs, which is the number a catalogue budget is built
from.

**Not a sampling region.** The catalogue's numeric bounds are catalogue-side and
not yet fixed (backlog B5), so this is a *spanning set*: the circle, elongation
at fixed `n1`, and both directions `n1` moves the boundary off an ellipse.

`rad = 200 nm`, `n_core = 3`, `λ = 551.4 nm`. Cost is `n(15)² + n(30)²`
normalised to the circle, since assembly is O(`n_pts`²) and the pair pays for
both rungs.

| shape | `R` at `n_pts = 200` | `n(R=15)` | `n(R=30)` | cost |
|---|---|---|---|---|
| circle | 29.25 | 104 | 206 | 1.00× |
| ellipse `b/a = 1.2` | 26.54 | 114 | 228 | 1.22× |
| ellipse `b/a = 2` | 18.97 | 160 | 318 | 2.38× |
| ellipse `b/a = 3` | 13.75 | 220 | 438 | 4.51× |
| `n1 = 1.5`, `b/a = 1.2` | 25.58 | 118 | 236 | 1.31× |
| `n1 = 4`, `b/a = 1.2` | 27.94 | 108 | 216 | 1.10× |

**Elongation is the whole cost story; `n1` is not.** Aspect ratio 3 costs 4.5×
the circle, while moving `n1` from 1.5 to 4 at fixed aspect spans only
1.10×–1.31×. A budget therefore has to be sized on the region's *aspect* bound
and can treat the `n1` direction as free. This also says where the R policy
earns its keep: at `n_pts = 200` the same nominal resolution reads 29.25 on the
circle and 13.75 at aspect 3, so a fixed-`n_pts` catalogue would silently be
running the elongated designs at half the resolution of the round ones —
precisely where the Jacobian is largest.

**`R` is λ-dependent and a quoted `R` is meaningless without its λ.** The
`R = 37.1` circle figure recorded against D17 was taken at a different
wavelength than the 29.25 here. `wavelength_over_ds` reads points per *interior*
wavelength, so it moves with λ and with `n_core`; the ladder scripts avoid this
by recomputing `R` at each rung's own λ rather than carrying a table.

**Not measured here:** whether the §12 two-rung extrapolation holds its 6.4e-4
at the elongated end of the family. The first-order premise is a property of the
discretisation and there is no reason for it to fail, but that is an argument,
not a check — backlog unit **A17**.
