# Two-ring loss estimator: tutorial, assumptions and limits

How to turn linewidth measurements on two ring resonators into an estimate of
a change in material absorption, while cancelling the slow drift of the glass
cladding. The numbers come from the study outputs in this folder
(`s4_cladding_drift.txt`, `s4c_bus.txt`).

## 1. The idea

A ring resonator cuts a dip into the light passing along its waveguide. The
dip's width tells you how fast light dies in the ring. Absorption in the ring
material widens it. So does a change in the refractive index of the cladding,
which happens slowly as the glass ages. One ring cannot tell the two apart.

Two rings of different size can. Both sit on the same chip and see the same
absorption change and the same cladding drift. They respond to the two in
different proportions. Two readings, two unknowns: solve the pair of equations.

## 2. Building the estimator

**Step 1: measure each ring's intrinsic Q.** Fit each ring's transmission dip
and separate the ring's own loss (intrinsic Q, `Q_i`) from the share the
waveguide takes (coupling Q). Each ring has its own waveguide, because both
rings resonate at the same 1550 nm and their dips would overlap on a shared one.

**Step 2: take a reference and form fractional changes.** Record each ring's
`Q_i` at the start. At every later measurement compute

```
u = Q_i,reference / Q_i,now − 1
```

`u` is the fractional change in the ring's loss rate.

**Step 3: write the response of each ring.** For small changes each `u` is a
linear mix of the absorption change `δα` (dB/m) and the cladding index
change `δn`:

```
u_k = a_k · δα + b_k · δn
```

`a_k` and `b_k` are the ring's sensitivities, from simulation or calibration.

**Step 4: invert the 2×2 system.**

```
δα = ( b_2 · u_1 − b_1 · u_2 ) / ( a_1 · b_2 − a_2 · b_1 )
```

This is a weighted difference of the two readings, with the weights chosen so
that cladding drift cancels exactly.

### Coefficients for the two candidate designs

| Design | Ring | Diameter (µm) | Dip Q | `a` (per dB/m) | `b` (per unit index) |
|---|---|---|---|---|---|
| Current, 58 + 62 (with waveguide) | n = 58 | 15.91 | 448k | 0.02410 | 35.13 |
| | n = 62 | 16.94 | 894k | 0.04829 | 25.79 |
| Dip Q < 0.5M, 46 + 58 (isolated rings) | n = 46 | 12.81 | 24k | 0.00125 | 35.02 |
| | n = 58 | 15.91 | 436k | 0.02344 | 35.63 |

Resulting estimators:

```
58 + 62:  δα = 32.69 · u_62 − 24.00 · u_58     (dB/m)
46 + 58:  δα = 45.12 · u_58 − 45.91 · u_46     (dB/m)
```

In 46 + 58 the small ring barely senses absorption (`a` is 19 times smaller).
It acts as a drift reference, and the weights subtract its reading almost
one-for-one.

### Worked example (58 + 62)

True change: `δα` = 0.1 dB/m and `δn` = 5×10⁻⁵.

| | n = 58 | n = 62 |
|---|---|---|
| `Q_i` reference | 896,070 | 1,787,150 |
| `Q_i` now | 892,352 | 1,776,282 |
| `u` | 0.00417 | 0.00612 |

`δα = 32.69 × 0.00612 − 24.00 × 0.00417 = 0.100 dB/m`. The main ring alone,
ignoring drift, reads 0.00417 / 0.0241 = 0.173 dB/m, 73 % too high.

## 3. How precise it is

Each width measurement is assumed to repeat to 1 part in 1000. At critical
coupling this makes each `u` uncertain by 0.002. The error carries through the
weights:

```
σ(δα) = 0.002 × √(w_1² + w_2²)
```

| Design | σ(δα), drift free | σ(δα), drift known | Averages for 3σ on 0.1 dB/m | Condition number |
|---|---|---|---|---|
| 58 + 62 | 0.081 dB/m | 0.037 dB/m | 6 | 4.1 |
| 46 + 58 | 0.129 dB/m | 0.085 dB/m | 15 | 2.6 |

σ is one standard deviation. A 0.1 dB/m step is only 0.8 to 1.2 σ on a single
measurement, so detecting it reliably needs averaging, and averaging only helps
while the noise is random rather than drifting. The condition number (columns
scaled to unit length) is low for both: the inversion is well behaved.

A useful rule: with drift known, resolution ≈ 37,000 / (dip Q) dB/m at this
repeatability, almost independent of ring size. Lowering the allowed Q costs
resolution in direct proportion.

## 4. Assumptions

1. **Two unknowns only.** Absorption and cladding index are the only things
   that change. Anything else that changes one ring (contamination, surface
   loss, stress) shows up as false absorption.
2. **Common changes.** Both rings see the same absorption change and the same
   cladding drift. A gradient across the chip breaks this.
3. **Small, linear changes.** Sensitivities are evaluated at the starting
   point; valid while changes stay small (here ~0.5 % in loss rate).
4. **Fixed repeatability.** 1 in 1000 on each dip width, for every ring and Q,
   with all the error falling on the ring's own loss.
5. **Independent readings.** The two rings' errors are uncorrelated.
6. **Critical coupling.** Each ring's waveguide gap is set so that waveguide
   and ring lose light at the same rate (gaps ≈ 1058 nm for n = 58,
   1134 nm for n = 62).
7. **Temperature held.** The stage holds 0.1 K; the main ring needs a 3.6 K
   swing to fake a 0.1 dB/m step. Temperature is not an unknown in the solve.

## 5. Limits of the model

- **Idealised simulation.** Flat 2D top view, bare material indices (2.0 core,
  1.45 cladding). Real rings confine light less and must be larger; the
  coefficients `a`, `b` will differ. Use them for design, not for data:
  measure or recompute them for the real device.
- **Coefficient errors leak drift.** The weights cancel drift by subtraction.
  A 1 % error in a coefficient can shift `δα` by up to ~4 % of the change
  measured and lets some drift through.
- **No surface roughness.** Scattering from sidewall roughness, which often
  limits real Q, is absent.
- **Splitting the dip at critical coupling.** The study's fit uses the complex
  through amplitude. With power alone, dip depth is insensitive to the
  intrinsic/coupling split near critical coupling, and the side (under or over)
  is ambiguous. Measure phase, or calibrate the coupling separately.
- **Timing and shared equipment.** Measuring the rings one after the other
  needs the 0.1 K limit to hold over that interval. A shared laser makes the
  two errors correlated, which the precision figures ignore.
- **46 + 58 is less checked.** Its coefficients come from the isolated-ring
  formula; the waveguide simulation done for 58 + 62 (which moved σ from
  0.083 to 0.081 dB/m) has not been run for it.
