# Half-space Sommerfeld quadrature study (v1.1, step 2a)

A prototype run in isolation from the package (nothing under `src/` is imported
except `BIESolver` / `ClusterBIESolver` for the end-to-end check). Run from the
repo root with `VECLIB_MAXIMUM_THREADS=4 uv run python docs/design/studies/halfspace/<script>.py`.
The theory is in [../../sommerfeld-holomorphy.md](../../sommerfeld-holomorphy.md).

## Result

A single deformed path gives G_ind, ∂x′G_ind and ∂z′G_ind to 1e-13–1e-15 at
real and complex k (Im k < 0). This holds for lossless dielectric, lossy and
plasmonic substrates, against three independent anchors. The separable (GEMM)
form is exact to rounding. **The legacy Chebyshev surrogate is unnecessary.**

## Path and node rule (`hs.py`)

- **Branch:** `α(q) = i·√(i(q−k))·√(−i(q+k))`, the vertical-cut sheet, with
  `α(0) = +k`.
- **Folding:** the integral is folded to q ≥ 0 and mirrored, `q(−t) = −q(t)`.
- **T** = 1.5·max Re{k₁, k₂ (if |Im k₂| < 2 Re k₁), q_sp}.
- **Hump:** `q = t − iδ·sin(πt/T)` on [0, T], Gauss–Legendre with
  `n_hump = max(48, ⌈13·T/δ⌉)`.
- **Depth:** `δ = max(min(0.4·Re k₁, 6/D), 3·max(0, −Im of every feature))`.
  - The 6/D cap bounds the growth, measured as `e^{0.45·δD}`.
  - The floor keeps the hump below +k₁, +k₂ and q_sp.
- **Tail:** the real axis from T to T + 38/Z_min, in 24-point GL panels that
  start at width T/2 and double up to `min(4·2π/D, 8/Z_min)`. The grading is
  necessary; uniform panels stall at 1e-7 (`t1b.py`).
- **Banding** (`gemm.py`): tail panels are applied only to the nodes with
  `h_i + h_j < 38/q`. At R = 500, gap = 5, this cuts M from 7398 to 2406 at no
  cost in accuracy.

## Convergence (`t5_convergence.py`; `s` scales every node count)

| case | kZ_min | kD | s=0.5 | s=0.75 | s=1 |
|---|---|---|---|---|---|
| PEC 500 nm, R500 gap5 | 0.126 | 12.6 | 1e-9 | 4e-14 | 2e-14 |
| PEC 1600(1+0.05i) TM | 0.039 | 3.9 | 3e-9 | 2e-13 | 2e-14 |
| glass 500 TE, R100 gap5 | 0.126 | 2.5 | 4e-13 | 2e-14 | 1e-14 |
| Si 1600 TE, R500 gap20 | 0.157 | 3.9 | 3e-9 | 3e-13 | 1e-14 |
| Ag 633 TM, R500 gap5 | 0.099 | 9.9 | 2e-10 | 2e-14 | 5e-15 |
| Ag 633(1−i/6) TM, Q=3 | 0.098 | 9.8 | 6e-7 | 5e-10 | 5e-13 |
| Au 800(1+0.05i) TM | 0.078 | 7.8 | 1e-8 | 8e-13 | 2e-14 |

## Anchors

| anchor | script | result |
|---|---|---|
| PEC closed form ∓(i/4)H₀(k₁ρ_img), real and complex k | `t1_pec.py` | 2e-13 (real), 1e-14 (complex) |
| ε₂ = ε₁ gives R ≡ 0 | `t2_anchors.py` | exactly 0 |
| QUADPACK on the real axis, legacy substitution, separate branch rule | `refquad.py`, `t2_anchors.py` | 1e-15–1e-14; the reference's own floor is 2e-10 at oscillatory TE points |
| Complex k by Chebyshev continuation of the real-k reference (path-independent) | `t3_complex.py` | 1e-15–4e-14 at Q=10; 1e-11–3e-10 at Q=3 (interpolation noise) |
| Plasmon pole k₁√(ε₂/(ε₁+ε₂)) is a zero of the TM denominator on the sheet; clearance over the shadow box | `t4_plasmon.py` | 1e-15; clearance ≥ 0.24·abs(k₁) down to Q=3 |
| Legacy real-axis `Im α ≥ 0` at Im k < 0 equals the incoming −(i/4)H₀^{(2)} | `t7_legacy_regression.py` | 1e-16 (defect confirmed) |
| End-to-end: circle over PEC vs `ClusterBIESolver` particle + mirror | `t11_e2e.py`, `t11b_sym.py` | φ, χ and field to 3e-15–1e-13, TE/TM, real/complex λ, n_clad ∈ {1, 1.33} |

The end-to-end PEC check is close to an algebraic identity. It pins the block
and right-hand-side sign wiring. The Fresnel R is anchored pointwise by the
QUADPACK and continuation rows.

## Timing (`t9_timing.py`; ms per M1+M2 block pair, Ag TM, 4 threads)

| case | nn | assemble_matrix real k | assemble_matrix complex k | banded GEMM |
|---|---|---|---|---|
| R100 gap5 | 64/128/256 | 0.2/0.7/2.8 | 4.8/18.9/75 | 4.3/6.0/11 |
| R500 gap5 | 64/128/256 | 0.2/0.8/3.3 | 9.4/38/151 | 6.4/9.1/14 |
| R500 gap20 | 64/128/256 | 0.3/0.8/3.3 | 9.4/38/151 | 4.2/5.9/10 |

The banded GEMM time is the same at real and complex k to within 5 %. At complex k
the reflected blocks are about 10× cheaper than the free-space assembly.

## Traps

1. Uniform tail panels near q = T stall at 1e-7. Use geometric grading.
2. The hump node count must scale with T/δ (Si failed at 1e-5 otherwise), and δ
   must be capped at 6/D (R = 4000 gave an O(1) error otherwise).
3. **Centre x on the particle.** An origin 200 µm away overflows (`|Im q|·x > 709`
   gives NaN); 50 µm costs about a digit.
4. **Trapezoid on the image kernel:** no Kress correction is needed, but the image
   is near-singular at distance 2·gap, so nn is set by `spacing/(2·gap)`. Reaching
   1e-12 needs ≲ 0.25 (TE) or ≲ 0.2 (TM) (`t12_nn_gap.py`).
5. **The TM R_∞ image subtraction does not pay** in the matrix path: it saves 1.5–1.8×
   in nodes but costs nn² Hankel evaluations, and loses accuracy when the
   shortened tail starts below abs(k₂) (`t8_rinf.py`).

## Open

- ε₂ → −ε₁ (q_sp → ∞) is not probed.
- A flat facet lying on the interface (Z_min → 0 over a span) is not probed; the
  `M ∝ D/Z_min` scaling would dominate.
- The δ floor can override the 6/D cap at large kD and low Q (kD = 100, Q = 10
  loses about 3 digits).
