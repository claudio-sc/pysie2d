# Spike: periodic (Bloch) kernel and a quasi-BIC grating cell

**Status: parked, 6 Oct 2026.** This is a sandbox and not part of the package. Nothing
here is imported by `pysie2d` or run by CI. It was built on v0.8.1 (`5da27b8`).

## Question

Can pysie2d's free-space BIE be turned into a 1-D periodic solver by adding the
smooth lattice remainder of the Green function, without touching the
Kress-corrected self term? And once it can, does a two-tooth cell (a quasi-BIC)
make a good test structure for measuring material loss ε″?

## Answer so far

- **Kernel.** `periodic_kernel.py` adds ΔG = (i/4)(S⁺ + S⁻) on top of the
  n = 0 free-space kernel, using Maradudin (2020) eqs. 1.87/1.92 (Sommerfeld form).
  It is checked against the damped direct lattice sum, Bloch folding and
  R + T = 1 in `check_kernel.py`. The energy residual is R + T − 1 = 8e-10 at
  `nn = 144`, against 1.6e-6 at 96.
- **Quasi-BIC** (`qbic.py`, `sweep.txt`). With δw = 0 the mode is bound:
  Q = 8e9, the round-off floor. With δw ≠ 0 the radiative width grows as δw².
  Im λ is linear in ε″: 0.317 → 0.635 nm as ε″ goes 0.01 → 0.02.
  `kb_sweep.txt` gives the band curvature away from Γ.
- **Cramér–Rao** (`spectra.py`). Inputs: zeroth-order R/T, 1 % noise, ε′ as a
  nuisance parameter. σ(ε″) is smallest at critical coupling (A_max ≈ 0.5), and
  fitting ε′ costs almost nothing.
- **1550 nm SiN port** (`sin.py`, `sin_spectra.py`). LPCVD SiN teeth in SiO₂,
  with ε″ = 1.1e-6 (0.1 dB/cm). Critical coupling is at δw ≈ 1.07 nm, and there
  σ(ε″) ≈ 1.7e-9, i.e. about 0.15 % of ε″. Linewidths are ~1e-4 nm, so each
  pole comes from Beyn rather than from interpolation. The FD step choice is
  explained in the `sin_spectra.py` docstring.

## Known limits (why it is not a feature)

- **No validation anchor that is independent of this repo.** Every check above
  is internal (direct sum, folding, energy). Following CLAUDE.md rule 3, a
  periodic feature would need e.g. a flat-grating / RCWA closed form first.
- Rayleigh anomalies (q± → 1) are not handled. At complex k the Sommerfeld
  integral converges only while Re k·(L − |Δx|) > |Δz|·Im k, and this is asserted.
- The background is homogeneous: there is no substrate, because the half-space
  background (v0.9) does not exist yet. A real SiN grating sits on a substrate.

## Picking it up

Run from this directory: `uv run python check_kernel.py`, then `qbic.py` /
`sin.py` with the arguments described in their `__main__` blocks. Each Beyn
call takes ~80 s at `nn = 144`. The `*.txt` / `*.json` files are the recorded
outputs of those runs. `sin_page_data.json` is a reduced copy of
`sin_spectra.json` (per δw: pole, A, R, CRB), made for plotting. No script in
this directory writes it.

Next steps, if this is resumed: (1) an external anchor for the periodic kernel;
(2) combine with the v0.9 half-space Green function to put a substrate under
the grating; (3) decide whether this becomes a `PeriodicCluster` in the
library or stays a study.
