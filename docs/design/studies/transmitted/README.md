# Transmitted-field prototypes (v1.1)

Throwaway prototypes behind [transmitted-spec.md](../../transmitted-spec.md) §3.
They are not package code and not lint-checked (`docs/design/studies` is
excluded from ruff). Each one runs in seconds, except `t3_far.py` (~20 s).

Run them from this directory, so the scripts can import `tproto`:

    uv run --project ../../../.. python t2_power.py

| Script | What it measures |
|---|---|
| `tproto.py` | Shared code: τ = T/α₁, pointwise G_t, the QUADPACK reference, the spectrum A(q, α₁), a_down, and the flux rule |
| `t1_green.py`, `t1b.py` | G_t vs `(i/4)H₀` at ε = 1, vs the reference, path vs path, and interface continuity |
| `t2_power.py` | Dipole power balance (G14 setup) per substrate, flux vs far field, 24 vs 48 nodes per panel |
| `t2b.py` | The `c_sca_up` critical-angle defect (spec finding 1) and its split fix; the lossless metal |
| `t2c.py` | Decade grading of the flux breakpoints (10³ vs 10⁶ widths) |
| `t3_far.py` | a_down vs the free-space far field at ε = 1; near-to-far consistency |
| `t3b.py` | **Not run yet**: the per-angle near-to-far error on glass (spec §7.1) |
