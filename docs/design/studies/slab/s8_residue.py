"""S8: mode normalisation and the residue identity  Res_β R = γ₁ u_m(z_top)² / (β_m N_m).

Derivation (README §S8): the 1D Green function of ∇·(p∇u) + k² w u = −δ has the
pole u_m(z)u_m(z')/((q² − β_m²)N_m), N_m = ∫ p u_m² dz; its reflected part in the
cover is (i/2)(R/α₁)e^{iα₁(h+h')} with α₁(β) = iγ₁ and u_m = u_m(z_top)e^{−γ₁h}.
Equating residues pins the normalisation, the 1/ε weight in TM and the sign.

Independence: the left side is a Cauchy circle on the stable recursion; the right
side is the (u, p u') transfer profile and a Gauss–Legendre norm, no R involved.
"""
import numpy as np

import sproto as S


def case(lam, n_clad, layers, n_bot):
    k = 2 * np.pi * n_clad / lam
    st = S.Stack([n**2 / n_clad**2 for n, _ in layers], [d for _, d in layers], n_bot**2 / n_clad**2)
    return k, st


cases = {
    "SOI 220 @1550": (1550, 1.444, [(3.476, 220)], 1.444),
    "air|Si 220|SiO2": (1550, 1.0, [(3.476, 220)], 1.444),
    "SiN 400 @800 (2 modes)": (800, 1.444, [(2.0, 400)], 1.444),
    "Si 2000 (9 modes)": (1550, 1.444, [(3.476, 2000)], 1.444),
    "Si/SiO2/Si coupled": (1550, 1.444, [(3.476, 220), (1.444, 300), (3.476, 220)], 1.444),
    "air|SiN 300|SiO2 near cutoff": (1550, 1.0, [(2.0, 300)], 1.444),
}
for name, (lam, nc, layers, nb) in cases.items():
    k, st = case(lam, nc, layers, nb)
    for pol in (2, 1):
        worst, worst_mm = 0.0, 0.0
        for b in S.guided_poles_gamma(st, k, pol):
            _, N, mm = S.mode_profile(st, k, pol, b)
            g1 = np.sqrt(b * b - k * k)
            cf = g1 / (b * N)
            lo = S.guided_window(st, k)[0]
            res = S.residue_R(st, k, pol, b, r=min(0.3 * (b - lo), 1e-3 * b))
            worst = max(worst, abs(res - cf) / abs(cf))
            worst_mm = max(worst_mm, mm)
        n = len(S.guided_poles_gamma(st, k, pol))
        print(f"{name:30s} {'TE' if pol == 2 else 'TM'}: {n} modes  max |Res_circle − γ₁/(βN)|/|·| = {worst:.1e}"
              f"   (profile bottom mismatch ≤ {worst_mm:.1e})")

print("\nwrong weights fail: TM with p = 1 instead of 1/ε, SOI 220")
k, st = case(1550, 1.444, [(3.476, 220)], 1.444)
b = S.guided_poles_gamma(st, k, 1)[0]
u, N, _ = S.mode_profile(st, k, 1, b)
h = np.linspace(-4000, 4000, 400001)
N_p1 = np.trapezoid(u(h) ** 2, h)  # unweighted ∫u² — the wrong norm
N_ok = np.trapezoid(u(h) ** 2 * np.where((h < 0) & (h > -220), 1 / st.eps[0], 1.0), h)
res = S.residue_R(st, k, 1, b)
g1 = np.sqrt(b * b - k * k)
print(f"  weighted N by trapezoid vs GL: {abs(N_ok - N) / abs(N):.1e};  "
      f"Res vs γ₁/(βN_weighted): {abs(res - g1 / (b * N_ok)) / abs(res):.1e};  "
      f"vs γ₁/(βN_unweighted): {abs(res - g1 / (b * N_p1)) / abs(res):.1e}")
