"""S2 (anchor A2): guided poles of the stable R = roots of the closed-form slab dispersion.

Three independent things must agree: the bracketed real roots of the
transfer-matrix D, the textbook κd = mπ + atan + atan roots, and the count by the
argument principle on D. Then the stable R must actually have a simple pole
there: |1/R| at round-off, D'(β) ≠ 0, residue finite by a Cauchy circle.
"""
import numpy as np

import sproto as S


def case(lam, n_clad, layers, n_bot):
    k = 2 * np.pi * n_clad / lam
    st = S.Stack([n**2 / n_clad**2 for n, _ in layers], [d for _, d in layers], n_bot**2 / n_clad**2)
    return k, st


cases = {
    "SOI 220 @1550 (sym)": (1550, 1.444, [(3.476, 220)], 1.444),
    "air|Si 220|SiO2 @1550": (1550, 1.0, [(3.476, 220)], 1.444),
    "SiN 400 @800 (sym)": (800, 1.444, [(2.0, 400)], 1.444),
    "Si 2000 @1550 (multimode)": (1550, 1.444, [(3.476, 2000)], 1.444),
    "air|SiN 300|SiO2 @1550 (near cutoff)": (1550, 1.0, [(2.0, 300)], 1.444),
}

print("three-layer slabs: D roots vs closed form vs winding count")
for name, (lam, nc, layers, nb) in cases.items():
    k, st = case(lam, nc, layers, nb)
    for pol in (2, 1):
        roots = S.guided_poles(st, k, pol)
        cf = S.slab3_closed_form(k, 1.0, st.eps[0], st.eps_bot, st.d[0], pol)
        wn = S.winding_count(st, k, pol)
        lo, hi = S.guided_window(st, k)
        dev = max([abs(a - b) / b for a, b in zip(roots, cf)] + [0.0])
        print(f"  {name:38s} {'TE' if pol == 2 else 'TM'}: n_D={len(roots)} n_cf={len(cf)} "
              f"winding={wn}  max rel |β_D − β_cf| = {dev:.1e}")
        for m, b in enumerate(roots):
            neff = b / (2 * np.pi / lam)
            inv = abs(1.0 / st.R(np.array([b + 0j]), k, pol)[0])
            dd = S.dD(st, k, pol, b)
            res = S.residue_R(st, k, pol, b)
            gap = (b - lo) / k  # distance to the nearest outer branch point
            print(f"      m={m}: n_eff={neff:.10f}  |1/R(β)|={inv:.1e}  |D'(β)|·k={abs(dd) * k:.2e}  "
                  f"|Res R|/k={abs(res) / k:.3e}  (β−k_out)/k={gap:.2e}  |Im D/Re D'|={abs(st.D(np.array([b+0j]),k,pol)[0].imag/dd.real) / k:.1e}")

print("coupled double core (no closed form): D roots vs winding count, simple poles")
k, st = case(1550, 1.444, [(3.476, 220), (1.444, 300), (3.476, 220)], 1.444)
for pol in (2, 1):
    roots = S.guided_poles(st, k, pol)
    print(f"  Si/SiO2 300/Si {'TE' if pol == 2 else 'TM'}: n_D={len(roots)} winding={S.winding_count(st, k, pol)}  "
          f"n_eff = {[f'{b / (2 * np.pi / 1550):.8f}' for b in roots]}  "
          f"min |D'|·k = {min(abs(S.dD(st, k, pol, b)) for b in roots) * k:.2e}")

print("Cauchy-circle residue vs N(β)/D'(β) from the transfer matrix (two routes)")
k, st = case(1550, 1.444, [(3.476, 220)], 1.444)
for pol in (2, 1):
    b = S.guided_poles(st, k, pol)[0]
    num = st._tm(np.array([b + 0j]), k, pol)[0][0]
    r1 = num / S.dD(st, k, pol, b)
    for rad in (1e-2, 1e-3, 1e-4):
        r2 = S.residue_R(st, k, pol, b, r=rad * b)
        print(f"  {'TE' if pol == 2 else 'TM'} r={rad:.0e}·β: |Res_circle − N/D'|/|Res| = {abs(r2 - r1) / abs(r1):.1e}")
