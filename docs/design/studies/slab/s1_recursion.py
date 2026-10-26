"""S1: the stable Stack recursion — reductions, energy, asymptote, and the naive form failing."""
import numpy as np

import sproto as S
from pysie2d import layered as L

lam, n_clad = 1550.0, 1.444
k = 2 * np.pi * n_clad / lam
e = lambda n: n**2 / n_clad**2
SI, AIR = e(3.476), e(1.0)
AG = (-129.0 + 3.3j) / n_clad**2  # Ag at 1550 nm, Johnson–Christy order of magnitude
qs = np.concatenate([np.linspace(0, 3 * k, 300)  # never exactly q = k₁: 0/0 there when ε = 1
                     , k * np.geomspace(3, 3000, 200)]).astype(complex)

print("(a) N = 0 recursion vs the package's fresnel_r / _tau_half_space (bitwise)")
for eps_b in (2.25, SI, AG):
    st = S.Stack([], [], eps_b)
    for pol in (1, 2):
        g = st._gammas(qs, k, pol)[0][0]
        r = L.fresnel_r(qs, k, eps_b, pol)
        t = st.tau(qs, k, pol)
        t0 = L._tau_half_space(qs, k, S._norm(eps_b), pol)
        print(f"  eps_b={eps_b:.4g} pol={pol}: R max|Δ| = {np.max(np.abs(g - r)):.1e}, "
              f"τ max|Δ| = {np.max(np.abs(t - t0)):.1e}")

print("(b) every layer and the bottom equal to the cover → R ≡ 0 exactly")
st = S.Stack([1.0, 1.0, 1.0], [100, 300, 50], 1.0)
for pol in (1, 2):
    print(f"  pol={pol}: max|R| = {np.max(np.abs(st.R(qs, k, pol))):.1e}, "
          f"max|α₁τ − e^{{iα₁·Σd}}| = "
          f"{np.max(np.abs(S.alpha(qs, k) * st.tau(qs, k, pol) - np.exp(1j * S.alpha(qs, k) * 450))):.1e}")

print("(c) lossless energy at propagating q: |R|² + Re(Y_b)/Y₁·|α₁τ|² = 1")
cases = {
    "SOI 220": S.Stack([SI], [220], 1.0),
    "Si/SiO2/Si": S.Stack([SI, 1.0, SI], [220, 300, 220], 1.0),
    "SiO2|Si 500|air": S.Stack([SI], [500], AIR),
}
for name, st in cases.items():
    _, kb = st.ks(k)
    qp = np.linspace(0, 0.999 * min(k, kb.real), 200).astype(complex)
    for pol in (1, 2):
        R = st.R(qp, k, pol)
        T = S.alpha(qp, k) * st.tau(qp, k, pol)
        Y1 = S.alpha(qp, k)
        Yb = S.alpha(qp, kb) / (1.0 if pol == 2 else st.eps_bot)
        err = np.abs(np.abs(R) ** 2 + (Yb.real / Y1.real) * np.abs(T) ** 2 - 1)
        print(f"  {name:28s} pol={pol}: max err = {err.max():.1e}")

print("(d) stable vs naive transfer matrix, and the large-q asymptote R → r(cover|layer 1)")
for name, st in {"SOI 220": S.Stack([SI], [220], 1.0),
                 "Si 2000": S.Stack([SI], [2000], 1.0),
                 "Ag 500 film": S.Stack([AG], [500], 1.0)}.items():
    top = S.Stack([], [], st.eps[0])
    for pol in (2, 1):
        for qk in (0.5, 2.0, 10.0, 50.0, 200.0):
            q = np.array([qk * k], dtype=complex)
            Rs, Rn = st.R(q, k, pol)[0], st.R_naive(q, k, pol)[0]
            a1 = S.alpha_int(q, k * np.sqrt(st.eps[0]))[0]
            tail = abs(np.exp(2j * a1 * st.d[0]))
            print(f"  {name:12s} pol={pol} q={qk:6.1f}k: |Rs−Rn| = {abs(Rs - Rn):.1e}  "
                  f"|Rs − r_top| = {abs(Rs - top.R(q, k, pol)[0]):.1e}  e^(-2|Imα|d) = {tail:.1e}")

print("(e) complex q on a hump and complex k (Q = 10): stable vs naive")
kc = k * (1 - 0.05j)
for name, st in {"SOI 220": S.Stack([SI], [220], 1.0), "Ag 500 film": S.Stack([AG], [500], 1.0)}.items():
    t = np.linspace(0, 6 * k, 400)
    qh = t - 1j * 0.4 * k * np.sin(np.pi * t / (6 * k))
    for pol in (2, 1):
        d = np.abs(st.R(qh, kc, pol) - st.R_naive(qh, kc, pol))
        print(f"  {name:12s} pol={pol}: max|Rs − Rn| on the hump = {d.max():.1e}")

print("(f) the naive form overflows on a real path: Σ|Im α_j|·d_j > 709 on the tail")
# tail end q_max = T + 38/z_min (layered.TAIL_DECAY); T ≈ 1.5·β_max ≲ 1.5·k_core
for name, st in {"SOI 220": S.Stack([SI], [220], 1.0),
                 "SiN 400": S.Stack([e(2.0)], [400], 1.0),
                 "Si/SiO2/Si": S.Stack([SI, 1.0, SI], [220, 300, 220], 1.0)}.items():
    for gap in (100.0, 20.0, 5.0):
        z_min = 2 * gap
        q_max = 1.5 * k * np.sqrt(max(np.real(st.eps))) + L.TAIL_DECAY / z_min
        T = 1.5 * k * np.sqrt(max(np.real(st.eps)))  # the tail: no pole on it
        q = np.linspace(T, q_max, 4000).astype(complex)
        with np.errstate(all="ignore"):
            Rn = st.R_naive(q, k, 2)
        Rs = st.R(q, k, 2)
        bad = ~np.isfinite(Rn)
        q_fail = q[bad][0].real if bad.any() else np.nan
        err = np.max(np.abs(Rs[~bad] - Rn[~bad]) / np.abs(Rs[~bad]))
        print(f"  {name:11s} gap={gap:5.1f}: q_max·Σd = {q_max * st.thickness:7.0f}  "
              f"naive NaN from q·Σd = {q_fail * st.thickness:6.0f}  "
              f"stable finite: {np.all(np.isfinite(Rs))}  max rel Δ where finite = {err:.1e}")
