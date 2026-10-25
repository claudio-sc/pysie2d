"""S9 (anchor A4): line-dipole emission into guided modes, two closed forms.

(a) Far along the stack the total G (free + reflected, on the path) tends to the
    guided sum  Σ_m (i/(2β_m N_m)) u_m(z) u_m(z₀) e^{iβ_m|X|}: the residue of g at
    +β, with the contour closed upward for X > 0. Everything else (radiation at
    grazing, lateral waves) must decay, like |X|^{-3/2}. This pins the mode
    amplitude, its normalisation and the phase reference e^{iβX} at the source.
(b) Dipole power balance as an equality, every term from a different integral:
        Im G(r₀, r₀) = 1/4 + Im G_refl            (deformed path, at X = 0)
                     = P_up + P_down + P_guided
    P_up   = (1/π)∫₀^{π/2} |e^{−iα₁h₀} + R e^{iα₁h₀}|²/4 dθ       (real axis, q = k sinθ)
    P_down = (1/π)∫₀^{k_b} Re(Y_b)|τ|²|e^{iα₁h₀}|²/4 dq           (real axis, τ of S1)
    P_guided = Σ_m u_m(z₀)²/(2β_m N_m)                           (profile and norm of S8)
    For the line source of (∇² + k²)G = −δ, the power per unit flux normalisation
    is Im G (free space: 1/4); the guided term is the iπ δ(q − β) part of Im g.
"""
import numpy as np
from scipy.integrate import quad
from scipy.special import hankel1

import sproto as S


def case(lam, n_clad, layers, n_bot):
    k = 2 * np.pi * n_clad / lam
    st = S.Stack([n**2 / n_clad**2 for n, _ in layers], [d for _, d in layers], n_bot**2 / n_clad**2)
    return k, st


cases = {
    "SOI 220 @1550": (1550, 1.444, [(3.476, 220)], 1.444),
    "air|Si 220|SiO2": (1550, 1.0, [(3.476, 220)], 1.444),
    "SiN 400 @800 (2 modes)": (800, 1.444, [(2.0, 400)], 1.444),
    "Si/SiO2/Si coupled": (1550, 1.444, [(3.476, 220), (1.444, 300), (3.476, 220)], 1.444),
}

print("(a) |G_total − guided sum| / Σ|guided terms| along the stack (source h₀ = 50, observer h = 100 nm)")
# Σ|terms|, not |Σ|: two supermodes beat, and |Σ| has near-zeros (double core, 20 µm).
h0, h = 50.0, 100.0
for name, (lam, nc, layers, nb) in cases.items():
    k, st = case(lam, nc, layers, nb)
    for pol in (2, 1):
        poles = S.guided_poles_gamma(st, k, pol)
        modes = [(b,) + S.mode_profile(st, k, pol, b)[:2] for b in poles]
        feats = S.stack_features(st, k, pol, poles)
        rows, prev = [], None
        for X in (2e3, 5e3, 1e4, 2e4, 4e4):
            p = S.build_path(k, feats, X, h + h0)
            g = 0.25j * hankel1(0, k * np.hypot(X, h - h0)) + S.g_path(p, st, k, pol, X, h + h0)
            terms = [1j / (2 * b * N) * u(np.array([h]))[0] * u(np.array([h0]))[0] * np.exp(1j * b * X)
                     for b, u, N in modes]
            gm, scale = sum(terms), sum(abs(t) for t in terms)
            rem = abs(g - gm)
            slope = np.log(rem / prev[1]) / np.log(X / prev[0]) if prev else np.nan
            rows.append(f"{X / 1e3:4.0f}µm {rem / scale:.1e}" + (f" (slope {slope:+.2f})" if prev else ""))
            prev = (X, rem)
        print(f"  {name:24s} {'TE' if pol == 2 else 'TM'}: " + "  ".join(rows))

print("\n(b) dipole power balance: Im G(r₀,r₀) [path] vs P_up + P_down + P_guided [real axis + profiles]")
kw = dict(epsabs=1e-17, epsrel=1e-13, limit=2000)
for name, (lam, nc, layers, nb) in cases.items():
    k, st = case(lam, nc, layers, nb)
    _, kb = st.ks(k)
    kb = float(kb.real)
    for pol in (2, 1):
        poles = S.guided_poles_gamma(st, k, pol)
        modes = [(b,) + S.mode_profile(st, k, pol, b)[:2] for b in poles]
        w_b = 1.0 if pol == 2 else st.eps_bot.real
        for h0 in (20.0, 100.0, 400.0):
            p = S.build_path(k, S.stack_features(st, k, pol, poles), 2 * h0, 2 * h0)
            lhs = 0.25 + S.g_path(p, st, k, pol, 0.0, 2 * h0).imag

            def up(th):
                q, a = k * np.sin(th), k * np.cos(th)
                R = st.R(np.array([q + 0j]), k, pol)[0]
                return abs(np.exp(-1j * a * h0) + R * np.exp(1j * a * h0)) ** 2 / 4

            def down(q):
                a1 = S.a_phys(k, q)
                ab = np.sqrt(max(kb * kb - q * q, 0.0))
                tau = st.tau(np.array([q + 0j]), k, pol)[0]
                return (ab / w_b) * abs(tau) ** 2 * abs(np.exp(1j * a1 * h0)) ** 2 / 4

            P_up = quad(up, 0, np.pi / 2, **kw)[0] / np.pi
            pts = [float(k)] if k < kb else None
            P_dn = quad(down, 0, kb, points=pts, **kw)[0] / np.pi
            P_g = sum((u(np.array([h0]))[0] ** 2 / (2 * b * N)).real for b, u, N in modes)
            rhs = P_up + P_dn + P_g
            print(f"  {name:24s} {'TE' if pol == 2 else 'TM'} h₀={h0:5.0f}: Im G = {lhs:.12f}  "
                  f"up {P_up:.4f} down {P_dn:.4f} guided {P_g:.4f}  |Δ|/ImG = {abs(lhs - rhs) / lhs:.1e}")
