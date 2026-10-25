"""S11 (anchor for R18): in-plane dipoles as derivative sources, emission into guided modes.

A current J in the cover gives, for TM (u = H_y, p = 1/ε),
    ∇·(p∇u) + k²u = −[∇×J]_y,
so with the Green function of (∇·p∇ + k²w)G = −δ, the field is the derivative source
    u(r) = −(b·∇')G(r, r')|_{r' = r₀},     b ∝ (p_z, −p_x)   (constant to pin in the spec).
The same form in TE (u = E_y) is an in-plane *magnetic* dipole; the algebra is pol-blind.

Closed forms, from S8/S9's guided pole g ⊃ u_m(z)u_m(z')/((q² − β²)N):
  amplitudes   c_m^± = (i/(2β_m N_m)) u_m(z₀) (±iβ_m b_x + γ₁ b_z)      (u_m' = −γ₁u_m in the cover)
  powers       P_m^± = u_m(z₀)² |±iβ_m b_x + γ₁ b_z|² / (4β_m N_m)
  one-way      c_m^− = 0  ⇔  b_z = iβ_m b_x/γ₁   (spin-locked, elliptical, |b_z/b_x| = β/γ > 1)
Power balance, P = Im[(b*·∇)(b·∇')G] at coincidence:
  LHS  k²|b|²/8 + Im (1/π)∫_Γ (i/2)(R/α₁)(q²|b_x|² − α₁²|b_z|²) e^{2iα₁h₀} dq   (odd cross terms vanish)
  RHS  P_up   = (1/2π)∫_{−k}^{k} |A↑e^{−iα₁h₀} + A↓R e^{iα₁h₀}|²/(4α₁) dq,  A↑ = i(qb_x + α₁b_z), A↓ = i(qb_x − α₁b_z)
       P_down = (1/2π)∫_{−k_b}^{k_b} Re(Y_b)|A↓|²|τ|²|e^{iα₁h₀}|²/4 dq
       P_guided = Σ_m P_m⁺ + P_m⁻
  Full ranges, not folded: the integrands are not even in q once b_x b_z ≠ 0.
"""
import numpy as np
from scipy.integrate import quad
from scipy.special import hankel1

import sproto as S


def case(lam, n_clad, layers, n_bot):
    k = 2 * np.pi * n_clad / lam
    st = S.Stack([n**2 / n_clad**2 for n, _ in layers], [d for _, d in layers], n_bot**2 / n_clad**2)
    return k, st


def amp(b, beta, N, u0, g1, sign):
    return 1j / (2 * beta * N) * u0 * (sign * 1j * beta * b[0] + g1 * b[1])


def field_b(st, k, pol, b, x, z, h0, path):
    """u = −(b·∇')G_total at (x, z) in the cover, source at (0, h0), z_top = 0."""
    X, dz = x, z - h0
    rho = np.hypot(X, dz)
    h1 = hankel1(1, k * rho)
    gx = 0.25j * k * h1 * X / rho  # ∂x' of (i/4)H₀(kρ)
    gz = 0.25j * k * h1 * dz / rho
    rx = S.g_path(path, st, k, pol, X, z + h0, which=1)
    rz = S.g_path(path, st, k, pol, X, z + h0, which=2)
    return -(b[0] * (gx + rx) + b[1] * (gz + rz))


cases = {
    "SOI 220 @1550": (1550, 1.444, [(3.476, 220)], 1.444),
    "air|Si 220|SiO2": (1550, 1.0, [(3.476, 220)], 1.444),
    "SiN 400 @800 (2 modes)": (800, 1.444, [(2.0, 400)], 1.444),
}

print("(a) far-x field of −(b·∇')G vs the guided closed form, both directions (h₀ = 50, observer h = 100 nm)")
h0, h = 50.0, 100.0
for name, (lam, nc, layers, nb) in cases.items():
    k, st = case(lam, nc, layers, nb)
    for pol in (1, 2):
        poles = S.guided_poles_gamma(st, k, pol)
        modes = [(b_,) + S.mode_profile(st, k, pol, b_)[:2] for b_ in poles]
        g1s = [np.sqrt(b_ * b_ - k * k) for b_ in poles]
        b0 = poles[0]
        bs = {"b=(1,0)": (1.0, 0.0), "b=(0,1)": (0.0, 1.0),
              "spin-locked to mode 0": (1.0, 1j * b0 / g1s[0])}
        feats = S.stack_features(st, k, pol, poles)
        for lab, b in bs.items():
            row = []
            for X in (1e4, 4e4):
                p = S.build_path(k, feats, X, h + h0)
                for sgn in (+1, -1):
                    f = field_b(st, k, pol, b, sgn * X, h, h0, p)
                    terms = [amp(b, bb, N, u(np.array([h0]))[0], g1, sgn) * u(np.array([h]))[0] * np.exp(1j * bb * X)
                             for (bb, u, N), g1 in zip(modes, g1s)]
                    scale = sum(abs(amp(b, bb, N, u(np.array([h0]))[0], g1, +1) * u(np.array([h]))[0])
                                + abs(amp(b, bb, N, u(np.array([h0]))[0], g1, -1) * u(np.array([h]))[0])
                                for (bb, u, N), g1 in zip(modes, g1s))
                    row.append(f"{'+' if sgn > 0 else '−'}{X / 1e3:.0f}µm {abs(f - sum(terms)) / scale:.1e}")
            print(f"  {name:24s} {'TM' if pol == 1 else 'TE'} {lab:22s}: |u − guided|/Σ|c| at " + "  ".join(row))

print("\n(b) power balance Im[(b*·∇)(b·∇')G] = P_up + P_down + Σ(P⁺ + P⁻), and the directionality")
kw = dict(epsabs=1e-17, epsrel=1e-12, limit=2000)
for name, (lam, nc, layers, nb) in cases.items():
    k, st = case(lam, nc, layers, nb)
    _, kb = st.ks(k)
    kb = float(kb.real)
    for pol in (1, 2):
        poles = S.guided_poles_gamma(st, k, pol)
        modes = [(b_,) + S.mode_profile(st, k, pol, b_)[:2] for b_ in poles]
        g10 = np.sqrt(poles[0] ** 2 - k * k)
        wb = 1.0 if pol == 2 else st.eps_bot.real
        for lab, b in {"(1,0)": (1.0, 0.0), "(0,1)": (0.0, 1.0), "(1, 0.7−0.4i)": (1.0, 0.7 - 0.4j),
                       "spin-locked": (1.0, 1j * poles[0] / g10)}.items():
            for hh in (20.0, 150.0):
                p = S.build_path(k, S.stack_features(st, k, pol, poles), 2 * hh, 2 * hh)
                q, w = p["q"], p["w"]
                a1 = S.alpha(q, k)
                integ = 0.5j * st.R(q, k, pol) / a1 * (q * q * abs(b[0]) ** 2 - a1 * a1 * abs(b[1]) ** 2) \
                    * np.exp(2j * a1 * hh)
                lhs = k * k * (abs(b[0]) ** 2 + abs(b[1]) ** 2) / 8 + (np.sum(w * integ) / np.pi).imag

                def up(th):
                    q_, a_ = k * np.sin(th), k * np.cos(th)
                    R = st.R(np.array([q_ + 0j]), k, pol)[0]
                    Au, Ad = 1j * (q_ * b[0] + a_ * b[1]), 1j * (q_ * b[0] - a_ * b[1])
                    return abs(Au * np.exp(-1j * a_ * hh) + Ad * R * np.exp(1j * a_ * hh)) ** 2 / 4

                def down(q_):
                    a1_ = S.a_phys(k, q_)
                    ab = np.sqrt(max(kb * kb - q_ * q_, 0.0))
                    Ad = 1j * (q_ * b[0] - a1_ * b[1])
                    tau = st.tau(np.array([q_ + 0j]), k, pol)[0]
                    return (ab / wb) * abs(Ad) ** 2 * abs(tau) ** 2 * abs(np.exp(1j * a1_ * hh)) ** 2 / 4

                P_up = quad(up, -np.pi / 2, np.pi / 2, **kw)[0] / (2 * np.pi)  # dq = α dθ cancels 1/α
                pts = [-k, k] if k < kb else None
                P_dn = quad(down, -kb, kb, points=pts, **kw)[0] / (2 * np.pi)
                Pp = Pm = 0.0
                for bb, u, N in modes:
                    u0 = u(np.array([hh]))[0].real
                    g1 = np.sqrt(bb * bb - k * k)
                    Pp += u0**2 * abs(1j * bb * b[0] + g1 * b[1]) ** 2 / (4 * bb * N.real)
                    Pm += u0**2 * abs(-1j * bb * b[0] + g1 * b[1]) ** 2 / (4 * bb * N.real)
                rhs = P_up + P_dn + Pp + Pm
                print(f"  {name:24s} {'TM' if pol == 1 else 'TE'} b={lab:14s} h₀={hh:5.0f}: |Δ|/P = {abs(lhs - rhs) / lhs:.1e}  "
                      f"guided {100 * (Pp + Pm) / lhs:5.1f} %  P⁺/(P⁺+P⁻) = {Pp / (Pp + Pm):.4f}")
