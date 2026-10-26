"""S7: transmitted G below the stack vs a real-axis reference (extends v1.1 GT2).

Path:  G_t = (i/2π) ∫_Γ⁺ τ(q) cos(qX) e^{iα₁h' + iα_b|h|} dq   (folded), τ from the recursion.
Reference: T by shooting upward from the bottom through the
transfer matrix (the forward form aR + b cancels on the tail, measured 1e37), radiation-condition branch rule,
θ/cosh substitutions, the guided poles by the symmetric-fold PV plus iπ·Res(τ), with Res(τ)
from a Cauchy circle on the recursion's τ.
"""
import numpy as np
from scipy.integrate import quad

import sproto as S


def T_ref(st, k, pol, q):
    """T by shooting upward from the bottom: u_b = 1, p u' = +iY_b·... (decaying below).

    The forward form T = aR + b cancels catastrophically on the tail (true T ~ e^{−qd}
    from O(1) terms); shooting from the evanescent side makes the growth the dominant
    direction, so T = 1/A_down is relatively accurate. Independent of the recursion.
    """
    kl, kb = st.ks(k)
    Y1 = S.a_phys(k, q)
    ab = S.a_phys(kb, q)
    Yb = ab if pol == 2 else ab / st.eps_bot
    # bottom field T e^{−iα_b(z − z_bot)} with T = 1: u = 1, v = p u' = −iY_b
    u, v = 1.0 + 0j, -1j * Yb
    for kj, ej, dj in zip(kl[::-1], st.eps[::-1], st.d[::-1]):
        a = np.sqrt(complex(kj * kj - q * q))
        w = 1.0 if pol == 2 else ej
        c = np.cos(a * dj)
        sinc = np.sinc(a * dj / np.pi) * dj
        u, v = c * u + w * sinc * v, -(a * a / w) * sinc * u + c * v  # upward by d (inverse of R_ref's M)
        if not (np.isfinite(u) and np.isfinite(v)) or abs(u) > 1e290:
            return 0.0
    A = 0.5 * (u - v / (1j * Y1))  # downward amplitude at z_top
    return 1.0 / A


def gt_ref(st, k, pol, X, hs, hb, poles, res, hw_frac=0.45):
    k1 = float(np.real(k))
    _, kb = st.ks(k)
    kbr = float(np.real(kb))

    def f1(th):
        q, a = k1 * np.sin(th), k1 * np.cos(th)
        ab = S.a_phys(kb, q)
        # τ dq = (T/α₁)·α₁ dθ
        return 1j / (2 * np.pi) * T_ref(st, k, pol, q) * np.cos(q * X) * np.exp(1j * a * hs + 1j * ab * hb)

    def f2(s):
        q, a = k1 * np.cosh(s), 1j * k1 * np.sinh(s)
        ab = S.a_phys(kb, q)
        # τ dq = (T/α₁)·k₁ sinh s ds = −iT ds, so (i/2π)·τ dq = T ds/2π
        return 1 / (2 * np.pi) * T_ref(st, k, pol, q) * np.cos(q * X) * np.exp(1j * a * hs + 1j * ab * hb)

    kw = dict(epsabs=1e-18, epsrel=1e-13, limit=4000)

    def qd(fn, a, b, pts=None):
        return quad(lambda t: fn(t).real, a, b, points=pts, **kw)[0] + 1j * quad(lambda t: fn(t).imag, a, b, points=pts, **kw)[0]

    pts1 = [np.arcsin(kbr / k1)] if kbr < k1 else None
    out = qd(f1, 0, np.pi / 2, pts1)
    sp = [np.arccosh(b / k1) for b in poles]
    edges = sorted(set([0.0] + ([np.arccosh(kbr / k1)] if kbr > k1 else [])))
    smax = np.arccosh(40.0 / (k1 * (hs + hb)) + max(list(poles) + [kbr, k1]) / k1)
    edges.append(smax)
    hw = hw_frac * min([abs(a - b) for a in sp for b in sp + edges if a != b] + [0.1])
    cuts = sorted(edges + [s - hw for s in sp] + [s + hw for s in sp])
    for a, b in zip(cuts[:-1], cuts[1:]):
        c = [s for s in sp if a < s < b]
        if c:
            s0 = c[0]
            out += qd(lambda u: f2(s0 + u) + f2(s0 - u), 0.0, b - s0)
        else:
            out += qd(f2, a, b)
    for b, r in zip(poles, res):
        a1 = 1j * np.sqrt(b * b - k1 * k1)
        ab = 1j * np.sqrt(b * b - kbr * kbr)
        out += 1j * np.pi * (1j / (2 * np.pi)) * r * np.cos(b * X) * np.exp(1j * a1 * hs + 1j * ab * hb)
    return out


def gt_path(path, st, k, pol, X, hs, hb):
    q, w = path["q"], path["w"]
    _, kb = st.ks(k)
    return 1j / (2 * np.pi) * np.sum(w * st.tau(q, k, pol) * np.cos(q * X)
                                     * np.exp(1j * S.alpha(q, k) * hs + 1j * S.alpha(q, kb) * hb))


nc = 1.444
cases = {
    "SOI 220": (1550, 1.444, [(3.476, 220)], 1.444),
    "air|Si 220|SiO2": (1550, 1.0, [(3.476, 220)], 1.444),
    "Si/SiO2/Si": (1550, 1.444, [(3.476, 220), (1.444, 300), (3.476, 220)], 1.444),
    "SiN 400 @800": (800, 1.444, [(2.0, 400)], 1.444),
}
pts = [(0.0, 20.0, 30.0), (300.0, 50.0, 10.0), (1500.0, 100.0, 200.0)]  # (X, h', |h|)
for name, (lam, ncl, layers, nb) in cases.items():
    k = 2 * np.pi * ncl / lam
    st = S.Stack([n**2 / ncl**2 for n, _ in layers], [d for _, d in layers], nb**2 / ncl**2)
    for pol in (2, 1):
        poles = S.guided_poles_gamma(st, k, pol)
        lo = S.guided_window(st, k)[0]
        res = []
        for b in poles:
            r = min(0.3 * (b - lo), 1e-3 * b)
            t = 2 * np.pi * np.arange(64) / 64
            z = b + r * np.exp(1j * t)
            res.append(np.mean(st.tau(z, k, pol) * r * np.exp(1j * t)))
        feats = S.stack_features(st, k, pol, poles)
        e_ref = e_s = e_d = e_f = 0.0
        for X, hs, hb in pts:
            p1 = S.build_path(k, feats, 1500.0, hs + hb)
            p2 = S.build_path(k, feats, 1500.0, hs + hb, nodes_scale=2.0)
            ph = S.build_path(k, feats, 1500.0, hs + hb, depth_scale=0.5)
            g = gt_path(p1, st, k, pol, X, hs, hb)
            ref = gt_ref(st, k, pol, X, hs, hb, poles, res)
            ref2 = gt_ref(st, k, pol, X, hs, hb, poles, res, hw_frac=0.2)
            sc = abs(ref)
            e_ref = max(e_ref, abs(g - ref) / sc)
            e_f = max(e_f, abs(ref2 - ref) / sc)
            e_s = max(e_s, abs(gt_path(p2, st, k, pol, X, hs, hb) - g) / sc)
            e_d = max(e_d, abs(gt_path(ph, st, k, pol, X, hs, hb) - g) / sc)
        print(f"{name:16s} {'TE' if pol == 2 else 'TM'}: path−ref = {e_ref:.0e}  (ref fold spread {e_f:.0e})  "
              f"s2−s1 = {e_s:.0e}  δ/2−δ = {e_d:.0e}")
