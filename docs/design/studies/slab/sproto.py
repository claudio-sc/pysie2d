"""Shared prototype code for the v1.2 slab study (not package code).

Frame as v0.9: cover above z_top, background-relative ε (cover ε = 1), k = k_bg.
Layers j = 1..N run downward from z_top with thickness d_j; the bottom
half-space starts at z_bot = z_top − Σd. Downward wave ∝ e^{−iα z}.

Admittance Y = α (TE, pol = 2, field E_y) or α/ε (TM, pol = 1, field H_y), so
one recursion serves both, and t = 1 + r = 2Y_a/(Y_a+Y_b) at every interface.

In layer j: ψ = A_j e^{−iα_j(z−z_j)} + B_j e^{+iα_j(z−z_j)}, z_j its top.
Γ_j = B_j/A_j at the top of j; at its bottom it is Γ_j e^{−2iα_j d_j}.
Across an interface a|b with b's Γ_b (at the interface):
    G_a = (r + Γ_b)/(1 + r Γ_b),   A_b = A_a·t/(1 + r Γ_b).
"""
import numpy as np
from scipy.integrate import quad
from scipy.optimize import brentq

from pysie2d import layered as L

PI = np.pi
alpha = L.alpha  # outer half-spaces: the package's vertical-cut sheet


def _norm(e):
    e = complex(e)
    return complex(e.real, e.imag + 0.0)


def alpha_int(q, kj):
    """Interior layer: R is even in α_j, so pick Im α_j ≥ 0 and |e^{2iα d}| ≤ 1."""
    a = np.sqrt(np.asarray(kj * kj - q * q, dtype=complex))
    return np.where(a.imag < 0, -a, a)


class Stack:
    """eps: layer permittivities relative to the cover, top first; d: thicknesses (nm)."""

    def __init__(self, eps, d, eps_bot):
        self.eps = [_norm(e) for e in eps]
        self.d = [float(x) for x in d]
        self.eps_bot = _norm(eps_bot)
        assert len(self.eps) == len(self.d)

    @property
    def thickness(self):
        return sum(self.d)

    def ks(self, k):
        return [k * np.sqrt(e) for e in self.eps], k * np.sqrt(self.eps_bot)

    def _adm(self, q, k, pol):
        kl, kb = self.ks(k)
        a1, ab = alpha(q, k), alpha(q, kb)
        al = [alpha_int(q, kj) for kj in kl]
        if pol == 2:
            Y = [a1] + al + [ab]
        else:
            Y = [a1] + [a / e for a, e in zip(al, self.eps)] + [ab / self.eps_bot]
        return Y, al

    def _gammas(self, q, k, pol):
        """Γ at the top of every medium 0..n+1 (Γ_{n+1} = 0), plus Y and α_j."""
        Y, al = self._adm(q, k, pol)
        n = len(self.d)
        Gam = [None] * (n + 2)
        Gam[n + 1] = np.zeros_like(Y[0])
        for i in range(n, -1, -1):  # interface between medium i (above) and i+1
            r = (Y[i] - Y[i + 1]) / (Y[i] + Y[i + 1])
            G = (r + Gam[i + 1]) / (1.0 + r * Gam[i + 1])
            Gam[i] = G if i == 0 else G * np.exp(2j * al[i - 1] * self.d[i - 1])
        return Gam, Y, al

    def R(self, q, k, pol):
        """Generalised reflection at z_top, cover side. Never forms e^{+|Im α| d}."""
        q = np.asarray(q, dtype=complex)
        if not self.d:  # N = 0: exactly the package's half-space expression
            return L.fresnel_r(q, k, self.eps_bot, pol)
        return self._gammas(q, k, pol)[0][0]

    def tau(self, q, k, pol):
        """τ = T/α₁: downward amplitude at z_bot per unit downward amplitude at z_top."""
        q = np.asarray(q, dtype=complex)
        Gam, Y, al = self._gammas(q, k, pol)
        n = len(self.d)
        out = 2.0 / (Y[0] + Y[1])  # t₀₁/α₁, finite at q = ±k₁ (Y₀ = α₁)
        for i in range(n + 1):
            r = (Y[i] - Y[i + 1]) / (Y[i] + Y[i + 1])
            if i > 0:
                out = out * 2.0 * Y[i] / (Y[i] + Y[i + 1]) * np.exp(1j * al[i - 1] * self.d[i - 1])
            out = out / (1.0 + r * Gam[i + 1])
        return out

    # -- naive transfer matrix: the form the stable recursion must beat -----
    def _tm(self, q, k, pol):
        """ψ, ψ'/w across the stack: cos, sin(αd)/α entries, entire in each α_j.

        Grows like e^{|Im α_j| d_j} on the tail, which is the instability.
        """
        kl, kb = self.ks(k)
        Y1 = alpha(q, k)
        ab = alpha(q, kb)
        Yb = ab if pol == 2 else ab / self.eps_bot
        M11, M12, M21, M22 = (np.ones_like(q), np.zeros_like(q), np.zeros_like(q), np.ones_like(q))
        for kj, ej, dj in zip(kl, self.eps, self.d):
            a = np.sqrt(kj * kj - q * q + 0j)
            w = 1.0 if pol == 2 else ej
            c = np.cos(a * dj)
            sinc = np.sinc(a * dj / PI) * dj  # sin(αd)/α
            m11, m12, m21, m22 = c, -w * sinc, (a * a / w) * sinc, c
            M11, M12, M21, M22 = (m11 * M11 + m12 * M21, m11 * M12 + m12 * M22,
                                  m21 * M11 + m22 * M21, m21 * M12 + m22 * M22)
        a_ = M11 + 1j * Y1 * M12
        b_ = M11 - 1j * Y1 * M12
        c_ = M21 + 1j * Y1 * M22
        d_ = M21 - 1j * Y1 * M22
        return d_ + 1j * Yb * b_, -1j * Yb * a_ - c_

    def R_naive(self, q, k, pol):
        num, den = self._tm(np.asarray(q, dtype=complex), k, pol)
        return num / den

    def D(self, q, k, pol):
        """Dispersion function, the transfer-matrix denominator of R.

        Real on the guided window for a lossless stack, so roots bracket.
        """
        return self._tm(np.asarray(q, dtype=complex), k, pol)[1]


# -- guided modes ------------------------------------------------------------
def guided_window(st, k):
    kl, kb = st.ks(k)
    lo = max(abs(k), abs(kb.real))
    hi = max(kj.real for kj in kl) if kl else lo
    return lo, hi


def guided_poles(st, k, pol, n_grid=4000):
    """Real roots of D on the guided window, bracketed by sign change (lossless only)."""
    lo, hi = guided_window(st, k)
    if hi <= lo:
        return []
    qs = np.linspace(lo, hi, n_grid + 1)[1:-1]

    def f(x):
        v = st.D(np.array(x, dtype=complex), k, pol)
        return float(np.real(v))

    vals = np.array([f(x) for x in qs])
    roots = []
    for a, b, fa, fb in zip(qs[:-1], qs[1:], vals[:-1], vals[1:]):
        if fa == 0.0:
            roots.append(a)
        elif fa * fb < 0:
            roots.append(brentq(f, a, b, xtol=1e-16 * hi, rtol=8.9e-16, maxiter=200))
    return sorted(roots, reverse=True)


def winding_count(st, k, pol, n=4096):
    """Zeros of D in a box around the open guided window, by the argument principle.

    D is analytic there: interior entries are entire, and the outer cuts run
    vertically from ±k₁, ±k_bot, left of the box.
    """
    lo, hi = guided_window(st, k)
    w = hi - lo
    a, b = lo + 1e-6 * w, hi + 0.05 * w
    h = 0.05 * w
    c, rx, ry = 0.5 * (a + b), 0.5 * (b - a), h
    t = 2 * PI * np.arange(n) / n
    z = c + rx * np.cos(t) + 1j * ry * np.sin(t)
    v = st.D(z, k, pol)
    ph = np.unwrap(np.angle(np.append(v, v[0])))
    return int(round((ph[-1] - ph[0]) / (2 * PI)))


def dD(st, k, pol, q0, r=None, n=64):
    """D'(q0) by Cauchy's formula on a small circle (trapezoid: spectral)."""
    r = r or 1e-3 * abs(q0)
    t = 2 * PI * np.arange(n) / n
    z = q0 + r * np.exp(1j * t)
    return np.mean(st.D(z, k, pol) * np.exp(-1j * t)) / r


def residue_R(st, k, pol, q0, r=None, n=64):
    """Res_{q0} R by Cauchy's formula on a small circle around the pole (stable R)."""
    r = r or 1e-3 * abs(q0)
    t = 2 * PI * np.arange(n) / n
    z = q0 + r * np.exp(1j * t)
    return np.mean(st.R(z, k, pol) * r * np.exp(1j * t))


def slab3_closed_form(k, eps_c, eps_f, eps_s, d, pol):
    """Three-layer slab: κd = mπ + atan(p_c/κ) + atan(p_s/κ), p = γ (TE) or γ ε_f/ε (TM).

    The textbook (Marcuse / Kogelnik) form, independent of the transfer matrix.
    """
    kc, kf, ks_ = (k * np.sqrt(e) for e in (eps_c, eps_f, eps_s))
    lo, hi = max(kc, ks_), kf
    roots = []
    m = 0
    while True:
        def g(b):
            kap = np.sqrt(kf**2 - b**2)
            gc, gs = np.sqrt(b**2 - kc**2), np.sqrt(b**2 - ks_**2)
            pc, ps = (gc, gs) if pol == 2 else (gc * eps_f / eps_c, gs * eps_f / eps_s)
            return kap * d - m * PI - np.arctan(pc / kap) - np.arctan(ps / kap)

        a, b = lo * (1 + 1e-15), hi * (1 - 1e-15)
        if g(a) <= 0:
            break
        roots.append(brentq(g, a, b, xtol=1e-16 * hi, rtol=8.9e-16, maxiter=200))
        m += 1
    return roots


# -- path with features (the v1.2 item 2 change) ------------------------------
def build_path(k, feats, D, z_min, depth_scale=1.0, nodes_scale=1.0):
    """layered.SommerfeldPath._build with an explicit feature list.

    feats: [(label, q)] right-hand singularities: outer branch points and poles.
    """
    k = complex(k)
    near = [f.real for _, f in feats if f.real > 0 and abs(f.imag) < 2 * k.real]
    T = L.T_FRAC * max(near)
    floor = L.FEATURE_FLOOR * max([-f.imag for _, f in feats] + [0.0])
    delta = depth_scale * max(min(L.HUMP_DEPTH_FRAC * k.real, L.GROWTH_CAP / D), floor)
    clear = {}
    for label, f in feats:
        gamma = delta * np.sin(PI * f.real / T) if 0 < f.real < T else 0.0
        clear[label] = (f.imag + gamma) / delta
        assert clear[label] >= L.CLEARANCE, (label, f, clear[label])
    n_hump = max(L.MIN_HUMP_NODES, int(np.ceil(nodes_scale * L.HUMP_NODES_PER_T_OVER_DELTA * T / delta)))
    t, wt = L._gauss_legendre(n_hump, 0.0, T)
    q = [t - 1j * delta * np.sin(PI * t / T)]
    w = [wt * (1 - 1j * delta * (PI / T) * np.cos(PI * t / T))]
    length = L.TAIL_DECAY / z_min
    w_max = min(L.TAIL_PERIODS_PER_PANEL * 2 * PI / D, L.TAIL_EFOLDS_PER_PANEL / z_min, length) / nodes_scale
    width, start = min(0.5 * T, w_max), T
    while start < T + length:
        stop = min(start + width, T + length)
        x, wx = L._gauss_legendre(L.TAIL_PANEL_POINTS, start, stop)
        q.append(x.astype(complex))
        w.append(wx.astype(complex))
        start, width = stop, min(2 * width, w_max)
    return dict(q=np.concatenate(q), w=np.concatenate(w), T=T, delta=delta, clear=clear)


def stack_features(st, k, pol, poles):
    _, kb = st.ks(k)
    f = [("+k1", complex(k)), ("+k_bot", complex(kb))]
    f += [(f"beta{m}", complex(b)) for m, b in enumerate(poles)]
    return f


def g_path(path, st, k, pol, X, Z, which=0):
    """Reflected G (which 0), ∂x'G (1), ∂z'G (2) on the path; Z = z + z' − 2 z_top."""
    q, w = path["q"], path["w"]
    a = alpha(q, k)
    fac = [np.cos(q * X), q * np.sin(q * X), 1j * a * np.cos(q * X)][which]
    return 1j / (2 * PI) * np.sum(w * st.R(q, k, pol) / a * fac * np.exp(1j * a * Z))


# -- independent real-axis reference with explicit pole indentation -----------
def a_phys(kk, q):
    a = np.sqrt(complex(kk * kk - q * q))
    if a.imag < 0 or (a.imag == 0 and a.real < 0):
        a = -a
    return a


def R_ref(st, k, pol, q):
    """Transfer-matrix R with the radiation-condition branch rule (real q, real k)."""
    kl, kb = st.ks(k)
    Y1 = a_phys(k, q)
    ab = a_phys(kb, q)
    Yb = ab if pol == 2 else ab / st.eps_bot
    M = np.eye(2, dtype=complex)
    for kj, ej, dj in zip(kl, st.eps, st.d):
        a = np.sqrt(complex(kj * kj - q * q))
        w = 1.0 if pol == 2 else ej
        # R is homogeneous of degree 0 in M: scaling each layer by e^{−|Im α|d} is
        # exact and removes the overflow, borrowing nothing from the recursion.
        # Formed from bounded exponentials, since cos itself overflows first.
        ep = np.exp(1j * a * dj - abs(a.imag) * dj)
        em = np.exp(-1j * a * dj - abs(a.imag) * dj)
        c = 0.5 * (ep + em)
        if abs(a * dj) > 1e-3:
            sinc = (ep - em) / (2j * a)
        else:
            sinc = np.sinc(a * dj / PI) * dj * np.exp(-abs(a.imag) * dj)
        M = np.array([[c, -w * sinc], [(a * a / w) * sinc, c]]) @ M
    a_ = M[0, 0] + 1j * Y1 * M[0, 1]
    b_ = M[0, 0] - 1j * Y1 * M[0, 1]
    c_ = M[1, 0] + 1j * Y1 * M[1, 1]
    d_ = M[1, 0] - 1j * Y1 * M[1, 1]
    return (d_ + 1j * Yb * b_) / (-1j * Yb * a_ - c_)


def g_ref(st, k, pol, X, Z, poles, residues, which=0, extra_pts=(), hw_frac=0.45):
    """Real-axis QUADPACK: θ-substitution on [0, k₁], cosh on [k₁, ∞).

    Real guided poles: PV by a symmetric fold, plus iπ·Res for the path
    passing below +β (radiation condition). Residues are supplied by the caller
    from an independent computation (Cauchy circle on the stable R).
    """
    k1 = float(np.real(k))

    def fac(q, a):
        return [np.cos(q * X), q * np.sin(q * X), 1j * a * np.cos(q * X)][which]

    def f1(th):
        q, a = k1 * np.sin(th), k1 * np.cos(th)
        return 1j / (2 * PI) * R_ref(st, k, pol, q) * fac(q, a) * np.exp(1j * a * Z)

    def f2(s):
        q, a = k1 * np.cosh(s), 1j * k1 * np.sinh(s)
        return 1 / (2 * PI) * R_ref(st, k, pol, q) * fac(q, a) * np.exp(-k1 * np.sinh(s) * Z)

    kw = dict(epsabs=1e-18, epsrel=1e-14, limit=4000)

    def qd(fn, a, b, pts=None):
        re = quad(lambda t: fn(t).real, a, b, points=pts or None, **kw)[0]
        im = quad(lambda t: fn(t).imag, a, b, points=pts or None, **kw)[0]
        return re + 1j * im

    _, kb = st.ks(k)
    brk = [float(np.real(kb))] + list(extra_pts)
    pts1 = [np.arcsin(f / k1) for f in brk if 0 < f < k1]
    s_pole = [np.arccosh(b / k1) for b in poles]
    pts2 = sorted(set([np.arccosh(f / k1) for f in brk if f > k1]))
    s_max = np.arccosh(40.0 / (k1 * Z) + max(list(poles) + brk + [k1]) / k1)
    out = qd(f1, 0, PI / 2, pts1)
    # PV by the symmetric fold ∫₀ʰ [f(s_p+u) + f(s_p−u)] du: the ±Res/u parts cancel,
    # so no residue enters the PV and a wrong residue cannot hide in it.
    edges = sorted(set([0.0, s_max] + pts2))
    hw = hw_frac * min([abs(a - b) for a in s_pole for b in s_pole + edges if a != b] + [0.1])
    cuts = sorted(edges + [s - hw for s in s_pole] + [s + hw for s in s_pole])
    for a, b in zip(cuts[:-1], cuts[1:]):
        c = [s for s in s_pole if a < s < b]
        if c:
            sp = c[0]
            out += qd(lambda u: f2(sp + u) + f2(sp - u), 0.0, b - sp)
        else:
            out += qd(f2, a, b)
    pre = 1j / (2 * PI)
    for b, res in zip(poles, residues):
        a1 = 1j * np.sqrt(b * b - k1 * k1)
        out += 1j * PI * pre * res / a1 * fac(b, a1) * np.exp(1j * a1 * Z)
    return out


def guided_poles_gamma(st, k, pol, n_grid=4000):
    """Real roots of D bracketed on a uniform grid in γ = √(q² − k_out²), from γ = 0.

    Near cutoff D is linear in γ (the outer Y = iγ), not in q, so a q-grid that
    starts one cell inside the window misses a root at (β − k_out)/k ≲ cell/k,
    while γ = 0 is an exact grid point here and D(k_out) is finite.
    """
    lo, hi = guided_window(st, k)
    if hi <= lo:
        return []
    g_hi = np.sqrt(hi * hi - lo * lo)
    gs = np.linspace(0.0, g_hi, n_grid + 1)[:-1]

    def f(g):
        return float(np.real(st.D(np.array(np.sqrt(lo * lo + g * g), dtype=complex), k, pol)))

    vals = np.array([f(g) for g in gs])
    roots = []
    for a, b, fa, fb in zip(gs[:-1], gs[1:], vals[:-1], vals[1:]):
        if fa * fb < 0:
            g0 = brentq(f, a, b, xtol=1e-300, rtol=8.9e-16, maxiter=300)
            roots.append(np.sqrt(lo * lo + g0 * g0))
    return sorted(roots, reverse=True)


def winding_count_gamma(st, k, pol, n=8192, h_frac=0.01):
    """Zeros of D(q(γ)) with Re γ > 0, by the argument principle in the γ-plane.

    q = √(k_out² + γ²) uniformises the outer branch point: the outer Y = iγ, so D
    is analytic through γ = 0, and the contour can run along Re γ = 0 itself
    (D(γ = 0) ≠ 0 generically). Improper zeros at Re γ < 0, which a mode becomes
    below cutoff, stay outside. A rectangle [0, 1.05 γ_hi] × [−h, h], trapezoid.

    Trap (S4): Re γ > 0 is the Im α ≥ 0 sheet, which differs from the code's
    vertical-cut sheet left of k_out (Re q² < k_out², i.e. |Im γ| > Re γ). A box
    of h = 0.05·γ_hi counted a leaky pole at q = 0.99655·k_out + 6e-5i·k (Si/6 µm
    SiO₂/Si), which is improper on the code's sheet. Keep the box thin.
    """
    lo, hi = guided_window(st, k)
    g_hi = 1.05 * np.sqrt(hi * hi - lo * lo)
    h = h_frac * g_hi
    m = n // 4
    s = np.linspace(0, 1, m, endpoint=False)
    zs = np.concatenate([g_hi * s - 1j * h, g_hi + 1j * h * (2 * s - 1),
                         g_hi * (1 - s) + 1j * h, 1j * h * (1 - 2 * s)])
    q = np.sqrt(lo * lo + zs * zs)
    v = st.D(q, k, pol)
    ph = np.unwrap(np.angle(np.append(v, v[0])))
    return int(round((ph[-1] - ph[0]) / (2 * PI)))


def g_ref_gamma(st, k, pol, X, Z, poles, residues, which=0, hw_frac=0.45):
    """Real-axis reference for near-cutoff poles: q ≥ k_out integrated in γ.

    q = √(k_out² + γ²) makes the outer branch point a smooth endpoint (the outer α
    is iγ and dq = γ/q dγ), so a pole at γ_p = √(β² − k_out²) is separated from it
    by γ_p ≈ √(2t)·k_out instead of t·k_out. [0, k₁] by θ, [k₁, k_out] (if
    k₁ < k_out) by cosh near k₁ and by q = √(k_out² − y²) near
    k_out, where a near-cutoff pole is t·k_out beyond the panel's end. PV by the same symmetric fold.
    """
    k1 = float(np.real(k))
    _, kb = st.ks(k)
    ko = max(k1, float(np.real(kb)))
    pre = 1j / (2 * PI)

    def fac(q, a):
        return [np.cos(q * X), q * np.sin(q * X), 1j * a * np.cos(q * X)][which]

    def f1(th):
        q, a = k1 * np.sin(th), k1 * np.cos(th)
        return pre * R_ref(st, k, pol, q) * fac(q, a) * np.exp(1j * a * Z)

    def f2(s):
        q, a = k1 * np.cosh(s), 1j * k1 * np.sinh(s)
        return 1 / (2 * PI) * R_ref(st, k, pol, q) * fac(q, a) * np.exp(-k1 * np.sinh(s) * Z)

    def f3(g):
        q = np.sqrt(ko * ko + g * g)
        a = 1j * np.sqrt(q * q - k1 * k1) if ko > k1 else 1j * g
        jac = g / q
        # R/α₁·dq/dγ: for k₁ = k_out, α₁ = iγ cancels γ in the Jacobian
        w = (1.0 / (1j * q)) if ko == k1 else jac / a
        return pre * R_ref(st, k, pol, q) * w * fac(q, a) * np.exp(1j * a * Z)

    kw = dict(epsabs=1e-18, epsrel=1e-14, limit=4000)

    def qd(fn, a, b):
        re = quad(lambda t: fn(t).real, a, b, **kw)[0]
        im = quad(lambda t: fn(t).imag, a, b, **kw)[0]
        return re + 1j * im

    def f4(y):  # q = √(k_out² − y²) on [mid, k_out]: the pole's left flank, uniformised
        q = np.sqrt(ko * ko - y * y)
        a = 1j * np.sqrt(q * q - k1 * k1)
        return pre * R_ref(st, k, pol, q) / a * (y / q) * fac(q, a) * np.exp(1j * a * Z)

    out = qd(f1, 0, PI / 2)
    if ko > k1:
        mid = 0.5 * (k1 + ko)
        out += qd(f2, 0, np.arccosh(mid / k1))
        out += qd(f4, 0, np.sqrt(ko * ko - mid * mid))
    g_pole = sorted(np.sqrt(b * b - ko * ko) for b in poles)
    g_max = np.sqrt((40.0 / Z + max(list(poles) + [ko])) ** 2 - ko * ko)
    edges = [0.0, g_max]
    hw = hw_frac * min([abs(a - b) for a in g_pole for b in g_pole + edges if a != b] + [g_max])
    cuts = sorted(edges + [g - hw for g in g_pole] + [g + hw for g in g_pole])
    for a, b in zip(cuts[:-1], cuts[1:]):
        c = [g for g in g_pole if a < g < b]
        if c:
            gp = c[0]
            out += qd(lambda u: f3(gp + u) + f3(gp - u), 0.0, b - gp)
        else:
            out += qd(f3, a, b)
    for b, res in zip(poles, residues):
        a1 = 1j * np.sqrt(b * b - k1 * k1)
        out += 1j * PI * pre * res / a1 * fac(b, a1) * np.exp(1j * a1 * Z)
    return out


# -- guided-mode profiles (S8, S9) ---------------------------------------------
def mode_profile(st, k, pol, beta, n_gl=200):
    """Profile u(z) and norm N = ∫ p u² dz of the guided mode at a real root β.

    ∇·(p∇u) + k² w u = 0 with p = 1, w = ε (TE) or p = 1/ε, w = 1 (TM); the modes are
    orthogonal under ∫ p u_m u_n dz (bilinear), which is the weight of the 1D Green
    function's pole, g ⊃ u_m(z)u_m(z')/((q² − β_m²) N_m). Normalised u(z_top) = 1.
    Built by the (u, v = p u') transfer of R_naive, *not* from R, so that S8 can
    compare it with R's residue.

    Returns:
        u: callable of h = z − z_top (array), N, and the bottom mismatch
        |v_b − p_b γ_b u_b| / |p_b γ_b u_b| (zero at an exact root).
    """
    kl, kb = st.ks(k)
    g1 = np.sqrt(beta * beta - k * k)
    gb = np.sqrt(beta * beta - kb * kb).real
    p_b = 1.0 if pol == 2 else 1.0 / st.eps_bot.real
    u0, v0 = 1.0 + 0j, -g1 + 0j  # cover: u = e^{−γ₁ h}, p = 1
    tops, N = [], 1.0 / (2 * g1)
    x, wq = np.polynomial.legendre.leggauss(n_gl)
    for kj, ej, dj in zip(kl, st.eps, st.d):
        a = np.sqrt(complex(kj * kj - beta * beta))
        wj = 1.0 if pol == 2 else ej  # 1/p
        tops.append((u0, v0, a, wj, dj))
        s = 0.5 * dj * (x + 1)
        us = np.cos(a * s) * u0 - wj * np.sinc(a * s / PI) * s * v0
        N += 0.5 * dj * np.sum(wq * us * us) / wj
        u0, v0 = (np.cos(a * dj) * u0 - wj * np.sinc(a * dj / PI) * dj * v0,
                  (a * a / wj) * np.sinc(a * dj / PI) * dj * u0 + np.cos(a * dj) * v0)
    N += p_b * u0 * u0 / (2 * gb)
    mismatch = abs(v0 - p_b * gb * u0) / abs(p_b * gb * u0)
    z_lay = np.concatenate([[0.0], -np.cumsum(st.d)])

    def u(h):
        h = np.asarray(h, dtype=float)
        out = np.empty(h.shape, dtype=complex)
        out[h >= 0] = np.exp(-g1 * h[h >= 0])
        for j, (ut, vt, a, wj, dj) in enumerate(tops):
            m = (h < z_lay[j]) & (h >= z_lay[j + 1])
            s = z_lay[j] - h[m]
            out[m] = np.cos(a * s) * ut - wj * np.sinc(a * s / PI) * s * vt
        m = h < z_lay[-1]
        out[m] = u0 * np.exp(gb * (h[m] - z_lay[-1]))
        return out

    return u, complex(N), mismatch


def continue_poles(st_from, st_to, k_from, k_to, pol, poles, steps=16):
    """Newton-continue real guided poles to a lossy stack and/or complex k.

    Homotopy in s ∈ [0, 1]: ε_j(s) = (1−s)ε_j⁰ + s·ε_j, k(s) likewise; Newton on the
    entire D (in q) at each step, D'(q) by a Cauchy circle. Returns the poles and the
    worst |D|/|D'·q| at the end (a relative step size).
    """
    out, worst = [], 0.0
    for b in poles:
        q = complex(b)
        for s in np.linspace(0, 1, steps + 1)[1:]:
            st = Stack([(1 - s) * e0 + s * e1 for e0, e1 in zip(st_from.eps, st_to.eps)], st_from.d,
                       (1 - s) * st_from.eps_bot + s * st_to.eps_bot)
            kk = (1 - s) * k_from + s * k_to
            for _ in range(30):
                dq = st.D(np.array([q]), kk, pol)[0] / dD(st, kk, pol, q, r=1e-4 * abs(q))
                q -= dq
                if abs(dq) < 1e-15 * abs(q):
                    break
        worst = max(worst, abs(dq) / abs(q))
        out.append(q)
    return out, worst
