"""Half-space reflected Green function on a deformed Sommerfeld path (prototype).

Frame: x horizontal, z vertical, interface z = z_int, all points in the cover.
Time exp(-iwt).  G_ind = (i/4pi) int R(q)/a1 exp(i q X + i a1 Z) dq,
X = x - x', Z = z + z' - 2 z_int.
"""
import numpy as np

PI = np.pi


def alpha(q, k):
    """sqrt(k^2 - q^2) with branch cuts running vertically: up from +k, down from -k.

    Continuous on any path that passes below +k and above -k (the Sommerfeld
    path for Im k of either sign).  Real tail q >> |k| gives alpha = i q.
    """
    return 1j * np.sqrt(1j * (q - k)) * np.sqrt(-1j * (q + k))


def fresnel(q, k1, eps1, eps2, pol, k0):
    """Cover-side r12 in the pysie2d pol convention (2 = TE E_y, 1 = TM H_y).

    eps1, eps2 absolute relative permittivities; k0 vacuum wavenumber.
    Special: eps2 = 'pec' -> R = -1 (TE) / +1 (TM).
    """
    if isinstance(eps2, str):
        return np.full(np.shape(q), -1.0 if pol == 2 else 1.0, dtype=complex)
    k2 = k0 * np.sqrt(eps2 + 0j)
    a1 = alpha(q, k1)
    a2 = alpha(q, k2)
    if pol == 2:
        return (a1 - a2) / (a1 + a2)
    return (a1 / eps1 - a2 / eps2) / (a1 / eps1 + a2 / eps2)


def gl(n, a, b):
    x, w = np.polynomial.legendre.leggauss(n)
    return 0.5 * (b - a) * x + 0.5 * (b + a), 0.5 * (b - a) * w


def features(k0, eps1, eps2, pol):
    k1 = k0 * np.sqrt(eps1 + 0j)
    f = [k1]
    if not isinstance(eps2, str):
        f.append(k0 * np.sqrt(eps2 + 0j))
        if pol == 1 and np.real(eps2) < 0:
            f.append(k1 * np.sqrt(eps2 / (eps1 + eps2)))
    return k1, f


def make_path(k0, eps1, eps2, pol, z_min, x_max, n_hump=None, osc=2.0,
              p_tail=24, delta_frac=0.4, T_frac=1.5, L_decay=38.0):
    """Half path t >= 0: hump q = t - i d sin(pi t/T) on [0,T], then real tail.

    Returns (q, wq) where wq = GL weight * dq/dt; the full path is q and -q
    (odd symmetry: q(-t) = -q(t), dq/dt even).
    """
    k1, feats = features(k0, eps1, eps2, pol)
    kr = np.real(k1)
    # Only features near the real axis constrain T (a metal's k2 sits far up).
    near = [np.real(f) for f in feats if abs(np.imag(f)) < 2 * kr and np.real(f) > 0]
    T = T_frac * max(near)
    # depth: 0.4 Re k1 bounds the exp(delta*D) growth; must also clear the most
    # negative Im of any feature below the axis (QNM: Im k < 0) with margin.
    # cap delta*D <= 6: GEMM/cos growth ~ exp(0.45 delta D) (measured, t10)
    d = max(min(delta_frac * kr, 6.0 / max(x_max, 1e-300)), 3.0 * max([-np.imag(f) for f in feats] + [0.0]))
    if n_hump is None:  # measured: 48 nodes at T/delta = 3.75 -> 1e-14
        n_hump = max(48, int(np.ceil(13.0 * T / d)))
    t, wh = gl(n_hump, 0.0, T)
    q_h = t - 1j * d * np.sin(PI * t / T)
    dq_h = 1.0 - 1j * d * (PI / T) * np.cos(PI * t / T)
    L = L_decay / z_min
    # tail panels: geometric grading from T (1/alpha varies on scale ~T there),
    # growing x2 up to w_max = min(osc_per_panel*2pi/x_max, decay 1/z_min*c)
    w_max = min(osc * 2 * PI / max(x_max, 1e-300), 8.0 / z_min, L)
    edges = [T]
    w = min(0.5 * T, w_max)
    while edges[-1] < T + L:
        edges.append(min(edges[-1] + w, T + L))
        w = min(2 * w, w_max)
    edges = np.array(edges)
    qt, wt = [], []
    for a, b in zip(edges[:-1], edges[1:]):
        x, ww = gl(p_tail, a, b)
        qt.append(x)
        wt.append(ww)
    q = np.concatenate([q_h, np.concatenate(qt)])
    wq = np.concatenate([wh * dq_h, np.concatenate(wt)])
    return q.astype(complex), wq.astype(complex), dict(T=T, delta=d, L=L)


def g_ind(X, Z, k0, eps1, eps2, pol, q, wq):
    """G_ind and source-side derivatives (d/dx', d/dz') at arrays X, Z."""
    k1 = k0 * np.sqrt(eps1 + 0j)
    a1 = alpha(q, k1)
    R = fresnel(q, k1, eps1, eps2, pol, k0)
    X = np.asarray(X, float)[..., None]
    Z = np.asarray(Z, float)[..., None]
    base = wq * R / a1 * np.exp(1j * a1 * Z)
    c = np.cos(q * X)
    s = np.sin(q * X)
    pre = 1j / (2 * PI)  # (i/4pi) * 2 from folding +-q
    G = pre * np.sum(base * c, -1)
    Gx = pre * np.sum(base * q * s, -1)          # d/dx' cos(q(x-x')) = q sin
    Gz = pre * np.sum(base * 1j * a1 * c, -1)    # d/dz' exp(i a Z) = i a
    return G, Gx, Gz


def pec_exact(X, Z, k1, pol):
    from scipy.special import hankel1
    s = -1.0 if pol == 2 else 1.0
    rho = np.hypot(X, Z)
    G = s * 0.25j * hankel1(0, k1 * rho)
    dH = -k1 * hankel1(1, k1 * rho)
    Gx = s * 0.25j * dH * (-X / rho)
    Gz = s * 0.25j * dH * (Z / rho)
    return G, Gx, Gz


def r_inf(eps1, eps2, pol):
    """Quasi-static limit of R as q -> inf on the path (TE: 0, TM: (e2-e1)/(e2+e1))."""
    if isinstance(eps2, str):
        return -1.0 if pol == 2 else 1.0
    return 0.0 if pol == 2 else (eps2 - eps1) / (eps2 + eps1)


def g_ind_sub(X, Z, k0, eps1, eps2, pol, q, wq):
    """Same as g_ind but with R_inf * (i/4) H0(k1 rho_img) subtracted from the
    integrand and added back in closed form (holomorphic in k)."""
    k1 = k0 * np.sqrt(eps1 + 0j)
    ri = r_inf(eps1, eps2, pol)
    a1 = alpha(q, k1)
    R = fresnel(q, k1, eps1, eps2, pol, k0) - ri
    X = np.asarray(X, float); Z = np.asarray(Z, float)
    base = wq * R / a1 * np.exp(1j * a1 * Z[..., None])
    c = np.cos(q * X[..., None]); s = np.sin(q * X[..., None])
    pre = 1j / (2 * PI)
    G = pre * np.sum(base * c, -1); Gx = pre * np.sum(base * q * s, -1); Gz = pre * np.sum(base * 1j * a1 * c, -1)
    # pec_exact carries sign -1 (pol 2) / +1 (pol 1); strip it and scale by R_inf
    e = pec_exact(X, Z, k1, 1)
    return G + ri * e[0], Gx + ri * e[1], Gz + ri * e[2]
