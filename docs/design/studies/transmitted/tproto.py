"""Prototype of the transmitted field over a half-space (spec verification only)."""

import numpy as np
from scipy.integrate import quad

from pysie2d.layered import (
    TAIL_DECAY,
    TAIL_EFOLDS_PER_PANEL,
    TAIL_PANEL_POINTS,
    TAIL_PERIODS_PER_PANEL,
    SommerfeldPath,
    _normalised,
    _singularities,
    alpha,
)

PI = np.pi


def tau(q, k, eps, pol):
    """T/α₁ on the code's sheet."""
    eps = _normalised(eps)
    a1, a2 = alpha(q, k), alpha(q, k * np.sqrt(eps))
    return 2.0 / (a1 + a2) if pol == 2 else 2.0 / (a1 + a2 / eps)


def t_green(path, pol, k, eps, x, z, xs, zs, z_int):
    """(G_t, dG_t/dx', dG_t/dz'), observation below, source above."""
    x, z, xs, zs = np.broadcast_arrays(*(np.asarray(v, float) for v in (x, z, xs, zs)))
    X, hs, hb = (x - xs).ravel(), (zs - z_int).ravel(), (z_int - z).ravel()
    eps = _normalised(eps)
    a1, a2 = alpha(path.q, k), alpha(path.q, k * np.sqrt(eps))
    base = path.w * tau(path.q, k, eps, pol)
    b = base * np.exp(1j * (np.outer(hs, a1) + np.outer(hb, a2)))
    qx = np.outer(X, path.q)
    c = np.cos(qx)
    pre = 1j / (2 * PI)
    g = pre * np.sum(b * c, 1)
    gx = pre * np.sum(b * path.q * np.sin(qx), 1)
    gz = pre * np.sum(b * 1j * a1 * c, 1)
    return g, gx, gz


def t_path(k, eps, pol, D, zmin, depth_scale=1.0):
    return SommerfeldPath.for_wavenumber(k, eps, pol, D, zmin, depth_scale)


# ---- independent real-axis reference (radiation-condition branch) ----------------
def _arad(k, q):
    a = np.sqrt(complex(k * k - q * q))
    if a.imag < 0 or (a.imag == 0 and a.real < 0):
        a = -a
    return a


def t_ref(X, hs, hb, k1, eps, pol, which=0):
    eps = complex(eps)
    k2 = k1 * np.sqrt(complex(eps.real, eps.imag + 0.0))

    def f(q):
        a1, a2 = _arad(k1, q), _arad(k2, q)
        t = 2 / (a1 + a2) if pol == 2 else 2 / (a1 + a2 / eps)
        fac = (np.cos(q * X), q * np.sin(q * X), 1j * a1 * np.cos(q * X))[which]
        return 1j / (2 * PI) * t * fac * np.exp(1j * (a1 * hs + a2 * hb))

    pts = [k1, k2.real]
    if pol == 1 and eps.real < 0:
        qsp = k1 * np.sqrt(eps / (1 + eps))
        qsp = qsp if qsp.real > 0 else -qsp
        pts.append(qsp.real)
        w = abs(qsp.imag)
        for m in (1, 10, 100, 1000):
            pts += [qsp.real - m * w, qsp.real + m * w]
    qmax = max(k1, abs(k2)) * 1.5 + 40.0 / (hs + hb)
    pts = sorted(p for p in pts if 0 < p < qmax)
    kw = {"epsabs": 1e-17, "epsrel": 1e-13, "limit": 4000, "points": pts}
    re = quad(lambda q: f(q).real, 0, qmax, **kw)[0]
    im = quad(lambda q: f(q).imag, 0, qmax, **kw)[0]
    return re + 1j * im


# ---- spectra, far field, flux ----------------------------------------------------
def spectrum(q, a1, bnds, z_top):
    """A(q, α₁) = Σ delt e^{−iqf + iα₁(g − z_top)} [i(q dg + α₁ df)φ − χ]."""
    out = np.zeros(np.shape(q), complex)
    for f, g, df, dg, delt, ei in bnds:
        nn = f.size
        ph = np.exp(-1j * np.outer(f, q) + 1j * np.outer(g - z_top, a1))
        poly = 1j * (np.outer(dg, q) + np.outer(df, a1)) * ei[:nn, None] - ei[nn:, None]
        out += delt * np.sum(ph * poly, 0)
    return out


def a_down(angles, bnds, k, eps, pol, z_int):
    eps = _normalised(eps)
    k2 = k * np.sqrt(eps)
    q = k2 * np.sin(angles)
    a1, a2 = alpha(q, k), alpha(q, k2)
    A = spectrum(q, a1, bnds, z_int)
    return a2 * tau(q, k, eps, pol) * np.exp(1j * a2 * z_int) * A


def flux_rule(k, eps, pol, D, zmin, n_pan=TAIL_PANEL_POINTS, widths=(1, 10, 100, 1000),
              sine_map=True):
    """Real-axis nodes/weights on (-qmax, qmax) for the interface flux."""
    eps = _normalised(eps)
    feats = _singularities(k, eps, pol)
    bp = {0.0}
    for _, s in feats:
        bp.add(abs(s.real))
        w = abs(s.imag)
        if w > 0:
            for m in widths:
                bp.update((abs(s.real) - m * w, abs(s.real) + m * w))
    k2 = k * np.sqrt(eps)
    lossless = eps.imag == 0.0
    if lossless:
        qmax = k2.real  # Re w ≡ 0 beyond Re k₂
    else:
        qmax = max(abs(s.real) for _, s in feats) * 1.5 + TAIL_DECAY / zmin
    wmax = min(TAIL_PERIODS_PER_PANEL * 2 * PI / D, TAIL_EFOLDS_PER_PANEL / zmin)
    edges = sorted(b for b in bp if 0.0 <= b < qmax) + [qmax]
    # subdivide panels wider than wmax
    fine = [edges[0]]
    for a, b in zip(edges[:-1], edges[1:]):
        n = max(1, int(np.ceil((b - a) / wmax)))
        fine += list(np.linspace(a, b, n + 1)[1:])
    x, wx = np.polynomial.legendre.leggauss(n_pan)
    qs, ws = [], []
    for a, b in zip(fine[:-1], fine[1:]):
        c, h = 0.5 * (a + b), 0.5 * (b - a)
        if sine_map:
            u = np.sin(0.5 * PI * x)
            du = 0.5 * PI * np.cos(0.5 * PI * x)
            qs.append(c + h * u)
            ws.append(h * du * wx)
        else:
            qs.append(c + h * x)
            ws.append(h * wx)
    q, w = np.concatenate(qs), np.concatenate(ws)
    return np.concatenate([-q[::-1], q]), np.concatenate([w[::-1], w])


def flux_weight(q, k, eps, pol):
    eps = _normalised(eps)
    a2 = alpha(q, k * np.sqrt(eps))
    return (a2 if pol == 2 else a2 / eps).real


def p_down_flux(q, w, A, k, eps, pol):
    return np.sum(w * flux_weight(q, k, eps, pol) * np.abs(tau(q, k, eps, pol) * A) ** 2) / (8 * PI)
