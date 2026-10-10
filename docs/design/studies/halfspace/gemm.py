"""Separable (GEMM) reflected blocks M1_ind, M2_ind on nn boundary nodes.

G_ind[i,j] = (i/4pi) sum_m c_m A_im B_jm,  c_m = w_m R_m / a_m  over the full
odd-symmetric path {q_m, -q_m};  A_im = exp(i q x_i + i a h_i), B_jm = exp(-i q x_j + i a h_j),
x measured from a chosen origin x_c (centring), h = z - z_int.
M2 = hstep * G,  M1 = hstep * (dG/dx' dg_j - dG/dz' df_j)  (assemble_cross_block convention).
"""
import numpy as np
from hs import alpha, fresnel, gl, features, PI


def full_path(q, wq):
    return np.concatenate([q, -q]), np.concatenate([wq, wq])


def blocks_from_path(q, wq, k0, eps1, eps2, pol, f, g, df, dg, z_int, x_c, rows=None, ri=0.0):
    k1 = k0 * np.sqrt(eps1 + 0j)
    a = alpha(q, k1)
    c = (1j / (4 * PI)) * wq * (fresnel(q, k1, eps1, eps2, pol, k0) - ri) / a
    x = f - x_c
    h = g - z_int
    if rows is not None:
        x, h, dfr, dgr = x[rows], h[rows], df[rows], dg[rows]
    else:
        dfr, dgr = df, dg
    A = np.exp(1j * (np.outer(x, q) + np.outer(h, a)))
    B = np.exp(1j * (-np.outer(x, q) + np.outer(h, a)))
    Ac = A * c
    B1 = B * (-1j * np.outer(dgr, q) - 1j * np.outer(dfr, a))
    G = Ac @ B.T
    G1 = Ac @ B1.T
    return G, G1, (np.abs(A) * np.abs(c)) @ np.abs(B).T


def reflected_blocks(k0, eps1, eps2, pol, geom_f, geom_g, geom_df, geom_dg, z_int,
                     s=1.0, x_c=None, banded=True, L_decay=38.0, p=24, subtract=False,
                     dD_cap=6.0, osc_hump=None):
    """Common-path GEMM for the whole nn x nn block. banded: the long quasi-static
    tail is applied only to the sub-block of low nodes that it can reach."""
    f, g, df, dg = geom_f, geom_g, geom_df, geom_dg
    nn = len(f)
    hstep = 2 * PI / nn
    if x_c is None:
        x_c = 0.5 * (f.max() + f.min())
    h = g - z_int
    hmin = h.min()
    k1, feats = features(k0, eps1, eps2, pol)
    kr = np.real(k1)
    near = [np.real(v) for v in feats if abs(np.imag(v)) < 2 * kr and np.real(v) > 0]
    T = 1.5 * max(near)
    D = np.ptp(f)
    d0 = 0.4 * kr if dD_cap is None else min(0.4 * kr, dD_cap / D)
    d = max(d0, 3.0 * max([-np.imag(v) for v in feats] + [0.0]))
    n_osc = 0 if osc_hump is None else np.ceil(T * D / (2 * PI) * osc_hump)
    nh = int(round(s * max(48, np.ceil(13.0 * T / d), n_osc)))
    t, wh = gl(nh, 0.0, T)
    qh = t - 1j * d * np.sin(PI * t / T)
    wqh = wh * (1.0 - 1j * d * (PI / T) * np.cos(PI * t / T))
    # hump + first tail stretch act on all nodes; build tail band by band
    G = np.zeros((nn, nn), complex); G1 = np.zeros((nn, nn), complex); Gabs = np.zeros((nn, nn))
    segs = [(np.arange(nn), qh, wqh)]
    qend = T + L_decay / (2 * hmin)
    a0 = T
    w = 0.5 * T
    M = 2 * nh
    while a0 < qend:
        # active nodes: pair decay exp(-q (h_i + h_j)) with q >= a0
        act = np.nonzero(h <= L_decay / a0 - hmin)[0] if banded else np.arange(nn)
        span = max(np.ptp(f[act]), 2 * hmin) if len(act) > 1 else 2 * hmin
        w_max = min(4.0 / s * 2 * PI / span, 8.0 / (2 * hmin))
        w = min(w, w_max)
        # group panels with the same active set into one band of length ~ 4 w_max
        b_end = min(a0 + max(w, 0.0), qend)
        qs, ws = [], []
        a = a0
        band_len = max(4 * w_max, w)
        while a < min(a0 + band_len, qend):
            b = min(a + w, qend)
            x_, w_ = gl(p, a, b)
            qs.append(x_); ws.append(w_)
            a = b
            w = min(2 * w, w_max)
        segs.append((act, np.concatenate(qs).astype(complex), np.concatenate(ws).astype(complex)))
        M += 2 * len(np.concatenate(qs))
        a0 = a
    ri = 0.0
    if subtract:
        from hs import r_inf, pec_exact
        ri = r_inf(eps1, eps2, pol)
        if ri != 0.0:
            X = f[:, None] - f[None, :]; Z = h[:, None] + h[None, :]
            Gi, Gxi, Gzi = pec_exact(X, Z, k1, 1)   # +(i/4)H0 and its source derivatives
            G += ri * Gi; G1 += ri * (Gxi * dg[None, :] - Gzi * df[None, :])
    for act, q, wq in segs:
        qf, wf = full_path(q, wq)
        Gs, G1s, Ga = blocks_from_path(qf, wf, k0, eps1, eps2, pol, f, g, df, dg, z_int, x_c, rows=act, ri=ri)
        ix = np.ix_(act, act)
        G[ix] += Gs; G1[ix] += G1s; Gabs[ix] += Ga
    return hstep * G1, hstep * G, M, Gabs
