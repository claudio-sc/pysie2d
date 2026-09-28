"""Periodic (Bloch) correction to the exterior Green function — spike sandbox.

G_per(r|r') = (i/4) Σ_n e^{i k_B n L} H₀⁽¹⁾(k|r − r' − nL x̂|). The n = 0 term is
pysie2d's free-space kernel (Kress-corrected already); this module supplies the
smooth remainder ΔG = (i/4)(S⁺ + S⁻) by Maradudin (2020) eqs. 1.87/1.92
(Veysoglu et al.), re-derived from the Sommerfeld integral with
β = k(1 + i u²), α = k u w, w = √(u² − 2i) (Re w > 0, Im w < 0):

    S⁺ = Σ_{p≥1} e^{i k_B pL} H₀(k√((pL−Δx)² + Δz²))
       = (4/πi) e^{−ikΔx} q⁺ ∫₀^∞ e^{−k(L−Δx)u²} / (1 − q⁺ e^{−kLu²}) · cos(Δz k u w)/w du,
    q⁺ = e^{i(k_B + k)L};   S⁻: Δx → −Δx, k_B → −k_B.

Valid for |Δx| < L. At complex k the cos factor grows like e^{|Δz| Im k u²}
against the decay e^{−Re k (L∓Δx) u²}: the integral exists only while
Re k·(L − |Δx|) > |Δz|·Im k, asserted below (the Maradudin derivation assumes
a real Hankel argument; the result is its analytic continuation where it
converges, checked against the damped direct sum in check_kernel.py).

Rayleigh anomalies (q± → 1) are not handled: the spike stays far from them
(L = 340 nm, λ ≥ 500 nm, k_B = 0 puts the first at λ = 340 nm).
"""

import numpy as np

from pysie2d.kernels import assemble_cross_block

_S_MAX = 7.0  # u·√(Re k·a) cut-off: e^{−49} ≈ 5e-22 below round-off
_N_U = 400  # Gauss–Legendre nodes on [0, _S_MAX]; ≈ 20 cos oscillations max
_S, _WS = np.polynomial.legendre.leggauss(_N_U)
_S = 0.5 * _S_MAX * (_S + 1.0)
_WS = 0.5 * _S_MAX * _WS


def _tail(k, kb, L, dx, dz):
    """S⁺-type tail and its Δx, Δz derivatives, vectorised over pairs.

    Returns (S, dS/dΔx, dS/dΔz) for Σ_{p≥1} e^{i kb pL} H₀(k√((pL−Δx)²+Δz²)).
    """
    a = L - dx
    ok = k.real * a > np.abs(dz) * abs(k.imag) + 1e-12
    if not np.all(ok):
        raise ValueError("Veysoglu integral diverges: Re k (L-|Δx|) <= |Δz| Im k")
    scale = 1.0 / np.sqrt(k.real * a)  # per-pair u range
    u = scale[..., None] * _S  # (..., N_U)
    wu = scale[..., None] * _WS
    u2 = u * u
    w = np.sqrt(u2 - 2j)
    w = np.where(w.real < 0, -w, w)
    q = np.exp(1j * (kb + k) * L)
    base = np.exp(-k * a[..., None] * u2) / (1.0 - q * np.exp(-k * L * u2))
    arg = dz[..., None] * k * u * w
    cw = np.cos(arg) / w
    pref = (4.0 / (np.pi * 1j)) * np.exp(-1j * k * dx) * q
    s = pref * np.sum(wu * base * cw, axis=-1)
    # d/dΔx of e^{−ikΔx} e^{kΔx u²} is k(u² − i)·(same)
    s_dx = pref * k * np.sum(wu * base * (u2 - 1j) * cw, axis=-1)
    s_dz = -pref * k * np.sum(wu * base * u * np.sin(arg), axis=-1)
    return s, s_dx, s_dz


def delta_g(k, kb, L, dx, dz):
    """ΔG and its derivatives w.r.t. Δx = x − x', Δz = z − z' (pysie2d normalisation)."""
    k = complex(k)
    p, p_dx, p_dz = _tail(k, kb, L, dx, dz)
    m, m_dx, m_dz = _tail(k, -kb, L, -dx, dz)
    return 0.25j * (p + m), 0.25j * (p_dx - m_dx), 0.25j * (p_dz + m_dz)


def delta_g_direct(k, kb, L, dx, dz, n_max):
    """Brute image sum Σ_{0<|n|≤n_max}; converges only for Im k > 0."""
    from pysie2d.kernels import hank0, hank1

    g = np.zeros(dx.shape, complex)
    gx = np.zeros(dx.shape, complex)
    gz = np.zeros(dx.shape, complex)
    for n in range(-n_max, n_max + 1):
        if n == 0:
            continue
        X = dx - n * L
        r = np.sqrt(X**2 + dz**2)
        ph = 0.25j * np.exp(1j * kb * n * L)
        g += ph * hank0(k * r)
        h1 = -ph * k * hank1(k * r) / r  # d/dr of (i/4)H₀ is −(i/4)k H₁
        gx += h1 * X
        gz += h1 * dz
    return g, gx, gz


def periodic_blocks(k, kb, L, gp, gq):
    """ΔG contribution to the (M1, M2) exterior block, field q ← source p.

    Same structure as assemble_cross_block: m1 = h(G_x' dg − G_z' df),
    m2 = h·G, with ∂/∂x' = −∂/∂Δx.
    """
    h = 2.0 * np.pi / gp.n_pts
    dx = gq.f[:, None] - gp.f[None, :]
    dz = gq.g[:, None] - gp.g[None, :]
    g, g_dx, g_dz = delta_g(k, kb, L, dx, dz)
    m1 = h * (-g_dx * gp.dg[None, :] + g_dz * gp.df[None, :])
    return m1, h * g


def periodic_blocks_direct(k, kb, L, gp, gq, n_max):
    """Reference: the same block by summing shifted assemble_cross_block copies."""
    m1 = np.zeros((gq.n_pts, gp.n_pts), complex)
    m2 = np.zeros_like(m1)
    for n in range(-n_max, n_max + 1):
        if n == 0:
            continue
        a, b = assemble_cross_block(
            k, gp.n_pts, gp.f + n * L, gp.g, gp.df, gp.dg, gq.f, gq.g
        )
        ph = np.exp(1j * kb * n * L)
        m1 += ph * a
        m2 += ph * b
    return m1, m2


def assemble_periodic(solver, wavelength, kb, L):
    """ClusterBIESolver matrix with every exterior block made Bloch-periodic."""
    cl = solver.cluster
    k = solver.materials[0].wnum_bg(wavelength)
    me = solver._assemble(wavelength).astype(complex)
    for q, gq in enumerate(cl.geometries):
        for p, gp in enumerate(cl.geometries):
            m1, m2 = periodic_blocks(k, kb, L, gp, gq)
            r0, c0 = cl.offsets[q], cl.offsets[p]
            me[r0 : r0 + gq.n_pts, c0 : c0 + gp.n_pts] += m1
            me[r0 : r0 + gq.n_pts, c0 + gp.n_pts : c0 + 2 * gp.n_pts] += m2
    return me


def orders(solver, ei, wavelength, kb, L, angle):
    """Reflectance and transmittance of the propagating Rayleigh orders.

    Upward order m of G_per = (i/2Lβ_m) e^{iα_mΔx + iβ_m|Δz|}; the scattered
    field uses the representation of pysie2d.fields._representation_at,
    u_s = h Σ [−(G_x' dg − G_z' df)φ − Gχ].
    """
    cl = solver.cluster
    k = solver.materials[0].wnum_bg(wavelength)
    kz_inc = k * np.cos(np.deg2rad(angle))
    R = T = 0.0
    m_max = int(np.ceil(abs(k) * L / (2 * np.pi))) + 1
    for m in range(-m_max, m_max + 1):
        al = kb + 2 * np.pi * m / L
        be = np.sqrt(complex(k**2 - al**2))
        if abs(be.imag) > 1e-12 * abs(k):
            continue
        up = dn = 0.0
        for p, gp in enumerate(cl.geometries):
            s = ei[cl.slice(p)]
            phi, chi = s[: gp.n_pts], s[gp.n_pts :]
            h = 2 * np.pi / gp.n_pts
            eu = np.exp(-1j * al * gp.f - 1j * be * gp.g)
            ed = np.exp(-1j * al * gp.f + 1j * be * gp.g)
            up += h * np.sum(eu * (-(-1j * al * gp.dg + 1j * be * gp.df) * phi - chi))
            dn += h * np.sum(ed * (-(-1j * al * gp.dg - 1j * be * gp.df) * phi - chi))
        c = 1j / (2 * L * be)
        A, B = c * up, c * dn + (1.0 if m == 0 else 0.0)
        R += abs(A) ** 2 * be.real / kz_inc.real
        T += abs(B) ** 2 * be.real / kz_inc.real
    return R, T
