"""Reflected Green function of a half-space, by Sommerfeld integrals on a deformed path.

Frame: x horizontal, z vertical, the interface horizontal at ``z = z_int``, the
cover above it. Every source and observation point lies in the cover. Time
convention ``exp(-iωt)``. With ``X = x − x'`` and ``Z = z + z' − 2·z_int > 0``,

    G_ind = (i/4π) ∫_Γ R(q)/α₁ · exp(i q X + i α₁ Z) dq,      α_j = √(k_j² − q²).

Everything here takes the **background** wavenumber ``k`` and the
**background-relative** permittivity ``eps_rel = ε_sub/n_clad²`` — never a
wavelength — so the vacuum-to-background conversion happens once, in
``Material.wnum_bg`` (conventions §2.3). The path is built in units of ``k``,
which is also what keeps scale covariance (§9) intact.

Why a deformed path and not the real axis: for complex ``k`` the real-axis
integral with a per-point ``Im α ≥ 0`` rule computes the *incoming* Green
function, not the continuation of the outgoing one (holomorphy note §1;
pinned by ``reference.sommerfeld.legacy_pec_green``). The deformed path winds
between the branch points ``±k`` and is holomorphic in ``k`` as long as no
singularity crosses it (clearance C1, asserted in :meth:`SommerfeldPath.for_box`).

The theory is ``docs/design/sommerfeld-holomorphy.md``; every constant below is
the one measured in ``docs/design/studies/halfspace/``.
"""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from .geometry import Geometry
from .kernels import hank0, hank1

PI = np.pi

# --- Path constants. Each is the value measured in studies/halfspace. ---------

# Hump depth δ ≤ 0.4·Re k₁: keeps the hump inside the region where the
# integrand is analytic without letting exp(i q X) grow faster than the
# Gauss–Legendre rate can absorb (t5, t3b).
HUMP_DEPTH_FRAC = 0.4
# δ·D ≤ 6: the GEMM factors grow like exp(0.45·δ·D) (t10); R = 4000 gave an O(1)
# error without the cap (study trap 2).
GROWTH_CAP = 6.0
# δ ≥ 3·(−Im) of every singularity: lifts the hump clear of +k₁, +k₂ and q_sp
# when Im k < 0 (the QNM case).
FEATURE_FLOOR = 3.0
# T = 1.5·max Re(singularity): the hump ends past every near-real singularity,
# and C2 (T² > max Re k²) holds with room.
T_FRAC = 1.5
# n_hump = max(48, ⌈13·T/δ⌉): 48 nodes at T/δ = 3.75 gave 1e-14, and Si failed
# at 1e-5 when the count did not scale with T/δ (study trap 2).
HUMP_NODES_PER_T_OVER_DELTA = 13.0
MIN_HUMP_NODES = 48
# Tail on the real axis to T + 38/Z_min: exp(−38) = 3e-17 is below round-off.
TAIL_DECAY = 38.0
# 24-point Gauss–Legendre panels; the width starts at T/2 and doubles. Uniform
# panels stall at 1e-7 near q = T (study trap 1).
TAIL_PANEL_POINTS = 24
# Panel width cap min(4·2π/D, 8/Z_min): four oscillations of exp(i q X) or
# 8 e-folds of exp(−q Z) per panel.
TAIL_PERIODS_PER_PANEL = 4.0
TAIL_EFOLDS_PER_PANEL = 8.0
# A singularity must clear the path by this fraction of the hump depth.
CLEARANCE = 0.2
# λi_max/λr_min ≤ tan 45° (Q ≥ 0.5): the branch points rotate towards the
# imaginary axis as Re λ → 0 and the depth diverges (holomorphy §8).
MAX_TAN_ARG = 1.0
# δ·D > 3·ln 10: the GEMM loses about δ·D/ln 10 digits (holomorphy §7).
MAX_DIGITS_LOST = 3.0
# |den|/|α₁| at the quasi-static plasmon root: ~1e-15 on the proper sheet,
# ~2 off it (t4, t13).
PLASMON_SHEET_TOL = 1e-6
# spacing/(2·gap) at which the trapezoid rule on the image kernel stops
# reaching 1e-12 (study trap 4, t12, t14).
GAP_RATIO_TE = 0.25
GAP_RATIO_TM = 0.2


class SommerfeldPrecisionWarning(UserWarning):
    """The reflected blocks are expected to lose digits (δ·D > 3·ln 10)."""


class InterfaceGapWarning(UserWarning):
    """The boundary is too coarse for its distance to the interface."""


def alpha(q: np.ndarray | complex, k: complex) -> np.ndarray | complex:
    """Vertical-cut sheet of ``√(k² − q²)``, with ``α(0) = +k``.

    The cuts run up from ``+k`` and down from ``−k``. The sheet is continuous on
    any path that passes below ``+k`` and above ``−k`` — the Sommerfeld path for
    ``Im k`` of either sign — and gives ``α → i|q|`` on both real tails.

    Args:
        q: Transverse wavenumber, any shape.
        k: Wavenumber (complex allowed).

    Returns:
        ``α(q)`` with the shape of ``q``.
    """
    return 1j * np.sqrt(1j * (q - k)) * np.sqrt(-1j * (q + k))


def _normalised(eps_rel: complex) -> complex:
    """Signed-zero normalisation: ``np.sqrt(complex(-4, -0.0)) == −2j`` otherwise."""
    eps = complex(eps_rel)
    return complex(eps.real, eps.imag + 0.0)


def fresnel_r(
    q: np.ndarray | complex, k1: complex, eps_rel: complex | None, pol: int
) -> np.ndarray:
    """Cover-side reflection coefficient of the half-space.

    TE (``pol = 2``): ``(α₁ − α₂)/(α₁ + α₂)``. TM (``pol = 1``), admittance form:
    ``(α₁ − α₂/ε)/(α₁ + α₂/ε)`` with ``ε = eps_rel``. ``eps_rel is None`` is
    PEC, the sentinel ``R ≡ −1`` (TE) / ``+1`` (TM).

    Args:
        q: Transverse wavenumber, any shape.
        k1: Cover wavenumber (the background ``k``).
        eps_rel: Substrate permittivity relative to the cover, or ``None``.
        pol: 2 = TE, 1 = TM.

    Returns:
        Complex ``R(q)`` with the shape of ``q``.
    """
    if eps_rel is None:
        return np.full(np.shape(q), -1.0 if pol == 2 else 1.0, dtype=complex)
    eps = _normalised(eps_rel)
    a1 = alpha(q, k1)
    a2 = alpha(q, k1 * np.sqrt(eps))
    if pol == 2:
        return (a1 - a2) / (a1 + a2)
    b2 = a2 / eps
    return (a1 - b2) / (a1 + b2)


def _singularities(
    k: complex, eps_rel: complex | None, pol: int
) -> list[tuple[str, complex]]:
    """Right-hand singularities of the integrand: ``+k₁``, ``+k₂``, and q_sp.

    The TM plasmon ``q_sp = k₁·√(ε/(1+ε))`` is included only if it is a zero of
    the TM denominator **on the code's own α sheet**: off that sheet there is
    nothing to clear (holomorphy §5), and at ``−1 < Re ε < 0`` it routinely is.
    """
    feats = [("+k₁", complex(k))]
    if eps_rel is None:
        return feats
    eps = _normalised(eps_rel)
    k2 = k * np.sqrt(eps)
    feats.append(("+k₂", complex(k2)))
    if pol == 1 and eps.real < 0:
        q_sp = k * np.sqrt(eps / (1.0 + eps))
        if q_sp.real < 0:  # α is even in q: −q_sp is the same zero
            q_sp = -q_sp
        den = alpha(q_sp, k) + alpha(q_sp, k2) / eps
        if abs(den) < PLASMON_SHEET_TOL * abs(alpha(q_sp, k)):
            feats.append(("plasmon q_sp", complex(q_sp)))
    return feats


@dataclass(frozen=True, eq=False)
class SommerfeldPath:
    """Half path ``q ≥ 0`` (the full path is ``±q``) with its weights and bands.

    ``q[:n_hump]`` is the hump ``q = t − iδ·sin(πt/T)``; the rest is the real
    tail in graded panels. ``bands`` groups the tail so that
    :func:`reflected_blocks` applies each group only to the nodes it can reach.

    Attributes:
        q: (n,) complex nodes on the half path.
        w: (n,) complex weights, Gauss–Legendre weight × ``dq/dt``.
        bands: ``(a0, i0, i1)`` triples. ``a0`` is the band's first ``q`` (0 for
            the hump, which acts on every node); ``q[i0:i1]`` its nodes.
        T: Hump length.
        delta: Hump depth.
        D: Horizontal extent the path was sized for.
        z_min: Smallest ``Z = z + z' − 2·z_int`` it was sized for.
    """

    q: np.ndarray
    w: np.ndarray
    bands: tuple[tuple[float, int, int], ...]
    T: float
    delta: float
    D: float
    z_min: float

    @classmethod
    def for_wavenumber(
        cls,
        k: complex,
        eps_rel: complex | None,
        pol: int,
        D: float,
        z_min: float,
        depth_scale: float = 1.0,
    ) -> SommerfeldPath:
        """Path for one wavenumber (driven solves).

        Args:
            k: Background wavenumber (rad/nm), complex allowed.
            eps_rel: Substrate permittivity relative to the cover, or ``None``
                for PEC.
            pol: 2 = TE, 1 = TM.
            D: Horizontal extent of everything the path serves (nm).
            z_min: Smallest ``z + z' − 2·z_int`` it serves (nm).
            depth_scale: Multiplies the hump depth δ. Only for checking path
                independence (G3, G9); the default is the measured rule.

        Returns:
            The path.

        Raises:
            ValueError: ``Re k ≤ 0``, the argument of ``k`` beyond 45°, or
                clearance C1 fails.
        """
        return cls._build(
            np.atleast_1d(np.asarray(k, dtype=complex)),
            eps_rel,
            pol,
            D,
            z_min,
            depth_scale,
        )

    @classmethod
    def for_box(
        cls,
        k_boundary: Sequence[complex] | np.ndarray,
        eps_rel: complex | None,
        pol: int,
        D: float,
        z_min: float,
        depth_scale: float = 1.0,
    ) -> SommerfeldPath:
        """One path for a whole QNM search box (Beyn needs one holomorphic M(λ)).

        ``Im k₁`` and ``Im q_sp`` are harmonic in λ, so the worst clearance is on
        the boundary of the **shadow box** B̄ = [λr_min, λr_max] × [0, λi_max],
        the box extended down to the real axis (holomorphy §4). The caller passes
        ``k`` sampled densely along ∂B̄; the path is sized for the worst sample
        and **asserts** clearance C1 on all of them, because a singularity
        crossing the path shows up as a cut that rank detection does not catch.

        Args:
            k_boundary: Background wavenumbers sampled along ∂B̄.
            eps_rel: Substrate permittivity relative to the cover, or ``None``.
            pol: 2 = TE, 1 = TM.
            D: Horizontal extent of everything the path serves (nm).
            z_min: Smallest ``z + z' − 2·z_int`` it serves (nm).
            depth_scale: As :meth:`for_wavenumber`.

        Returns:
            The path.

        Raises:
            ValueError: As :meth:`for_wavenumber`; the message names the
                singularity and the offending sample.
        """
        return cls._build(
            np.atleast_1d(np.asarray(k_boundary, dtype=complex)),
            eps_rel,
            pol,
            D,
            z_min,
            depth_scale,
        )

    @classmethod
    def _build(
        cls,
        ks: np.ndarray,
        eps_rel: complex | None,
        pol: int,
        D: float,
        z_min: float,
        depth_scale: float,
    ) -> SommerfeldPath:
        if not (D > 0.0 and z_min > 0.0):
            raise ValueError(f"need D > 0 and z_min > 0, got D = {D}, z_min = {z_min}")
        if np.any(ks.real <= 0.0):
            raise ValueError(f"Re k must be positive, got k = {ks[ks.real <= 0][0]}")
        tan_arg = -ks.imag / ks.real
        if np.any(tan_arg > MAX_TAN_ARG):
            worst = ks[np.argmax(tan_arg)]
            raise ValueError(
                f"|Im k|/Re k = {tan_arg.max():.3g} exceeds tan 45° at k = {worst}: "
                "the hump depth diverges as Re λ → 0 (holomorphy note §8)"
            )

        samples = [(i, _singularities(k, eps_rel, pol)) for i, k in enumerate(ks)]
        near = [
            f.real
            for i, feats in samples
            for _, f in feats
            if f.real > 0 and abs(f.imag) < 2.0 * ks[i].real
        ]
        T = T_FRAC * max(near)
        floor = FEATURE_FLOOR * max(
            [-f.imag for _, feats in samples for _, f in feats] + [0.0]
        )
        delta = depth_scale * max(
            min(HUMP_DEPTH_FRAC * ks.real.min(), GROWTH_CAP / D), floor
        )

        for i, feats in samples:
            for label, f in feats:
                gamma = delta * np.sin(PI * f.real / T) if 0.0 < f.real < T else 0.0
                if f.imag + gamma < CLEARANCE * delta:
                    raise ValueError(
                        f"clearance C1 fails: {label} = {f:.6g} at k = {ks[i]:.6g} "
                        f"(sample {i}) is within {f.imag + gamma:.3g} of the "
                        f"Sommerfeld path (hump depth {delta:.3g}, length {T:.3g})"
                    )

        if delta * D > MAX_DIGITS_LOST * np.log(10.0):
            warnings.warn(
                f"δ·D = {delta * D:.2f} exceeds 3·ln 10: the reflected blocks lose "
                f"about {delta * D / np.log(10.0):.1f} digits",
                SommerfeldPrecisionWarning,
                stacklevel=3,
            )

        n_hump = max(
            MIN_HUMP_NODES, int(np.ceil(HUMP_NODES_PER_T_OVER_DELTA * T / delta))
        )
        t, wt = _gauss_legendre(n_hump, 0.0, T)
        q = [t - 1j * delta * np.sin(PI * t / T)]
        w = [wt * (1.0 - 1j * delta * (PI / T) * np.cos(PI * t / T))]
        bands = [(0.0, 0, n_hump)]

        length = TAIL_DECAY / z_min
        w_max = min(
            TAIL_PERIODS_PER_PANEL * 2.0 * PI / D, TAIL_EFOLDS_PER_PANEL / z_min, length
        )
        width = min(0.5 * T, w_max)
        start = T
        band_start, band_i0, n_nodes = T, n_hump, n_hump
        while start < T + length:
            stop = min(start + width, T + length)
            if start >= band_start + TAIL_PERIODS_PER_PANEL * w_max:
                bands.append((band_start, band_i0, n_nodes))
                band_start, band_i0 = start, n_nodes
            x, wx = _gauss_legendre(TAIL_PANEL_POINTS, start, stop)
            q.append(x.astype(complex))
            w.append(wx.astype(complex))
            n_nodes += TAIL_PANEL_POINTS
            start, width = stop, min(2.0 * width, w_max)
        bands.append((band_start, band_i0, n_nodes))
        return cls(
            q=np.concatenate(q),
            w=np.concatenate(w),
            bands=tuple(bands),
            T=float(T),
            delta=float(delta),
            D=float(D),
            z_min=float(z_min),
        )


def _gauss_legendre(n: int, a: float, b: float) -> tuple[np.ndarray, np.ndarray]:
    x, w = np.polynomial.legendre.leggauss(n)
    return 0.5 * (b - a) * x + 0.5 * (b + a), 0.5 * (b - a) * w


def _check_in_cover(z: np.ndarray, z_int: float, what: str) -> None:
    if np.any(z <= z_int):
        raise ValueError(
            f"{what} must lie strictly above the interface z_int = {z_int}; "
            f"lowest z = {np.min(z)}"
        )


def check_interface_gap(geometries: Sequence[Geometry], z_int: float, pol: int) -> None:
    """Warn when the boundary is too coarse for its distance to the interface.

    The image kernel is near-singular at distance ``2·gap`` and the plain
    trapezoid rule on it reaches 1e-12 only for ``spacing/(2·gap) ≲ 0.25`` (TE) or
    ``0.2`` (TM) (study trap 4). The spacing is the **local** one at each node,
    not the mean, so a flat facet lying near the interface is caught (t14).

    Args:
        geometries: Every particle boundary.
        z_int: Interface height (nm).
        pol: 2 = TE, 1 = TM.

    Raises:
        ValueError: A node is at or below the interface (D5).
    """
    limit = GAP_RATIO_TE if pol == 2 else GAP_RATIO_TM
    for geom in geometries:
        _check_in_cover(geom.g, z_int, "every boundary node")
        spacing = np.hypot(geom.df, geom.dg) * geom.delt
        ratio = float(np.max(spacing / (2.0 * (geom.g - z_int))))
        if ratio > limit:
            warnings.warn(
                f"local node spacing / (2·gap) = {ratio:.3g} exceeds {limit}: the "
                "image kernel is under-resolved; raise n_pts",
                InterfaceGapWarning,
                stacklevel=3,
            )


def _pec_terms(
    k: complex, pol: int, X: np.ndarray, Z: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """∓(i/4)·H₀^{(1)}(kρ) and its source-side derivatives; also ρ.

    The image of the source through a PEC plane: the exact reflected field, so
    no Sommerfeld integral and no path (owner decision 5).
    """
    s = -1.0 if pol == 2 else 1.0
    rho = np.hypot(X, Z)
    u = k * rho
    h1 = hank1(u)
    g = s * 0.25j * hank0(u)
    gx = s * 0.25j * k * h1 * (X / rho)
    gz = -s * 0.25j * k * h1 * (Z / rho)
    return g, gx, gz, rho


def reflected_green(
    path: SommerfeldPath | None,
    pol: int,
    k: complex,
    eps_rel: complex | None,
    x: np.ndarray | float,
    z: np.ndarray | float,
    xs: np.ndarray | float,
    zs: np.ndarray | float,
    z_int: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Reflected Green function at observation ``(x, z)`` from a source ``(xs, zs)``.

    Args:
        path: The Sommerfeld path (unused, may be ``None``, for PEC).
        pol: 2 = TE, 1 = TM.
        k: Background wavenumber.
        eps_rel: Substrate permittivity relative to the cover, or ``None`` for PEC.
        x: Observation x; broadcast against the source's.
        z: Observation z.
        xs: Source x.
        zs: Source z.
        z_int: Interface height.

    Returns:
        ``(G, dG/dx', dG/dz')``, the source-side derivatives, broadcast shape.

    Raises:
        ValueError: A point is at or below the interface.
    """
    x, z, xs, zs = np.broadcast_arrays(
        *(np.asarray(v, dtype=float) for v in (x, z, xs, zs))
    )
    _check_in_cover(z, z_int, "observation points")
    _check_in_cover(zs, z_int, "source points")
    X, Z = x - xs, z + zs - 2.0 * z_int
    if eps_rel is None:
        return _pec_terms(k, pol, X, Z)[:3]

    assert path is not None
    a = alpha(path.q, k)
    base = path.w * fresnel_r(path.q, k, eps_rel, pol) / a
    pre = 1j / (2.0 * PI)  # (i/4π)·2: the integral is folded onto q ≥ 0
    shape = X.shape
    Xf, Zf = X.ravel(), Z.ravel()
    out = np.empty((3, Xf.size), dtype=complex)
    chunk = max(1, 4_000_000 // path.q.size)
    for lo in range(0, Xf.size, chunk):
        sl = slice(lo, lo + chunk)
        b = base * np.exp(1j * np.outer(Zf[sl], a))
        qx = np.outer(Xf[sl], path.q)
        c = np.cos(qx)
        out[0, sl] = pre * np.sum(b * c, axis=1)
        out[1, sl] = pre * np.sum(b * path.q * np.sin(qx), axis=1)
        out[2, sl] = pre * np.sum(b * 1j * a * c, axis=1)
    return out[0].reshape(shape), out[1].reshape(shape), out[2].reshape(shape)


def _reflected_blocks(
    path: SommerfeldPath | None,
    pol: int,
    k: complex,
    eps_rel: complex | None,
    tgt: Geometry,
    src: Geometry,
    z_int: float,
    x_c: float,
    derivative: bool,
) -> tuple[np.ndarray, np.ndarray]:
    h_t, h_s = tgt.g - z_int, src.g - z_int
    _check_in_cover(tgt.g, z_int, "field-particle boundary")
    _check_in_cover(src.g, z_int, "source-particle boundary")
    step = src.delt

    if eps_rel is None:
        X = tgt.f[:, None] - src.f[None, :]
        Z = h_t[:, None] + h_s[None, :]
        s = -1.0 if pol == 2 else 1.0
        rho = np.hypot(X, Z)
        u = k * rho
        c = X * src.dg[None, :] + Z * src.df[None, :]
        if derivative:
            # d/dk[k·H₁(kρ)] = k·ρ·H₀(kρ), and d/dk H₀(kρ) = −ρ·H₁(kρ).
            return (
                step * s * 0.25j * k * hank0(u) * c,
                -step * s * 0.25j * rho * hank1(u),
            )
        return (
            step * s * 0.25j * k * hank1(u) / rho * c,
            step * s * 0.25j * hank0(u),
        )

    assert path is not None
    x_t, x_s = tgt.f - x_c, src.f - x_c
    h_min_t, h_min_s = h_t.min(), h_s.min()
    m1 = np.zeros((tgt.n_pts, src.n_pts), dtype=complex)
    m2 = np.zeros_like(m1)
    eps = _normalised(eps_rel)
    k2 = k * np.sqrt(eps)

    for a0, i0, i1 in path.bands:
        if a0 == 0.0:
            rows, cols = np.arange(tgt.n_pts), np.arange(src.n_pts)
        else:
            # |A·B| ~ exp(−q·(h_i + h_j)): a pair is reached by this band only if
            # h_i + h_j < 38/q for some q ≥ a0.
            lim = TAIL_DECAY / a0
            rows = np.flatnonzero(h_t < lim - h_min_s)
            cols = np.flatnonzero(h_s < lim - h_min_t)
            if rows.size == 0 or cols.size == 0:
                continue
        # The folded half path and its mirror: A·diag(c)·Bᵀ needs the full path.
        q = np.concatenate([path.q[i0:i1], -path.q[i0:i1]])
        w = np.concatenate([path.w[i0:i1], path.w[i0:i1]])
        a1 = alpha(q, k)
        r = fresnel_r(q, k, eps_rel, pol)
        coef = (1j / (4.0 * PI)) * w * r / a1
        e_t = np.exp(1j * (np.outer(x_t[rows], q) + np.outer(h_t[rows], a1)))
        e_s = np.exp(1j * (-np.outer(x_s[cols], q) + np.outer(h_s[cols], a1)))
        poly = -1j * (np.outer(src.dg[cols], q) + np.outer(src.df[cols], a1))
        ix = np.ix_(rows, cols)
        if not derivative:
            left = e_t * coef
            m2[ix] += left @ e_s.T
            m1[ix] += left @ (e_s * poly).T
            continue

        # d/dk at fixed nodes (the path is fixed per call context, so this is the
        # exact derivative). dα₁/dk = k/α₁ and dα₂/dk = k·ε/α₂ on the sheet.
        da1 = k / a1
        a2 = alpha(q, k2)
        if pol == 2:
            dr = 2.0 * (a2 * da1 - a1 * k * eps / a2) / (a1 + a2) ** 2
        else:
            b2 = a2 / eps
            dr = 2.0 * (b2 * da1 - a1 * k / a2) / (a1 + b2) ** 2
        dcoef = (1j / (4.0 * PI)) * w * (dr / a1 - r * da1 / a1**2)
        d_t = 1j * np.outer(h_t[rows], da1)
        d_s = 1j * np.outer(h_s[cols], da1)
        dpoly = d_s * poly - 1j * np.outer(src.df[cols], da1)
        left, dleft = e_t * coef, e_t * (dcoef + coef * d_t)
        m2[ix] += dleft @ e_s.T + left @ (e_s * d_s).T
        m1[ix] += dleft @ (e_s * poly).T + left @ (e_s * dpoly).T
    return step * m1, step * m2


def reflected_blocks(
    path: SommerfeldPath | None,
    pol: int,
    k: complex,
    eps_rel: complex | None,
    tgt: Geometry,
    src: Geometry,
    z_int: float,
    x_c: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Reflected M1/M2 blocks between two boundaries above the interface.

    Field node ``i`` on ``tgt``, source node ``j`` on ``src`` (``tgt = src`` for a
    particle's own block). The convention is :func:`assemble_cross_block`'s:
    ``M1 = h·(∂x'G·dg − ∂z'G·df)`` and ``M2 = h·G`` with ``h = 2π/src.n_pts``, so
    these add to the exterior rows with the same sign as the free-space blocks.
    Computed as a banded separable GEMM, ``A·diag(w·R/α₁)·Bᵀ``, on the path.

    ``x_c`` is the **centroid of the whole particle set**, never exposed to a
    user: an origin far from the particle overflows ``exp(−i q x)`` once
    ``|Im q|·|x| ≳ 700`` and costs a digit at 50 µm (study trap 3).

    Args:
        path: The Sommerfeld path, sized for the whole set's extent ``D``
            (``None`` for PEC).
        pol: 2 = TE, 1 = TM.
        k: Background wavenumber, complex allowed.
        eps_rel: Substrate permittivity relative to the cover, or ``None`` for PEC.
        tgt: Field boundary.
        src: Source boundary.
        z_int: Interface height (nm).
        x_c: Horizontal centre of the particle set.

    Returns:
        ``(m1_ind, m2_ind)``, each ``(tgt.n_pts, src.n_pts)``.

    Raises:
        ValueError: A boundary node is at or below the interface.
    """
    return _reflected_blocks(path, pol, k, eps_rel, tgt, src, z_int, x_c, False)


def reflected_blocks_dk(
    path: SommerfeldPath | None,
    pol: int,
    k: complex,
    eps_rel: complex | None,
    tgt: Geometry,
    src: Geometry,
    z_int: float,
    x_c: float,
) -> tuple[np.ndarray, np.ndarray]:
    """``d/dk`` of :func:`reflected_blocks`, for Newton refinement of a QNM.

    The path is fixed, so differentiating at fixed nodes is exact.

    Args:
        path: The Sommerfeld path (``None`` for PEC).
        pol: 2 = TE, 1 = TM.
        k: Background wavenumber, complex allowed.
        eps_rel: Substrate permittivity relative to the cover, or ``None``.
        tgt: Field boundary.
        src: Source boundary.
        z_int: Interface height (nm).
        x_c: Horizontal centre of the particle set.

    Returns:
        ``(dm1_dk, dm2_dk)``, each ``(tgt.n_pts, src.n_pts)``.
    """
    return _reflected_blocks(path, pol, k, eps_rel, tgt, src, z_int, x_c, True)
