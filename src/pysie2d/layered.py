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

import functools
import math
import warnings
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

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


class NoFarFieldError(ValueError):
    """The substrate has no downward far field (absorbing, ``Re ε ≤ 0`` or PEC)."""


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
    def for_features(
        cls,
        k: complex,
        features: Sequence[tuple[str, complex]],
        D: float,
        z_min: float,
        depth_scale: float = 1.0,
    ) -> SommerfeldPath:
        """Path for one wavenumber with an explicit singularity list (a stack).

        Every v0.9 rule applies unchanged — T, δ, clearance C1, node counts,
        banding — only the right-hand singularities come from the caller
        instead of :func:`_singularities`: a stack's outer branch points and
        guided poles (slab-spec §2.4).

        Args:
            k: Background wavenumber (rad/nm), complex allowed.
            features: ``(label, q)`` right-hand singularities to clear.
            D: Horizontal extent of everything the path serves (nm).
            z_min: Smallest decay length sum it serves (nm).
            depth_scale: As :meth:`for_wavenumber`.

        Returns:
            The path.

        Raises:
            ValueError: As :meth:`for_wavenumber`.
        """
        return cls._build(
            np.atleast_1d(np.asarray(k, dtype=complex)),
            None,
            0,
            D,
            z_min,
            depth_scale,
            features,
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
        features: Sequence[tuple[str, complex]] | None = None,
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

        if features is None:
            samples = [(i, _singularities(k, eps_rel, pol)) for i, k in enumerate(ks)]
        else:
            samples = [(0, [(label, complex(f)) for label, f in features])]
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


@dataclass(frozen=True, eq=False)
class Points:
    """Observation or source points standing in for a field boundary.

    Lets :func:`reflected_blocks` evaluate the reflected field at many points as
    one separable GEMM instead of a point-by-node loop of exponentials.

    Attributes:
        f: (n,) x coordinates (nm).
        g: (n,) z coordinates (nm).
    """

    f: np.ndarray
    g: np.ndarray

    @property
    def n_pts(self) -> int:
        """Number of points."""
        return int(self.f.size)


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
    r_fn: Callable[[np.ndarray], np.ndarray] | None = None,
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
        r_fn: ``q → R(q)`` replacing :func:`fresnel_r` (a stack's recursion);
            ``eps_rel`` is then unused.

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
    if eps_rel is None and r_fn is None:
        return _pec_terms(k, pol, X, Z)[:3]

    assert path is not None
    a = alpha(path.q, k)
    r = fresnel_r(path.q, k, eps_rel, pol) if r_fn is None else r_fn(path.q)
    base = path.w * r / a
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
    tgt: Geometry | Points,
    src: Geometry,
    z_int: float,
    x_c: float,
    derivative: bool,
    r_fn: Callable[[np.ndarray], np.ndarray] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    h_t, h_s = tgt.g - z_int, src.g - z_int
    _check_in_cover(tgt.g, z_int, "field-particle boundary")
    _check_in_cover(src.g, z_int, "source-particle boundary")
    step = src.delt

    if eps_rel is None and r_fn is None:
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
        r = fresnel_r(q, k, eps_rel, pol) if r_fn is None else r_fn(q)
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
        eps = _normalised(eps_rel)
        da1 = k / a1
        a2 = alpha(q, k * np.sqrt(eps))
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
    tgt: Geometry | Points,
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
        tgt: Field boundary, or :class:`Points`.
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
    tgt: Geometry | Points,
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
        tgt: Field boundary, or :class:`Points`.
        src: Source boundary.
        z_int: Interface height (nm).
        x_c: Horizontal centre of the particle set.

    Returns:
        ``(dm1_dk, dm2_dk)``, each ``(tgt.n_pts, src.n_pts)``.
    """
    return _reflected_blocks(path, pol, k, eps_rel, tgt, src, z_int, x_c, True)


@dataclass(frozen=True, eq=False)
class Reflection:
    """The reflected-field machinery for one call context.

    Bundles the path, wavenumber, substrate permittivity and interface height
    that a driven solve needs, so the facade, the right-hand sides, the near
    field and the LDOS all use one path built once. ``eps_rel is None`` is PEC.

    Attributes:
        pol: 2 = TE, 1 = TM.
        k: Background wavenumber (rad/nm), complex allowed.
        eps_rel: Substrate permittivity relative to the cover, or ``None``.
        z_int: Interface height (nm).
        path: The Sommerfeld path, ``None`` for PEC.
        x_c: Horizontal centre of everything the path serves.
    """

    pol: int
    k: complex
    eps_rel: complex | None
    z_int: float
    path: SommerfeldPath | None
    x_c: float

    @classmethod
    def build(
        cls,
        pol: int,
        k: complex,
        eps_rel: complex | None,
        z_int: float,
        f: np.ndarray,
        g: np.ndarray,
        path: SommerfeldPath | None = None,
    ) -> Reflection | None:
        """Reflection for every point ``(f, g)`` the call will touch.

        Args:
            pol: 2 = TE, 1 = TM.
            k: Background wavenumber.
            eps_rel: Substrate permittivity relative to the cover, or ``None``.
            z_int: Interface height (nm).
            f: x coordinates of every node, source and observation point served.
            g: Their z coordinates.
            path: A path to use as is (QNM: one per search box). By default one
                is built for this ``k``, sized for the points given.

        Returns:
            The reflection, or ``None`` when ``eps_rel == 1`` exactly: no
            interface, so no reflected term is built at all (G7).

        Raises:
            ValueError: A point is at or below the interface.
        """
        if eps_rel is not None and eps_rel == 1.0:
            return None
        f, g = np.asarray(f, dtype=float), np.asarray(g, dtype=float)
        _check_in_cover(g, z_int, "every node, source and observation point")
        z_min = 2.0 * float((g - z_int).min())
        if path is None and eps_rel is not None:
            # D never below z_min: a single point has no extent, and the length
            # must come from the geometry so the path stays scale covariant.
            path = SommerfeldPath.for_wavenumber(
                k, eps_rel, pol, max(float(np.ptp(f)), z_min), z_min
            )
        return cls(pol, k, eps_rel, z_int, path, 0.5 * float(f.max() + f.min()))

    def blocks(
        self, tgt: Geometry | Points, src: Geometry
    ) -> tuple[np.ndarray, np.ndarray]:
        """:func:`reflected_blocks` with this context."""
        return reflected_blocks(
            self.path, self.pol, self.k, self.eps_rel, tgt, src, self.z_int, self.x_c
        )

    def blocks_dk(
        self, tgt: Geometry | Points, src: Geometry
    ) -> tuple[np.ndarray, np.ndarray]:
        """:func:`reflected_blocks_dk` with this context."""
        return reflected_blocks_dk(
            self.path, self.pol, self.k, self.eps_rel, tgt, src, self.z_int, self.x_c
        )

    def green(
        self,
        x: np.ndarray | float,
        z: np.ndarray | float,
        xs: np.ndarray | float,
        zs: np.ndarray | float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """:func:`reflected_green` with this context."""
        return reflected_green(
            self.path, self.pol, self.k, self.eps_rel, x, z, xs, zs, self.z_int
        )

    def plane_wave(self, angle: float, f: np.ndarray, g: np.ndarray) -> np.ndarray:
        """Fresnel-reflected plane wave at ``(f, g)``, for a wave from the cover.

        The incident wave of :func:`pysie2d.sources.plane_wave_rhs` travels
        downward as ``exp(i k (f sin θ − g cos θ))``. Its reflection is
        ``r(q)·exp(i q f + i α₁ (g − 2·z_int))`` with ``q = k sin θ`` and
        ``α₁ = k cos θ``. Only ``r`` at the one real ``q`` is needed, so no path.

        Args:
            angle: Incidence angle θ (degrees), from the normal, |θ| < 90°.
            f: x coordinates (nm).
            g: z coordinates (nm).

        Returns:
            The reflected wave, shape of ``f``.

        Raises:
            ValueError: ``|angle| ≥ 90°``: grazing, and a wave from below the
                interface is outside this milestone.
        """
        if abs(angle) >= 90.0:
            raise ValueError(
                f"incidence angle {angle}° must satisfy |angle| < 90° over a half-space"
            )
        theta = np.deg2rad(angle)
        q, a1 = self.k * np.sin(theta), self.k * np.cos(theta)
        r = fresnel_r(np.asarray(q), self.k, self.eps_rel, self.pol)
        return r * np.exp(1j * (q * f + a1 * (g - 2.0 * self.z_int)))


# --- Transmitted field below the interface (v1.1) -----------------------------


def _tau_half_space(q: np.ndarray, k: complex, eps: complex, pol: int) -> np.ndarray:
    """``τ = T/α₁`` of a half-space: TE ``2/(α₁+α₂)``, TM ``2/(α₁+α₂/ε)``.

    T carries a factor α₁ for any stack (``t₁₂ = 2α₁/(α₁+α₂)``), so τ is finite
    at ``q = ±k₁``, where ``T/α₁`` evaluated literally is 0/0. The denominator
    is :func:`fresnel_r`'s, so ``T = 1 + R`` and the singularities are R's.
    """
    a1, a2 = alpha(q, k), alpha(q, k * np.sqrt(eps))
    return 2.0 / (a1 + a2) if pol == 2 else 2.0 / (a1 + a2 / eps)


def _check_below(z: np.ndarray, z_int: float, what: str) -> None:
    if np.any(z >= z_int):
        raise ValueError(
            f"{what} must lie strictly below the interface z_int = {z_int}; "
            f"highest z = {np.max(z)}"
        )


@dataclass(frozen=True, eq=False)
class Transmission:
    """The transmitted-field machinery for one call context (spec v1.1 §1).

    Written against a generic coefficient ``τ = T/α₁`` and the bottom layer's
    wavenumber, never against ``eps_rel`` directly, so that a multilayer only
    supplies its own ``tau``, ``k_bot`` and ``features`` (D3). A half-space has
    ``z_top = z_bot = z_int``.

    Attributes:
        pol: 2 = TE, 1 = TM.
        k: Cover wavenumber k₁, complex allowed.
        k_bot: Bottom-layer wavenumber ``k₁·√ε_bot``, principal root after the
            signed-zero normalisation.
        eps_bot: Bottom-layer permittivity relative to the cover.
        z_top: Top interface: the phase reference on the cover side.
        z_bot: Bottom interface: the phase reference in the bottom layer.
        tau: ``q → T(q)/α₁(q)`` on the code's sheet.
        features: Singularities ``(label, q)``, for the flux breakpoints.
        path: The Sommerfeld path, ``None`` until built for a point set.
        x_c: Horizontal centre of everything the path serves.
    """

    pol: int
    k: complex
    k_bot: complex
    eps_bot: complex
    z_top: float
    z_bot: float
    tau: Callable[[np.ndarray], np.ndarray]
    features: tuple[tuple[str, complex], ...]
    path: SommerfeldPath | None
    x_c: float

    @classmethod
    def half_space(
        cls, pol: int, k: complex, eps_rel: complex, z_int: float
    ) -> Transmission:
        """Transmission of a (non-PEC) half-space, with no path yet.

        Args:
            pol: 2 = TE, 1 = TM.
            k: Background wavenumber.
            eps_rel: Substrate permittivity relative to the cover.
            z_int: Interface height (nm).

        Returns:
            The transmission, ``path = None`` and ``x_c = 0``.
        """
        eps = _normalised(eps_rel)
        return cls(
            pol=pol,
            k=k,
            k_bot=k * np.sqrt(eps),
            eps_bot=eps,
            z_top=z_int,
            z_bot=z_int,
            tau=lambda q: _tau_half_space(q, k, eps, pol),
            features=tuple(_singularities(k, eps, pol)),
            path=None,
            x_c=0.0,
        )

    @classmethod
    def build(
        cls,
        pol: int,
        k: complex,
        eps_rel: complex | None,
        z_int: float,
        f_src: np.ndarray,
        g_src: np.ndarray,
        f_obs: np.ndarray,
        g_obs: np.ndarray,
    ) -> Transmission | None:
        """Half-space transmission with a path sized for sources and observers.

        The integrand decays like ``exp(−q(h' + |h|))``, so the path's ``z_min``
        is the smallest source height plus the smallest observation depth
        (finding 3: a path sized for anything larger truncates the TM tail).

        Args:
            pol: 2 = TE, 1 = TM.
            k: Background wavenumber.
            eps_rel: Substrate permittivity relative to the cover, or ``None``.
            z_int: Interface height (nm).
            f_src: x of every source (boundary nodes, dipole), above.
            g_src: Their z.
            f_obs: x of every observation point, below.
            g_obs: Their z.

        Returns:
            The transmission, or ``None`` for PEC (no field below, D14) and
            for ``eps_rel == 1`` exactly (no interface: the caller uses the
            free-space representation, D19).

        Raises:
            ValueError: A source at or below, or an observation point at or
                above, the interface.
        """
        if eps_rel is None or eps_rel == 1.0:
            return None
        f_src, g_src = np.asarray(f_src, float), np.asarray(g_src, float)
        f_obs, g_obs = np.asarray(f_obs, float), np.asarray(g_obs, float)
        _check_in_cover(g_src, z_int, "every node and source")
        _check_below(g_obs, z_int, "every substrate observation point")
        z_min = float((g_src - z_int).min() + (z_int - g_obs).min())
        fx = np.concatenate([f_src, f_obs])
        # D never below z_min, as in Reflection.build: the length must come from
        # the geometry so the path stays scale covariant.
        path = SommerfeldPath.for_wavenumber(
            k, eps_rel, pol, max(float(np.ptp(fx)), z_min), z_min
        )
        tr = cls.half_space(pol, k, eps_rel, z_int)
        return cls(
            tr.pol,
            tr.k,
            tr.k_bot,
            tr.eps_bot,
            tr.z_top,
            tr.z_bot,
            tr.tau,
            tr.features,
            path,
            0.5 * float(fx.max() + fx.min()),
        )

    def blocks(self, tgt: Points, src: Geometry) -> tuple[np.ndarray, np.ndarray]:
        """:func:`transmitted_blocks` with this context."""
        assert self.path is not None
        return transmitted_blocks(self.path, self, tgt, src, self.x_c)

    def green(
        self,
        x: np.ndarray | float,
        z: np.ndarray | float,
        xs: np.ndarray | float,
        zs: np.ndarray | float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """:func:`transmitted_green` with this context."""
        assert self.path is not None
        return transmitted_green(self.path, self, x, z, xs, zs)

    def plane_wave(self, angle: float, f: np.ndarray, g: np.ndarray) -> np.ndarray:
        """Transmitted plane wave at ``(f, g)`` below, for a wave from the cover.

        The incident wave ``exp(i k (f sin θ − g cos θ))`` becomes
        ``α₁·τ(q)·exp(i q f − i α₁ z_top − i α_bot (g − z_bot))`` with
        ``q = k sin θ``, ``α₁ = k cos θ``. Under total internal reflection
        ``α_bot = +i|…|`` on the sheet, so the wave decays with depth. Only the
        one real ``q`` is needed, so no path.

        Args:
            angle: Incidence angle θ (degrees), from the normal, |θ| < 90°.
            f: x coordinates (nm).
            g: z coordinates (nm), below the interface.

        Returns:
            The transmitted wave, shape of ``f``.

        Raises:
            ValueError: ``|angle| ≥ 90°``.
        """
        if abs(angle) >= 90.0:
            raise ValueError(
                f"incidence angle {angle} deg must satisfy |angle| < 90 deg "
                "over a half-space"
            )
        theta = np.deg2rad(angle)
        q, a1 = self.k * np.sin(theta), self.k * np.cos(theta)
        a_bot = alpha(np.asarray(q), self.k_bot)
        t = a1 * self.tau(np.asarray(q))
        return t * np.exp(1j * (q * f - a1 * self.z_top - a_bot * (g - self.z_bot)))


def transmitted_green(
    path: SommerfeldPath,
    tr: Transmission,
    x: np.ndarray | float,
    z: np.ndarray | float,
    xs: np.ndarray | float,
    zs: np.ndarray | float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Transmitted Green function at ``(x, z)`` below from a source ``(xs, zs)`` above.

    ``G_t = (i/4π) ∫_Γ τ(q)·exp(i q X + i α₁ h' + i α_bot |h|) dq``, with
    ``h' = z' − z_top`` and ``|h| = z_bot − z``, folded onto ``q ≥ 0`` as
    :func:`reflected_green` is. Source derivatives: ``∂x' → −iq`` and
    ``∂z' → +iα₁`` (holomorphy §6).

    Args:
        path: The Sommerfeld path, sized for ``h' + |h|`` of these points.
        tr: The transmission context (τ, wavenumbers, phase references).
        x: Observation x; broadcast against the source's.
        z: Observation z, at or below ``z_bot``.
        xs: Source x.
        zs: Source z, strictly above ``z_top``.

    Returns:
        ``(G_t, dG_t/dx', dG_t/dz')``, broadcast shape.

    Raises:
        ValueError: A source at or below ``z_top``, or an observation point
            above ``z_bot``.
    """
    x, z, xs, zs = np.broadcast_arrays(
        *(np.asarray(v, dtype=float) for v in (x, z, xs, zs))
    )
    _check_in_cover(zs, tr.z_top, "source points")
    if np.any(z > tr.z_bot):
        raise ValueError(
            f"observation points must lie at or below z_bot = {tr.z_bot}; "
            f"highest z = {np.max(z)}"
        )
    a1 = alpha(path.q, tr.k)
    a_bot = alpha(path.q, tr.k_bot)
    base = path.w * tr.tau(path.q)
    pre = 1j / (2.0 * PI)  # (i/4π)·2: the integral is folded onto q ≥ 0
    shape = x.shape
    Xf = (x - xs).ravel()
    hs, hb = (zs - tr.z_top).ravel(), (tr.z_bot - z).ravel()
    out = np.empty((3, Xf.size), dtype=complex)
    chunk = max(1, 4_000_000 // path.q.size)
    for lo in range(0, Xf.size, chunk):
        sl = slice(lo, lo + chunk)
        b = base * np.exp(1j * (np.outer(hs[sl], a1) + np.outer(hb[sl], a_bot)))
        qx = np.outer(Xf[sl], path.q)
        c = np.cos(qx)
        out[0, sl] = pre * np.sum(b * c, axis=1)
        out[1, sl] = pre * np.sum(b * path.q * np.sin(qx), axis=1)
        out[2, sl] = pre * np.sum(b * 1j * a1 * c, axis=1)
    return out[0].reshape(shape), out[1].reshape(shape), out[2].reshape(shape)


def transmitted_blocks(
    path: SommerfeldPath,
    tr: Transmission,
    tgt: Points,
    src: Geometry,
    x_c: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Transmitted M1/M2 blocks from a boundary above to points below.

    The convention is :func:`reflected_blocks`': ``M1 = h·(∂x'G_t·dg −
    ∂z'G_t·df)`` and ``M2 = h·G_t``, so the substrate field is
    ``−(m1 @ φ + m2 @ χ)``. Computed as the banded separable GEMM
    ``A·diag(w·τ)·Bᵀ`` with ``A = e^{iq(x−x_c) + iα_bot|h|}`` and
    ``B = e^{−iq(x'−x_c) + iα₁h'}``; measured equal to the pointwise
    :func:`transmitted_green` sum to 2.5e-16 (t4_gemm), the banding dropping
    nothing.

    Args:
        path: The Sommerfeld path, sized for these points.
        tr: The transmission context.
        tgt: Observation points, at or below ``z_bot``.
        src: Source boundary, above ``z_top``.
        x_c: Horizontal centre of the point set (overflow guard, study trap 3).

    Returns:
        ``(m1_t, m2_t)``, each ``(tgt.n_pts, src.n_pts)``.
    """
    hb, hs = tr.z_bot - tgt.g, src.g - tr.z_top  # observation depth, source height
    hb_min, hs_min = hb.min(), hs.min()
    xt, xs = tgt.f - x_c, src.f - x_c
    m1 = np.zeros((tgt.n_pts, src.n_pts), dtype=complex)
    m2 = np.zeros_like(m1)
    for a0, i0, i1 in path.bands:
        if a0 == 0.0:
            rows, cols = np.arange(tgt.n_pts), np.arange(src.n_pts)
        else:
            # |A·B| ~ exp(−q·(|h_i| + h'_j)): a pair is reached by this band only
            # if |h_i| + h'_j < 38/q for some q ≥ a0.
            lim = TAIL_DECAY / a0
            rows = np.flatnonzero(hb < lim - hs_min)
            cols = np.flatnonzero(hs < lim - hb_min)
            if rows.size == 0 or cols.size == 0:
                continue
        # The folded half path and its mirror: A·diag(c)·Bᵀ needs the full path.
        q = np.concatenate([path.q[i0:i1], -path.q[i0:i1]])
        w = np.concatenate([path.w[i0:i1], path.w[i0:i1]])
        a1, a_bot = alpha(q, tr.k), alpha(q, tr.k_bot)
        coef = (1j / (4.0 * PI)) * w * tr.tau(q)
        e_t = np.exp(1j * (np.outer(xt[rows], q) + np.outer(hb[rows], a_bot)))
        e_s = np.exp(1j * (-np.outer(xs[cols], q) + np.outer(hs[cols], a1)))
        poly = -1j * (np.outer(src.dg[cols], q) + np.outer(src.df[cols], a1))
        left = e_t * coef
        ix = np.ix_(rows, cols)
        m2[ix] += left @ e_s.T
        m1[ix] += left @ (e_s * poly).T
    return src.delt * m1, src.delt * m2


# --- N-layer stack (v1.2, slab-spec §2) ----------------------------------------
#
# Frame as v0.9: cover above z_top, layers downward, bottom below z_bot, every
# value background-relative (cover ε = 1), k the background wavenumber. Media
# are numbered 0 (cover) … n+1 (bottom), so layer j of the tuples is medium j+1.
# Admittance Y = α (TE) or α/ε (TM), so one recursion serves both. Outer α on
# the vertical-cut sheet (`alpha`); interior α_j on Im α_j ≥ 0, which bounds
# every e^{2iα_j d_j} by 1 (R and τ are even in every interior α_j).

# γ-grid density for the guided-pole bracketing: n = max(4000, ⌈400(1 + k_max·Σd/π)⌉)
# (S2, S4: a q-grid misses near-cutoff roots at t ≤ 1e-4; the γ-grid finds all).
POLE_GRID_MIN = 4000
POLE_GRID_PER_MODE = 400.0
# Argument-principle box [0, 1.05γ_hi] × [−h, h], h = 0.01·γ_hi. It must stay
# thin: Re γ > 0 is the Im α ≥ 0 sheet, which leaves the code's sheet left of
# k_out (S4 counted a leaky pole at h = 0.05).
WINDING_OVERSHOOT = 1.05
WINDING_HALF_HEIGHT = 0.01
WINDING_POINTS_MIN = 8192
WINDING_POINTS_PER_K_D = 64.0
# A shortfall of the bracketing against the count refines the grid ×10, twice.
POLE_GRID_REFINEMENTS = 2
# Continuation of the real roots to complex k or a lossy stack (S6): 16 linear
# homotopy steps, Newton with D′ from a 64-point Cauchy circle of radius
# 1e-4·|q|, |Δq| < 1e-15·|q|, at most 30 steps per stage.
CONTINUATION_STEPS = 16
CAUCHY_POINTS = 64
CAUCHY_RADIUS = 1e-4
NEWTON_RTOL = 1e-15
NEWTON_MAX_STEPS = 30
# |α_j d_j| below which sin(αd)/α is formed as np.sinc: (e^{iαd} − e^{−iαd})/2iα
# cancels there.
SMALL_PHASE = 1e-3


def _alpha_interior(q: np.ndarray, kj: complex) -> np.ndarray:
    """Interior ``α_j = √(k_j² − q²)`` with ``Im α_j ≥ 0``: ``|e^{2iα_j d_j}| ≤ 1``."""
    a = np.sqrt(kj * kj - q * q)
    return np.where(a.imag < 0.0, -a, a)


def _stack_media(
    q: np.ndarray,
    k: complex,
    eps_layers: tuple[complex, ...],
    d: tuple[float, ...],
    eps_bot: complex,
    pol: int,
) -> tuple[list, list, list, list, list]:
    """Admittances, interior α, Γ at every medium top, and the interface terms.

    Bottom up, ``Γ_{n+1} = 0``; across interface ``i | i+1`` the division-free
    form ``G_i = num_i/den_i`` with ``num_i = (Y_i − Y_{i+1}) + (Y_i + Y_{i+1})Γ_{i+1}``
    and ``den_i = (Y_i + Y_{i+1}) + (Y_i − Y_{i+1})Γ_{i+1}``, then
    ``Γ_i = G_i e^{2iα_i d_i}`` (medium ``i ≥ 1``). Never forms ``e^{+|Im α|d}``:
    the naive transfer matrix overflows at ``q·Σd ≈ 709`` (S1, finding 1).
    """
    n = len(eps_layers)
    al = [_alpha_interior(q, k * np.sqrt(e)) for e in eps_layers]
    a_bot = alpha(q, k * np.sqrt(eps_bot))
    if pol == 2:
        Y = [alpha(q, k), *al, a_bot]
    else:
        Y = [alpha(q, k), *(a / e for a, e in zip(al, eps_layers, strict=True))]
        Y.append(a_bot / eps_bot)
    gam: list = [None] * (n + 2)
    gam[n + 1] = np.zeros_like(Y[0])
    num: list = [None] * (n + 1)
    den: list = [None] * (n + 1)
    for i in range(n, -1, -1):
        s, t = Y[i] + Y[i + 1], Y[i] - Y[i + 1]
        num[i] = t + s * gam[i + 1]
        den[i] = s + t * gam[i + 1]
        g = num[i] / den[i]
        gam[i] = g if i == 0 else g * np.exp(2j * al[i - 1] * d[i - 1])
    return Y, al, gam, num, den


def stack_r(
    q: np.ndarray | complex,
    k: complex,
    eps_layers: tuple[complex, ...],
    d: tuple[float, ...],
    eps_bot: complex,
    pol: int,
) -> np.ndarray:
    """Cover-side generalised reflection coefficient of a stack at ``z_top``.

    Args:
        q: Transverse wavenumber, any shape.
        k: Cover (background) wavenumber, complex allowed.
        eps_layers: Layer permittivities relative to the cover, top first;
            never empty (T6 routes an empty stack to :func:`fresnel_r`).
        d: Layer thicknesses (nm).
        eps_bot: Bottom permittivity relative to the cover.
        pol: 2 = TE, 1 = TM.

    Returns:
        Complex ``R(q)`` with the shape of ``q``.
    """
    q = np.asarray(q, dtype=complex)
    return _stack_media(q, k, eps_layers, d, eps_bot, pol)[2][0]


def stack_tau(
    q: np.ndarray | complex,
    k: complex,
    eps_layers: tuple[complex, ...],
    d: tuple[float, ...],
    eps_bot: complex,
    pol: int,
) -> np.ndarray:
    """``τ = T/α₁``: downward amplitude at ``z_bot`` per unit downward at ``z_top``.

    ``τ = (2/den₀)·Π_{i≥1} 2Y_i e^{iα_i d_i}/den_i``, which is slab-spec §2.1's
    ``[2/(Y₀+Y₁)]·Π[2Y_i/(Y_i+Y_{i+1})]e^{iα_i d_i}·Π 1/(1 + r_iΓ_{i+1})`` with
    each ``(Y_i+Y_{i+1})`` cancelled. Finite at ``q = ±k₁`` (``Y₀ = α₁``).

    Args:
        q: Transverse wavenumber, any shape.
        k: Cover wavenumber.
        eps_layers: Layer permittivities relative to the cover, top first.
        d: Layer thicknesses (nm).
        eps_bot: Bottom permittivity relative to the cover.
        pol: 2 = TE, 1 = TM.

    Returns:
        Complex ``τ(q)`` with the shape of ``q``.
    """
    q = np.asarray(q, dtype=complex)
    Y, al, _, _, den = _stack_media(q, k, eps_layers, d, eps_bot, pol)
    out = 2.0 / den[0]
    for i in range(1, len(eps_layers) + 1):
        out = out * 2.0 * Y[i] * np.exp(1j * al[i - 1] * d[i - 1]) / den[i]
    return out


def stack_layer_factor(
    q: np.ndarray,
    k: complex,
    eps_layers: tuple[complex, ...],
    d: tuple[float, ...],
    eps_bot: complex,
    pol: int,
    s: np.ndarray,
    j: int,
) -> np.ndarray:
    """``W/α₁`` in layer ``j`` at depth ``s`` below its top (slab-spec §2.2).

    Per unit downward amplitude at ``z_top``, the field in layer ``j`` is
    ``W = A_j[e^{iα_j s} + G_j^{bot} e^{iα_j(2d_j − s)}]`` with ``s ∈ [0, d_j]``,
    ``A₁/α₁ = 2/den₀`` and ``A_{j+1} = A_j e^{iα_j d_j}·2Y_j/den_j``; every
    exponential is bounded. ``W/α₁`` is finite at ``q = ±k₁``.

    Args:
        q: (M,) transverse wavenumbers.
        k: Cover wavenumber.
        eps_layers: Layer permittivities relative to the cover, top first.
        d: Layer thicknesses (nm).
        eps_bot: Bottom permittivity relative to the cover.
        pol: 2 = TE, 1 = TM.
        s: (P,) depths below the top of layer ``j`` (nm), in ``[0, d_j]``.
        j: Layer index, 0 for the top layer.

    Returns:
        complex (P, M) array.
    """
    q = np.asarray(q, dtype=complex)
    Y, al, gam, num, den = _stack_media(q, k, eps_layers, d, eps_bot, pol)
    amp = 2.0 / den[0]
    for i in range(1, j + 1):
        amp = amp * 2.0 * Y[i] * np.exp(1j * al[i - 1] * d[i - 1]) / den[i]
    m = j + 1
    g_bot = num[m] / den[m]
    a = al[j][None, :]
    s = np.asarray(s, dtype=float)[:, None]
    return amp[None, :] * (
        np.exp(1j * a * s) + g_bot[None, :] * np.exp(1j * a * (2.0 * d[j] - s))
    )


def _dispersion(
    q: np.ndarray,
    k: complex,
    eps_layers: tuple[complex, ...],
    d: tuple[float, ...],
    eps_bot: complex,
    pol: int,
    scale_at: complex | None = None,
) -> np.ndarray:
    """Scaled transfer-matrix denominator of R; see :func:`stack_dispersion`.

    Each layer's matrix is multiplied by ``e^{−|Im α_j| d_j}``, formed from
    bounded exponentials (``cos`` itself overflows first). The factor is
    positive, so roots, signs and the winding number are those of the unscaled
    D. With ``scale_at`` the factor is frozen at that one ``q``, so the
    function is holomorphic in ``q`` around it (Newton, Cauchy D′).
    """
    q = np.asarray(q, dtype=complex)
    y1 = alpha(q, k)
    a_bot = alpha(q, k * np.sqrt(eps_bot))
    y_bot = a_bot if pol == 2 else a_bot / eps_bot
    m11, m12 = np.ones_like(q), np.zeros_like(q)
    m21, m22 = np.zeros_like(q), np.ones_like(q)
    for e, dj in zip(eps_layers, d, strict=True):
        kj = k * np.sqrt(e)
        a = np.sqrt(kj * kj - q * q)  # the entries are even in a: any sheet
        a_s = a if scale_at is None else np.sqrt(kj * kj - scale_at * scale_at)
        sc = np.abs(np.imag(a_s)) * dj
        ep, em = np.exp(1j * a * dj - sc), np.exp(-1j * a * dj - sc)
        c = 0.5 * (ep + em)
        with np.errstate(divide="ignore", invalid="ignore"):
            sinc = np.where(
                np.abs(a * dj) > SMALL_PHASE,
                (ep - em) / (2j * a),
                np.sinc(a * dj / PI) * dj * np.exp(-sc),
            )
        w = 1.0 if pol == 2 else e
        n12, n21 = -w * sinc, (a * a / w) * sinc
        m11, m12, m21, m22 = (
            c * m11 + n12 * m21,
            c * m12 + n12 * m22,
            n21 * m11 + c * m21,
            n21 * m12 + c * m22,
        )
    return -1j * y_bot * (m11 + 1j * y1 * m12) - (m21 + 1j * y1 * m22)


def stack_dispersion(
    q: np.ndarray | complex,
    k: complex,
    eps_layers: tuple[complex, ...],
    d: tuple[float, ...],
    eps_bot: complex,
    pol: int,
) -> np.ndarray:
    """Scaled dispersion function D(q): zero at every guided pole (slab-spec §2.3).

    The transfer-matrix denominator of R with each layer's matrix scaled by
    ``e^{−|Im α_j| d_j}``: same roots, real on the guided window of a lossless
    stack, so the roots bracket.

    Args:
        q: Transverse wavenumber, any shape.
        k: Cover wavenumber.
        eps_layers: Layer permittivities relative to the cover, top first.
        d: Layer thicknesses (nm).
        eps_bot: Bottom permittivity relative to the cover.
        pol: 2 = TE, 1 = TM.

    Returns:
        Complex ``D(q)`` with the shape of ``q``.
    """
    return _dispersion(np.asarray(q, dtype=complex), k, eps_layers, d, eps_bot, pol)


def _window(
    k: float, eps_layers: tuple[complex, ...], eps_bot: complex
) -> tuple[float, float]:
    """Guided window ``(k_out, k_max)``: ``max(k₁, Re k_bot)``, ``max Re k_j``."""
    k_out = max(k, float(np.real(k * np.sqrt(eps_bot))))
    k_max = max(float(np.real(k * np.sqrt(e))) for e in eps_layers)
    return k_out, k_max


def _winding(
    k: float,
    eps_layers: tuple[complex, ...],
    d: tuple[float, ...],
    eps_bot: complex,
    pol: int,
    g_lo: float,
    g_hi: float,
    h: float,
    n: int,
) -> int:
    """Zeros of D(q(γ)) in the γ-box ``[g_lo, g_hi] × [−h, h]``, argument principle.

    ``q = √(k_out² + γ²)`` uniformises the outer branch point (the outer
    ``Y = iγ``), so D is analytic through γ = 0 and the box can sit on it.
    """
    k_out, _ = _window(k, eps_layers, eps_bot)
    m = n // 4
    s = np.linspace(0.0, 1.0, m, endpoint=False)
    w = g_hi - g_lo
    gs = np.concatenate(
        [
            g_lo + w * s - 1j * h,
            g_hi + 1j * h * (2.0 * s - 1.0),
            g_hi - w * s + 1j * h,
            g_lo + 1j * h * (1.0 - 2.0 * s),
        ]
    )
    v = _dispersion(np.sqrt(k_out * k_out + gs * gs), k, eps_layers, d, eps_bot, pol)
    ph = np.unwrap(np.angle(np.append(v, v[0])))
    return int(round((ph[-1] - ph[0]) / (2.0 * PI)))


def _bracket(
    k: float,
    eps_layers: tuple[complex, ...],
    d: tuple[float, ...],
    eps_bot: complex,
    pol: int,
    n_grid: int,
) -> np.ndarray:
    """Real roots of D bracketed on a uniform γ-grid from γ = 0, refined by brentq.

    Near cutoff D is linear in γ, not in q, so the γ-grid sees a root at
    ``(β − k_out)/k = t`` that a q-grid one cell inside the window misses (S4).
    """
    from scipy.optimize import brentq

    k_out, k_max = _window(k, eps_layers, eps_bot)
    g_hi = np.sqrt(k_max * k_max - k_out * k_out)

    def f(g: np.ndarray | float) -> np.ndarray:
        q = np.sqrt(k_out * k_out + np.asarray(g, dtype=float) ** 2) + 0j
        return _dispersion(np.atleast_1d(q), k, eps_layers, d, eps_bot, pol).real

    gs = np.linspace(0.0, g_hi, n_grid + 1)
    vals = f(gs)
    roots = []
    for i in np.flatnonzero(vals[:-1] * vals[1:] < 0.0):
        g0 = brentq(lambda g: f(g)[0], gs[i], gs[i + 1], xtol=1e-300, rtol=8.9e-16)
        roots.append(np.sqrt(k_out * k_out + g0 * g0))
    return np.array(sorted(roots, reverse=True))


def _pole_grid(k: float, eps_layers: tuple[complex, ...], d: tuple[float, ...]) -> int:
    k_max = max(float(np.real(k * np.sqrt(e))) for e in eps_layers)
    return max(
        POLE_GRID_MIN, math.ceil(POLE_GRID_PER_MODE * (1.0 + k_max * sum(d) / PI))
    )


def _winding_box(
    k: float, eps_layers: tuple[complex, ...], d: tuple[float, ...], eps_bot: complex
) -> tuple[float, float, int]:
    """``(γ_box, h, n)`` of the count check: the box ``[0, 1.05γ_hi] × [−h, h]``."""
    k_out, k_max = _window(k, eps_layers, eps_bot)
    g_hi = np.sqrt(k_max * k_max - k_out * k_out)
    n = max(WINDING_POINTS_MIN, math.ceil(WINDING_POINTS_PER_K_D * k_max * sum(d)))
    return WINDING_OVERSHOOT * g_hi, WINDING_HALF_HEIGHT * g_hi, n


def guided_poles(
    k: float,
    eps_layers: tuple[complex, ...],
    d: tuple[float, ...],
    eps_bot: complex,
    pol: int,
) -> tuple[np.ndarray, int]:
    """Guided poles of a lossless stack at real ``k``, and their count.

    The real roots of the scaled D, bracketed on a uniform γ-grid from γ = 0
    and refined by ``brentq``; the count is the argument principle on the thin
    γ-box. Close pairs vanish in pairs from any bracketing grid (finding 3), so
    a shortfall refines the grid ×10, twice; the caller compares the two.

    Args:
        k: Real cover wavenumber.
        eps_layers: Real layer permittivities relative to the cover.
        d: Layer thicknesses (nm).
        eps_bot: Real bottom permittivity relative to the cover.
        pol: 2 = TE, 1 = TM.

    Returns:
        ``(roots, count)``: β in descending order, and the winding count.
    """
    k_out, k_max = _window(k, eps_layers, eps_bot)
    if k_max <= k_out:
        return np.empty(0), 0
    g_box, h, n = _winding_box(k, eps_layers, d, eps_bot)
    count = _winding(k, eps_layers, d, eps_bot, pol, 0.0, g_box, h, n)
    n_grid = _pole_grid(k, eps_layers, d)
    for refine in range(POLE_GRID_REFINEMENTS + 1):
        roots = _bracket(k, eps_layers, d, eps_bot, pol, n_grid * 10**refine)
        if roots.size >= count:
            break
    return roots, count


def _unresolved_cell(
    k: float,
    eps_layers: tuple[complex, ...],
    d: tuple[float, ...],
    eps_bot: complex,
    pol: int,
    roots: np.ndarray,
) -> tuple[float, float]:
    """The q-interval of one grid cell holding zeros the bracketing missed.

    Bisects the count box, descending into the half whose winding exceeds the
    roots found in it, down to the finest grid's cell width.
    """
    k_out, k_max = _window(k, eps_layers, eps_bot)
    g_box, h, n = _winding_box(k, eps_layers, d, eps_bot)
    cell = np.sqrt(k_max * k_max - k_out * k_out) / (
        _pole_grid(k, eps_layers, d) * 10**POLE_GRID_REFINEMENTS
    )
    g_roots = np.sqrt(np.maximum(roots**2 - k_out * k_out, 0.0))
    lo, hi = 0.0, g_box
    while hi - lo > cell:
        mid = 0.5 * (lo + hi)
        found = np.count_nonzero((g_roots > lo) & (g_roots < mid))
        if _winding(k, eps_layers, d, eps_bot, pol, lo, mid, h, n) > found:
            hi = mid
        else:
            lo = mid
    return float(np.sqrt(k_out**2 + lo**2)), float(np.sqrt(k_out**2 + hi**2))


def _cauchy_derivative(
    q0: complex,
    k: complex,
    eps_layers: tuple[complex, ...],
    d: tuple[float, ...],
    eps_bot: complex,
    pol: int,
) -> complex:
    """D′(q₀) by Cauchy's formula on a circle of radius 1e-4·|q₀| (trapezoid)."""
    r = CAUCHY_RADIUS * abs(q0)
    t = 2.0 * PI * np.arange(CAUCHY_POINTS) / CAUCHY_POINTS
    z = q0 + r * np.exp(1j * t)
    v = _dispersion(z, k, eps_layers, d, eps_bot, pol, scale_at=q0)
    return complex(np.mean(v * np.exp(-1j * t)) / r)


def continue_poles(
    k_from: complex,
    k_to: complex,
    eps_from: tuple[tuple[complex, ...], complex],
    eps_to: tuple[tuple[complex, ...], complex],
    d: tuple[float, ...],
    pol: int,
    poles: np.ndarray,
) -> np.ndarray:
    """Continue real guided poles to complex ``k`` and/or a lossy stack (S6).

    Homotopy in 16 steps, ``ε(s)`` and ``k(s)`` linear; Newton on D at each
    step, with the scale factor frozen at the current iterate so D is
    holomorphic there and D′ comes from a Cauchy circle.

    Args:
        k_from: Wavenumber the poles were found at (real).
        k_to: Target wavenumber, complex allowed.
        eps_from: ``(layer ε, ε_bot)`` the poles were found for (real).
        eps_to: Target ``(layer ε, ε_bot)``, lossy allowed.
        d: Layer thicknesses (nm).
        pol: 2 = TE, 1 = TM.
        poles: The real roots at the start.

    Returns:
        complex array of the continued poles, in the order given.

    Raises:
        ValueError: Newton does not reach ``|Δq| < 1e-15·|q|`` within 30 steps
            at some stage.
    """
    (lay0, bot0), (lay1, bot1) = eps_from, eps_to
    out = []
    for beta in poles:
        q = complex(beta)
        for s in np.linspace(0.0, 1.0, CONTINUATION_STEPS + 1)[1:]:
            lay = tuple((1.0 - s) * a + s * b for a, b in zip(lay0, lay1, strict=True))
            bot = (1.0 - s) * bot0 + s * bot1
            kk = (1.0 - s) * k_from + s * k_to
            for _ in range(NEWTON_MAX_STEPS):
                dq = _dispersion(np.array([q]), kk, lay, d, bot, pol, scale_at=q)[0]
                dq /= _cauchy_derivative(q, kk, lay, d, bot, pol)
                q -= dq
                if abs(dq) < NEWTON_RTOL * abs(q):
                    break
            else:
                raise ValueError(
                    f"guided-pole continuation from β = {beta:.12g} did not converge "
                    f"at homotopy step s = {s:.4g} (last |Δq|/|q| = {abs(dq / q):.1e})"
                )
        out.append(q)
    return np.array(out, dtype=complex)


def stack_features(
    k: complex, k_bot: complex, poles: Sequence[complex]
) -> tuple[tuple[str, complex], ...]:
    """Right-hand singularities of a stack's integrand: ``+k₁``, ``+k_bot``, the β_m.

    The interior ``k_j`` are not singularities: R and τ are even in every
    interior α_j.
    """
    feats = [("+k₁", complex(k)), ("+k_bot", complex(k_bot))]
    return tuple(feats + [(f"β_{m}", complex(b)) for m, b in enumerate(poles)])


@functools.lru_cache(maxsize=64)
def _stack_poles(
    k: complex,
    eps_layers: tuple[complex, ...],
    d: tuple[float, ...],
    eps_bot: complex,
    pol: int,
) -> tuple[
    tuple[complex, ...], tuple[tuple[str, complex], ...], tuple[float, float] | None
]:
    """Poles at ``k`` (continued if needed), the path features, and any unresolved cell.

    The roots are found at ``Re k`` with ``Re ε`` and continued when ``Im k ≠ 0``
    or any ``Im ε ≠ 0``. If the count check is still short after refinement,
    the unresolved γ-cell's centre becomes a feature: the path for fields needs
    only clearance, and only the guided outputs raise (§2.4).
    """
    kr = float(np.real(k))
    lay_re = tuple(complex(e.real, 0.0) for e in eps_layers)
    bot_re = complex(eps_bot.real, 0.0)
    roots, count = guided_poles(kr, lay_re, d, bot_re, pol)
    cell = None
    if count > roots.size:
        cell = _unresolved_cell(kr, lay_re, d, bot_re, pol, roots)
    poles: np.ndarray = roots.astype(complex)
    lossy = any(e.imag != 0.0 for e in eps_layers) or eps_bot.imag != 0.0
    if roots.size and (np.imag(k) != 0.0 or lossy):
        poles = continue_poles(
            kr, k, (lay_re, bot_re), (eps_layers, eps_bot), d, pol, roots
        )
    feats = stack_features(k, k * np.sqrt(eps_bot), poles)
    if cell is not None:
        feats += (("unresolved pair", complex(0.5 * (cell[0] + cell[1]))),)
    return tuple(complex(p) for p in poles), feats, cell


# (layer ε_rel, thicknesses, ε_bot,rel, z_top): a stack after the T6 normalisation.
StackRel = tuple[tuple[complex, ...], tuple[float, ...], complex, float]


@dataclass(frozen=True, eq=False)
class StackReflection:
    """The reflected-field machinery of a stack for one call context.

    The stack's counterpart of :class:`Reflection`, with the same methods, so
    the solve, the right-hand sides, the cover near field and the LDOS reach
    it unchanged. ``r`` is the stable recursion :func:`stack_r`; the path's
    features are the stack's outer branch points and guided poles.

    Attributes:
        pol: 2 = TE, 1 = TM.
        k: Background wavenumber (rad/nm), complex allowed.
        r: ``q → R(q)`` on the code's sheet.
        features: The path's singularities.
        z_top: Top interface (nm), after the T6 normalisation.
        path: The Sommerfeld path.
        x_c: Horizontal centre of everything the path serves.
        stack: The normalised relative stack the context was built for.
    """

    pol: int
    k: complex
    r: Callable[[np.ndarray], np.ndarray]
    features: tuple[tuple[str, complex], ...]
    z_top: float
    path: SommerfeldPath
    x_c: float
    stack: StackRel

    @classmethod
    def build(
        cls,
        pol: int,
        k: complex,
        stack_rel: StackRel,
        f: np.ndarray,
        g: np.ndarray,
        *,
        z_min: float | None = None,
    ) -> StackReflection | None:
        """Reflection for every point ``(f, g)`` the call will touch, all in the cover.

        Sized as :meth:`Reflection.build`: ``z_min = 2·min(g − z_top)`` and
        ``D = max(ptp f, z_min)``.

        Args:
            pol: 2 = TE, 1 = TM.
            k: Background wavenumber.
            stack_rel: The normalised relative stack (:meth:`Stack.eps_rel`).
            f: x of every node, source and observation point served.
            g: Their z.
            z_min: Overrides ``2·min(g − z_top)`` for a set whose sources and
                observers are separate, where ``Z ≥ h_obs + h_src`` (the
                incident field, as in v0.9's ``_background_field``).

        Returns:
            The context, or ``None`` when no layer is left (T6).

        Raises:
            ValueError: A point at or below ``z_top``; a pole continuation that
                does not converge.
        """
        eps_layers, d, eps_bot, z_top = stack_rel
        if not eps_layers:
            return None
        f, g = np.asarray(f, dtype=float), np.asarray(g, dtype=float)
        _check_in_cover(g, z_top, "every node, source and observation point")
        if z_min is None:
            z_min = 2.0 * float((g - z_top).min())
        feats = _stack_poles(complex(k), eps_layers, d, eps_bot, pol)[1]
        path = SommerfeldPath.for_features(
            k, feats, max(float(np.ptp(f)), z_min), z_min
        )
        return cls(
            pol,
            k,
            lambda q: stack_r(q, k, eps_layers, d, eps_bot, pol),
            feats,
            z_top,
            path,
            0.5 * float(f.max() + f.min()),
            stack_rel,
        )

    def blocks(
        self, tgt: Geometry | Points, src: Geometry
    ) -> tuple[np.ndarray, np.ndarray]:
        """Reflected M1/M2 blocks: :func:`reflected_blocks` with the stack's R."""
        return _reflected_blocks(
            self.path,
            self.pol,
            self.k,
            None,
            tgt,
            src,
            self.z_top,
            self.x_c,
            False,
            self.r,
        )

    def green(
        self,
        x: np.ndarray | float,
        z: np.ndarray | float,
        xs: np.ndarray | float,
        zs: np.ndarray | float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """``(G_ind, ∂x'G_ind, ∂z'G_ind)``: :func:`reflected_green`, stack R."""
        return reflected_green(
            self.path, self.pol, self.k, None, x, z, xs, zs, self.z_top, self.r
        )

    def green_grad_src(
        self,
        x: np.ndarray | float,
        z: np.ndarray | float,
        xs: np.ndarray | float,
        zs: np.ndarray | float,
    ) -> tuple[np.ndarray, np.ndarray]:
        """``(∂x'G_ind, ∂z'G_ind)``, path factors ``−iq``, ``+iα₁`` (in-plane RHS)."""
        return self.green(x, z, xs, zs)[1:]

    def plane_wave(self, angle: float, f: np.ndarray, g: np.ndarray) -> np.ndarray:
        """Reflected plane wave at ``(f, g)``: :meth:`Reflection.plane_wave`, stack R.

        Args:
            angle: Incidence angle θ (degrees), from the normal, |θ| < 90°.
            f: x coordinates (nm).
            g: z coordinates (nm).

        Returns:
            The reflected wave, shape of ``f``.

        Raises:
            ValueError: ``|angle| ≥ 90°``.
        """
        if abs(angle) >= 90.0:
            raise ValueError(
                f"incidence angle {angle}° must satisfy |angle| < 90° over a stack"
            )
        theta = np.deg2rad(angle)
        q, a1 = self.k * np.sin(theta), self.k * np.cos(theta)
        r = self.r(np.asarray(q))
        return r * np.exp(1j * (q * f + a1 * (g - 2.0 * self.z_top)))


# Gauss–Legendre points per layer for the mode norm N = ∫ p u² dz (S8: the
# residue identity holds to 4e-14 with 200, 16 modes per polarisation).
NORM_GL_POINTS = 200


@dataclass(frozen=True, eq=False)
class GuidedMode:
    """A guided mode of a lossless stack at one real wavelength.

    ``u(z)`` solves ``∇·(p∇u) + k²w u = 0`` with ``p = 1, w = ε`` (TE) or
    ``p = 1/ε, w = 1`` (TM), all relative to the cover, normalised so that
    ``u(z_top) = 1``. The norm is the **bilinear** ``N = ∫ p u² dz``, which is
    the weight of the 1-D Green function's pole, so ``Res_β R = γ₁/(βN)``
    (S8; the TM weight 1/ε is load-bearing, 74 % off without it). The
    power-normalised mode is ``ũ = u/√(βN)`` (slab-spec §1.3).

    Attributes:
        order: 0 for the highest β.
        pol: 2 = TE, 1 = TM.
        beta: Propagation constant (rad/nm); background-independent.
        n_eff: ``β/(2π/λ_vac)``.
        norm: ``N`` (nm), with ``u(z_top) = 1``.
        gamma: Cover decay constant ``γ₁ = √(β² − k₁²)`` (rad/nm).
        wavelength: **Vacuum** wavelength (nm).
    """

    order: int
    pol: int
    beta: float
    n_eff: float
    norm: float
    gamma: float
    wavelength: float
    _z_top: float = field(repr=False)
    _tops: tuple[tuple[complex, complex, complex, complex, float], ...] = field(
        repr=False
    )
    _bottom: tuple[float, float] = field(repr=False)

    def profile(self, z: np.ndarray | float) -> np.ndarray:
        """``u(z)`` at any real height, cover, layers and bottom; ``u(z_top) = 1``.

        Args:
            z: Heights (nm), any shape.

        Returns:
            Real array with the shape of ``z``.
        """
        h = np.asarray(z, dtype=float) - self._z_top
        out = np.empty(h.shape, dtype=complex)
        up = h >= 0.0
        out[up] = np.exp(-self.gamma * h[up])
        top = 0.0
        for u0, v0, a, w, d in self._tops:
            m = (h < top) & (h >= top - d)
            s = top - h[m]
            out[m] = np.cos(a * s) * u0 - w * np.sinc(a * s / PI) * s * v0
            top -= d
        u_b, g_b = self._bottom
        m = h < top
        out[m] = u_b * np.exp(g_b * (h[m] - top))
        return out.real


def _guided_mode(
    order: int,
    pol: int,
    beta: float,
    k: float,
    eps_layers: tuple[complex, ...],
    d: tuple[float, ...],
    eps_bot: complex,
    z_top: float,
    wavelength: float,
) -> GuidedMode:
    """Profile and bilinear norm by the ``(u, v = p·u′)`` transfer from ``z_top`` (S8).

    Built from the transfer matrix, *not* from R, so the residue identity
    ``Res_β R = γ₁/(βN)`` compares two independent computations (GW7).
    """
    g1 = float(np.sqrt(beta * beta - k * k))
    k_bot = k * np.sqrt(eps_bot.real)
    g_b = float(np.sqrt(beta * beta - k_bot * k_bot))
    p_b = 1.0 if pol == 2 else 1.0 / eps_bot.real
    u0, v0 = 1.0 + 0j, -g1 + 0j  # cover: u = e^{−γ₁h}, p = 1
    norm = 1.0 / (2.0 * g1)
    x, wq = np.polynomial.legendre.leggauss(NORM_GL_POINTS)
    tops = []
    for e, dj in zip(eps_layers, d, strict=True):
        a = np.sqrt(complex(k * k * e.real - beta * beta))
        w = 1.0 if pol == 2 else e.real  # 1/p
        tops.append((u0, v0, a, w, dj))
        s = 0.5 * dj * (x + 1.0)
        us = np.cos(a * s) * u0 - w * np.sinc(a * s / PI) * s * v0
        norm += 0.5 * dj * float(np.sum(wq * (us * us).real)) / w
        sinc = np.sinc(a * dj / PI) * dj
        u0, v0 = (
            np.cos(a * dj) * u0 - w * sinc * v0,
            (a * a / w) * sinc * u0 + np.cos(a * dj) * v0,
        )
    norm += p_b * float((u0 * u0).real) / (2.0 * g_b)
    return GuidedMode(
        order=order,
        pol=pol,
        beta=float(beta),
        n_eff=float(beta * wavelength / (2.0 * PI)),
        norm=norm,
        gamma=g1,
        wavelength=float(wavelength),
        _z_top=float(z_top),
        _tops=tuple(tops),
        _bottom=(float(u0.real), g_b),
    )
