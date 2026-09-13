"""Smooth monotone circle map ``θ = w(t)`` for near-uniform node placement.

Implements `docs/design/parametrisation-spec.md` §3–§6: closed-form Gielis
θ-derivatives to third order, the Fourier/Newton inversion of the arc-length
map, the curvature-driven near-uniform density, and the frozen
``Parametrisation`` object that carries them.

This module is **additive**. It is not wired into ``Geometry`` — the API break
that replaces the v0.5 ``np.interp`` inversion belongs to the migration spec
(v0.6-architecture §5), not here.

Conventions (spec §2, conventions §13):
    Boundary ``f = r sin θ + x0``, ``g = r cos θ + z0``; primes are ``d/dθ``,
    dots are ``d/dt``. Nodes sit at ``t_j = 2π(j + ½)/nn``. The map is anchored
    at ``T(0) = 0``, hence ``w(0) = 0``. ``σ`` is a *relative* density: its
    overall scale cancels in ``w`` and is read only by the ``nn`` derivation.
"""

from dataclasses import dataclass

import numpy as np

TWOPI = 2.0 * np.pi
EPS = float(np.finfo(float).eps)

# §4.2: the doubling loop for the shape-intrinsic representation of σγ.
N_FINE_START = 1024
N_FINE_CAP = 1 << 20

# §4.3: Newton is declared converged when the largest step is at the round-off
# floor of θ ~ 2π; the factor 8 keeps the trigger above the level at which the
# step itself is pure noise, so the two extra steps land on a settled iterate.
NEWTON_FLOOR = TWOPI * 8.0 * EPS
NEWTON_MAX_ITER = 50


# ---------------------------------------------------------------------------
# §3.1 The superformula and its u-derivatives
# ---------------------------------------------------------------------------


def _abs_pow(x: np.ndarray, expo: float) -> np.ndarray:
    """``|x|^expo``, as an exact integer power when ``expo`` is even.

    The even-integer case is the analytic one (spec §10, safe shapes): there
    ``|x|^n`` *is* ``x^n``, and writing it as the plain power keeps the value
    finite at ``x = 0`` where ``|x|`` raised to a negative float would overflow.
    Callers must skip terms whose coefficient vanishes *before* calling, since
    the ``0·∞`` those produce at ``n2 = 2`` is a ``nan``, not a zero (spec §3.1,
    implementation note).

    Args:
        x: Base, ``cos u`` or ``sin u``.
        expo: Exponent.

    Returns:
        ``|x|^expo`` elementwise.
    """
    if float(expo).is_integer() and int(expo) % 2 == 0:
        return x ** int(expo)
    return np.abs(x) ** expo


def _lobe_derivatives(
    base: np.ndarray, other: np.ndarray, scale: float, expo: float, sign: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """One superformula lobe term and its first three ``u``-derivatives.

    ``C`` and ``S`` of spec §3.1 are the same function of their own base:
    ``S`` follows from ``C`` under ``u → π/2 − u``, which swaps ``c`` with ``s``
    and flips the odd orders. ``sign = −1`` gives ``C`` (base ``c``, other
    ``s``), ``sign = +1`` gives ``S`` (base ``s``, other ``c``).

    Args:
        base: ``cos u`` for the ``C`` term, ``sin u`` for the ``S`` term.
        other: The complementary trig function.
        scale: ``a^{−n2}`` resp. ``b^{−n3}``.
        expo: ``n2`` resp. ``n3``.
        sign: ``−1`` for ``C``, ``+1`` for ``S``.

    Returns:
        The term and its ``d/du`` derivatives to third order.
    """
    n = float(expo)
    val = scale * _abs_pow(base, n)

    # Every term is coefficient-then-power: a zero coefficient (n = 0, 1 or 2
    # in the third-order term) must drop the power factor entirely, because at
    # n = 2 that factor is |c|^{-2} = ∞ at c = 0 and 0·∞ is nan.
    d1 = sign * n * scale * _abs_pow(base, n - 2.0) * base * other

    c2a = n * (n - 1.0)
    d2 = -n * val
    if c2a != 0.0:
        d2 = d2 + c2a * scale * _abs_pow(base, n - 2.0) * other**2

    c3a = n * (n - 1.0) * (n - 2.0)
    c3b = n * (3.0 * n - 2.0)
    d3 = np.zeros_like(val)
    if c3a != 0.0:
        d3 = d3 + sign * c3a * scale * _abs_pow(base, n - 4.0) * base * other**3
    if c3b != 0.0:
        d3 = d3 - sign * c3b * scale * _abs_pow(base, n - 2.0) * base * other

    return val, d1, d2, d3


def _radius_derivatives(
    theta: np.ndarray,
    rad: float,
    a: float,
    b: float,
    m: int,
    n1: float,
    n2: float,
    n3: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """``r(θ)`` and its first three θ-derivatives, in closed form (spec §3.1–3.2).

    The chain runs through ``ℓ = ln r = ln rad − (ln P)/n1`` rather than through
    ``P^{−1/n1}`` directly: ``P`` spans many decades on a sharp star, and the
    ratios ``P^{(k)}/P`` stay O(1) where the product form underflows (spec §3.2).

    Args:
        theta: Angles (rad).
        rad: Scale radius (nm).
        a: Scale factor of the cosine term.
        b: Scale factor of the sine term.
        m: Rotational symmetry order.
        n1: Gielis shape exponent.
        n2: Gielis shape exponent.
        n3: Gielis shape exponent.

    Returns:
        ``r``, ``r'``, ``r''``, ``r'''`` at every angle.
    """
    fac = m / 4.0
    u = fac * theta
    c, s = np.cos(u), np.sin(u)

    c_val, c_u, c_uu, c_uuu = _lobe_derivatives(c, s, a ** (-n2), n2, -1.0)
    s_val, s_u, s_uu, s_uuu = _lobe_derivatives(s, c, b ** (-n3), n3, +1.0)

    p0 = c_val + s_val
    p1 = fac * (c_u + s_u)
    p2 = fac**2 * (c_uu + s_uu)
    p3 = fac**3 * (c_uuu + s_uuu)

    q1 = p1 / p0
    q2 = p2 / p0 - q1**2
    q3 = p3 / p0 - 3.0 * p1 * p2 / p0**2 + 2.0 * q1**3

    l1, l2, l3 = -q1 / n1, -q2 / n1, -q3 / n1
    r = rad * np.exp(-np.log(p0) / n1)
    return (
        r,
        r * l1,
        r * (l2 + l1**2),
        r * (l3 + 3.0 * l1 * l2 + l1**3),
    )


def _speed_curvature(
    r: np.ndarray, r1: np.ndarray, r2: np.ndarray, r3: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Boundary speed ``γ`` and signed curvature ``κ`` from ``r`` (spec §3.3).

    ``N = r² + 2r'² − r r''`` is minus the cross product ``f'g'' − g'f''``: the
    Gielis curve runs clockwise in ``(x, z)`` because ``x`` comes from ``sin``
    and ``z`` from ``cos``.

    Args:
        r: Radius.
        r1: ``r'``.
        r2: ``r''``.
        r3: ``r'''`` (unused for ``κ`` itself; kept so the caller's tuple is one
            object and ``κ'`` stays one line away).

    Returns:
        ``γ = √(r² + r'²)`` and ``κ = N/γ³``.
    """
    del r3
    gam = np.sqrt(r**2 + r1**2)
    return gam, (r**2 + 2.0 * r1**2 - r * r2) / gam**3


# ---------------------------------------------------------------------------
# §5 The near-uniform density
# ---------------------------------------------------------------------------


def _layer_bandwidth(kappa: np.ndarray) -> float:
    """Gaussian smoothing width ``M = 2π / w_layer`` of the density (spec §5.3).

    ``w_layer`` is the angular width on which ``ln|κ|`` sits within 1 of its
    maximum — on a star, the angular extent of the arm tip, which narrows as the
    exponent rises. Tying ``M`` to it puts the bandwidth where the curvature
    layer is; tying it to ``nn`` would make ``w`` a different map at every
    resolution (invariant 13.3).

    ``M`` is real, not ``ceil``'d: an integer-valued shape functional steps in
    parameter space, and a step inside a continuation trajectory is read as a
    defect. The remaining step is the Heaviside count itself, a documented
    non-goal (spec §1).

    Args:
        kappa: ``|κ(θ)|`` on the fine grid.

    Returns:
        The real Gaussian bandwidth ``M``.
    """
    lk = np.log(kappa)
    inside = float((lk > lk.max() - 1.0).sum())
    return kappa.size / inside


def _density(
    kappa: np.ndarray, gam: np.ndarray, contrast: float, alpha: float
) -> tuple[np.ndarray, float, float]:
    """Relative node density ``σ(θ)`` on the fine grid (spec §5.1–5.2).

    ``u = |κ|·L/2π`` is identically 1 on a circle and carries no absolute
    length, which is what keeps scale covariance (conventions §9) structural.
    ``s̃`` is ``ln u`` Gaussian-smoothed; a positive kernel cannot leave the range
    of ``ln u``, where a sharp spectral cutoff would Gibbs-undershoot and dig an
    artificial minimum that then sets ``nn``.

    The exponent is the single smooth formula ``α_eff = α ln C/(α Δ + ln C)``:
    ``C = 1`` gives ``α_eff = 0`` and a circle gives ``Δ = 0``, both through the
    same code path, and the realised log-contrast ``α_eff Δ < ln C`` can never
    exceed the band, so no clamp is needed (spec §5.2).

    Args:
        kappa: ``|κ(θ)|`` on the fine grid.
        gam: ``γ(θ)`` on the fine grid.
        contrast: ``C = R_max/R_min ≥ 1``, the requested contrast.
        alpha: Upper bound on the curvature exponent.

    Returns:
        ``σ`` on the grid (mean-anchored), ``α_eff``, and the realised
        log-contrast ``α_eff Δ``.
    """
    n_f = kappa.size
    perimeter = float(gam.mean()) * TWOPI
    log_u = np.log(kappa * perimeter / TWOPI)

    bandwidth = _layer_bandwidth(kappa)
    freq = np.fft.fftfreq(n_f, d=1.0 / n_f)
    taper = np.exp(-0.5 * (freq / bandwidth) ** 2)
    smooth = np.real(np.fft.ifft(np.fft.fft(log_u) * taper))

    spread = float(smooth.max() - smooth.min())
    log_c = np.log(contrast)
    # 0/0 only when C = 1 *and* the shape has no curvature contrast at all; the
    # limit is 0 along either axis, so the realised grading is σ ≡ 1 either way.
    denom = alpha * spread + log_c
    alpha_eff = alpha * log_c / denom if denom > 0.0 else 0.0

    # The mean, not the max: an anchor at an extremum puts a max over a grid
    # inside the map. σ's scale cancels in T (§4.1), so the anchor is free.
    return np.exp(alpha_eff * (smooth - smooth.mean())), alpha_eff, alpha_eff * spread


# ---------------------------------------------------------------------------
# The object
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Nodes:
    """Quadrature nodes of a :class:`Parametrisation` at one ``nn``.

    Attributes:
        t: ``t_j = 2π(j + ½)/nn``, the equispaced quadrature parameter.
        theta: ``θ_j = w(t_j)``.
        dw: ``w'(t_j)``.
        ddw: ``w''(t_j)``.
    """

    t: np.ndarray
    theta: np.ndarray
    dw: np.ndarray
    ddw: np.ndarray


@dataclass(frozen=True)
class Parametrisation:
    """A smooth monotone circle map ``θ = w(t)``, frozen at construction.

    ``w`` is the inverse of ``T(θ) = (2π/Z)∫₀^θ σγ``, held as the truncated
    real trig series of ``σγ`` from which ``T`` is the exact antiderivative.
    ``w'`` and ``w''`` are taken from that same series, never from the analytic
    ``γ``: the Jacobian must be the exact derivative of the map whose root was
    found, or node and Jacobian disagree at the truncation level (spec §4.1).

    ``w`` depends on the shape, ``r_band``, ``n_core``, ``wavelength_ref`` and
    ``alpha`` — and on nothing else. Not on ``nn`` (invariant 13.3), not on the
    solve wavelength (13.2), not on ``rad`` (conventions §9).

    Attributes:
        a0: Mean of ``σγ``; ``Z = 2π a_0``.
        coef: ``c_k`` for ``k = 1…K`` of ``σγ = a_0 + 2 Re Σ c_k e^{ikθ}``.
        theta_fine: The uniform θ grid the series was sampled on.
        t_fine: ``T`` on that grid, used to seed Newton.
        nn_from_band: ``nn`` derived from the ``R`` band (spec §5.4).
        contrast_realised: ``exp(α_eff Δ) ≤ C`` — the bound, not the promise.
        alpha_eff: The curvature exponent actually used.
        n_fine: ``N_f``, shape-intrinsic (spec §4.2).
        n_terms: ``K``, the number of retained harmonics.
    """

    a0: float
    coef: np.ndarray
    theta_fine: np.ndarray
    t_fine: np.ndarray
    nn_from_band: int
    contrast_realised: float
    alpha_eff: float
    n_fine: int
    n_terms: int

    # -- the series ---------------------------------------------------------

    def _series(self, theta: np.ndarray, order: int = 0) -> np.ndarray:
        """Evaluate the truncated ``σγ`` series, a derivative, or its integral.

        Args:
            theta: Angles (rad).
            order: ``0`` for ``σγ``, ``1`` for ``(σγ)'``, ``−1`` for the
                antiderivative with the secular ``a_0 θ`` term dropped (the
                caller adds it, since it is not periodic).

        Returns:
            The requested quantity at every angle.
        """
        modes = np.arange(1, self.coef.size + 1)
        fac = (1j * modes) ** order if order >= 0 else 1.0 / (1j * modes)
        phase = np.exp(1j * np.outer(theta, modes))
        val = 2.0 * np.real(phase @ (self.coef * fac))
        return val + (self.a0 if order == 0 else 0.0)

    def _t_of(self, theta: np.ndarray) -> np.ndarray:
        """``T(θ)``, the exact antiderivative of the truncated series.

        Anchored at ``T(0) = 0`` (spec §2), so ``w(0) = 0``. The normalisation
        ``2π/Z = 1/a_0`` makes ``T(2π) = 2π`` exactly, hence ``w`` a circle map.

        Args:
            theta: Angles (rad).

        Returns:
            ``T(θ)``.
        """
        anti = self._series(theta, -1) - self._series(np.zeros(1), -1)[0]
        return (self.a0 * theta + anti) / self.a0

    # -- the inverse --------------------------------------------------------

    def nodes(self, nn: int) -> Nodes:
        """Place ``nn`` nodes equispaced in ``t`` and return ``w``, ``w'``, ``w''``.

        Pure: the same ``nn`` returns the same arrays. ``nn`` enters here and
        nowhere in the map, so the nodes of two resolutions lie on one curve
        (invariant 13.3).

        Newton is seeded by linear interpolation on the fine grid — a
        convergence aid only, since Newton on a strictly monotone smooth ``T``
        reaches the unique root from any seed in the bracket, so the seed's C⁰
        character does not reach ``w``. Steps that leave the bracketing fine-grid
        interval bisect instead. Iteration stops two full steps *after* the
        largest step first falls to the round-off floor: a tolerance-based stop
        leaves a residual that jumps as a shape parameter moves, which is
        exactly the step a continuation trajectory reads as a defect (spec §4.3).

        Args:
            nn: Number of nodes.

        Returns:
            The :class:`Nodes` at ``t_j = 2π(j + ½)/nn``.

        Raises:
            ValueError: If Newton has not converged within 50 iterations.
        """
        t = TWOPI * (np.arange(nn) + 0.5) / nn
        # The grid is closed with (T, θ) = (2π, 2π) so that a node in the last
        # interval is seeded and bracketed like any other. T is strictly
        # increasing, so searchsorted locates the bracketing interval exactly.
        theta_ext = np.append(self.theta_fine, TWOPI)
        theta = np.interp(t, self.t_fine, theta_ext)
        idx = np.clip(np.searchsorted(self.t_fine, t) - 1, 0, self.n_fine - 1)
        lo, hi = theta_ext[idx], theta_ext[idx + 1]

        extra = -1
        for _ in range(NEWTON_MAX_ITER):
            resid = self._t_of(theta) - t
            trial = theta - resid * self.a0 / self._series(theta)  # T' = σγ/a_0
            # The bracket is the *static* fine-grid interval holding the root
            # (spec §4.3). Shrinking it onto the iterate the way a textbook
            # rtsafe does is what breaks here: once an endpoint sits within
            # round-off of the root, the converged Newton trial falls a few ulp
            # outside it and gets bisected back out to the interval midpoint,
            # a 1e-8 excursion that then repeats. The bracket is also closed,
            # so a root landing exactly on a grid point is not read as an escape.
            outside = (trial < lo) | (trial > hi)
            trial = np.where(outside, 0.5 * (lo + hi), trial)
            step = np.max(np.abs(trial - theta))
            theta = trial
            if extra >= 0:
                extra += 1
                if extra == 2:
                    break
            elif step <= NEWTON_FLOOR:
                extra = 0
        else:
            raise ValueError(
                "Newton inversion of the parametrisation did not converge "
                "in 50 iterations"
            )

        dw = self.a0 / self._series(theta)
        return Nodes(
            t=t, theta=theta, dw=dw, ddw=-self._series(theta, 1) / self.a0 * dw**3
        )

    # -- construction -------------------------------------------------------

    @classmethod
    def gielis(
        cls,
        *,
        rad: float,
        a: float,
        b: float,
        m: int,
        n1: float,
        n2: float,
        n3: float,
        r_band: tuple[float, float] = (15.0, 15.0),
        n_core: float,
        wavelength_ref: float = 1550.0,
        alpha: float = 0.5,
    ) -> "Parametrisation":
        """Build the map for a Gielis boundary.

        Every shape goes through §4–§5; there is no special case for the circle
        or for ``C = 1`` (spec §6, contract 2), so the Mie anchor exercises the
        code the stars run.

        The representation of ``σγ`` is a function of the shape alone: ``N_f``
        doubles from 1024 until the top quarter of the spectrum has decayed to
        the round-off plateau, and ``K`` is the last coefficient above ``eps``
        *relative to* ``a_0``. Relative, so that ``rad → s·rad`` leaves ``N_f``
        and ``K`` identical and conventions §9 holds structurally.

        Args:
            rad: Scale radius (nm).
            a: Scale factor of the cosine term.
            b: Scale factor of the sine term.
            m: Rotational symmetry order.
            n1: Gielis shape exponent.
            n2: Gielis shape exponent. Even integers are the analytic shapes.
            n3: Gielis shape exponent. Even integers are the analytic shapes.
            r_band: ``(R_min, R_max)`` in points per interior wavelength
                (conventions §12). ``R_min`` fixes ``nn``; the ratio
                ``C = R_max/R_min`` is the requested grading contrast. The
                default is a degenerate band, ``C = 1``: uniform arc length is
                the reference design (spec §1).
            n_core: Core refractive index, which sets the interior wavelength.
            wavelength_ref: The map's own fixed reference λ_vac (nm), never
                taken from a solve — a λ-dependent map destroys the holomorphy
                Beyn's contour rests on (conventions §13.2).
            alpha: Upper bound on the curvature exponent of the density.

        Returns:
            The frozen :class:`Parametrisation`.

        Raises:
            ValueError: If ``σγ`` has not resolved by ``N_f = 2^20`` — a curve
                that needs more is outside the safe-shape envelope, and a silent
                truncation is a plausible wrong map.
        """
        contrast = r_band[1] / r_band[0]
        n_f = N_FINE_START
        while True:
            theta_f = np.linspace(0.0, TWOPI, n_f, endpoint=False)
            r, r1, r2, r3 = _radius_derivatives(theta_f, rad, a, b, m, n1, n2, n3)
            gam, kappa = _speed_curvature(r, r1, r2, r3)
            sigma, alpha_eff, log_realised = _density(
                np.abs(kappa), gam, contrast, alpha
            )
            spec = np.fft.rfft(sigma * gam) / n_f
            decay = np.abs(spec[1:]) / np.abs(spec[0].real)
            # 4·eps over the top quarter, not eps: the FFT of an analytic
            # function plateaus at a few eps, and reading that plateau as
            # "unconverged" would double N_f forever.
            if decay[3 * n_f // 8 :].max() <= 4.0 * EPS:
                break
            n_f *= 2
            if n_f > N_FINE_CAP:
                raise ValueError(
                    "sigma*gamma does not resolve below N_f = 2**20; the shape "
                    "is outside the safe-shape envelope (even n2, n3)"
                )

        # K = 0 (no harmonic above eps) is the circle, and an empty series is
        # the right answer there: T(θ) = θ exactly, through the same code path.
        n_terms = int(np.max(np.flatnonzero(decay > EPS) + 1, initial=0))
        obj = cls(
            a0=float(spec[0].real),
            coef=spec[1 : n_terms + 1].copy(),
            theta_fine=theta_f,
            t_fine=np.empty(0),
            nn_from_band=0,
            contrast_realised=float(np.exp(log_realised)),
            alpha_eff=float(alpha_eff),
            n_fine=n_f,
            n_terms=n_terms,
        )
        # T on the fine grid seeds Newton; it needs the series, so it is filled
        # in after construction. The last entry is the 2π closure T(2π) = 2π,
        # which brackets the nodes in the final interval.
        t_fine = np.concatenate([obj._t_of(theta_f), [TWOPI]])
        object.__setattr__(obj, "t_fine", t_fine)

        # §5.4, nn from the lower end of the band. Node spacing in arc length is
        # Δs_j = γ(θ_j) w'(t_j) (2π/nn), so the worst-resolved node carries
        # R = (λ_ref/n_core)/max Δs. The first estimate uses the *continuum*
        # minimum of σ, i.e. Z_min = ∫(σ/min σ)γ dθ — conservative, because nodes
        # are sparsest exactly where σ is smallest and a narrow minimum may hold
        # no node at all, over-resolving by up to ~2×. σ's scale cancels in the
        # map, so it is reinstated here and only here.
        lam_core = wavelength_ref / n_core

        def worst_r(nn: int) -> float:
            nodes = obj.nodes(nn)
            rr = _radius_derivatives(nodes.theta, rad, a, b, m, n1, n2, n3)
            gam_n, _ = _speed_curvature(*rr)
            return float((lam_core / (gam_n * nodes.dw * TWOPI / nn)).min())

        z_min = TWOPI * obj.a0 / float(sigma.min())
        nn = int(np.ceil(r_band[0] * z_min * n_core / wavelength_ref))
        nn += nn % 2
        # Descend from the estimate, **verifying** each candidate rather than
        # extrapolating: the nodes move when nn changes, so an extrapolated nn
        # can land below the band it was derived from.
        for _ in range(8):
            cand = int(np.ceil(nn * r_band[0] / worst_r(nn)))
            cand += cand % 2
            if cand >= nn or cand < 4 or worst_r(cand) < r_band[0]:
                break
            nn = cand
        object.__setattr__(obj, "nn_from_band", nn)
        return obj
