"""Gielis super-formula boundary parameterisation.

Pure-geometry functions for the Gielis particle boundary. No EM physics.

Public API:
    gielis: raw Gielis coordinates at arbitrary theta values.
    boundary_setup: complete discretised boundary (f, g, df, dg, ddf, ddg, delt).
    perimeter: closed-curve arc length.
    Geometry: high-level boundary object built via the ``gielis`` factory.
"""

import numpy as np

PI = np.pi


# ---------------------------------------------------------------------------
# Gielis super-formula
# ---------------------------------------------------------------------------


def gielis(
    theta: np.ndarray,
    rad: float,
    a: float,
    b: float,
    m: int,
    n1: float,
    n2: float,
    n3: float,
    x0: float,
    z0: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Gielis super-formula boundary coordinates.

    Args:
        theta: Angular parameter values (rad).
        rad: Scale radius (nm).
        a: Scale factor for the cosine term.
        b: Scale factor for the sine term.
        m: Rotational symmetry order.
        n1: Gielis shape exponent.
        n2: Gielis shape exponent.
        n3: Gielis shape exponent.
        x0: Centre x-coordinate (nm).
        z0: Centre z-coordinate (nm).

    Returns:
        f, g: Boundary x and z coordinates (nm).
        r: Radial distance from centre.
        co, se: Intermediate quantities |cos(m θ/4)/a|^n2, |sin(m θ/4)/b|^n3.
        arg: m θ / 4.
    """
    arg = m * theta / 4.0
    co = np.abs(np.cos(arg) / a) ** n2
    se = np.abs(np.sin(arg) / b) ** n3
    r = rad * (co + se) ** (-1.0 / n1)
    f = r * np.sin(theta) + x0
    g = r * np.cos(theta) + z0
    return f, g, r, co, se, arg


def _rderiv(
    rad: float,
    n1: float,
    n2: float,
    n3: float,
    fact_n2: float,
    fact_n3: float,
    co: np.ndarray,
    se: np.ndarray,
    arg: np.ndarray,
) -> np.ndarray:
    """Compute dr/dtheta for the Gielis formula, safe at sin/cos zeros.

    At arg = k*pi, sin(arg)=0 so se*(cos/sin) is 0/0; the true limit is 0
    for n3 > 1 (se ~ |sin|^n3 → 0 faster than 1/|sin|). Same logic applies
    to co*(sin/cos) at arg = pi/2 + k*pi. We avoid the division by replacing
    the zero denominator with 1 and masking the whole term to 0.

    Args:
        rad: Scale radius (nm).
        n1: Gielis shape exponent.
        n2: Gielis shape exponent.
        n3: Gielis shape exponent.
        fact_n2: Prefactor -n2 * m / 4 for the cosine term.
        fact_n3: Prefactor n3 * m / 4 for the sine term.
        co: Intermediate quantity |cos(m θ/4)/a|^n2.
        se: Intermediate quantity |sin(m θ/4)/b|^n3.
        arg: m θ / 4.

    Returns:
        dr/dtheta evaluated at every theta.
    """
    cos_arg = np.cos(arg)
    sin_arg = np.sin(arg)
    safe_cos = np.where(cos_arg == 0.0, 1.0, cos_arg)
    safe_sin = np.where(sin_arg == 0.0, 1.0, sin_arg)
    term_c = np.where(cos_arg == 0.0, 0.0, co * sin_arg / safe_cos)
    term_s = np.where(sin_arg == 0.0, 0.0, se * cos_arg / safe_sin)
    return (
        rad
        * ((-1.0 / n1) * (co + se) ** ((-1.0 / n1) - 1.0))
        * (fact_n2 * term_c + fact_n3 * term_s)
    )


def _der_real_3(arr: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Central-difference derivative; one-sided at endpoints (DER_REAL_3).

    Args:
        arr: Sampled function values.
        x: Sample abscissae (same shape as arr).

    Returns:
        d(arr)/dx approximated at every sample point.
    """
    d_arr = np.empty_like(arr)
    d_arr[1:-1] = (arr[2:] - arr[:-2]) / (x[2:] - x[:-2])
    d_arr[0] = (arr[1] - arr[0]) / (x[1] - x[0])
    d_arr[-1] = (arr[-1] - arr[-2]) / (x[-1] - x[-2])
    return d_arr


# ---------------------------------------------------------------------------
# Uniform-theta parameterisation  (translated from subroutine etoil)
# Note: the parameter order m, n2, n1, n3 is preserved from the original
# Fortran translation.  Callers pass actual_n2 to the n2 position and
# actual_n1 to the n1 position; gielis() then receives them correctly.
# ---------------------------------------------------------------------------


def _etoil(
    nn: int,
    delt: float,
    x0: float,
    z0: float,
    rad: float,
    m: int,
    n2: float,
    n1: float,
    n3: float,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """Boundary parameterisation via the Gielis super-formula (uniform theta).

    Args:
        nn: Number of boundary quadrature points.
        delt: Angular step = 2π/nn.
        x0: Centre x-coordinate (nm).
        z0: Centre z-coordinate (nm).
        rad: Scale radius (nm).
        m: Rotational symmetry order.
        n2: Gielis exponent (note historic argument order).
        n1: Gielis exponent (note historic argument order).
        n3: Gielis exponent (note historic argument order).

    Returns:
        f, g: Boundary coordinates (nn,).
        df, dg: First derivatives w.r.t. θ.
        ddf, ddg: Second derivatives w.r.t. θ.
        theta: (nn,) the sample angles themselves.
    """
    theta = (np.arange(1, nn + 1) - 0.5) * delt
    f, g, r, co, se, arg = gielis(theta, rad, 1, 1, m, n1, n2, n3, x0, z0)
    fact_n2 = -n2 * m / 4.0
    fact_n3 = n3 * m / 4.0

    rderiv = _rderiv(rad, n1, n2, n3, fact_n2, fact_n3, co, se, arg)

    df = r * np.cos(theta) + np.sin(theta) * rderiv
    dg = -r * np.sin(theta) + np.cos(theta) * rderiv
    ddf = _der_real_3(df, theta)
    ddg = _der_real_3(dg, theta)

    return f, g, df, dg, ddf, ddg, theta


# ---------------------------------------------------------------------------
# Arc-length parameterisation  (translated from subroutine etoil_arc)
# ---------------------------------------------------------------------------


def _validated_theta(theta: np.ndarray) -> np.ndarray:
    """Check an externally supplied node set before it becomes a boundary.

    A frozen θ set is normally handed straight back from another Geometry, so
    the checks are cheap insurance against the two ways it gets mangled in
    transit — reordering and wrapping past 2π. Both produce a *plausible*
    boundary rather than a crash: ``delt`` is a bare ``np.diff``, so a
    reordered set gives negative quadrature weights and a boundary integral
    that silently counts part of the curve backwards.

    Args:
        theta: (nn,) candidate sample angles.

    Returns:
        The same angles as a float array.

    Raises:
        ValueError: If the angles are not strictly increasing, or do not fit
            inside a single 2π span.
    """
    theta = np.asarray(theta, dtype=float)
    if theta.ndim != 1 or theta.size < 3:
        raise ValueError(
            f"theta must be a 1-D array of at least 3 angles; got {theta.shape}"
        )
    if not np.all(np.diff(theta) > 0.0):
        raise ValueError(
            "theta must be strictly increasing; a reordered node set gives "
            "negative quadrature weights rather than an error"
        )
    if theta[-1] - theta[0] >= 2.0 * PI:
        raise ValueError(
            f"theta must span less than 2*pi; got {theta[-1] - theta[0]:.6f}, "
            "which traverses part of the boundary twice"
        )
    return theta


def _uniform_arc_theta(
    nn: int,
    rad: float,
    a: float,
    b: float,
    m: int,
    n1: float,
    n2: float,
    n3: float,
    x0: float = 0.0,
    z0: float = 0.0,
    n_fine: int | None = None,
) -> tuple[np.ndarray, float]:
    """Return nn theta values giving uniform arc-length spacing on the curve.

    Args:
        nn: Desired number of boundary points.
        rad: Gielis scale radius.
        a: Gielis cosine scale factor.
        b: Gielis sine scale factor.
        m: Rotational symmetry order.
        n1: Gielis exponent.
        n2: Gielis exponent.
        n3: Gielis exponent.
        x0: Centre x-coordinate (default 0).
        z0: Centre z-coordinate (default 0).
        n_fine: Fine-grid resolution (default max(10*nn, 4096)).

    Returns:
        theta_uniform: (nn,) theta values with uniform arc-length spacing.
        contour_length: Total perimeter of the curve.
    """
    if n_fine is None:
        # A function of nn alone, and it must stay one: the whole assembly is
        # scale covariant (docs/conventions.md §9) only because the theta nodes
        # this returns are rad-independent. Choosing n_fine from an absolute
        # chord length in nm would break that silently.
        n_fine = max(10 * nn, 4096)

    # Dense uniform-theta forward pass
    theta_fine = np.linspace(0, 2 * PI, n_fine, endpoint=False)
    f_fine, g_fine, *_ = gielis(theta_fine, rad, a, b, m, n1, n2, n3, x0, z0)

    # Cumulative arc length (chord-length approximation, closed curve)
    df = np.diff(f_fine, append=f_fine[0])
    dg = np.diff(g_fine, append=g_fine[0])
    ds = np.sqrt(df**2 + dg**2)
    s_fine = np.concatenate([[0.0], np.cumsum(ds[:-1])])
    contour_length = s_fine[-1] + ds[-1]  # include the closing segment

    # Invert: interpolate theta as a function of arc length
    s_uniform = np.linspace(0, contour_length, nn, endpoint=False)
    theta_uniform = np.interp(s_uniform, s_fine, theta_fine)

    return theta_uniform, contour_length


def _etoil_arc(
    nn: int,
    x0: float,
    z0: float,
    rad: float,
    a: float,
    b: float,
    m: int,
    n2: float,
    n1: float,
    n3: float,
    n_fine: int | None = None,
    theta: np.ndarray | None = None,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """Boundary parameterisation with uniform arc-length spacing.

    Drop-in replacement for _etoil() for shapes where uniform-theta sampling
    produces strongly non-uniform dipole distances (e.g. stars, high-m shapes).
    Theta values are chosen so consecutive points are evenly spaced in arc
    length; derivatives are still w.r.t. theta, but delt is a per-point array.

    Args:
        nn: Number of boundary quadrature points.
        x0: Centre x-coordinate (nm).
        z0: Centre z-coordinate (nm).
        rad: Scale radius (nm).
        a: Gielis cosine scale factor.
        b: Gielis sine scale factor.
        m: Rotational symmetry order.
        n2: Gielis exponent (note historic argument order).
        n1: Gielis exponent (note historic argument order).
        n3: Gielis exponent (note historic argument order).
        n_fine: Fine-grid resolution for arc-length inversion.
        theta: (nn,) sample angles to use **instead of** re-inverting arc
            length. This is the frozen-node path of ``docs/conventions.md``
            §10: the spacing is then uniform in arc length on whatever shape
            the angles were computed for, and only approximately so on this
            one. ``nn`` is ignored when it is given.

    Returns:
        f, g: Boundary coordinates (nn,).
        df, dg: First derivatives w.r.t. theta.
        ddf, ddg: Second derivatives w.r.t. theta.
        delt: (nn,) per-point theta step (non-uniform).
        theta: (nn,) the sample angles, uniform in arc length unless supplied.
    """
    if theta is None:
        theta, _ = _uniform_arc_theta(
            nn, rad, a, b, m, n1, n2, n3, x0=0.0, z0=0.0, n_fine=n_fine
        )
    else:
        theta = _validated_theta(theta)

    # Per-point theta differences; last step wraps around to theta[0] + 2π
    delt = np.diff(theta, append=theta[0] + 2 * PI)

    f, g, r, co, se, arg = gielis(theta, rad, a, b, m, n1, n2, n3, x0, z0)

    # Analytical first derivatives w.r.t. theta
    fact_n2 = -n2 * m / 4.0
    fact_n3 = n3 * m / 4.0
    rderiv = _rderiv(rad, n1, n2, n3, fact_n2, fact_n3, co, se, arg)
    df_theta = r * np.cos(theta) + np.sin(theta) * rderiv
    dg_theta = -r * np.sin(theta) + np.cos(theta) * rderiv
    ddf_theta = _der_real_3(df_theta, theta)
    ddg_theta = _der_real_3(dg_theta, theta)

    return f, g, df_theta, dg_theta, ddf_theta, ddg_theta, delt, theta


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def boundary_setup(
    nn: int,
    rad: float,
    a: float,
    b: float,
    m: int,
    n1: float,
    n2: float,
    n3: float,
    x0: float = 0.0,
    z0: float = 0.0,
    arc_length: bool = True,
    n_fine: int | None = None,
    theta: np.ndarray | None = None,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    float | np.ndarray,
    np.ndarray,
]:
    """Discretise the Gielis particle boundary.

    Single entry point for all downstream BIE and QNM code. Returns the
    complete set of quadrature data needed by assemble_matrix, far_field,
    and eval_field.

    Args:
        nn: Number of boundary quadrature points.
        rad: Gielis scale radius (nm).
        a: Gielis cosine scale factor.
        b: Gielis sine scale factor.
        m: Rotational symmetry order.
        n1: Gielis shape exponent.
        n2: Gielis shape exponent.
        n3: Gielis shape exponent.
        x0: Centre x-coordinate (nm). Default 0.
        z0: Centre z-coordinate (nm). Default 0.
        arc_length: If True (default) use uniform arc-length sampling
            (_etoil_arc). If False use uniform-theta sampling (_etoil; a=b=1
            forced).
        n_fine: Fine-grid resolution for arc-length inversion. None → auto.
        theta: (nn,) sample angles to use instead of re-inverting arc length;
            the frozen-node path (``docs/conventions.md`` §10). Arc-length
            sampling only.

    Returns:
        f, g: (nn,) boundary x and z coordinates (nm).
        df, dg: (nn,) first derivatives of f, g w.r.t. theta.
        ddf, ddg: (nn,) second derivatives of f, g w.r.t. theta.
        delt: Quadrature theta step. Scalar for uniform-theta; per-point array
            for arc-length sampling.
        theta: (nn,) the sample angles the boundary was evaluated at.

    Raises:
        ValueError: If ``theta`` is given with ``arc_length=False``, which has
            its own fixed node set and would ignore it.
    """
    if arc_length:
        # Note: _etoil_arc uses the historic (n2, n1, n3) argument order
        return _etoil_arc(
            nn, x0, z0, rad, a, b, m, n2, n1, n3, n_fine=n_fine, theta=theta
        )
    if theta is not None:
        raise ValueError(
            "theta is only meaningful for arc_length=True; the uniform-theta "
            "path has its own fixed node set and would ignore it"
        )
    delt = 2.0 * PI / nn
    f, g, df, dg, ddf, ddg, theta_uniform = _etoil(nn, delt, x0, z0, rad, m, n2, n1, n3)
    return f, g, df, dg, ddf, ddg, delt, theta_uniform


def perimeter(
    rad: float,
    a: float,
    b: float,
    m: int,
    n1: float,
    n2: float,
    n3: float,
    x0: float = 0.0,
    z0: float = 0.0,
    n_fine: int = 8000,
) -> float:
    """Closed-curve arc length of the Gielis boundary.

    Args:
        rad: Gielis scale radius (nm).
        a: Gielis cosine scale factor.
        b: Gielis sine scale factor.
        m: Rotational symmetry order.
        n1: Gielis shape exponent.
        n2: Gielis shape exponent.
        n3: Gielis shape exponent.
        x0: Centre x-coordinate (nm). Default 0.
        z0: Centre z-coordinate (nm). Default 0.
        n_fine: Number of points for the chord-sum approximation. Default 8000.

    Returns:
        Total perimeter (nm).
    """
    theta = np.linspace(0, 2 * PI, n_fine, endpoint=False)
    f, g, *_ = gielis(theta, rad, a, b, m, n1, n2, n3, x0, z0)
    df = np.diff(f, append=f[0])
    dg = np.diff(g, append=g[0])
    return float(np.sqrt(df**2 + dg**2).sum())


# ---------------------------------------------------------------------------
# Geometry — high-level boundary object
# ---------------------------------------------------------------------------


class Geometry:
    """Discretized Gielis boundary for BIE computation.

    Holds the quadrature arrays produced by :func:`boundary_setup` and
    exposes them through named attributes rather than a positional tuple.
    Build via the :meth:`gielis` factory; direct construction is also
    supported when arrays come from another source.

    Attributes:
        f, g: (n_pts,) boundary x and z coordinates (nm).
        df, dg: (n_pts,) first derivatives w.r.t. arc-length parameter.
        ddf, ddg: (n_pts,) second derivatives.
        delt: Quadrature weight (scalar for uniform-theta; per-point for
            arc-length).
        theta: (n_pts,) the sample angles the boundary was evaluated at.
            Stored rather than recomputed because it is the **node set**, and
            a shape derivative needs to hold it fixed across a finite
            difference (``docs/conventions.md`` §10). Recovering it after the
            fact from ``arctan2(g - z0, f - x0)`` is not equivalent: it is
            ambiguous for a boundary that is not star-convex about the centre,
            and it loses the branch for one that winds past 2π.
        rad: Gielis scale radius (nm).
        x0, z0: Particle centre coordinates (nm).
    """

    def __init__(
        self,
        f: np.ndarray,
        g: np.ndarray,
        df: np.ndarray,
        dg: np.ndarray,
        ddf: np.ndarray,
        ddg: np.ndarray,
        delt: float | np.ndarray,
        *,
        theta: np.ndarray,
        rad: float,
        x0: float = 0.0,
        z0: float = 0.0,
    ) -> None:
        """Store the boundary quadrature arrays; see the class docstring."""
        self.f = f
        self.g = g
        self.df = df
        self.dg = dg
        self.ddf = ddf
        self.ddg = ddg
        self.delt = delt
        self.theta = theta
        self.rad = rad
        self.x0 = x0
        self.z0 = z0

    @property
    def n_pts(self) -> int:
        """Number of boundary quadrature points."""
        return len(self.f)

    @property
    def is_circle(self) -> bool:
        """True if every boundary point sits at ``rad`` from the centre.

        Tested numerically rather than from the Gielis ``m`` parameter, so it
        also answers for a Geometry built directly from externally supplied
        arrays. The tolerance is ``1e-12`` relative — the circle branch of the
        superformula sets ``r = rad`` in closed form (``arg = 0`` makes
        ``co = 1``, ``se = 0``), so the deviation is rounding, not discretisation,
        and anything larger is a genuinely non-circular shape.

        Only a circle has a size parameter. Several Gielis parameter sets give
        one — ``m = 0``, and also ``n1 = n2 = n3 = 2`` at any ``m`` — which is
        why this is a numerical test rather than a check on ``m``.
        """
        r = np.hypot(self.f - self.x0, self.g - self.z0)
        return bool(np.all(np.abs(r - self.rad) <= 1e-12 * self.rad))

    @classmethod
    def gielis(
        cls,
        rad: float,
        n_pts: int = 200,
        *,
        m: int = 4,
        n1: float = 2.0,
        n2: float = 2.0,
        n3: float = 2.0,
        a: float = 1.0,
        b: float = 1.0,
        x0: float = 0.0,
        z0: float = 0.0,
        arc_length: bool = True,
        theta: np.ndarray | None = None,
    ) -> "Geometry":
        """Create a Geometry from Gielis superformula parameters.

        A circle is recovered with ``m=0`` (the superformula reduces to a
        constant radius): ``arg = 0`` makes ``co = 1`` and ``se = 0``, so
        ``r = rad`` for every theta. It is *also* recovered at
        ``n1 = n2 = n3 = 2`` with ``a = b = 1`` for **any** ``m``, which is why
        :attr:`Geometry.is_circle` tests the coordinates rather than ``m``.
        The same exponents at ``a != b`` give an exact ellipse, with semi-axis
        ``b·rad`` along x and ``a·rad`` along z — the factor on the *cosine*
        term sets the *z* semi-axis, not the x one.

        Args:
            rad: Scale radius (nm).
            n_pts: Number of boundary quadrature points.
            m: Rotational symmetry order.
            n1: Shape exponent.
            n2: Shape exponent.
            n3: Shape exponent.
            a: Cosine scale factor. **Ignored unless ``arc_length=True``.**
            b: Sine scale factor. **Ignored unless ``arc_length=True``.**
            x0: Centre x-coordinate (nm).
            z0: Centre z-coordinate (nm).
            arc_length: Use uniform arc-length sampling (default True). The
                uniform-theta path (``False``) **forces ``a = b = 1``** and
                raises nothing when passed anything else, so a derivative taken
                with respect to ``a`` or ``b`` through that path is identically
                zero rather than wrong-looking. Leave this True whenever ``a``
                or ``b`` is not 1.
            theta: (n_pts,) sample angles to use instead of re-inverting arc
                length — normally another Geometry's ``theta``. This is how a
                shape derivative holds the node set fixed across a finite
                difference; see ``docs/conventions.md`` §10 for why it must.
                ``n_pts`` is ignored when it is given.
        """
        f, g, df, dg, ddf, ddg, delt, theta_used = boundary_setup(
            n_pts,
            rad,
            a,
            b,
            m,
            n1,
            n2,
            n3,
            x0=x0,
            z0=z0,
            arc_length=arc_length,
            theta=theta,
        )
        return cls(
            f, g, df, dg, ddf, ddg, delt, theta=theta_used, rad=rad, x0=x0, z0=z0
        )
