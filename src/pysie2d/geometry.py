"""Gielis super-formula boundary parameterisation.

Pure-geometry functions for the Gielis particle boundary. No EM physics.

Public API:
    gielis: raw Gielis coordinates at arbitrary theta values.
    perimeter: closed-curve arc length.
    Geometry: high-level boundary object built via the ``gielis`` factory, on
        nodes equispaced in the quadrature parameter t of a
        :class:`~pysie2d.parametrisation.Parametrisation`.
"""

import numpy as np

from .parametrisation import Nodes, Parametrisation

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


def _rderiv2(
    rad: float,
    n1: float,
    n2: float,
    n3: float,
    fact_n2: float,
    fact_n3: float,
    m: int,
    co: np.ndarray,
    se: np.ndarray,
    arg: np.ndarray,
) -> np.ndarray:
    """Compute d²r/dθ² for the Gielis formula, safe at sin/cos zeros.

    Closed form, differentiating S = co + se a second time (S' is
    ``_rderiv``'s ``fact_n2*co*tan(arg) + fact_n3*se*cot(arg)``):

        S'' = co*[fact_n2²·tan²(arg) + fact_n2·(m/4)·sec²(arg)]
            + se*[fact_n3²·cot²(arg) - fact_n3·(m/4)·csc²(arg)]

    then r = rad·S^p with p = -1/n1 gives
    r'' = rad·p(p-1)·S^(p-2)·(S')² + rad·p·S^(p-1)·S''. The zero-denominator
    guard mirrors ``_rderiv``: at cos(arg)=0, co·tan²(arg) ~ |cos|^(n2-2),
    which is the analytic limit 0 for n2 > 2 but a genuine singularity for
    n2 < 2 (a real corner, not a masking artefact) — this function is only
    valid away from that regime, matching the "no true corners" scope.

    Args:
        rad: Scale radius (nm).
        n1: Gielis shape exponent.
        n2: Gielis shape exponent.
        n3: Gielis shape exponent.
        fact_n2: Prefactor -n2 * m / 4 for the cosine term.
        fact_n3: Prefactor n3 * m / 4 for the sine term.
        m: Rotational symmetry order (needed directly, not just via the
            fact_n2/fact_n3 already-scaled prefactors).
        co: Intermediate quantity |cos(m θ/4)/a|^n2.
        se: Intermediate quantity |sin(m θ/4)/b|^n3.
        arg: m θ / 4.

    Returns:
        d²r/dθ² evaluated at every theta.
    """
    cos_arg = np.cos(arg)
    sin_arg = np.sin(arg)
    safe_cos = np.where(cos_arg == 0.0, 1.0, cos_arg)
    safe_sin = np.where(sin_arg == 0.0, 1.0, sin_arg)
    tan_arg = np.where(cos_arg == 0.0, 0.0, sin_arg / safe_cos)
    cot_arg = np.where(sin_arg == 0.0, 0.0, cos_arg / safe_sin)
    sec2 = np.where(cos_arg == 0.0, 0.0, 1.0 + tan_arg**2)
    csc2 = np.where(sin_arg == 0.0, 0.0, 1.0 + cot_arg**2)
    term_co = np.where(
        cos_arg == 0.0,
        0.0,
        co * (fact_n2**2 * tan_arg**2 + fact_n2 * (m / 4.0) * sec2),
    )
    term_se = np.where(
        sin_arg == 0.0,
        0.0,
        se * (fact_n3**2 * cot_arg**2 - fact_n3 * (m / 4.0) * csc2),
    )
    s_dd = term_co + term_se

    s = co + se
    s_d = np.where(cos_arg == 0.0, 0.0, fact_n2 * co * tan_arg) + np.where(
        sin_arg == 0.0, 0.0, fact_n3 * se * cot_arg
    )
    p = -1.0 / n1
    return (
        rad * p * (p - 1.0) * s ** (p - 2.0) * s_d**2 + rad * p * s ** (p - 1.0) * s_dd
    )


# ---------------------------------------------------------------------------
# Discretised boundary on a parametrisation
# ---------------------------------------------------------------------------


class NonClosingBoundaryError(ValueError):
    """The superformula boundary does not close after one turn, θ → θ + 2π.

    Odd ``m`` closes only in the symmetric case ``a == b, n2 == n3``; ``m = 1``
    is the case most likely to be hit by accident (no ``m``-fold symmetry to
    mask an asymmetric ``a``/``b`` or ``n2``/``n3``), so this is its own type
    rather than a bare ``ValueError`` — a caller can catch it without matching
    on message text. Subclasses ``ValueError`` so existing ``except
    ValueError`` and ``pytest.raises(ValueError, ...)`` call sites still work.
    """


def _closes(m: float, a: float, b: float, n2: float, n3: float) -> bool:
    """Whether the superformula boundary closes after one turn, θ → θ + 2π.

    ``|cos u|^n2/a^n2 + |sin u|^n3/b^n3`` with ``u = mθ/4`` has period π in
    ``u``, and period π/2 only when the swap ``cos ↔ sin`` leaves it unchanged,
    i.e. ``a = b`` and ``n2 = n3``. One turn advances ``u`` by ``mπ/2``, so the
    curve closes for every even integer ``m``, and for odd ``m`` only in the
    symmetric case (D5, generalised to ``n2 ≠ n3``). A non-closing ``r(θ)`` is
    not 2π-periodic, and the periodic quadrature would integrate a curve with
    a jump at θ = 0 while raising nothing.

    Args:
        m: Rotational symmetry order.
        a: Scale factor of the cosine term.
        b: Scale factor of the sine term.
        n2: Exponent of the cosine term.
        n3: Exponent of the sine term.

    Returns:
        True if ``r(θ + 2π) = r(θ)``.
    """
    return float(m).is_integer() and (int(m) % 2 == 0 or (a == b and n2 == n3))


def _checked_parametrisation(parametrisation: object) -> None:
    """Refuse anything that is not a :class:`Parametrisation`, naming the v0.5 call.

    The one caller this exists for passes a v0.5 ``theta`` array. Duck typing
    would let an array fail later as an ``AttributeError`` on ``nodes`` with no
    hint of the API change behind it.

    Raises:
        TypeError: If ``parametrisation`` is not ``None`` or a Parametrisation.
    """
    if parametrisation is not None and not isinstance(parametrisation, Parametrisation):
        raise TypeError(
            "parametrisation must be a pysie2d.Parametrisation, got "
            f"{type(parametrisation).__name__}. v0.6 freezes the node *map*, not "
            "the angles: replace theta=other.theta with "
            "parametrisation=other.parametrisation"
        )


def _boundary_arrays(
    nodes: Nodes,
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
    """Boundary coordinates and their t-derivatives at the nodes of a map.

    The superformula is differentiated in θ in closed form (``_rderiv``,
    ``_rderiv2``) and carried to the quadrature parameter by the chain rule
    through ``θ = w(t)``:

        ẋ = x'·w',        ẍ = x''·w'² + x'·w''

    Kress needs derivatives in ``t``, the variable the nodes are equispaced in
    (conventions §13.1). On the identity map ``w' = 1`` and ``w'' = 0`` and the
    two coincide.

    Args:
        nodes: Nodes of a :class:`Parametrisation` at the wanted ``nn``.
        rad: Scale radius (nm).
        a: Scale factor of the cosine term.
        b: Scale factor of the sine term.
        m: Rotational symmetry order.
        n1: Gielis shape exponent.
        n2: Gielis shape exponent.
        n3: Gielis shape exponent.
        x0: Centre x-coordinate (nm).
        z0: Centre z-coordinate (nm).

    Returns:
        f, g, df, dg, ddf, ddg at the nodes, derivatives with respect to t.
    """
    theta = nodes.theta
    f, g, r, co, se, arg = gielis(theta, rad, a, b, m, n1, n2, n3, x0, z0)
    fact_n2 = -n2 * m / 4.0
    fact_n3 = n3 * m / 4.0
    rderiv = _rderiv(rad, n1, n2, n3, fact_n2, fact_n3, co, se, arg)
    rderiv2 = _rderiv2(rad, n1, n2, n3, fact_n2, fact_n3, m, co, se, arg)

    sin_t = np.sin(theta)
    cos_t = np.cos(theta)
    df_theta = r * cos_t + sin_t * rderiv
    dg_theta = -r * sin_t + cos_t * rderiv
    ddf_theta = rderiv2 * sin_t + 2.0 * rderiv * cos_t - r * sin_t
    ddg_theta = rderiv2 * cos_t - 2.0 * rderiv * sin_t - r * cos_t

    dw = nodes.dw
    ddw = nodes.ddw
    return (
        f,
        g,
        df_theta * dw,
        dg_theta * dw,
        ddf_theta * dw**2 + df_theta * ddw,
        ddg_theta * dw**2 + dg_theta * ddw,
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


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

    Holds the boundary sampled at nodes **equispaced in a quadrature parameter
    t**, with derivatives taken with respect to that same ``t``
    (``docs/conventions.md`` §5, §13.1). Build via the :meth:`gielis` factory;
    direct construction is also supported when arrays come from another source,
    under the same contract — see :meth:`__init__`.

    Attributes:
        f, g: (n_pts,) boundary x and z coordinates (nm).
        df, dg: (n_pts,) first derivatives with respect to t.
        ddf, ddg: (n_pts,) second derivatives with respect to t.
        parametrisation: The :class:`~pysie2d.parametrisation.Parametrisation`
            the arrays were sampled on, or ``None`` for a boundary whose arrays
            came from somewhere with no map to report. It is the **frozen
            object** of a shape derivative (conventions §10, §13.3):
            :meth:`pysie2d.qnm.QNMResult.sensitivity` requires every perturbed
            geometry to be built on the base geometry's map, and refuses a
            geometry that has none.
        nodes: ``parametrisation.nodes(n_pts)`` — ``t``, ``θ = w(t)``, ``w'``,
            ``w''`` — or ``None`` with the parametrisation.
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
        *,
        rad: float,
        x0: float = 0.0,
        z0: float = 0.0,
        parametrisation: Parametrisation | None = None,
    ) -> None:
        """Store the boundary quadrature arrays.

        The arrays must be samples at ``t_j = 2π(j + ½)/n_pts`` (any fixed
        offset is equivalent) with ``df … ddg`` differentiated with respect to
        ``t``, traversing the boundary in the same sense as :meth:`gielis`.
        Nothing here can check that, and arrays sampled any other way assemble
        into a plausible matrix with first-order error.

        Args:
            f: (n_pts,) boundary x coordinates (nm).
            g: (n_pts,) boundary z coordinates (nm).
            df: (n_pts,) df/dt.
            dg: (n_pts,) dg/dt.
            ddf: (n_pts,) d²f/dt².
            ddg: (n_pts,) d²g/dt².
            rad: Gielis scale radius (nm).
            x0: Centre x-coordinate (nm).
            z0: Centre z-coordinate (nm).
            parametrisation: The map the arrays were sampled on. Optional to
                store, mandatory to differentiate (conventions §10).

        Raises:
            TypeError: If ``parametrisation`` is not a Parametrisation.
        """
        _checked_parametrisation(parametrisation)
        self.f = f
        self.g = g
        self.df = df
        self.dg = dg
        self.ddf = ddf
        self.ddg = ddg
        self.rad = rad
        self.x0 = x0
        self.z0 = z0
        self.parametrisation = parametrisation
        self.nodes = None if parametrisation is None else parametrisation.nodes(len(f))

    @property
    def n_pts(self) -> int:
        """Number of boundary quadrature points."""
        return len(self.f)

    @property
    def delt(self) -> float:
        """Trapezoid step ``h = 2π/n_pts`` in the quadrature parameter t.

        Not a free quantity: Kress's weights presume exactly this step, so it
        is derived from ``n_pts`` rather than stored.
        """
        return 2.0 * PI / self.n_pts

    @property
    def theta(self) -> np.ndarray | None:
        """(n_pts,) node angles ``θ_j = w(t_j)``, or ``None`` without a map.

        Where the nodes are, for plotting and diagnostics. It is **not** the
        frozen object: assembly also needs ``w'`` and ``w''``, which cannot be
        recovered from the angles (conventions §13.3).
        """
        return None if self.nodes is None else self.nodes.theta

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
        n_pts: int = 100,
        *,
        m: int = 4,
        n1: float = 2.0,
        n2: float = 2.0,
        n3: float = 2.0,
        a: float = 1.0,
        b: float = 1.0,
        x0: float = 0.0,
        z0: float = 0.0,
        parametrisation: Parametrisation | None = None,
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
            n_pts: Number of boundary quadrature points. The default of
                100 is comfortably past the ``nn ≈ 30–40`` where the
                circle anchors reach round-off under Kress quadrature;
                the first-order scheme it replaced needed 200.
            m: Rotational symmetry order.
            n1: Shape exponent.
            n2: Shape exponent.
            n3: Shape exponent.
            a: Cosine scale factor.
            b: Sine scale factor.
            x0: Centre x-coordinate (nm).
            z0: Centre z-coordinate (nm).
            parametrisation: The node map ``θ = w(t)``. ``None`` (default) is
                :meth:`Parametrisation.uniform_theta`, nodes equispaced in θ.
                Pass another geometry's ``parametrisation`` to hold the map
                fixed across a shape derivative (conventions §10).
                ``Parametrisation.gielis(...)`` gives uniform arc length, which
                is not more accurate on any shape measured, spiky stars at low
                resolution included (conventions §13).

        Raises:
            TypeError: If ``parametrisation`` is not a Parametrisation — in
                particular a v0.5 ``theta`` array.
            NonClosingBoundaryError: If the boundary does not close after one
                turn: odd ``m`` unless ``a = b`` and ``n2 = n3``, or
                non-integer ``m``.
        """
        _checked_parametrisation(parametrisation)
        if not _closes(m, a, b, n2, n3):
            raise NonClosingBoundaryError(
                f"the boundary does not close at m={m}, a={a}, b={b}, n2={n2}, "
                f"n3={n3}: r(θ + 2π) ≠ r(θ). Odd m needs a == b and n2 == n3"
            )
        if parametrisation is None:
            parametrisation = Parametrisation.uniform_theta()
        f, g, df, dg, ddf, ddg = _boundary_arrays(
            parametrisation.nodes(n_pts), rad, a, b, m, n1, n2, n3, x0, z0
        )
        return cls(
            f,
            g,
            df,
            dg,
            ddf,
            ddg,
            rad=rad,
            x0=x0,
            z0=z0,
            parametrisation=parametrisation,
        )
