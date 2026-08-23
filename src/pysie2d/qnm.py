"""QNMSolver / QNMResult façade for quasi-normal-mode extraction.

A quasi-normal mode is a source-free solution: a complex wavelength where the
BIE operator M(λ) is singular. :class:`QNMSolver` finds every one inside a
rectangle of the complex λ-plane by contour integration (:mod:`pysie2d.beyn`),
with no initial guess and no scan.

Mirrors the driven solver: same ``(geometry, material)`` construction, one
method returning a result object, derived physics as properties.

Conventions (docs/conventions.md §§2, 8). Wavelengths are vacuum nm, in and
out: the search rectangle is given in vacuum λ and the mode wavelengths come
back in vacuum λ. There is no conversion on the return leg because the contour
is drawn directly on :meth:`pysie2d.solver.BIESolver.assemble`, which is itself
the single vacuum-to-background conversion point. Under
``exp(-iωt)`` a decaying mode has ``Im ω < 0``, hence ``Im k < 0``, hence
**``Im λ > 0``** — a search box must lie in the upper half λ-plane, and in
``Re λ > 0`` so that every Hankel argument stays off the ``H^(1)`` branch cut.
Both are asserted. Poles do *not* come in conjugate pairs: the reality
condition is ``λ → −λ̄``, which puts mirror partners at negative ``Re λ``.
"""

from collections.abc import Callable
from dataclasses import dataclass, replace

import numpy as np

from .beyn import beyn_modes, newton_refine
from .geometry import Geometry
from .material import Material
from .solver import BIESolver, size_parameter

# Two eigenvalues closer than this, relatively, are the same mode: the ±n
# partners of a circle. *(measured: a true degenerate pair agrees to 2.4e-13,
# while the nearest distinct mode in the same spectrum is 20 nm away, so the
# grouping is not a close call.)*
DEGENERACY_RTOL = 1.0e-9

# Above this condition number the bordered Newton system has no single
# well-defined solution and its step is discarded. Bordered Newton assumes a
# one-dimensional null space; a degenerate pole has a two-dimensional one, so
# the Jacobian is singular in exact arithmetic. The threshold is not a close
# call *(measured: cond(J) = 1.5e3 on the simple TE n=0 pole against 3.2e15 and
# 5.4e15 on the degenerate TE n=3 pair — twelve orders of magnitude apart)*,
# which is why a single fixed number is defensible here rather than a tuned or
# adaptive one: anything from ~1e8 to ~1e14 classifies this spectrum
# identically.
#
# Exported because it is the one number a user needs to reproduce a decision
# the library made on their behalf. ``QNMResult.refine`` silently keeps the
# contour estimate wherever ``cond_jacobian`` exceeds this, so a caller reading
# ``converged == False`` can only tell "degenerate, skipped by design" from
# "Newton failed" by comparing against the same threshold — and hard-coding
# 1e12 at the call site would silently drift the day this value changes.
DEGENERATE_COND = 1.0e12

# Central-difference step for dM/dp, in the parameter's own units
# (docs/conventions.md §10). Under the frozen node set the h-ladder is exactly
# ×100 per decade until the cancellation floor at ~1e-8, with truncation at
# ~1e-7 a decade above 1e-5 — so this is chosen on having about a decade of
# margin on each side, not on the best value at one design point: the
# truncation coefficient scales with the parameter's geometric leverage, which
# moves across a shape catalogue.
SHAPE_STEP = 1.0e-5


@dataclass(frozen=True)
class QNMResult:
    """Quasi-normal modes found inside a search rectangle.

    Attributes:
        wavelengths: complex (K,) mode wavelengths in vacuum nm, sorted by
            ``Re λ``. Degenerate partners appear as separate, numerically
            equal entries — see ``multiplicity``.
        vectors: complex (2·n_pts, K) boundary-field vectors of each mode, unit
            columns in the ``φ`` / ``χ`` layout of the driven solver. Not
            normalised as mode fields; that needs a QNM norm, which is out of
            scope (v0.5).
        multiplicity: int (K,) how many entries share this wavelength. Every
            ``n ≥ 1`` mode of a circle is doubly degenerate through
            ``exp(±inθ)``; only ``n = 0`` is simple.
        sigma_ratio: float (K,) ``σ_min/σ_max`` of M at each mode. The
            singularity measure, and dimensionless — an absolute ``σ_min``
            means nothing, since it carries the scale of the operator.
        sv_ratio: float (n_probe,) singular-value spectrum of the Beyn moment
            matrix, divided by its largest. The rank gap lives here.
        max_gap: float largest ratio between consecutive singular values.
        rank: int rank detected in the moment matrix. This can exceed
            ``n_modes``: a pole just *outside* the contour leaks a rank
            direction in through imperfect quadrature cancellation, and its
            eigenvalue is then discarded by the in-contour filter.
        cancellation: float how completely the contour integral cancelled.
        edge_margin: float (K,) distance from each mode to the nearest contour
            edge, as a fraction of the shorter side. A value near zero means
            the box is clipping the pole and the result is not trustworthy.
        converged: bool (K,) whether bordered Newton converged on each mode.
            Straight out of :meth:`QNMSolver.modes` this is **all False, which
            means no refinement was attempted** — not that anything failed. A
            mode from a well-drawn contour is already converged to far below
            the discretisation error and needs no refinement; see
            :meth:`refine`, which is the only thing that sets this True.
        cond_jacobian: float (K,) condition number of the bordered Newton
            Jacobian at each mode, ``NaN`` where refinement was not attempted.
            Above ``DEGENERATE_COND`` the pole is degenerate — the bordered
            system assumes a one-dimensional null space and a degenerate pole
            has two — so :meth:`refine` keeps the contour estimate and reports
            it here rather than returning a polished-looking number. Read
            alongside ``multiplicity``: the two detect degeneracy by
            independent means, eigenvalue spacing versus the conditioning of
            the actual linear algebra, and a disagreement is informative. A
            ``multiplicity`` of 1 with a singular Jacobian says a degenerate
            partner is missing — clipped by the box, most likely — or that two
            distinct modes are closer than ``DEGENERACY_RTOL`` resolves.
        z_lo: complex bottom-left corner of the search rectangle.
        z_hi: complex top-right corner.
        geometry: The scatterer boundary.
        material: The scatterer optical properties.
    """

    wavelengths: np.ndarray
    vectors: np.ndarray
    multiplicity: np.ndarray
    sigma_ratio: np.ndarray
    sv_ratio: np.ndarray
    max_gap: float
    rank: int
    cancellation: float
    edge_margin: np.ndarray
    converged: np.ndarray
    cond_jacobian: np.ndarray
    z_lo: complex
    z_hi: complex
    geometry: Geometry
    material: Material

    @property
    def quality_factors(self) -> np.ndarray:
        """Q = Re λ / (2 Im λ), exactly equal to −Re ω / (2 Im ω).

        Invariant under the vacuum/background rescaling of λ, since that is a
        real positive factor common to numerator and denominator.
        """
        return self.wavelengths.real / (2.0 * self.wavelengths.imag)

    @property
    def size_parameters(self) -> np.ndarray:
        """Complex size parameters x = 2π·n_clad·rad/λ_vac of the modes.

        The coordinate the analytic anchor (:mod:`pysie2d.reference.mie`) works
        in. Derived only — :meth:`QNMSolver.modes` searches rectangles in λ, and
        a box in ``x`` is *not* a box in λ: ``λ = 2π·n_clad·rad/x`` is a Möbius
        map, so it does not carry corners to corners. See :func:`~pysie2d.solver.
        size_parameter`.

        Raises:
            ValueError: If the geometry is not circular.
        """
        return np.array(
            [
                size_parameter(self.geometry, self.material, lam)
                for lam in self.wavelengths
            ]
        )

    @property
    def n_modes(self) -> int:
        """Number of modes found inside the rectangle."""
        return int(self.wavelengths.size)

    def refine(self, *, tol: float = 1.0e-9, max_iter: int = 30) -> "QNMResult":
        """Polish each mode with bordered Newton, returning a new result.

        **This is insurance, not accuracy, and it is opt-in for that reason.**
        On a well-drawn contour the extraction is already converged to ~1e-8 nm
        while the *discretisation* error is 0.38 nm at ``n_pts = 200`` — so
        100 % of the error against the analytic pole is ``n_pts`` and 0 % is
        extraction, and refining changes the answer in the eighth decimal of a
        number that is wrong in the first. What this buys is the recovery path
        for a contour too coarse or too badly placed to locate the pole, and
        the ``converged`` flag. **If a mode is not accurate enough, raise
        ``n_pts``** — refining is tuning the wrong knob.

        Degenerate poles are found, reported, and deliberately **not** refined:
        bordered Newton assumes a simple eigenvalue, so on the doubly
        degenerate ``n ≥ 1`` modes of a circle — the common case, not the
        corner case — the contour estimate is kept and ``cond_jacobian``
        records why. Nothing raises.

        Requires a **non-dispersive** material, inherited from the analytic
        derivative this drives
        (:func:`pysie2d.kernels.assemble_matrix_dwn`).

        Cost is ``max_iter``-bounded but typically one iteration per mode on a
        simple pole, each iteration two assemblies *(measured: 2.05 assemblies,
        the derivative being 1.05× the matrix)*, plus one assembly and SVD per
        mode to bring ``sigma_ratio`` with it.

        Args:
            tol: Convergence threshold on the Newton step size, in nm.
                **Absolute**, which makes it the only scale-dependent
                quantity in the QNM path: everything else obeys the exact
                covariance of ``docs/conventions.md`` §9, so a caller relying
                on ``λ(s·rad) = s·λ(rad)`` should read that section or work
                from the unrefined :meth:`QNMSolver.modes` output.
            max_iter: Iteration cap per mode.

        Returns:
            A new :class:`QNMResult`. ``wavelengths`` carry the polished values
            where refinement succeeded and the contour estimate everywhere
            else; ``converged`` and ``cond_jacobian`` are filled in;
            ``multiplicity``, ``sigma_ratio`` and ``edge_margin`` are
            recomputed at the new wavelengths so no diagnostic describes a
            wavelength that is no longer in the result. ``vectors`` are
            polished alongside their wavelengths — the bordered system solves
            for the pair jointly — and renormalised to unit columns in the
            gauge the contour estimate set. Degenerate modes keep their contour
            vectors untouched, along with their wavelengths. Unit-norm is still
            not a *mode* normalisation: a QNM norm remains out of scope (D4).
        """
        bie = BIESolver(self.geometry, self.material)
        lams = self.wavelengths.copy()
        vectors = self.vectors.copy()
        converged = np.zeros(self.n_modes, dtype=bool)
        cond_jacobian = np.full(self.n_modes, np.nan)

        for k in range(self.n_modes):
            polished = newton_refine(
                bie.assemble,
                bie.assemble_derivative,
                self.wavelengths[k],
                self.vectors[:, k],
                self.z_lo,
                self.z_hi,
                tol=tol,
                max_iter=max_iter,
            )
            cond_jacobian[k] = polished.cond_jacobian
            # newton_refine reports the conditioning but does not act on it —
            # it will happily take a step on a singular bordered system, and
            # the line search only guarantees the residual fell, not that the
            # step meant anything. Discarding it here is what makes "flagged,
            # skipped, never raises" true.
            if polished.cond_jacobian > DEGENERATE_COND:
                continue
            lams[k] = polished.eigenvalue
            converged[k] = polished.converged
            # newton_refine normalises to its anchor, conj(v0)·v = 1, while
            # these columns are unit. Rescaling by a positive real does not
            # touch the phase, so the polished column keeps the gauge the
            # contour estimate set and the two stay comparable.
            #
            # The division is a no-op on every path reachable today: beyn_modes
            # emits unit columns, which makes the anchor scale 1. It is kept
            # because "unit columns" is this class's documented contract and
            # the alternative is an unstated dependency on two other modules'
            # conventions agreeing. Not covered by a test for the same reason
            # it is a no-op — test_beyn's scaled-input case is what shows the
            # two normalisations are genuinely different constraints.
            vectors[:, k] = polished.vector / np.linalg.norm(polished.vector)

        return replace(
            self,
            wavelengths=lams,
            vectors=vectors,
            multiplicity=_multiplicity(lams),
            sigma_ratio=np.array([_sigma_ratio(bie, lam) for lam in lams]),
            edge_margin=_edge_margin(lams, self.z_lo, self.z_hi),
            converged=converged,
            cond_jacobian=cond_jacobian,
        )

    def sensitivity(
        self,
        at: Callable[[float], tuple[Geometry, Material]],
        *,
        step: float = SHAPE_STEP,
    ) -> np.ndarray:
        """Adjoint derivative dλ/dp of every mode with respect to one parameter.

        The identity (spec §5.1) for a **simple** pole of M(λ):

            ``dλ/dp = − uᴴ (∂M/∂p) v / [ uᴴ (∂M/∂λ) v ]``

        with ``v`` the right and ``u`` the left null vector of M(λ)
        (:func:`_null_vectors` — ``u`` is not ``conj(v)`` here, M is not
        complex-symmetric), ``∂M/∂λ`` analytic
        (:meth:`~pysie2d.solver.BIESolver.assemble_derivative`) and ``∂M/∂p``
        one central difference. Two extra assemblies and one SVD per mode; no
        eigenvalue is re-extracted, which is the point of the adjoint.

        **Frozen nodes are enforced, not assumed** (``docs/conventions.md``
        §10). Both perturbed geometries must carry *exactly* the θ of the base
        geometry, and a mismatch raises. Letting the arc-length inversion
        re-place nodes between ``p₀−h`` and ``p₀+h`` differentiates the
        parametrisation gauge along with the physics and puts an O(h) term into
        the quotient that grows with ``n_pts``.

        Simple poles only. A degenerate pole has a two-dimensional null space,
        the quotient above picks an arbitrary vector out of it, and the answer
        is meaningless rather than merely inaccurate — so it raises instead.

        Args:
            at: ``δ → (geometry, material)`` at parameter offset ``δ`` from the
                base point, in the parameter's own units, with ``δ = 0`` the
                base point itself. Returning both halves is what lets one
                signature cover shape parameters and ``n_core``/``n_clad``
                alike. The geometry must be built on ``self.geometry.theta``.
            step: Central-difference step ``h``, in the parameter's own units.
                Defaults to :data:`SHAPE_STEP`.

        Returns:
            complex (K,) ``dλ/dp`` in nm per unit of ``p``, one per mode, in
            the order of :attr:`wavelengths`.

        Raises:
            ValueError: If any mode is degenerate, or if a perturbed geometry
                does not carry the base node set.
        """
        if np.any(self.multiplicity > 1):
            raise ValueError(
                "sensitivity() is the simple-pole quotient; modes "
                f"{np.flatnonzero(self.multiplicity > 1).tolist()} are "
                "degenerate and need the secular branch"
            )

        base = BIESolver(self.geometry, self.material)
        plus = self._perturbed_solver(at, +step)
        minus = self._perturbed_solver(at, -step)

        out = np.empty(self.n_modes, dtype=complex)
        for k, lam in enumerate(self.wavelengths):
            # ∂M/∂p is a function of λ as much as of p, so the difference is
            # taken at *this* mode's wavelength; a single ∂M/∂p shared across
            # modes would be the derivative at the wrong point for all but one.
            dm_dp = (plus.assemble(lam) - minus.assemble(lam)) / (2.0 * step)
            u, v = _null_vectors(base, lam)
            dm_dlam = base.assemble_derivative(lam)
            out[k] = -(u.conj() @ dm_dp @ v) / (u.conj() @ dm_dlam @ v)
        return out

    def _perturbed_solver(
        self,
        at: Callable[[float], tuple[Geometry, Material]],
        delta: float,
    ) -> BIESolver:
        """Solver at one displaced parameter value, on the frozen node set.

        The node-set check is here rather than in the caller because it is the
        one precondition a user cannot see the violation of: with re-inverted
        nodes every number downstream still looks plausible
        (``docs/conventions.md`` §10).
        """
        geom, mat = at(delta)
        if geom.theta.shape != self.geometry.theta.shape or not np.array_equal(
            geom.theta, self.geometry.theta
        ):
            raise ValueError(
                "perturbed geometry must carry the base node set exactly "
                "(docs/conventions.md §10); build it with "
                "Geometry.gielis(..., theta=result.geometry.theta)"
            )
        return BIESolver(geom, mat)


class QNMSolver:
    """Quasi-normal-mode solver for a single particle in a homogeneous background.

    Attributes:
        geometry: Discretized particle boundary.
        material: Optical properties of the scatterer.

    Examples:
        >>> geom = Geometry.gielis(rad=200, n_pts=200, m=0)
        >>> mat = Material(n_core=3.0, n_clad=1.0, pol=2)
        >>> qnm = QNMSolver(geom, mat)
        >>> res = qnm.modes(z_lo=520 + 15j, z_hi=545 + 40j)
        >>> res.wavelengths
        >>> res.quality_factors
    """

    def __init__(self, geometry: Geometry, material: Material) -> None:
        """Compose a geometry and a material into a reusable mode solver."""
        self.geometry = geometry
        self.material = material
        # Deliberately the driven solver's own assembly rather than a parallel
        # path: a QNM is a singularity of the *scattering* operator, and that
        # claim only holds if the two are literally the same matrix.
        self._bie = BIESolver(geometry, material)

    def modes(
        self,
        z_lo: complex,
        z_hi: complex,
        *,
        n_quad_per_side: int = 12,
        n_probe: int = 12,
        rank_tol: float = 1.0e-8,
        rng_seed: int = 0,
    ) -> QNMResult:
        """Find every quasi-normal mode inside a rectangle of the λ-plane.

        Cost is ``4·n_quad_per_side`` complex assemblies and LU factorisations,
        independent of how many modes are inside.

        Args:
            z_lo: Bottom-left corner of the search rectangle, **vacuum** nm.
            z_hi: Top-right corner, **vacuum** nm.
            n_quad_per_side: Gauss-Legendre nodes per contour edge. The
                default resolves the contour integral far below the
                discretisation error in ``n_pts`` *(measured: identical modes
                to 1e-8 nm from 6 nodes per side upward, against a 0.38 nm
                discretisation error at n_pts = 200)*.
            n_probe: Probe columns; must exceed the number of modes inside,
                counting rank leaked from poles just outside.
            rank_tol: Relative singular-value floor for rank detection.
            rng_seed: Probe-matrix seed. Results must not depend on it.

        Returns:
            The modes inside the rectangle, with their diagnostics.

        Raises:
            ValueError: If the rectangle is not strictly inside the physical
                quadrant ``Re λ > 0``, ``Im λ > 0``, or if the modes inside
                outnumber ``n_probe``.
        """
        self._validate_region(z_lo, z_hi)

        found = beyn_modes(
            self._bie.assemble,
            z_lo,
            z_hi,
            n_quad_per_side=n_quad_per_side,
            n_probe=n_probe,
            rank_tol=rank_tol,
            rng_seed=rng_seed,
        )
        lams = found.eigenvalues

        return QNMResult(
            wavelengths=lams,
            vectors=found.vectors,
            multiplicity=_multiplicity(lams),
            sigma_ratio=np.array([_sigma_ratio(self._bie, lam) for lam in lams]),
            sv_ratio=found.sv_ratio,
            max_gap=found.max_gap,
            rank=found.rank,
            cancellation=found.cancellation,
            edge_margin=_edge_margin(lams, z_lo, z_hi),
            # No refinement has been attempted: all-False plus all-NaN is the
            # unambiguous reading of "not tried", and neither is a verdict on
            # the modes. QNMResult.refine fills both in.
            converged=np.zeros(lams.size, dtype=bool),
            cond_jacobian=np.full(lams.size, np.nan),
            z_lo=z_lo,
            z_hi=z_hi,
            geometry=self.geometry,
            material=self.material,
        )

    def _sigma_ratio(self, wavelength: complex) -> float:
        """σ_min/σ_max of M at one **vacuum** wavelength, on this solver.

        Bound method over :func:`_sigma_ratio` so the singularity measure has
        one implementation while both the solver and
        :meth:`QNMResult.refine` — which has a result, not a solver — can reach
        it.
        """
        return _sigma_ratio(self._bie, wavelength)

    @staticmethod
    def _validate_region(z_lo: complex, z_hi: complex) -> None:
        """Reject rectangles outside the physical quadrant.

        ``Im λ > 0`` is the decaying half-plane under ``exp(-iωt)``; a box
        below it searches for growing modes. ``Re λ > 0`` keeps every Hankel
        argument off the ``H^(1)`` branch cut on the negative real axis, which
        is what makes M(λ) holomorphic on the rectangle — and holomorphy is the
        premise of the whole contour argument, not a detail.
        """
        if z_hi.real <= z_lo.real or z_hi.imag <= z_lo.imag:
            raise ValueError(
                f"z_hi must be strictly above and right of z_lo; "
                f"got z_lo={z_lo}, z_hi={z_hi}"
            )
        # Exception text is ASCII on purpose: it may be printed to a cp1252
        # console, where a bare "λ" raises UnicodeEncodeError and hides the
        # actual error. Docstrings keep the Unicode notation.
        if z_lo.real <= 0.0:
            raise ValueError(
                f"search region must lie in Re(lambda) > 0 to keep Hankel "
                f"arguments off the H^(1) branch cut; got Re z_lo = {z_lo.real}"
            )
        if z_lo.imag <= 0.0:
            raise ValueError(
                f"search region must lie in Im(lambda) > 0: under exp(-i.omega.t) "
                f"a decaying mode has Im(lambda) > 0, so a box reaching "
                f"Im(lambda) = {z_lo.imag} is searching for growing modes"
            )


def _sigma_ratio(bie: BIESolver, wavelength: complex) -> float:
    """σ_min/σ_max of M at one **vacuum** wavelength.

    A full SVD of the 2·n_pts matrix, run once per mode. Module-level rather
    than a QNMSolver method because :meth:`QNMResult.refine` needs the same
    number at the moved wavelengths, and a diagnostic with two implementations
    is a diagnostic that will eventually disagree with itself.

    Inverse iteration on a factorisation would be ~100× cheaper (spec §6.3);
    with a handful of modes the full SVD is not the cost that matters.
    """
    s = np.linalg.svd(bie.assemble(wavelength), compute_uv=False)
    return float(s[-1] / s[0])


def _multiplicity(lams: np.ndarray) -> np.ndarray:
    """Count how many entries share each wavelength.

    Degenerate partners are reported separately rather than collapsed: the ±n
    pair carries two independent mode vectors, and merging them would make the
    mode count disagree with the analytic one.
    """
    if lams.size == 0:
        return np.zeros(0, dtype=int)
    close = np.abs(lams[:, None] - lams[None, :]) <= DEGENERACY_RTOL * np.abs(
        lams[None, :]
    )
    return close.sum(axis=1).astype(int)


def _edge_margin(lams: np.ndarray, z_lo: complex, z_hi: complex) -> np.ndarray:
    """Distance to the nearest contour edge, as a fraction of the shorter side."""
    if lams.size == 0:
        return np.zeros(0, dtype=float)
    side = min(z_hi.real - z_lo.real, z_hi.imag - z_lo.imag)
    dist = np.minimum.reduce(
        [
            lams.real - z_lo.real,
            z_hi.real - lams.real,
            lams.imag - z_lo.imag,
            z_hi.imag - lams.imag,
        ]
    )
    return dist / side


def _null_vectors(bie: BIESolver, wavelength: complex) -> tuple[np.ndarray, np.ndarray]:
    """Left and right null vectors of M at one **vacuum** wavelength.

    The adjoint quotient ``dλ/dp = −uᴴ(∂M/∂p)v / uᴴ(∂M/∂λ)v`` needs both. The
    right vector ``v`` is the one :attr:`QNMResult.vectors` already carries; the
    left vector ``u`` is **not** obtainable from it, because M is not
    complex-symmetric — each of the four ``n_pts`` blocks is symmetric to 1e-14
    but the off-diagonal blocks are not transposes of one another
    *(measured: ‖M − Mᵀ‖/‖M‖ = 1.05)*. Assuming ``u = v̄`` would give a quotient
    that is wrong by an O(1) factor and would still look plausible.

    From ``M = U Σ Vᴴ`` at a pole, the smallest singular triplet is the null
    pair: ``v = V[:, -1]`` satisfies ``Mv = σ_min u`` and ``u = U[:, -1]``
    satisfies ``uᴴM = σ_min vᴴ``. Both residuals are therefore ``σ_min``
    exactly, and ``σ_min/σ_max`` — the number :attr:`QNMResult.sigma_ratio`
    already reports — is how singular the pole actually came out.

    Costs one assembly plus one full SVD with vectors, i.e. what
    :func:`_sigma_ratio` already spends plus the vector accumulation.

    Args:
        bie: Solver carrying the geometry and material.
        wavelength: Vacuum wavelength (nm), complex at a mode.

    Returns:
        ``(u, v)``, each complex ``(2·n_pts,)`` with unit norm.
    """
    u, _, vh = np.linalg.svd(bie.assemble(wavelength))
    return u[:, -1], vh[-1].conj()
