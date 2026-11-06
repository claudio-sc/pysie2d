"""Clusters of non-overlapping particles sharing one background.

Degree-of-freedom layout (conventions §14, D4): particle ``p`` occupies the
contiguous slice ``[o_p : o_p + 2·nn_p]`` of the coupled unknown vector, with
``o_p = Σ_{q<p} 2·nn_q``; within that slice the first ``nn_p`` entries are φ_p
and the next ``nn_p`` are χ_p. Contiguity is what lets each particle's
sub-vector go straight into the existing single-particle primitives unchanged.

The background wavenumber ``k_bg = 2π·n_clad/λ_vac`` is **common to the whole
cluster** (D3b): a cluster sits in exactly one background, so the per-particle
materials may differ in ``nc``/``eps`` (which are background-relative,
conventions §2) but not in ``n_clad``.

Overlapping boundaries are rejected rather than approximated. The coupled
boundary-integral formulation represents the field in the region exterior to
every particle and interior to exactly one; where two boundaries cross, that
partition does not exist and the unknowns φ, χ have no meaning. This is a limit
of the formulation, not of the implementation, so no amount of resolution
repairs it.
"""

import warnings
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from scipy.linalg import lu_factor, lu_solve

from . import layered
from .background import HalfSpace, Stack, _effective, _top
from .beyn import beyn_modes, newton_refine
from .fields import (
    _background_field,
    _check_far_field_below,
    _cross_sections,
    _downward_cross_section,
    _downward_far_field,
    _far_field_at,
    _representation_at,
    _upward_cross_sections,
    _upward_far_field,
)
from .geometry import Geometry
from .kernels import (
    _interior_rows,
    _real_if_real,
    assemble_cross_block,
    assemble_matrix,
)
from .layered import GuidedMode
from .material import Material
from .solver import wavelength_over_ds
from .sources import (
    _in_plane_dipole_rhs,
    _point_inside,
    in_plane_moment_b,
    line_dipole_rhs,
    plane_wave_rhs,
)
from .waveguide import _mode_rhs, _resolve_mode, _StackResult

PI = np.pi

# A resolution spread wider than this between the best- and worst-resolved
# particle warns. The coupled system is only as accurate as its worst block,
# and that is not visible from any one particle. Four is two halvings of nn,
# which in the M2 pilot is the difference between round-off and 1e-6 at a
# fixed gap (docs/design/studies/cluster-gap-envelope.md).
RESOLUTION_SPREAD_WARN = 4.0

# Nodes needed before the near-gap integrand's analyticity strip stops being
# resolved. For a circle of radius a, a near-singularity at distance `gap` sits
# at conformal half-width ~ sqrt(2·gap/a), so the trapezoid rule's rate
# exp(-nn·sqrt(2·gap/a)) fixes the form nn >= C·sqrt(a/gap). Measured by BIE
# self-convergence in docs/design/studies/cluster-gap-envelope.md over
# x = 1.0, 1.79, 3.5 in both polarisations: the smallest nn reaching round-off
# is the *same* at every x and pol once gap/a <= 0.83, which is what licenses a
# single constant. Measured nn·sqrt(gap/a) spans 41-101 on a sqrt(2) nn grid,
# worst at the tightest gap tested (gap/a = 0.032) — the exponential-rate
# estimate behind nn ~ C·sqrt(a/gap) is a leading-order approximation, and the
# strict 1e-13 criterion used here needs relatively more nodes right at the
# near-singularity than the pilot's looser read suggested. 100 covers the
# worst measured point rather than the mid-range typical value: a guard that
# silently under-warns exactly where degradation is worst is worse than one
# that warns early in the common case (docs/design/studies/
# cluster-gap-envelope.md). Validated over gap/a in [0.032, 5.06]; below that
# range this is extrapolation, not measurement, and the guard should be
# treated as diagnostic rather than a hard bound.
GAP_ENVELOPE_C = 100.0

# Singular values kept when compressing the neighbour coupling, relative to the
# largest. Truncating harder is false economy: 1e-4 gains 1.3× per permittivity
# and costs nine digits (docs/design/studies/cluster-reduction/README.md).
COUPLING_RANK_TOL = 1.0e-13

# x = a − R·y cancels when an isolated particle is near resonance and the cluster is
# not, by up to 5e-5 at |ε − ε_iso| = 1e-12. No a-priori ratio detects it: ‖a‖/‖x‖ stays
# at 0.26 for a plane wave that does not drive the resonance, so every solve refines
# once and reads the error off the step. An ordinary solve's first step is ≲ 1e-12 and
# ends there; 5e-5 takes three to reach 4e-13, the comparison's floor
# (docs/design/studies/cluster-reduction/README.md).
MAX_REFINE_STEPS = 3
REFINE_CONVERGED = 1.0e-12
REFINE_WARN = 1.0e-10

# Step for dD/dε. Applied to the interior rows, which are smooth in ε: a difference
# of G itself would straddle G's isolated-particle poles, which at high Q lie far
# closer than this.
INTERIOR_DERIVATIVE_STEP = 1.0e-6


def _upper_branch(eps: complex) -> complex:
    """Return ``√ε`` with its cut on ``[0, ∞)``, the branch modal contours need.

    The principal root makes ``M(ε)`` singular at ``H_n^{(1)}(√ε·x) = 0``, where no
    physical mode exists. This one keeps ``Im √ε > 0`` off the cut, so ``H^{(1)}``
    has no zeros there. The two coincide for Im ε > 0 (conventions §18).

    Args:
        eps: Background-relative permittivity, off the cut.

    Returns:
        ``i·√(−ε)``.
    """
    return 1j * np.sqrt(-complex(eps))


class ClusterOverlapError(ValueError):
    """Two particle boundaries intersect; the formulation does not apply."""


class ClusterGapWarning(UserWarning):
    """A gap is too small for the quadrature resolution in use."""


class ClusterResolutionWarning(UserWarning):
    """One particle is far more coarsely resolved than its neighbours."""


class ClusterCancellationWarning(UserWarning):
    """A fast solve lost accuracy to a particle resonance, despite refining."""


def _check_materials(materials: Sequence[Material], n_particles: int, pol: int) -> None:
    """Validate one material set against a cluster.

    Args:
        materials: One material per particle.
        n_particles: Number of particles in the cluster.
        pol: The cluster's polarisation.

    Raises:
        ValueError: If the number of materials does not match the number of
            particles, if any material's ``pol`` differs from ``pol``, or if the
            materials disagree on ``n_clad``.
    """
    if len(materials) != n_particles:
        raise ValueError(f"{len(materials)} materials for {n_particles} particles")
    for i, mat in enumerate(materials):
        if mat.pol != pol:
            raise ValueError(
                f"material {i} has pol = {mat.pol}, but the cluster is "
                f"being solved at pol = {pol}. Polarisation is a property "
                f"of the problem, not of a particle (conventions sec. 1)."
            )
        if mat.n_clad != materials[0].n_clad:
            raise ValueError(
                f"material {i} has n_clad = {mat.n_clad}, material 0 has "
                f"{materials[0].n_clad}. A cluster has one background, and "
                f"k_bg = 2*pi*n_clad/lambda_vac must be unambiguous. "
                f"Per-particle optical properties go in n_core and epsi, "
                f"which enter as the background-relative nc and eps "
                f"(conventions sec. 2)."
            )


class Cluster:
    """An arrangement of non-overlapping particle boundaries.

    Geometry only: materials live on the solver (D2), so one arrangement can be
    swept over material sets without being rebuilt. Construction validates that
    no two boundaries intersect and records the smallest node-to-node gap, both
    of which are properties of the arrangement alone.

    Attributes:
        geometries: The particle boundaries, in DOF order.
        offsets: ``len(self) + 1`` cumulative DOF offsets, ``offsets[p] = o_p``.
        n_dof: Total number of unknowns ``N = 2·Σ_p nn_p``.
        min_gap: Smallest node-to-node distance between distinct particles
            (nm); ``inf`` for a single-particle cluster.
    """

    def __init__(self, geometries: Sequence[Geometry]) -> None:
        """Build a cluster from a sequence of boundaries.

        Args:
            geometries: One or more :class:`~pysie2d.geometry.Geometry`
                boundaries. Resolutions ``n_pts`` may differ between them.

        Raises:
            ValueError: If ``geometries`` is empty.
            ClusterOverlapError: If two boundaries intersect.
        """
        if len(geometries) == 0:
            raise ValueError("a Cluster needs at least one geometry")
        self.geometries = tuple(geometries)
        sizes = [2 * g.n_pts for g in self.geometries]
        self.offsets = tuple(np.cumsum([0] + sizes).tolist())
        self.n_dof = self.offsets[-1]
        self._check_overlap()
        self.min_gap = self._min_gap()

    def __len__(self) -> int:
        """Return the number of particles in the cluster."""
        return len(self.geometries)

    def slice(self, p: int) -> slice:
        """Return the ``2·nn_p`` slice of particle ``p`` (conventions §14).

        Args:
            p: Particle index.

        Returns:
            The slice ``[o_p : o_p + 2·nn_p]`` into a coupled vector of length
            ``n_dof``; its first half is φ_p and its second half χ_p.
        """
        return slice(self.offsets[p], self.offsets[p + 1])

    def _radii(self, p: int) -> tuple[float, float]:
        """Return the inscribed and circumscribed node radii of particle ``p``.

        Both are measured from the stored centre over the boundary *nodes*, not
        taken from ``Geometry.rad``: on a rounded square the superformula scale
        is 2.29× smaller than the true circumscribing radius, the same trap
        ``ScatterResult.multipoles`` guards against, and an overlap test built
        on it would accept intersecting boundaries.

        Args:
            p: Particle index.

        Returns:
            ``(r_insc, r_circ)`` in nm — the min and max node radius.
        """
        g = self.geometries[p]
        r = np.hypot(g.f - g.x0, g.g - g.z0)
        return float(r.min()), float(r.max())

    def _check_overlap(self) -> None:
        """Reject any pair of intersecting boundaries.

        Three bands per pair, cheapest first: separated circumscribing circles
        certainly do not intersect; overlapping inscribed circles certainly do;
        in between, every node of each boundary is ray-cast against the other.

        The test is a strong filter, not a decision procedure. **Two boundaries
        can cross in a lens-shaped sliver that contains no node of either**, and
        band 3 misses exactly that case. The arrangement's ``min_gap`` is then
        tiny, so the resolution-versus-gap envelope check (§4.9) warns — that
        warning is the backstop for this residual failure mode.

        Raises:
            ClusterOverlapError: If a pair is found to intersect.
        """
        for p in range(len(self)):
            gp = self.geometries[p]
            rp_in, rp_out = self._radii(p)
            for q in range(p + 1, len(self)):
                gq = self.geometries[q]
                rq_in, rq_out = self._radii(q)
                d = float(np.hypot(gq.x0 - gp.x0, gq.z0 - gp.z0))
                if rp_out + rq_out < d:
                    continue  # certainly disjoint
                if rp_in + rq_in > d:  # certainly overlapping
                    raise ClusterOverlapError(
                        f"particles {p} and {q} overlap: their inscribed "
                        f"circles ({rp_in:.4g} + {rq_in:.4g} nm) already "
                        f"exceed the centre separation {d:.4g} nm"
                    )
                hit = any(
                    _point_inside(x, z, gp.f, gp.g)
                    for x, z in zip(gq.f, gq.g, strict=True)
                ) or any(
                    _point_inside(x, z, gq.f, gq.g)
                    for x, z in zip(gp.f, gp.g, strict=True)
                )
                if hit:
                    raise ClusterOverlapError(
                        f"particles {p} and {q} overlap: a boundary node of "
                        f"one lies inside the other"
                    )

    def _min_gap(self) -> float:
        """Return the smallest node-to-node distance between particles (nm).

        This slightly **overestimates** the true surface-to-surface gap: the
        closest approach of two smooth boundaries generally falls between
        nodes, so the discrete minimum is an upper bound that tightens as
        ``n_pts`` grows. It is used as a diagnostic scale, where an error of
        order the node spacing is immaterial.

        Cost is ``O((Σ nn)²)`` — 9e6 distances at ``Np = 10``, ``nn = 300``,
        milliseconds, far below one assembly — so it is computed once in
        ``__init__``.

        Returns:
            The minimum distance in nm, or ``inf`` for a single particle.
        """
        best = np.inf
        for p in range(len(self)):
            gp = self.geometries[p]
            for q in range(p + 1, len(self)):
                gq = self.geometries[q]
                d = np.hypot(
                    gq.f[:, None] - gp.f[None, :], gq.g[:, None] - gp.g[None, :]
                )
                best = min(best, float(d.min()))
        return best


class ClusterBIESolver:
    """Coupled boundary-integral solver for a finite cluster of particles.

    The diagonal blocks are the single-particle matrix, :func:`~pysie2d.kernels
    .assemble_matrix`, called verbatim at each particle's own ``nn_p``, ``nc_p``
    and ``eps_p``; the off-diagonal blocks couple particles through the
    background Green function alone and so have a zero lower half
    (docs/design/multiparticle-spec.md §3.1). The whole v0.6 Kress–Martensen
    validation therefore carries over untouched.

    ``pol`` is an argument of the *problem*, not of a particle (conventions
    §1), and the background index ``n_clad`` is a property of the *cluster*
    (D3b): both are validated equal across the materials, because a
    disagreement in either is a meaningless coupled system that nothing else
    would notice. The background-relative ``nc`` and ``eps`` (conventions §2)
    may differ freely between particles.

    There is deliberately no public ``assemble`` (D12): the only caller for one
    would be a cluster quasi-normal-mode search, and the half-plane argument of
    conventions §8 has not been re-examined for a cluster.

    Attributes:
        cluster: The arrangement being solved.
        materials: Per-particle optical properties, in DOF order.
        pol: Polarisation: 1 = TM (H_y), 2 = TE (E_y).
        background: A :class:`~pysie2d.background.HalfSpace` or
            :class:`~pysie2d.background.Stack` the whole cluster
            sits above, or ``None`` (default) for the homogeneous cladding,
            bit-identical to a solver without the argument. Every particle pair
            is coupled through the substrate, including a particle with itself.
    """

    def __init__(
        self,
        cluster: Cluster,
        materials: Sequence[Material],
        pol: int = 2,
        background: HalfSpace | Stack | None = None,
    ) -> None:
        """Bind materials and a polarisation to an arrangement.

        Args:
            cluster: The particle arrangement.
            materials: One :class:`~pysie2d.material.Material` per particle, in
                the cluster's own order.
            pol: Polarisation: 1 = TM, 2 = TE.
            background: A half-space substrate below the cluster, or ``None``.

        Raises:
            ValueError: If the number of materials does not match the number of
                particles, if any material's ``pol`` differs from ``pol``, or
                if the materials disagree on ``n_clad``; with a background, a
                boundary node at or below the interface.

        Warns:
            InterfaceGapWarning: With a background, if a boundary is too coarse
                for its distance to the interface.
        """
        _check_materials(materials, len(cluster), pol)
        self.cluster = cluster
        self.materials = tuple(materials)
        self.pol = pol
        self.background = background
        if background is not None:
            layered.check_interface_gap(cluster.geometries, _top(background), pol)

    def at_wavelength(self, wavelength: float | complex) -> "ClusterEigenEps":
        """Prepare fast solves and eigenpermittivities at one wavelength.

        Use it when many materials are solved at one wavelength: see
        :class:`ClusterEigenEps`. A sweep in wavelength rebuilds everything at each
        step, so :meth:`scatter` remains the tool for that, except at very large
        clusters.

        Args:
            wavelength: Vacuum wavelength λ_vac in nm. May be complex for driven
                solves; eigenpermittivities need it real.

        Returns:
            The prepared :class:`ClusterEigenEps`.

        Warns:
            ClusterGapWarning: If the closest gap falls below the quadrature
                envelope of ``GAP_ENVELOPE_C``.
        """
        self._check_gap(wavelength)
        return ClusterEigenEps(self, wavelength)

    def _reflection(
        self,
        wavelength: float | complex,
        f: np.ndarray | None = None,
        g: np.ndarray | None = None,
    ) -> layered.Reflection | layered.StackReflection | None:
        """Reflected-field context for the whole cluster, or ``None`` without one.

        One path serves every particle pair, so it is sized for the **cluster's**
        horizontal extent, not any one particle's (holomorphy note §7).

        Args:
            wavelength: Vacuum wavelength λ_vac in nm.
            f: x coordinates of every point to serve; every node by default.
            g: Their z coordinates.
        """
        mat = self.materials[0]
        bg = _effective(self.background, mat.n_clad, wavelength)
        if bg is None:
            return None
        geoms = self.cluster.geometries
        if isinstance(bg, Stack):
            return layered.StackReflection.build(
                self.pol,
                mat.wnum_bg(wavelength),
                bg.eps_rel(mat.n_clad, wavelength),
                np.concatenate([gp.f for gp in geoms]) if f is None else f,
                np.concatenate([gp.g for gp in geoms]) if g is None else g,
            )
        return layered.Reflection.build(
            self.pol,
            mat.wnum_bg(wavelength),
            bg.eps_rel(mat.n_clad, wavelength),
            bg.z_int,
            np.concatenate([gp.f for gp in geoms]) if f is None else f,
            np.concatenate([gp.g for gp in geoms]) if g is None else g,
        )

    def _assemble(self, wavelength: float | complex) -> np.ndarray:
        """Build the coupled system matrix at one wavelength.

        Args:
            wavelength: Vacuum wavelength λ_vac in nm. May be complex.

        Returns:
            complex ``(n_dof, n_dof)`` matrix in the layout of conventions §14.
        """
        cl = self.cluster
        k_bg = _real_if_real(self.materials[0].wnum_bg(wavelength))
        me = np.zeros((cl.n_dof, cl.n_dof), dtype=complex)

        # Each task writes a disjoint block, so this loop is embarrassingly
        # parallel, and hank0/hank1 both release the GIL — threading it is the
        # proven 5.02x pattern of contour_moments (docs/design/performance.md
        # §3.1). Serial until the Np ceiling study says otherwise (D8); the
        # task list is the seam that keeps that a one-line change.
        tasks = [("self", p, p) for p in range(len(cl))]
        tasks += [
            ("cross", q, p) for q in range(len(cl)) for p in range(len(cl)) if q != p
        ]

        for kind, q, p in tasks:
            gp = cl.geometries[p]
            if kind == "self":
                mat = self.materials[p]
                me[cl.slice(p), cl.slice(p)] = assemble_matrix(
                    self.pol,
                    gp.n_pts,
                    gp.f,
                    gp.g,
                    gp.df,
                    gp.dg,
                    gp.ddf,
                    gp.ddg,
                    k_bg,
                    mat.nc,
                    mat.eps,
                )
                continue
            gq = cl.geometries[q]
            m1, m2 = assemble_cross_block(
                k_bg, gp.n_pts, gp.f, gp.g, gp.df, gp.dg, gq.f, gq.g
            )
            r0 = cl.offsets[q]
            c0 = cl.offsets[p]
            me[r0 : r0 + gq.n_pts, c0 : c0 + gp.n_pts] = m1
            me[r0 : r0 + gq.n_pts, c0 + gp.n_pts : c0 + 2 * gp.n_pts] = m2
            # The lower half of a cross block stays at its initialised zero:
            # that is the M3/M4 result of §3.1, not an omission. Particle p's
            # interior Green function is confined to p's own volume, so
            # coupling enters only through the exterior background kernel.

        refl = self._reflection(wavelength)
        if refl is not None:
            # The substrate couples every pair, q = p included, and only in the
            # exterior rows, like the free-space cross blocks.
            for q in range(len(cl)):
                for p in range(len(cl)):
                    gq, gp = cl.geometries[q], cl.geometries[p]
                    m1, m2 = refl.blocks(gq, gp)
                    r0, c0 = cl.offsets[q], cl.offsets[p]
                    me[r0 : r0 + gq.n_pts, c0 : c0 + gp.n_pts] += m1
                    me[r0 : r0 + gq.n_pts, c0 + gp.n_pts : c0 + 2 * gp.n_pts] += m2
        return me

    def _exterior_rows(
        self,
        q: int,
        wavelength: float | complex,
        refl: layered.Reflection | layered.StackReflection | None,
    ) -> np.ndarray:
        """Build one particle's exterior rows of the coupled matrix.

        These rows do not depend on any particle's material, so they are all a
        fixed-wavelength sweep needs from the assembly, and building them one
        particle at a time is what keeps the dense matrix out of memory.

        Args:
            q: The field particle whose rows are built.
            wavelength: Vacuum wavelength λ_vac in nm. May be complex.
            refl: The reflected-field context of :meth:`_reflection`.

        Returns:
            complex ``(nn_q, n_dof)``: ``_assemble(wavelength)``'s rows of ``q``.
        """
        cl = self.cluster
        k_bg = _real_if_real(self.materials[0].wnum_bg(wavelength))
        gq = cl.geometries[q]
        rows = np.zeros((gq.n_pts, cl.n_dof), dtype=complex)
        for p, gp in enumerate(cl.geometries):
            c0 = cl.offsets[p]
            if p == q:
                # ri = kd = 1: the upper half of assemble_matrix sees neither.
                rows[:, cl.slice(p)] = assemble_matrix(
                    self.pol,
                    gp.n_pts,
                    gp.f,
                    gp.g,
                    gp.df,
                    gp.dg,
                    gp.ddf,
                    gp.ddg,
                    k_bg,
                    1.0,
                    1.0,
                )[: gp.n_pts]
            else:
                m1, m2 = assemble_cross_block(
                    k_bg, gp.n_pts, gp.f, gp.g, gp.df, gp.dg, gq.f, gq.g
                )
                rows[:, c0 : c0 + gp.n_pts] = m1
                rows[:, c0 + gp.n_pts : c0 + 2 * gp.n_pts] = m2
            if refl is not None:
                m1, m2 = refl.blocks(gq, gp)
                rows[:, c0 : c0 + gp.n_pts] += m1
                rows[:, c0 + gp.n_pts : c0 + 2 * gp.n_pts] += m2
        return rows

    def _rhs_plane_wave(
        self,
        wavelength: float,
        angle: float,
        refl: layered.Reflection | layered.StackReflection | None = None,
    ) -> np.ndarray:
        """Right-hand side of a plane wave, in the layout of conventions §14.

        Args:
            wavelength: Vacuum wavelength λ_vac in nm.
            angle: Incidence angle in degrees; 0 propagates along −z.
            refl: A reflected-field context already built for every node, because
                building one is most of a slab solve's cost per material; by default
                it is built here.

        Returns:
            complex ``(n_dof,)`` right-hand side.
        """
        cl = self.cluster
        k_bg = self.materials[0].wnum_bg(wavelength)
        rhs = np.zeros(cl.n_dof, dtype=complex)
        for p, gp in enumerate(cl.geometries):
            rhs[cl.slice(p)] = plane_wave_rhs(gp.n_pts, angle, k_bg, gp.f, gp.g)
        if refl is None:
            refl = self._reflection(wavelength)
        if refl is not None:
            for p, gp in enumerate(cl.geometries):
                rhs[cl.offsets[p] : cl.offsets[p] + gp.n_pts] += refl.plane_wave(
                    angle, gp.f, gp.g
                )
        return rhs

    def _rhs_dipole(
        self,
        wavelength: float,
        x_s: float,
        z_s: float,
        moment: tuple[complex, complex] | None,
    ) -> np.ndarray:
        """Right-hand side of a line dipole, in the layout of conventions §14.

        Args:
            wavelength: Vacuum wavelength λ_vac in nm.
            x_s: Source x-coordinate (nm).
            z_s: Source z-coordinate (nm).
            moment: ``(p_x, p_z)`` of an in-plane electric dipole, TM only.

        Returns:
            complex ``(n_dof,)`` right-hand side.

        Raises:
            ValueError: If the source lies inside, or too close to, any
                particle — raised by :func:`~pysie2d.sources.line_dipole_rhs`;
                a ``moment`` at ``pol = 2`` (T7).
        """
        if moment is not None and self.pol != 1:
            raise ValueError(
                "moment= is accepted at pol = 1 (TM) only: in TE the same source "
                "would be an in-plane magnetic dipole (slab-spec T7)"
            )
        b = None if moment is None else in_plane_moment_b(moment)
        cl = self.cluster
        k_bg = self.materials[0].wnum_bg(wavelength)
        rhs = np.zeros(cl.n_dof, dtype=complex)
        for p, gp in enumerate(cl.geometries):
            if b is not None:
                rhs[cl.slice(p)] = _in_plane_dipole_rhs(
                    gp.n_pts, k_bg, gp.f, gp.g, x_s, z_s, b
                )
                continue
            # line_dipole_rhs's own inside/too-close guards run once per
            # particle, which is exactly the cluster condition: the source must
            # lie outside every particle. No extra check is needed here and
            # none should be added.
            rhs[cl.slice(p)] = line_dipole_rhs(gp.n_pts, k_bg, gp.f, gp.g, x_s, z_s)
        refl = self._reflection(
            wavelength,
            np.append(np.concatenate([gp.f for gp in cl.geometries]), x_s),
            np.append(np.concatenate([gp.g for gp in cl.geometries]), z_s),
        )
        if refl is not None:
            for p, gp in enumerate(cl.geometries):
                gs = refl.green(gp.f, gp.g, x_s, z_s)
                term = gs[0] if b is None else -(b[0] * gs[1] + b[1] * gs[2])
                rhs[cl.offsets[p] : cl.offsets[p] + gp.n_pts] += term
        return rhs

    def _rhs_mode(
        self,
        wavelength: float,
        mode: GuidedMode | int,
        direction: int,
        x_ref: float,
    ) -> tuple[np.ndarray, GuidedMode]:
        """Right-hand side of an incident guided mode, and the mode it resolved to.

        Args:
            wavelength: Real vacuum wavelength λ_vac in nm.
            mode: A guided mode of this stack, or its index.
            direction: +1 towards +x, −1 towards −x.
            x_ref: Phase reference along x (nm).

        Returns:
            ``(rhs, mode)``: complex ``(n_dof,)`` right-hand side and the
            :class:`~pysie2d.layered.GuidedMode`.

        Raises:
            ValueError: Without a stack; complex λ or a lossy stack (T2); a
                direction other than ±1; a mode foreign to this stack.
        """
        n_clad = self.materials[0].n_clad
        if not isinstance(_effective(self.background, n_clad, wavelength), Stack):
            raise ValueError("scatter_mode needs a Stack background that guides")
        m = _resolve_mode(
            self.background, wavelength, self.pol, n_clad, mode, direction
        )
        cl = self.cluster
        rhs = np.zeros(cl.n_dof, dtype=complex)
        for p, gp in enumerate(cl.geometries):
            rhs[cl.offsets[p] : cl.offsets[p] + gp.n_pts] = _mode_rhs(
                m, direction, x_ref, gp.f, gp.g
            )
        return rhs, m

    def scatter(self, wavelength: float, angle: float = 0.0) -> "ClusterScatterResult":
        """Solve the coupled system for plane-wave excitation.

        Args:
            wavelength: Vacuum wavelength λ_vac in nm.
            angle: Incidence angle in degrees; 0 propagates along −z.

        Returns:
            The :class:`ClusterScatterResult` carrying the coupled solution.

        Warns:
            ClusterResolutionWarning: If the particles' resolutions differ by
                more than ``RESOLUTION_SPREAD_WARN``.
            ClusterGapWarning: If the closest gap falls below the quadrature
                envelope of ``GAP_ENVELOPE_C``.
        """
        self._check_resolution(wavelength)
        self._check_gap(wavelength)
        rhs = self._rhs_plane_wave(wavelength, angle)
        ei = np.linalg.solve(self._assemble(wavelength), rhs)
        return ClusterScatterResult(
            ei,
            self.cluster,
            self.materials,
            self.pol,
            wavelength,
            angle,
            "plane_wave",
            self.background,
        )

    def scatter_dipole(
        self,
        wavelength: float,
        x_s: float,
        z_s: float,
        *,
        moment: tuple[complex, complex] | None = None,
    ) -> "ClusterScatterResult":
        """Solve the coupled system for a line-dipole source.

        Args:
            wavelength: Vacuum wavelength λ_vac in nm.
            x_s: Source x-coordinate (nm).
            z_s: Source z-coordinate (nm).
            moment: ``(p_x, p_z)`` of an in-plane electric dipole, TM only;
                see :meth:`pysie2d.solver.BIESolver.scatter_dipole`.

        Returns:
            The :class:`ClusterScatterResult` carrying the coupled solution.

        Raises:
            ValueError: If the source lies inside, or too close to, any
                particle — raised by :func:`~pysie2d.sources.line_dipole_rhs`;
                a ``moment`` at ``pol = 2`` (T7).

        Warns:
            ClusterResolutionWarning: If the particles' resolutions differ by
                more than ``RESOLUTION_SPREAD_WARN``.
            ClusterGapWarning: If the closest gap falls below the quadrature
                envelope of ``GAP_ENVELOPE_C``.
        """
        self._check_resolution(wavelength)
        self._check_gap(wavelength)
        rhs = self._rhs_dipole(wavelength, x_s, z_s, moment)
        ei = np.linalg.solve(self._assemble(wavelength), rhs)
        return ClusterScatterResult(
            ei,
            self.cluster,
            self.materials,
            self.pol,
            wavelength,
            0.0,
            "dipole",
            self.background,
            source=(x_s, z_s),
            moment=None if moment is None else (complex(moment[0]), complex(moment[1])),
        )

    def scatter_mode(
        self,
        wavelength: float,
        mode: GuidedMode | int,
        direction: int,
        *,
        x_ref: float,
    ) -> "ClusterScatterResult":
        """Solve for a guided mode of the stack incident on the cluster.

        See :meth:`pysie2d.solver.BIESolver.scatter_mode`: the RHS φ-half is
        ``ũ_m(z_j)e^{±iβ(x_j − x_ref)}`` on every particle, the χ-half 0.

        Args:
            wavelength: Real vacuum wavelength λ_vac in nm.
            mode: A guided mode of this stack, or its index.
            direction: +1 towards +x, −1 towards −x.
            x_ref: Phase reference along x (nm), required (R14).

        Returns:
            The :class:`ClusterScatterResult`, ``excitation = "mode"``.

        Raises:
            ValueError: Without a stack; complex λ or a lossy stack (T2); a
                direction other than ±1; a mode foreign to this stack.
        """
        self._check_resolution(wavelength)
        self._check_gap(wavelength)
        rhs, m = self._rhs_mode(wavelength, mode, direction, x_ref)
        ei = np.linalg.solve(self._assemble(wavelength), rhs)
        return ClusterScatterResult(
            ei,
            self.cluster,
            self.materials,
            self.pol,
            wavelength,
            0.0,
            "mode",
            self.background,
            mode=m,
            direction=direction,
            x_ref=x_ref,
        )

    def _check_gap(self, wavelength: float) -> None:
        """Warn when the closest gap is finer than the quadrature can resolve.

        The cross-block integrand of two nearly touching boundaries has a
        near-singularity a conformal half-width ~ sqrt(2·gap/a) off the real
        axis; below ``GAP_ENVELOPE_C·sqrt(a/gap)`` nodes the trapezoid rule
        stops resolving it and the coupled solution degrades **silently** — the
        matrix stays well conditioned and every single-particle diagnostic
        stays clean. It is also the backstop for the one overlap case
        ``_check_overlap`` cannot see (a node-free lens sliver).

        Args:
            wavelength: Vacuum wavelength λ_vac in nm. Unused — the envelope is
                a purely geometric ratio (conventions §9 scale covariance) —
                but taken for symmetry with ``_check_resolution``.
        """
        cl = self.cluster
        if len(cl) < 2:
            return
        gap = cl.min_gap
        a = max(float(np.hypot(g.f - g.x0, g.g - g.z0).max()) for g in cl.geometries)
        nn_needed = GAP_ENVELOPE_C * np.sqrt(a / gap)
        nn_worst = min(g.n_pts for g in cl.geometries)
        if nn_worst >= nn_needed:
            return
        circular = all(g.is_circle for g in cl.geometries)
        detail = (
            ""
            if circular
            else (
                " This envelope was measured on circles and is not validated "
                "for non-circular facing boundaries (spec D9); treat it as a "
                "diagnostic, not a bound."
            )
        )
        warnings.warn(
            f"minimum gap {gap:.4g} nm needs about n_pts = {nn_needed:.0f} "
            f"per particle at this configuration, but the coarsest particle "
            f"has {nn_worst}. The cross-block quadrature degrades silently "
            f"below the envelope.{detail}",
            ClusterGapWarning,
            stacklevel=3,
        )

    def _check_resolution(
        self,
        wavelength: float | complex,
        materials: Sequence[Material] | None = None,
    ) -> None:
        """Warn when one particle is far coarser than its best-resolved peer.

        The user picks ``n_pts`` per particle (D7) — that freedom is the point
        of ragged resolution — but the coupled system is only as accurate as
        its worst block, and a particle that is under-resolved for its own size
        and index cannot be spotted from any other particle's numbers.

        Args:
            wavelength: Vacuum wavelength λ_vac in nm.
            materials: The set to check; the solver's own by default.
        """
        mats = self.materials if materials is None else materials
        res = [
            wavelength_over_ds(g, m, wavelength)
            for g, m in zip(self.cluster.geometries, mats, strict=True)
        ]
        if len(res) < 2:
            return
        worst = int(np.argmin(res))
        best = int(np.argmax(res))
        if res[best] <= RESOLUTION_SPREAD_WARN * res[worst]:
            return
        warnings.warn(
            f"particle {worst} is resolved at {res[worst]:.3g} points per "
            f"interior wavelength while particle {best} reaches "
            f"{res[best]:.3g}; the coupled solution is only as accurate as its "
            f"worst block",
            ClusterResolutionWarning,
            stacklevel=3,
        )


class ClusterScatterResult(_StackResult):
    """Result of a single-wavelength coupled cluster solve.

    Observables are additive over boundaries, which is what makes the façade
    thin: each particle's ``2·nn_p`` sub-vector (conventions §14) goes straight
    into the existing single-particle primitives and the contributions are
    summed. The coordinates ``f_p``, ``g_p`` are **absolute**, so each
    particle's geometric phase is already inside the far-field integrand and
    there is no extra phase factor to apply (§3.4) — a tempting and wrong
    addition.

    There is deliberately no ``efficiencies()``: it normalises by ``2·rad``,
    and a cluster has no ``rad`` (D5); ``cross_sections()`` is the absolute
    quantity that replaces it. There is deliberately no ``multipoles()``: the
    v0.7 expansion is about one particle centre, and a cluster needs an
    output-side translation theorem that ``multipole.py`` does not implement
    (§1.3). Both absences are decisions, not omissions.

    Attributes:
        ei: complex ``(n_dof,)`` coupled solution vector.
        cluster: The arrangement solved.
        materials: Per-particle optical properties.
        pol: Polarisation: 1 = TM, 2 = TE.
        wavelength: Vacuum wavelength λ_vac in nm.
        angle: Plane-wave incidence angle in degrees (0.0 for a dipole solve).
        background: The substrate the cluster sits on, or ``None``. With one,
            ``far_field`` is the upward amplitude and ``cross_sections`` returns
            ``{'c_sca_up', 'c_sca_down', 'c_abs'}``.
    """

    def __init__(
        self,
        ei: np.ndarray,
        cluster: Cluster,
        materials: Sequence[Material],
        pol: int,
        wavelength: float,
        angle: float,
        excitation: str,
        background: HalfSpace | Stack | None = None,
        *,
        source: tuple[float, float] | None = None,
        mode: GuidedMode | None = None,
        direction: int | None = None,
        x_ref: float | None = None,
        moment: tuple[complex, complex] | None = None,
    ) -> None:
        """Store a coupled solution and the problem that produced it.

        Args:
            ei: complex ``(n_dof,)`` coupled solution vector.
            cluster: The arrangement solved.
            materials: One material per particle, in the cluster's order.
            pol: Polarisation: 1 = TM, 2 = TE.
            wavelength: Vacuum wavelength λ_vac in nm.
            angle: Plane-wave incidence angle in degrees.
            excitation: ``"plane_wave"`` or ``"dipole"``.
            background: The substrate the cluster sits on, or ``None``.
            source: The dipole position ``(x_s, z_s)``, or ``None``.
            mode: The incident guided mode of a mode excitation, or ``None``.
            direction: Its direction, +1 or −1, or ``None``.
            x_ref: Its phase reference along x (nm), or ``None``.
            moment: In-plane dipole moment ``(p_x, p_z)``, or ``None``.
        """
        self.ei = ei
        self.cluster = cluster
        self.materials = tuple(materials)
        self.pol = pol
        self.wavelength = wavelength
        self.angle = angle
        self._excitation = excitation
        self.background = background
        self.source = source
        self.mode = mode
        self.direction = direction
        self.x_ref = x_ref
        self.moment = moment

    @property
    def _pol(self) -> int:
        return self.pol

    @property
    def _n_clad(self) -> float:
        return self.materials[0].n_clad

    @property
    def _excitation_name(self) -> str:
        return self._excitation

    def _boundaries(self) -> list:
        return [
            (gp.f, gp.g, gp.df, gp.dg, gp.delt, self.ei_particle(p))
            for p, gp in enumerate(self.cluster.geometries)
        ]

    def _substrate(self) -> tuple:
        """Arguments the half-space far-field helpers share."""
        bg = self._bg
        assert isinstance(bg, HalfSpace)
        return (
            self._boundaries(),
            self.wnum_bg,
            bg.eps_rel(self.materials[0].n_clad, self.wavelength),
            self.pol,
            bg.z_int,
        )

    @property
    def wnum_bg(self) -> complex:
        """Background wavenumber k_bg = 2π·n_clad/λ_vac (rad/nm)."""
        return self.materials[0].wnum_bg(self.wavelength)

    def ei_particle(self, p: int) -> np.ndarray:
        """Return particle ``p``'s own sub-vector, in conventions §4 order.

        Args:
            p: Particle index.

        Returns:
            complex ``(2·nn_p,)`` slice: φ_p first, then χ_p — exactly what the
            single-particle primitives take.
        """
        return self.ei[self.cluster.slice(p)]

    def far_field(self, n_angles: int = 3000) -> tuple[np.ndarray, np.ndarray]:
        """Total far-field amplitude of the cluster.

        Args:
            n_angles: Number of observation angles over [−π, π] inclusive.

        Returns:
            ``(amp, angles)``: complex ``(n_angles,)`` amplitude and the
            observation angles in radians, measured from +z. With a background
            this is the upward amplitude on ``[−π/2, π/2]`` inclusive.
        """
        bg = self._bg
        if isinstance(bg, Stack):
            return self._stack_far_field(n_angles)
        if bg is not None:
            return _upward_far_field(n_angles, *self._substrate())
        angles = -PI + np.arange(n_angles) * 2.0 * PI / (n_angles - 1.0)
        return self._amp_at(angles), angles

    def far_field_down(self, n_angles: int = 3000) -> tuple[np.ndarray, np.ndarray]:
        """Downward far-field amplitude ``a_down(θ)`` below the substrate.

        ``ψ ~ (i/4)·√(2/(π k_bot ρ))·e^{i(k_bot ρ − π/4)}·a_down(θ)`` about the
        origin, ``k_bot = k·√ε_rel`` the substrate wavenumber: :meth:`far_field`'s
        convention in the substrate. ``a_down = 0`` at both grazing ends unless
        ``ε_rel = 1``. At complex λ over a real ``ε > 0`` this is the analytic
        continuation, as :meth:`far_field` is (D16).

        Args:
            n_angles: Number of observation angles.

        Returns:
            amplitude: complex (n_angles,) amplitude.
            angles: float (n_angles,) angles from +z, uniform on ``[π/2, 3π/2]``
                inclusive.

        Raises:
            ValueError: Without a background: :meth:`far_field` already covers
                the full circle (D20).
            NoFarFieldError: Over PEC, an absorbing substrate (``Im ε ≠ 0``) or
                ``Re ε ≤ 0``, where no wave reaches infinity below (D2).
        """
        bg = self._bg
        if bg is None:
            raise ValueError(
                "far_field_down needs a background: without one far_field() "
                "already covers the full circle"
            )
        if isinstance(bg, Stack):
            return self._stack_far_field_down(n_angles)
        args = self._substrate()
        _check_far_field_below(args[2])
        return _downward_far_field(n_angles, *args)

    def _amp_at(self, angles: np.ndarray) -> np.ndarray:
        """Far-field amplitude at arbitrary angles.

        Args:
            angles: (M,) observation angles in radians.

        Returns:
            complex (M,) total amplitude.
        """
        # The absolute coordinates f_p, g_p already carry each particle's
        # geometric phase inside exp(-i k (f sinθ + g cosθ)); adding one here
        # would apply it twice (§3.4).
        k = self.wnum_bg
        total = np.zeros(len(angles), dtype=complex)
        for p, gp in enumerate(self.cluster.geometries):
            total += _far_field_at(
                angles,
                gp.n_pts,
                k,
                gp.f,
                gp.g,
                gp.df,
                gp.dg,
                gp.delt,
                self.ei_particle(p),
            )
        return total

    def cross_sections(self, n_angles: int = 500) -> dict[str, float]:
        """Absolute scattering, extinction and absorption cross-sections (nm).

        The cluster observable of §3.6, and the same quantity
        :meth:`pysie2d.solver.ScatterResult.cross_sections` reports for one
        particle, so the two paths are comparable.

        Args:
            n_angles: Far-field grid size for the C_sca quadrature.

        Returns:
            ``{"c_sca", "c_ext", "c_abs"}`` in nm. With a background, instead
            ``{"c_sca_up", "c_sca_down", "c_abs"}`` (see
            :meth:`pysie2d.solver.ScatterResult.cross_sections`, which states
            the plasmon leak over a lossless TM metal); the free-space keys are
            absent rather than raising, and ``c_sca_down`` is absent at
            complex λ.

        Raises:
            ValueError: If this result did not come from plane-wave excitation.
        """
        if self._excitation != "plane_wave":
            raise ValueError(
                "cross_sections() is defined against a unit-amplitude "
                "incident plane wave; this result came from "
                f"{self._excitation} excitation"
            )
        bg = self._bg
        if isinstance(bg, Stack):
            return self._stack_cross_sections(n_angles)
        if bg is not None:
            args = self._substrate()
            c = _upward_cross_sections(n_angles, *args)
            if np.imag(self.wavelength) == 0.0:  # D18: no real-axis flux otherwise
                c["c_sca_down"] = _downward_cross_section(*args)
            return c
        amp, _ = self.far_field(n_angles)
        # π − α lands on the far-field grid only by accident, so the forward
        # direction is evaluated exactly rather than picked off it.
        amp_fwd = self._amp_at(np.array([PI - np.deg2rad(self.angle)]))[0]
        return _cross_sections(amp, amp_fwd, self.wnum_bg)

    def eval_field(self, x: np.ndarray, z: np.ndarray) -> np.ndarray:
        """Evaluate the field at arbitrary points, inside or outside.

        Outside every particle the field is the sum of the representation
        integral over all boundaries at ``k_bg``; inside particle ``p`` it is
        that boundary alone at ``k_core,p = nc_p·k_bg``. Overlap being
        excluded, a point is inside at most one particle.

        Classification uses ray casting (:func:`pysie2d.sources._point_inside`)
        rather than the nearest-point normal test
        :func:`pysie2d.fields._is_outside` that
        :meth:`pysie2d.solver.ScatterResult.eval_field` uses — the latter is
        documented unreliable for concave superformula shapes, and a cluster
        runs the test once per particle, so a misclassification is that many
        times more likely (§3.5).

        Never evaluate on or very near a boundary: the representation-formula
        integrand is near-singular there. Keep points at least ~5 node spacings
        clear of every particle.

        Args:
            x: Observation x-coordinates (nm); any shape, flattened.
            z: Observation z-coordinates (nm); same length as ``x``.

        Returns:
            complex (M,) field at each point.

        Raises:
            ValueError: With a background, a point exactly at the interface
                (D17: the field is continuous there, but step off it).

        With a background this is the **scattered** field: outside every
        particle the representation integral gains the reflected-image term of
        each boundary, ``−Σ delt·((∂x'G_ind·dg − ∂z'G_ind·df)·φ + G_ind·χ)``.
        Interior points are unchanged. The total exterior field is scattered +
        incident + reflected incident; the last two are the caller's.

        Below the interface it is the transmitted term of every boundary,
        ``−Σ delt·((∂x'G_t·dg − ∂z'G_t·df)·φ + G_t·χ)``, with no free-space
        term; 0 over PEC (D14), and the free-space representation at
        ``ε_rel = 1`` exactly (D19). Particles are in the cover, so no
        substrate point needs classifying. Points above the interface are
        computed exactly as without any substrate point, bit for bit.
        """
        x = np.asarray(x, dtype=float).ravel()
        z = np.asarray(z, dtype=float).ravel()
        bg = self._bg
        if isinstance(bg, Stack):
            return self._stack_eval_field(x, z)
        if bg is not None:
            z_int = bg.z_int
            if np.any(z == z_int):
                raise ValueError(
                    f"observation point exactly at the interface z_int = {z_int}: "
                    "step off it (the field is continuous there)"
                )
            below = z < z_int
            if below.any():
                field = np.empty(z.size, dtype=complex)
                if not below.all():
                    field[~below] = self._cover_field(x[~below], z[~below])
                field[below] = self._substrate_field(x[below], z[below])
                return field
        return self._cover_field(x, z)

    def _substrate_field(self, x: np.ndarray, z: np.ndarray) -> np.ndarray:
        """Scattered field at points below the interface."""
        bg = self._bg
        assert isinstance(bg, HalfSpace)
        geoms = self.cluster.geometries
        eps_rel = bg.eps_rel(self.materials[0].n_clad, self.wavelength)
        if eps_rel is None:
            return np.zeros(x.size, dtype=complex)
        # One path for every particle: sized for the whole cluster's extent.
        tr = layered.Transmission.build(
            self.pol,
            self.wnum_bg,
            eps_rel,
            bg.z_int,
            np.concatenate([gp.f for gp in geoms]),
            np.concatenate([gp.g for gp in geoms]),
            x,
            z,
        )
        if tr is None:  # ε_rel = 1: no interface, the free-space representation
            return self._cover_field(x, z)
        pts = layered.Points(x, z)
        field = np.zeros(x.size, dtype=complex)
        for p, gp in enumerate(geoms):
            m1, m2 = tr.blocks(pts, gp)
            ei_p = self.ei_particle(p)
            field -= m1 @ ei_p[: gp.n_pts] + m2 @ ei_p[gp.n_pts :]
        return field

    def _cover_field(self, x: np.ndarray, z: np.ndarray) -> np.ndarray:
        """Field at points above the interface (the v0.9 path)."""
        k = _real_if_real(self.wnum_bg)
        refl = None
        bg = self._bg
        if isinstance(bg, Stack):
            refl = self._stack_cover_reflection(x, z)
        elif bg is not None:
            geoms = self.cluster.geometries
            refl = layered.Reflection.build(
                self.pol,
                self.wnum_bg,
                bg.eps_rel(self.materials[0].n_clad, self.wavelength),
                bg.z_int,
                np.concatenate([gp.f for gp in geoms] + [x]),
                np.concatenate([gp.g for gp in geoms] + [z]),
            )
        inside = np.full(len(x), -1, dtype=int)
        for p, gp in enumerate(self.cluster.geometries):
            for j in range(len(x)):
                if inside[j] < 0 and _point_inside(x[j], z[j], gp.f, gp.g):
                    inside[j] = p
        field = np.zeros(len(x), dtype=complex)
        out = inside < 0
        if out.any():
            for p, gp in enumerate(self.cluster.geometries):
                field[out] += _representation_at(
                    self.ei_particle(p),
                    gp.n_pts,
                    gp.f,
                    gp.df,
                    gp.g,
                    gp.dg,
                    gp.delt,
                    k,
                    x[out],
                    z[out],
                )
            if refl is not None:
                pts = layered.Points(x[out], z[out])
                for p, gp in enumerate(self.cluster.geometries):
                    m1, m2 = refl.blocks(pts, gp)
                    ei_p = self.ei_particle(p)
                    field[out] -= m1 @ ei_p[: gp.n_pts] + m2 @ ei_p[gp.n_pts :]
        for p, gp in enumerate(self.cluster.geometries):
            sel = inside == p
            if sel.any():
                mat = self.materials[p]
                k_core = _real_if_real(mat.nc * self.wnum_bg)
                # Interior representation (conventions §4): χ is the
                # exterior-side derivative, eps·χ inside in TM, and the
                # interior integral carries the opposite sign.
                ei_in = self.ei_particle(p).copy()
                if mat.pol == 1:
                    ei_in[gp.n_pts :] *= mat.eps
                field[sel] = -_representation_at(
                    ei_in,
                    gp.n_pts,
                    gp.f,
                    gp.df,
                    gp.g,
                    gp.dg,
                    gp.delt,
                    k_core,
                    x[sel],
                    z[sel],
                )
        return field

    def _background_field(self, x: np.ndarray, z: np.ndarray) -> np.ndarray:
        """The unmasked background field; see :meth:`incident_field`."""
        bg = self._bg
        if isinstance(bg, Stack):
            return self._stack_background(x, z)
        n_clad = self.materials[0].n_clad
        return _background_field(
            x,
            z,
            self.wnum_bg,
            self.pol,
            self._excitation,
            self.angle,
            self.source,
            None if bg is None else bg.eps_rel(n_clad, self.wavelength),
            None if bg is None else bg.z_int,
            self._b,
        )

    def incident_field(self, x: np.ndarray, z: np.ndarray) -> np.ndarray:
        """Background field the cluster was driven by, at arbitrary points (nm).

        ``eval_field + incident_field`` is the total field at every point (D13);
        see :meth:`pysie2d.solver.ScatterResult.incident_field`. 0 inside every
        particle, classified by ray casting as :meth:`eval_field` does.

        Args:
            x: Observation x-coordinates (nm).
            z: Observation z-coordinates (nm).

        Returns:
            complex (M,) field.

        Raises:
            ValueError: At a point exactly on the interface (D17), or at the
                dipole's own position.
        """
        x = np.asarray(x, dtype=float).ravel()
        z = np.asarray(z, dtype=float).ravel()
        field = self._background_field(x, z)
        z_int = -np.inf if self.background is None else _top(self.background)
        for j in np.flatnonzero(z > z_int):  # particles lie in the cover
            if any(
                _point_inside(x[j], z[j], gp.f, gp.g) for gp in self.cluster.geometries
            ):
                field[j] = 0.0
        return field

    def resolution(self) -> tuple[float, ...]:
        """Points per interior wavelength, per particle (D7).

        Returns:
            One :func:`~pysie2d.solver.wavelength_over_ds` value per particle,
            in the cluster's order.
        """
        return tuple(
            wavelength_over_ds(g, m, self.wavelength)
            for g, m in zip(self.cluster.geometries, self.materials, strict=True)
        )


@dataclass(frozen=True)
class EigenpermittivityResult:
    """Eigenpermittivities of a cluster inside a rectangle, at one real wavelength.

    Attributes:
        eps: complex (K,) **absolute** permittivities, the value
            :meth:`pysie2d.material.Material.from_eps` takes, sorted by real part
            then imaginary part.
        vectors: complex (n_dof, K) unit source-free solutions, one column per
            eigenvalue, in the layout of conventions §14.
        converged: bool (K,) whether each Newton polish met its tolerance.
        rank: Rank of the contour moment, before the in-contour filter.
        sv_ratio: Singular values of the contour moment over the largest; the rank
            gap is visible here.
    """

    eps: np.ndarray
    vectors: np.ndarray
    converged: np.ndarray
    rank: int
    sv_ratio: np.ndarray


@dataclass(frozen=True)
class _Factored:
    """One material set's factors, reusable for every right-hand side."""

    blocks: list[np.ndarray]
    lus: list[tuple[np.ndarray, np.ndarray]]
    responses: list[np.ndarray]
    reduced: np.ndarray


class ClusterEigenEps:
    """A cluster at one fixed wavelength: fast material sweeps and eigenpermittivities.

    **What it is for.** Solving one cluster for many materials at one wavelength.
    :meth:`ClusterBIESolver.scatter` rebuilds and factorises the whole coupled matrix
    on every call. Here the part that does not depend on the materials is built once,
    so each later solve costs only a set of small per-particle problems and one small
    coupled problem. The answer is the dense solve's, to round-off.

    Measured per material set with 100 nodes per particle, distinct particles and an
    absorbing ε (docs/design/studies/cluster-reduction/README.md):

    ========= ================= ================ ========
    particles ``scatter``, free ``scatter``, slab here
    ========= ================= ================ ========
    4         0.03 s            0.18 s           11 ms
    16        0.46 s            8.5 s            50 ms
    32        2.8 s             66 s             0.12 s
    64        27 s              not run          0.33 s
    ========= ================= ================ ========

    Setup costs about one ``scatter`` call (slab: 65 s at 32 particles, nearly all of
    it the reflected-block assembly), so it pays for itself within the first few
    materials. It does **not** speed up a sweep in wavelength, where everything changes at each step
    and only very large clusters gain.

    **How.** Split the coupled matrix as ``M(ε) = D(ε) + U·Y``, with ``D`` the isolated
    particles, substrate included, and ``U·Y`` the SVD-compressed neighbour coupling.
    Only the interior rows of ``D`` depend on ε (conventions §14), and Woodbury leaves
    the small matrix ``G = I + Y·D⁻¹·U``. The rank per particle saturates with cluster
    size, because it measures what one small particle can receive.

    Build it with :meth:`ClusterBIESolver.at_wavelength`.

    Attributes:
        solver: The solver this was prepared from. Its materials are not used for
            solving; only their shared ``n_clad`` is.
        wavelength: Vacuum wavelength λ_vac in nm.
        coupling_ranks: Retained coupling rank per particle.
    """

    def __init__(self, solver: ClusterBIESolver, wavelength: float | complex) -> None:
        """Build and compress the exterior rows, one particle at a time.

        They are never held together: that is what keeps the dense matrix, 2.6 GB at
        64 particles against 380 MB for the compressed form, out of memory.

        Args:
            solver: The cluster, polarisation and background.
            wavelength: Vacuum wavelength λ_vac in nm. May be complex.
        """
        self.solver = solver
        self.wavelength = wavelength
        cl = solver.cluster
        refl = solver._reflection(wavelength)
        self._refl = refl
        self._top: list[np.ndarray] = []
        self._u: list[np.ndarray] = []
        self._y: list[np.ndarray] = []
        for q in range(len(cl)):
            rows = solver._exterior_rows(q, wavelength, refl)
            self._top.append(rows[:, cl.slice(q)].copy())
            rows[:, cl.slice(q)] = 0.0
            u, sv, _ = np.linalg.svd(rows, full_matrices=False)
            rank = int(np.sum(sv > COUPLING_RANK_TOL * sv[0]))
            self._u.append(u[:, :rank])
            self._y.append(u[:, :rank].conj().T @ rows)
        self.coupling_ranks = tuple(u.shape[1] for u in self._u)
        self._offsets = np.cumsum((0, *self.coupling_ranks))

    def _interior(self, p: int, ri: complex, kd: complex) -> np.ndarray:
        """Interior rows of particle ``p`` at a background-relative ``ri``, ``kd``."""
        g = self.solver.cluster.geometries[p]
        return _interior_rows(
            self.solver.pol,
            g.n_pts,
            g.f,
            g.g,
            g.df,
            g.dg,
            g.ddf,
            g.ddg,
            self.solver.materials[0].wnum_bg(self.wavelength),
            ri,
            kd,
        )

    def _factor(self, optics: Sequence[tuple[complex, complex]]) -> _Factored:
        """Factorise the isolated particles and build the small coupled matrix.

        Args:
            optics: ``(ri, kd)`` per particle, background-relative.

        Returns:
            The factors of ``D`` and the matrix ``G``.
        """
        cl = self.solver.cluster
        blocks = [
            np.vstack([self._top[p], self._interior(p, *optics[p])])
            for p in range(len(cl))
        ]
        lus = [lu_factor(b) for b in blocks]
        responses = [
            lu_solve(lus[p], self._embed(p, self._u[p])) for p in range(len(cl))
        ]
        o = self._offsets
        reduced = np.eye(o[-1], dtype=complex)
        for p in range(len(cl)):
            for q in range(len(cl)):
                reduced[o[p] : o[p + 1], o[q] : o[q + 1]] += (
                    self._y[p][:, cl.slice(q)] @ responses[q]
                )
        return _Factored(blocks, lus, responses, reduced)

    def _embed(self, p: int, exterior: np.ndarray) -> np.ndarray:
        """Pad exterior rows with zero interior rows: the coupling has none (§14)."""
        return np.vstack([exterior, np.zeros_like(exterior)])

    def _apply(
        self,
        fac: _Factored,
        reduced_lu: tuple[np.ndarray, np.ndarray] | None,
        rhs: np.ndarray,
    ) -> np.ndarray:
        """Solve ``M x = rhs`` by Woodbury.

        Args:
            fac: The factors.
            reduced_lu: The LU factors of ``fac.reduced``; ``None`` for one particle,
                where there is nothing to couple.
            rhs: complex ``(n_dof,)``, exterior and interior rows both, because a
                refinement residual has non-zero interior rows.

        Returns:
            complex ``(n_dof,)`` solution.
        """
        cl = self.solver.cluster
        o = self._offsets
        alone = [lu_solve(fac.lus[p], rhs[cl.slice(p)]) for p in range(len(cl))]
        x = np.empty(cl.n_dof, dtype=complex)
        y = np.empty(0, dtype=complex)
        if reduced_lu is not None:
            g = np.zeros(o[-1], dtype=complex)
            for p in range(len(cl)):
                for q in range(len(cl)):
                    g[o[p] : o[p + 1]] += self._y[p][:, cl.slice(q)] @ alone[q]
            y = lu_solve(reduced_lu, g)
        for p in range(len(cl)):
            x[cl.slice(p)] = alone[p] - fac.responses[p] @ y[o[p] : o[p + 1]]
        return x

    def _residual(self, fac: _Factored, rhs: np.ndarray, x: np.ndarray) -> np.ndarray:
        """Return ``rhs − M·x``, without forming ``M``."""
        cl = self.solver.cluster
        r = rhs.copy()
        for p in range(len(cl)):
            sl = cl.slice(p)
            r[sl] -= fac.blocks[p] @ x[sl]
            r[sl.start : sl.start + self._u[p].shape[0]] -= self._u[p] @ (
                self._y[p] @ x
            )
        return r

    def _solve(self, materials: Sequence[Material], rhs: np.ndarray) -> np.ndarray:
        """Solve for one material set, refining against the Woodbury cancellation.

        Args:
            materials: One material per particle, already validated.
            rhs: complex ``(n_dof,)`` right-hand side.

        Returns:
            complex ``(n_dof,)`` solution.

        Warns:
            ClusterCancellationWarning: If refining did not converge.
        """
        fac = self._factor([(m.nc, m.eps) for m in materials])
        reduced_lu = lu_factor(fac.reduced) if self._offsets[-1] else None
        x = self._apply(fac, reduced_lu, rhs)
        change = np.inf
        for _ in range(MAX_REFINE_STEPS):
            dx = self._apply(fac, reduced_lu, self._residual(fac, rhs, x))
            x = x + dx
            change = float(np.linalg.norm(dx) / np.linalg.norm(x))
            if change < REFINE_CONVERGED:
                break
        if change > REFINE_WARN:
            warnings.warn(
                f"a particle is so close to resonance on its own that the fast solve "
                f"lost accuracy: the last refinement step changed x by {change:.1e}. "
                f"ClusterBIESolver.scatter solves this case directly.",
                ClusterCancellationWarning,
                stacklevel=3,
            )
        return x

    def _validated(self, materials: Sequence[Material]) -> tuple[Material, ...]:
        """Check one material set against this cluster and return it.

        Args:
            materials: One material per particle.

        Returns:
            The materials as a tuple.

        Raises:
            ValueError: As :class:`ClusterBIESolver` does, or if ``n_clad`` differs
                from the solver's: the exterior rows were built for that one.
        """
        solver = self.solver
        _check_materials(materials, len(solver.cluster), solver.pol)
        if materials[0].n_clad != solver.materials[0].n_clad:
            raise ValueError(
                f"materials have n_clad = {materials[0].n_clad}, but the exterior "
                f"rows were built for n_clad = {solver.materials[0].n_clad}"
            )
        solver._check_resolution(self.wavelength, materials)
        return tuple(materials)

    def scatter(
        self, materials: Sequence[Material], angle: float = 0.0
    ) -> ClusterScatterResult:
        """Solve for plane-wave excitation with one material set.

        Args:
            materials: One material per particle.
            angle: Incidence angle in degrees; 0 propagates along −z.

        Returns:
            The same result :meth:`ClusterBIESolver.scatter` returns.

        Raises:
            ValueError: If ``materials`` does not fit this cluster.

        Warns:
            ClusterResolutionWarning: See :meth:`ClusterBIESolver.scatter`.
            ClusterCancellationWarning: If a particle is too close to resonance on
                its own for this solve to be accurate.
        """
        mats = self._validated(materials)
        solver = self.solver
        rhs = solver._rhs_plane_wave(self.wavelength, angle, self._refl)
        return ClusterScatterResult(
            self._solve(mats, rhs),
            solver.cluster,
            mats,
            solver.pol,
            self.wavelength,
            angle,
            "plane_wave",
            solver.background,
        )

    def scatter_dipole(
        self,
        materials: Sequence[Material],
        x_s: float,
        z_s: float,
        *,
        moment: tuple[complex, complex] | None = None,
    ) -> ClusterScatterResult:
        """Solve for a line-dipole source with one material set.

        Args:
            materials: One material per particle.
            x_s: Source x-coordinate (nm).
            z_s: Source z-coordinate (nm).
            moment: See :meth:`ClusterBIESolver.scatter_dipole`.

        Returns:
            The same result :meth:`ClusterBIESolver.scatter_dipole` returns.

        Raises:
            ValueError: If ``materials`` does not fit this cluster; for the source,
                as :meth:`ClusterBIESolver.scatter_dipole`.

        Warns:
            ClusterResolutionWarning: See :meth:`ClusterBIESolver.scatter`.
            ClusterCancellationWarning: See :meth:`scatter`.
        """
        mats = self._validated(materials)
        solver = self.solver
        rhs = solver._rhs_dipole(self.wavelength, x_s, z_s, moment)
        return ClusterScatterResult(
            self._solve(mats, rhs),
            solver.cluster,
            mats,
            solver.pol,
            self.wavelength,
            0.0,
            "dipole",
            solver.background,
            source=(x_s, z_s),
            moment=None if moment is None else (complex(moment[0]), complex(moment[1])),
        )

    def scatter_mode(
        self,
        materials: Sequence[Material],
        mode: GuidedMode | int,
        direction: int,
        *,
        x_ref: float,
    ) -> ClusterScatterResult:
        """Solve for an incident guided mode with one material set.

        Args:
            materials: One material per particle.
            mode: A guided mode of this stack, or its index.
            direction: +1 towards +x, −1 towards −x.
            x_ref: Phase reference along x (nm), required (R14).

        Returns:
            The same result :meth:`ClusterBIESolver.scatter_mode` returns.

        Raises:
            ValueError: If ``materials`` does not fit this cluster; otherwise as
                :meth:`ClusterBIESolver.scatter_mode`.

        Warns:
            ClusterResolutionWarning: See :meth:`ClusterBIESolver.scatter`.
            ClusterCancellationWarning: See :meth:`scatter`.
        """
        mats = self._validated(materials)
        solver = self.solver
        rhs, m = solver._rhs_mode(self.wavelength, mode, direction, x_ref)
        return ClusterScatterResult(
            self._solve(mats, rhs),
            solver.cluster,
            mats,
            solver.pol,
            self.wavelength,
            0.0,
            "mode",
            solver.background,
            mode=m,
            direction=direction,
            x_ref=x_ref,
        )

    def _reduced_at(self, eps: complex) -> np.ndarray:
        """Beyn's operator at a background-relative ``eps``, shared by all particles.

        ``G⁻¹ = I − Y·M⁻¹·U`` has poles only at cluster modes, the isolated-particle
        poles of ``G`` itself cancelling, so ``G`` is a legal contour operator despite
        ``beyn_modes``' warning about poles of M (conventions §18). With one particle
        there is no ``G``, and this returns the particle's own matrix.

        Args:
            eps: Background-relative permittivity.

        Returns:
            complex ``G(eps)``, or ``M_1(eps)`` for one particle.
        """
        ri = _upper_branch(eps)
        n = len(self.solver.cluster)
        if self._offsets[-1] == 0:
            return np.vstack([self._top[0], self._interior(0, ri, eps)])
        return self._factor([(ri, eps)] * n).reduced

    def _reduced_derivative(self, eps: complex) -> np.ndarray:
        """Return ``dG/dε = −Y·D⁻¹·(dD/dε)·D⁻¹·U`` (``dM_1/dε`` for one particle).

        Differences only the interior rows, which are smooth in ε. A difference of
        ``G`` itself would straddle its isolated-particle poles, which at high Q lie
        far closer than any usable step.

        Args:
            eps: Background-relative permittivity.

        Returns:
            complex matrix of the same shape as :meth:`_reduced_at`.
        """
        cl = self.solver.cluster
        h = INTERIOR_DERIVATIVE_STEP

        def d_interior(p: int) -> np.ndarray:
            up = self._interior(p, _upper_branch(eps + h), eps + h)
            down = self._interior(p, _upper_branch(eps - h), eps - h)
            return (up - down) / (2.0 * h)

        if self._offsets[-1] == 0:
            nn = cl.geometries[0].n_pts
            return np.vstack([np.zeros((nn, 2 * nn), dtype=complex), d_interior(0)])
        fac = self._factor([(_upper_branch(eps), eps)] * len(cl))
        o = self._offsets
        out = np.zeros_like(fac.reduced)
        for q in range(len(cl)):
            moved = d_interior(q) @ fac.responses[q]
            back = lu_solve(fac.lus[q], np.vstack([np.zeros_like(moved), moved]))
            for p in range(len(cl)):
                out[o[p] : o[p + 1], o[q] : o[q + 1]] = (
                    -self._y[p][:, cl.slice(q)] @ back
                )
        return out

    def _mode_vector(self, eps: complex, y: np.ndarray) -> np.ndarray:
        """Recover the coupled solution from a null vector of ``G``.

        Args:
            eps: Background-relative permittivity.
            y: Null vector of :meth:`_reduced_at`.

        Returns:
            complex ``(n_dof,)`` unit vector in the layout of conventions §14:
            ``x_p = −R_p·y_p``, or ``y`` itself for one particle.
        """
        cl = self.solver.cluster
        if self._offsets[-1] == 0:
            return y / np.linalg.norm(y)
        fac = self._factor([(_upper_branch(eps), eps)] * len(cl))
        o = self._offsets
        x = np.empty(cl.n_dof, dtype=complex)
        for p in range(len(cl)):
            x[cl.slice(p)] = -fac.responses[p] @ y[o[p] : o[p + 1]]
        return x / np.linalg.norm(x)

    def eigenpermittivities(
        self,
        eps_lo: complex,
        eps_hi: complex,
        *,
        n_quad_per_side: int = 32,
        n_probe: int = 16,
        rank_tol: float = 1.0e-8,
        tol: float = 1.0e-12,
        max_iter: int = 30,
    ) -> EigenpermittivityResult:
        """Find the permittivities at which the cluster supports a source-free field.

        All particles share the unknown ε, at this object's real wavelength. Unlike
        :class:`~pysie2d.qnm.QNMSolver`, this works over a
        :class:`~pysie2d.background.Stack`, because the background is held fixed.

        **Over a Stack the result has no external check.** It has been compared only
        with a pole fitted to this package's own driven solves, agreeing to about 1e-6;
        a dielectric half-space likewise. Free space and PEC are checked against
        exact references, the Graf addition theorem and a mirror image (conventions
        §18).

        A contour that ends with no rank gap raises, usually because a mode sits close
        to its edge: move the edge. TM plasmonic modes accumulate at ``ε = −n_clad²``
        on the real axis, so keep the top edge away from it.

        Args:
            eps_lo: Bottom-left corner, as an **absolute** permittivity.
            eps_hi: Top-right corner, as an **absolute** permittivity.
            n_quad_per_side: Quadrature nodes per side of the contour.
            n_probe: Random probes; must exceed the number of modes inside. Capped at
                the size of the small system.
            rank_tol: Rank threshold of the contour moment.
            tol: Newton convergence on background-relative ε.
            max_iter: Newton iteration cap.

        Returns:
            The modes inside the rectangle; ``eps`` is absolute, ready for
            :meth:`pysie2d.material.Material.from_eps`.

        Raises:
            ValueError: If the wavelength is complex, or the rectangle meets the real
                ray ``ε ≥ 0``: the branch cut of ``√ε``, with ``ε = 0`` on it.
        """
        if np.imag(self.wavelength) != 0.0:
            raise ValueError(
                "eigenpermittivities needs a real wavelength: a modal permittivity "
                "is defined at a fixed real frequency"
            )
        n_clad2 = self.solver.materials[0].n_clad ** 2
        lo, hi = complex(eps_lo) / n_clad2, complex(eps_hi) / n_clad2
        if lo.imag <= 0.0 <= hi.imag and hi.real >= 0.0:
            raise ValueError(
                f"the rectangle {eps_lo}..{eps_hi} meets the real ray eps >= 0, the "
                f"branch cut of sqrt(eps) (conventions sec. 18). Keep it below "
                f"the positive real axis, or strictly left of 0."
            )
        size = self._reduced_at(lo).shape[0]
        found = beyn_modes(
            self._reduced_at,
            lo,
            hi,
            n_quad_per_side=n_quad_per_side,
            n_probe=min(n_probe, size),
            rank_tol=rank_tol,
        )
        polished = [
            newton_refine(
                self._reduced_at,
                self._reduced_derivative,
                eps,
                found.vectors[:, k],
                lo,
                hi,
                tol=tol,
                max_iter=max_iter,
            )
            for k, eps in enumerate(found.eigenvalues)
        ]
        eps_rel = np.array([m.eigenvalue for m in polished], dtype=complex)
        order = np.lexsort((eps_rel.imag, eps_rel.real))
        vectors = np.empty((self.solver.cluster.n_dof, order.size), dtype=complex)
        for col, k in enumerate(order):
            vectors[:, col] = self._mode_vector(eps_rel[k], polished[k].vector)
        return EigenpermittivityResult(
            eps=eps_rel[order] * n_clad2,
            vectors=vectors,
            converged=np.array([polished[k].converged for k in order], dtype=bool),
            rank=found.rank,
            sv_ratio=found.sv_ratio,
        )
