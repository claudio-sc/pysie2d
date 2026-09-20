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

from collections.abc import Sequence

import numpy as np

from .geometry import Geometry
from .sources import _point_inside

# A resolution spread wider than this between the best- and worst-resolved
# particle warns. The coupled system is only as accurate as its worst block,
# and that is not visible from any one particle. Four is two halvings of nn,
# which in the M2 pilot is the difference between round-off and 1e-6 at a
# fixed gap (docs/design/studies/cluster-gap-envelope.md).
RESOLUTION_SPREAD_WARN = 4.0


class ClusterOverlapError(ValueError):
    """Two particle boundaries intersect; the formulation does not apply."""


class ClusterGapWarning(UserWarning):
    """A gap is too small for the quadrature resolution in use."""


class ClusterResolutionWarning(UserWarning):
    """One particle is far more coarsely resolved than its neighbours."""


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
