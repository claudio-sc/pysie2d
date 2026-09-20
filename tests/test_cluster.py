"""The `Cluster` container and its geometric guards.

A cluster is a purely geometric object (spec D2): it fixes the coupled
degree-of-freedom layout of conventions §14 and rejects arrangements the
boundary-integral formulation cannot represent. These tests pin both — the
layout arithmetic under *ragged* `n_pts`, where an off-by-one offset silently
mixes one particle's φ with another's χ, and the three-band overlap test in
each of its bands.
"""

import numpy as np
import pytest

from pysie2d import Cluster, ClusterOverlapError, Geometry

RAD = 100.0


def _circle(n_pts: int, x0: float = 0.0, z0: float = 0.0) -> Geometry:
    return Geometry.gielis(RAD, n_pts, m=0, x0=x0, z0=z0)


def test_dof_layout_is_contiguous_per_particle_with_ragged_resolutions():
    """Offsets and slices follow conventions §14 for unequal `n_pts`.

    Ragged resolutions are the case the layout exists for: with equal `n_pts`
    almost any offset rule happens to work, so the test uses three different
    ones. Each slice must have length exactly 2·nn_p and the slices must tile
    range(n_dof) without gap or overlap — a property that cannot hold by
    accident if any offset is wrong, since a single misplaced boundary both
    shortens one slice and lengthens its neighbour.
    """
    n_pts = (40, 61, 33)
    cluster = Cluster([_circle(n, x0=400.0 * i) for i, n in enumerate(n_pts)])

    assert len(cluster) == 3
    assert cluster.n_dof == 2 * sum(n_pts)
    assert cluster.offsets == (0, 80, 202, 268)

    covered = np.zeros(cluster.n_dof, dtype=int)
    for p, n in enumerate(n_pts):
        sl = cluster.slice(p)
        assert sl.stop - sl.start == 2 * n
        covered[sl] += 1
    assert np.all(covered == 1)


def test_single_particle_cluster_is_allowed():
    """One particle is a legal cluster; §6.1 compares it to `BIESolver`.

    There is no pair, so no overlap test and no finite gap.
    """
    cluster = Cluster([_circle(40)])
    assert len(cluster) == 1
    assert cluster.n_dof == 80
    assert cluster.slice(0) == slice(0, 80)
    assert cluster.min_gap == np.inf


def test_empty_cluster_is_rejected():
    """An empty cluster has no unknowns and no background to scatter in."""
    with pytest.raises(ValueError, match="at least one geometry"):
        Cluster([])


def test_band_one_clearly_disjoint_particles_are_accepted():
    """Separated circumscribing circles end the test at band 1, silently.

    Two circles of radius 100 nm whose centres are 500 nm apart cannot touch;
    accepting them is the common case and must cost no warning.
    """
    cluster = Cluster([_circle(60), _circle(60, x0=500.0)])
    assert len(cluster) == 2


def test_band_two_overlapping_inscribed_circles_raise():
    """Interpenetrating circles are rejected before any node is sampled.

    Centres 150 nm apart with radii 100 nm each: the inscribed circles alone
    already overlap, so the cheap band-2 test must catch it. The error names
    both particles, because in a ten-particle cluster the index is the only
    way to find the offender.
    """
    with pytest.raises(ClusterOverlapError, match="particles 0 and 1 overlap"):
        Cluster([_circle(60), _circle(60, x0=150.0)])


def test_band_three_node_inside_the_other_boundary_raises():
    """A crossing missed by both circle tests is caught by node sampling.

    Two perpendicular ellipses (semi-axes 200 × 100 nm, the second rotated a
    quarter turn) with centres 250 nm apart. Their inscribed radii sum to
    200 < 250, so band 2 accepts; their circumscribed radii sum to 400 > 250,
    so band 1 does not accept. Only the ray-cast of band 3 sees that the tips
    interpenetrate — which is why this configuration, and not a pair of
    circles, is the one that exercises it.
    """
    wide = Geometry.gielis(RAD, 120, m=4, b=2.0)
    tall = Geometry.gielis(RAD, 120, m=4, a=2.0, x0=250.0)
    with pytest.raises(ClusterOverlapError, match="boundary node"):
        Cluster([wide, tall])


def test_min_gap_matches_the_analytic_surface_separation():
    """`min_gap` recovers the true gap from above, to the node spacing.

    Two circles of radius 100 nm with centres 300 nm apart have a surface gap
    of exactly 100 nm along the line of centres. The node-to-node minimum can
    only overestimate it: the closest approach generally falls between nodes,
    and at n_pts = 120 the two boundaries each contribute at most
    RAD·(1 − cos(π/n_pts)) ≈ 0.034 nm of sagitta, so 1e-3 relative is a bound
    on that geometric slop and not a fitted tolerance. The one-sidedness is
    checked separately — an underestimate would mean the guard is measuring
    something else entirely.
    """
    cluster = Cluster([_circle(120), _circle(120, x0=300.0)])
    assert cluster.min_gap >= 100.0
    assert cluster.min_gap == pytest.approx(100.0, rel=1e-3)
