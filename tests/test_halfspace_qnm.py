"""Quasi-normal modes over a half-space: G8, the path-independence half of G9,
and the wiring around them.

A QNM needs one holomorphic M(λ) on the whole search box, so the Sommerfeld path
is built once per box. The checks here are that the poles are the right ones
(G8: over PEC against Beyn on the mirror-cluster matrix), that they do not depend
on the path (G9: hump depth ×1.5), and that the analytic dM/dλ is the exact
derivative of the matrix Beyn integrates.
"""

import numpy as np
import pytest

from pysie2d import (
    BIESolver,
    Cluster,
    ClusterBIESolver,
    Geometry,
    HalfSpace,
    Material,
    QNMSolver,
)
from pysie2d.beyn import beyn_modes, newton_refine
from pysie2d.qnm import _box_solver

RAD = 100.0
AG = -18.3 + 0.48j


def _mat(pol):
    return Material(n_core=3.0, n_clad=1.0, pol=pol)


# --- G8: PEC poles equal the mirror cluster's poles ---------------------------------

# Beyn on the half-space matrix against Beyn on the two-cylinder (particle + mirror
# image) matrix. The cluster also carries the modes of the opposite mirror symmetry,
# which a PEC plane excludes, so the half-space poles are a *subset*: each must be
# found in the cluster, and the cluster must have at least one the half-space
# lacks (otherwise the test would pass for a half-space that found everything).
# Measured at nn = 160: 7.8e-13 nm (TE, gap 60), 2.9e-13 nm (TE, gap 150),
# 2.6e-13 nm (TM, gap 60); the bar is 1e-11 nm, 10× the worst.
TOL_PEC_POLE = 1e-11

PEC_CASES = [
    (2, 60.0, 430 + 20j, 520 + 75j),
    (2, 150.0, 430 + 20j, 520 + 75j),
    (1, 60.0, 700 + 20j, 800 + 75j),
    (1, 150.0, 800 + 20j, 900 + 60j),
]


@pytest.mark.parametrize("pol,gap,lo,hi", PEC_CASES)
def test_pec_poles_are_a_subset_of_the_mirror_cluster_poles(pol, gap, lo, hi):
    mat = _mat(pol)
    geo = Geometry.gielis(RAD, 160, m=0, x0=0.0, z0=RAD + gap)
    mirror = Geometry.gielis(RAD, 160, m=0, x0=0.0, z0=-(RAD + gap))
    half = QNMSolver(geo, mat, HalfSpace.pec()).modes(lo, hi, n_probe=12)
    cluster = ClusterBIESolver(Cluster([geo, mirror]), [mat, mat], pol=pol)
    ref = beyn_modes(cluster._assemble, lo, hi, n_quad_per_side=12, n_probe=24)
    assert half.n_modes >= 1
    for lam in half.wavelengths:
        assert np.abs(ref.eigenvalues - lam).min() < TOL_PEC_POLE
    assert ref.eigenvalues.size >= half.n_modes
    # σ_min/σ_max at the pole is the tell for a held-up holomorphy (conventions
    # §13.2): a returned λ that is not a singularity of the integrated operator
    # shows here and nowhere else. Measured ≤ 1.3e-15.
    assert half.sigma_ratio.max() < 1e-12


def test_pec_cluster_has_modes_the_half_space_excludes():
    # Guards the subset test: at gap 60 the TE mirror cluster has a second pole
    # (482.98+50.65i, of the symmetry a conductor forbids) that the half-space
    # must not report.
    mat = _mat(2)
    geo = Geometry.gielis(RAD, 160, m=0, x0=0.0, z0=RAD + 60.0)
    mirror = Geometry.gielis(RAD, 160, m=0, x0=0.0, z0=-(RAD + 60.0))
    lo, hi = 430 + 20j, 520 + 75j
    half = QNMSolver(geo, mat, HalfSpace.pec()).modes(lo, hi, n_probe=12)
    ref = beyn_modes(
        ClusterBIESolver(Cluster([geo, mirror]), [mat, mat], pol=2)._assemble,
        lo,
        hi,
        n_quad_per_side=12,
        n_probe=24,
    )
    assert ref.eigenvalues.size == 2 and half.n_modes == 1


# --- G9, the path-independence half -------------------------------------------------

# Poles with the hump depth × 1.5 against the default, both polished by Newton at
# tol = 1e-9 nm. Measured ≤ 1.8e-13 nm (refined) and ≤ 1.3e-12 nm (contour) over
# silver TE/TM at gaps 100–349 nm; the bar is 10× the refine tolerance, 1e-8 nm,
# as the spec states it, and sits four decades above what was measured.
REFINE_TOL = 1e-9

DEPTH_CASES = [
    (2, 349.33, 440 + 19j, 540 + 78j),
    (2, 100.0, 440 + 19j, 540 + 78j),
    (1, 200.0, 700 + 30j, 950 + 140j),
]


def _refined(geo, mat, background, lo, hi, depth):
    bie = _box_solver(geo, mat, background, lo, hi, depth)
    found = beyn_modes(bie.assemble, lo, hi, n_quad_per_side=12, n_probe=12)
    return [
        newton_refine(
            bie.assemble,
            bie.assemble_derivative,
            lam,
            vec,
            lo,
            hi,
            tol=REFINE_TOL,
            max_iter=30,
        )
        for lam, vec in zip(found.eigenvalues, found.vectors.T, strict=True)
    ]


# The ×1.5 path breaks the δ·D ≤ 6 cap on purpose (δ·D reaches 9–12), which is the
# point of the variation and exactly what SommerfeldPrecisionWarning reports.
@pytest.mark.filterwarnings("ignore::pysie2d.SommerfeldPrecisionWarning")
@pytest.mark.parametrize("pol,gap,lo,hi", DEPTH_CASES)
@pytest.mark.parametrize(
    "background", [HalfSpace(AG, 0.0), HalfSpace(2.25, 0.0)], ids=["silver", "glass"]
)
def test_poles_do_not_depend_on_the_hump_depth(pol, gap, lo, hi, background):
    mat = _mat(pol)
    geo = Geometry.gielis(RAD, 320, m=0, x0=0.0, z0=RAD + gap)
    base = _refined(geo, mat, background, lo, hi, 1.0)
    deep = _refined(geo, mat, background, lo, hi, 1.5)
    assert len(base) == len(deep)
    for a, b in zip(base, deep, strict=True):
        assert a.converged and b.converged
        assert abs(a.eigenvalue - b.eigenvalue) < 10 * REFINE_TOL


# --- the analytic derivative, and the path built once per box -----------------------

# The reflected part of dM/dλ — (with background) − (without), so the free-space
# block cannot dominate the norm and hide it — against a central difference at the
# box's fixed nodes, where the path cannot move between the two evaluations.
# Measured 3.4e-10 at the optimal step h = 1e-3 nm (4e-9 at 1e-2, 3e-9 at 1e-4: the
# truncation and cancellation arms), the same for PEC, which has no path at all.
# The bar is 1e-8, a decade and a half above the floor of the difference itself.
RTOL_DM = 1e-8


@pytest.mark.parametrize("pol", [2, 1])
@pytest.mark.parametrize(
    "background", [HalfSpace(AG, 0.0), HalfSpace.pec()], ids=["silver", "pec"]
)
def test_reflected_dm_dlambda_is_the_derivative_of_the_reflected_matrix(
    pol, background
):
    geo = Geometry.gielis(RAD, 96, m=0, x0=0.0, z0=RAD + 60.0)
    bie = _box_solver(geo, _mat(pol), background, 440 + 19j, 540 + 78j)
    free = BIESolver(geo, _mat(pol))
    lam, h = 489.0 + 39.0j, 1e-3

    def reflected(x):
        return bie.assemble(x) - free.assemble(x)

    fd = (reflected(lam + h) - reflected(lam - h)) / (2 * h)
    an = bie.assemble_derivative(lam) - free.assemble_derivative(lam)
    # The reflected part is not negligible: it is the thing under test.
    assert np.abs(an).max() > 1e-5
    assert np.abs(fd - an).max() / np.abs(an).max() < RTOL_DM


def test_the_path_is_built_once_per_box_and_reused_for_every_wavelength():
    geo = Geometry.gielis(RAD, 96, m=0, x0=0.0, z0=RAD + 60.0)
    bie = _box_solver(geo, _mat(2), HalfSpace(AG, 0.0), 440 + 19j, 540 + 78j)
    path = bie._path
    assert path is not None
    bie.assemble(480.0 + 30.0j)
    bie.assemble(500.0 + 50.0j)
    assert bie._path is path


def test_refine_works_over_a_substrate_and_reports_a_converged_pole():
    geo = Geometry.gielis(RAD, 160, m=0, x0=0.0, z0=RAD + 349.33)
    res = QNMSolver(geo, _mat(2), HalfSpace(AG, 0.0)).modes(440 + 19j, 540 + 78j)
    assert res.n_modes == 1
    polished = res.refine()
    assert polished.converged.all()
    assert abs(polished.wavelengths[0] - res.wavelengths[0]) < 1e-8
    assert polished.background is res.background
    assert polished.sigma_ratio[0] < 1e-12


def test_modes_with_a_substrate_equal_to_the_cladding_are_the_isolated_modes():
    # ε_sub = n_clad²: no interface, no path, and the isolated circle's pole
    # (473.076+43.176i at nn = 64, the spec's own measurement) comes back.
    geo = Geometry.gielis(RAD, 64, m=0, x0=0.0, z0=RAD + 50.0)
    box = (430 + 20j, 520 + 75j)
    plain = QNMSolver(geo, _mat(2)).modes(*box)
    same = QNMSolver(geo, _mat(2), HalfSpace(1.0)).modes(*box)
    assert np.array_equal(plain.wavelengths, same.wavelengths)
    assert abs(plain.wavelengths[0] - (473.076 + 43.176j)) < 1e-3


# --- refusals -----------------------------------------------------------------------


def test_a_callable_substrate_is_refused_because_it_is_not_holomorphic():
    geo = Geometry.gielis(RAD, 64, m=0, x0=0.0, z0=RAD + 50.0)
    with pytest.raises(ValueError, match="holomorphic"):
        QNMSolver(geo, _mat(2), HalfSpace(lambda w: AG))


def test_sensitivity_is_not_available_over_a_substrate():
    geo = Geometry.gielis(RAD, 160, m=0, x0=0.0, z0=RAD + 349.33)
    res = QNMSolver(geo, _mat(2), HalfSpace(AG, 0.0)).modes(440 + 19j, 540 + 78j)
    with pytest.raises(NotImplementedError, match="sensitivity"):
        res.sensitivity(lambda d: (geo, _mat(2)))


def test_a_box_forcing_a_crossing_is_refused_with_the_singularity_named():
    # Q < 0.5 over the box: Im λ/Re λ > 1 rotates the branch points to the
    # imaginary axis (holomorphy §8).
    geo = Geometry.gielis(RAD, 64, m=0, x0=0.0, z0=RAD + 50.0)
    with pytest.raises(ValueError, match="tan 45"):
        QNMSolver(geo, _mat(2), HalfSpace(AG, 0.0)).modes(300 + 10j, 400 + 400j)
