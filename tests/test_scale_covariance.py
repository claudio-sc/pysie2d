"""Scale covariance: M depends on (rad, λ) only through the ratio ``k_bg·rad``.

That combination is the size parameter of §2.4 **when the boundary is a
circle**; on a Gielis star it is only a dimensionless ratio, since a
non-circular shape has no single physical radius and ``size_parameter`` refuses
to return one.

Every length and every wavenumber in :func:`pysie2d.kernels.assemble_matrix`
appears in one of exactly four combinations, each of total degree zero under
``rad → s·rad``, ``λ → s·λ``:

    k·r                (all off-diagonal Hankel arguments)
    k·delt/(2e)·gamma  (both singular diagonals)
    k²·cij             (c1 and c3 against the boundary cross products)
    deriv/gamma²       (the M1 and M3 diagonals)

``delt`` and the θ-nodes are degree 0 — including on the arc-length path, where
the chord-length arc estimate is *inexact as an arc length but exactly
homogeneous of degree 1 in rad*, and ``np.interp`` is homogeneous of degree 0 in
(query, table) jointly. Covariance needs the homogeneity, not the accuracy.

Hence ``M(s·rad, s·λ) = M(rad, λ)`` **entrywise**, at any ``n_pts``, and
therefore ``λ(s·rad) = s·λ(rad)``, ``dλ/drad = λ/rad`` and ``dQ/drad = 0``.

Two things this is *not*:

- It is not a validation of the physics. The discrete pole sits at a fixed
  ``x_disc(n_pts) ≠ x_Mie``: covariance is exact while the wavelength is still
  wrong in the first decimal. A wholly incorrect ``M`` passes every assertion
  here. ``test_qnm.py`` is what pins the accuracy.
- It is not a statement about degeneracy. Since ``∂M/∂rad = −(λ/rad)·∂M/∂λ``,
  a dilation perturbs the operator only along the trivial λ-direction: the 2×2
  secular problem of the ``±n`` pair reduces to a multiple of the identity, so
  a dilation can never split a degeneracy and this file says nothing about
  what does.

The load-bearing hypothesis is **non-dispersion**: ``ri`` and ``kd`` are
degree 0 only because :class:`pysie2d.material.Material` holds constant indices.
The day dispersion is added these tests fail, and they should — ``dQ/drad = 0``
fails as physics at the same moment.

The algebra holds for any boundary, but the *conditioning* at a scale ratio
with no exact binary representation can fail on a boundary that is not C¹ —
see :func:`test_a_cusped_boundary_can_lose_the_covariance_at_a_generic_ratio`.

What these tests detect, stated so nobody over-reads them: expressions that are
not dimensionally homogeneous, and absolute lengths in nm that are *in range* at
``rad = 200`` under the ratios used here. A hard-coded length below the ~6 nm
point spacing at ``n_pts = 200`` never binds and stays invisible.
"""

import numpy as np
import pytest

from conftest import QNM_N_CORE, RAD
from pysie2d import BIESolver, Geometry, Material, QNMSolver

# Taken from the shared fixture rather than restated: the mode counts asserted
# below are properties of this contrast and this radius, and a local copy would
# let them drift apart silently.
N_CORE = QNM_N_CORE
N_PTS = 200

# An off-anchor wavelength: the matrix identity holds at every λ, and using a
# pole would invite the misreading that this is a statement about modes.
LAM = 530.83214 + 26.37850j

# Powers of two only. In binary floating point fl(2·a ∘ b) = 2·fl(a ∘ b) for
# ∘ ∈ {+,−,×,÷} and fl(√(4x)) = 2·fl(√x), so every intermediate — rad·(…),
# df²+dg², gamma, r, k = 2πn/(s·λ) — scales by exactly s and every degree-zero
# combination comes out bit-identical. Bit-identity here is therefore a
# *theorem*, which is what lets the assertion be np.array_equal with no
# tolerance at all: a missing or extra power of s is a 100 % error against a
# noise floor of exactly zero.
EXACT_RATIOS = (2.0, 0.5)

# Ratios with no exact binary representation, where the identity survives only
# up to round-off. These are the cases that could fail from conditioning —
# catastrophic cancellation in a degree-zero combination — which the exact
# ratios cannot see.
INEXACT_RATIOS = (1.7, 0.37)

# Round-off floor on the relative max entry difference at an inexact ratio.
# *(measured over s ∈ {1.7, 0.37, 3.0, 0.61} × n_pts ∈ {100, 200, 400} × both
# polarisations: worst 4.3e-15 on the uniform-θ path, worst 1.7e-13 on the
# arc-length path.)* The arc-length excess is the fine-grid inversion — one ulp
# of difference between `linspace(0, s·L, nn)` and `s·linspace(0, L, nn)` maps
# into δθ through dθ/ds — and it does not have a clean n_pts law: `n_fine` is
# `max(10·nn, 4096)`, hence 4096 throughout that sweep, so the floor varies
# with the shape rather than with the resolution.
#
# The headroom over the measurement is ~3×, for a different BLAS or libm
# rounding, and it is not room to absorb a real failure: a missing power of s
# at s = 1.7 is a 40 % discrepancy, twelve orders above this floor, and the
# measured sensitivity is a relative error of 5e-13 in λ. There is no regime in
# which widening these would rescue anything.
ATOL_UNIFORM_THETA = 2.0e-14
ATOL_ARC_LENGTH = 5.0e-13

CIRCLE = {"m": 0}
# The examples' star. Note that Geometry.gielis defaults to n1 = n2 = n3 = 2,
# which gives r = rad at *any* m — a circle. A non-circular shape has to name
# its exponents, and this file needs one: the arc-length inversion only does
# real work where the uniform-θ spacing is non-uniform, and that is the path a
# rough boundary will take.
STAR = {"m": 6, "n1": 6.0, "n2": 12.0, "n3": 12.0}
# Superformula exponent 1 puts |cos|-kinks on the boundary, so the tangent is
# discontinuous and the shape is not C¹. See the two tests for what that costs.
CUSPED_STAR = {"m": 6, "n1": 1.0, "n2": 1.0, "n3": 1.0}

# (id, shape kwargs, arc_length, tolerance).
CONDITIONED_SHAPES = [
    ("circle-uniform-theta", CIRCLE, False, ATOL_UNIFORM_THETA),
    ("circle-arc-length", CIRCLE, True, ATOL_ARC_LENGTH),
    ("star-arc-length", STAR, True, ATOL_ARC_LENGTH),
]
# The exact-ratio test carries the cusped star too: the algebra is exact
# whatever the boundary regularity, and only the conditioning is not.
EXACT_SHAPES = CONDITIONED_SHAPES + [
    ("cusped-star-arc-length", CUSPED_STAR, True, None)
]


def _params(rows):
    """pytest.param triples with readable ids, dropping the id column."""
    return [pytest.param(kw, arc, atol, id=name) for name, kw, arc, atol in rows]


def assemble(rad, lam, shape, arc_length, pol):
    """M(λ) for a Gielis boundary of scale radius ``rad``, centred on origin."""
    return BIESolver(
        Geometry.gielis(rad=rad, n_pts=N_PTS, arc_length=arc_length, **shape),
        Material(n_core=N_CORE, n_clad=1.0, pol=pol),
    ).assemble(lam)


@pytest.mark.parametrize("s", EXACT_RATIOS)
@pytest.mark.parametrize("shape, arc_length, _atol", _params(EXACT_SHAPES))
@pytest.mark.parametrize("pol", [1, 2])
def test_matrix_is_bit_identical_under_binary_scaling(s, shape, arc_length, _atol, pol):
    """M(s·rad, s·λ) == M(rad, λ) to the last bit, for s a power of two.

    Also pins the two premises separately, so a failure says which one broke:
    the θ-nodes (hence ``delt``) are rad-independent, and the coordinates are
    homogeneous of degree 1. ``n_fine`` in the arc-length inversion depends on
    ``nn`` alone; the natural-looking "improvement" of choosing it from an
    absolute chord length in nm would break the first assertion here.

    The cusped star passes exactly like the rest: boundary regularity has
    nothing to do with the algebra.
    """
    g1 = Geometry.gielis(rad=RAD, n_pts=N_PTS, arc_length=arc_length, **shape)
    g2 = Geometry.gielis(rad=s * RAD, n_pts=N_PTS, arc_length=arc_length, **shape)

    assert np.array_equal(np.atleast_1d(g1.delt), np.atleast_1d(g2.delt))
    assert np.array_equal(s * g1.f, g2.f)
    assert np.array_equal(s * g1.dg, g2.dg)
    assert np.array_equal(s * g1.ddf, g2.ddf)

    m1 = assemble(RAD, LAM, shape, arc_length, pol)
    m2 = assemble(s * RAD, s * LAM, shape, arc_length, pol)
    assert np.array_equal(m1, m2)


def test_a_cusped_boundary_can_lose_the_covariance_at_a_generic_ratio():
    """The conditioning limit, pinned rather than left as a surprise.

    At exponent 1 the boundary has kinks, where ``dr/dθ`` takes opposite values
    on the two sides and ``_rderiv`` returns whichever side its node fell on.
    A node sitting numerically *on* a kink therefore takes a finite jump in
    ``df, dg`` — and hence in the cross product ``cij`` — from the ulp-level
    node displacement an inexact ratio introduces. The matrix then differs by
    O(1).

    **This is a knife edge, not a property of inexact ratios.** It happens at
    s = 1.7, where ``dg[100]`` flips from +299.999999999999 to −299.999999999997;
    it does not happen at the other ratio this file uses. *(measured on the
    cusped star: 0.264 at s = 1.7 and 0.264 at s = 0.61, against 3.2e-13 at
    s = 0.37 and 1.7e-13 at s = 3.0 — and 1.5e-13 for the smooth star at every
    one of them. Stable in resolution: 0.305 / 0.264 / 0.248 at n_pts =
    100 / 200 / 400.)* The bimodality is why the threshold below is not a close
    call.

    This is a statement about discretising a corner, not about the covariance —
    the same shape is bit-identical at s = 2 above. It is recorded because the
    roughness programme will feed this solver arbitrary boundaries, and one
    that is not C¹ can inherit exactly this.
    """
    m1 = assemble(RAD, LAM, CUSPED_STAR, True, 2)
    m2 = assemble(1.7 * RAD, 1.7 * LAM, CUSPED_STAR, True, 2)
    rel = np.abs(m2 - m1).max() / np.abs(m1).max()
    assert rel > 1.0e-3  # measured 0.264; the smooth shapes hold to 5e-13


@pytest.mark.parametrize("s", INEXACT_RATIOS)
@pytest.mark.parametrize("shape, arc_length, atol", _params(CONDITIONED_SHAPES))
@pytest.mark.parametrize("pol", [1, 2])
def test_matrix_covariance_survives_a_generic_ratio(s, shape, arc_length, atol, pol):
    """The identity is well conditioned, not merely exact on binary ratios.

    This is the variant that can fail from cancellation rather than from
    algebra, and it is what calibrates the floor every downstream scale
    argument inherits.
    """
    m1 = assemble(RAD, LAM, shape, arc_length, pol)
    m2 = assemble(s * RAD, s * LAM, shape, arc_length, pol)
    rel = np.abs(m2 - m1).max() / np.abs(m1).max()
    assert rel < atol


# ---------------------------------------------------------------------------
# The mode-level consequence: dλ/drad = λ/rad and dQ/drad = 0.
#
# Nothing in the Beyn machinery weakens the matrix identity. With the search box
# scaled with the radius, the contour nodes scale by s and the weights by s, so
# A0 → s·A0 and A1 → s²·A1; every decision the extractor makes is taken on a
# *ratio* — sv_ratio = s/s[0], max_gap, cancellation = ‖A0‖/(Σ|w|·‖x‖),
# edge_margin — and is therefore invariant, while the recovered eigenvalues
# scale by exactly s. The probe matrix depends on (n_dim, n_probe, seed) alone,
# so the gauge is the same too. The mode-level claim is exactly as strong as the
# matrix-level one.
#
# The one scale-dependent knob in the QNM path is `QNMResult.refine`'s `tol`,
# which is a step size in absolute nm: `step` scales with s while `tol` does
# not, so a step landing between tol and s·tol stops the iteration at different
# points at the two radii. It does bite *(measured on the simple anchor at
# s = 2: refined wavelengths bit-identical for tol = 1e-9, 1e-7, 1e-6, 1e-4,
# 1e-3, 1e-2 — and differing by 7.1e-15 relative at tol = 1e-5, with both radii
# reporting converged)*. That is a real scale dependence, narrow because a
# quadratically convergent step passes through the marginal band only for a
# thin set of tol. These tests therefore assert on the **unrefined** modes(),
# where the covariance is a theorem rather than a measurement.
# ---------------------------------------------------------------------------

# Boxes from test_qnm.py, in vacuum nm, with the mode count each must return —
# every assertion below is over arrays, so a box that found nothing would pass
# them all vacuously. Simple and degenerate are both tested because the second
# is where the claim could plausibly weaken: rank detection in a
# two-dimensional null space.
BOX_TE_SIMPLE = (520.0 + 15.0j, 545.0 + 40.0j)
BOX_TE_DEGENERATE = (745.0 + 2.0j, 775.0 + 15.0j)
BOXES = [("simple", BOX_TE_SIMPLE, 1), ("degenerate", BOX_TE_DEGENERATE, 2)]

N_SIDE = 6  # as in test_qnm.py: identical modes to 1e-8 nm against the default

# Round-off floor on |λ(s·rad)/s − λ(rad)| / |λ|, at an inexact ratio.
# **Measured, and not derivable**: the map from the matrix floor above to a
# floor on λ is the eigenvalue condition number, a property of the operator
# rather than of the arithmetic. *(measured: worst 7.8e-16 over
# s ∈ {1.7, 0.37} × both anchors.)* The bound below is ~6× that, and the same
# argument as for the matrix applies — a missing power of s is a 40 % error at
# s = 1.7, not a 1e-15 one.
RTOL_LAM = 5.0e-15


def qnm_modes(rad, box, s):
    """Modes of a circle of radius ``rad`` in the box scaled by ``s``.

    Every knob is passed explicitly: a change to a default must not be able to
    turn this into a different test.
    """
    solver = QNMSolver(
        Geometry.gielis(rad=rad, n_pts=N_PTS, m=0, arc_length=True),
        Material(n_core=N_CORE, n_clad=1.0, pol=2),
    )
    return solver.modes(s * box[0], s * box[1], n_quad_per_side=N_SIDE)


@pytest.fixture(scope="module")
def unscaled():
    """The reference spectra at rad = 200 nm, ~1.5 s per box."""
    return {name: qnm_modes(RAD, box, 1.0) for name, box, _ in BOXES}


@pytest.mark.parametrize("s", EXACT_RATIOS)
@pytest.mark.parametrize("name, box, n_modes", BOXES)
def test_qnm_is_bit_identical_under_binary_scaling(unscaled, s, name, box, n_modes):
    """λ(s·rad) = s·λ(rad) and Q(s·rad) = Q(rad), to the last bit.

    This is the sharpest statement available: not "dλ/drad agrees with λ/rad to
    some tolerance" but the finite identity it integrates to, at zero
    tolerance. The finite form is strictly stronger — it implies the derivative
    by differentiating at s = 1, while a finite-difference derivative test
    would carry an O(h) truncation and an O(ε/h) cancellation floor around
    1e-8, seven orders worse.

    Only two of the assertions can independently fail. ``size_parameters`` and
    ``quality_factors`` are built from operands that scale exactly, so at a
    power-of-two ratio bit-equality of λ propagates into both: they are the
    claim restated in the reader's coordinate, not extra detection.
    ``vectors`` is the one that adds information — the SVD's phase gauge.
    """
    ref = unscaled[name]
    got = qnm_modes(s * RAD, box, s)

    # A mode within an ulp of a contour edge would make the in-contour filter a
    # knife edge and the comparison meaningless for a trivial reason.
    assert np.all(ref.edge_margin > 0.05)
    assert ref.n_modes == n_modes and got.n_modes == n_modes
    assert np.array_equal(ref.multiplicity, got.multiplicity)

    assert np.array_equal(got.wavelengths / s, ref.wavelengths)
    assert np.array_equal(got.size_parameters, ref.size_parameters)
    assert np.array_equal(got.quality_factors, ref.quality_factors)
    # Free, and it says the SVD's phase convention is scale-stable — which any
    # downstream gauge-dependent quantity (an adjoint sensitivity, say) needs.
    assert np.array_equal(got.vectors, ref.vectors)


@pytest.mark.parametrize("s", INEXACT_RATIOS)
@pytest.mark.parametrize("name, box, n_modes", BOXES)
def test_qnm_scale_covariance_at_a_generic_ratio(unscaled, s, name, box, n_modes):
    """The same identity where only round-off, not exact arithmetic, protects it.

    The Q tolerance is *derived* from the λ one rather than measured. With
    λ' = λ(1+δ), |δ| ≤ ε and ρ = Re λ / Im λ,

        |ΔQ|/Q ≤ |Δa|/a + |Δb|/b ≤ ε·√(1+ρ²)·(1+ρ)/ρ,   ρ = 2Q

    so the amplification factor tends to 1 + 2Q and equals it to within 1 % at
    these Q (99.01 against 99.0 at Q = 48.93) — a discrepancy in the direction
    that can only make the test fire spuriously, never hide a regression. It is
    99× here, which is why dQ/drad = 0 is the more demanding half of the pair
    and must not be given a flat tolerance.

    A note, not an assertion: a semisimple pair responds to a matrix
    perturbation of size η as δλ = O(η), while at an exceptional point — the
    ±n partners coalesced, uᴴ(∂M/∂λ)v → 0 — it would be O(√η), i.e. ~4e-7
    against the 1.7e-13 matrix floor measured above. The observed 7.8e-16 is
    nine orders below that, which is consistent with a semisimple pair and not
    with a defective one. ``cond_jacobian`` cannot make that distinction, since
    it reports both cases as singular — but this is an argument from a
    round-off floor, not an anchor, and it is not what the assertions test.
    """
    ref = unscaled[name]
    got = qnm_modes(s * RAD, box, s)

    assert ref.n_modes == n_modes and got.n_modes == n_modes
    rel_lam = np.abs(got.wavelengths / s - ref.wavelengths) / np.abs(ref.wavelengths)
    assert np.all(rel_lam < RTOL_LAM)

    rtol_q = (1.0 + 2.0 * ref.quality_factors) * RTOL_LAM
    rel_q = np.abs(got.quality_factors - ref.quality_factors) / ref.quality_factors
    assert np.all(rel_q < rtol_q)
