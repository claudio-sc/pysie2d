"""Scale covariance: M depends on (rad, λ) only through the size parameter.

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
"""

import numpy as np
import pytest

from pysie2d import BIESolver, Geometry, Material, QNMSolver

N_CORE = 3.0  # the QNM fixture contrast; isolated modes up to Q ≈ 2289
RAD = 200.0
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
# polarisations: worst 4.3e-15 on the uniform-θ path, worst 1.1e-13 on the
# arc-length path.)* The arc-length excess is the fine-grid inversion: one ulp
# of error in `s_uniform` maps into δθ through dθ/ds, so it falls as the fine
# grid densifies (n_fine = 10·nn) — 1.1e-13 at n_pts = 100 down to 3.5e-14 at
# n_pts = 400 — and the bound below is the coarsest case, not the tested one.
#
# The headroom over the measurement is ~3×, for a different BLAS or libm
# rounding, and it is not room to absorb a real failure: a missing power of s
# at s = 1.7 is a 40 % discrepancy, thirteen orders above this floor. There is
# no regime in which widening these would rescue anything.
ATOL_UNIFORM_THETA = 2.0e-14
ATOL_ARC_LENGTH = 3.0e-13

# (m, arc_length, tolerance). The circle is the anchor shape; m = 6 is there
# because the arc-length inversion only does real work on a shape whose
# uniform-θ spacing is non-uniform, and it is the path a rough boundary will
# take.
SHAPES = [
    (0, False, ATOL_UNIFORM_THETA),
    (0, True, ATOL_ARC_LENGTH),
    (6, True, ATOL_ARC_LENGTH),
]


def assemble(rad, lam, m, arc_length, pol):
    """M(λ) for a Gielis boundary of scale radius ``rad``, centred on origin."""
    return BIESolver(
        Geometry.gielis(rad=rad, n_pts=N_PTS, m=m, arc_length=arc_length),
        Material(n_core=N_CORE, n_clad=1.0, pol=pol),
    ).assemble(lam)


@pytest.mark.parametrize("s", EXACT_RATIOS)
@pytest.mark.parametrize("m, arc_length, _atol", SHAPES)
@pytest.mark.parametrize("pol", [1, 2])
def test_matrix_is_bit_identical_under_binary_scaling(s, m, arc_length, _atol, pol):
    """M(s·rad, s·λ) == M(rad, λ) to the last bit, for s a power of two.

    Also pins the two premises separately, so a failure says which one broke:
    the θ-nodes (hence ``delt``) are rad-independent, and the coordinates are
    homogeneous of degree 1. ``n_fine`` in the arc-length inversion depends on
    ``nn`` alone; the natural-looking "improvement" of choosing it from an
    absolute chord length in nm would break the first assertion here.
    """
    g1 = Geometry.gielis(rad=RAD, n_pts=N_PTS, m=m, arc_length=arc_length)
    g2 = Geometry.gielis(rad=s * RAD, n_pts=N_PTS, m=m, arc_length=arc_length)

    assert np.array_equal(np.atleast_1d(g1.delt), np.atleast_1d(g2.delt))
    assert np.array_equal(s * g1.f, g2.f)
    assert np.array_equal(s * g1.dg, g2.dg)
    assert np.array_equal(s * g1.ddf, g2.ddf)

    m1 = assemble(RAD, LAM, m, arc_length, pol)
    m2 = assemble(s * RAD, s * LAM, m, arc_length, pol)
    assert np.array_equal(m1, m2)


@pytest.mark.parametrize("s", INEXACT_RATIOS)
@pytest.mark.parametrize("m, arc_length, atol", SHAPES)
@pytest.mark.parametrize("pol", [1, 2])
def test_matrix_covariance_survives_a_generic_ratio(s, m, arc_length, atol, pol):
    """The identity is well conditioned, not merely exact on binary ratios.

    This is the variant that can fail from cancellation rather than from
    algebra, and it is what calibrates the floor every downstream scale
    argument inherits.
    """
    m1 = assemble(RAD, LAM, m, arc_length, pol)
    m2 = assemble(s * RAD, s * LAM, m, arc_length, pol)
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
# not, so a step landing between tol and s·tol would stop the iteration at
# different points at the two radii. It does not bite on either anchor below
# *(measured: refined wavelengths bit-identical at s = 2 for tol from 1e-9 to
# 1e-2, because the first Newton step already falls decades below any of
# them)*, and these tests deliberately assert on the **unrefined** modes(),
# where the covariance is a theorem rather than a measurement.
# ---------------------------------------------------------------------------

# Anchors and boxes from test_qnm.py, in vacuum nm. Simple and degenerate are
# both tested because the second is where the claim could plausibly weaken:
# rank detection in a two-dimensional null space.
BOX_TE_SIMPLE = (520.0 + 15.0j, 545.0 + 40.0j)
BOX_TE_DEGENERATE = (745.0 + 2.0j, 775.0 + 15.0j)
BOXES = [("simple", BOX_TE_SIMPLE), ("degenerate", BOX_TE_DEGENERATE)]

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
    return {name: qnm_modes(RAD, box, 1.0) for name, box in BOXES}


@pytest.mark.parametrize("s", EXACT_RATIOS)
@pytest.mark.parametrize("name, box", BOXES)
def test_qnm_is_bit_identical_under_binary_scaling(unscaled, s, name, box):
    """λ(s·rad) = s·λ(rad) and Q(s·rad) = Q(rad), to the last bit.

    This is the sharpest statement available: not "dλ/drad agrees with λ/rad to
    some tolerance" but the finite identity it integrates to, at zero
    tolerance. The finite form is strictly stronger — it implies the derivative
    by differentiating at s = 1, while a finite-difference derivative test
    would carry an O(h) truncation and an O(ε/h) cancellation floor around
    1e-8, seven orders worse.

    ``size_parameters`` is the same claim in the analytic anchor's coordinate
    and is the most direct expression of it: the discrete pole sits at a fixed
    x, whatever the radius.
    """
    ref = unscaled[name]
    got = qnm_modes(s * RAD, box, s)

    # A mode within an ulp of a contour edge would make the in-contour filter a
    # knife edge and the comparison meaningless for a trivial reason.
    assert np.all(ref.edge_margin > 0.05)
    assert ref.n_modes == got.n_modes
    assert np.array_equal(ref.multiplicity, got.multiplicity)

    assert np.array_equal(got.wavelengths / s, ref.wavelengths)
    assert np.array_equal(got.size_parameters, ref.size_parameters)
    assert np.array_equal(got.quality_factors, ref.quality_factors)
    # Free, and it says the SVD's phase convention is scale-stable — which any
    # downstream gauge-dependent quantity (an adjoint sensitivity, say) needs.
    assert np.array_equal(got.vectors, ref.vectors)


@pytest.mark.parametrize("s", INEXACT_RATIOS)
@pytest.mark.parametrize("name, box", BOXES)
def test_qnm_scale_covariance_at_a_generic_ratio(unscaled, s, name, box):
    """The same identity where only round-off, not exact arithmetic, protects it.

    The Q tolerance is *derived* from the λ one rather than measured. With
    λ' = λ(1+δ), |δ| ≤ ε and ρ = Re λ / Im λ,

        |ΔQ|/Q ≤ |Δa|/a + |Δb|/b ≤ ε·√(1+ρ²)·(1+ρ)/ρ  →  (1 + 2Q)·ε

    so the amplification factor is exactly 1 + 2Q — 99 on the degenerate anchor
    at Q ≈ 49, which is why dQ/drad = 0 is the more demanding half of the pair
    and must not be given a flat tolerance.

    On the degenerate pair this doubles as a semisimplicity screen. A first-
    order (semisimple) response to a matrix perturbation of size η gives
    δλ = O(η); at an exceptional point, where the ±n partners have coalesced
    and uᴴ(∂M/∂λ)v → 0, it would be O(√η) — √(5e-14) ≈ 2e-7 here. The measured
    8e-16 sits eight orders below that floor, so the pair is semisimple, not
    defective. ``cond_jacobian`` cannot make that distinction: it reports both
    cases as singular.
    """
    ref = unscaled[name]
    got = qnm_modes(s * RAD, box, s)

    assert ref.n_modes == got.n_modes
    rel_lam = np.abs(got.wavelengths / s - ref.wavelengths) / np.abs(ref.wavelengths)
    assert np.all(rel_lam < RTOL_LAM)

    rtol_q = (1.0 + 2.0 * ref.quality_factors) * RTOL_LAM
    rel_q = np.abs(got.quality_factors - ref.quality_factors) / ref.quality_factors
    assert np.all(rel_q < rtol_q)
