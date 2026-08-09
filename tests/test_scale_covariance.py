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

from pysie2d import BIESolver, Geometry, Material

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
