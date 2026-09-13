"""Tests for the smooth parametrisation θ = w(t) (parametrisation-spec §3-§6).

Each check names the spec measurement it stands for (M-2, M-4 …) and states
what physical or structural property it pins.
"""

import numpy as np
import pytest

from pysie2d.parametrisation import (
    EPS,
    TWOPI,
    Parametrisation,
    _radius_derivatives,
    _speed_curvature,
)

# The §8.2 ladder plus one asymmetric shape; all exponents even (safe shapes).
CIRCLE = {"a": 1.0, "b": 1.0, "m": 4, "n1": 2.0, "n2": 2.0, "n3": 2.0}
MILD = {"a": 1.0, "b": 1.0, "m": 4, "n1": 2.0, "n2": 4.0, "n3": 4.0}
SHARP = {"a": 1.0, "b": 1.0, "m": 4, "n1": 12.0, "n2": 24.0, "n3": 24.0}
SKEW = {"a": 1.3, "b": 0.8, "m": 4, "n1": 6.0, "n2": 12.0, "n3": 4.0}

# Round-off floor of an angle of size 2π. Every θ-level tolerance below is a
# multiple of it, never of 1.
THETA_FLOOR = TWOPI * EPS


def _spectral(vals):
    """First Fourier derivative of a smooth 2π-periodic function on a uniform grid."""
    n = vals.size
    j = np.fft.fftfreq(n, d=1.0 / n)
    return np.real(np.fft.ifft(np.fft.fft(vals) * 1j * j))


# ---------------------------------------------------------------------------
# §3 closed-form derivatives (M-0)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("shape", [SKEW, MILD, SHARP])
def test_radius_derivatives_match_spectral_differentiation(shape):
    """§3.1-3.2: r', r'', r''' in closed form agree with spectral derivatives.

    Independent check: the closed forms come from the product/log chain, the
    reference from the FFT of r itself. They cannot agree by construction.

    Each order is checked against **one** spectral derivative of the order below
    it, not against the k-th derivative of r: differentiating k times in Fourier
    space amplifies round-off by ~K^k, so the direct comparison measures the
    reference's noise (3e-8 at k = 3, spec §3.3) rather than the closed form.
    One order at a time holds every rung at the single-order floor ~K·eps,
    measured 5e-13 here, so a wrong r''' cannot hide inside the tolerance.
    """
    theta = np.linspace(0.0, TWOPI, 4096, endpoint=False)
    r, r1, r2, r3 = _radius_derivatives(theta, 1.0, **shape)
    for lower, closed in ((r, r1), (r1, r2), (r2, r3)):
        ref = _spectral(lower)
        assert np.abs(closed - ref).max() / np.abs(ref).max() < 5e-12


def test_circle_derivatives_vanish_exactly_through_the_even_integer_path():
    """§3.1 implementation note: the 0·∞ term is skipped, not evaluated to nan.

    On the circle n2 = n3 = 2, so C_uuu's first coefficient n2(n2−1)(n2−2) is
    zero while its power factor |c|^{−2} is infinite at c = 0. Evaluated as
    written that is nan; built coefficient-first it is absent. r is constant
    here, so every derivative must be *exactly* zero — a floor, not a tolerance.
    """
    theta = np.linspace(0.0, TWOPI, 4096, endpoint=False)
    _, r1, r2, r3 = _radius_derivatives(theta, 200.0, **CIRCLE)
    assert np.abs(r1).max() == 0.0
    assert np.abs(r2).max() == 0.0
    assert np.abs(r3).max() == 0.0


def test_curvature_of_a_circle_is_one_over_the_radius():
    """§3.3: κ = N/γ³ reduces to 1/rad on the circle, γ to rad.

    Closed-form anchor, exact in theory; relative bound is a few eps for the
    handful of round-off-level operations in √(r²+r'²) and the ratio.
    """
    theta = np.linspace(0.0, TWOPI, 64, endpoint=False)
    gam, kappa = _speed_curvature(*_radius_derivatives(theta, 200.0, **CIRCLE))
    assert np.abs(gam / 200.0 - 1.0).max() < 4.0 * EPS
    assert np.abs(kappa * 200.0 - 1.0).max() < 4.0 * EPS


# ---------------------------------------------------------------------------
# §4 the inversion
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("contrast", [1.0, 1.5])
def test_circle_maps_to_the_identity_through_the_full_path(contrast):
    """M-2: on a circle w(t) = t, at both ends of the contrast range.

    The circle takes the same §4-§5 code path as every star (contract 2): no
    is_circle test, no C = 1 branch. σγ is constant, so T(θ) = θ and the Newton
    root must land on t to round-off.

    Bound 16·eps·2π: round-off at θ ~ 2π is eps·2π = 1.4e-15 and the FFT plus
    Newton accumulate a small multiple of it (spec M-2).
    """
    par = Parametrisation.gielis(
        rad=200.0, **CIRCLE, r_band=(15.0, 15.0 * contrast), n_core=3.0
    )
    nodes = par.nodes(30)
    assert np.abs(nodes.theta - nodes.t).max() <= 16.0 * THETA_FLOOR
    # w' ≡ 1 and w'' ≡ 0 follow from the same statement and are what Kress
    # consumes; checking them separately catches a Jacobian taken from the
    # analytic γ instead of from the truncated series (§4.1).
    assert np.abs(nodes.dw - 1.0).max() <= 16.0 * EPS
    assert np.abs(nodes.ddw).max() <= 16.0 * EPS


@pytest.mark.parametrize("shape", [CIRCLE, MILD, SHARP, SKEW])
def test_the_nodes_invert_the_map_to_round_off(shape):
    """§4.3: T(w(t_j)) = t_j at the round-off floor on every rung.

    This is the defining property of the map and the one Newton is run for;
    the stopping rule (round-off plus exactly two more steps) exists so that
    the residual is at the floor rather than anywhere below a tolerance.
    """
    par = Parametrisation.gielis(rad=200.0, **shape, r_band=(15.0, 22.5), n_core=3.0)
    nodes = par.nodes(90)
    assert np.abs(par._t_of(nodes.theta) - nodes.t).max() <= 16.0 * THETA_FLOOR


@pytest.mark.parametrize("shape", [MILD, SHARP])
def test_the_map_is_strictly_monotone_and_closes_the_circle(shape):
    """§4.1: w is a monotone circle map, so the node order is the boundary order.

    σγ > 0 makes T strictly increasing; a non-monotone w would cross nodes over
    each other and silently reorder the boundary. T(2π) = 2π is the closure that
    makes w 2π-periodic, exact by the normalisation Z = 2π a_0 (floor bound).
    """
    par = Parametrisation.gielis(rad=200.0, **shape, r_band=(15.0, 22.5), n_core=3.0)
    theta = par.nodes(200).theta
    assert np.all(np.diff(theta) > 0.0)
    assert 0.0 < theta[0] and theta[-1] < TWOPI
    assert abs(float(par._t_of(np.array([TWOPI]))[0]) - TWOPI) <= 8.0 * THETA_FLOOR


def test_the_representation_is_shape_intrinsic_not_nn_dependent():
    """§4.2: N_f and K are set by the decay of σγ, and the cap raises.

    A representation that moved with nn would make w a different map at every
    resolution (invariant 13.3). The sharp rung must ask for more than the
    1024 the mild one needs — that ordering is the whole content of a
    shape-intrinsic rule.
    """
    mild = Parametrisation.gielis(rad=200.0, **MILD, r_band=(15.0, 22.5), n_core=3.0)
    sharp = Parametrisation.gielis(rad=200.0, **SHARP, r_band=(15.0, 22.5), n_core=3.0)
    assert sharp.n_fine > mild.n_fine
    assert sharp.n_terms > mild.n_terms
    assert mild.n_terms < mild.n_fine // 2


# ---------------------------------------------------------------------------
# §5 the density
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("shape", [MILD, SHARP, SKEW])
def test_density_residual_is_flat_along_the_boundary(shape):
    """M-6: σγ·w' is constant, i.e. the nodes realise the requested density.

    |dx/dt| = γ(w)·w' and the design says that equals Z/2π·σ… the residual
    below is the statement that the delivered Jacobian and the density that
    defined the map are the same object (§4.1).

    Bound 1e-13: G2 measured 4e-16; the margin covers the sharp rung's K ≈ 1350
    term summation, which accumulates round-off linearly in K.
    """
    par = Parametrisation.gielis(rad=200.0, **shape, r_band=(15.0, 22.5), n_core=3.0)
    nodes = par.nodes(120)
    rho = par._series(nodes.theta) * nodes.dw
    assert np.abs(rho / rho.mean() - 1.0).max() <= 1e-13


def test_contrast_realised_never_exceeds_the_band():
    """§5.2: α_eff Δ < ln C by construction — the band is a bound, no clamp.

    Exact inequality from the rational form α_eff = α lnC/(αΔ + lnC), so the
    comparison is exact up to the round-off of the exponential; a code path
    that clamped instead would sit *at* the bound, not below it.
    """
    contrast = 1.5
    for shape in (MILD, SHARP, SKEW):
        par = Parametrisation.gielis(
            rad=200.0, **shape, r_band=(15.0, 15.0 * contrast), n_core=3.0
        )
        assert 1.0 < par.contrast_realised < contrast
        assert par.alpha_eff <= 0.5


def test_a_circle_grades_nothing_however_wide_the_band():
    """§5.2: Δ = 0 on a circle, so σ ≡ 1 with no `if spread <= 0` branch.

    A circle has no curvature contrast to spend the band on. The realised
    contrast is exactly 1 (exp of an exactly-zero exponent), which is why the
    equality below is not a tolerance.
    """
    par = Parametrisation.gielis(rad=200.0, **CIRCLE, r_band=(15.0, 30.0), n_core=3.0)
    assert par.contrast_realised == 1.0
    assert par.n_terms == 0  # σγ is constant: the series is empty, T(θ) = θ


def test_the_map_is_continuous_in_the_contrast_at_unity():
    """M-7: ‖w(C) − w(1)‖ ∝ ln C as C → 1 — no branch at the reference design.

    A `if contrast <= 1: return ones` branch shows up here as a jump: the ratio
    below would not tend to a constant. α_eff ∝ ln C at leading order, so the
    ratio converges; the 10 % window is the spec's M-7 criterion, and k starts
    at 2 because at C = 1.1 the O(ln²C) term is a genuine 17 % of the signal.
    """

    def theta_of(contrast):
        par = Parametrisation.gielis(
            rad=200.0, **MILD, r_band=(15.0, 15.0 * contrast), n_core=3.0
        )
        return par.nodes(64).theta

    base = theta_of(1.0)
    slopes = []
    for k in range(2, 9):
        contrast = 1.0 + 10.0**-k
        slopes.append(np.linalg.norm(theta_of(contrast) - base) / np.log(contrast))
    limit = slopes[-1]
    assert max(abs(s / limit - 1.0) for s in slopes) < 0.10


def test_nn_from_band_resolves_the_worst_node():
    """§5.4: every node meets R_min, and a wider band or finer R costs more nn.

    R_j = (λ_ref/n_core)/Δs_j with Δs_j = γ w' (2π/nn). The descent loop may
    stop above the estimate but never below the band, so the *worst* node is
    the one that has to be checked.
    """
    r_min, n_core, lam = 15.0, 3.0, 1550.0
    par = Parametrisation.gielis(
        rad=200.0, **MILD, r_band=(r_min, 22.5), n_core=n_core, wavelength_ref=lam
    )
    nodes = par.nodes(par.nn_from_band)
    gam, _ = _speed_curvature(*_radius_derivatives(nodes.theta, 200.0, **MILD))
    r_nodes = (lam / n_core) / (gam * nodes.dw * TWOPI / par.nn_from_band)
    assert r_nodes.min() >= r_min
    assert par.nn_from_band % 2 == 0
    # Twice the radius at the same reference wavelength is twice the boundary
    # to resolve: nn tracks it, which is what makes R a physical band.
    bigger = Parametrisation.gielis(
        rad=400.0, **MILD, r_band=(r_min, 22.5), n_core=n_core, wavelength_ref=lam
    )
    assert bigger.nn_from_band > par.nn_from_band


# ---------------------------------------------------------------------------
# §6 the contract
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("shape", [CIRCLE, MILD, SHARP])
def test_nn_does_not_change_the_map(shape):
    """M-4 / invariant 13.3: nodes(nn) and nodes(3nn) lie on one curve.

    t_j(nn) = t_{3j+1}(3nn), so the two runs ask for the same point of the same
    map. If anything in the construction read nn — the series length, the
    density bandwidth, the fine grid — the two would disagree at the level of
    that dependence, which is orders above round-off.

    The spec asks for bit-for-bit. That is unreachable in floating point and
    not the map's fault: 2π(3j+1.5)/(3nn) and 2π(j+0.5)/nn differ by up to one
    ulp *in the requested t itself* (asserted below so the claim is visible),
    so the comparison is at the round-off floor of θ ~ 2π, with a factor 4 for
    the Newton step that the 1-ulp difference in t propagates through.
    """
    par = Parametrisation.gielis(rad=200.0, **shape, r_band=(15.0, 22.5), n_core=3.0)
    coarse, fine = par.nodes(40), par.nodes(120)
    assert np.abs(coarse.t - fine.t[1::3]).max() <= THETA_FLOOR
    assert np.abs(coarse.theta - fine.theta[1::3]).max() <= 4.0 * THETA_FLOOR
    # w' is a K-term trig sum whose round-off accumulates linearly in K, so the
    # bound is K·eps, not a constant multiple of eps (K = 1352 on the sharp rung).
    assert np.abs(coarse.dw / fine.dw[1::3] - 1.0).max() <= par.n_terms * EPS


def test_nodes_is_pure():
    """§6 contract 3: the same nn returns bit-identical arrays.

    No tolerance: the call reads only frozen state, so anything but exact
    equality means hidden mutable state in the object.
    """
    par = Parametrisation.gielis(rad=200.0, **SKEW, r_band=(15.0, 22.5), n_core=3.0)
    first, second = par.nodes(48), par.nodes(48)
    for name in ("t", "theta", "dw", "ddw"):
        assert np.array_equal(getattr(first, name), getattr(second, name))


def test_the_map_carries_no_absolute_length():
    """M-5 / conventions §9: rad and λ_ref scaled together leave w unchanged.

    The density is built from u = |κ|L/2π, which is dimensionless, and R is a
    ratio, so the only length that can leak in is through nn. Scaling both
    lengths by 7 must therefore reproduce N_f, K and nn exactly and the nodes to
    round-off — the FFT coefficients of a scaled γ are not bit-identical, hence
    16·eps·2π rather than equality.
    """
    kw = {**SKEW, "r_band": (15.0, 22.5), "n_core": 3.0}
    small = Parametrisation.gielis(rad=200.0, wavelength_ref=1550.0, **kw)
    big = Parametrisation.gielis(rad=1400.0, wavelength_ref=7 * 1550.0, **kw)
    assert (big.n_fine, big.n_terms) == (small.n_fine, small.n_terms)
    assert big.nn_from_band == small.nn_from_band
    assert big.alpha_eff == small.alpha_eff
    assert (
        np.abs(big.nodes(60).theta - small.nodes(60).theta).max() <= 16.0 * THETA_FLOOR
    )


def test_the_object_is_frozen():
    """§6 contract 4: the parametrisation is what gets frozen for ∂M/∂p.

    A shape derivative assembles M(p ± h) from *one* map; if a caller could
    mutate it between the two evaluations the O(h) term freezing removes comes
    straight back (invariant 13.3).
    """
    par = Parametrisation.gielis(rad=200.0, **MILD, n_core=3.0)
    with pytest.raises(AttributeError):
        par.alpha_eff = 0.1
