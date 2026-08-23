"""Guards for the vacuum-wavelength and background-index conventions (§2).

Every other test in the suite runs at ``n_clad = 1.0``, where the vacuum and
background readings of a wavelength coincide and every ``n_clad`` factor is
1. That is exactly why these tests exist: without them nothing in CI can tell
the two conventions apart, and the package documented one while implementing
the other for three releases.
"""

import numpy as np
import pytest

from conftest import N_CLAD, N_CORE, POL_TAG, QNM_N_CORE, RAD, size_parameter
from pysie2d import (
    SHAPE_STEP,
    BIESolver,
    Geometry,
    Material,
    QNMSolver,
    relative_ldos,
    relative_ldos_map,
    self_green,
    wavelength_over_ds,
)
from pysie2d import size_parameter as public_size_parameter
from pysie2d.qnm import _degenerate_groups
from pysie2d.reference import mie
from pysie2d.reference.mie import self_green_cylinder

# The physics is invariant under a common rescaling of the background: what a
# scatterer does is set by the *relative* index m = n_core/n_clad and the size
# parameter x = 2π·n_clad·a/λ_vac, not by n_clad on its own. Holding both fixed
# while changing n_clad must therefore leave every efficiency bit-for-bit
# comparable. 1e-12 is a round-off budget, not a physics tolerance: the two
# runs execute the same arithmetic on inputs that agree to ~1 ulp.
RTOL_SCALE_INVARIANCE = 1.0e-12

# Reference case and its exact rescaling. n_clad 1.0 → 1.3 with n_core scaled by
# the same factor keeps m = 1.5; λ_vac must scale by 1.3 as well, or x picks up
# the factor and the comparison fails on a different size parameter rather than
# on the convention. This is the trap the spec's §2.5 wording left open.
BASE = {"n_clad": 1.0, "n_core": 1.5, "wavelength": 600.0}
SCALED = {"n_clad": 1.3, "n_core": 1.95, "wavelength": 780.0}


@pytest.mark.parametrize("pol", [1, 2])
@pytest.mark.parametrize("epsi_base", [0.0, 0.5])
def test_vacuum_wavelength_scaling(circle, pol, epsi_base):
    # If the solver read its wavelength as a medium wavelength, the two cases
    # would see different size parameters and disagree at the 1e-1 level.
    #
    # The lossy leg (epsi_base = 0.5) is simultaneously the exact test of the
    # absolute-epsi convention: epsi is scaled by n_clad² so that the *relative*
    # permittivity is unchanged. Under the relative reading the scaled case would
    # carry 1.69× the loss of the base case and qabs would differ by ~40 %.
    geom = circle(300)

    def run(case, epsi):
        mat = Material(n_core=case["n_core"], n_clad=case["n_clad"], pol=pol, epsi=epsi)
        return BIESolver(geom, mat).scatter(wavelength=case["wavelength"])

    base = run(BASE, epsi_base)
    scaled = run(SCALED, epsi_base * SCALED["n_clad"] ** 2)

    # The premise of the comparison: identical relative permittivity, identical
    # relative index m, identical size parameter x.
    assert scaled.material.eps == pytest.approx(base.material.eps)
    assert complex(scaled.material.nc) == pytest.approx(complex(base.material.nc))
    assert scaled.size_parameter == pytest.approx(base.size_parameter, rel=1e-15)

    eff_base = base.efficiencies()
    eff_scaled = scaled.efficiencies()
    for key in ("qsca", "qext", "qabs"):
        assert eff_scaled[key] == pytest.approx(
            eff_base[key], rel=RTOL_SCALE_INVARIANCE
        )


@pytest.mark.parametrize("pol", [1, 2])
def test_efficiencies_match_mie_in_cladding(circle, pol):
    # The analytic anchor at n_clad ≠ 1. Mie theory is stated in (x, m), both of
    # which are background-relative, so the same closed form must hold with no
    # extra n_clad factor anywhere. RTOL is the unchanged RTOL_MIE = 5e-3 of
    # test_efficiencies: the discretisation error is a property of the boundary
    # quadrature and does not care what n_clad is.
    geom = circle(300)
    mat = Material(n_core=SCALED["n_core"], n_clad=SCALED["n_clad"], pol=pol)
    result = BIESolver(geom, mat).scatter(wavelength=SCALED["wavelength"])

    ref = mie.efficiencies(result.size_parameter, complex(mat.nc))
    tag = POL_TAG[pol]

    assert result.efficiencies()["qsca"] == pytest.approx(
        ref[f"Q_sca_{tag}"], rel=5.0e-3
    )
    assert result.efficiencies()["qext"] == pytest.approx(
        ref[f"Q_ext_{tag}"], rel=5.0e-3
    )


@pytest.mark.parametrize("pol", [1, 2])
def test_absorbing_particle_in_cladding(circle, pol):
    # The lossy analytic anchor at n_clad ≠ 1. epsi is absolute, so reaching the
    # relative permittivity 2.25 + 0.5j — the case test_efficiencies already
    # validates at n_clad = 1 — takes epsi = 0.5·n_clad². Landing on that same
    # physical case is deliberate: it inherits the justified RTOL_MIE = 5e-3
    # rather than needing a fresh tolerance argument. Under the *relative*
    # reading of epsi this call would instead model 2.25 + 0.845j and miss the
    # reference by ~30 %, far outside the tolerance.
    geom = circle(300)
    epsi_abs = 0.5 * SCALED["n_clad"] ** 2
    mat = Material(
        n_core=SCALED["n_core"], n_clad=SCALED["n_clad"], pol=pol, epsi=epsi_abs
    )
    assert mat.eps == pytest.approx(complex(2.25, 0.5))
    result = BIESolver(geom, mat).scatter(wavelength=SCALED["wavelength"])

    ref = mie.efficiencies(result.size_parameter, complex(mat.nc))
    tag = POL_TAG[pol]

    assert result.efficiencies()["qabs"] > 0.0
    assert result.efficiencies()["qabs"] == pytest.approx(
        ref[f"Q_abs_{tag}"], rel=5.0e-3
    )


@pytest.mark.parametrize("pol", [1, 2])
def test_self_green_in_cladding(pol):
    # The near-field paths — scatter_dipole → line_dipole_rhs, eval_field — have
    # their own wavenumber plumbing, separate from the far-field path the tests
    # above exercise. Without this test a double-applied n_clad in
    # ScatterResult.wnum_bg or in green.relative_ldos_map passes the whole suite.
    #
    # Anchor and tolerance are inherited from test_green's
    # test_self_green_vs_analytic_cylinder: the Graf-addition-theorem closed form
    # at n_pts = 1000, where the near-field quantity converges at first order and
    # reaches 1 %. Neither the anchor nor the convergence rate depends on n_clad.
    geom = Geometry.gielis(rad=RAD, n_pts=1000, m=0)
    mat = Material(n_core=SCALED["n_core"], n_clad=SCALED["n_clad"], pol=pol)
    solver = BIESolver(geom, mat)
    lam = SCALED["wavelength"]
    d = 2.0 * RAD

    s_bie = self_green(solver, lam, d, 0.0)
    s_ref = self_green_cylinder(
        public_size_parameter(geom, mat, lam), complex(mat.nc), mat.wnum_bg(lam), d, pol
    )

    assert s_bie.real == pytest.approx(s_ref.real, rel=1e-2)
    assert s_bie.imag == pytest.approx(s_ref.imag, rel=1e-2)


def test_ldos_map_matches_scalar_ldos_in_cladding():
    # relative_ldos_map builds its own wnum_bg alongside solver.assemble
    # (green.py) — the one site in the package where two wavenumbers must agree
    # by construction rather than by sharing a call. Pinning the batched path to
    # the scalar one at n_clad ≠ 1 is what makes a drift between them visible.
    # Both run the same arithmetic, so they must agree to round-off, not physics.
    geom = Geometry.gielis(rad=RAD, n_pts=300, m=0)
    mat = Material(n_core=SCALED["n_core"], n_clad=SCALED["n_clad"], pol=2)
    solver = BIESolver(geom, mat)
    lam = SCALED["wavelength"]

    pts = np.array([2.0 * RAD, 3.0 * RAD, -2.5 * RAD])
    mapped = relative_ldos_map(solver, lam, pts, np.zeros_like(pts))
    scalar = np.array([relative_ldos(solver, lam, x, 0.0) for x in pts])

    assert np.allclose(mapped, scalar, rtol=1e-12, atol=0.0)


def test_epsi_is_absolute():
    # The defining statement of the convention, as arithmetic: epsi is referred
    # to vacuum exactly like n_core, so both divide by n_clad² on their way into
    # the relative permittivity the operator sees.
    mat = Material(n_core=1.95, n_clad=1.3, epsi=0.5)
    assert mat.eps.real == pytest.approx((1.95 / 1.3) ** 2)
    assert mat.eps.imag == pytest.approx(0.5 / 1.3**2)
    # nc² == eps holds for a passive material; the principal-root form does not
    # reproduce a gain medium (epsi < 0), which is outside the validated scope.
    assert complex(mat.nc) ** 2 == pytest.approx(mat.eps)

    # At n_clad = 1 the absolute and relative readings coincide — which is why
    # the rest of the suite cannot see the difference.
    vac = Material(n_core=1.5, n_clad=1.0, epsi=0.5)
    assert vac.eps == complex(2.25, 0.5)


def test_wnum_bg_carries_the_background_index():
    # The single conversion point. k_bg = 2π·n_clad/λ_vac, so a denser
    # background shortens the wavelength the operator actually sees.
    mat = Material(n_core=1.5, n_clad=1.3)
    assert mat.wnum_bg(600.0) == pytest.approx(2.0 * np.pi * 1.3 / 600.0)
    assert Material(n_core=1.5).wnum_bg(600.0) == pytest.approx(2.0 * np.pi / 600.0)

    # Complex wavelengths (the QNM path) pass through in kind.
    assert isinstance(mat.wnum_bg(600.0 + 20.0j), complex)


def test_primitives_take_a_wavenumber_not_a_wavelength():
    # A regression guard on the layering itself: the primitives must consume
    # their third argument *as a wavenumber*. Asserting the closed form is what
    # makes this discriminating — merely checking that two different arguments
    # give two different answers would also pass if the primitive re-derived a
    # wavelength from it, which is exactly the regression being guarded.
    from pysie2d import plane_wave_rhs

    geom = Geometry.gielis(rad=RAD, n_pts=64, m=0)
    mat = Material(n_core=N_CORE, n_clad=1.3)
    wnum_bg = mat.wnum_bg(600.0)

    rhs = plane_wave_rhs(geom.n_pts, 0.0, wnum_bg, geom.f, geom.g)
    expected = np.exp(1j * wnum_bg * -geom.g)  # θ = 0: fb = f·sin0 − g·cos0
    assert np.allclose(rhs[: geom.n_pts], expected, rtol=0, atol=1e-14)
    assert np.allclose(rhs[geom.n_pts :], 0.0)


@pytest.mark.parametrize("wavelength", [600.0, 600.0 + 40.0j])
def test_assemble_derivative_applies_the_wavelength_chain_factor(circle, wavelength):
    # kernels.assemble_matrix_dwn differentiates with respect to k_bg, and
    # BIESolver.assemble_derivative is the single place the chain factor
    # dk/dλ = −k/λ is applied. Differencing `assemble` — the façade method that
    # takes a vacuum wavelength — is what makes this test see the whole chain:
    # n_clad = 1.3 so a factor of n_clad dropped from k gives a 23 % error, and
    # the minus sign is the difference between 9e-8 and 2.0.
    geom = circle(64)
    mat = Material(n_core=N_CORE, n_clad=1.3, pol=2)
    solver = BIESolver(geom, mat)

    dm = solver.assemble_derivative(wavelength)
    h = 1e-4 * abs(wavelength)
    fd = (solver.assemble(wavelength + h) - solver.assemble(wavelength - h)) / (2.0 * h)

    # Measured 9.0e-8 (real λ) and 9.4e-8 (complex λ); the bound sits ~10×
    # above, and the residual is the O(h²) truncation of the difference, not
    # the derivative. Order 2 in h is asserted in test_kernels.py, on the
    # wavenumber derivative this one wraps.
    assert np.max(np.abs(fd - dm)) / np.max(np.abs(dm)) < 1e-6


def test_size_parameter_is_referred_to_the_cladding(circle):
    geom = circle(64)
    mat = Material(n_core=N_CORE, n_clad=1.3)
    assert public_size_parameter(geom, mat, 600.0) == pytest.approx(
        2.0 * np.pi * 1.3 * RAD / 600.0
    )


def test_size_parameter_matches_fixture(circle):
    # Pins the test-local helper to the shipped one, so the suite cannot drift
    # from the public definition.
    geom = circle(64)
    mat = Material(n_core=N_CORE, n_clad=N_CLAD)
    assert public_size_parameter(geom, mat, 600.0) == pytest.approx(
        size_parameter(600.0)
    )


def test_size_parameter_rejects_non_circular_geometry():
    # x needs a physical radius. Geometry.rad is a Gielis scale parameter, so on
    # a star it is a number with no size-parameter meaning — refuse rather than
    # return it.
    star = Geometry.gielis(rad=RAD, n_pts=200, m=6, n1=1.0, n2=1.0, n3=1.0)
    assert not star.is_circle
    with pytest.raises(ValueError, match="circular"):
        public_size_parameter(star, Material(n_core=N_CORE), 600.0)


def test_frozen_nodes_restore_second_order_convergence():
    """§10: freezing the node set is what makes ∂M/∂p second-order in h.

    The check is the *rate*, not a value. A central difference must fall by 100
    per decade of h; the unfrozen path does not, because re-inverting arc length
    between the two evaluations adds an O(h) term from ``np.interp`` cell
    crossings. Both halves are asserted, and the second is what stops this
    passing by accident — a test that only checked the frozen path would still
    pass if the freeze silently stopped being applied.

    The ratio is asserted above 50 rather than at 100 because the plateau
    carries the next term of the expansion too (O(h⁴), relatively 1e-2 here), so
    a factor-of-two band around the ideal is the honest statement. The unfrozen
    ratio is measured at 2.6, twenty times below the bound, so the two cases are
    not a close call. Steps 1e-3 and 1e-4 sit inside the D12 window and a decade
    clear of the 1e-8 cancellation floor at either end.
    """
    shape = {"m": 4, "b": 1.20}
    lam = 700.0 + 8.0j  # complex λ: the QNM path, not a real fast path
    n_pts = 120  # the rate is n_pts-independent; 120 keeps the test quick

    def deriv(h, theta):
        mats = []
        for sign in (+1.0, -1.0):
            geo = Geometry.gielis(
                RAD, n_pts, **{**shape, "b": shape["b"] + sign * h}, theta=theta
            )
            mats.append(
                BIESolver(geo, Material(n_core=QNM_N_CORE, pol=2)).assemble(lam)
            )
        return (mats[0] - mats[1]) / (2.0 * h)

    frozen = Geometry.gielis(RAD, n_pts, **shape).theta
    for theta, name in ((frozen, "frozen"), (None, "unfrozen")):
        ref = deriv(1.0e-6, theta)
        coarse = np.linalg.norm(deriv(1.0e-3, theta) - ref)
        fine = np.linalg.norm(deriv(1.0e-4, theta) - ref)
        rate = coarse / fine
        if name == "frozen":
            assert rate > 50.0, f"frozen path is not second order: {rate:.1f}"
        else:
            assert rate < 10.0, f"unfrozen path unexpectedly clean: {rate:.1f}"


def test_wavelength_over_ds_counts_the_interior_wavelength(circle):
    """§10: R is referred to n_core, not to vacuum or to the cladding.

    The BIE carries an interior and an exterior kernel and the interior one
    oscillates faster by n_core, so it sets the resolution requirement. Getting
    this wrong overstates the resolution by exactly n_core — a factor of 3 on
    the QNM fixture — which is the difference between a study at 37 points per
    wavelength and one at 12. Checked against the closed form on the circle,
    where Δs = 2π·rad/n_pts exactly, so the assertion carries no discretisation
    error of its own and the tolerance is pure float round-off.
    """
    geo = circle(200)
    mat = Material(n_core=QNM_N_CORE, pol=2)
    lam = 700.0

    ds = 2.0 * np.pi * RAD / geo.n_pts
    # Chord vs arc: the polygon inscribed in a circle is shorter than the arc by
    # 1 − sinc(π/n_pts) ≈ 1.4e-4 relative at n_pts = 200, so R comes out
    # slightly *higher* than the arc-length closed form. rtol covers exactly
    # that and nothing else.
    assert wavelength_over_ds(geo, mat, lam) == pytest.approx(
        lam / QNM_N_CORE / ds, rel=2.0e-4
    )

    # And it is neither the vacuum nor the cladding reading: both omit n_core,
    # so both are 3x larger here. 0.5 separates them with room to spare.
    assert wavelength_over_ds(geo, mat, lam) < 0.5 * lam / ds


def test_wavelength_over_ds_uses_the_real_part_and_the_worst_node(circle):
    """§10: oscillation is set by Re λ, and the coarsest node decides.

    Two conventions in one test because they are the two ways the scalar could
    have been defined otherwise. The decay of a QNM is not oscillation, so a
    large Im λ must not inflate R; and the risk being diagnosed is
    under-resolution, so the largest gap governs rather than the mean.
    """
    geo = circle(200)
    mat = Material(n_core=QNM_N_CORE, pol=2)

    # Im λ of 200 nm against Re λ 700 would move |λ| by 4 %; it must move R by 0.
    assert wavelength_over_ds(geo, mat, 700.0 + 200.0j) == wavelength_over_ds(
        geo, mat, 700.0
    )

    # A frozen node set on a perturbed shape is the case where Δs stops being
    # uniform. R must then track max(Δs), so it can only fall relative to the
    # shape the nodes were equidistributed for.
    base = Geometry.gielis(RAD, 200, m=4, b=1.20)
    stretched = Geometry.gielis(RAD, 200, m=4, b=1.60, theta=base.theta)
    equidistributed = Geometry.gielis(RAD, 200, m=4, b=1.60)

    assert wavelength_over_ds(stretched, mat, 700.0) < wavelength_over_ds(
        equidistributed, mat, 700.0
    )


def test_wavelength_over_ds_flags_an_elongated_shape_as_under_resolved():
    """§10: R is what makes a fixed n_pts mean different things across shapes.

    The point of the criterion. An aspect-3 ellipse has more than twice the
    perimeter of the circle at the same rad, so the same n_pts resolves it half
    as well — 17.5 against 37.1, i.e. inside the 10–15 "cheap" band rather than
    the 30–50 "accurate" one. A global n_pts must therefore be sized against the
    worst shape in the sampled region, which is a statement this test pins in
    numbers rather than in prose.
    """
    mat = Material(n_core=QNM_N_CORE, pol=2)
    circle_r = wavelength_over_ds(Geometry.gielis(RAD, 200, m=0), mat, 700.0)
    ellipse_r = wavelength_over_ds(Geometry.gielis(RAD, 200, m=4, b=3.0), mat, 700.0)

    assert circle_r == pytest.approx(37.1, abs=0.1)
    assert ellipse_r == pytest.approx(17.5, abs=0.1)
    assert ellipse_r < 0.5 * circle_r

    # Restoring the ratio needs n_pts up by the perimeter ratio, not a tweak.
    recovered = wavelength_over_ds(Geometry.gielis(RAD, 426, m=4, b=3.0), mat, 700.0)
    assert recovered == pytest.approx(circle_r, rel=0.02)


def test_sensitivity_step_is_in_the_parameters_own_units():
    """§11: `step` and the returned dλ/dp are in the parameter's own units.

    The convention is that the caller owns the parametrisation, so a
    reparametrisation must show up as the plain chain-rule factor and nothing
    else. Here rad is expressed in nm and in units of 10 nm: the derivative
    must scale by exactly 10, with no hidden normalisation by rad, by λ, or by
    the step. Scale covariance (§9) makes both sides exact, so the tolerance is
    the cancellation floor of the difference quotient (~ε/h = 1e-11) with two
    decades of margin, not a fitted number.
    """
    geom = Geometry.gielis(rad=RAD, n_pts=200, m=0)
    mat = Material(n_core=QNM_N_CORE, n_clad=1.0, pol=2)
    res = QNMSolver(geom, mat).modes(z_lo=520 + 15j, z_hi=545 + 40j, n_quad_per_side=6)
    theta = res.geometry.theta

    def per_nm(delta):
        return Geometry.gielis(rad=RAD + delta, n_pts=200, m=0, theta=theta), mat

    def per_ten_nm(delta):
        return (
            Geometry.gielis(rad=RAD + 10.0 * delta, n_pts=200, m=0, theta=theta),
            mat,
        )

    d_nm = res.sensitivity(per_nm, step=SHAPE_STEP)[0]
    d_ten = res.sensitivity(per_ten_nm, step=SHAPE_STEP)[0]
    assert d_ten == pytest.approx(10.0 * d_nm, rel=1.0e-9)


def test_sensitivity_degenerate_dispatch_agrees_with_multiplicity():
    """§11: the secular branch is taken exactly where `multiplicity` says so.

    Two criteria for "same pole" would eventually disagree, and the branch
    taken would then contradict the multiplicity reported next to it. The
    grouping is asserted against `multiplicity` itself rather than against a
    hard-coded pair count, and the degenerate group must return as many
    derivatives as it has members — a k-fold pole has k of them, not one.
    """
    geom = Geometry.gielis(rad=RAD, n_pts=200, m=0)
    mat = Material(n_core=QNM_N_CORE, n_clad=1.0, pol=2)
    res = QNMSolver(geom, mat).modes(z_lo=745 + 2j, z_hi=775 + 15j, n_quad_per_side=6)
    theta = res.geometry.theta

    groups = _degenerate_groups(res.wavelengths)
    assert [len(g) for g in groups] == [int(res.multiplicity[g[0]]) for g in groups]

    got = res.sensitivity(
        lambda d: (Geometry.gielis(rad=RAD + d, n_pts=200, m=0, theta=theta), mat)
    )
    assert got.shape == res.wavelengths.shape


def test_sensitivity_left_vector_is_not_conj_of_the_right_one():
    """§11: M is not complex-symmetric, so u must come from the SVD's U.

    If M *were* complex-symmetric this convention would be vacuous and
    `conj(v)` would do. The measurement that says otherwise is asserted here so
    the convention text cannot quietly stop being true: ‖M − Mᵀ‖/‖M‖ is O(1),
    not O(1e-14), even though each of the four n_pts blocks is symmetric.
    """
    geom = Geometry.gielis(rad=RAD, n_pts=200, m=0)
    mat = Material(n_core=QNM_N_CORE, n_clad=1.0, pol=2)
    m = BIESolver(geom, mat).assemble(530.83214 + 26.37850j)

    assert np.linalg.norm(m - m.T) / np.linalg.norm(m) > 1.0
    nn = geom.n_pts
    block = m[:nn, :nn]
    assert np.linalg.norm(block - block.T) / np.linalg.norm(block) < 1.0e-13


def test_geometry_without_a_node_set_is_allowed_but_cannot_be_differentiated():
    """§10: `theta` is optional to store, and mandatory to differentiate.

    The two halves are a pair. A `Geometry` assembled from arrays that came
    from somewhere else has no node set to report, and refusing to construct it
    would break a scattering-only user for a reason that has nothing to do with
    scattering — the whole solver runs without ever reading `theta`. But a
    shape derivative *is* the node set (§10), so the frozen-node path must
    refuse such a geometry rather than fall back to anything.

    The failure therefore belongs at the point of use, not at construction, and
    it has to name which of the two geometries is missing: `at` supplies one
    and the result carries the other, and "theta is None" alone does not say
    which one the caller has to fix.
    """
    src = Geometry.gielis(rad=RAD, n_pts=120, m=4, b=1.2)
    bare = Geometry(src.f, src.g, src.df, src.dg, src.ddf, src.ddg, src.delt, rad=RAD)
    assert bare.theta is None

    # The solver itself never reads theta, so a bare geometry scatters exactly
    # like the geometry it was copied from — bit for bit, since the arrays are
    # the same objects and no node set enters the assembly.
    mat = Material(n_core=QNM_N_CORE, n_clad=1.0, pol=2)
    lam = 530.83214 + 26.37850j
    assert np.array_equal(
        BIESolver(bare, mat).assemble(lam), BIESolver(src, mat).assemble(lam)
    )

    res = QNMSolver(src, mat).modes(520 + 5j, 620 + 45j)
    with pytest.raises(ValueError, match="perturbed geometry carries no node set"):
        res.sensitivity(lambda d: (bare, mat))

    # And the mirror: a base result with no node set is refused too, naming the
    # base rather than the perturbed one. Without this branch `None == None`
    # would compare equal and two unrelated discretisations would be accepted.
    bare_res = QNMSolver(bare, mat).modes(520 + 5j, 620 + 45j)
    with pytest.raises(ValueError, match="base geometry carries no node set"):
        bare_res.sensitivity(lambda d: (src, mat))
