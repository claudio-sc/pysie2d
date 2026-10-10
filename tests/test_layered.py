"""Gates G1–G5 of docs/design/layered-spec.md: the half-space primitives.

Every comparison here is against something that is not the code under test: a
closed form (PEC image, legacy Hankel identity), a second quadrature on the real
axis with a different branch rule (``reference.sommerfeld``), or a continuation
that uses no complex path at all.
"""

import warnings

import numpy as np
import pytest
from numpy.polynomial import chebyshev as cheb
from scipy.special import hankel1, hankel2

from pysie2d import Geometry, HalfSpace, InterfaceGapWarning, layered
from pysie2d.reference.sommerfeld import g_ref, legacy_pec_green

PI = np.pi


@pytest.fixture
def pec_by_integral(monkeypatch):
    """Make the *integral* path carry R ≡ ∓1.

    ``eps_rel=None`` bypasses the path (owner decision 5), so the Sommerfeld
    machinery would never meet the PEC closed form. Patching ``fresnel_r`` lets
    the same quadrature that serves a dielectric be checked against it.
    """

    def constant(q, k1, eps_rel, pol):
        return np.full(np.shape(q), -1.0 if pol == 2 else 1.0, dtype=complex)

    monkeypatch.setattr(layered, "fresnel_r", constant)


def _worst(a, b):
    return max(np.abs(x - y).max() / np.abs(y).max() for x, y in zip(a, b, strict=True))


# --- G1: PEC closed form ------------------------------------------------------

# Measured worst 4.0e-14 over 32 (λ, Im λ, pol, R, h) combinations at real and
# complex k; the study's own table has 2e-13 (real) and 1e-14 (complex). 1e-12
# is a decade above the study floor and nine below an O(1e-3) sign or branch bug.
RTOL_PEC = 1e-12


@pytest.mark.parametrize("pol", [2, 1])
@pytest.mark.parametrize("imag", [0.0, 0.05])
@pytest.mark.parametrize("wavelength", [500.0, 1600.0])
@pytest.mark.parametrize("extent,height", [(100, 5), (500, 20)])
def test_path_integral_with_pec_reflection_is_the_image_hankel(
    pec_by_integral, wavelength, imag, pol, extent, height
):
    k = 2 * PI / (wavelength * (1 + 1j * imag))
    z = np.array([2 * height, 2 * height, 2 * height + extent, 2 * height + 4 * extent])
    z = np.append(z, 2 * height + 2 * extent)
    x = np.array([0.0, 2 * extent, 1.4 * extent, 0.0, -2 * extent])
    path = layered.SommerfeldPath.for_wavenumber(
        k, 2.25 + 0j, pol, 2 * extent, 2 * height
    )
    got = layered.reflected_green(path, pol, k, 2.25 + 0j, x, z / 2, 0 * x, z / 2, 0.0)
    exact = layered.reflected_green(None, pol, k, None, x, z / 2, 0 * x, z / 2, 0.0)
    assert _worst(got, exact) < RTOL_PEC


def test_pec_reflection_signs():
    # TE E_y vanishes on a conductor (R = −1); TM H_y has zero normal derivative
    # (R = +1). A swapped convention flips the sign of every reflected term.
    q = np.array([0.0, 0.01])
    assert np.all(layered.fresnel_r(q, 0.01, None, 2) == -1.0)
    assert np.all(layered.fresnel_r(q, 0.01, None, 1) == 1.0)
    # The closed form carries the same sign: G_ind = ∓(i/4)H₀^{(1)}(kρ_img).
    k, rho = 0.01, 40.0
    g = layered.reflected_green(None, 2, k, None, 0.0, 20.0, 0.0, 20.0, 0.0)[0]
    assert g == pytest.approx(-0.25j * hankel1(0, k * rho), rel=1e-14)


def test_equal_permittivity_reflects_nothing():
    # ε_sub = ε_cover leaves no interface: R is exactly zero.
    r = layered.fresnel_r(np.linspace(0, 0.05, 7), 0.01 + 0.0j, 1.0 + 0j, 2)
    assert np.all(r == 0.0)
    r = layered.fresnel_r(np.linspace(0, 0.05, 7), 0.01 + 0.0j, 1.0 + 0j, 1)
    assert np.all(r == 0.0)


def test_alpha_sheet_starts_at_plus_k_and_decays_on_both_tails():
    k = 0.01 - 0.002j
    assert layered.alpha(0.0, k) == pytest.approx(k, abs=1e-18)
    # The tails are α ≈ i|q|: decaying exp(iαZ), on either side.
    for q in (5.0, -5.0):
        assert layered.alpha(q, k).imag > 0


# --- G2: real-axis QUADPACK reference ------------------------------------------

# The independent reference has its own floor at oscillatory lossless-TE points:
# measured 2.3e-10 here against the study's 2e-10, on G only. Everywhere else
# the worst of 112 comparisons is 1.8e-13, so 1e-12 holds there.
RTOL_REF = 1e-12
RTOL_REF_LOSSLESS_TE = 1e-9

REF_CASES = [
    ("glass", 500.0, 2.25),
    ("glass", 1600.0, 2.25),
    ("Si", 1600.0, 12.25),
    ("lossy", 800.0, 2.25 + 1.0j),
    ("Ag", 633.0, -18.3 + 0.48j),
    ("Au", 800.0, -24.1 + 1.5j),
    ("Ag-weak-loss", 900.0, -38.0 + 0.5j),
]


@pytest.mark.parametrize("pol", [2, 1])
@pytest.mark.parametrize(
    "name,wavelength,eps", REF_CASES, ids=[c[0] + str(c[1]) for c in REF_CASES]
)
@pytest.mark.parametrize("extent,height", [(100, 5), (500, 20)])
def test_deformed_path_matches_real_axis_quadrature(
    name, wavelength, eps, pol, extent, height
):
    k = 2 * PI / wavelength
    pts = [
        (0, 2 * height),
        (2 * extent, 2 * height),
        (1.4 * extent, 2 * height + extent),
    ]
    pts.append((0, 2 * height + 4 * extent))
    path = layered.SommerfeldPath.for_wavenumber(k, eps, pol, 2 * extent, 2 * height)
    lossless_te = pol == 2 and np.imag(eps) == 0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # quad's own roundoff notices
        for which in range(3):
            num = np.array(
                [
                    layered.reflected_green(
                        path, pol, k, eps, x, z / 2, 0.0, z / 2, 0.0
                    )[which]
                    for x, z in pts
                ]
            )
            ref = np.array([g_ref(x, z, k, eps, pol, which) for x, z in pts])
            err = np.abs(num - ref).max() / np.abs(ref).max()
            limit = RTOL_REF_LOSSLESS_TE if lossless_te and which == 0 else RTOL_REF
            assert err < limit


# --- G3: complex k, by continuation that uses no complex path --------------------

# Chebyshev interpolation in real k of the G2 reference, evaluated at complex k
# (G is analytic for Re k > 0). Measured worst 7.7e-14 at Q = 10 over six cases
# and three points, with the Chebyshev tail at 2e-15 so the interpolant is
# converged; 1e-12 leaves a decade over the study's 4e-14. At Q = 3 the study's
# floor is 3e-10 (interpolation noise), which is why this gate stays at Q = 10.
RTOL_CONTINUATION = 1e-12
# The hump depth ×1.5 must not move the answer: measured worst 1.4e-14.
RTOL_DEPTH = 1e-12

CONTINUATION_CASES = [
    ("glass", 800.0, 2.25, 2),
    ("glass", 800.0, 2.25, 1),
    ("Si", 1600.0, 12.25, 1),
    ("Ag", 633.0, -18.3 + 0.48j, 1),
    ("Au", 800.0, -24.1 + 1.5j, 1),
    ("Au", 800.0, -24.1 + 1.5j, 2),
]


@pytest.mark.parametrize("name,wavelength,eps,pol", CONTINUATION_CASES)
def test_complex_k_matches_chebyshev_continuation_of_the_real_reference(
    name, wavelength, eps, pol
):
    n = 32
    k_real = 2 * PI / wavelength
    a, b = 0.6 * k_real, 1.4 * k_real
    nodes = np.cos(PI * (np.arange(n) + 0.5) / n)
    ks = 0.5 * (b - a) * nodes + 0.5 * (a + b)
    k = 2 * PI / (wavelength * (1 + 1j / 20.0))  # Q = 10
    u = (k - 0.5 * (a + b)) / (0.5 * (b - a))
    for x, z in [(0.0, 10.0), (150.0, 40.0), (-300.0, 210.0)]:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            vals = np.array([g_ref(x, z, kk, eps, pol) for kk in ks])
        re = cheb.chebfit(nodes, vals.real, n - 1)
        im = cheb.chebfit(nodes, vals.imag, n - 1)
        continued = cheb.chebval(u, re) + 1j * cheb.chebval(u, im)
        path = layered.SommerfeldPath.for_wavenumber(k, eps, pol, max(abs(x), 1.0), z)
        got = layered.reflected_green(path, pol, k, eps, x, z / 2, 0.0, z / 2, 0.0)[0]
        assert abs(got - continued) / abs(continued) < RTOL_CONTINUATION
        deeper = layered.SommerfeldPath.for_wavenumber(
            k, eps, pol, max(abs(x), 1.0), z, depth_scale=1.5
        )
        other = layered.reflected_green(deeper, pol, k, eps, x, z / 2, 0.0, z / 2, 0.0)[
            0
        ]
        assert abs(got - other) / abs(got) < RTOL_DEPTH


# --- G4: the legacy failure, pinned -----------------------------------------------

# Measured: legacy vs −(i/4)H₀^{(2)} ≤ 2.3e-16; deformed vs +(i/4)H₀^{(1)} ≤ 8.6e-15.
# quad was asked for epsrel 1e-13, so 1e-12 is its floor with a decade to spare.
RTOL_LEGACY = 1e-12


@pytest.mark.parametrize("q_factor", [10.0, 3.0])
@pytest.mark.parametrize("x,z", [(0.0, 50.0), (200.0, 100.0), (-400.0, 300.0)])
@pytest.mark.filterwarnings("ignore::scipy.integrate.IntegrationWarning")
def test_legacy_real_axis_integral_at_complex_k_is_the_incoming_wave(
    monkeypatch, q_factor, x, z
):
    # R ≡ 1 here so the closed forms are bare Hankel functions.
    monkeypatch.setattr(
        layered, "fresnel_r", lambda q, k1, e, pol: np.ones(np.shape(q), dtype=complex)
    )
    k = 2 * PI / (800.0 * (1 + 1j / (2 * q_factor)))
    rho = np.hypot(x, z)
    outgoing = 0.25j * hankel1(0, k * rho)
    incoming = -0.25j * hankel2(0, k * rho)
    legacy = legacy_pec_green(x, z, k)
    path = layered.SommerfeldPath.for_wavenumber(k, 2.25 + 0j, 1, max(abs(x), 1.0), z)
    deformed = layered.reflected_green(
        path, 1, k, 2.25 + 0j, x, z / 2, 0.0, z / 2, 0.0
    )[0]

    assert abs(legacy - incoming) / abs(incoming) < RTOL_LEGACY
    # The legacy answer is O(1) away from the outgoing wave (measured 1.1–1.7),
    # so this cannot pass if the two coincide.
    assert abs(legacy - outgoing) / abs(outgoing) > 0.5
    assert abs(deformed - outgoing) / abs(outgoing) < RTOL_LEGACY


# --- G5: plasmon pole and the clearance guard --------------------------------------

AG = -18.3 + 0.48j


def _shadow_box_k(lam_r, lam_i):
    """k sampled along ∂B̄ = [λr_lo, λr_hi] × [0, λi_max] (the shadow box)."""
    lo, hi = lam_r
    edge = np.linspace(0.0, 1.0, 9)
    lam = np.concatenate(
        [
            lo + (hi - lo) * edge,
            hi + 1j * lam_i * edge,
            hi - (hi - lo) * edge + 1j * lam_i,
            lo + 1j * lam_i * (1 - edge),
        ]
    )
    return 2 * PI / lam


def test_plasmon_pole_is_a_zero_of_the_tm_denominator_on_the_codes_own_sheet():
    # The pole is classified with the quadrature's α, so a root that is on the
    # other sheet cannot be mistaken for one that needs clearing.
    for lam_r in np.linspace(600, 900, 7):
        for q_factor in (np.inf, 30, 10, 5, 3):
            k = 2 * PI / (lam_r * (1 + 1j / (2 * q_factor)))
            feats = dict(layered._singularities(k, AG, 1))
            q_sp = feats["plasmon q_sp"]
            den = layered.alpha(q_sp, k) + layered.alpha(q_sp, k * np.sqrt(AG)) / AG
            # Measured ≤ 1e-15·|α₁|; 1e-12 is the sheet test's resolving power.
            assert abs(den) < 1e-12 * abs(layered.alpha(q_sp, k))
            assert q_sp == pytest.approx(k * np.sqrt(AG / (1 + AG)), rel=1e-14)


def test_plasmon_off_the_proper_sheet_is_not_a_singularity_to_clear():
    # −ε₁ < Re ε₂ < 0: the root exists but is not a zero of the code's
    # denominator (|den|/|α₁| ≈ 2 in the study), so there is nothing to clear.
    eps = -0.5 + 0.01j
    k = 2 * PI / 800.0
    assert "plasmon q_sp" not in dict(layered._singularities(k, eps, 1))


def test_for_box_clears_the_plasmon_over_the_silver_shadow_box():
    ks = _shadow_box_k((600.0, 900.0), 900.0 / 6.0)  # Q ≥ 3 over the whole box
    path = layered.SommerfeldPath.for_box(ks, AG, 1, 400.0, 20.0)
    assert path.delta > 0


def test_for_box_raises_when_the_hump_cannot_clear_a_singularity():
    # The guard that must fail: a hump a tenth as deep as the rule gives leaves
    # +k₁ (Im k < 0 at Q = 3) above the path, and the integral would then jump
    # across a cut that Beyn's rank detection does not see.
    ks = _shadow_box_k((600.0, 900.0), 900.0 / 6.0)
    with pytest.raises(ValueError, match=r"clearance C1 fails: \+k₁"):
        layered.SommerfeldPath.for_box(ks, AG, 1, 400.0, 20.0, depth_scale=0.1)


def test_for_box_raises_on_a_high_index_substrate_that_pins_the_hump_low():
    # ε = 100: T = 1.5·k₂ = 15·k₁, so the hump is nearly flat over +k₁.
    ks = _shadow_box_k((600.0, 900.0), 900.0 / 6.0)
    with pytest.raises(ValueError, match="clearance C1 fails"):
        layered.SommerfeldPath.for_box(ks, 100.0 + 0j, 2, 400.0, 20.0)


def test_path_refuses_a_low_q_box():
    # λi/λr > tan 45° rotates the branch points to the imaginary axis (§8).
    k = 2 * PI / (600.0 * (1 + 1.2j))
    with pytest.raises(ValueError, match="tan 45"):
        layered.SommerfeldPath.for_wavenumber(k, 2.25 + 0j, 2, 100.0, 10.0)
    with pytest.raises(ValueError, match="Re k must be positive"):
        layered.SommerfeldPath.for_wavenumber(-0.01 + 0j, 2.25 + 0j, 2, 100.0, 10.0)


def test_precision_warning_when_depth_times_extent_exceeds_the_bound():
    # Q = 3 at k·D ≈ 40: the floor 3·|Im k| overrides the 6/D cap, δ·D > 3·ln 10.
    k = 2 * PI / (600.0 * (1 + 1j / 6.0))
    with pytest.warns(layered.SommerfeldPrecisionWarning, match="digits"):
        layered.SommerfeldPath.for_wavenumber(k, 2.25 + 0j, 2, 4000.0, 20.0)


# --- Blocks --------------------------------------------------------------------------

# GEMM against the PEC closed form, banded: measured 8.5e-16 (M1) and 1.8e-15 (M2)
# relative to the block norm, real and complex k, TE and TM. 1e-13 is the study's
# end-to-end floor (3e-15–1e-13).
RTOL_BLOCKS = 1e-13


def _circle(z0=110.0, n=128, x0=30.0, rad=100.0):
    return Geometry.gielis(rad=rad, n_pts=n, m=0, x0=x0, z0=z0)


@pytest.mark.parametrize("pol", [2, 1])
@pytest.mark.parametrize("imag", [0.0, 0.05])
def test_banded_gemm_with_pec_reflection_equals_the_closed_form_blocks(
    pec_by_integral, pol, imag
):
    k = 2 * PI / (500.0 * (1 + 1j * imag))
    geom = _circle()
    x_c = 0.5 * (geom.f.max() + geom.f.min())
    path = layered.SommerfeldPath.for_wavenumber(
        k, 2.25 + 0j, pol, np.ptp(geom.f), 2 * geom.g.min()
    )
    assert len(path.bands) > 1  # the banding is actually exercised
    got = layered.reflected_blocks(path, pol, k, 2.25 + 0j, geom, geom, 0.0, x_c)
    exact = layered.reflected_blocks(None, pol, k, None, geom, geom, 0.0, x_c)
    for g, e in zip(got, exact, strict=True):
        assert np.abs(g - e).max() / np.abs(e).max() < RTOL_BLOCKS


# Blocks from the GEMM against blocks built node-by-node from the folded cos/sin
# integral, on a lossy metal and a lossless dielectric, with different resolutions
# on the two sides: a different summation of the same path, so it pins the banding
# and the (tgt, src) bookkeeping, not the physics. Measured below 1e-14.
RTOL_GEMM_VS_POINTWISE = 1e-12


@pytest.mark.parametrize("pol,eps", [(2, 2.25 + 0j), (1, AG)])
def test_gemm_blocks_equal_pointwise_blocks_across_two_resolutions(pol, eps):
    k = 2 * PI / (633.0 * (1 + 1j / 20))
    tgt = _circle(z0=110.0, n=96, x0=-150.0, rad=60.0)
    src = _circle(z0=140.0, n=64, x0=120.0, rad=80.0)
    x_c = 0.5 * (tgt.f.min() + src.f.max())
    extent = src.f.max() - tgt.f.min()
    path = layered.SommerfeldPath.for_wavenumber(
        k, eps, pol, extent, tgt.g.min() + src.g.min()
    )
    m1, m2 = layered.reflected_blocks(path, pol, k, eps, tgt, src, 0.0, x_c)
    g, gx, gz = layered.reflected_green(
        path,
        pol,
        k,
        eps,
        tgt.f[:, None],
        tgt.g[:, None],
        src.f[None, :],
        src.g[None, :],
        0.0,
    )
    h = 2 * PI / src.n_pts
    assert np.abs(m2 - h * g).max() / np.abs(m2).max() < RTOL_GEMM_VS_POINTWISE
    ref1 = h * (gx * src.dg[None, :] - gz * src.df[None, :])
    assert np.abs(m1 - ref1).max() / np.abs(m1).max() < RTOL_GEMM_VS_POINTWISE


# The analytic d/dk against a central difference at fixed nodes (the path is
# fixed per call context, so this is the exact derivative of the discrete
# blocks). The difference's own floor is ε/h ≈ 1e-16/1e-5 = 1e-11 plus h²
# truncation ≈ 1e-10·(derivative scale); measured 2.9e-10–9.5e-10 at h = 1e-6|k|
# and the PEC closed form, which has no path, sits at the same level (3.5e-10),
# so the floor is the difference's. 1e-8 is a decade over that.
RTOL_DK = 1e-8


@pytest.mark.parametrize("pol,eps", [(2, 2.25 + 0.3j), (1, AG), (None, None)])
def test_dk_blocks_match_a_central_difference_at_fixed_nodes(pol, eps):
    pol_used = 2 if pol is None else pol
    for imag in (0.0, 0.05):
        k = 2 * PI / (500.0 * (1 + 1j * imag))
        geom = _circle()
        x_c = 0.5 * (geom.f.max() + geom.f.min())
        path = (
            None
            if eps is None
            else layered.SommerfeldPath.for_wavenumber(
                k, eps, pol_used, np.ptp(geom.f), 2 * geom.g.min()
            )
        )
        d = 1e-6 * abs(k)
        up = layered.reflected_blocks(path, pol_used, k + d, eps, geom, geom, 0.0, x_c)
        dn = layered.reflected_blocks(path, pol_used, k - d, eps, geom, geom, 0.0, x_c)
        an = layered.reflected_blocks_dk(path, pol_used, k, eps, geom, geom, 0.0, x_c)
        for a, u, v in zip(an, up, dn, strict=True):
            assert np.abs((u - v) / (2 * d) - a).max() / np.abs(a).max() < RTOL_DK


def test_nodes_at_or_below_the_interface_are_refused():
    k = 2 * PI / 500.0
    geom = _circle(z0=50.0)  # circle of rad 100 dips to z = −50
    path = layered.SommerfeldPath.for_wavenumber(k, 2.25 + 0j, 2, 200.0, 10.0)
    with pytest.raises(ValueError, match="strictly above the interface"):
        layered.reflected_blocks(path, 2, k, 2.25 + 0j, geom, geom, 0.0, 30.0)
    with pytest.raises(ValueError, match="strictly above the interface"):
        layered.reflected_green(path, 2, k, 2.25 + 0j, 0.0, 0.0, 0.0, 10.0, 0.0)


def test_interface_gap_warning_uses_the_local_node_spacing():
    # 64 nodes on a rad-100 circle 5 nm over the interface: spacing 9.8 nm,
    # spacing/(2·gap) ≈ 1 ≫ 0.25. A mean-spacing rule would also fire here; the
    # discriminating case is the second one.
    with pytest.warns(InterfaceGapWarning):
        layered.check_interface_gap([_circle(z0=105.0, n=64, x0=0.0)], 0.0, 2)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        layered.check_interface_gap([_circle(z0=105.0, n=1024, x0=0.0)], 0.0, 2)
    # A flat facet near the interface: a superellipse whose bottom is nearly
    # flat is resolved on average but not where it lies close to the plane.
    flat = Geometry.gielis(rad=100.0, n_pts=128, m=4, n1=8.0, n2=8.0, n3=8.0, z0=105.0)
    with pytest.warns(InterfaceGapWarning):
        layered.check_interface_gap([flat], 0.0, 1)


# --- HalfSpace -------------------------------------------------------------


def test_halfspace_eps_rel_is_background_relative():
    assert HalfSpace(2.25 + 0.1j).eps_rel(1.5) == pytest.approx((2.25 + 0.1j) / 2.25)
    assert HalfSpace.pec().eps_rel(1.33) is None
    assert HalfSpace.pec().is_pec and not HalfSpace(2.25).is_pec


def test_halfspace_rejects_gain_and_the_lossless_plasmon_resonance():
    with pytest.raises(ValueError, match="gain"):
        HalfSpace(2.25 - 0.1j)
    with pytest.raises(ValueError, match="plasmon"):
        HalfSpace(-1.0).eps_rel(1.0)
    with pytest.raises(ValueError, match="plasmon"):
        HalfSpace(-(1.33**2)).eps_rel(1.33)
    # A small loss moves it off the resonance and is accepted.
    assert HalfSpace(-1.0 + 1e-3j).eps_rel(1.0) is not None


def test_halfspace_normalises_signed_zero_to_the_principal_sheet():
    # np.sqrt(complex(-4, -0.0)) is −2j: a silent flip of the substrate's
    # wavenumber to the wrong half-plane.
    assert np.sqrt(complex(-4.0, -0.0)) == -2j
    eps = HalfSpace(complex(-4.0, -0.0)).eps_rel(1.0)
    assert np.sqrt(eps) == 2j


def test_callable_substrate_is_evaluated_once_at_a_real_wavelength():
    calls = []

    def drude(wavelength):
        calls.append(wavelength)
        return -18.3 + 0.48j

    hs = HalfSpace(drude)
    assert hs.eps_rel(1.0, 633.0) == pytest.approx(-18.3 + 0.48j)
    assert calls == [633.0]
    with pytest.raises(ValueError, match="needs a wavelength"):
        hs.eps_rel(1.0)
    with pytest.raises(ValueError, match="real wavelength"):
        hs.eps_rel(1.0, 633.0 + 20.0j)
    with pytest.raises(ValueError, match="gain"):
        HalfSpace(lambda w: 2.0 - 1.0j).eps_rel(1.0, 600.0)
