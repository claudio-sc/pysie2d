"""v1.2 slab waveguide: an N-layer Stack background (slab-spec.md §5, gates GW1–GW18).

Each gate's bar is 10× the value measured in the commit that added it, and the
measured number is quoted beside it. Independent anchors: the three-layer
closed form (GW6), the real-axis references of ``reference.sommerfeld`` with a
different R, branch rule and quadrature (GW4, GW5), power balance as an
equality with every term from its own integral (GW9, GW12, GW13), reciprocity
(GW14), the Born limit (GW15) and the exact translation law (GW17).
"""

import numpy as np
import pytest
from scipy.special import hankel1

from pysie2d import (
    BIESolver,
    Geometry,
    HalfSpace,
    Material,
    NoFarFieldError,
    QNMSolver,
    layered,
)
from pysie2d.background import NearDegenerateModesError, Stack
from pysie2d.fields import _stack_background_field
from pysie2d.layered import SommerfeldPath, _singularities
from pysie2d.reference.sommerfeld import (
    g_ref_stack,
    g_ref_stack_gamma,
    gt_ref_stack,
    slab3_roots,
    stack_r_ref,
)
from pysie2d.waveguide import _amplitudes

PI = np.pi
N_CLAD = 1.444
K = 2 * PI * N_CLAD / 1550.0


# --- GW1: the feature-list path is the v0.9 path when given v0.9's features -----


@pytest.mark.parametrize("pol", [2, 1])
@pytest.mark.parametrize("eps_rel", [2.25 / N_CLAD**2, (-18.3 + 0.48j) / N_CLAD**2])
@pytest.mark.parametrize("k", [K, K * (1 - 0.05j)], ids=["real", "complex"])
def test_feature_path_with_half_space_features_is_bit_identical(k, eps_rel, pol):
    # The only change to _build is where the singularities come from; handed
    # _singularities' own list it must reproduce for_wavenumber bit for bit, so
    # every v0.9 rule (T, δ, C1, node counts, banding) is provably unchanged.
    ref = SommerfeldPath.for_wavenumber(k, eps_rel, pol, 900.0, 40.0)
    got = SommerfeldPath.for_features(k, _singularities(k, eps_rel, pol), 900.0, 40.0)
    assert np.array_equal(ref.q, got.q) and np.array_equal(ref.w, got.w)
    assert ref.bands == got.bands and (ref.T, ref.delta) == (got.T, got.delta)


# --- fixtures --------------------------------------------------------------------

N_SI, N_SIN = 3.476, 2.0
E_SI = (N_SI / N_CLAD) ** 2


def _rel(lam, n_clad, layers, n_bot):
    """(k, layer ε_rel, d, ε_bot,rel): the relative stack the primitives take."""
    k = 2 * PI * n_clad / lam
    lay = tuple(complex((n / n_clad) ** 2) for n, _ in layers)
    return k, lay, tuple(float(d) for _, d in layers), complex((n_bot / n_clad) ** 2)


# The five s3 stacks: one, nine modes, a coupled double core, an air cover.
STACKS = {
    "SOI": (1550, 1.444, [(N_SI, 220)], 1.444),
    "air|Si|SiO2": (1550, 1.0, [(N_SI, 220)], 1.444),
    "SiN@800": (800, 1.444, [(N_SIN, 400)], 1.444),
    "Si2000": (1550, 1.444, [(N_SI, 2000)], 1.444),
    "double": (1550, 1.444, [(N_SI, 220), (1.444, 300), (N_SI, 220)], 1.444),
}


def _green(
    k, lay, d, eb, pol, x_sep, z_sum, depth_scale=1.0, extent=1500.0, z_min=20.0
):
    """(G, ∂x'G, ∂z'G) on the stack's feature path, z_sum = z + z' (z_top = 0)."""
    feats = layered._stack_poles(complex(k), lay, d, eb, pol)[1]
    path = SommerfeldPath.for_features(k, feats, extent, z_min, depth_scale)
    r = lambda q: layered.stack_r(q, k, lay, d, eb, pol)  # noqa: E731
    return layered.reflected_green(
        path, pol, k, None, x_sep, z_sum / 2, 0.0, z_sum / 2, 0.0, r
    )


# --- GW2: the T6 reductions are bit-identical at the solve level -----------------


def _particle(pol):
    geom = Geometry.gielis(rad=60.0, n_pts=96, m=4, x0=10.0, z0=120.0)
    return geom, Material(n_core=N_SI, n_clad=N_CLAD, pol=pol)


@pytest.mark.parametrize("pol", [2, 1])
@pytest.mark.parametrize(
    "stack, reference",
    [
        # Every layer equal to the bottom: one HalfSpace at z_top.
        (Stack(((2.25, 100.0), (2.25, 50.0)), 2.25, 0.0), HalfSpace(2.25, 0.0)),
        # Every layer equal to the cover: dropped, the HalfSpace moves to z_bot.
        (Stack(((N_CLAD**2, 30.0),), 2.25, 0.0), HalfSpace(2.25, -30.0)),
        # Everything equal to the cover: no background at all.
        (Stack(((N_CLAD**2, 30.0), (N_CLAD**2, 5.0)), N_CLAD**2, 0.0), None),
    ],
    ids=["layers=bottom", "layers=cover", "all=cover"],
)
def test_stack_reductions_are_bit_identical_to_half_space_and_free(
    stack, reference, pol
):
    # T6 removes r = 0 interfaces exactly, so a stack with nothing left must take
    # the HalfSpace / free code path itself, not a numerically equal one: any
    # recursion or path built for it would differ in the last bits.
    geom, mat = _particle(pol)
    got, ref = BIESolver(geom, mat, stack), BIESolver(geom, mat, reference)
    assert np.array_equal(got.scatter(1550.0, 20.0).ei, ref.scatter(1550.0, 20.0).ei)
    assert np.array_equal(
        got.scatter_dipole(1550.0, 150.0, 60.0).ei,
        ref.scatter_dipole(1550.0, 150.0, 60.0).ei,
    )


# --- GW3: the recursion against the scaled transfer matrix, energy, asymptote -----

# s1(d)'s points. Absolute |ΔR|, as s1 measured it: relative error is the wrong
# measure where R → O(Δk²/q²) at large q (both forms cancel in Y₁ − Y_j, 4e-10
# relative at q = 2600k) and next to a pole, where it is the pole's condition.
Q_OVER_K = np.array([0.5, 2.0, 10.0, 50.0, 200.0])
GW3_STACKS = {
    "SOI": STACKS["SOI"],
    "double": STACKS["double"],
    "air-bottom": (1550, 1.444, [(N_SI, 500)], 1.0),
}


@pytest.mark.parametrize("pol", [2, 1])
@pytest.mark.parametrize("case", GW3_STACKS, ids=str)
def test_recursion_matches_the_scaled_transfer_matrix(case, pol):
    k, lay, d, eb = _rel(*GW3_STACKS[case])
    q = Q_OVER_K * k
    got = layered.stack_r(q + 0j, k, lay, d, eb, pol)
    ref = np.array([stack_r_ref(qi, k, lay, d, eb, pol) for qi in q])
    # Measured 4.3e-14 (double core, TE, q = 2k); spec bar 1e-13.
    assert np.max(np.abs(got - ref)) < 1e-13
    # Large-q asymptote: R → r(cover | layer 1), the rest is e^{−2|Im α|d} ≤ 1e-56.
    top = layered.fresnel_r(q[-2:] + 0j, k, lay[0], pol)
    assert np.max(np.abs(got[-2:] - top)) < 1e-15  # measured 9.7e-17


@pytest.mark.parametrize("pol", [2, 1])
@pytest.mark.parametrize("case", GW3_STACKS, ids=str)
def test_recursion_conserves_energy_at_propagating_q(case, pol):
    # |R|² + (Re Y_b/Y₁)|α₁τ|² = 1 for a lossless stack: R and τ from one
    # recursion, so a wrong τ product or sheet breaks the equality.
    k, lay, d, eb = _rel(*GW3_STACKS[case])
    k_bot = (k * np.sqrt(eb)).real
    q = np.linspace(0.0, 0.999 * min(k, k_bot), 200) + 0j
    r = layered.stack_r(q, k, lay, d, eb, pol)
    t = layered.alpha(q, k) * layered.stack_tau(q, k, lay, d, eb, pol)
    y_bot = layered.alpha(q, k * np.sqrt(eb)) / (1.0 if pol == 2 else eb)
    err = np.abs(
        np.abs(r) ** 2 + (y_bot.real / layered.alpha(q, k).real) * np.abs(t) ** 2 - 1
    )
    assert err.max() < 2e-14  # measured 2.2e-15


@pytest.mark.parametrize("pol", [2, 1])
def test_recursion_is_finite_where_the_naive_transfer_matrix_overflows(pol):
    # Finding 1: SOI at a 5 nm gap runs the tail to q·Σd ≈ 840 > 709, where
    # cos(α_j d_j) itself overflows. The recursion never forms e^{+|Im α|d}.
    k, lay, d, eb = _rel(*STACKS["SOI"])
    feats = layered._stack_poles(complex(k), lay, d, eb, pol)[1]
    path = SommerfeldPath.for_features(k, feats, 300.0, 10.0)
    with np.errstate(all="ignore"):
        naive = np.cos(np.sqrt(k * k * lay[0] - path.q**2) * d[0])
    assert not np.all(np.isfinite(naive))
    assert np.all(np.isfinite(layered.stack_r(path.q, k, lay, d, eb, pol)))


# --- GW4: reflected G against the real-axis reference ----------------------------

PTS = [(0.0, 20.0), (300.0, 20.0), (1500.0, 100.0), (40.0, 400.0)]  # (x_sep, z_sum) nm


@pytest.mark.filterwarnings("ignore::scipy.integrate.IntegrationWarning")
@pytest.mark.parametrize("pol", [2, 1])
@pytest.mark.parametrize("case", STACKS, ids=str)
def test_reflected_green_matches_the_real_axis_reference(case, pol):
    # The reference shares nothing but the formula: transfer-matrix R, the
    # radiation-condition branch, QUADPACK on the real axis, and the guided
    # poles by a symmetric-fold PV plus iπ·Res from its own Cauchy circle.
    k, lay, d, eb = _rel(*STACKS[case])
    poles = tuple(p.real for p in layered._stack_poles(complex(k), lay, d, eb, pol)[0])
    for x_sep, z_sum in PTS:
        g = _green(k, lay, d, eb, pol, x_sep, z_sum)
        for which in (0, 2):
            ref = g_ref_stack(x_sep, z_sum, k, lay, d, eb, pol, poles, which=which)
            # Measured 2.6e-10 (Si 2000 TM); the reference floor (s3: 4e-10).
            assert abs(g[which] - ref) / abs(ref) < 4e-9


def _near_cutoff(n_clad, n_core, n_bot, m, pol, t):
    """A slab with mode m at (β − k_out)/k_out = t exactly, by the closed form."""
    k = 2 * PI * n_clad / 1550.0
    ef, es = (n_core / n_clad) ** 2, (n_bot / n_clad) ** 2
    k_out = max(k, k * np.sqrt(es))
    beta = k_out * (1 + t)
    kap, gc = np.sqrt(k * k * ef - beta**2), np.sqrt(beta**2 - k * k)
    gs = np.sqrt(beta**2 - k * k * es)
    pc, ps = (gc, gs) if pol == 2 else (gc * ef, gs * ef / es)
    d = (m * PI + np.arctan(pc / kap) + np.arctan(ps / kap)) / kap
    return k, (complex(ef),), (float(d),), complex(es), beta


@pytest.mark.filterwarnings("ignore::scipy.integrate.IntegrationWarning")
@pytest.mark.parametrize("pol", [2, 1])
@pytest.mark.parametrize(
    "slab",
    [(1.444, N_SI, 1.444, 1), (1.0, N_SIN, 1.444, 0)],
    ids=["SOI-TE1", "air|SiN"],
)
def test_reflected_green_near_cutoff_matches_the_gamma_reference(slab, pol):
    # A pole 1e-6·k_out past the branch point (finding 2): the γ-reference
    # integrates q ≥ k_out in γ, where the cosh variable swings by O(1).
    k, lay, d, eb, beta = _near_cutoff(*slab, pol, 1e-6)
    poles = tuple(p.real for p in layered._stack_poles(complex(k), lay, d, eb, pol)[0])
    assert min(abs(p - beta) for p in poles) < 1e-14 * beta  # found, not missed
    for x_sep, z_sum in PTS[::2] + PTS[3:]:
        ref = g_ref_stack_gamma(x_sep, z_sum, k, lay, d, eb, pol, poles)
        g = _green(k, lay, d, eb, pol, x_sep, z_sum)[0]
        assert abs(g - ref) / abs(ref) < 8e-9  # measured 6.6e-10


@pytest.mark.filterwarnings("ignore::scipy.integrate.IntegrationWarning")
@pytest.mark.parametrize("pol", [2, 1])
@pytest.mark.parametrize("im_n", [1e-3, 5e-2])
def test_reflected_green_over_a_lossy_core_matches_plain_quadpack(im_n, pol):
    # Loss lifts the poles off the axis (Im β > 0), so the reference needs no
    # PV and no residue: a second quadrature that cannot share a pole error.
    k = 2 * PI * N_CLAD / 1550.0
    lay, d, eb = (complex((N_SI + 1j * im_n) ** 2 / N_CLAD**2),), (220.0,), 1 + 0j
    poles = layered._stack_poles(complex(k), lay, d, eb, pol)[0]
    assert all(p.imag > 0 for p in poles)
    for x_sep, z_sum in PTS[:3]:
        ref = g_ref_stack(
            x_sep, z_sum, k, lay, d, eb, pol, (), extra_pts=tuple(p.real for p in poles)
        )
        g = _green(k, lay, d, eb, pol, x_sep, z_sum)[0]
        assert abs(g - ref) / abs(ref) < 5e-13  # measured 6.1e-14


# --- GW8: complex k and loss: continued poles and path independence --------------


@pytest.mark.parametrize("pol", [2, 1])
@pytest.mark.parametrize(
    "eps_core, k_factor",
    [
        ((N_SI + 1e-3j) ** 2, 1.0),
        ((N_SI + 5e-2j) ** 2, 1.0),
        (N_SI**2, 1 - 0.05j),
        (N_SI**2, 1 - 1j / 6),
    ],
    ids=["Im n=1e-3", "Im n=5e-2", "Q=10", "Q=3"],
)
def test_continued_poles_are_zeros_of_the_dispersion(eps_core, k_factor, pol):
    # No external anchor at complex k (v0.9 G3 style): the continued pole must
    # satisfy D = 0 to round-off, measured relative to D′·β, and sit on the
    # side the physics puts it — Im β > 0 for loss, < 0 for a decaying k.
    k = 2 * PI * N_CLAD / 1550.0 * k_factor
    lay, d, eb = (complex(eps_core / N_CLAD**2),), (220.0,), 1 + 0j
    (beta,) = layered._stack_poles(complex(k), lay, d, eb, pol)[0]
    resid = layered._dispersion(np.array([beta]), k, lay, d, eb, pol, scale_at=beta)[0]
    resid /= layered._cauchy_derivative(beta, k, lay, d, eb, pol) * beta
    assert abs(resid) < 1e-14  # measured 3.7e-16
    assert np.sign(beta.imag) == (1.0 if k_factor == 1.0 else -1.0)


@pytest.mark.parametrize("pol", [2, 1])
def test_reflected_green_at_complex_k_is_independent_of_the_hump_depth(pol):
    # Holomorphy: deforming the path where nothing is crossed cannot change G.
    k = 2 * PI * N_CLAD / 1550.0 * (1 - 0.05j)
    lay, d, eb = (complex(E_SI),), (220.0,), 1 + 0j
    for x_sep, z_sum in PTS[:3]:
        g = _green(k, lay, d, eb, pol, x_sep, z_sum)[0]
        g15 = _green(k, lay, d, eb, pol, x_sep, z_sum, depth_scale=1.5)[0]
        assert abs(g15 - g) / abs(g) < 4e-12  # measured 4.3e-13


def test_qnm_solver_refuses_a_stack():
    geom, mat = _particle(2)
    with pytest.raises(ValueError, match="out of scope"):
        QNMSolver(geom, mat, Stack(((E_SI * N_CLAD**2, 220.0),), N_CLAD**2))


@pytest.mark.parametrize(
    "layers, eps_bot",
    [
        ((), 2.0),
        (((2.0, 0.0),), 2.0),
        (((2.0 - 0.1j, 10.0),), 2.0),
        (((-4.0, 10.0),), 2.0),
    ],
    ids=["no layers", "zero thickness", "gain", "metal layer"],
)
def test_stack_refuses_invalid_layers(layers, eps_bot):
    with pytest.raises(ValueError):
        Stack(layers, eps_bot)


# --- GW6: guided modes against the closed form; the count check ------------------

SLABS3 = {
    "SOI": (1550, 1.444, N_SI, 220, 1.444),
    "air|Si|SiO2": (1550, 1.0, N_SI, 220, 1.444),
    "SiN@800": (800, 1.444, N_SIN, 400, 1.444),
    "Si2000": (1550, 1.444, N_SI, 2000, 1.444),
    "air|SiN300|SiO2": (1550, 1.0, N_SIN, 300, 1.444),
}


@pytest.mark.parametrize("pol", [2, 1])
@pytest.mark.parametrize("case", SLABS3, ids=str)
def test_guided_modes_match_the_three_layer_closed_form(case, pol):
    # κd = mπ + atan(p_c/κ) + atan(p_s/κ) shares nothing with the transfer
    # matrix; one to nine modes, two near cutoff, both polarisations.
    lam, n_clad, n_film, d, n_sub = SLABS3[case]
    modes = Stack(((n_film**2, d),), n_sub**2).guided_modes(lam, pol, n_clad)
    k = 2 * PI * n_clad / lam
    ref = slab3_roots(k, 1.0, (n_film / n_clad) ** 2, (n_sub / n_clad) ** 2, d, pol)
    assert len(modes) == len(ref)
    for m, beta in zip(modes, ref, strict=True):
        assert abs(m.beta - beta) / beta < 5e-15  # measured 2.8e-16
        assert m.n_eff == pytest.approx(beta * lam / (2 * PI), rel=5e-15)
    assert [m.order for m in modes] == list(range(len(modes)))


@pytest.mark.parametrize("pol", [2, 1])
def test_three_counts_agree_on_the_coupled_double_core(pol):
    # No closed form: a q-grid finder, the γ-grid finder and the argument
    # principle must all see the same two supermodes.
    k, lay, d, eb = _rel(*STACKS["double"])
    roots, count = layered.guided_poles(k, lay, d, eb, pol)
    k_out, k_max = layered._window(k, lay, eb)
    q = np.linspace(k_out, k_max, 4001)[1:-1]
    v = layered.stack_dispersion(q + 0j, k, lay, d, eb, pol).real
    assert np.count_nonzero(v[:-1] * v[1:] < 0) == roots.size == count == 2


@pytest.mark.parametrize("pol", [2, 1])
def test_a_mode_1e8_past_cutoff_is_found(pol):
    # S4: a q-grid one cell inside the window misses t ≤ 1e-4; the γ-grid from
    # γ = 0 must not miss t = 1e-8.
    # The count may undercount here (S4: wind_γ = 1 of 2 at t = 1e-8, the box's
    # left edge leaving the code's sheet); only a shortfall of the bracketing
    # against it is an error, so no error may be raised.
    k, lay, d, eb, beta = _near_cutoff(1.444, N_SI, 1.444, 1, pol, 1e-8)
    roots, count = layered.guided_poles(k, lay, d, eb, pol)
    assert count <= roots.size == 2
    assert min(abs(roots - beta)) / beta < 5e-15


@pytest.mark.parametrize("pol", [2, 1])
def test_near_degenerate_supermodes_raise_by_name(pol):
    # Si / 3 µm SiO₂ / Si: Δβ/β ≲ 1e-8, below even the ×100 grid (S4 b), so
    # the pair vanishes from every bracketing while the count still sees it.
    stack = Stack(((N_SI**2, 220.0), (N_CLAD**2, 3000.0), (N_SI**2, 220.0)), N_CLAD**2)
    with pytest.raises(NearDegenerateModesError, match="beat length"):
        stack.guided_modes(1550.0, pol, N_CLAD)


def test_guided_modes_refuse_complex_wavelength_and_loss():
    with pytest.raises(ValueError, match="real wavelength"):
        Stack(((N_SI**2, 220.0),), N_CLAD**2).guided_modes(1550.0 + 1j, 2, N_CLAD)
    with pytest.raises(ValueError, match="lossless"):
        Stack((((N_SI + 1e-3j) ** 2, 220.0),), N_CLAD**2).guided_modes(
            1550.0, 2, N_CLAD
        )


# --- GW7: the residue identity pins the normalisation ----------------------------

GW7_STACKS = {**STACKS, "air|SiN300|SiO2": (1550, 1.0, [(N_SIN, 300)], 1.444)}


@pytest.mark.parametrize("pol", [2, 1])
@pytest.mark.parametrize("case", GW7_STACKS, ids=str)
def test_residue_of_r_is_gamma_over_beta_n(case, pol):
    # Res_β R = γ₁u(z_top)²/(βN): left a Cauchy circle on the recursion, right
    # the transfer-matrix profile and a Gauss–Legendre norm, no R involved. The
    # 1/ε TM weight in N is load-bearing (S8: 74 % off without it).
    lam, n_clad, layers, n_bot = GW7_STACKS[case]
    stack = Stack(tuple((n**2, t) for n, t in layers), n_bot**2)
    k, lay, d, eb = _rel(*GW7_STACKS[case])
    k_out = layered._window(k, lay, eb)[0]
    for m in stack.guided_modes(lam, pol, n_clad):
        r = min(0.3 * (m.beta - k_out), 1e-3 * m.beta)
        t = 2 * PI * np.arange(64) / 64
        z = m.beta + r * np.exp(1j * t)
        res = np.mean(layered.stack_r(z, k, lay, d, eb, pol) * r * np.exp(1j * t))
        assert m.profile(0.0) == 1.0
        assert abs(res - m.gamma / (m.beta * m.norm)) / abs(res) < 4e-13  # 4.2e-14


# --- GW5: transmitted G below the stack against upward shooting ------------------

GW5_STACKS = {k: STACKS[k] for k in ("SOI", "air|Si|SiO2", "double", "SiN@800")}


@pytest.mark.filterwarnings("ignore::scipy.integrate.IntegrationWarning")
@pytest.mark.parametrize("pol", [2, 1])
@pytest.mark.parametrize("case", GW5_STACKS, ids=str)
def test_transmitted_green_below_the_stack_matches_upward_shooting(case, pol):
    # The forward T = aR + b cancels catastrophically on the tail (finding 4);
    # the reference shoots T upward from the bottom, a second quadrature with a
    # different T, branch rule and pole treatment.
    k, lay, d, eb = _rel(*GW5_STACKS[case])
    poles = tuple(p.real for p in layered._stack_poles(complex(k), lay, d, eb, pol)[0])
    z_bot = -sum(d)
    for x_sep, hs, hb in [
        (0.0, 20.0, 30.0),
        (300.0, 50.0, 10.0),
        (1500.0, 100.0, 200.0),
    ]:
        tr = layered.Transmission.build_stack(
            pol, k, (lay, d, eb, 0.0), [0.0], [hs], [x_sep], [z_bot - hb]
        )
        g = tr.green(x_sep, z_bot - hb, 0.0, hs)[0]
        ref = gt_ref_stack(x_sep, hs, hb, k, lay, d, eb, pol, poles)
        assert abs(g - ref) / abs(ref) < 4e-10  # measured 1.8e-11


# --- GW11: the in-layer field ------------------------------------------------------

GW11_STACKS = {k: STACKS[k] for k in ("SOI", "double", "air|Si|SiO2")}


def _ds(fn, s0, rho=1.0, n=16):
    """d/ds by a Cauchy circle in complex depth: W is entire in s.

    16 points at radius 1 nm: truncation (|α|·ρ)^16/16! is nil, and round-off
    is ε·|W|/(ρ|α|) ~ 1e-14 relative.
    """
    t = 2 * PI * np.arange(n) / n
    v = np.array([fn(s0 + rho * np.exp(1j * ti)) for ti in t])
    return np.mean(v * np.exp(-1j * t)[:, None], axis=0) / rho


@pytest.mark.parametrize("pol", [2, 1])
@pytest.mark.parametrize("case", GW11_STACKS, ids=str)
def test_layer_field_satisfies_the_interface_conditions(case, pol):
    # u and p·∂z u continuous across every interface, each side from its own
    # formula — layer j at its bottom, layer j+1 (or the τ bottom wave) at its
    # top — on real and complex q; and at z_top the layer formula is the cover's
    # 1 + R per unit downward amplitude.
    k, lay, d, eb = _rel(*GW11_STACKS[case])
    q = np.linspace(0.01, 6.0, 7) * k + 0j
    q = np.concatenate([q, q - 0.3j * k])
    p = [1.0, *([1.0] * len(lay) if pol == 2 else [1 / e for e in lay])]
    p.append(1.0 if pol == 2 else 1 / eb)
    a1 = layered.alpha(q, k)
    r = layered.stack_r(q, k, lay, d, eb, pol)

    def w(s, j):
        return (
            a1 * layered.stack_layer_factor(q, k, lay, d, eb, pol, np.array([s]), j)[0]
        )

    assert np.max(np.abs(1 + r - w(0.0, 0)) / np.abs(1 + r)) < 3e-14  # measured 6.0e-16
    sides = [(1 + r, -1j * a1 * (1 - r))]  # cover at z_top: u and p·∂z u (p = 1)
    for j in range(len(lay)):
        top = (w(0.0, j), -_ds(lambda s, j=j: w(s, j), 0.0) * p[j + 1])
        bot = (w(d[j], j), -_ds(lambda s, j=j: w(s, j), d[j]) * p[j + 1])
        sides += [top, bot]
    tau = a1 * layered.stack_tau(q, k, lay, d, eb, pol)
    sides.append((tau, -1j * layered.alpha(q, k * np.sqrt(eb)) * tau * p[-1]))
    for (ua, da), (ub, db) in zip(sides[::2], sides[1::2], strict=True):
        # Measured 2.6e-13 worst (double core TM), the Cauchy derivative's floor.
        assert np.max(np.abs(ua - ub) / np.abs(ua)) < 3e-12
        assert np.max(np.abs(da - db) / np.abs(da)) < 3e-12


@pytest.mark.parametrize("pol", [2, 1])
def test_core_field_far_along_the_stack_is_the_guided_sum(pol):
    # The public incident field inside the core, 30 µm from a line source, must
    # be the residue sum Σ (i/2βN)·u(z)u(z_s)·e^{iβX} (S9): radiation and lateral
    # waves decay like X^{−3/2}. This wires _layer_field end to end.
    stack = Stack(((N_SI**2, 220.0),), N_CLAD**2)
    geom = Geometry.gielis(rad=20.0, n_pts=64, m=4, x0=-3000.0, z0=200.0)
    res = BIESolver(geom, Material(n_core=1.5, n_clad=N_CLAD, pol=pol), stack)
    res = res.scatter_dipole(1550.0, 0.0, 50.0)
    zc, x_far = np.array([-60.0, -110.0]), 3e4
    u = res.incident_field(np.full(2, x_far), zc)
    total, scale = 0.0, 0.0
    for m in stack.guided_modes(1550.0, pol, N_CLAD):
        t = 1j / (2 * m.beta * m.norm) * m.profile(zc) * m.profile(50.0)
        total, scale = total + t * np.exp(1j * m.beta * x_far), scale + np.abs(t)
    assert np.max(np.abs(u - total) / scale) < 0.05  # measured 1.6e-2 (TM)


def test_fields_refuse_interface_planes_and_lossy_far_field_below():
    stack = Stack(((N_SI**2, 220.0),), (N_CLAD + 0.01j) ** 2)
    geom, mat = _particle(2)
    res = BIESolver(geom, mat, stack).scatter(1550.0)
    with pytest.raises(ValueError, match="interface plane"):
        res.eval_field(np.array([0.0]), np.array([-220.0]))
    with pytest.raises(NoFarFieldError):
        res.far_field_down()
    assert {"c_sca_up", "c_sca_down", "c_abs"} <= res.cross_sections().keys()


# --- shared particle cases (s12) ---------------------------------------------------


def _shifted(geom, dz, rot_deg=0.0):
    """Rotate about the origin, then lift by dz: an x-asymmetric shape (s12)."""
    c, s = np.cos(np.deg2rad(rot_deg)), np.sin(np.deg2rad(rot_deg))

    def rot(x, z):
        return c * x - s * z, s * x + c * z

    f, g = rot(geom.f, geom.g)
    df, dg = rot(geom.df, geom.dg)
    ddf, ddg = rot(geom.ddf, geom.ddg)
    return Geometry(f=f, g=g + dz, df=df, dg=dg, ddf=ddf, ddg=ddg, rad=geom.rad, z0=dz)


SOI = Stack(((N_SI**2, 220.0),), N_CLAD**2)
SIN = Stack(((N_SIN**2, 400.0),), N_CLAD**2)
CIRC = Geometry.gielis(rad=80.0, n_pts=160, m=4)
TRI = Geometry.gielis(rad=90.0, n_pts=200, m=3, n1=4.0, n2=6.0, n3=6.0)
PARTICLES = {
    # (stack, λ, geometry, n_core, epsi, dipole position)
    "SOI circle": (SOI, 1550.0, _shifted(CIRC, 110.0), N_SI, 0.0, (150.0, 60.0)),
    "SOI lossy circle": (SOI, 1550.0, _shifted(CIRC, 110.0), N_SI, 0.8, (150.0, 60.0)),
    "SOI lossy triangle": (
        SOI,
        1550.0,
        _shifted(TRI, 140.0, 17.0),
        2.0,
        0.3,
        (-170.0, 50.0),
    ),
    "SiN triangle, 2 modes": (
        SIN,
        800.0,
        _shifted(TRI, 140.0, 17.0),
        N_SI,
        0.0,
        (-170.0, 50.0),
    ),
}


def _solver(case, pol):
    stack, lam, geom, n, epsi, src = PARTICLES[case]
    mat = Material(n_core=n, n_clad=N_CLAD, pol=pol, epsi=epsi)
    return BIESolver(geom, mat, stack), lam, src


# --- GW9: dipole power balance with no particle --------------------------------------

GW9_STACKS = {
    "SOI": (SOI, N_CLAD, 1550.0),
    "air|Si|SiO2": (SOI, 1.0, 1550.0),
    "SiN@800": (SIN, N_CLAD, 800.0),
    "double": (
        Stack(((N_SI**2, 220.0), (N_CLAD**2, 300.0), (N_SI**2, 220.0)), N_CLAD**2),
        N_CLAD,
        1550.0,
    ),
}


@pytest.mark.parametrize("pol", [2, 1])
@pytest.mark.parametrize("case", GW9_STACKS, ids=str)
def test_line_source_power_balance_over_a_bare_stack(case, pol):
    # 1/4 + Im G_refl(r₀, r₀) on the deformed path must equal P_up + P_down +
    # Σ P_guided, each from its own integral: the real-axis cover and bottom
    # fluxes and the mode profiles' norms. An equality, not a bound (G14).
    stack, n_clad, lam = GW9_STACKS[case]
    k = 2 * PI * n_clad / lam
    rel = stack.eps_rel(n_clad, lam)
    modes = stack.guided_modes(lam, pol, n_clad)
    for h0 in (20.0, 100.0, 400.0):
        refl = layered.StackReflection.build(
            pol, k, rel, np.array([0.0]), np.array([h0])
        )
        p_src = 0.25 + refl.green(0.0, h0, 0.0, h0)[0].imag
        amps = _amplitudes(
            [], k, pol, rel, modes, 0.0, "dipole", (0.0, h0, None), p_src
        )
        assert amps.balance() < 5e-14  # measured 7.0e-15


# --- GW10: the far field along the stack is the guided closed form ------------------

# SOI TM is left out: the symmetric slab's TM remainder decays as X^{-0.84}, not
# the X^{-3/2} radiation tail (S9: 1.0e-1 at 40 µm), so 40 µm is not far enough.
GW10_CASES = [("SOI", 2), ("air|Si|SiO2", 2), ("air|Si|SiO2", 1)]
GW10_CASES += [("SiN@800", 2), ("SiN@800", 1), ("double", 2), ("double", 1)]


@pytest.mark.parametrize("case, pol", GW10_CASES, ids=str)
def test_far_field_along_the_stack_is_the_guided_sum(case, pol):
    # The total line-source field at ±x on the path must tend to
    # Σ s_±·ũ(z)e^{±iβ(x − x_ref)}, with s_± as mode_amplitudes reports them:
    # this pins the amplitude, its normalisation and the phase reference.
    stack, n_clad, lam = GW9_STACKS[case]
    k = 2 * PI * n_clad / lam
    rel = stack.eps_rel(n_clad, lam)
    modes = stack.guided_modes(lam, pol, n_clad)
    amps = _amplitudes([], k, pol, rel, modes, 0.0, "dipole", (0.0, 50.0, None), 1.0)
    for sign in (1, -1):
        rem = []
        for x_far in (1e4, 4e4):
            u = _stack_background_field(
                np.array([sign * x_far]),
                np.array([100.0]),
                k,
                pol,
                "dipole",
                0.0,
                (0.0, 50.0),
                rel,
            )[0]
            s = amps.s_plus if sign > 0 else amps.s_minus
            terms = [
                s[i]
                * m.profile(100.0)
                / np.sqrt(m.beta * m.norm)
                * np.exp(1j * m.beta * x_far)
                for i, m in enumerate(modes)
            ]
            scale = sum(abs(t) for t in terms)
            rem.append(abs(u - sum(terms)) / scale)
            assert abs(u + sum(terms)) / scale > 1.0  # a sign error cannot pass
        assert rem[1] < 0.1 and rem[1] < rem[0]  # measured ≤ 4.9e-2 at 40 µm


# --- GW12–GW14: a particle over the stack ---------------------------------------------


@pytest.mark.parametrize("pol", [2, 1])
@pytest.mark.parametrize("case", PARTICLES, ids=str)
def test_dipole_and_particle_power_balance(case, pol):
    # P_source = 1/4 + Im G_refl + Im u_s at the source, against cover, bottom,
    # absorption and both guided directions — each from its own integral, with
    # loss in the particle and an x-asymmetric shape.
    solver, lam, src = _solver(case, pol)
    amps = solver.scatter_dipole(lam, *src).mode_amplitudes(0.0)
    assert amps.balance() < 1e-14  # measured 4.0e-15 (SiN triangle TE)


@pytest.mark.parametrize("pol", [2, 1])
@pytest.mark.parametrize("case", PARTICLES, ids=str)
def test_mode_incidence_conserves_power_and_is_reciprocal(case, pol):
    # Σ|S|² + P_rad + P_abs = 1 for every incident mode and direction (GW13), and
    # t_{n←m}(L→R) = t_{m←n}(R→L) (GW14): reciprocity holds for a lossy,
    # x-asymmetric particle, where no symmetry could make it pass by accident.
    solver, lam, _ = _solver(case, pol)
    n_modes = len(solver.background.guided_modes(lam, pol, N_CLAD))
    s = {}
    for m in range(n_modes):
        for d in (1, -1):
            s[m, d] = solver.scatter_mode(lam, m, d, x_ref=37.0).mode_amplitudes(37.0)
            assert s[m, d].balance() < 5e-15  # measured 2.4e-15
    for m in range(n_modes):
        for n in range(n_modes):
            assert abs(s[m, 1].s_plus[n] - s[n, -1].s_minus[m]) < 2e-15  # 1.1e-16


# --- GW15: the Born limit -------------------------------------------------------------


@pytest.mark.parametrize(
    "stack, lam, bar", [(SOI, 1550.0, 1e-3), (SIN, 800.0, 2.4e-3)], ids=["SOI", "SiN"]
)
def test_small_circle_s_matrix_tends_to_first_born(stack, lam, bar):
    # u_s ≈ k²(ε − 1)πa²·G(r, r_p)·u_inc(r_p) with the full stack G; its guided
    # part fixes S − δ in closed form. The error must fall like a² (ratio ≈ 4
    # per halving): a wrong formula would have to share both law and constant.
    k, eps, (xp, zp) = 2 * PI * N_CLAD / lam, (2.0 / N_CLAD) ** 2, (23.0, 60.0)
    modes = stack.guided_modes(lam, 2, N_CLAD)
    errs = []
    for a in (4.0, 2.0):
        geom = Geometry.gielis(rad=a, n_pts=64, m=4, x0=xp, z0=zp)
        solver = BIESolver(geom, Material(n_core=2.0, n_clad=N_CLAD, pol=2), stack)
        err = 0.0
        for m in modes:
            amps = solver.scatter_mode(lam, m.order, 1, x_ref=0.0).mode_amplitudes(0.0)
            for n, mn in enumerate(modes):
                pre = k * k * (eps - 1) * PI * a * a * 1j / (2 * mn.beta * mn.norm)
                pre *= mn.profile(zp) * m.profile(zp) * np.exp(1j * m.beta * xp)
                pre *= np.sqrt(mn.beta * mn.norm / (m.beta * m.norm))
                fwd = amps.s_plus[n] - (n == m.order)
                err = max(err, abs(fwd - pre * np.exp(-1j * mn.beta * xp)) / abs(pre))
                err = max(
                    err,
                    abs(amps.s_minus[n] - pre * np.exp(1j * mn.beta * xp)) / abs(pre),
                )
        errs.append(err)
    assert errs[1] < bar  # measured 5.0e-4 (SOI), 1.2e-3 (SiN)
    assert 3.2 < errs[0] / errs[1] < 4.2  # measured 3.43–3.66


# --- façade behaviour ---------------------------------------------------------------


def test_mode_outputs_refuse_what_is_not_a_channel():
    solver, lam, src = _solver("SOI circle", 2)
    with pytest.raises(ValueError, match="direction"):
        solver.scatter_mode(lam, 0, 0, x_ref=0.0)
    res = solver.scatter_mode(lam, 0, 1, x_ref=5.0)
    with pytest.raises(ValueError, match="x_ref"):
        res.mode_amplitudes(0.0)
    plane = solver.scatter(lam)
    assert plane.mode_amplitudes(0.0).p_source is None
    assert "c_guided" in plane.cross_sections() and "qguided" in plane.efficiencies()
    with pytest.raises(ValueError, match="plane wave"):
        plane.mode_amplitudes(0.0).balance()
    lossy = Stack((((N_SI + 1e-3j) ** 2, 220.0),), N_CLAD**2)
    res = BIESolver(solver.geometry, solver.material, lossy).scatter(lam)
    assert "c_guided" not in res.cross_sections()  # T8: absent, not raising
    with pytest.raises(ValueError, match="lossless"):
        res.mode_amplitudes(0.0)


# --- in-plane dipoles (R18, T7) ----------------------------------------------------


def _in_plane_source_power(k, rel, b, h):
    """k²|b|²/8 + Im[(b*·∇)(b·∇′)G_refl] at a source h above a bare stack (S11)."""
    refl = layered.StackReflection.build(1, k, rel, np.array([0.0]), np.array([h]))
    p = refl.path
    a1 = layered.alpha(p.q, k)
    wt = p.q**2 * abs(b[0]) ** 2 - a1**2 * abs(b[1]) ** 2
    self_r = np.sum(p.w * 0.5j * refl.r(p.q) / a1 * wt * np.exp(2j * a1 * h)) / PI
    return k * k * (abs(b[0]) ** 2 + abs(b[1]) ** 2) / 8 + self_r.imag


IN_PLANE_STACKS = {k: GW9_STACKS[k] for k in ("SOI", "air|Si|SiO2", "SiN@800")}


def _b_cases(mode):
    # b = (p_z, −p_x): x-, z-directed, elliptical, and spin-locked to mode 0,
    # b_z = iβb_x/γ₁, which couples one way only (S11).
    return [
        (1.0, 0.0),
        (0.0, 1.0),
        (1.0, 0.7 - 0.4j),
        (1.0, 1j * mode.beta / mode.gamma),
    ]


@pytest.mark.parametrize("case", IN_PLANE_STACKS, ids=str)
def test_in_plane_dipole_power_balance_over_a_bare_stack(case):
    # The full-range spectra are not even in q once b_x·b_z ≠ 0, so a folded
    # integral or a wrong cross term breaks the equality; the spin-locked
    # source must put all its guided power into +x.
    stack, n_clad, lam = IN_PLANE_STACKS[case]
    k = 2 * PI * n_clad / lam
    rel = stack.eps_rel(n_clad, lam)
    modes = stack.guided_modes(lam, 1, n_clad)
    for b in _b_cases(modes[0]):
        for h in (20.0, 150.0):
            p_src = _in_plane_source_power(k, rel, b, h)
            a = _amplitudes([], k, 1, rel, modes, 0.0, "dipole", (0.0, h, b), p_src)
            assert a.balance() < 5e-14  # measured 6.4e-15
    if len(modes) == 1:
        assert a.p_minus[0] / a.p_plus[0] < 1e-28  # one-way: c⁻ = 0 exactly


@pytest.mark.parametrize("case", IN_PLANE_STACKS, ids=str)
def test_in_plane_far_field_along_the_stack_is_the_guided_sum(case):
    # GW10 for −(b·∇′)G: the amplitudes (i/2βN)u(z_s)(±iβb_x + γ₁b_z) must be
    # the field's own far-x content in both directions, for every b.
    stack, n_clad, lam = IN_PLANE_STACKS[case]
    k = 2 * PI * n_clad / lam
    rel = stack.eps_rel(n_clad, lam)
    modes = stack.guided_modes(lam, 1, n_clad)
    for b in _b_cases(modes[0])[::-1][:3]:
        a = _amplitudes([], k, 1, rel, modes, 0.0, "dipole", (0.0, 50.0, b), 1.0)
        prof = np.array([m.profile(100.0) / np.sqrt(m.beta * m.norm) for m in modes])
        scale = np.sum((np.abs(a.s_plus) + np.abs(a.s_minus)) * np.abs(prof))
        for sign in (1, -1):
            s = a.s_plus if sign > 0 else a.s_minus
            rem = []
            for x_far in (1e4, 4e4):
                u = _stack_background_field(
                    np.array([sign * x_far]),
                    np.array([100.0]),
                    k,
                    1,
                    "dipole",
                    0.0,
                    (0.0, 50.0),
                    rel,
                    b,
                )[0]
                beta = np.array([m.beta for m in modes])
                rem.append(
                    abs(u - np.sum(s * prof * np.exp(1j * beta * x_far))) / scale
                )
            assert rem[1] < 0.1 and rem[1] < rem[0]  # measured ≤ 3.8e-2 at 40 µm


IN_PLANE_PARTICLES = {k: PARTICLES[k] for k in ("SOI circle", "SOI lossy triangle")}


@pytest.mark.parametrize("case", IN_PLANE_PARTICLES, ids=str)
def test_in_plane_dipole_and_particle_power_balance(case):
    # The source power needs the analytic ∇u_s at the source; a finite
    # difference would cap this at its h⁴ floor (s12b: 2e-11).
    stack, lam, geom, _, epsi, src = PARTICLES[case]
    solver = BIESolver(
        geom, Material(n_core=N_SI, n_clad=N_CLAD, pol=1, epsi=epsi), stack
    )
    m0 = stack.guided_modes(lam, 1, N_CLAD)[0]
    for b in _b_cases(m0)[:2] + _b_cases(m0)[3:]:
        res = solver.scatter_dipole(lam, *src, moment=(-b[1], b[0]))
        assert res.mode_amplitudes(0.0).balance() < 1.3e-14  # measured 1.3e-15


def test_analytic_scattered_gradient_matches_finite_differences():
    stack, lam, geom, _, epsi, (xs, zs) = PARTICLES["SOI lossy triangle"]
    solver = BIESolver(
        geom, Material(n_core=N_SI, n_clad=N_CLAD, pol=1, epsi=epsi), stack
    )
    res = solver.scatter_dipole(lam, xs, zs, moment=(0.3, 1.0 - 0.2j))
    ux, uz = res._scattered_gradient(xs, zs)
    h, c = 0.25, np.array([1, -8, 8, -1]) / 3.0
    off = np.array([-2, -1, 1, 2]) * h
    fx = sum(
        ci * res.eval_field(np.array([xs + o]), np.array([zs]))[0]
        for ci, o in zip(c, off, strict=True)
    ) / (4 * h)
    fz = sum(
        ci * res.eval_field(np.array([xs]), np.array([zs + o]))[0]
        for ci, o in zip(c, off, strict=True)
    ) / (4 * h)
    # 5-point FD at h = 0.25 nm: measured ≤ 3.0e-10 from the analytic value,
    # the FD's own h⁴ floor; 10× that.
    assert abs(ux - fx) / abs(ux) < 3e-9 and abs(uz - fz) / abs(uz) < 3e-9


@pytest.mark.parametrize("moment", [(1.0, 0.0), (0.0, 1.0), (1.0, 0.5 - 0.3j)])
def test_in_plane_free_space_pattern_is_that_of_an_electric_dipole(moment):
    # −(b·∇′)(i/4)H₀ = −(i/4)k·H₁(kρ)·b·r̂ exactly, so |u|²·16/(k²|H₁|²) is the
    # pattern |p_x cos θ − p_z sin θ|² at any radius, θ from +z: this pins the
    # b = (p_z, −p_x) map and its sign.
    geom = Geometry.gielis(rad=20.0, n_pts=64, m=4, x0=-5000.0, z0=0.0)
    mat = Material(n_core=1.5, n_clad=N_CLAD, pol=1)
    res = BIESolver(geom, mat).scatter_dipole(1550.0, 0.0, 0.0, moment=moment)
    th, rho = np.linspace(0, 2 * PI, 64, endpoint=False), 3000.0
    u = res.incident_field(rho * np.sin(th), rho * np.cos(th))
    k = res.wnum_bg
    pattern = np.abs(u) ** 2 * 16 / (k * k * np.abs(hankel1(1, k * rho)) ** 2)
    ref = np.abs(moment[0] * np.cos(th) - moment[1] * np.sin(th)) ** 2
    assert np.max(np.abs(pattern - ref)) / ref.max() < 2.4e-14  # measured 2.4e-15


def test_moment_is_tm_only():
    solver, lam, src = _solver("SOI circle", 2)
    with pytest.raises(ValueError, match="pol = 1"):
        solver.scatter_dipole(lam, *src, moment=(1.0, 0.0))
