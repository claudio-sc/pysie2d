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

from pysie2d import BIESolver, Geometry, HalfSpace, Material, QNMSolver, layered
from pysie2d.background import Stack
from pysie2d.layered import SommerfeldPath, _singularities
from pysie2d.reference.sommerfeld import g_ref_stack, g_ref_stack_gamma, stack_r_ref

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
