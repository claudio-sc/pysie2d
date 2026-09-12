"""G2 prototype: curvature-adaptive node density with a smooth clamp.

Study script — touches no package code, and production code is written from the
code-specs, never from here (study plan §2).

What it builds
--------------
The ``Parametrisation`` of architecture §3.2 as a prototype: a smooth
2π-periodic monotone ``θ = w(t)`` with ``w'`` and ``w''``, defined by a density
``|dx/dt| = ρ(t)``, plus the curvature-driven ρ of §4 and its smooth clamp.

The four pieces, and why each is shaped the way it is:

1. **Dimensionless density shape.** ρ̂ is built from ``u = |κ|·L/2π`` (``u ≡ 1``
   on a circle), so it carries **no absolute length**. That is what keeps scale
   covariance (conventions §9) exact: the R band fixes only ``nn``, and ``nn``
   is unchanged when ``rad`` and λ scale together, so ``w`` is identical and
   ``M(s·rad, s·λ) = M(rad, λ)`` entrywise still holds. A density that read λ_ref
   against an absolute ``Δs`` would move the nodes under ``rad`` scaling alone.

2. **Smooth clamp, not ``np.clip``.** ``softplus(s) − softplus(s − ln C)`` is
   C^∞, monotone, and saturates at both ends, so the band is honoured *and* the
   analyticity the trapezoid rule pays for survives. ``np.clip`` puts two kinks
   per lobe on the boundary (architecture §4). Curvature itself is sharply
   peaked at high exponent, so the log-curvature is Fourier low-passed with
   bandwidth tied to ``nn`` *before* saturating — a trigonometric polynomial is
   C^∞, and composing the analytic saturation with it stays analytic while the
   bound stays exact.

3. **``nn`` derived from the band.** Node spacing in arc length is
   ``Δs(θ) = Z / (nn · σ(θ))`` with ``Z = ∫σγ dθ`` and ``min σ = 1``, so the
   worst-resolved node is ``Δs_max = Z/nn`` and ``R_min`` pins
   ``nn = ⌈R_min · Z · n_core / λ_ref⌉``. The peaks then read ``R_min·C = R_max``
   automatically: both ends of the band are honoured by construction rather
   than checked afterwards. Rungs in ``R``, never in raw ``n_pts`` (§12, D17).

4. **Newton inversion, not ``np.interp``.** ``T(θ) = 2π/Z ∫₀^θ σγ`` is
   evaluated from its exact Fourier antiderivative and inverted by Newton.
   ``np.interp`` appears only as the initial guess, so the delivered map is
   C^∞ — that distinction is the whole of G1. ``w'' = −T''(w)·(w')³`` closes the
   triple analytically.

Run: ``python docs/design/studies/adaptive_density.py``
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))
from pysie2d.geometry import gielis  # noqa: E402

PI = np.pi
N_FINE = 4096  # a function of nothing absolute, so §9 is safe
REF_WAVELENGTH = 1550.0  # nm, architecture §4: design-time, never from the solve


# ---------------------------------------------------------------------------
# Spectral helpers on a uniform periodic θ grid
# ---------------------------------------------------------------------------


def _dtheta(vals: np.ndarray, order: int = 1) -> np.ndarray:
    """Fourier derivative of a smooth 2π-periodic sampled function."""
    n = vals.size
    j = np.fft.fftfreq(n, d=1.0 / n)
    return np.real(np.fft.ifft(np.fft.fft(vals) * (1j * j) ** order))


def _smooth(vals: np.ndarray, n_modes: int) -> np.ndarray:
    """Gaussian (heat-kernel) smoothing with Fourier width ``n_modes``.

    **Not a sharp cutoff.** Truncating the spectrum Gibbs-undershoots a peaked
    log-curvature, which digs a narrow artificial minimum; since ``min σ = 1``
    is what pins ``nn``, the band then gets set by the undershoot and every
    real node is over-resolved — measured on the mild star, where a sharp
    cutoff put every node at R ≈ 28.7 for a requested band of 15–30. A
    Gaussian kernel is positive, so the smoothed field cannot leave the range
    of the original, and it is analytic, which a trig polynomial also is.
    """
    n = vals.size
    j = np.fft.fftfreq(n, d=1.0 / n)
    taper = np.exp(-0.5 * (j / n_modes) ** 2)
    return np.real(np.fft.ifft(np.fft.fft(vals) * taper))


@dataclass(frozen=True)
class _Series:
    """A real 2π-periodic trig polynomial, evaluable at arbitrary θ."""

    mean: float
    modes: np.ndarray  # (M,) integers >= 1
    coef: np.ndarray  # (M,) complex FFT coefficients, already normalised

    @classmethod
    def fit(cls, vals: np.ndarray, n_modes: int) -> "_Series":
        n = vals.size
        spec = np.fft.fft(vals) / n
        modes = np.arange(1, n_modes + 1)
        return cls(float(np.real(spec[0])), modes, spec[modes])

    def __call__(self, theta: np.ndarray, order: int = 0) -> np.ndarray:
        """Evaluate the series (order 0), or its θ-derivative / antiderivative.

        order = -1 returns the antiderivative with the secular term dropped;
        the caller adds ``mean·θ`` itself.
        """
        fac = (1j * self.modes) ** order if order >= 0 else 1.0 / (1j * self.modes)
        phase = np.exp(1j * np.outer(theta, self.modes))
        val = 2.0 * np.real(phase @ (self.coef * fac))
        return val + (self.mean if order == 0 else 0.0)


# ---------------------------------------------------------------------------
# The density
# ---------------------------------------------------------------------------


def _softclamp(s: np.ndarray, width: float, beta: float) -> np.ndarray:
    """C^∞ monotone saturation of ``s`` onto ``[0, width]``.

    ``softplus(βs)/β − softplus(β(s−width))/β``. Analytic everywhere, unlike
    ``np.clip``, which is C⁰ at both breakpoints.
    """

    def softplus(x: np.ndarray) -> np.ndarray:
        return np.logaddexp(0.0, beta * x) / beta

    return softplus(s) - softplus(s - width)


def layer_width(kappa: np.ndarray, drop: float = 1.0) -> float:
    """Angular width of the curvature layer, in radians.

    The fraction of the boundary on which ``ln|κ|`` sits within ``drop`` of its
    maximum. On a Gielis star that is the angular extent of the arm tip, and it
    narrows as the exponent rises — the geometric signature of the thinning
    analyticity strip that architecture §4 invokes. Measured on the raw κ,
    which is analytic, so no smoothing is needed to define it and there is no
    circularity with the bandwidth it feeds.

    Args:
        kappa: |κ(θ)| on the fine grid.
        drop: How far ``ln|κ|`` may fall below its peak and still count as
            inside the layer.

    Returns:
        The layer width in radians.
    """
    s_raw = np.log(np.maximum(kappa, 1e-300))
    inside = int((s_raw > s_raw.max() - drop).sum())
    return max(inside, 1) / kappa.size * 2.0 * PI


def density_bandwidth(kappa: np.ndarray, harmonics_per_layer: float = 1.0) -> int:
    """Gaussian smoothing width for the density: ``2π / layer width``.

    **Not tied to ``nn``**, which architecture §4 suggests and measurement
    rejects: a bandwidth ∝ nn makes ``w`` a different map at every resolution,
    so refinement chases a moving target and ``∫w' dt = 2π`` stalls near 1e-3
    instead of converging. It is also invariant 13.3 in another guise — the
    frozen object has to be one map.

    **Not a faithful fit to κ either.** Retaining all but 1e-4 of the
    log-curvature's spectral energy asks for M = 1253 on the near-corner star,
    i.e. no smoothing at all: the density is then as sharply peaked as κ and
    the map loses the wide analyticity strip that is the whole point.

    Tying M to the *layer width* instead puts the bandwidth where the physics
    is. It is also the rule that orders correctly with sharpness — a lobe-count
    rule gave the **sharpest** star the **lowest** bandwidth, which is backwards
    — and it makes the cost of the near-corner regime explicit before any solve:
    M = 4, 18, 38, 94 across the four stars, so a graded map needs
    ``nn ≳ 4M`` = 16, 72, 152, 376.

    Args:
        kappa: |κ(θ)| on the fine grid.
        harmonics_per_layer: Gaussian widths retained per layer width.

    Returns:
        The mode cutoff M.
    """
    return max(4, int(np.ceil(harmonics_per_layer * 2.0 * PI / layer_width(kappa))))


def density_shape(
    kappa: np.ndarray,
    gam: np.ndarray,
    contrast: float,
    n_modes: int,
    alpha: float = 0.5,
    beta_span: float = 8.0,
    hard: bool = False,
    cap_alpha: bool = True,
) -> tuple[np.ndarray, float]:
    """Dimensionless node-density factor σ(θ), with ``min σ = 1``.

    σ ∝ |κ|^α_eff, **anchored at the curvature maximum**: the finest sampling
    (σ = contrast) sits at the sharpest point and σ = 1 along the smoothest,
    which is doc A §4's specification. Anchoring at the minimum instead makes
    the *upper* bound bind almost everywhere, and the grading degenerates into
    uniform arc length with a few artificially coarsened points.

    **α is derived from the band, not pinned.** The 4-peak star carries a
    curvature contrast of ~8.8e3, so σ ∝ |κ|^{1/2} spans ~93× against a
    requested band of 2×: the clamp then saturates over the whole boundary and
    the density is two-level, not graded. Taking

        α_eff = min(α, ln(contrast) / range(ln|κ|))

    makes σ span exactly ``[1, contrast]`` smoothly, uses the whole band, and
    leaves the clamp **inactive** — so the clamp is a guard for a caller who
    pins α, not part of the normal path.

    Args:
        kappa: |κ(θ)| on the fine grid (1/nm).
        gam: γ(θ) = |x'(θ)| on the fine grid (nm).
        contrast: C = R_max/R_min, the finest-to-coarsest sampling ratio.
        n_modes: Gaussian smoothing width, from :func:`density_bandwidth`.
        alpha: Upper bound on the curvature exponent.
        beta_span: β·ln C, the saturation sharpness in units of the band.
        hard: Use ``np.clip`` instead of the smooth saturation — the C⁰ control
            case for G2.
        cap_alpha: Apply the ``α_eff`` cap above. Setting it False pins α and
            makes the clamp bind, which is the only regime where the choice of
            clamp matters at all.

    Returns:
        σ(θ) on the fine grid with ``min σ = 1`` and ``max σ <= contrast``, and
        the α actually used.
    """
    if contrast <= 1.0:
        return np.ones_like(kappa), 0.0  # const density: the uniform-arc baseline

    perimeter = float(np.mean(gam) * 2.0 * PI)
    u = np.maximum(kappa * perimeter / (2.0 * PI), 1e-12)  # ≡ 1 on a circle

    # Smooth first: κ is sharply peaked at high exponent, and the density is a
    # design profile that has to track it loosely and be analytic.
    s_lp = _smooth(np.log(u), n_modes)
    width = np.log(contrast)
    spread = float(s_lp.max() - s_lp.min())
    if spread <= 0.0:
        alpha_eff = 0.0
    else:
        alpha_eff = min(alpha, width / spread) if cap_alpha else alpha

    # Anchored at the maximum, so s <= 0 and the *floor* is the active bound.
    s = alpha_eff * (s_lp - s_lp.max())
    s = np.clip(s, -width, 0.0) if hard else -_softclamp(-s, width, beta_span / width)

    sigma = np.exp(s)
    return sigma / sigma.min(), alpha_eff  # min σ = 1 exactly, so R_min pins nn


# ---------------------------------------------------------------------------
# The parametrisation
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Parametrisation:
    """Prototype of the v0.6 object: a smooth monotone ``θ = w(t)``."""

    theta: np.ndarray  # w(t_j) at the nn equispaced quadrature nodes
    dw: np.ndarray  # w'(t_j)
    ddw: np.ndarray  # w''(t_j)
    t: np.ndarray
    nn: int
    sigma_nodes: np.ndarray
    r_achieved: tuple[float, float]  # (min, max) over nodes, in arc length
    r_chord: tuple[float, float]  # the same, on chords: the package reading
    perimeter: float
    n_modes: int  # Gaussian width of the density; nn >= 4·n_modes is required
    alpha_eff: float  # curvature exponent actually used, derived from the band


def build(
    shape: dict,
    r_band: tuple[float, float],
    n_core: float,
    wavelength: float = REF_WAVELENGTH,
    alpha: float = 0.5,
    hard: bool = False,
    uniform_arc: bool = False,
    per_layer: float = 1.0,
    cap_alpha: bool = True,
    nn_force: int | None = None,
) -> Parametrisation:
    """Build the parametrisation for one shape from an R band.

    Args:
        shape: Keyword arguments for :func:`pysie2d.geometry.gielis` minus θ.
        r_band: ``(R_min, R_max)`` in the units of conventions §12 — points per
            *interior* wavelength, at ``wavelength``/``n_core``.
        n_core: Core index, which sets the interior wavelength.
        wavelength: The parametrisation's own fixed reference λ_vac (nm). Fixed
            at construction and never read from a solve, so ``M(λ)`` keeps the
            holomorphy Beyn's contour needs (conventions §13.2).
        alpha: Curvature exponent of the density.
        hard: Use the C⁰ ``np.clip`` clamp (the G2 control).
        per_layer: Gaussian widths retained per curvature-layer width.
        cap_alpha: Derive the curvature exponent from the band (the default);
            False pins ``alpha`` and lets the clamp bind.
        nn_force: Sample the same map at this nn instead of the one the band
            derives. For diagnostics only — comparing two clamps at their own
            derived nn would compare two different grids as well as two maps.
        uniform_arc: Ignore curvature entirely, σ ≡ 1 — the const-density
            baseline adaptive grading has to beat.

    Returns:
        The :class:`Parametrisation`.
    """
    r_min, r_max = r_band
    contrast = 1.0 if uniform_arc else r_max / r_min

    theta_f = np.linspace(0.0, 2.0 * PI, N_FINE, endpoint=False)
    f, g, *_ = gielis(theta_f, **shape)
    df, dg = _dtheta(f), _dtheta(g)
    ddf, ddg = _dtheta(f, 2), _dtheta(g, 2)
    gam = np.hypot(df, dg)
    kappa = np.abs(df * ddg - dg * ddf) / gam**3
    perimeter = float(np.mean(gam) * 2.0 * PI)

    # The map is built first and depends on nn nowhere; nn is then read off
    # the band. One direction only — an nn-dependent map is invariant 13.3 in
    # another guise.
    n_modes = density_bandwidth(kappa, harmonics_per_layer=per_layer)
    sigma, alpha_eff = density_shape(
        kappa, gam, contrast, n_modes, alpha=alpha, hard=hard, cap_alpha=cap_alpha
    )
    z_norm = float(np.mean(sigma * gam) * 2.0 * PI)

    # T(θ) = 2π/Z ∫₀^θ σγ, from the exact Fourier antiderivative of σγ. The
    # series bandwidth is a *representation* choice set by the fine grid, and
    # deliberately not the density bandwidth: truncating it near nn leaves a
    # γ-shaped residual that shows up as a non-constant Δs even at σ ≡ 1.
    integrand = _Series.fit(sigma * gam, N_FINE // 2 - 1)
    scale = 2.0 * PI / z_norm
    anti0 = integrand(np.zeros(1), order=-1)[0]

    def t_of(theta: np.ndarray) -> np.ndarray:
        return scale * (integrand.mean * theta + integrand(theta, -1) - anti0)

    def invert(nn: int):
        """Place nn equispaced-in-t nodes on the map, with w' and w''."""
        t_nodes = 2.0 * PI * (np.arange(nn) + 0.5) / nn
        # np.interp only seeds Newton; the delivered map is the analytic
        # inverse, which is the whole difference from the v0.5 C⁰ inversion.
        theta = np.interp(t_nodes, t_of(theta_f), theta_f)
        for _ in range(40):
            step = (t_of(theta) - t_nodes) / (scale * integrand(theta))
            theta -= step
            if np.max(np.abs(step)) < 1e-14:
                break
        dw = 1.0 / (scale * integrand(theta))
        return t_nodes, theta, dw, -scale * integrand(theta, 1) * dw**3

    def worst_r(nn: int) -> float:
        _, theta, dw, _ = invert(nn)
        ds = np.interp(theta, theta_f, gam) * dw * (2.0 * PI / nn)
        return float(((wavelength / n_core) / ds).min())

    # nn from the band. The first estimate uses min σ = 1 over the continuum,
    # which is conservative: nodes are sparsest exactly where σ is smallest, so
    # if that minimum is narrow no node lands in it and the estimate
    # over-resolves by up to ~2×. Descend from it, **verifying** each candidate
    # rather than extrapolating — the nodes move when nn changes, so the
    # extrapolated nn can land below the band it was derived from.
    nn = int(np.ceil(r_min * z_norm * n_core / wavelength))
    nn += nn % 2
    for _ in range(8):
        cand = int(np.ceil(nn * r_min / worst_r(nn)))
        cand += cand % 2
        if cand >= nn or cand < 4 or worst_r(cand) < r_min:
            break
        nn = cand
    if nn_force is not None:
        nn = nn_force
    t_nodes, theta, dw, ddw = invert(nn)

    tp = scale * integrand(theta)
    del tp
    # σ at the nodes, normalised on the fine grid where min σ = 1 holds
    # exactly; normalising on the nodes would rescale by whatever the node set
    # happened to sample and report a contrast above the cap.
    sigma_nodes = integrand(theta) / np.interp(theta, theta_f, gam)
    ds_arc = np.interp(theta, theta_f, gam) * dw * (2.0 * PI / nn)
    r_nodes = (wavelength / n_core) / ds_arc
    return Parametrisation(
        theta=theta,
        dw=dw,
        ddw=ddw,
        t=t_nodes,
        nn=nn,
        sigma_nodes=sigma_nodes,
        r_achieved=(float(r_nodes.min()), float(r_nodes.max())),
        r_chord=_chord_band(shape, theta, wavelength, n_core),
        perimeter=perimeter,
        n_modes=n_modes,
        alpha_eff=alpha_eff,
    )


def _chord_band(shape, theta, wavelength, n_core) -> tuple[float, float]:
    """R from consecutive chords — how :func:`wavelength_over_ds` reads it.

    It overstates R at a sharp tip, where the chord is measurably shorter than
    the arc it subtends; the band is exact in arc length and only approximate
    in this reading.
    """
    f, g, *_ = gielis(theta, **shape)
    ds = np.hypot(np.diff(f, append=f[0]), np.diff(g, append=g[0]))
    r = (wavelength / n_core) / ds
    return float(r.min()), float(r.max())


# ---------------------------------------------------------------------------
# Self-checks
# ---------------------------------------------------------------------------


def checks(shape: dict, par: Parametrisation) -> dict:
    """Verify the map, its derivatives and the band on one shape.

    Three things have to hold, and each fails silently if it does not:
    monotonicity (a non-monotone w traverses part of the curve twice),
    ``∫w' dt = 2π`` (the trapezoid rule on w' — so this doubles as a read on
    whether nn resolves the density it was handed), and ``|dx/dt| ∝ 1/σ``,
    which is the definition of the density actually being delivered.
    """
    theta_f = np.linspace(0.0, 2.0 * PI, N_FINE, endpoint=False)
    f, g, *_ = gielis(theta_f, **shape)
    gam = np.hypot(_dtheta(f), _dtheta(g))
    speed = np.interp(par.theta, theta_f, gam) * par.dw  # |dx/dt|
    target = speed * par.sigma_nodes  # must be constant
    # w' against central differences of w, which needs w off the node grid
    return {
        "nn": par.nn,
        "monotone": bool(np.all(np.diff(par.theta) > 0)),
        "R_arc": tuple(round(v, 2) for v in par.r_achieved),
        "R_chord": tuple(round(v, 2) for v in par.r_chord),
        "sigma_max": round(float(par.sigma_nodes.max()), 3),
        "density_resid": float(np.max(np.abs(target / target.mean() - 1.0))),
        "int_dw_err": abs(float(np.sum(par.dw) * 2 * PI / par.nn) - 2 * PI),
        # The density must itself be resolved by the nodes the band asks for.
        # Not a coupling — the map stays nn-independent — but a *check*: if it
        # fails, the band is too coarse to carry a graded map on this shape.
        "M": par.n_modes,
        "nn_min": 4 * par.n_modes,
        "density_resolved": par.nn >= 4 * par.n_modes,
        "alpha_eff": round(par.alpha_eff, 4),
    }


def _fourier_decay(shape, r_band, n_core, n_eval: int = 1024, **kw) -> np.ndarray:
    """|Fourier coefficients| of ``w(t) − t`` on a fixed fine t grid.

    The smoothness diagnostic, and the measurement that has to justify the
    smooth clamp: an analytic ``w`` decays geometrically, a C⁰ one decays
    algebraically (~1/j²) and never stops mattering. Sampled at a fixed
    ``n_eval`` for every variant, so the comparison is about the map alone.
    """
    par = build(shape, r_band, n_core, nn_force=n_eval, **kw)
    return np.abs(np.fft.rfft(par.theta - par.t)) / n_eval


def _star(n1: float, n2: float) -> dict:
    """A 4-peak Gielis star; ``m = 4`` gives four lobes in this convention."""
    return {
        "rad": 200.0,
        "a": 1,
        "b": 1,
        "m": 4,
        "n1": n1,
        "n2": n2,
        "n3": n2,
        "x0": 0,
        "z0": 0,
    }


# A sharpness ladder at fixed symmetry: the curvature layer at the arm tips
# narrows with the exponent, which is the regime graded quadrature is for.
STARS = [
    ("mild  n1=2, n2=n3=4", _star(2, 4)),
    ("baseline  n1=6, n2=n3=12", _star(6, 12)),
    ("sharp  n1=12, n2=n3=24", _star(12, 24)),
    ("near-corner  n1=20, n2=n3=50", _star(20, 50)),
]
R_BAND = (20.0, 100.0)
N_CORE = 1.5


def main() -> None:
    """Run the four stars and write the figure."""
    out = Path(__file__).with_name("star_adaptive_sampling.png")
    fig, axes = plt.subplots(4, 3, figsize=(13.5, 16.5))

    for row, (label, shape) in enumerate(STARS):
        ad = build(shape, R_BAND, N_CORE)
        # Equal nn, redistributed: the fair const-density baseline. Compared at
        # its *own* band-derived nn instead, uniform arc looks cheaper (96 vs
        # 172 on the mild star) — because grading at a fixed R_min adds nodes at
        # the peaks rather than moving them. Grading is extra resolution where
        # the geometry needs it, not a node-count saving.
        un = build(shape, R_BAND, N_CORE, uniform_arc=True, nn_force=ad.nn)
        hd = build(shape, R_BAND, N_CORE, hard=True)
        print(f"\n{label}")
        for name, par in (("adaptive", ad), ("uniform-arc", un), ("hard clip", hd)):
            fields = ", ".join(f"{k}={v}" for k, v in checks(shape, par).items())
            print(f"  {name:12s} {fields}")

        # --- column 1: the boundary and its nodes
        ax = axes[row, 0]
        theta_f = np.linspace(0, 2 * PI, 2000)
        f_f, g_f, *_ = gielis(theta_f, **shape)
        dense = np.linspace(0, 2 * PI, N_FINE, endpoint=False)
        fd, gd, *_ = gielis(dense, **shape)
        dfd, dgd = _dtheta(fd), _dtheta(gd)
        gam_d = np.hypot(dfd, dgd)
        kap = np.abs(dfd * _dtheta(gd, 2) - dgd * _dtheta(fd, 2)) / gam_d**3
        ax.plot(f_f, g_f, "-", color="0.75", lw=1.0, zorder=1)
        fa, ga, *_ = gielis(ad.theta, **shape)
        sc = ax.scatter(fa, ga, c=ad.sigma_nodes, s=14, cmap="viridis", zorder=3)
        fu, gu, *_ = gielis(un.theta, **shape)
        ax.scatter(
            fu,
            gu,
            s=5,
            marker="x",
            color="crimson",
            alpha=0.55,
            zorder=2,
            label=f"uniform arc, same nn={un.nn}",
        )
        ax.set_aspect("equal")
        need = 4 * ad.n_modes
        verdict = "resolved" if ad.nn >= need else f"UNDER-RESOLVED, needs {need}"
        ax.set_title(
            f"{label}\nlayer {layer_width(kap):.3f} rad → M={ad.n_modes}\n"
            f"nn={ad.nn} from the band · {verdict}",
            fontsize=9.5,
        )
        ax.legend(fontsize=6.5, loc="lower left")
        fig.colorbar(sc, ax=ax, fraction=0.046, label="σ (density factor)")

        # --- column 2: resolution per node, against the requested band
        ax = axes[row, 1]
        for par, name, col in ((ad, "adaptive", "C0"), (un, "uniform arc", "crimson")):
            fj, gj, *_ = gielis(par.theta, **shape)
            theta_f = np.linspace(0, 2 * PI, N_FINE, endpoint=False)
            ff, gg, *_ = gielis(theta_f, **shape)
            gam_j = np.interp(par.theta, theta_f, np.hypot(_dtheta(ff), _dtheta(gg)))
            ds = gam_j * par.dw * (2 * PI / par.nn)  # arc length per node
            ax.plot(
                par.t,
                (REF_WAVELENGTH / N_CORE) / ds,
                ".",
                ms=3.5,
                color=col,
                label=name,
            )
        ax.axhline(R_BAND[0], color="0.3", ls="--", lw=0.9)
        ax.axhline(R_BAND[1], color="0.3", ls="--", lw=0.9)
        ax.set_xlabel("t")
        ax.set_ylabel("R = (λ/n_core)/Δs")
        ax.set_title(
            f"R in [{ad.r_achieved[0]:.0f}, {ad.r_achieved[1]:.0f}] of a requested "
            f"[{R_BAND[0]:.0f}, {R_BAND[1]:.0f}] · α_eff={ad.alpha_eff:.2f}",
            fontsize=9,
        )
        ax.legend(fontsize=6.5, loc="upper left", framealpha=0.85)
        axd = ax.twinx()
        axd.plot(ad.t, ad.sigma_nodes, "-", color="0.55", lw=0.9)
        axd.set_ylabel("σ (density factor)", color="0.45", fontsize=8)

        # --- column 3: what the clamp costs, in the regime where it binds.
        # With α derived from the band the clamp is inactive and smooth and
        # hard are identical — which is the result, not a missing measurement.
        # Pinning α = 0.5 makes the band bind and separates them.
        ax = axes[row, 2]
        specs = {}
        for kw, name, col, style in (
            ({}, "α from band (clamp inactive)", "0.45", "-"),
            ({"cap_alpha": False}, "pinned α = 1/2, smooth clamp", "C0", "-"),
            (
                {"cap_alpha": False, "hard": True},
                "pinned α = 1/2, np.clip",
                "darkorange",
                "-",
            ),
        ):
            spec = _fourier_decay(shape, R_BAND, N_CORE, **kw)
            specs[name] = spec
            ax.semilogy(
                np.arange(1, len(spec)),
                np.maximum(spec[1:], 1e-18),
                style,
                color=col,
                lw=1.0,
                label=name,
            )
        gain = specs["pinned α = 1/2, np.clip"][500] / max(
            specs["pinned α = 1/2, smooth clamp"][500], 1e-30
        )
        ax.set_xlabel("Fourier mode j of w(t) − t")
        ax.set_ylabel("|coefficient|")
        ax.set_title(
            f"pinned α: np.clip tail at j=500 is {gain:.0f}× worse", fontsize=9
        )
        ax.legend(fontsize=6.5, loc="lower left")

    fig.suptitle(
        "v0.6 G2 prototype — curvature-adaptive node density on the "
        "4-peak Gielis star\n"
        f"cheap on the smooth shapes, expensive at the corner: band "
        f"{R_BAND[0]:.0f}\u2013{R_BAND[1]:.0f} at λ_ref = {REF_WAVELENGTH:g} nm, "
        f"n_core = {N_CORE}; nn derived from the band, uniform arc at the same nn",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.965))
    fig.savefig(out, dpi=140)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
