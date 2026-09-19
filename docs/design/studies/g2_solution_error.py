"""G2, solution-error half: does the adaptive density buy accuracy under Kress?

Closes the three "not established here" items of the 2026-09-12 G2 findings in
`quadrature-study-plan.md`, all of which are about *solution* error rather than
map smoothness, and none of which could be asked before Kress landed
(commit 771bc9b):

1. a convergence rate for adaptive vs constant density at equal `nn`;
2. smooth (Gaussian) clamp vs hard `np.clip` clamp on solution error;
3. whether `α_eff` derived from the R band is accuracy-optimal or merely
   band-filling.

Everything here calls **production** code: `Parametrisation` for the maps,
`Geometry.gielis(parametrisation=...)` and `BIESolver.scatter` for the Kress
assembly. The only thing this script builds itself is the *hard-clamp* density,
which production deliberately does not ship — and it builds it by reusing the
production smoothing and series machinery so that the clamp is the only
difference between the two maps.

Observable: **`qext`**, TE (`pol = 2`). `qext` is read from a single forward
far-field amplitude, so it does not go through the angular quadrature that G5
has yet to fix; `qsca` does (study plan, "Probe with `qext`" trap).

Reference: **high-`nn` uniform-θ Kress on the same shape** (doc B's rule — there
is no closed form for a star, and an arc-length reference would bake one of the
competing maps into the answer). Its own convergence is measured, so the error
floor is a number rather than an assumption.
"""

import numpy as np

from pysie2d.geometry import Geometry
from pysie2d.material import Material
from pysie2d.parametrisation import (
    EPS,
    N_FINE_START,
    TWOPI,
    Parametrisation,
    _density,
    _layer_bandwidth,
    _radius_derivatives,
    _speed_curvature,
)
from pysie2d.solver import BIESolver

# The G2 "baseline" star of the cost-curve table: layer width 0.362 rad, M = 18,
# nn from the band 80, 4M = 72. Informative and not yet in G3's near-corner
# regime, where the density is the binding constraint rather than the band.
SHAPE = {"rad": 200.0, "a": 1.0, "b": 1.0, "m": 4, "n1": 6.0, "n2": 12.0, "n3": 12.0}
N_CORE = 1.5
N_CLAD = 1.0
POL = 2  # TE (conventions §1)
WAVELENGTH = 600.0  # λ_vac, nm; also the map's own fixed λ_ref (§13.2)
R_BAND = (20.0, 100.0)  # the G2 band; C = 5
ALPHA = 0.5  # the spec's upper bound on the curvature exponent

NN_LADDER = (40, 60, 80, 100, 120, 160, 200)
NN_REF = 800
NN_REF_CHECK = 640  # second reference rung → the reference's own error floor


# ---------------------------------------------------------------------------
# Maps
# ---------------------------------------------------------------------------


def _fine_grid(n_f: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """θ grid, |κ| and γ of the shape at resolution `n_f`.

    Args:
        n_f: Grid size.

    Returns:
        ``theta_f``, ``|κ|``, ``γ``.
    """
    theta_f = np.linspace(0.0, TWOPI, n_f, endpoint=False)
    r, r1, r2, r3 = _radius_derivatives(theta_f, **SHAPE)
    gam, kappa = _speed_curvature(r, r1, r2, r3)
    return theta_f, np.abs(kappa), gam


def _smoothed_log_curvature(kappa: np.ndarray, gam: np.ndarray) -> np.ndarray:
    """`s̃` of spec §5.1: Gaussian-smoothed ``ln u``, exactly as `_density` does.

    Reproducing the two lines rather than importing them is unavoidable — they
    live inside `_density` — but the bandwidth comes from the production
    `_layer_bandwidth`, so the smoothing kernel is the shipped one.

    Args:
        kappa: ``|κ|`` on the fine grid.
        gam: ``γ`` on the fine grid.

    Returns:
        ``s̃`` on the fine grid.
    """
    log_u = np.log(kappa * float(gam.mean()))
    bandwidth = _layer_bandwidth(kappa)
    freq = np.fft.fftfreq(kappa.size, d=1.0 / kappa.size)
    taper = np.exp(-0.5 * (freq / bandwidth) ** 2)
    return np.real(np.fft.ifft(np.fft.fft(log_u) * taper))


def _from_sigma(sigma_of, force_n_f: int | None = None) -> Parametrisation:
    """Build a `Parametrisation` from a relative density ``σ(θ)``.

    The tail of `Parametrisation.gielis` verbatim — including its `N_f` doubling
    loop and its *relative* truncation, which is what keeps conventions §9
    structural — with the density supplied instead of computed. `nn` is not
    derived from a band here: this study fixes `nn` and compares maps at equal
    `nn`, which is what the G2 pass criterion asks.

    Args:
        sigma_of: Callable ``(kappa, gam) → σ`` on the fine grid.
        force_n_f: Skip the doubling loop and use this ``N_f`` with the full
            series. Needed only for the hard clip, whose ``σγ`` is C⁰ and whose
            Fourier coefficients therefore *never* reach the round-off plateau
            the production loop waits for — the loop is itself a measurement,
            reported below.

    Returns:
        The frozen map.

    Raises:
        ValueError: If ``σγ`` has not resolved by ``N_f = 2**20``.
    """
    if force_n_f is not None:
        theta_f, kappa, gam = _fine_grid(force_n_f)
        sigma = sigma_of(kappa, gam)
        spec = np.fft.rfft(sigma * gam) / force_n_f
        decay = np.abs(spec[1:]) / np.abs(spec[0].real)
        n_f = force_n_f
        n_terms = decay.size - 1
        return _pack(spec, n_terms, theta_f, sigma, n_f)

    n_f = N_FINE_START
    while True:
        theta_f, kappa, gam = _fine_grid(n_f)
        sigma = sigma_of(kappa, gam)
        spec = np.fft.rfft(sigma * gam) / n_f
        decay = np.abs(spec[1:]) / np.abs(spec[0].real)
        if decay[3 * n_f // 8 :].max() <= 4.0 * EPS:
            break
        n_f *= 2
        if n_f > 1 << 20:
            raise ValueError("σγ does not resolve below N_f = 2**20")
    n_terms = int(np.max(np.flatnonzero(decay > EPS) + 1, initial=0))
    return _pack(spec, n_terms, theta_f, sigma, n_f)


def _pack(spec, n_terms, theta_f, sigma, n_f) -> Parametrisation:
    """Assemble the frozen object from a truncated ``σγ`` spectrum.

    Args:
        spec: ``rfft(σγ)/N_f``.
        n_terms: ``K``, retained harmonics.
        theta_f: The fine θ grid.
        sigma: The density on that grid.
        n_f: ``N_f``.

    Returns:
        The frozen map.
    """
    obj = Parametrisation(
        a0=float(spec[0].real),
        coef=spec[1 : n_terms + 1].copy(),
        theta_fine=theta_f,
        t_fine=np.empty(0),
        nn_from_band=None,
        contrast_realised=float(sigma.max() / sigma.min()),
        alpha_eff=float("nan"),
        n_fine=n_f,
        n_terms=n_terms,
    )
    object.__setattr__(obj, "t_fine", np.concatenate([obj._t_of(theta_f), [TWOPI]]))
    return obj


def map_uniform_arc() -> Parametrisation:
    """Constant density — the fair const-density baseline (G2 correction 6)."""
    return _from_sigma(lambda kappa, gam: np.ones_like(gam))


def map_alpha(alpha_eff: float) -> Parametrisation:
    """Adaptive map at a **prescribed** curvature exponent ``σ ∝ exp(α_eff s̃)``.

    This is the shipped smooth-clamp family: `_density` differs only in choosing
    ``α_eff`` from the band. Prescribing it is what lets question 3 be asked.

    Args:
        alpha_eff: The curvature exponent.

    Returns:
        The frozen map.
    """

    def sigma(kappa, gam):
        s = _smoothed_log_curvature(kappa, gam)
        return np.exp(alpha_eff * (s - s.mean()))

    return _from_sigma(sigma)


HARD_N_F = 4096  # fixed: the hard clip never resolves, so N_f must be pinned


def _spread() -> float:
    """``Δ = range(s̃)``, the smoothed log-curvature spread of the shape."""
    _, kappa, gam = _fine_grid(N_FINE_START)
    s = _smoothed_log_curvature(kappa, gam)
    return float(s.max() - s.min())


def map_hard_clip() -> Parametrisation:
    """Hard `np.clip` clamp into the same ``[R_min, R_max]`` band.

    α is **pinned** at `ALPHA`, because with α derived from the band the clamp is
    never active and the two schemes coincide (G2 finding 3). The clip is on the
    log-density, so the realised contrast is exactly ``C`` and the band is the
    same one the smooth map fills — the clamp is the only difference.

    Returns:
        The frozen map.
    """

    def sigma(kappa, gam):
        s = _smoothed_log_curvature(kappa, gam)
        clipped = np.clip(ALPHA * (s - s.min()), 0.0, np.log(R_BAND[1] / R_BAND[0]))
        return np.exp(clipped - clipped.mean())

    return _from_sigma(sigma, force_n_f=HARD_N_F)


def map_hard_clip_smooth_twin() -> Parametrisation:
    """The smooth clamp at the pinned α — the hard clip's like-for-like twin.

    Same pinned α, same band, saturated smoothly instead of clipped: the exact
    ``α_eff = α ln C/(α Δ + ln C)`` of spec §5.2, which is what the shipped code
    does to keep the realised contrast inside the band without a kink.

    Returns:
        The frozen map.
    """
    holder = {}

    def sigma(kappa, gam):
        sig, alpha_eff, _ = _density(kappa, gam, R_BAND[1] / R_BAND[0], ALPHA)
        holder["a"] = alpha_eff
        return sig

    par = _from_sigma(sigma)
    object.__setattr__(par, "alpha_eff", holder["a"])
    return par


# ---------------------------------------------------------------------------
# The observable
# ---------------------------------------------------------------------------


def qext(par: Parametrisation, nn: int) -> float:
    """`qext` at `nn` nodes on the map `par`, through the production solver.

    Args:
        par: The node map.
        nn: Number of boundary nodes.

    Returns:
        TE extinction efficiency at `WAVELENGTH`.
    """
    geom = Geometry.gielis(n_pts=nn, parametrisation=par, **SHAPE)
    mat = Material(n_core=N_CORE, n_clad=N_CLAD, pol=POL)
    return BIESolver(geom, mat).scatter(WAVELENGTH).efficiencies()["qext"]


def ladder(par: Parametrisation, ref: float) -> list[float]:
    """Relative `qext` error against the reference, over `NN_LADDER`.

    Args:
        par: The node map.
        ref: Reference `qext`.

    Returns:
        One relative error per rung.
    """
    return [abs(qext(par, nn) - ref) / abs(ref) for nn in NN_LADDER]


FIT_FROM = 80  # fit the exponential rate only on resolved rungs (nn >= 4M = 72)


def _rate(errs: list[float]) -> float:
    """Exponential decay constant ``b`` in ``err ~ exp(-b·nn)``, per node.

    A local algebraic order is the wrong summary for a spectral scheme, and it
    is also unstable here: ``qext`` error passes through zero as it converges,
    so consecutive-rung orders swing between −11 and +19 on the same ladder. A
    least-squares slope of ``ln err`` against ``nn`` over the resolved rungs is
    the quantity that actually distinguishes the maps.

    Args:
        errs: Relative errors over `NN_LADDER`.

    Returns:
        ``b``; larger is faster.
    """
    nn = np.array([n for n in NN_LADDER if n >= FIT_FROM], dtype=float)
    ee = np.log(np.array(errs[-nn.size :]))
    return float(-np.polyfit(nn, ee, 1)[0])


def _report(name: str, errs: list[float]) -> None:
    """Print one ladder with its observed order between consecutive rungs."""
    cells = []
    for i, e in enumerate(errs):
        if i == 0:
            cells.append(f"{e:.2e}")
        else:
            p = np.log(errs[i - 1] / e) / np.log(NN_LADDER[i] / NN_LADDER[i - 1])
            cells.append(f"{e:.2e}(p={p:4.1f})")
    print(f"{name:28s} " + "  ".join(cells) + f"   b={_rate(errs):.4f}")


def main() -> None:
    """Run the reference, the three maps, and the α_eff sweep."""
    global WAVELENGTH
    uni = Parametrisation.uniform_theta()
    ref = qext(uni, NN_REF)
    ref_check = qext(uni, NN_REF_CHECK)
    floor = abs(ref - ref_check) / abs(ref)
    print(f"shape {SHAPE}, TE, λ_vac = {WAVELENGTH} nm, n_core = {N_CORE}")
    print(f"reference uniform-θ qext(nn={NN_REF}) = {ref:.12f}")
    print(
        f"reference self-consistency |q({NN_REF_CHECK}) - q({NN_REF})|/q "
        f"= {floor:.2e}   <- error floor of every number below"
    )
    print("nn:" + "".join(f"{n:>16d}" for n in NN_LADDER))

    band_par = Parametrisation.gielis(
        **SHAPE,
        r_band=R_BAND,
        n_core=N_CORE,
        wavelength_ref=WAVELENGTH,
        alpha=ALPHA,
    )
    print(
        f"shipped map: α_eff = {band_par.alpha_eff:.4f}, "
        f"contrast_realised = {band_par.contrast_realised:.3f}, "
        f"nn_from_band = {band_par.nn_from_band}"
    )

    _report("uniform θ (default)", ladder(uni, ref))
    _report("uniform arc length", ladder(map_uniform_arc(), ref))
    _report("adaptive, shipped α_eff", ladder(band_par, ref))

    print(
        "\nclamp comparison at pinned α = %.2f, band C = %.1f"
        % (ALPHA, R_BAND[1] / R_BAND[0])
    )
    # The clamp's C⁰ kink, before any solve: the hard-clipped σγ's spectrum
    # decays algebraically and never reaches the 4·eps plateau the production
    # constructor's doubling loop waits for, so that loop runs to N_f = 2**20
    # and raises. That is why HARD_N_F has to be pinned at all.
    for label, fn in (("smooth", None), ("hard clip", map_hard_clip)):
        n_f = HARD_N_F
        _, kappa, gam = _fine_grid(n_f)
        if fn is None:
            sig, _, _ = _density(kappa, gam, R_BAND[1] / R_BAND[0], ALPHA)
        else:
            sm = _smoothed_log_curvature(kappa, gam)
            cl = np.clip(ALPHA * (sm - sm.min()), 0.0, np.log(R_BAND[1] / R_BAND[0]))
            sig = np.exp(cl - cl.mean())
        sp = np.fft.rfft(sig * gam) / n_f
        top = np.abs(sp[3 * n_f // 8 :]).max() / abs(sp[0].real)
        print(
            f"  σγ spectral tail at N_f = {n_f}, {label:9s}: {top:.1e} "
            f"(production accepts <= {4.0 * EPS:.1e})"
        )

    smooth_twin = map_hard_clip_smooth_twin()
    print(f"  smooth-saturation α_eff = {smooth_twin.alpha_eff:.4f}")
    _report("  smooth clamp (pinned α)", ladder(smooth_twin, ref))
    matched = map_alpha(np.log(R_BAND[1] / R_BAND[0]) / _spread())
    print(
        f"  contrast-matched smooth α_eff = "
        f"{np.log(R_BAND[1] / R_BAND[0]) / _spread():.4f} (realised C = 5 exactly)"
    )
    _report("  smooth, C matched", ladder(matched, ref))
    _report("  hard np.clip (pinned α)", ladder(map_hard_clip(), ref))

    print("\nα_eff sweep (accuracy vs band-filling); shipped value marked")
    for a in (0.0, 0.1, 0.2, 0.3, band_par.alpha_eff, 0.5, 0.7):
        tag = " <- shipped" if abs(a - band_par.alpha_eff) < 1e-12 else ""
        _report(f"  α_eff = {a:.4f}{tag}", ladder(map_alpha(a), ref))

    # Two more wavelengths: `qext` error crosses zero as it converges, so a
    # ranking read at one λ could be an accident of where the crossing fell.
    # Only `b` is reported — the ladders themselves are in the block above.
    print("\nrobustness of the rate b across λ_vac (map λ_ref moves with it)")
    base_wl = WAVELENGTH
    for wl in (450.0, 900.0):
        WAVELENGTH = wl
        rw = qext(uni, NN_REF)
        rows = [
            ("uniform θ", uni),
            ("uniform arc", map_uniform_arc()),
            ("α_eff = 0.20", map_alpha(0.20)),
            ("α_eff = 0.2155 (shipped)", map_alpha(band_par.alpha_eff)),
            ("α_eff = 0.25", map_alpha(0.25)),
        ]
        cells = "  ".join(f"{n}: b={_rate(ladder(m, rw)):.4f}" for n, m in rows)
        print(f"  λ = {wl:5.0f} nm   {cells}")
    WAVELENGTH = base_wl

    print("\nα_eff is set by the band, and the band is the free knob")
    for c in (2.0, 5.0, 10.0, 20.0):
        pc = Parametrisation.gielis(
            **SHAPE,
            r_band=(R_BAND[0], R_BAND[0] * c),
            n_core=N_CORE,
            wavelength_ref=WAVELENGTH,
            alpha=ALPHA,
        )
        print(
            f"  C = {c:4.1f}: α_eff = {pc.alpha_eff:.4f}, "
            f"realised C = {pc.contrast_realised:.3f}, "
            f"nn_from_band = {pc.nn_from_band}"
        )


if __name__ == "__main__":
    main()
