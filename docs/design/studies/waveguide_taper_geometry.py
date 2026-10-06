"""Prototype: a straight waveguide with tapered, rounded terminations.

Study script — touches no package code. It builds a closed boundary
``(f(θ), g(θ))`` for a finite-length dielectric waveguide segment: a flat
straight section of width ``w_guide``, narrowing through a smooth taper to a
finite neck width ``w_tip``, then closing into a rounded nose at each end —
and demonstrates that it can be fed straight into
:class:`pysie2d.geometry.Geometry` (which accepts externally supplied
``f, g, df, dg, ddf, ddg`` arrays; see its ``__init__`` docstring).

Why this needs its own construction
------------------------------------
Every shape :mod:`pysie2d.geometry` ships (the Gielis superformula) is a
closed, 2π-periodic, **entire-analytic** function of θ — no true corners
*and* no curvature discontinuities, which is what gives Kress quadrature its
spectral convergence (conventions §13). A waveguide-with-tapered-ends is not
naturally analytic: the obvious "flat sides + circular end caps" (a stadium)
has *continuous tangent* at each cap but *discontinuous curvature* (0 on the
straight side, 1/R on the arc) — the same "near-corner" defect the package's
own docs flag for flat-sided superellipses. That gives fast algebraic
convergence, not spectral.

The construction here avoids any join at all. Following the same idea as an
ellipse closing smoothly at its own vertical tangent point, the whole
boundary is one global formula:

    g(θ) = x_end · cos θ                  (long axis, spans ±x_end)
    f(θ) = Y(g(θ)) · sin θ                (transverse half-width)

``Y(x)`` is a smooth *envelope* — a ``tanh``-windowed step from the guide
half-width down to the tip half-width, built entirely from ``cos``, ``sin``,
``tanh`` (all entire on ℝ) — so ``f`` and ``g`` are themselves entire in θ,
with no piecewise gluing anywhere. The ``sin θ`` factor is what supplies the
rounded closure at each tip, exactly as it does for a plain ellipse: y → 0
with the curve staying regular even though dy/dx → ∞ there.

The one honest cost, flagged to and accepted by the user before writing this:
``tanh`` only *approaches* its asymptotes, so the "flat" middle and the
"tip" neck width are reached to within a tunable, exponentially small
residual (set by the margin between the taper transition and x_end) — never
exactly. That trade is what buys true spectral (not merely algebraic)
convergence.

Run: ``python docs/design/studies/waveguide_taper_geometry.py``
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
from pysie2d.geometry import Geometry  # noqa: E402
from pysie2d.parametrisation import Parametrisation  # noqa: E402

TWOPI = 2.0 * np.pi


# ---------------------------------------------------------------------------
# The envelope Y(x) and its x-derivatives
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TaperShape:
    """Parameters of a straight waveguide with tapered, rounded ends.

    Attributes:
        w_guide: Full width of the straight waveguide section (nm).
        w_tip: Full neck width the taper narrows to, before the final rounded
            closure (nm). Must be less than ``w_guide`` and greater than 0 —
            this is the "finite tip width," not a cusp.
        l_straight: Length of the flat straight section (nm).
        l_taper: Length over which the width transitions from ``w_guide`` to
            (approximately) ``w_tip``, on each end.
        taper_sharpness: Number of ``tanh`` decay lengths that fit inside
            ``l_taper``. Larger is a crisper, more corner-like transition
            (worse Fourier decay, more nn needed); smaller is gentler and
            resolves at lower nn. 4 is a reasonable default (residual
            ``1 - tanh(4) ≈ 7e-4`` of the step remains at the nominal taper
            end).
        nose_margin: Extra run, in units of the tanh width, appended beyond
            ``l_straight/2 + l_taper`` before the curve closes at ``x_end``.
            This is what keeps the "flat tip before rounding" reading
            honest: without it, the envelope would still be visibly
            transitioning right where the ``sin θ`` closure kicks in.
    """

    w_guide: float
    w_tip: float
    l_straight: float
    l_taper: float
    taper_sharpness: float = 4.0
    nose_margin: float = 2.0

    def __post_init__(self) -> None:
        """Guard the one silent failure mode: an overlapping window.

        ``_window`` is two tanh edges at ``±x1``; if ``x1`` is not
        comfortably larger than ``tanh_width`` they overlap and the
        "plateau" is never flat — it reads as a single smooth bump peaking
        well below ``w_guide/2``, with no visible straight section at all.
        """
        if self.x1 < 3.0 * self.tanh_width:
            raise ValueError(
                f"l_straight/2 = {self.x1:.1f} nm is not >> tanh_width = "
                f"{self.tanh_width:.1f} nm: the two taper edges overlap and "
                "there is no flat straight section. Increase l_straight or "
                "shrink l_taper / raise taper_sharpness."
            )

    @property
    def x1(self) -> float:
        """Half-length of the flat straight section (nm)."""
        return self.l_straight / 2.0

    @property
    def tanh_width(self) -> float:
        """``w`` in ``tanh((x∓x1)/w)`` — the taper's decay length (nm)."""
        return self.l_taper / (2.0 * self.taper_sharpness)

    @property
    def x_end(self) -> float:
        """Tip-to-tip half-length of the whole particle (nm)."""
        return self.x1 + self.l_taper + self.nose_margin * self.tanh_width


def _window(x: np.ndarray, x1: float, w: float) -> np.ndarray:
    """Smoothed rectangle: ≈1 for ``|x| ≲ x1``, ≈0 for ``|x| ≳ x1``.

    ``0.5·[tanh((x+x1)/w) − tanh((x−x1)/w)]``, even in ``x`` by construction
    (both terms swap and flip sign together under ``x → −x``).

    Args:
        x: Positions along the long axis (nm).
        x1: Half-width of the plateau (nm).
        w: Transition (tanh decay) length (nm).

    Returns:
        The window value at every ``x``.
    """
    return 0.5 * (np.tanh((x + x1) / w) - np.tanh((x - x1) / w))


def _window_d1(x: np.ndarray, x1: float, w: float) -> np.ndarray:
    """``d/dx`` of :func:`_window`, via ``sech² = 1 − tanh²``."""
    tp, tm = np.tanh((x + x1) / w), np.tanh((x - x1) / w)
    return 0.5 * ((1.0 - tp**2) - (1.0 - tm**2)) / w


def _window_d2(x: np.ndarray, x1: float, w: float) -> np.ndarray:
    """``d²/dx²`` of :func:`_window`, via ``d(sech²)/du = −2 sech²·tanh``."""
    tp, tm = np.tanh((x + x1) / w), np.tanh((x - x1) / w)
    sp2, sm2 = 1.0 - tp**2, 1.0 - tm**2
    return (-sp2 * tp + sm2 * tm) / w**2


def envelope(x: np.ndarray, shape: TaperShape, order: int = 0) -> np.ndarray:
    """Half-width envelope ``Y(x)`` (or a derivative), nm.

    ``Y = w_tip/2 + (w_guide/2 − w_tip/2) · window(x)`` — the tip half-width
    plus the guide/tip difference, scaled by the plateau window.

    Args:
        x: Positions along the long axis (nm).
        shape: The taper geometry.
        order: 0 for ``Y``, 1 for ``Y'``, 2 for ``Y''``.

    Returns:
        The requested quantity at every ``x``.
    """
    delta = 0.5 * (shape.w_guide - shape.w_tip)
    fn = {0: _window, 1: _window_d1, 2: _window_d2}[order]
    base = 0.5 * shape.w_tip if order == 0 else 0.0
    return base + delta * fn(x, shape.x1, shape.tanh_width)


# ---------------------------------------------------------------------------
# The closed boundary (f(θ), g(θ)) and its θ-derivatives
# ---------------------------------------------------------------------------


def boundary_arrays(
    theta: np.ndarray, shape: TaperShape
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """``f, g`` and their θ-derivatives for the tapered-waveguide boundary.

    ``g(θ) = x_end·cos θ`` is the long axis; ``f(θ) = Y(g(θ))·sin θ`` is the
    transverse half-width, closing smoothly at ``θ = 0, π`` exactly as an
    ellipse's ``b·sin θ`` does. Chain rule through ``g(θ)`` for ``Y``'s
    θ-derivatives, product rule for ``f``.

    Args:
        theta: Angles (rad).
        shape: The taper geometry.

    Returns:
        ``f, g, df, dg, ddf, ddg`` (θ-derivatives) at every angle.
    """
    x_end = shape.x_end
    g = x_end * np.cos(theta)
    dg = -x_end * np.sin(theta)
    ddg = -x_end * np.cos(theta)

    y = envelope(g, shape, 0)
    y1 = envelope(g, shape, 1)
    y2 = envelope(g, shape, 2)
    dy = y1 * dg
    ddy = y2 * dg**2 + y1 * ddg

    sin_t, cos_t = np.sin(theta), np.cos(theta)
    f = y * sin_t
    df = dy * sin_t + y * cos_t
    ddf = ddy * sin_t + 2.0 * dy * cos_t - y * sin_t
    return f, g, df, dg, ddf, ddg


# ---------------------------------------------------------------------------
# Self-check: analytic derivatives vs. complex-step differentiation
# ---------------------------------------------------------------------------


def check_derivatives(shape: TaperShape) -> None:
    """Verify the closed-form θ-derivatives against complex-step derivatives.

    ``f`` and ``g`` are built entirely from ``cos``, ``sin`` and ``tanh``, all
    of which numpy evaluates on complex arrays — so ``Im f(θ + ih)/h`` is a
    first derivative accurate to machine precision with no subtractive
    cancellation, and the same trick applied to the closed-form ``df/dθ``
    gives ``d²f/dθ²`` the same way. This checks the hand-differentiation
    above, nothing about the physics.

    Args:
        shape: The taper geometry.

    Raises:
        AssertionError: If a derivative disagrees with its complex-step
            counterpart beyond ``1e-9`` absolute (h = 1e-25 leaves no
            truncation error worth naming; the tolerance only guards against
            a genuine formula mistake).
    """
    h = 1e-25
    theta = np.linspace(0.3, TWOPI - 0.3, 37)  # avoid the branch seam at 0/2π

    def f_of(th: np.ndarray) -> np.ndarray:
        return boundary_arrays(th, shape)[0]

    def g_of(th: np.ndarray) -> np.ndarray:
        return boundary_arrays(th, shape)[1]

    f, g, df, dg, ddf, ddg = boundary_arrays(theta, shape)

    df_cs = np.imag(f_of(theta + 1j * h)) / h
    dg_cs = np.imag(g_of(theta + 1j * h)) / h
    assert np.max(np.abs(df - df_cs)) < 1e-9, "df/dθ disagrees with complex-step"
    assert np.max(np.abs(dg - dg_cs)) < 1e-9, "dg/dθ disagrees with complex-step"

    def df_of(th: np.ndarray) -> np.ndarray:
        return boundary_arrays(th, shape)[2]

    def dg_of(th: np.ndarray) -> np.ndarray:
        return boundary_arrays(th, shape)[3]

    ddf_cs = np.imag(df_of(theta + 1j * h)) / h
    ddg_cs = np.imag(dg_of(theta + 1j * h)) / h
    assert np.max(np.abs(ddf - ddf_cs)) < 1e-9, "d²f/dθ² disagrees with complex-step"
    assert np.max(np.abs(ddg - ddg_cs)) < 1e-9, "d²g/dθ² disagrees with complex-step"
    print("derivative self-check: analytic matches complex-step to < 1e-9")


# ---------------------------------------------------------------------------
# Spectral resolution: Fourier decay of the speed function
# ---------------------------------------------------------------------------


def harmonics_to_roundoff(shape: TaperShape, n_fine: int = 1 << 16) -> int:
    """Number of Fourier harmonics of the speed function above round-off.

    Mirrors the resolution check in ``Parametrisation.gielis``: samples the
    speed ``γ(θ) = √(f'² + g'²)`` on a fine uniform grid, takes its real FFT,
    and reports the last harmonic whose relative magnitude exceeds machine
    epsilon. This is the direct evidence for the "spectral, not algebraic"
    claim above — a fast-decaying spectrum here is what lets Kress quadrature
    reach round-off at a small ``nn``.

    Args:
        shape: The taper geometry.
        n_fine: Fine-grid size for the FFT.

    Returns:
        The index of the last harmonic above ``4·eps`` relative to the mean.
    """
    theta = np.linspace(0.0, TWOPI, n_fine, endpoint=False)
    _, _, df, dg, _, _ = boundary_arrays(theta, shape)
    gam = np.sqrt(df**2 + dg**2)
    spec = np.fft.rfft(gam) / n_fine
    decay = np.abs(spec[1:]) / np.abs(spec[0].real)
    eps = np.finfo(float).eps
    above = np.flatnonzero(decay > 4.0 * eps)
    return int(above[-1] + 1) if above.size else 0


def main() -> None:
    """Build the shape, verify derivatives, report resolution, plot it."""
    shape = TaperShape(
        w_guide=400.0,
        w_tip=150.0,
        l_straight=12000.0,  # >> tanh_width, else the two taper edges overlap
        l_taper=8000.0,  # adiabatic: taper length >> width change, per user request
        taper_sharpness=3.0,
    )
    print(f"x_end (tip-to-tip half-length) = {shape.x_end:.1f} nm")
    print(f"tanh transition width          = {shape.tanh_width:.1f} nm")

    check_derivatives(shape)

    for sharpness in (2.0, 4.0, 8.0):
        # Hold l_straight/2 at 4x this sharpness's tanh_width so every entry
        # clears the __post_init__ overlap guard, whatever sharpness does to
        # the transition length.
        w_at_sharpness = shape.l_taper / (2.0 * sharpness)
        s = TaperShape(
            w_guide=shape.w_guide,
            w_tip=shape.w_tip,
            l_straight=8.0 * w_at_sharpness,
            l_taper=shape.l_taper,
            taper_sharpness=sharpness,
        )
        k = harmonics_to_roundoff(s)
        print(f"taper_sharpness={sharpness:>4.1f}  ->  harmonics to round-off K={k}")

    k_main = harmonics_to_roundoff(shape)
    print(f"this shape (sharpness={shape.taper_sharpness}) -> K={k_main}")
    nn = 1 << (k_main * 2).bit_length()  # comfortably above K, for a resolved plot
    theta = TWOPI * (np.arange(nn) + 0.5) / nn
    f, g, df, dg, ddf, ddg = boundary_arrays(theta, shape)
    geom = Geometry(
        f, g, df, dg, ddf, ddg,
        rad=shape.x_end,
        parametrisation=Parametrisation.uniform_theta(),
    )
    print(f"Geometry built: n_pts={geom.n_pts}, is_circle={geom.is_circle}")

    # Long axis (g) on the horizontal, transverse width (f) on the vertical.
    # At true (equal) aspect a ~30:1 long:wide waveguide is an unreadable
    # sliver, so the top panel is that honest to-scale view and the bottom
    # panel exaggerates the transverse axis so the taper profile itself is
    # actually visible.
    gg = np.append(geom.g, geom.g[0])
    ff = np.append(geom.f, geom.f[0])
    fig, (ax_scale, ax_zoom) = plt.subplots(2, 1, figsize=(14, 6))

    ax_scale.plot(gg, ff, "-", lw=0.8)
    ax_scale.set_aspect("equal")
    ax_scale.set_ylabel("f (transverse, nm)")
    ax_scale.set_title("To scale (equal aspect)")

    ax_zoom.plot(gg, ff, "-", lw=0.8)
    ax_zoom.set_xlabel("g (long axis, nm)")
    ax_zoom.set_ylabel("f (transverse, nm)")
    ax_zoom.set_title("Transverse axis exaggerated, to show the taper profile")

    fig.suptitle("Tapered waveguide boundary prototype (adiabatic taper)")
    fig.tight_layout()
    out = Path(__file__).with_name("waveguide_taper_geometry.png")
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"figure written to {out}")


if __name__ == "__main__":
    main()
