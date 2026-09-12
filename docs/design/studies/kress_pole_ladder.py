"""The orientation ladder again, under Kress instead of the shipped scheme.

Same question as `g2_pole_ladder.py` and the same shapes — equal-area ellipses
from the circle to aspect 4, tracking the TE n = 0 pole — but assembled with
`kress_t.kress_matrix` instead of shipped 0.5.0. The first ladder found that
grading *loses* on a first-order scheme because the error is set by the
worst-resolved node; the point of repeating it here is that Kress removes that
mechanism, so this is where grading can finally be asked to pay.

Two changes follow from the accuracy, and both simplify the study:

- **The reference is the full-precision analytic Mie pole on the circle**, and
  a high-`nn` Kress solution on the ellipses. The anchor constant quoted in
  `tests/test_qnm.py` is rounded to five decimals, which reads as a floor of
  1.259e-6 — invisible against the shipped 0.38 nm error, and the dominant term
  under Kress. It is recomputed here by Newton on `qnm_denominator`.
- **`nn` is swept directly and the two schemes are compared at equal `nn`**,
  rather than through an `R` band. Grading is set by the band's *contrast*; the
  band's level is what `nn` would otherwise fix, and fixing `nn` by hand is the
  simplest way to put the two node distributions on the same budget.

Run: ``python docs/design/studies/kress_pole_ladder.py``
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import newton

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))

from adaptive_density import build  # noqa: E402
from g2_pole_ladder import ellipse  # noqa: E402
from kress_t import kress_matrix  # noqa: E402

from pysie2d import Material  # noqa: E402
from pysie2d.beyn import beyn_modes  # noqa: E402
from pysie2d.geometry import gielis  # noqa: E402
from pysie2d.reference.mie import qnm_denominator  # noqa: E402

RAD = 200.0
MAT = Material(n_core=3.0, n_clad=1.0, pol=2)
BAND = (20.0, 80.0)  # sets the grading contrast; nn is swept independently
N_SIDE = 12  # 6 leaves a 5e-6 contour floor, which Kress sits below
ASPECT_STOPS = (1.0, 2.0, 3.0, 4.0)
ASPECTS = np.round(np.arange(1.0, 4.001, 0.1), 3)
NN_SWEEP = (20, 30, 40, 60, 80, 120)
NN_REF = 200


def exact_circle_pole() -> complex:
    """The TE n=0 Mie pole at full precision, not the 5-dp table constant."""
    seed = 2.0 * np.pi * RAD / (530.83214 + 26.37850j)
    x = newton(
        lambda z: qnm_denominator(0, z, MAT.nc, 2), seed, tol=1.0e-14, maxiter=100
    )
    return complex(2.0 * np.pi * RAD / x)


EXACT = exact_circle_pole()


def nodes(shape: dict, nn: int, lam_ref: float, *, uniform: bool):
    """Boundary samples at nodes equispaced in `t`, graded or constant density."""
    par = build(
        shape, BAND, MAT.n_core, wavelength=lam_ref, uniform_arc=uniform, nn_force=nn
    )
    f, g, *_ = gielis(par.theta, **shape)
    return f, g


def find_pole(f, g, predict: complex, *, half=6.0, n_probe=10):
    """Beyn in a small box centred on a secant prediction.

    A saturated probe is caught rather than raised: at the coarse end of the
    sweep on an elongated shape the discrete operator is so far from the
    continuous one that the box fills with spurious spectrum, which is a
    *result* of that resolution and not an error to abort the sweep on.
    """

    def builder(lam):
        k_bg = MAT.wnum_bg(lam)
        return kress_matrix(f, g, k_bg, MAT.nc * k_bg)

    for factor in (1.0, 2.0, 4.0):
        h = half * factor
        lo = complex(predict.real - h, max(predict.imag - h, 0.5))
        hi = complex(predict.real + h, predict.imag + h)
        try:
            res = beyn_modes(builder, lo, hi, n_quad_per_side=N_SIDE, n_probe=n_probe)
        except ValueError:
            return None, np.nan
        if res.eigenvalues.size:
            w = res.eigenvalues
            return complex(w[np.argmin(np.abs(w - predict))]), factor
    return None, np.nan


def trajectory(nn: int, *, uniform: bool):
    """Secant continuation of the pole from the circle to aspect 4."""
    pts = []
    for aspect in ASPECTS:
        if len(pts) >= 2:
            (a0, l0), (a1, l1) = pts[-2], pts[-1]
            predict = l1 + (l1 - l0) * (aspect - a1) / (a1 - a0)
        else:
            predict = pts[-1][1] if pts else EXACT
        shape = ellipse(float(aspect))
        f, g = nodes(shape, nn, predict.real, uniform=uniform)
        lam, factor = find_pole(f, g, predict)
        if lam is None:
            print(f"  A={aspect:4.2f} LOST")
            break
        if factor > 1.0:
            print(f"  A={aspect:4.2f} box widened x{factor:.0f}")
        pts.append((float(aspect), lam))
    return pts


def main() -> None:
    """Reference trajectory, then the equal-nn sweep at each aspect stop."""
    t0 = time.time()
    print(f"exact circle pole (Newton on qnm_denominator): {EXACT!r}")

    ref_pts = trajectory(NN_REF, uniform=True)
    ref = dict(ref_pts)
    print(
        f"reference trajectory at nn={NN_REF}: "
        f"circle err {abs(ref[1.0] - EXACT):.3e} nm, "
        f"aspect 4 at {ref[4.0].real:.4f}{ref[4.0].imag:+.4f}j"
    )

    rows = {}
    print("\nequal-nn sweep (error in nm)")
    print(f"{'A':>4} {'nn':>5} {'adaptive':>12} {'const density':>14} {'ratio':>7}")
    for aspect in ASPECT_STOPS:
        target = EXACT if aspect == 1.0 else ref[aspect]
        shape = ellipse(aspect)
        for nn in NN_SWEEP:
            out = []
            for uniform in (False, True):
                f, g = nodes(shape, nn, target.real, uniform=uniform)
                lam, _ = find_pole(f, g, target, half=8.0)
                out.append(abs(lam - target) if lam is not None else np.nan)
            rows[(aspect, nn)] = out
            ratio = out[0] / out[1] if out[1] else np.nan
            print(f"{aspect:4.1f} {nn:5d} {out[0]:12.3e} {out[1]:14.3e} {ratio:7.2f}")

    # ---------------- figure ----------------
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.6))

    ax = axes[0]
    lams = np.array([p[1] for p in ref_pts])
    sc = ax.scatter(
        lams.real, lams.imag, c=[p[0] for p in ref_pts], cmap="viridis", s=26, zorder=3
    )
    ax.plot(lams.real, lams.imag, "-", color="0.6", lw=0.9, zorder=2)
    ax.scatter(
        [EXACT.real],
        [EXACT.imag],
        marker="*",
        s=170,
        color="k",
        zorder=4,
        label="analytic Mie pole",
    )
    fig.colorbar(sc, ax=ax, label="aspect b/a", fraction=0.046)
    ax.set_xlabel("Re λ (nm)")
    ax.set_ylabel("Im λ (nm)")
    ax.set_title(f"same trajectory, Kress at nn={NN_REF}", fontsize=10)
    ax.legend(fontsize=8)

    # Aspect 1 is the anchored panel; aspect 3 shows the crossover. Aspect 4 is
    # omitted from the figure because the mode is not identifiable at the coarse
    # end there under either scheme — the table carries it.
    for j, aspect in enumerate((1.0, 3.0)):
        ax = axes[j + 1]
        for k, (name, col) in enumerate(
            (("adaptive", "C0"), ("const density", "crimson"))
        ):
            errs = [rows[(aspect, nn)][k] for nn in NN_SWEEP]
            ax.semilogy(
                NN_SWEEP, np.maximum(errs, 1e-14), "-o", ms=4, color=col, label=name
            )
        ax.set_xlabel("nn")
        ax.set_ylabel("|λ − reference| (nm)")
        kind = "analytic Mie" if aspect == 1.0 else f"Kress nn={NN_REF}"
        ax.set_title(f"aspect {aspect:.0f} · vs {kind}", fontsize=10)
        ax.legend(fontsize=8)

    fig.suptitle(
        "v0.6 — the orientation ladder under Kress: spectral convergence, and "
        "what grading does once the error is no longer set by max Δs",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    out = Path(__file__).with_name("kress_pole_ladder.png")
    fig.savefig(out, dpi=140)
    print(f"\nwrote {out}  ({time.time() - t0:.0f} s)")


if __name__ == "__main__":
    main()
