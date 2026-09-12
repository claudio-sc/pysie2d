"""Orientation study: adaptive nodes along a circle → sharp-ellipse ladder.

Preliminary and for orientation only. Study script — touches no package code,
and production code is written from the code-specs, never from here.

What it measures, and what it cannot
------------------------------------
The adaptive density of `adaptive_density.py` is fed to **shipped 0.5.0**
through ``Geometry.gielis(theta=...)``. Kress is not implemented, so the
quadrature here is the Maradudin scheme and convergence is **first order by
construction**: grading can move the error *constant*, never the rate. That is
the whole point of running it anyway — it is the one question that can be asked
before G0, and handoff §9 shows the answer is not obvious (there, uniform arc
length was markedly *worse* than uniform θ on a star, which is backwards).

The ladder
----------
Equal-area ellipses, `m = 4`, `n1 = n2 = n3 = 2`, `a = 1/√A`, `b = √A`, so
``a·b = 1`` and the area is the circle's at every aspect `A` — no rescaling.
Scale covariance (conventions §9) makes a pure size change exactly
``λ → s·λ``, so holding the area removes that trivial motion and what remains
in the pole trajectory is the shape effect, which is the thing being tracked.

The tracked mode is **TE n = 0 at 530.83214 + 26.37850j** (rad 200 nm,
n_core 3.0, n_clad 1.0), the simple anchor of ``tests/test_qnm.py``. Angular
order 0, so it stays simple under elongation and there is no degenerate pair to
swap labels. It is crowded in Re λ (TE n=2 at 550.47+20.24j), so the box
follows the pole and the nearest *other* mode is reported at every step as the
ambiguity tell.

The two references
------------------
- **Circle**: the analytic Mie pole. The only genuinely independent anchor.
- **Ellipse**: two-rung Richardson in ``R`` (§12, exponent pinned at 1). A
  self-convergence reference — adequate to measure a constant, and **not** an
  independent validation anchor (CLAUDE.md non-negotiable 3). An ellipse
  against Mathieu functions remains the open item.

`λ_ref` for the density is fixed per shape at the previous step's Re λ, never
taken from the contour point being assembled — invariant 13.2, without which
``M(λ)`` loses the holomorphy Beyn's premise needs.

Run: ``python docs/design/studies/g2_pole_ladder.py``
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))

from adaptive_density import build  # noqa: E402

from pysie2d import (  # noqa: E402
    Geometry,
    Material,
    QNMSolver,
    richardson_limit,
)

# The D2 fixture of tests/conftest.py, and the simple TE anchor of
# tests/test_qnm.py — the analytic Mie pole, vacuum nm.
RAD = 200.0
N_CORE = 3.0
POL = 2
ANCHOR = 530.83214 + 26.37850j
BAND = (20.0, 80.0)
N_SIDE = 6  # measured in tests/test_qnm.py to be far below the n_pts error
N_PROBE = 8

# Rungs in R, never in raw n_pts (D17). Contrast held at 4 so every rung asks
# for the same *grading* and only the level moves.
R_RUNGS = (10.0, 20.0, 40.0)
ASPECT_STOPS = (1.0, 2.0, 3.0, 4.0)
# Step 0.05, not 0.25. At 0.25 the pole moves ~9 nm in Im λ per step — the box
# half-height — and nearest-neighbour tracking hops onto a different branch:
# measured Q swinging 10 → 48 → 25 → 40 along what should be a smooth
# trajectory, and two walks with different step sizes landing on different
# modes at the same aspect. Mode identification is the fragile part of this
# study, not the quadrature.
ASPECTS = np.round(np.arange(1.0, 4.001, 0.05), 3)


def ellipse(aspect: float) -> dict:
    """Equal-area ellipse as Gielis parameters.

    ``n1 = n2 = n3 = 2`` with ``a ≠ b`` is an *exact* ellipse, semi-axis
    ``b·rad`` along x and ``a·rad`` along z. With ``a·b = 1`` the area is
    ``π·rad²`` for every aspect, so nothing rescales.

    Args:
        aspect: ``b/a``, the x-to-z semi-axis ratio. 1 is the circle.

    Returns:
        Keyword arguments for :func:`pysie2d.geometry.gielis`, minus theta.
    """
    return {
        "rad": RAD,
        "a": 1.0 / np.sqrt(aspect),
        "b": np.sqrt(aspect),
        "m": 4,
        "n1": 2.0,
        "n2": 2.0,
        "n3": 2.0,
        "x0": 0.0,
        "z0": 0.0,
    }


def geometry(shape: dict, band, lam_ref: float, *, uniform=False, nn_force=None):
    """Build a Geometry on adaptive (or const-density) nodes from an R band."""
    par = build(
        shape,
        band,
        N_CORE,
        wavelength=lam_ref,
        uniform_arc=uniform,
        nn_force=nn_force,
    )
    geom = Geometry.gielis(
        rad=shape["rad"],
        m=shape["m"],
        n1=shape["n1"],
        n2=shape["n2"],
        n3=shape["n3"],
        a=shape["a"],
        b=shape["b"],
        theta=par.theta,
    )
    return geom, par


def find_pole(geom, predict: complex, *, half=6.0, widen=(1.0, 2.0, 4.0)):
    """Beyn in a box centred on a *predicted* λ; return the nearest mode.

    The centre is a secant prediction, not the previous λ: with a predictor the
    box only has to contain the curvature of the trajectory, so it can be small
    and the nearest-neighbour pick is unambiguous. Centred on the previous λ
    instead, the box has to be as large as the whole step, and in a crowded
    region the tracker walks onto a neighbouring branch silently.

    Widens the box only if nothing is found, and reports the width used — a
    widened box is the tell that the step was too long for the predictor.

    Returns:
        (λ, edge_margin, gap to the nearest other mode, mode count, box factor)
        with λ None if no mode is found even at the widest box.
    """
    solver = QNMSolver(geom, Material(n_core=N_CORE, n_clad=1.0, pol=POL))
    for factor in widen:
        h = half * factor
        lo = complex(predict.real - h, max(predict.imag - h, 0.5))
        hi = complex(predict.real + h, predict.imag + h)
        res = solver.modes(lo, hi, n_quad_per_side=N_SIDE, n_probe=N_PROBE)
        if res.wavelengths.size:
            k = int(np.argmin(np.abs(res.wavelengths - predict)))
            others = np.delete(res.wavelengths, k)
            gap = (
                float(np.min(np.abs(others - res.wavelengths[k])))
                if others.size
                else np.inf
            )
            return (
                complex(res.wavelengths[k]),
                float(res.edge_margin[k]),
                gap,
                int(res.wavelengths.size),
                factor,
            )
    return None, np.nan, np.nan, 0, np.nan


def trajectory(scheme="adaptive", verbose=True):
    """Follow the mode from the circle to aspect 4 by secant continuation.

    Returns:
        A list of ``(aspect, λ, nn, edge_margin, gap, box factor)`` per step.
    """
    pts = []
    if verbose:
        print(f"\n{scheme} trajectory, band {BAND[0]:.0f}-{BAND[1]:.0f}")
    for i, aspect in enumerate(ASPECTS):
        # Secant predictor: linear extrapolation of the last two steps.
        if len(pts) >= 2:
            (a0, l0, *_), (a1, l1, *_) = pts[-2], pts[-1]
            predict = l1 + (l1 - l0) * (aspect - a1) / (a1 - a0)
        else:
            predict = pts[-1][1] if pts else ANCHOR
        shape = ellipse(float(aspect))
        geom, par = geometry(
            shape, BAND, predict.real, uniform=(scheme == "uniform")
        )
        lam, margin, gap, count, factor = find_pole(geom, predict)
        if lam is None:
            print(f"  A={aspect:4.2f} LOST — no mode found")
            break
        pts.append((float(aspect), lam, par.nn, margin, gap, factor))
        if verbose and (i % 5 == 0 or factor > 1.0):
            flag = "" if factor == 1.0 else f"  BOX x{factor:.0f}"
            print(
                f"  A={aspect:4.2f} nn={par.nn:4d} λ={lam.real:9.4f}"
                f"{lam.imag:+8.4f}j  Q={lam.real / (2 * lam.imag):7.2f}"
                f"  margin={margin:5.3f}  gap={gap:6.2f}  modes={count}{flag}"
            )
    return pts


def convergence(traj):
    """Resolution ladder in R at each aspect stop, adaptive vs const density.

    The guess at each stop comes from the **fine** trajectory, never from a
    coarser re-walk: two walks with different step sizes were measured landing
    on different modes at the same aspect.
    """
    by_aspect = {p[0]: p[1] for p in traj}
    rows = []
    for aspect in ASPECT_STOPS:
        shape = ellipse(aspect)
        guess = by_aspect.get(aspect)
        if guess is None:
            print(f"  A={aspect:3.1f} skipped — trajectory did not reach it")
            continue
        for r_min in R_RUNGS:
            band = (r_min, 4.0 * r_min)
            geom_a, par_a = geometry(shape, band, guess.real)
            lam_a, margin_a, _, _, _ = find_pole(geom_a, guess)
            # Const density at the SAME nn: the fair baseline, since grading at
            # fixed R_min adds nodes rather than saving them.
            geom_u, par_u = geometry(
                shape, band, guess.real, uniform=True, nn_force=par_a.nn
            )
            lam_u, _, _, _, _ = find_pole(geom_u, guess)
            if lam_a is None or lam_u is None:
                print(f"  A={aspect:3.1f} R={r_min:5.1f} LOST — no mode found")
                continue
            rows.append(
                {
                    "aspect": aspect,
                    "r_min": r_min,
                    "nn": par_a.nn,
                    "lam_a": lam_a,
                    "lam_u": lam_u,
                    "margin": margin_a,
                }
            )
            print(
                f"  A={aspect:3.1f} R={r_min:5.1f} nn={par_a.nn:4d}"
                f"  adaptive {lam_a.real:9.4f}{lam_a.imag:+8.4f}j"
                f"  uniform {lam_u.real:9.4f}{lam_u.imag:+8.4f}j"
            )
    return rows


def _order(errs, nns):
    """Observed convergence order between consecutive rungs."""
    return [
        np.log(abs(errs[i]) / abs(errs[i + 1])) / np.log(nns[i + 1] / nns[i])
        for i in range(len(errs) - 1)
    ]


def main() -> None:
    """Run both halves, print the tables, write the figure."""
    t0 = time.time()
    traj = {s: trajectory(s) for s in ("adaptive", "uniform")}

    print("\nconvergence ladder")
    rows = convergence(traj["adaptive"])

    # Per aspect: Richardson limit from the two finest rungs, then the error of
    # each rung against it. On the circle the analytic Mie pole is used instead
    # — the one genuinely independent reference available.
    print("\nreference and orders")
    summary = {}
    for aspect in ASPECT_STOPS:
        sel = [r for r in rows if r["aspect"] == aspect]
        nns = [r["nn"] for r in sel]
        for key in ("lam_a", "lam_u"):
            lams = [r[key] for r in sel]
            if aspect == 1.0:
                ref = ANCHOR
                ref_kind = "analytic Mie"
            else:
                ref = richardson_limit(lams[-2], lams[-1], nns[-2], nns[-1])
                ref_kind = "Richardson (self)"
            errs = [lam - ref for lam in lams]
            summary[(aspect, key)] = (nns, errs, ref, ref_kind)
            orders = _order([e.real for e in errs], nns)
            print(
                f"  A={aspect:3.1f} {key[-1]}  ref={ref.real:9.4f}{ref.imag:+8.4f}j"
                f" ({ref_kind:17s}) |err| "
                + " ".join(f"{abs(e):.3e}" for e in errs)
                + "  order(Re) "
                + " ".join(f"{o:.2f}" for o in orders)
            )

    # ---------------- figure ----------------
    fig = plt.figure(figsize=(13.5, 9.5))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.15, 1.0], hspace=0.38, wspace=0.32)

    # (a) the trajectory in the complex λ-plane
    ax = fig.add_subplot(gs[0, :2])
    for scheme, col, mk in (("adaptive", "C0", "o"), ("uniform", "crimson", "x")):
        pts = traj[scheme]
        lams = np.array([p[1] for p in pts])
        ax.plot(lams.real, lams.imag, "-", color=col, lw=1.0, alpha=0.7)
        sc = ax.scatter(lams.real, lams.imag, c=[p[0] for p in pts], cmap="viridis",
                        s=34, marker=mk, zorder=3, label=f"{scheme} nodes")
    ax.scatter([ANCHOR.real], [ANCHOR.imag], marker="*", s=180, color="k", zorder=4,
               label="analytic Mie pole (circle)")
    fig.colorbar(sc, ax=ax, label="aspect b/a", fraction=0.04)
    ax.set_xlabel("Re λ (nm)")
    ax.set_ylabel("Im λ (nm)")
    ax.set_title("TE n=0 pole, circle → aspect 4 at equal area", fontsize=10)
    ax.legend(fontsize=8)

    # (b) the physical identification check, and the cost
    ax = fig.add_subplot(gs[0, 2])
    pts = traj["adaptive"]
    asp = np.array([p[0] for p in pts])
    lams = np.array([p[1] for p in pts])
    ax.plot(asp, lams.real / ANCHOR.real, "-o", ms=3, color="C0",
            label="Re λ(A) / Re λ(1)")
    ax.plot(asp, 1.0 / np.sqrt(asp), "--", color="0.4", lw=1.0,
            label="minor semi-axis, $A^{-1/2}$")
    ax.set_xlabel("aspect b/a")
    ax.set_ylabel("relative Re λ")
    ax.set_title("identification: the n=0 mode\nfollows the minor axis",
                 fontsize=10)
    ax.legend(fontsize=8)

    ax = fig.add_subplot(gs[1, 2])
    for scheme, col in (("adaptive", "C0"), ("uniform", "crimson")):
        pts = traj[scheme]
        ax.plot([p[0] for p in pts], [p[2] for p in pts], "-o", ms=3, color=col,
                label=f"{scheme} nn")
    ax.set_xlabel("aspect b/a")
    ax.set_ylabel("nn from the band")
    ax.set_title(f"cost of holding R ≥ {BAND[0]:.0f}", fontsize=10)
    ax.legend(fontsize=8, loc="upper left")
    axq = ax.twinx()
    pts = traj["adaptive"]
    axq.plot([p[0] for p in pts], [p[1].real / (2 * p[1].imag) for p in pts],
             "-", color="0.5", lw=1.0)
    axq.set_ylabel("Q = Re λ / 2 Im λ", color="0.45", fontsize=9)

    # (c) convergence at the circle (genuine Mie anchor) and the sharp end
    for j, aspect in enumerate((ASPECT_STOPS[0], ASPECT_STOPS[-1])):
        ax = fig.add_subplot(gs[1, j])
        for key, col, name in (("lam_a", "C0", "adaptive"),
                               ("lam_u", "crimson", "const density")):
            nns, errs, ref, kind = summary[(aspect, key)]
            ax.loglog(nns, [abs(e) for e in errs], "-o", ms=4, color=col, label=name)
        guide = np.array(nns, dtype=float)
        ax.loglog(guide, abs(errs[0]) * guide[0] / guide, "--", color="0.6", lw=0.9,
                  label="first order")
        ax.set_xlabel("nn")
        ax.set_ylabel("|λ − reference| (nm)")
        ax.set_title(f"aspect {aspect:.0f} · {kind}", fontsize=10)
        ax.legend(fontsize=7.5)

    fig.suptitle(
        "v0.6 orientation study — adaptive nodes on shipped (first-order) "
        "quadrature, R band 20–80\n"
        "equal-area ellipses; the rate is 1 by construction, so what is being "
        "read here is the error constant",
        fontsize=12,
    )
    out = Path(__file__).with_name("pole_ladder.png")
    fig.savefig(out, dpi=140, bbox_inches="tight")
    print(f"\nwrote {out}  ({time.time() - t0:.0f} s)")


if __name__ == "__main__":
    main()
