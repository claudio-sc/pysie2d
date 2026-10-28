"""S2 — gradient anchors for S and T on cosine-profile boundaries.

Study script — touches no package code. Plan: ``README.md`` §S2.

Boundary r(θ) = r₀ + Σ a_m cos mθ on the uniform-θ map, with f = r sin θ,
g = r cos θ as in ``Geometry.gielis``, so the profile is mirror-symmetric under
x → −x (about the waveguide normal). The mode is the n = 46 p = 1 WGM
(D ≈ 12.8 µm), the cheapest member of S1's set. Its ±n pair is degenerate on
the circle and goes through the secular branch of ``sensitivity`` throughout.

- **A1 — scale (§9).** S and T at r₀·(1 ± 10⁻³) equal S and T at r₀: M sees
  only k·rad, so both are functions of the mode alone.
- **A2 — selection rule.** In first-order degenerate perturbation theory the
  cos mθ harmonic acts on the {e^{inθ}, e^{−inθ}} pair through ∫cos mθ dθ
  (diagonal) and ∫cos mθ e^{±2inθ} dθ (off-diagonal). Both vanish unless
  m = 0 or m = 2n, so dλ/da_m = 0 for 1 ≤ m ≤ 30 at n = 46 — an analytic zero,
  independent of the BIE. Positive control, so the zero cannot pass by
  ``sensitivity`` returning nothing: m = 2n = 92 must split the pair ±, and
  agree with a central difference of contour-located poles.
- **A2b — curvature at the circle.** The same rule makes the Hessian of λ in
  the a_m diagonal at the circle for m, m' ≤ 30 (a cross term needs
  m + m' = 2n at second order), and keeps the pair degenerate. So the 30
  diagonal second differences of S and of the drift ratio R = |T|/S decide
  whether the circle is a local optimum of the cosine family.
- **A3 — a generic shape.** (a) Shape sensitivity dλ/da_m against a central
  difference of contour-located poles. (b) dS/da_m and dT/da_m by the cheap
  route S3 would use — poles at p ± h *predicted* as λ ± h·dλ/da_m, then the
  ε″ and n_core sensitivities there — against the brute-force route with the
  poles located by the contour at p ± h. The prediction misses ½h²·λ'' at
  both ends; it is the same at both, so it cancels in the central difference.

Run: ``uv run python docs/design/studies/interior-enhancement/s2_gradient_anchor.py``
"""

from __future__ import annotations

import dataclasses
import sys
import time
from pathlib import Path

import numpy as np

from pysie2d import Geometry, Material, Parametrisation, QNMResult, QNMSolver

sys.path.insert(0, str(Path(__file__).resolve().parent))
from s0_qrad_budget import N_CLAD, N_CORE, POL  # noqa: E402
from s1_circle_sensitivity import EPSI0, analytic, gain  # noqa: E402

N_MODE, N_PTS = 46, 260  # S1: n_pts = 260 is at the floor for n = 46
PM = Parametrisation.uniform_theta()
# Shape step in nm. ε·r₀/h ≈ 1.4e-10 cancellation at r₀ ≈ 6.4 µm; truncation
# (h·k·m)² is far below that. SHAPE_STEP = 1e-5 nm would sit at 1.4e-7.
H_SHAPE = 1.0e-2
H_MAT = 1.0e-5  # S1: O(h²) truncation 6e-10 on dλ/dε″ at this step


def cosine_geometry(r0: float, a: dict[int, float], n_pts: int = N_PTS) -> Geometry:
    """r(θ) = r₀ + Σ a_m cos mθ, sampled on the shared uniform-θ map."""
    nodes = PM.nodes(n_pts)
    th = nodes.theta
    r, rp, rpp = np.full_like(th, r0), np.zeros_like(th), np.zeros_like(th)
    for m, am in a.items():
        r += am * np.cos(m * th)
        rp -= am * m * np.sin(m * th)
        rpp -= am * m * m * np.cos(m * th)
    s, c = np.sin(th), np.cos(th)
    # Uniform map: w' = 1, w'' = 0, so θ-derivatives are t-derivatives.
    return Geometry(
        r * s,
        r * c,
        r * c + s * rp,
        -r * s + c * rp,
        rpp * s + 2.0 * rp * c - r * s,
        rpp * c - 2.0 * rp * s - r * c,
        rad=r0,
        parametrisation=PM,
    )


def material(d_eps: float = 0.0, d_n: float = 0.0) -> Material:
    """The S1 baseline material, optionally displaced in ε″ or n_core."""
    return Material(n_core=N_CORE + d_n, n_clad=N_CLAD, pol=POL, epsi=EPSI0 + d_eps)


def find(geom: Geometry, guess: complex) -> QNMResult:
    """Contour-locate the ±n pair near `guess`; the box is square on Im λ."""
    h = 0.6 * guess.imag
    res = QNMSolver(geom, material()).modes(
        complex(guess.real - h, 0.4 * guess.imag),
        complex(guess.real + h, 1.6 * guess.imag),
    )
    if res.n_modes != 2:
        raise RuntimeError(f"expected the ±n pair, found {res.n_modes} modes")
    return res


def s_and_t(res: QNMResult) -> tuple[np.ndarray, np.ndarray]:
    """S and T of each mode of `res`, in sorted-λ order."""
    geom = res.geometry
    d_eps = res.sensitivity(lambda d: (geom, material(d_eps=d)), step=H_MAT)
    d_n = res.sensitivity(lambda d: (geom, material(d_n=d)), step=H_MAT)
    lam = np.sort_complex(res.wavelengths)
    return (
        np.array([gain(w, d) for w, d in zip(lam, d_eps, strict=True)]),
        np.array([gain(w, d) for w, d in zip(lam, d_n, strict=True)]),
    )


def shape_sens(res: QNMResult, r0: float, a: dict[int, float], m: int) -> np.ndarray:
    """dλ/da_m of each mode (sorted) at shape `a`, by the adjoint."""

    def at(d: float) -> tuple[Geometry, Material]:
        return cosine_geometry(r0, {**a, m: a.get(m, 0.0) + d}), material()

    return res.sensitivity(at, step=H_SHAPE)


def a1_scale(r0: float, lam0: complex) -> None:
    """S and T at three radii: equal to the S1 floor by scale covariance."""
    st = []
    for s in (1.0 - 1e-3, 1.0, 1.0 + 1e-3):
        st.append(s_and_t(find(cosine_geometry(r0 * s, {}), lam0 * s)))
    s_all = np.array([x[0] for x in st])
    t_all = np.array([x[1] for x in st])
    print(
        f"A1  spread over r₀(1±1e-3): S {np.ptp(s_all) / abs(s_all.mean()):.1e}, "
        f"T {np.ptp(t_all) / abs(t_all.mean()):.1e}"
    )


def a2_selection(r0: float, lam0: complex) -> None:
    """dλ/da_m = 0 at the circle for m ≤ 30; m = 2n splits the pair."""
    base = find(cosine_geometry(r0, {}), lam0)
    scale = abs(lam0) / r0  # |dλ/dr₀| = |λ|/r₀ exactly (§9)
    worst = max(
        float(np.max(np.abs(shape_sens(base, r0, {}, m)))) / scale for m in range(1, 31)
    )
    print(f"A2  max_m≤30 |dλ/da_m| / (|λ|/r₀) = {worst:.1e}")
    m = 2 * N_MODE
    adj = shape_sens(base, r0, {}, m)
    # A first-order split of |dλ/da| ≈ |λ|/r₀ moves the poles by h·0.2: h must
    # keep that inside the box (±0.6·Im λ ≈ ±0.01 nm). The contour floor
    # 1e-12 nm over 2h stays ≤ 5e-10 nm/nm, below 1e-8 of |dλ/da|.
    for h in (1.0e-3, 4.0e-3):
        lp = np.sort_complex(find(cosine_geometry(r0, {m: h}), lam0).wavelengths)
        lm = np.sort_complex(find(cosine_geometry(r0, {m: -h}), lam0).wavelengths)
        # A ± split sorts the partners in opposite order at ±h: partner k at +h
        # continues partner 1−k at −h.
        fd = (lp - lm[::-1]) / (2.0 * h)
        fd = np.sort_complex(fd)
        print(
            f"A2  m={m}: adjoint {adj[0]:.6e}, {adj[1]:.6e} (/(|λ|/r₀) = "
            f"{abs(adj[0]) / scale:.3f}); FD h={h}: rel diff "
            f"{np.max(np.abs(fd - adj)) / np.max(np.abs(adj)):.1e}"
        )


def a2b_curvature(r0: float, lam0: complex) -> None:
    """Diagonal second differences of S and ln R at the circle, m = 1..30."""
    s0, t0 = s_and_t(find(cosine_geometry(r0, {}), lam0))
    s0, r_0 = s0.mean(), abs(t0.mean()) / s0.mean()
    print(f"A2b circle: S = {s0:.6g}, R = |T|/S = {r_0:.4e}")
    for h in (0.5, 1.0):
        curv_s, curv_r = [], []
        for m in range(1, 31):
            # dλ/da_m = 0 here (A2), so the pole moves only at O(h²) and the
            # circle's box still holds it.
            sp, tp = s_and_t(find(cosine_geometry(r0, {m: h}), lam0))
            sm, tm = s_and_t(find(cosine_geometry(r0, {m: -h}), lam0))
            curv_s.append((sp.mean() + sm.mean() - 2 * s0) / h**2 / s0)
            rp, rm = abs(tp.mean()) / sp.mean(), abs(tm.mean()) / sm.mean()
            curv_r.append((np.log(rp) + np.log(rm) - 2 * np.log(r_0)) / h**2)
        cs, cr = np.array(curv_s), np.array(curv_r)
        print(f"A2b h={h}: (1/S)·∂²S/∂a_m² [nm⁻²], m = 1..30:")
        print("     " + " ".join(f"{v:.2e}" for v in cs))
        print(f"A2b h={h}: ∂² ln R/∂a_m² [nm⁻²], m = 1..30:")
        print("     " + " ".join(f"{v:.2e}" for v in cr))
        print(
            f"A2b h={h}: max (1/S)S'' = {cs.max():.2e} (m={cs.argmax() + 1}), "
            f"min (ln R)'' = {cr.min():.2e} (m={cr.argmin() + 1})"
        )


def continue_to(r0: float, a: dict[int, float], lam0: complex, steps: int) -> QNMResult:
    """Follow the pair from the circle to shape `a` in `steps` equal steps."""
    guess = lam0
    for j in range(1, steps + 1):
        res = find(
            cosine_geometry(r0, {m: am * j / steps for m, am in a.items()}), guess
        )
        guess = complex(np.mean(res.wavelengths))
    return res


def a3_generic(r0: float, lam0: complex) -> None:
    """Shape gradient of λ, S and T on a generic shape, adjoint vs brute force."""
    a0 = {2: 30.0, 3: 20.0, 5: 12.0, 7: 8.0}
    base = continue_to(r0, a0, lam0, steps=8)
    lam = np.sort_complex(base.wavelengths)
    s0, t0 = s_and_t(base)
    print(
        f"A3  shape {a0} nm: λ = {lam[0]:.6f}, {lam[1]:.6f}; "
        f"S = {s0[0]:.6g}, {s0[1]:.6g}; T = {t0[0]:.5g}, {t0[1]:.5g}"
    )
    for m in (2, 5, 12):
        adj = shape_sens(base, r0, a0, m)
        for h in (0.125, 0.5, 2.0):
            ap = {**a0, m: a0.get(m, 0.0) + h}
            am_ = {**a0, m: a0.get(m, 0.0) - h}
            # Brute force: contour-locate the pair at both ends, the box centred
            # on the predicted pole so a first-order shift cannot leave it.
            res_p = find(cosine_geometry(r0, ap), complex(np.mean(lam + h * adj)))
            res_m = find(cosine_geometry(r0, am_), complex(np.mean(lam - h * adj)))
            fd_lam = (
                np.sort_complex(res_p.wavelengths) - np.sort_complex(res_m.wavelengths)
            ) / (2 * h)
            sp, tp = s_and_t(res_p)
            sm, tm = s_and_t(res_m)
            ds_bf, dt_bf = (sp - sm) / (2 * h), (tp - tm) / (2 * h)
            # Cheap: predicted poles, no contour.
            pred_p = dataclasses.replace(
                base, wavelengths=lam + h * adj, geometry=cosine_geometry(r0, ap)
            )
            pred_m = dataclasses.replace(
                base, wavelengths=lam - h * adj, geometry=cosine_geometry(r0, am_)
            )
            sp_c, tp_c = s_and_t(pred_p)
            sm_c, tm_c = s_and_t(pred_m)
            ds_c, dt_c = (sp_c - sm_c) / (2 * h), (tp_c - tm_c) / (2 * h)

            def rel(x: np.ndarray, y: np.ndarray) -> float:
                return float(np.max(np.abs(x - y)) / np.max(np.abs(y)))

            print(
                f"A3  m={m:2d} h={h}: dλ/da {adj[0]:.4e}  "
                f"adj-vs-FD {rel(adj, fd_lam):.1e} | "
                f"dS/da {ds_bf[0]:.5g} cheap-vs-BF {rel(ds_c, ds_bf):.1e} | "
                f"dT/da {dt_bf[0]:.5g} cheap-vs-BF {rel(dt_c, dt_bf):.1e}"
            )


def main() -> None:
    """Run A1–A3 on the n = 46 WGM."""
    t_start = time.perf_counter()
    ref = analytic(N_MODE)
    r0, lam0 = ref["rad"], ref["lam"]
    g = Geometry.gielis(rad=r0, n_pts=N_PTS, m=0)
    c = cosine_geometry(r0, {})
    dev = max(
        float(np.max(np.abs(getattr(g, k) - getattr(c, k))))
        for k in ("f", "g", "df", "dg", "ddf", "ddg")
    )
    print(
        f"n={N_MODE}, r₀={r0:.3f} nm, n_pts={N_PTS}; "
        f"cosine builder vs gielis circle: {dev:.1e} nm"
    )
    a1_scale(r0, lam0)
    a2_selection(r0, lam0)
    a2b_curvature(r0, lam0)
    a3_generic(r0, lam0)
    print(f"{time.perf_counter() - t_start:.0f} s")


if __name__ == "__main__":
    main()
