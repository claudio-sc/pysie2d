"""G4: holomorphy and the QNM path under Kress, on uniform θ and adaptive maps.

Everything runs through **production** code (`Parametrisation`,
`Geometry.gielis`, `QNMSolver`, `QNMResult.refine`/`sensitivity`). Uniform arc
length is not measured: it is disqualified as a default (conventions §13) and
kept as legacy only. The one exception to production is the superellipse's
adaptive map, which production cannot build (flat points, G3 defect) and which
takes G3's smooth relative curvature floor `hypot(κ, 1e-3·max|κ|)`.

Material throughout is the QNM test fixture: rad 200, n_core 3.0, TE.

Four jobs, one per process, each printing JSON lines:

- `holo` — the premise. (a) Invariant 13.2 structurally: the adaptive map's
  nodes are bit-identical across `wavelength_ref`. (b) Cauchy–Riemann on the
  assembled matrix: the central difference of `M` along Re λ and along Im λ
  must agree with each other and with `assemble_derivative`, decaying as h²
  — on three non-circular shapes and both maps. (c) A **negative control**:
  a circle on the graded map `θ = t + ε sin 2t` whose `ε` is made to depend on
  Re λ. At every fixed λ that map gives the Mie pole (parametrisation
  invariance), so any error Beyn returns is holomorphy loss and nothing else;
  the point is to see which diagnostics notice.
- `circle` — Mie. A three-mode box (TE n=0, simple; TE n=2, degenerate) on
  uniform θ and on a production adaptive map borrowed from the `m = 4` star,
  so the circle is sampled on a genuinely graded, C₄-symmetric node set. Rate,
  mode count, multiplicity, and the discrete splitting of the n=2 pair.
- `shape NAME MODE` — continuation from the circle along a smooth path to
  `ellipse2` (equal-area, aspect 2), `star6` (m=4, 6/12/12) or `super32`
  (superellipse n=32, G3's grading-pays class); then, at the endpoint, the
  QNM error ladder per map against uniform θ at `NN_REF`, contour-independence
  (shifted box, denser contour), and `dλ/dp` on the frozen map: its ladder,
  cross-map agreement, adjoint vs re-extracted poles, and what
  `richardson_limit` does to it.

Usage: `uv run python docs/design/studies/g4_holomorphy_qnm.py holo|circle|shape NAME MODE`
"""

import json
import sys
import time
import warnings

import numpy as np

import pysie2d.parametrisation as pmod
from pysie2d import BIESolver, Geometry, Material, Parametrisation, QNMSolver
from pysie2d import richardson_limit
from pysie2d.beyn import beyn_modes
from pysie2d.geometry import _boundary_arrays, gielis
from pysie2d.parametrisation import Nodes
from pysie2d.qnm import _edge_margin

RAD = 200.0
MAT = Material(n_core=3.0, n_clad=1.0, pol=2)
N_SIDE = 12  # 6 leaves a 5e-6 nm contour floor; refine() removes what is left
BAND = (20.0, 100.0)
FLOOR = 1.0e-3  # G3's value; only the super32 adaptive map reads it
THETA = Parametrisation.uniform_theta()

# Newton on reference.mie.qnm_denominator, full precision (tests/test_qnm.py).
MIE = {
    "n0": (530.8321407288238 + 26.37849897377869j, 1),
    "n2": (550.4688885949248 + 20.235785293593015j, 2),
    "n3": (760.6866483138609 + 7.9477105871857985j, 2),
}
NNS = (40, 60, 80, 120, 160, 240, 320)  # all ≡ 0 mod 4: keeps C₄ exact on nodes
NN_REF, NN_REF_CHECK = 480, 400
STEPS = 40
_SC = pmod._speed_curvature


def _gielis_shape(name: str, s: float) -> dict:
    if name == "ellipse2":
        area_aspect = 1.0 + s
        return dict(
            a=area_aspect**-0.5, b=area_aspect**0.5, m=4, n1=2.0, n2=2.0, n3=2.0
        )
    if name == "star6":
        return dict(
            a=1.0, b=1.0, m=4, n1=2.0 + 4.0 * s, n2=2.0 + 10.0 * s, n3=2.0 + 10.0 * s
        )
    if name == "super32":
        n = 2.0 * 16.0**s
        return dict(a=1.0, b=1.0, m=4, n1=n, n2=n, n3=n)
    raise ValueError(name)


def shape(name: str, s: float) -> dict:
    """Gielis parameters at path position `s ∈ [0, 1]`; `s = 0` is the circle.

    `rad` is rescaled so the area is the circle's at every `s`. Scale
    covariance (§9) makes a pure size change exactly `λ → s·λ`; raising the
    exponents swells the shape (~8 % area on the first 1/20 of the star path,
    ~20 nm of Re λ), which would put the step outside any tight box and is not
    the shape effect being tracked.
    """
    par = _gielis_shape(name, s)
    theta = np.linspace(0.0, 2.0 * np.pi, 8192, endpoint=False)
    _, _, r, *_ = gielis(
        theta,
        1.0,
        par["a"],
        par["b"],
        par["m"],
        par["n1"],
        par["n2"],
        par["n3"],
        0.0,
        0.0,
    )
    area = 0.5 * np.mean(r**2) * 2.0 * np.pi
    return dict(rad=RAD * np.sqrt(np.pi / area), **par)


# The differentiated parameter per shape: b breaks the ellipse's aspect, n1
# sharpens the star/superellipse while keeping C₄v, so a degenerate pair must
# stay degenerate under it — a free check on the secular branch.
PARAM = {"ellipse2": "b", "star6": "n1", "super32": "n1"}


def geom(sh: dict, nn: int, par: Parametrisation) -> Geometry:
    return Geometry.gielis(n_pts=nn, parametrisation=par, **sh)


def adaptive(name: str) -> Parametrisation:
    """Production adaptive map at the endpoint; floored only on super32."""
    sh = shape(name, 1.0)
    if name == "super32":
        pmod._speed_curvature = lambda *rr: (
            lambda gk: (gk[0], np.hypot(gk[1], FLOOR * np.abs(gk[1]).max()))
        )(_SC(*rr))
    try:
        return Parametrisation.gielis(**sh, r_band=BAND, n_core=MAT.n_core)
    finally:
        pmod._speed_curvature = _SC


def emit(**kw) -> None:
    def conv(v):
        if isinstance(v, complex | np.complexfloating):
            return [float(v.real), float(v.imag)]
        if isinstance(v, np.ndarray):
            return [conv(x) for x in v.tolist()]
        if isinstance(v, dict):
            return {k: conv(x) for k, x in v.items()}
        if isinstance(v, list | tuple):
            return [conv(x) for x in v]
        if isinstance(v, np.floating | np.integer | np.bool_):
            return v.item()
        return v

    print(json.dumps({k: conv(v) for k, v in kw.items()}), flush=True)


def box(centre: complex, half: float) -> tuple[complex, complex]:
    """Square box about `centre`, kept strictly inside Im λ > 0."""
    lo = complex(centre.real - half, max(centre.imag - half, 0.3 * centre.imag))
    return lo, complex(centre.real + half, centre.imag + half)


def pick(lams: np.ndarray, centre: complex, k: int):
    """The k modes nearest `centre`, their mean, and the next one's distance."""
    order = np.argsort(np.abs(lams - centre))
    chosen = lams[order[:k]]
    other = float(np.abs(lams[order[k]] - centre)) if lams.size > k else np.inf
    return chosen, complex(chosen.mean()), other


# ---------------------------------------------------------------------------
# holo
# ---------------------------------------------------------------------------


def graded_circle(nn: int, eps: float) -> Geometry:
    """Circle sampled on `θ = t + ε sin 2t` — a valid map for |ε| < ½."""
    t = 2.0 * np.pi * (np.arange(nn) + 0.5) / nn
    nodes = Nodes(
        t=t,
        theta=t + eps * np.sin(2 * t),
        dw=1 + 2 * eps * np.cos(2 * t),
        ddw=-4 * eps * np.sin(2 * t),
    )
    arrays = _boundary_arrays(nodes, RAD, 1.0, 1.0, 0, 2.0, 2.0, 2.0, 0.0, 0.0)
    return Geometry(*arrays, rad=RAD)


def cauchy_riemann(assemble, dassemble, lam0: complex, hs=(1e-1, 1e-2, 1e-3)):
    """‖D_re − D_im‖/‖D‖ and ‖D_re − dM/dλ‖/‖D‖ over a ladder of steps h."""
    d_an = dassemble(lam0) if dassemble else None
    out = []
    for h in hs:
        d_re = (assemble(lam0 + h) - assemble(lam0 - h)) / (2 * h)
        d_im = (assemble(lam0 + 1j * h) - assemble(lam0 - 1j * h)) / (2j * h)
        scale = np.linalg.norm(d_re)
        row = {"h": h, "cr": np.linalg.norm(d_re - d_im) / scale}
        if d_an is not None:
            row["vs_analytic"] = np.linalg.norm(d_re - d_an) / scale
        out.append(row)
    return out


def holo() -> None:
    # (a) 13.2 structurally: wavelength_ref may set nn_from_band, never w.
    for name in ("ellipse2", "star6"):
        sh = shape(name, 1.0)
        maps = [
            Parametrisation.gielis(
                **sh, r_band=BAND, n_core=MAT.n_core, wavelength_ref=w
            )
            for w in (450.0, 600.0, 1550.0)
        ]
        n0 = maps[0].nodes(120)
        same = all(
            np.array_equal(getattr(p.nodes(120), f), getattr(n0, f))
            for p in maps[1:]
            for f in ("theta", "dw", "ddw")
        )
        emit(
            job="lambda_ref_invariance",
            shape=name,
            nodes_bit_identical=same,
            nn_from_band=[p.nn_from_band for p in maps],
        )

    # (b) Cauchy–Riemann on the assembled matrix, both maps, three shapes.
    lam0 = MIE["n0"][0]
    for name in ("ellipse2", "star6", "super32"):
        for label, par in (("theta", THETA), ("adapt", adaptive(name))):
            bie = BIESolver(geom(shape(name, 1.0), 120, par), MAT)
            emit(
                job="cauchy_riemann",
                shape=name,
                map=label,
                ladder=cauchy_riemann(bie.assemble, bie.assemble_derivative, lam0),
            )

    # (c) Negative control: a λ-dependent map on the circle.
    nn, lo, hi = 80, 520.0 + 15.0j, 545.0 + 40.0j
    for slope in (0.0, 1e-4, 1e-3, 1e-2, 1e-1):

        def eps(lam, slope=slope):
            return 0.3 + slope * (lam.real - 530.0) / 25.0

        def assemble(lam, eps=eps):
            return BIESolver(graded_circle(nn, eps(lam)), MAT).assemble(lam)

        try:
            res = beyn_modes(assemble, lo, hi, n_quad_per_side=N_SIDE)
        except ValueError as exc:
            emit(job="lambda_dependent_map", slope_per_25nm=slope, raised=str(exc)[:90])
            continue
        lams = res.eigenvalues
        sig = [
            float(
                (lambda s: s[-1] / s[0])(np.linalg.svd(assemble(x), compute_uv=False))
            )
            for x in lams
        ]
        emit(
            job="lambda_dependent_map",
            slope_per_25nm=slope,
            n_modes=res.n_modes,
            rank=res.rank,
            max_gap=res.max_gap,
            cancellation=res.cancellation,
            edge_margin=_edge_margin(lams, lo, hi),
            sigma_ratio=sig,
            err_vs_mie=np.abs(lams - MIE["n0"][0]),
            cr=cauchy_riemann(assemble, None, lam0, hs=(1e-2,))[0]["cr"],
        )


# ---------------------------------------------------------------------------
# circle
# ---------------------------------------------------------------------------


def circle() -> None:
    graded = adaptive("star6")
    lo, hi = 515.0 + 5.0j, 565.0 + 35.0j  # TE n=0 (simple) + TE n=2 (pair)
    for label, par in (("theta", THETA), ("star6_map", graded)):
        for nn in (20, 30, 40, 60, 80, 120, 160, 240, 320, 480):
            g = Geometry.gielis(RAD, nn, m=0, parametrisation=par)
            try:
                res = QNMSolver(g, MAT).modes(lo, hi, n_quad_per_side=N_SIDE)
            except ValueError as exc:
                emit(job="circle", map=label, nn=nn, error=str(exc)[:80])
                continue
            fine = res.refine(tol=1e-11)
            row = dict(
                job="circle",
                map=label,
                nn=nn,
                n_modes=res.n_modes,
                multiplicity=res.multiplicity,
                rank=res.rank,
                max_gap=res.max_gap,
                cancellation=res.cancellation,
                min_edge=float(res.edge_margin.min()) if res.n_modes else None,
            )
            if res.n_modes >= 1:
                _, l0, _ = pick(fine.wavelengths, MIE["n0"][0], 1)
                row["err_n0"] = abs(l0 - MIE["n0"][0])
                row["beyn_vs_newton_n0"] = abs(
                    pick(res.wavelengths, MIE["n0"][0], 1)[1] - l0
                )
            if res.n_modes >= 3:
                pair, _, _ = pick(fine.wavelengths, MIE["n2"][0], 2)
                row["err_n2"] = float(np.abs(pair - MIE["n2"][0]).max())
                row["split_n2"] = abs(pair[0] - pair[1])
            emit(**row)


# ---------------------------------------------------------------------------
# shape
# ---------------------------------------------------------------------------


def extract(g: Geometry, centre: complex, k: int, half: float = 5.0, n_side=N_SIDE):
    """Refined modes in a box; the k nearest `centre` and the diagnostics."""
    for widen in (1.0, 2.0):
        try:
            res = QNMSolver(g, MAT).modes(
                *box(centre, half * widen), n_quad_per_side=n_side
            )
        except ValueError:
            return None
        if res.n_modes >= k:
            break
    else:
        return None
    fine = res.refine(tol=1e-11)
    chosen, lam, other = pick(fine.wavelengths, centre, k)
    return dict(
        res=fine,
        lam=lam,
        chosen=chosen,
        other=other,
        n_modes=res.n_modes,
        rank=res.rank,
        max_gap=res.max_gap,
        widen=widen,
        min_edge=float(res.edge_margin.min()),
        beyn_vs_newton=abs(pick(res.wavelengths, centre, k)[1] - lam),
    )


def continuation(name: str, mode: str, nn: int = 120):
    """Secant continuation of a labelled Mie mode from the circle, uniform θ."""
    target, k = MIE[mode]
    pts, tells = [], []
    for s in np.linspace(0.0, 1.0, STEPS + 1):
        pred = 2 * pts[-1] - pts[-2] if len(pts) >= 2 else (pts[-1] if pts else target)
        hit = extract(geom(shape(name, s), nn, THETA), pred, k, half=3.0)
        if hit is None:
            tells.append(dict(s=float(s), lost=True))
            break
        pts.append(hit["lam"])
        tells.append(
            dict(
                s=float(s),
                lam=hit["lam"],
                n_modes=hit["n_modes"],
                widen=hit["widen"],
                miss_pred=abs(hit["lam"] - pred),
                other=hit["other"],
                split=float(np.ptp(np.abs(hit["chosen"]))),
                min_edge=hit["min_edge"],
            )
        )
    lam = np.array(pts)
    d2 = np.abs(np.diff(lam, 2)) if lam.size >= 3 else np.full(1, np.nan)
    return (
        lam,
        tells,
        dict(
            max_d2=float(d2.max()),
            median_d2=float(np.median(d2)),
            circle_err=abs(lam[0] - target) if lam.size else None,
        ),
    )


def shape_job(name: str, mode: str) -> None:
    t0 = time.time()
    target, k = MIE[mode]
    traj, tells, smooth = continuation(name, mode)
    emit(job="continuation", shape=name, mode=mode, tells=tells, **smooth)
    if len(traj) < STEPS + 1:
        return

    sh = shape(name, 1.0)
    pname = PARAM[name]
    maps = {"theta": THETA, "adapt": adaptive(name)}
    emit(
        job="adaptive_map",
        shape=name,
        alpha_eff=maps["adapt"].alpha_eff,
        nn_from_band=maps["adapt"].nn_from_band,
        n_terms=maps["adapt"].n_terms,
    )

    def jac(hit, par, nn):
        def at(d):
            return geom({**sh, pname: sh[pname] + d}, nn, par), MAT

        jj = hit["res"].sensitivity(at)
        idx = [
            int(np.argmin(np.abs(hit["res"].wavelengths - c))) for c in hit["chosen"]
        ]
        vals = jj[idx]
        return complex(vals.mean()), float(np.ptp(np.abs(vals))) if k > 1 else 0.0

    ref = extract(geom(sh, NN_REF, THETA), traj[-1], k)
    j_ref, j_split = jac(ref, THETA, NN_REF)
    check_theta = extract(geom(sh, NN_REF_CHECK, THETA), ref["lam"], k)
    check_adapt = extract(geom(sh, NN_REF, maps["adapt"]), ref["lam"], k)
    j_adapt_top, _ = jac(check_adapt, maps["adapt"], NN_REF)
    emit(
        job="reference",
        shape=name,
        mode=mode,
        lam=ref["lam"],
        split=np.ptp(np.abs(ref["chosen"])),
        continuation_end_vs_ref=abs(traj[-1] - ref["lam"]),
        floor_theta=abs(check_theta["lam"] - ref["lam"]),
        floor_adapt_same_nn=abs(check_adapt["lam"] - ref["lam"]),
        j_ref=j_ref,
        j_split=j_split,
        j_adapt_vs_theta_at_ref=abs(j_adapt_top - j_ref) / abs(j_ref),
    )

    jlad = {}
    for label, par in maps.items():
        for nn in NNS:
            hit = extract(geom(sh, nn, par), ref["lam"], k)
            if hit is None:
                emit(job="ladder", shape=name, mode=mode, map=label, nn=nn, lost=True)
                continue
            j, js = jac(hit, par, nn)
            jlad[(label, nn)] = j
            emit(
                job="ladder",
                shape=name,
                mode=mode,
                map=label,
                nn=nn,
                err=abs(hit["lam"] - ref["lam"]),
                split=float(np.ptp(np.abs(hit["chosen"]))),
                n_modes=hit["n_modes"],
                rank=hit["rank"],
                max_gap=hit["max_gap"],
                min_edge=hit["min_edge"],
                widen=hit["widen"],
                beyn_vs_newton=hit["beyn_vs_newton"],
                j_err=abs(j - j_ref) / abs(j_ref),
                j_split=js,
            )

        # Contour independence at nn = 160: a shifted, larger box and a denser
        # contour must return the same poles — Cauchy's theorem at work.
        g160 = geom(sh, 160, par)
        base = extract(g160, ref["lam"], k)
        shifted = extract(g160, ref["lam"] + 1.5 + 1.0j, k, half=7.0)
        dense = extract(g160, ref["lam"], k, n_side=16)
        emit(
            job="contour_independence",
            shape=name,
            mode=mode,
            map=label,
            shifted=abs(shifted["lam"] - base["lam"]),
            dense=abs(dense["lam"] - base["lam"]),
            counts=[base["n_modes"], shifted["n_modes"], dense["n_modes"]],
        )

        # Adjoint against re-extracted poles on the frozen map (conventions §11 Gate 3).
        j160 = jlad.get((label, 160))
        fd = []
        for step in (1e-2, 1e-3):
            lam_pm = [
                extract(
                    geom({**sh, pname: sh[pname] + sgn * step}, 160, par),
                    base["lam"],
                    k,
                )["lam"]
                for sgn in (1, -1)
            ]
            fd.append(abs((lam_pm[0] - lam_pm[1]) / (2 * step) - j160) / abs(j160))
        emit(
            job="adjoint_vs_reextracted",
            shape=name,
            mode=mode,
            map=label,
            steps=[1e-2, 1e-3],
            rel_err=fd,
            ratio=fd[0] / fd[1],
        )

        if (label, 160) in jlad and (label, 320) in jlad:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                rich = richardson_limit(
                    jlad[(label, 160)], jlad[(label, 320)], 160, 320
                )
            emit(
                job="richardson",
                shape=name,
                mode=mode,
                map=label,
                err_320=abs(jlad[(label, 320)] - j_ref) / abs(j_ref),
                err_richardson=abs(rich - j_ref) / abs(j_ref),
            )
    emit(job="done", shape=name, mode=mode, seconds=time.time() - t0)


if __name__ == "__main__":
    if sys.argv[1] == "holo":
        holo()
    elif sys.argv[1] == "circle":
        circle()
    else:
        shape_job(sys.argv[2], sys.argv[3])
