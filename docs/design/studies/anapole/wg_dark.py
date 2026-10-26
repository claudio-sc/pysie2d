"""Anapoles under guided drive, the loss law, and the dark-seeded Pareto front. Single core.

Phase 1 — guided-drive anapoles: minimise log10 P_scatt (P_up + P_down, E5) of a lossless
  Si particle (R_eq 500 nm, gap 100 nm, CD rules) under TE0 of the SiN slab. Free-space
  darkness does not carry over (wg_evanescent.py: the drive is a plane wave at complex angle,
  order m weighted ≈ 3.1^|m|, plus slab feedback), so darkness is searched for directly.
Phase 2 — the loss law: on the best distinct dark designs sweep Im ε. The anapole-note
  reality argument makes the absorbed power the imaginary part of the interior condition,
  P_abs = k0²·Im ε·∫|u|² (wg_abs_check.py: ratio 1 to 1e-14), so near a lossless anapole
  P_abs ∝ Im ε and the extra scattering ∝ (Im ε)²: slopes 1 and 2 in log–log.
Phase 3 — Pareto front (E6) seeded from the dark designs, Im ε a free variable.

Wall-clock deadlines per phase; results written after every optimisation.
Run:  uv run python docs/design/studies/anapole/wg_dark.py   (→ results/wg_dark.json)
"""

import json
import os
import sys
import time
from pathlib import Path

for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[v] = "1"  # the owner's budget is one core
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import numpy as np  # noqa: E402
from scipy.optimize import minimize  # noqa: E402

from campaign_500 import AREA, CONS, M, feasible  # noqa: E402
from campaign_cd import width_gap_ok  # noqa: E402
from shapes import N_SI, geometry  # noqa: E402
from pysie2d import BIESolver, Geometry, Material, Stack  # noqa: E402

LAM, GAP, NN = 1550.0, 100.0, 192  # nn 384 moved P_abs by ≤ 7e-4 on the wg_opt optima
STACK = Stack(layers=((2.0**2, 400.0),), eps_bot=1.444**2, z_top=0.0)
SCALE = float(os.environ.get("WG_DARK_SCALE", "1"))  # < 1 only for smoke tests
T1, T2, T3 = (np.array([100, 125, 230]) * 60 * SCALE)
N_START1 = 40
CAPS = (0.05, 0.02, 0.01, 0.005, 0.002)
EPSIS = np.logspace(-3, 0, 13)
EPS_RE = np.linspace(1.5, 25.0, 471)  # Re ε sweep, step 0.05; Si is 12.08
FD_H = 1e-5  # see wg_opt.py
OUT = HERE / ("results/wg_dark.json" if SCALE == 1 else "results/wg_dark_smoke.json")
T0 = time.time()
state = {"phase1": [], "phase2": [], "eps_sweep": [], "circle_anapoles": {}, "phase3": []}


def elapsed():
    return time.time() - T0


def save():
    OUT.write_text(json.dumps(state, indent=1))


def solve(a, epsi, nn=NN, eps_re=N_SI**2):
    g0 = geometry(np.asarray(a, float), AREA, nn)
    dz = GAP - g0.g.min()
    g = Geometry(g0.f, g0.g + dz, g0.df, g0.dg, g0.ddf, g0.ddg, rad=g0.rad, z0=dz)
    s = BIESolver(g, Material(n_core=float(np.sqrt(eps_re)), n_clad=1.0, pol=2, epsi=float(epsi)), background=STACK)
    ma = s.scatter_mode(LAM, 0, +1, x_ref=0.0).mode_amplitudes(x_ref=0.0)
    return {"abs": float(ma.p_abs), "scatt": float(ma.p_up + ma.p_down), "T": float(np.sum(ma.p_plus)),
            "R": float(np.sum(ma.p_minus)), "balance": float(ma.balance())}


class Cached:
    def __init__(self, fun):
        self.fun, self.memo = fun, {}

    def __call__(self, x):
        key = np.asarray(x, float).tobytes()
        if key not in self.memo:
            self.memo[key] = self.fun(np.asarray(x, float))
        return self.memo[key]

    def grad(self, x, q):
        x = np.asarray(x, float)
        out = np.empty(len(x))
        for k in range(len(x)):
            e = np.zeros(len(x))
            e[k] = FD_H
            out[k] = (q(self(x + e)) - q(self(x - e))) / (2 * FD_H)
        return out


def shrink_feasible(a0):
    a0 = np.array(a0, float)
    while not feasible(a0):
        a0 *= 0.85
    return a0


def phase1(starts):
    c = Cached(lambda a: solve(a, 0.0))
    obj = lambda r: np.log10(r["scatt"])  # noqa: E731
    for name, a0 in starts:
        if elapsed() > T1:
            break
        a0 = shrink_feasible(a0)
        t = time.time()
        res = minimize(lambda a: obj(c(a)), a0, jac=lambda a: c.grad(a, obj), method="SLSQP",
                       bounds=[(-0.6, 0.6)] * M, constraints=CONS, options={"maxiter": 80, "ftol": 1e-10})
        a = res.x
        r = c(a)
        state["phase1"].append({"start": name, "a": a.tolist(), **r, "scatt_384": solve(a, 0.0, 384)["scatt"],
                                "feasible": feasible(a), "width_gap_ok": width_gap_ok(a, AREA),
                                "nit": res.nit, "seconds": time.time() - t})
        save()
        print(f"[{elapsed()/60:5.1f} min] P1 {name:12s}: P_scatt {r['scatt']:.3e} (nn384 {state['phase1'][-1]['scatt_384']:.3e})"
              f" T {r['T']:.3f} R {r['R']:.3f} feas {feasible(a)} nit {res.nit}", flush=True)
        c.memo.clear()


def distinct_dark(k=4):
    ok = sorted([o for o in state["phase1"] if o["feasible"] and o["width_gap_ok"]], key=lambda o: o["scatt"])
    picks = []
    for o in ok:
        if all(np.linalg.norm(np.subtract(o["a"], p["a"])) > 0.02 for p in picks):
            picks.append(o)
        if len(picks) == k:
            break
    if len(picks) < k:  # SLSQP often stops at maxiter just outside the CD set: pull those back in
        for o in sorted(state["phase1"], key=lambda o: o["scatt"]):
            if o["feasible"] and o["width_gap_ok"]:
                continue
            a = shrink_feasible(o["a"])
            if width_gap_ok(a, AREA) and all(np.linalg.norm(a - np.asarray(p["a"])) > 0.02 for p in picks):
                picks.append({"start": o["start"] + "*", "a": a.tolist(), **solve(a, 0.0)})
            if len(picks) == k:
                break
    return picks


def phase2(designs):
    for o in designs:
        if elapsed() > T2:
            break
        rows = [solve(o["a"], e) for e in EPSIS]
        ab = np.array([r["abs"] for r in rows])
        sc = np.array([r["scatt"] for r in rows]) - o["scatt"]
        lo = slice(0, 5)  # Im ε ≤ 1e-2: the perturbative end
        s_abs = np.polyfit(np.log10(EPSIS[lo]), np.log10(ab[lo]), 1)[0]
        s_sc = np.polyfit(np.log10(EPSIS[lo]), np.log10(np.abs(sc[lo])), 1)[0]
        state["phase2"].append({"start": o["start"], "scatt0": o["scatt"], "epsi": EPSIS.tolist(), "rows": rows,
                                "slope_abs": s_abs, "slope_dscatt": s_sc})
        save()
        print(f"[{elapsed()/60:5.1f} min] P2 {o['start']:12s}: P_scatt(0) {o['scatt']:.2e}  slope P_abs {s_abs:.3f}"
              f"  slope ΔP_scatt {s_sc:.3f}  max P_abs {ab.max():.3f} at Im ε {EPSIS[ab.argmax()]:.3g}", flush=True)


def circle_anapoles(nmax=6):
    """Exact TM channel anapoles of the disk (anapole note Eq. twopoints), pysie2d pol 2.

    Λ_n(w) = √w J_n'(√w)/J_n(√w); roots of Λ_n(εX²) = Λ_n(X²) in EPS_RE, poles (J_n(√ε X) = 0)
    excluded by requiring the bracket to be continuous.
    """
    from scipy.optimize import brentq
    from scipy.special import jv, jvp
    X = 2 * np.pi / LAM * np.sqrt(AREA / np.pi)
    lam = lambda n, w: np.sqrt(w) * jvp(n, np.sqrt(w)) / jv(n, np.sqrt(w))  # noqa: E731
    out = {}
    for n in range(nmax + 1):
        F = lambda e: lam(n, e * X**2) - lam(n, X**2)  # noqa: E731
        es = np.linspace(EPS_RE[0], EPS_RE[-1], 20000)
        v = F(es)
        roots = []
        for k in range(len(es) - 1):
            if np.sign(v[k]) != np.sign(v[k + 1]) and abs(v[k] - v[k + 1]) < 5:  # skip poles
                roots.append(brentq(F, es[k], es[k + 1], xtol=1e-12))
        out[str(n)] = roots
    return out


def eps_sweep(designs):
    """Lossless Re ε sweep at fixed shape: P_scatt dips are the guided-drive anapole permittivities."""
    for name, a in designs:
        if elapsed() > T2:
            break
        rows = [solve(a, 0.0, eps_re=e) for e in EPS_RE]
        sc = np.array([r["scatt"] for r in rows])
        dips = [int(k) for k in range(1, len(sc) - 1) if sc[k] < sc[k - 1] and sc[k] < sc[k + 1]]
        state["eps_sweep"].append({"name": name, "eps": EPS_RE.tolist(), "rows": rows,
                                   "dips": [{"eps": float(EPS_RE[k]), "scatt": float(sc[k])} for k in dips]})
        save()
        print(f"[{elapsed()/60:5.1f} min] ε-sweep {name:12s}: dips " +
              ", ".join(f"ε {EPS_RE[k]:.2f} ({sc[k]:.1e})" for k in dips), flush=True)


def phase3(seeds):
    # x = (a, log10 Im ε); Im ε ∈ [1e-3, 3]
    cons_a = [{"type": "ineq", "fun": (lambda x, c=c: c["fun"](x[:M]))} for c in CONS]
    for cap in CAPS:
        for name, a0, e0 in seeds:
            if elapsed() > T3:
                return
            c = Cached(lambda x: solve(x[:M], 10 ** x[M]))
            cons = cons_a + [{"type": "ineq", "fun": lambda x: (cap - c(x)["scatt"]) / cap,
                              "jac": lambda x: -c.grad(x, lambda r: r["scatt"]) / cap}]
            x0 = np.r_[shrink_feasible(a0), np.log10(e0)]
            t = time.time()
            res = minimize(lambda x: -c(x)["abs"], x0, jac=lambda x: -c.grad(x, lambda r: r["abs"]), method="SLSQP",
                           bounds=[(-0.6, 0.6)] * M + [(-3, np.log10(3))], constraints=cons,
                           options={"maxiter": 60, "ftol": 1e-8})
            x = res.x
            r = c(x)
            r384 = solve(x[:M], 10 ** x[M], 384)
            state["phase3"].append({"cap": cap, "start": name, "a": x[:M].tolist(), "epsi": float(10 ** x[M]), **r,
                                    "abs_384": r384["abs"], "feasible": feasible(x[:M]),
                                    "width_gap_ok": width_gap_ok(x[:M], AREA), "cap_ok": r["scatt"] <= cap * 1.01,
                                    "nit": res.nit, "seconds": time.time() - t})
            save()
            print(f"[{elapsed()/60:5.1f} min] P3 cap {cap:5.3f} {name:16s}: P_abs {r['abs']:.4f} (nn384 {r384['abs']:.4f})"
                  f" P_scatt {r['scatt']:.4f} Im ε {10**x[M]:.3g} T {r['T']:.3f} R {r['R']:.3f}"
                  f" feas {feasible(x[:M])} cap {r['scatt'] <= cap*1.01}", flush=True)


def main():
    c5 = json.loads((HERE / "results/campaign_500.json").read_text())
    fs_dark = min((o for o in c5["phase_a"] if o["feasible"]), key=lambda o: o["f"])["a"]
    rng = np.random.default_rng(2026)
    starts = [("circle", np.zeros(M)), ("fs-dark", np.array(fs_dark))]
    starts += [(f"rnd{k}", rng.normal(0, rng.uniform(0.08, 0.25), M) / np.arange(1, M + 1) ** 1.5) for k in range(N_START1 - 2)]
    print(f"phase 1 to {T1/60:.0f} min", flush=True)
    phase1(starts)
    dark = distinct_dark()
    print(f"phase 2 to {T2/60:.0f} min on {[o['start'] for o in dark]}", flush=True)
    phase2(dark)
    state["circle_anapoles"] = circle_anapoles()
    print("circle TM channel anapoles (ε):", {n: [round(r, 3) for r in v] for n, v in state["circle_anapoles"].items()}, flush=True)
    eps_sweep([("circle", np.zeros(M)), ("fs-dark", np.array(fs_dark))] + [(o["start"], np.array(o["a"])) for o in dark])
    prev = json.loads((HERE / "results/wg_opt.json").read_text())["runs"]
    best_prev = max((r for r in prev if r["cap"] == 0.01 and r["feasible"]), key=lambda r: r["abs"])
    seeds = [(f"dark:{o['start']}", np.array(o["a"]), 0.03) for o in dark[:3]]
    seeds.append(("wg_opt cap0.01", np.array(best_prev["a"]), 0.3))
    print(f"phase 3 to {T3/60:.0f} min", flush=True)
    phase3(seeds)
    print(f"done at {elapsed()/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
