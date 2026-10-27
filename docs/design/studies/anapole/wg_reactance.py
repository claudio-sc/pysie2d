"""Anapole note §6 method (C): anapoles of a non-separable shape from the scattering matrix.

Free space (the S-matrix of the particle alone), pol 2 = the note's TM, 1550 nm, R_eq 500.
Basis i^n J_n(kρ)e^{inθ} in, i^m H_m^(1)(kρ)e^{imθ} out (pysie2d.multipole, θ from +z
toward +x): driving with the n-th regular wave, decompose() returns column n of T, and
S = I + 2T (J = (H1 + H2)/2). Lossless ⇒ S unitary (anchor 1). Mirror x → −x maps n → −n, so
S splits into even/odd blocks. Eigenphases S = e^{2iδ}; K = i(1−S)(1+S)^{-1} = −tan δ is Herglotz
in ε, δ rotates monotonically, and anapoles are the zero crossings of an active δ. On the circle
those must be the exact channel anapoles (anchor 2, wg_dark.circle_anapoles).

At ε = 12.08 (Si) each drive is projected on the eigenchannels: its scattered power is
Σ_j |⟨v_j, a⟩|² sin²δ_j, so a drive is dark only if every lit channel sits at δ ≈ 0.

Run:  uv run python docs/design/studies/anapole/wg_reactance.py   (→ results/wg_reactance.json)
"""
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import numpy as np  # noqa: E402
from scipy.special import jv  # noqa: E402

from shapes import N_SI, geometry  # noqa: E402
from wg_dark import AREA, LAM, STACK, circle_anapoles  # noqa: E402
from pysie2d import BIESolver, Material  # noqa: E402

M = 9  # |n| ≤ 9: T_nn ~ (kR_c/2)^{2n}/(n!)² < 1e-8 beyond, below the eigenphase tolerance
NN = 256
N = np.arange(-M, M + 1)
K0 = 2 * np.pi / LAM
P = np.eye(2 * M + 1)[::-1]  # n → −n
TOL_ACTIVE = 1e-3  # a channel whose |δ| never exceeds this over the sweep is truncation, not physics


def regular_rhs(n, x0, z0):
    def rhs(nn, k, f, g):
        rho, th = np.hypot(f - x0, g - z0), np.arctan2(f - x0, g - z0)
        ei = np.zeros(2 * nn, dtype=complex)
        ei[:nn] = 1j**n * jv(n, k * rho) * np.exp(1j * n * th)
        return ei
    return rhs


def smatrix(geo, eps):
    s = BIESolver(geo, Material(n_core=float(np.sqrt(eps)), n_clad=1.0, pol=2))
    T = np.empty((2 * M + 1, 2 * M + 1), complex)
    for j, n in enumerate(N):
        T[:, j] = s.scatter(LAM, incident_rhs=regular_rhs(n, geo.x0, geo.z0)).multipoles(mmax=M).c
    return np.eye(2 * M + 1) + 2 * T


def blocks():
    """Orthonormal bases of the even and odd subspaces of P."""
    ev = [np.eye(2 * M + 1)[M]] + [(np.eye(2 * M + 1)[M + n] + np.eye(2 * M + 1)[M - n]) / np.sqrt(2) for n in range(1, M + 1)]
    od = [(np.eye(2 * M + 1)[M + n] - np.eye(2 * M + 1)[M - n]) / np.sqrt(2) for n in range(1, M + 1)]
    return {"even": np.array(ev).T, "odd": np.array(od).T}


B = blocks()


def phases(S, Q):
    w, v = np.linalg.eig(Q.T @ S @ Q)
    return np.angle(w) / 2, Q @ v  # δ ∈ (−π/2, π/2]


def drive_coeffs(field, x0, z0, r0=300.0):
    """a_n of a background field in the i^n J_n e^{inθ} basis, from samples on a circle inside its validity."""
    th = 2 * np.pi * np.arange(256) / 256
    u = field(x0 + r0 * np.sin(th), z0 + r0 * np.cos(th))
    c = np.fft.fft(u) / 256
    return np.array([c[n % 256] / (1j**n * jv(n, K0 * r0)) for n in N])


def sweep(name, geo, eps_grid):
    """Eigenphases followed by eigenvector overlap; anapoles = sign changes of an active channel's δ.

    Matching by nearest δ fails near δ = 0, where the truncation channels sit (δ ~ 1e-8):
    it hands an active channel's crossing to an inactive one.
    """
    with ProcessPoolExecutor(4) as ex:  # owner: 4 cores
        Ss = list(ex.map(smatrix, [geo] * len(eps_grid), eps_grid, chunksize=16))
    unit = max(float(np.abs(S.conj().T @ S - np.eye(len(S))).max()) for S in Ss)
    roots, series = {}, {}
    for b, Q in B.items():
        d_prev, V_prev = phases(Ss[0], Q)
        track, vecs = [d_prev], [V_prev]
        for S in Ss[1:]:
            d, V = phases(S, Q)
            O = np.abs(V_prev.conj().T @ V)  # overlap[i, j]: old channel i with new j
            perm = np.full(len(d), -1)
            for _ in range(len(d)):  # greedy assignment by largest overlap
                i, j = np.unravel_index(np.argmax(O), O.shape)
                perm[i] = j
                O[i, :] = -1
                O[:, j] = -1
            d, V = d[perm], V[:, perm]
            track.append(d)
            vecs.append(V)
            d_prev, V_prev = d, V
        D = np.array(track)
        active = np.abs(D).max(axis=0) > TOL_ACTIVE
        found = []
        for j in np.flatnonzero(active):
            for k in range(len(eps_grid) - 1):
                x, y = D[k, j], D[k + 1, j]
                if x * y < 0 and abs(x) < 0.6 and abs(y) < 0.6:  # through 0, not the ±π/2 wrap
                    r = float(eps_grid[k] - x * (eps_grid[k + 1] - eps_grid[k]) / (y - x))
                    # a high-Q resonance wraps ±π/2 inside one step and fakes a sign change:
                    # re-solve at the interpolated root and require δ there to be small
                    d_r, V_r = phases(smatrix(geo, r), Q)
                    d_j = d_r[np.argmax(np.abs(V_r.conj().T @ vecs[k][:, j]))]  # the same channel at the root
                    if abs(d_j) < 0.1 * max(abs(x), abs(y)) and all(abs(r - f) > 0.02 for f in found):
                        found.append(r)
        roots[b] = sorted(found)
        series[b] = D.tolist()
        print(f"  {name} {b}: {int(active.sum())} active channels; anapoles ε = {[round(r, 3) for r in roots[b]]}", flush=True)
    return series, roots, unit


def projection(name, geo, drives):
    S = smatrix(geo, N_SI**2)
    out = {}
    for dname, a in drives.items():
        # no normalisation: unit field at the centre for every drive; a truncated-norm would be set by |n| = M
        tot = 0.0
        chans = []
        for b, Q in B.items():
            d, V = phases(S, Q)
            for dj, vj in zip(d, V.T):
                w = abs(np.vdot(vj, a)) ** 2
                chans.append({"block": b, "delta": float(dj), "weight": float(w), "contrib": float(w * np.sin(dj) ** 2)})
                tot += w * np.sin(dj) ** 2
        chans.sort(key=lambda c: -c["contrib"])
        out[dname] = {"total": tot, "channels": chans[:6]}
        print(f"  {name} @ε 12.08, {dname:12s}: Σ w sin²δ {tot:.3e};  top: " +
              "; ".join(f"{c['block']} δ {c['delta']:+.3f} w {c['weight']:.2e}" for c in chans[:3]), flush=True)
    return out


def main():
    c5 = json.loads((HERE / "results/campaign_500.json").read_text())
    dk = json.loads((HERE / "results/wg_dark.json").read_text())
    fs = min((o for o in c5["phase_a"] if o["feasible"]), key=lambda o: o["f"])["a"]
    an = min((o for o in dk["phase1"] if o["feasible"] and o["width_gap_ok"]), key=lambda o: o["scatt"])["a"]
    n_eff = STACK.guided_modes(LAM, 2, 1.0)[0].beta / K0
    beta, gamma = n_eff * K0, K0 * np.sqrt(n_eff**2 - 1)
    exact = circle_anapoles()
    print("exact circle anapoles:", {n: [round(r, 4) for r in v] for n, v in exact.items() if v}, flush=True)
    eps_grid = np.arange(1.5, 25.0001, 0.02)  # linear interpolation of δ: error ~ (0.02)² curvature
    out = {"exact_circle": exact, "eps": eps_grid.tolist(), "shapes": {}}
    for name, a in (("circle", np.zeros(10)), ("anapole", np.array(an)), ("fs-dark", np.array(fs))):
        geo = geometry(a, AREA, NN)
        series, roots, unit = sweep(name, geo, eps_grid)
        drives = {k: drive_coeffs(fn, geo.x0, geo.z0) for k, fn in {
            "plane z": lambda x, z: np.exp(-1j * K0 * z),  # pysie2d angle 0: exp(ik(x sin0 − z cos0))
            "plane x": lambda x, z: np.exp(1j * K0 * x),
            "guided tail": lambda x, z: np.exp(1j * beta * x - gamma * z)}.items()}
        out["shapes"][name] = {"roots": roots, "unitarity_max": unit, "delta": series,
                               "projection": projection(name, geo, drives)}
        print(f"  {name}: max |S†S − I| over sweep {out['shapes'][name]['unitarity_max']:.1e}", flush=True)
        (HERE / "results/wg_reactance.json").write_text(json.dumps(out))


if __name__ == "__main__":
    main()
