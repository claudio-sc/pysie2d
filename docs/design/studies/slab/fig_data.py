"""Data for the slab near-field showcase (Plotly page). Writes one JSON file.

Usage: python fig_data.py OUT.json. Every number comes from the study prototype
(sproto, nearfield), validated in S1–S9 and nearfield.py's checks.
"""
import json
import sys

import numpy as np
from scipy.integrate import quad
from scipy.optimize import brentq

import nearfield as NF
import sproto as S

LAM, NC, NSI = 1550.0, 1.444, 3.476
k = 2 * np.pi * NC / LAM
ESI = (NSI / NC) ** 2
out = {"lam": LAM}


def r3(a, sig=4):
    a = np.asarray(a, dtype=float)
    return [float(f"{v:.{sig}g}") for v in a.ravel()]


def guided_field(st, kk, pol, h0, x, z):
    poles = S.guided_poles_gamma(st, kk, pol)
    G = np.zeros((z.size, x.size), dtype=complex)
    for b in poles:
        u, N, _ = S.mode_profile(st, kk, pol, b)
        G += (1j / (2 * b * N) * u(np.array([h0]))[0]) * np.outer(u(z), np.exp(1j * b * np.abs(x)))
    return G, poles


# --- F1/F2: SOI near field, TE and TM -------------------------------------------
h0 = 50.0
st = S.Stack([ESI], [220.0], 1.0)
xh = np.linspace(0, 5000, 251)
z = np.linspace(-900, 900, 181)
f1 = {"x": r3(np.concatenate([-xh[:0:-1], xh])), "z": r3(z), "h0": h0, "core": [-220.0, 0.0]}
for pol, tag in ((2, "TE"), (1, "TM")):
    poles = S.guided_poles_gamma(st, k, pol)
    p = S.build_path(k, S.stack_features(st, k, pol, poles), 5000.0, h0)
    U = NF.field_map(st, k, pol, h0, xh, z, p)
    U = np.concatenate([U[:, :0:-1], U], axis=1)
    x = np.concatenate([-xh[:0:-1], xh])
    Gm, _ = guided_field(st, k, pol, h0, x, z)
    norm = np.max(np.abs(Gm))
    rad = U - Gm
    big = np.hypot(*np.meshgrid(x, z - h0)) < 15.0  # the source singularity, masked
    lg = np.log10(np.abs(U) / norm)
    lr = np.log10(np.abs(rad) / norm)
    lg[big] = np.nan
    lr[big] = np.nan
    f1[tag] = {"re": r3(np.where(big, np.nan, U.real / norm), 3), "im": r3(np.where(big, np.nan, U.imag / norm), 3),
               "log": r3(lg, 3), "lograd": r3(lr, 3),
               "neff": [float(b / (2 * np.pi / LAM)) for b in poles]}
    print("F1", tag, "done; n_eff", f1[tag]["neff"])
out["f1"] = f1

# --- F3: power partition vs dipole height, SOI TE and TM -------------------------
kw = dict(epsabs=1e-17, epsrel=1e-12, limit=2000)
hs = np.geomspace(5, 1500, 45)
f3 = {"h": r3(hs)}
for pol, tag in ((2, "TE"), (1, "TM")):
    poles = S.guided_poles_gamma(st, k, pol)
    modes = [(b,) + S.mode_profile(st, k, pol, b)[:2] for b in poles]
    rows = {"up": [], "down": [], "guided": [], "total": [], "err": []}
    for hh in hs:
        p = S.build_path(k, S.stack_features(st, k, pol, poles), 2 * hh, 2 * hh)
        tot = 0.25 + S.g_path(p, st, k, pol, 0.0, 2 * hh).imag
        up = quad(lambda th: abs(np.exp(-1j * k * np.cos(th) * hh) + st.R(np.array([k * np.sin(th) + 0j]), k, pol)[0]
                                 * np.exp(1j * k * np.cos(th) * hh)) ** 2 / 4, 0, np.pi / 2, **kw)[0] / np.pi
        wb = 1.0 if pol == 2 else st.eps_bot.real
        dn = quad(lambda q: np.sqrt(max(k * k - q * q, 0)) / wb * abs(st.tau(np.array([q + 0j]), k, pol)[0]) ** 2
                  * abs(np.exp(1j * S.a_phys(k, q) * hh)) ** 2 / 4, 0, k, **kw)[0] / np.pi
        gd = sum((u(np.array([hh]))[0] ** 2 / (2 * b * N)).real for b, u, N in modes)
        for key, v in (("up", up), ("down", dn), ("guided", gd), ("total", tot)):
            rows[key].append(v)
        rows["err"].append(abs(tot - up - dn - gd) / tot)
    f3[tag] = {kk_: r3(np.array(v) / 0.25 if kk_ != "err" else v, 5) for kk_, v in rows.items()}
    print("F3", tag, "max balance error", max(rows["err"]))
out["f3"] = f3

# --- F4: mode chart, n_eff vs Si thickness ---------------------------------------
ts = np.linspace(20, 1200, 237)
f4 = {"t": r3(ts)}
for pol, tag in ((2, "TE"), (1, "TM")):
    branches = {}
    for t in ts:
        s4 = S.Stack([ESI], [t], 1.0)
        lo, hi = S.guided_window(s4, k)
        gg = np.linspace(0, np.sqrt(hi * hi - lo * lo), 3001)[:-1]
        qg = np.sqrt(lo * lo + gg * gg) + 0j
        v = s4.D(qg, k, pol).real
        f = lambda g: float(s4.D(np.array(np.sqrt(lo * lo + g * g) + 0j), k, pol).real)
        roots = [brentq(f, a, b) for a, b, fa, fb in zip(gg[:-1], gg[1:], v[:-1], v[1:]) if fa * fb < 0]
        betas = sorted((np.sqrt(lo * lo + g * g) for g in roots), reverse=True)
        for m, b in enumerate(betas):
            branches.setdefault(m, {}).update({float(t): float(b / (2 * np.pi / LAM))})
    f4[tag] = {str(m): [br.get(float(t)) for t in ts] for m, br in branches.items()}
    print("F4", tag, "orders", len(branches))
out["f4"] = f4

# --- F5: per-mode coupling vs height, multimode Si 700 nm, TE --------------------
st5 = S.Stack([ESI], [700.0], 1.0)
f5 = {"h": r3(hs)}
for pol, tag in ((2, "TE"), (1, "TM")):
    poles = S.guided_poles_gamma(st5, k, pol)
    modes = [(b,) + S.mode_profile(st5, k, pol, b)[:2] for b in poles]
    per = []
    for b, u, N in modes:
        per.append([(u(np.array([hh]))[0] ** 2 / (2 * b * N)).real / 0.25 for hh in hs])
    tot = []
    for hh in hs:
        p = S.build_path(k, S.stack_features(st5, k, pol, poles), 2 * hh, 2 * hh)
        tot.append((0.25 + S.g_path(p, st5, k, pol, 0.0, 2 * hh).imag) / 0.25)
    f5[tag] = {"modes": [r3(np.array(r) / np.array(tot), 5) for r in per],
               "neff": [float(b / (2 * np.pi / LAM)) for b in poles]}
    print("F5", tag, "modes", len(poles))
out["f5"] = f5

# --- F6: directional coupler, Si/SiO2 300/Si, TE ------------------------------------
st6 = S.Stack([ESI, 1.0, ESI], [220.0, 300.0, 220.0], 1.0)
poles6 = S.guided_poles_gamma(st6, k, 2)
x6 = np.linspace(0, 40000, 801)
z6 = np.linspace(-1150, 450, 161)
p6 = S.build_path(k, S.stack_features(st6, k, 2, poles6), 40000.0, h0)
U6 = NF.field_map(st6, k, 2, h0, x6, z6, p6)
Gm6, _ = guided_field(st6, k, 2, h0, x6, z6)
n6 = np.max(np.abs(Gm6))
big = np.hypot(*np.meshgrid(x6, z6 - h0)) < 15.0
A6 = np.abs(U6) / n6
A6[big] = np.nan
zf = np.linspace(-740, 0, 741)
uf = NF.field_map(st6, k, 2, h0, x6, zf, p6)
top = np.trapezoid(np.abs(uf[zf >= -220]) ** 2, zf[zf >= -220], axis=0)
bot = np.trapezoid(np.abs(uf[zf <= -520]) ** 2, zf[zf <= -520], axis=0)
Lb = np.pi / (poles6[0] - poles6[1])
out["f6"] = {"x": r3(x6), "z": r3(z6), "abs": r3(A6, 3), "top": r3(top / (top + bot), 4),
             "bot": r3(bot / (top + bot), 4), "Lb": float(Lb),
             "neff": [float(b / (2 * np.pi / LAM)) for b in poles6], "layers": [[-220, 0], [-740, -520]]}
print("F6 beat length", Lb)

with open(sys.argv[1], "w") as fh:
    json.dump(out, fh, separators=(",", ":"), allow_nan=True)
print("wrote", sys.argv[1])
