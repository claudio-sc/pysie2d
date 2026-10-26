"""S10 (anchor A5): small-circle limit of the mode S-matrix, TE, against first Born.

A circle of radius a and relative ε at r_p = (x_p, z_p) in the cover. First Born with the
full stack Green function (the 'dressing' by the stack is inside G):
    u_s(r) ≈ k²(ε − 1) π a² G(r, r_p) u_inc(r_p)
Its guided part (S9) gives, for incident ũ_m e^{iβ_m x} (power 1) and x_ref = 0,
    S_{n←m}^{fwd} − δ_nm ≈ k²(ε−1)πa² · (i/(2β_n N_n)) u_n(z_p) e^{−iβ_n x_p} ũ_m(z_p) e^{iβ_m x_p} · √(β_n N_n)
    S_{n←m}^{bwd}        ≈ same with e^{+iβ_n x_p}
Relative error must fall like a² (field variation over the disk, self-interaction): an
accidental pass would need the wrong formula to share the a² law and the constant.
"""
import numpy as np

import sproto as S
from pysie2d import Geometry, Material
from s12_particle import Ctx, shifted

nc = 1.444
cases = {"SOI 220 @1550": (S.Stack([(3.476 / nc) ** 2], [220.0], 1.0), 1550),
         "SiN 400 @800 (2 modes)": (S.Stack([(2.0 / nc) ** 2], [400.0], 1.0), 800)}
zp, xp, n_part = 60.0, 23.0, 2.0
for name, (st, lam) in cases.items():
    k = 2 * np.pi * nc / lam
    eps = (n_part / nc) ** 2
    prev = None
    for a in (16.0, 8.0, 4.0, 2.0):
        geom = Geometry.gielis(rad=a, n_pts=64, m=4, x0=xp, z0=zp)
        mat = Material(n_core=n_part, n_clad=nc, pol=2)
        ctx = Ctx(st, lam, nc, mat, geom, 2)
        M = ctx.matrix()
        err = 0.0
        for mi, (bm, um, Nm) in enumerate(ctx.modes):
            am = 1 / np.sqrt((bm * Nm).real)
            rhs = np.zeros(2 * geom.n_pts, dtype=complex)
            rhs[: geom.n_pts] = am * um(geom.g) * np.exp(1j * bm * geom.f)
            ei = np.linalg.solve(M, rhs)
            for ni, ((bn, un, Nn), (cp, cm)) in enumerate(zip(ctx.modes, ctx.overlap(ei))):
                s = np.sqrt((bn * Nn).real)
                pre = k * k * (eps - 1) * np.pi * a * a * 1j / (2 * bn * Nn) * un(np.array([zp]))[0] * am * um(np.array([zp]))[0] * np.exp(1j * bm * xp) * s
                for c, born in ((cp * s, pre * np.exp(-1j * bn * xp)), (cm * s, pre * np.exp(1j * bn * xp))):
                    err = max(err, abs(c - born) / abs(born))
        rate = f"  ratio {prev / err:.2f}" if prev else ""
        print(f"  {name:24s} a = {a:4.1f} nm (k·a = {k * a:.3f}): max |S_BIE − S_Born|/|S_Born| = {err:.2e}{rate}")
        prev = err
