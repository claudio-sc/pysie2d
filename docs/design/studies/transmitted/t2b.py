"""P_up with a split at the cover critical angle; lossless metal with P_down = 0."""
import numpy as np
from pysie2d import HalfSpace, relative_ldos
from pysie2d.fields import _upward_amplitude, _upward_cross_sections
from pysie2d.layered import _normalised, fresnel_r
from t2_power import LAM, XS, ZS, solver, powers
PI = np.pi

def p_up_split(s, n=200, split=True):
    res = s.scatter_dipole(LAM, XS, ZS)
    k, pol, bg = res.wnum_bg, s.material.pol, s.background
    eps = bg.eps_rel(1.33, LAM)
    z_int = bg.z_int
    brk = [-0.5*PI, 0.5*PI]
    e = _normalised(eps) if eps is not None else None
    if split and e is not None and e.imag == 0 and 0 < e.real < 1:
        tc = np.arcsin(np.sqrt(e.real)); brk = [-0.5*PI, -tc, tc, 0.5*PI]
    xg, wg = np.polynomial.legendre.leggauss(n); tot = 0.0
    for lo, hi in zip(brk[:-1], brk[1:]):
        c, h = 0.5*(lo+hi), 0.5*(hi-lo)
        th = c + h*np.sin(0.5*PI*xg); ww = h*0.5*PI*np.cos(0.5*PI*xg)*wg
        a_up = _upward_amplitude(th, *res._substrate())
        r = fresnel_r(k*np.sin(th), k, eps, pol)
        sd = np.exp(-1j*k*(XS*np.sin(th)+ZS*np.cos(th))) + r*np.exp(-1j*k*(XS*np.sin(th)+(2*z_int-ZS)*np.cos(th)))
        tot += np.sum(ww*np.abs(sd+a_up)**2)
    return tot/(8*PI)

if __name__ == '__main__':
    for name, bg in (("low-n", HalfSpace(1.0)), ("glass", HalfSpace(2.25))):
        for pol in (2, 1):
            s = solver(pol, bg)
            p_in, p_up, p_abs, p_down, _ = powers(s)
            up_s = p_up_split(s); up_s2 = p_up_split(s, 400)
            print(f"{name} pol={pol}: G14-style up err {(p_up-up_s)/p_in:+.1e}; split balance {(up_s+p_down+p_abs-p_in)/p_in:+.1e}; split 200 vs 400 {(up_s-up_s2)/p_in:+.1e}")
            # what v0.9's c_sca_up loses for a plane wave
            res = s.scatter(LAM, 20.0)
            for n in (500, 2000, 8000):
                print(f"   c_sca_up n_angles={n}: {_upward_cross_sections(n, *res._substrate())['c_sca_up']:.15f}")
    # lossless metal: P_down = 0
    for pol in (2, 1):
        s = solver(pol, HalfSpace(-4.0+0j))
        res = s.scatter_dipole(LAM, XS, ZS)
        x, w = np.polynomial.legendre.leggauss(500); th = 0.5*PI*x
        k = res.wnum_bg; eps = s.background.eps_rel(1.33, LAM)
        a_up = _upward_amplitude(th, *res._substrate())
        r = fresnel_r(k*np.sin(th), k, eps, pol)
        sd = np.exp(-1j*k*(XS*np.sin(th)+ZS*np.cos(th))) + r*np.exp(-1j*k*(XS*np.sin(th)+(-ZS)*np.cos(th)))
        p_up = np.sum(0.5*PI*w*np.abs(sd+a_up)**2)/(8*PI)
        g = s.geometry; nn = g.n_pts
        p_abs = -np.imag(g.delt*np.sum(np.conj(res.ei[:nn])*res.ei[nn:]))
        p_in = relative_ldos(s, LAM, XS, ZS)/4
        print(f"lossless metal eps=-4 pol={pol}: (up+abs-in)/in = {(p_up+p_abs-p_in)/p_in:+.2e}")
