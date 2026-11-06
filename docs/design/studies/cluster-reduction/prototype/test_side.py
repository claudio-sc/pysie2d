import numpy as np
from scipy.special import jn_zeros
from pysie2d import BIESolver, Geometry, Material
rad = 100.0; nn = 64
geom = Geometry.gielis(rad=rad, n_pts=nn, m=0)
def ratio(m):
    u, s, vh = np.linalg.svd(m); return s[-1]/s[0], np.linalg.norm(u[nn:, -1]), np.linalg.norm(u[:nn, -1])
# (1) existing lambda-QNM solver: interior k_c = n_core*k_bg; H_2^(1)(z)=0 at z=0.429485-1.281374j
z = 0.429485 - 1.281374j
for n_core in (1.5, 3.5):
    lam_s = 2*np.pi*n_core*rad/z
    s = BIESolver(geom, Material(n_core=n_core, pol=2))
    r, ui, ue = ratio(s.assemble(lam_s)); r2, *_ = ratio(s.assemble(lam_s*1.02))
    print(f"QNM side: n_core={n_core} lam_s={lam_s:.2f} nm  Q={lam_s.real/(2*lam_s.imag):.2f}  "
          f"smin/smax={r:.1e} (2% away {r2:.1e}); |u_int|={ui:.3f} |u_ext|={ue:.3f}")
# (2) exterior rows: interior Dirichlet eigenvalue of k_bg, J_0(k_bg a)=0
j01 = jn_zeros(0, 1)[0]
lam_d = 2*np.pi*rad/j01
for eps in (-1.5-0.3j, 2.0-0.5j, 9.0-1.0j):
    m = Material.from_eps(eps, pol=1); s = BIESolver(geom, m)
    r, ui, ue = ratio(s.assemble(lam_d)); r2, *_ = ratio(s.assemble(lam_d*1.03))
    print(f"exterior side: lam={lam_d:.2f} nm (k_bg a = j01) eps={eps}: smin/smax={r:.1e} (3% away {r2:.1e}); "
          f"|u_ext|={ue:.3f}")
