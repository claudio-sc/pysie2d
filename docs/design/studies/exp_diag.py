import sys, os, shutil, re
import numpy as np

SRC = "/home/claude/pysie2d-0.5.0/src"

# Build a patched copy of the package with a tunable self-patch constant.
# kernels.py uses  arg_d = wnum * delt / (2.0 * e) * gamma  with e = np.e.
# Setting e = C/2 makes the effective argument  wnum * delt * gamma / C.
def make_variant(tag, C):
    dst = f"/home/claude/var_{tag}/pysie2d"
    if os.path.exists(f"/home/claude/var_{tag}"):
        shutil.rmtree(f"/home/claude/var_{tag}")
    shutil.copytree(f"{SRC}/pysie2d", dst)
    p = f"{dst}/kernels.py"
    s = open(p).read()
    n = s.count("e = np.e")
    s = s.replace("e = np.e", f"e = {C!r} / 2.0")
    open(p, "w").write(s)
    assert n >= 1, "constant not found"
    return f"/home/claude/var_{tag}"


CANDIDATES = {
    "current 2e": 2 * np.e,
    "pi*exp(gamma)": np.pi * np.exp(np.euler_gamma),
}
for tag, C in CANDIDATES.items():
    make_variant(re.sub(r"\W", "", tag), C)

RAD, NCORE, LAM = 200.0, 1.5, 600.0
TWOPI = 2 * np.pi


def run(path, nns):
    """Run in a subprocess so each variant imports its own copy."""
    code = f'''
import sys; sys.path.insert(0, {path!r})
import numpy as np
from pysie2d import BIESolver, Geometry, Material
from pysie2d.reference import mie
mat = Material(n_core={NCORE}, n_clad=1.0, pol=2)
x = 2*np.pi*{RAD}/{LAM}
qm = mie.efficiencies(x, complex({NCORE}))["Q_ext_TE"]
out = []
for nn in {list(nns)!r}:
    g = Geometry.gielis(rad={RAD}, n_pts=nn, m=0)
    r = BIESolver(g, mat).scatter(wavelength={LAM})
    amp, _ = r.far_field(4000)
    d = 2*np.pi/3999.0
    q = amp[int((2*np.pi - np.deg2rad(r.angle))/d)].imag / (r.wnum_bg*2*{RAD})
    out.append(abs(q-qm)/qm)
print(" ".join(f"{{v:.6e}}" for v in out))
'''
    import subprocess
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    if r.returncode:
        print(r.stderr[-800:])
        raise SystemExit(1)
    return [float(v) for v in r.stdout.split()]


NNS = (60, 120, 240, 480)
print("Relative error of qext against analytic Mie, circle rad=200nm, TE, 600nm\n")
print(f"{'self-patch constant':22s}" + "".join(f"{n:>12d}" for n in NNS) + "    rates")
for tag in CANDIDATES:
    path = f"/home/claude/var_{re.sub(chr(92)+'W', '', tag) if False else re.sub(r'W', '', '')}"
for tag, C in CANDIDATES.items():
    path = f"/home/claude/var_{re.sub(r'[^A-Za-z0-9]', '', tag)}"
    e = run(path, NNS)
    rates = [np.log2(e[i] / e[i + 1]) for i in range(len(e) - 1)]
    print(f"{tag:22s}" + "".join(f"{v:12.3e}" for v in e)
          + "    " + " ".join(f"{r:5.2f}" for r in rates))
