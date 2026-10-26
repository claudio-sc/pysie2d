from pathlib import Path
import numpy as np
for pol, ms in [(1, (1, 3, 5)), (2, (0, 4, 8))]:
    d = np.load(Path(__file__).resolve().parent / f"results/ite_pol{pol}.npz"); es, e0, O = d["es"], d["e0"], d["orders"]
    mmax = (O.shape[1] - 1) // 2
    print(f"pol={pol} ITE {e0:.6f}")
    for k in range(0, len(es), 8):
        print(f"  eps {es[k]:9.5f} rel {es[k]/e0-1:+.4f}  " + "  ".join(f"|c{m}| {O[k, m+mmax]:.2e}" for m in ms) + f"  class P {d['cl'][k]:.2e}")
    for m in ms:
        i = np.argmin(O[:, m + mmax]); print(f"  min |c{m}| at rel {es[i]/e0-1:+.2e}: {O[i, m+mmax]:.2e}")
