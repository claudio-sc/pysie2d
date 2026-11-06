import numpy as np, json
import os
HERE = os.path.dirname(os.path.abspath(__file__))
d = np.load(os.path.join(HERE, "expand2.npz")); eps, exact, poles = d["eps"], d["exact"], d["poles"]
re = eps.real; fitted = poles.copy()
for msk in ((re <= -1.3), (re > -1.3) & (re < -0.7), (re >= -0.7)):
    idx = np.flatnonzero(msk)[[0, msk.sum()//2, -1]]
    c = np.linalg.solve(np.vander(eps[idx], 3, increasing=True), exact[idx]-poles[idx])
    fitted[msk] = poles[msk] + np.vander(eps[msk], 3, increasing=True) @ c
k = slice(0, None, 2)
print(json.dumps({"x": [f"{v:.2f}" for v in re[k]], "exact": np.round(np.abs(exact[k]), 5).tolist(),
                  "poles": np.round(np.abs(poles[k]), 5).tolist(), "fit": np.round(np.abs(fitted[k]), 5).tolist()}))
