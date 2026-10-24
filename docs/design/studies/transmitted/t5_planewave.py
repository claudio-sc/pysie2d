"""T5: transmitted plane wave, interface conditions, energy, TIR decay (spec §7.2)."""

import numpy as np

from pysie2d.layered import _normalised, alpha, fresnel_r
from tproto import tau

Z_INT = -37.0  # off zero, so a wrong z_top/z_bot phase reference shows


def fields(k, eps, pol, theta, x, z):
    """(ψ above, ∂zψ above, ψ below, ∂zψ below) at (x, z), z just at the interface."""
    q, a1 = k * np.sin(theta), k * np.cos(theta)
    eps = _normalised(eps)
    a2 = alpha(np.asarray(q), k * np.sqrt(eps))
    r = fresnel_r(np.asarray(q), k, eps, pol)
    inc = np.exp(1j * k * (x * np.sin(theta) - z * np.cos(theta)))
    ref = r * np.exp(1j * (q * x + a1 * (z - 2 * Z_INT)))
    tra = a1 * tau(np.asarray(q), k, eps, pol) * np.exp(1j * (q * x - a1 * Z_INT - a2 * (z - Z_INT)))
    up, dup = inc + ref, -1j * a1 * inc + 1j * a1 * ref
    return up, dup, tra, -1j * a2 * tra, a2


def main():
    x = 123.0
    cases = [("glass", 2.25 / 1.33**2), ("low-n", 1 / 1.33**2), ("lossy", (2.25 + 0.1j) / 1.33**2),
             ("silver", (-18.3 + 0.48j) / 1.33**2), ("metal -4", -4.0 / 1.33**2)]
    for kname, k in (("real k", 2 * np.pi * 1.33 / 633), ("complex k", 2 * np.pi * 1.33 / (633 + 31.6j))):
        for name, eps in cases:
            for pol in (2, 1):
                worst_v = worst_d = worst_e = 0.0
                for th in np.deg2rad([0.0, 20.0, 45.0, 60.0, 80.0, -35.0]):
                    up, dup, dn, ddn, a2 = fields(k, eps, pol, th, x, Z_INT)
                    w = 1.0 if pol == 2 else _normalised(eps)  # TM: (1/ε)∂zψ continuous
                    worst_v = max(worst_v, abs(up - dn) / abs(up + 1e-300) if abs(up) > 0 else abs(dn))
                    worst_d = max(worst_d, abs(dup - ddn / w) / abs(k))
                    if kname == "real k" and np.imag(eps) == 0:
                        # lossless: |r|² + Re(w_bot)/α₁·|T|² = 1 (TM weight 1/ε)
                        q, a1 = k * np.sin(th), k * np.cos(th)
                        r = fresnel_r(np.asarray(q), k, eps, pol)
                        T = a1 * tau(np.asarray(q), k, eps, pol)
                        e = abs(abs(r) ** 2 + (a2 / w).real / a1 * abs(T) ** 2 - 1)
                        worst_e = max(worst_e, e)
                print(f"{kname:9s} {name:8s} pol={pol}  value {worst_v:.1e}  "
                      f"deriv {worst_d:.1e}  energy {worst_e:.1e}")
    # TIR (low-n at 60° > θ_c = 48.8°): decays like exp(−|α₂|·depth)
    k = 2 * np.pi * 1.33 / 633
    for pol in (2, 1):
        _, _, d0, _, a2 = fields(k, 1 / 1.33**2, pol, np.deg2rad(60.0), x, Z_INT)
        _, _, d1, _, _ = fields(k, 1 / 1.33**2, pol, np.deg2rad(60.0), x, Z_INT - 200.0)
        rate = -np.log(abs(d1) / abs(d0)) / 200.0
        print(f"TIR pol={pol}: Im α₂ = {a2.imag:.6e}, Re α₂ = {a2.real:.1e}, "
              f"measured decay {rate:.6e}, rel {abs(rate - a2.imag) / a2.imag:.1e}")


if __name__ == "__main__":
    main()
