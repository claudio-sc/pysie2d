"""v1.2 slab waveguide: an N-layer Stack background (slab-spec.md §5, gates GW1–GW18).

Each gate's bar is 10× the value measured in the commit that added it, and the
measured number is quoted beside it. Independent anchors: the three-layer
closed form (GW6), the real-axis references of ``reference.sommerfeld`` with a
different R, branch rule and quadrature (GW4, GW5), power balance as an
equality with every term from its own integral (GW9, GW12, GW13), reciprocity
(GW14), the Born limit (GW15) and the exact translation law (GW17).
"""

import numpy as np
import pytest

from pysie2d.layered import SommerfeldPath, _singularities

PI = np.pi
N_CLAD = 1.444
K = 2 * PI * N_CLAD / 1550.0


# --- GW1: the feature-list path is the v0.9 path when given v0.9's features -----


@pytest.mark.parametrize("pol", [2, 1])
@pytest.mark.parametrize("eps_rel", [2.25 / N_CLAD**2, (-18.3 + 0.48j) / N_CLAD**2])
@pytest.mark.parametrize("k", [K, K * (1 - 0.05j)], ids=["real", "complex"])
def test_feature_path_with_half_space_features_is_bit_identical(k, eps_rel, pol):
    # The only change to _build is where the singularities come from; handed
    # _singularities' own list it must reproduce for_wavenumber bit for bit, so
    # every v0.9 rule (T, δ, C1, node counts, banding) is provably unchanged.
    ref = SommerfeldPath.for_wavenumber(k, eps_rel, pol, 900.0, 40.0)
    got = SommerfeldPath.for_features(k, _singularities(k, eps_rel, pol), 900.0, 40.0)
    assert np.array_equal(ref.q, got.q) and np.array_equal(ref.w, got.w)
    assert ref.bands == got.bands and (ref.T, ref.delta) == (got.T, got.delta)
