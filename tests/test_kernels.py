import numpy as np
import pytest

from pysie2d.geometry import Geometry
from pysie2d.kernels import assemble_matrix, assemble_matrix_reference
from pysie2d.material import Material


@pytest.mark.parametrize("pol", [1, 2])
@pytest.mark.parametrize("wn", [2.0 * np.pi / 600.0, 2.0 * np.pi / 600.0 + 1e-4j])
def test_fast_assembly_matches_reference(pol, wn):
    # Two functions implement identical arithmetic in a different loop order,
    # so they must agree to rounding. Any looser agreement means the port
    # broke an expression.
    nn = 64
    geom = Geometry.gielis(rad=200.0, n_pts=nn, m=0)
    mat = Material(n_core=1.5, n_clad=1.0, pol=pol)

    args = (
        pol,
        nn,
        geom.f,
        geom.g,
        geom.df,
        geom.dg,
        geom.ddf,
        geom.ddg,
        geom.delt,
        wn,
        mat.nc,
        mat.eps,
    )
    m_fast = assemble_matrix(*args)
    m_ref = assemble_matrix_reference(*args)

    assert np.allclose(m_fast, m_ref, rtol=1e-12)
