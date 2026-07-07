"""pysie2d — 2-D boundary-integral scattering solver (homogeneous background).

Public API:
    Geometry: Gielis-superformula boundary parameterisation.
    Material: optical properties of the scatterer.
    BIESolver: solver façade; call ``scatter`` to obtain a ``ScatterResult``.
    ScatterResult: carries the solution and exposes far/near-field analysis.
    assemble_matrix: the vectorised BIE system-matrix assembly.
    plane_wave_rhs: plane-wave excitation right-hand side.
    eval_field, far_field: field-evaluation primitives.
"""

from .fields import eval_field, far_field
from .geometry import Geometry
from .kernels import assemble_matrix, assemble_matrix_reference
from .material import Material
from .solver import BIESolver, ScatterResult
from .sources import plane_wave_rhs

__version__ = "0.1.0"

__all__ = [
    "BIESolver",
    "Geometry",
    "Material",
    "ScatterResult",
    "assemble_matrix",
    "assemble_matrix_reference",
    "eval_field",
    "far_field",
    "plane_wave_rhs",
]
