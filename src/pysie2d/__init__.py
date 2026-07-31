"""pysie2d — 2-D boundary-integral scattering solver (homogeneous background).

Public API:
    Geometry: Gielis-superformula boundary parameterisation.
    Material: optical properties of the scatterer.
    BIESolver: solver façade; call ``scatter``/``scatter_dipole`` to obtain a
        ``ScatterResult``.
    ScatterResult: carries the solution and exposes far/near-field analysis.
    assemble_matrix: the vectorised BIE system-matrix assembly.
    plane_wave_rhs, line_dipole_rhs: excitation right-hand sides.
    eval_field, far_field: field-evaluation primitives.
    self_green, relative_ldos, relative_ldos_map: self-Green function and
        LDOS / Purcell-effect analysis (line-dipole excitation).
    QNMSolver: quasi-normal-mode façade; call ``modes`` to obtain a
        ``QNMResult``.
    QNMResult: mode wavelengths, vectors, and extraction diagnostics.
"""

from .fields import eval_field, far_field
from .geometry import Geometry
from .green import relative_ldos, relative_ldos_map, self_green
from .kernels import assemble_matrix, assemble_matrix_reference
from .material import Material
from .qnm import QNMResult, QNMSolver
from .solver import BIESolver, ScatterResult
from .sources import line_dipole_rhs, plane_wave_rhs

__version__ = "0.2.0"

__all__ = [
    "BIESolver",
    "Geometry",
    "Material",
    "QNMResult",
    "QNMSolver",
    "ScatterResult",
    "assemble_matrix",
    "assemble_matrix_reference",
    "eval_field",
    "far_field",
    "line_dipole_rhs",
    "plane_wave_rhs",
    "relative_ldos",
    "relative_ldos_map",
    "self_green",
]
