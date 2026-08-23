"""pysie2d — 2-D boundary-integral scattering solver (homogeneous background).

Public API:
    Geometry: Gielis-superformula boundary parameterisation.
    Material: optical properties of the scatterer.
    BIESolver: solver façade; call ``scatter``/``scatter_dipole`` to obtain a
        ``ScatterResult``.
    ScatterResult: carries the solution and exposes far/near-field analysis.
    assemble_matrix, assemble_matrix_reference: the vectorised BIE
        system-matrix assembly and its loop-order reference implementation,
        kept as a validation anchor.
    plane_wave_rhs, line_dipole_rhs: excitation right-hand sides.
    eval_field, far_field: field-evaluation primitives.
    self_green, relative_ldos, relative_ldos_map: self-Green function and
        LDOS / Purcell-effect analysis (line-dipole excitation).
    QNMSolver: quasi-normal-mode façade; call ``modes`` to obtain a
        ``QNMResult``.
    QNMResult: mode wavelengths, vectors, and extraction diagnostics; call
        ``refine`` for bordered-Newton polishing.
    DEGENERATE_COND: the ``cond_jacobian`` threshold above which a pole is
        taken to be degenerate and left unrefined.
    size_parameter: derived Mie size parameter x = 2π·n_clad·rad/λ_vac.
    wavelength_over_ds: boundary points per interior wavelength (§10).

All public wavelengths are **vacuum** wavelengths in nm; the low-level
primitives (``assemble_matrix``, ``assemble_matrix_reference``,
``eval_field``, ``far_field``, ``plane_wave_rhs``, ``line_dipole_rhs``)
take a background wavenumber
``wnum_bg = 2π·n_clad/λ_vac`` instead. See ``docs/conventions.md`` §2.
"""

from .fields import eval_field, far_field
from .geometry import Geometry
from .green import relative_ldos, relative_ldos_map, self_green
from .kernels import assemble_matrix, assemble_matrix_reference
from .material import Material
from .qnm import DEGENERATE_COND, QNMResult, QNMSolver
from .solver import (
    BIESolver,
    ScatterResult,
    size_parameter,
    wavelength_over_ds,
)
from .sources import line_dipole_rhs, plane_wave_rhs

__version__ = "0.4.2"

__all__ = [
    "BIESolver",
    "DEGENERATE_COND",
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
    "size_parameter",
    "wavelength_over_ds",
]
