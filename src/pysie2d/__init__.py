"""pysie2d — 2-D boundary-integral scattering solver (homogeneous background).

Public API:
    Geometry: Gielis-superformula boundary, sampled on a Parametrisation.
        ``Geometry.gielis`` raises ``pysie2d.geometry.NonClosingBoundaryError``
        (a ``ValueError``) when the superformula does not close.
    Parametrisation: the frozen node map θ = w(t); ``uniform_theta()`` is the
        default, ``gielis(...)`` places nodes near-uniformly in arc length.
    Material: optical properties of the scatterer.
    BIESolver: solver façade; call ``scatter``/``scatter_dipole`` to obtain a
        ``ScatterResult``.
    ScatterResult: carries the solution and exposes far/near-field analysis.
    assemble_matrix, assemble_matrix_reference: the vectorised BIE
        system-matrix assembly and its loop-order reference implementation,
        kept as a validation anchor.
    assemble_cross_block: the M1/M2 coupling blocks between two
        non-overlapping boundaries (conventions §14).
    plane_wave_rhs, line_dipole_rhs: excitation right-hand sides.
    eval_field, far_field: field-evaluation primitives.
    self_green, relative_ldos, relative_ldos_map: self-Green function and
        LDOS / Purcell-effect analysis (line-dipole excitation).
    multipole_decompose, multipole_reconstruct: cylindrical-harmonic
        decomposition of the scattered field outside the circumscribing
        circle, and its inverse. ``ScatterResult.multipoles`` is the façade.
    Multipoles: the signed-order coefficients c_m, with the symmetric /
        antisymmetric pair ``c_plus``/``c_minus`` derived from them.
    Cluster: an arrangement of non-overlapping particle boundaries sharing one
        background (conventions §14). Construction raises
        ``ClusterOverlapError`` when two boundaries intersect.
    ClusterBIESolver: coupled cluster solver; call ``scatter``/
        ``scatter_dipole`` to obtain a ``ClusterScatterResult``. ``pol`` and
        ``n_clad`` are validated equal across the materials.
    ClusterScatterResult: the coupled solution, with the total far field,
        absolute cross-sections and near fields. Deliberately no
        ``efficiencies()`` (a cluster has no ``rad``) and no ``multipoles()``.
    ClusterOverlapError, ClusterGapWarning, ClusterResolutionWarning: the
        cluster geometry guards — overlapping boundaries, a gap too small for
        the resolution in use, and a particle far coarser than its neighbours.
    QNMSolver: quasi-normal-mode façade; call ``modes`` to obtain a
        ``QNMResult``.
    QNMResult: mode wavelengths, vectors, and extraction diagnostics; call
        ``refine`` for bordered-Newton polishing.
    DEGENERATE_COND: the ``cond_jacobian`` threshold above which a pole is
        taken to be degenerate and left unrefined.
    SHAPE_STEP: default central-difference step for ``QNMResult.sensitivity``,
        in the parameter's own units (§§10, 11).
    size_parameter: derived Mie size parameter x = 2π·n_clad·rad/λ_vac.
    wavelength_over_ds: boundary points per interior wavelength (§10).
    richardson_limit: two-rung extrapolation of a quantity converging at
        first order in ``n_pts``. Since v0.6 neither λ nor dλ/dp is such a
        quantity (§12); **deprecated** (F7), emits ``DeprecationWarning``.

All public wavelengths are **vacuum** wavelengths in nm; the low-level
primitives (``assemble_matrix``, ``assemble_matrix_reference``,
``eval_field``, ``far_field``, ``plane_wave_rhs``, ``line_dipole_rhs``)
take a background wavenumber
``wnum_bg = 2π·n_clad/λ_vac`` instead. See ``docs/conventions.md`` §2.
"""

from .cluster import (
    Cluster,
    ClusterBIESolver,
    ClusterGapWarning,
    ClusterOverlapError,
    ClusterResolutionWarning,
    ClusterScatterResult,
)
from .fields import eval_field, far_field
from .geometry import Geometry
from .green import relative_ldos, relative_ldos_map, self_green
from .kernels import (
    assemble_cross_block,
    assemble_matrix,
    assemble_matrix_reference,
)
from .material import Material
from .multipole import Multipoles
from .multipole import decompose as multipole_decompose
from .multipole import reconstruct as multipole_reconstruct
from .parametrisation import Parametrisation
from .qnm import (
    DEGENERATE_COND,
    SHAPE_STEP,
    QNMResult,
    QNMSolver,
    richardson_limit,
)
from .solver import (
    BIESolver,
    ScatterResult,
    size_parameter,
    wavelength_over_ds,
)
from .sources import line_dipole_rhs, plane_wave_rhs

__version__ = "0.7.0"

__all__ = [
    "BIESolver",
    "Cluster",
    "ClusterBIESolver",
    "ClusterGapWarning",
    "ClusterOverlapError",
    "ClusterResolutionWarning",
    "ClusterScatterResult",
    "DEGENERATE_COND",
    "Geometry",
    "Material",
    "Multipoles",
    "Parametrisation",
    "QNMResult",
    "QNMSolver",
    "SHAPE_STEP",
    "richardson_limit",
    "ScatterResult",
    "assemble_cross_block",
    "assemble_matrix",
    "assemble_matrix_reference",
    "eval_field",
    "far_field",
    "line_dipole_rhs",
    "multipole_decompose",
    "multipole_reconstruct",
    "plane_wave_rhs",
    "relative_ldos",
    "relative_ldos_map",
    "self_green",
    "size_parameter",
    "wavelength_over_ds",
]
