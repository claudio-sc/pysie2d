"""Frozen cross-section spectrum files — the format, and its reader/writer.

This module is the machine-readable half of the v1.0 external-validation
contract (``docs/design/v1.0-external-validation.md`` §6). It is **not part of
the installed package**: it lives beside the external drivers that produce the
frozen data, and the repo's own test imports it from here so that the schema
has exactly one definition.

A frozen file carries a cross-section spectrum *and its whole provenance*:
what geometry and material were simulated, in which polarisation, by which
tool at which version, on what discretisation, with what evidence that the
discretisation had converged, and what agreement the repo is therefore
entitled to assert. A reader opening one file learns all of that without
running anything — which is the point of freezing it rather than quoting a
number in a docstring.

Everything is JSON: a spectrum is a few hundred floats, so the numbers stay
reviewable in a diff, and reading one needs nothing but the standard library.

Functions:
    write: serialise a :class:`Spectrum` to a JSON file.
    read: load a JSON file back, validating the schema version and the
        array lengths.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

SCHEMA = "pysie2d-external-validation/1"
"""Schema tag written into every file.

Bumped when a field changes meaning. ``read`` refuses anything else rather
than silently reinterpreting an older layout.
"""

_SPECTRA = ("wavelength_nm", "c_sca_nm", "c_ext_nm", "c_abs_nm")


@dataclass(frozen=True)
class Convergence:
    """Gate-2 evidence that the external discretisation had stopped moving.

    The comparison tolerance is set by *this* floor, not by ours: the external
    error dominates pysie2d's by orders of magnitude. Recording the refinement
    that was actually run is what makes the tolerance measured rather than
    assumed (CLAUDE.md non-negotiable §4).

    Attributes:
        parameter: What was refined, in the tool's own vocabulary
            (``"resolution"``, ``"element order"``, ``"n_pts"``).
        coarse: The coarser setting compared against.
        fine: The setting the frozen spectrum was produced at.
        max_rel_drift: Largest pointwise relative change in ``c_ext_nm``
            between the two, over the whole grid. This is the floor.
        note: Anything a reader needs to interpret the drift — a resonance
            that dominates it, a wavelength where it is worst.
    """

    parameter: str
    coarse: Any
    fine: Any
    max_rel_drift: float
    note: str = ""


@dataclass(frozen=True)
class Tolerance:
    """The agreement the repo's test asserts, and where the number came from.

    Attributes:
        rel: Relative tolerance on ``c_sca_nm`` and ``c_ext_nm``.
        abs_nm: Absolute floor (nm) added to the relative tolerance, and the
            sole tolerance on ``c_abs_nm`` — which is zero for a lossless
            particle, where a relative tolerance is meaningless (§5 gate 3).
        justification: Prose tying both numbers to a measurement. Not
            optional: a tolerance without one is a tolerance that will be
            widened later to make a test pass.
    """

    rel: float
    abs_nm: float
    justification: str


@dataclass(frozen=True)
class Spectrum:
    """One frozen comparison case.

    Attributes:
        case_id: Filename stem and the name the test reports on failure.
        claim: One sentence stating what this file is evidence *for*.
        tool: ``{"name": ..., "version": ...}`` of the code that produced the
            spectrum. Not pysie2d, except in the bootstrap case where the
            reference is the repo's own analytic Mie anchor.
        geometry: Everything needed to rebuild the boundary — ``rad`` in nm,
            the Gielis ``m``/``n1``/``n2``/``n3``, and the ``n_pts`` pysie2d
            should use when reproducing it.
        material: ``n_core``, ``n_clad`` (absolute indices) and ``epsi``
            (absolute imaginary permittivity), per conventions §2.
        pol: pysie2d polarisation — 2 = TE (``E_y``), 1 = TM (``H_y``).
        pol_mapping: How that maps onto the external tool's own naming. The
            invariant axis here is **y**; every external 2-D formulation is
            z-invariant, so this is a relabelling plus a time-convention
            check, and it is where a sign error hides.
        angle_deg: Plane-wave incidence angle (degrees), pysie2d convention.
        wavelength_nm: **Vacuum** wavelength grid (nm), ascending.
        c_sca_nm, c_ext_nm, c_abs_nm: Absolute cross-sections (nm) — a 2-D
            cross-section is a length. Efficiencies are deliberately not
            frozen: the ``2·rad`` normalisation is only approximate off a
            circle and would inject an arbitrary reference length.
        convergence: Gate-2 evidence, above.
        tolerance: The asserted agreement, above.
        notes: Free-text provenance — the environment recipe used, anything
            surprising in the run.
    """

    case_id: str
    claim: str
    tool: dict[str, str]
    geometry: dict[str, float]
    material: dict[str, float]
    pol: int
    pol_mapping: str
    angle_deg: float
    wavelength_nm: np.ndarray
    c_sca_nm: np.ndarray
    c_ext_nm: np.ndarray
    c_abs_nm: np.ndarray
    convergence: Convergence
    tolerance: Tolerance
    notes: str = ""
    schema: str = field(default=SCHEMA)

    def __post_init__(self) -> None:
        """Reject a file whose four spectra disagree on length."""
        n = len(self.wavelength_nm)
        for name in _SPECTRA:
            if len(getattr(self, name)) != n:
                raise ValueError(
                    f"{self.case_id}: {name} has {len(getattr(self, name))} "
                    f"points, wavelength_nm has {n}"
                )


def write(spectrum: Spectrum, path: str | Path) -> Path:
    """Serialise a spectrum to JSON.

    Args:
        spectrum: The case to freeze.
        path: Destination file. Its stem is not required to match
            ``case_id``, but keeping them equal is what makes a failing test
            point at the right file.

    Returns:
        The path written.
    """
    payload = asdict(spectrum)
    for name in _SPECTRA:
        payload[name] = [float(v) for v in payload[name]]
    path = Path(path)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    return path


def read(path: str | Path) -> Spectrum:
    """Load a frozen spectrum, validating schema and array lengths.

    Args:
        path: The JSON file to read.

    Returns:
        The :class:`Spectrum`, with the four spectra as float arrays.

    Raises:
        ValueError: If the schema tag is not the one this module writes, or
            the spectra disagree on length.
    """
    payload = json.loads(Path(path).read_text())
    tag = payload.pop("schema", None)
    if tag != SCHEMA:
        raise ValueError(f"{path}: schema {tag!r}, expected {SCHEMA!r}")
    payload["convergence"] = Convergence(**payload["convergence"])
    payload["tolerance"] = Tolerance(**payload["tolerance"])
    for name in _SPECTRA:
        payload[name] = np.asarray(payload[name], dtype=float)
    return Spectrum(**payload)
