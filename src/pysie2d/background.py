"""Backgrounds other than the homogeneous cladding.

A :class:`HalfSpace` puts a substrate below the horizontal plane ``z = z_int``
and the cladding above it. The reflected Green function is computed by
:mod:`pysie2d.layered`, which takes background-relative quantities only; this
module is where the absolute permittivity of the substrate meets ``n_clad``.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

# |n_clad² + ε_sub| below this fraction of |ε_sub| is a lossless plasmon
# resonance: R_∞ = (ε₂ − ε₁)/(ε₂ + ε₁) and the pole q_sp are infinite (t13).
PLASMON_RESONANCE_RTOL = 1e-12


@dataclass(frozen=True)
class HalfSpace:
    """A half-space substrate below ``z = z_int``; the cladding is above.

    ``eps_sub`` is an **absolute** complex permittivity (referred to vacuum, like
    ``Material.epsi``), not a ``Material``: a ``Material`` builds Re ε from a real
    ``n_core`` and so cannot express Re ε < 0. Gain (``Im ε < 0``) is refused,
    because the continuation argument behind the QNM contour needs passive media
    (holomorphy note §2).

    Attributes:
        eps_sub: Absolute substrate permittivity; a callable ``ε(λ_vac)`` for
            driven solves (evaluated once per call, at a real wavelength,
            because tabulated data is not holomorphic and ``QNMSolver`` refuses
            it); or ``None`` for a perfect electric conductor.
        z_int: Interface height (nm). Every particle, source and observation
            point must lie strictly above it.

    Examples:
        >>> HalfSpace(2.25, z_int=0.0).eps_rel(1.0)
        (2.25+0j)
        >>> HalfSpace.pec().is_pec
        True
    """

    eps_sub: complex | Callable[[float], complex] | None
    z_int: float = 0.0

    def __post_init__(self) -> None:
        """Refuse a constant substrate with gain."""
        if self.eps_sub is None or callable(self.eps_sub):
            return
        _check_passive(complex(self.eps_sub))

    @classmethod
    def pec(cls, z_int: float = 0.0) -> HalfSpace:
        """A perfect electric conductor: ``R ≡ −1`` (TE), ``+1`` (TM).

        Evaluated as the exact image Hankel term, with no Sommerfeld integral.
        """
        return cls(eps_sub=None, z_int=z_int)

    @property
    def is_pec(self) -> bool:
        """Whether the substrate is a perfect conductor."""
        return self.eps_sub is None

    def eps_rel(self, n_clad: float, wavelength: float | None = None) -> complex | None:
        """Substrate permittivity relative to the cladding, ``ε_sub/n_clad²``.

        Args:
            n_clad: Absolute cladding index.
            wavelength: Real **vacuum** wavelength (nm); needed only for a
                callable ``eps_sub``.

        Returns:
            The relative permittivity with signed zero normalised
            (``np.sqrt(complex(-4, -0.0))`` is ``−2j``, the wrong sheet), or
            ``None`` for PEC.

        Raises:
            ValueError: A callable with a complex or missing wavelength; a
                value with ``Im ε < 0``; or a lossless plasmon resonance,
                ``ε_sub = −n_clad²``.
        """
        if self.eps_sub is None:
            return None
        if callable(self.eps_sub):
            if wavelength is None:
                raise ValueError("a callable eps_sub needs a wavelength")
            if np.imag(wavelength) != 0.0:
                raise ValueError(
                    f"a callable eps_sub is evaluated at a real wavelength, got "
                    f"{wavelength}: tabulated data is not holomorphic (QNMs refuse it)"
                )
            eps_abs = complex(self.eps_sub(float(np.real(wavelength))))
            _check_passive(eps_abs)
        else:
            eps_abs = complex(self.eps_sub)
        if abs(n_clad**2 + eps_abs) <= PLASMON_RESONANCE_RTOL * abs(eps_abs):
            raise ValueError(
                f"ε_sub = {eps_abs} = −n_clad²: lossless surface-plasmon "
                "resonance, R_∞ and the plasmon pole are infinite"
            )
        eps = eps_abs / n_clad**2
        return complex(eps.real, eps.imag + 0.0)


def _check_passive(eps: complex) -> None:
    if eps.imag < 0.0:
        raise ValueError(f"Im ε_sub = {eps.imag} < 0: gain is not supported")
