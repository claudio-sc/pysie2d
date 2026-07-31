"""Material — physical properties of the scatterer.

This module owns the only place where the background index ``n_clad`` enters a
wavenumber. Public API wavelengths are **vacuum** wavelengths (nm);
:meth:`Material.wnum_bg` converts one into the background wavenumber that every
low-level primitive consumes. See ``docs/conventions.md`` §2.
"""

from dataclasses import dataclass

import numpy as np


@dataclass
class Material:
    """Physical properties of the scatterer.

    ``n_core`` and ``n_clad`` are independent **absolute** refractive indices,
    and ``epsi`` is an **absolute** imaginary permittivity. The operator sees
    only background-relative quantities: the relative index :attr:`nc` and the
    relative permittivity :attr:`eps`, both of which divide out ``n_clad``.

    Attributes:
        n_core: Refractive index of the particle (real part), absolute.
        n_clad: Refractive index of the surrounding medium, absolute.
            Default 1.0 (vacuum).
        pol: Polarisation: 2 = TE (default), 1 = TM.
        epsi: Imaginary part of the particle permittivity, **absolute** (i.e.
            referred to vacuum, matching ``n_core``). The relative permittivity
            that enters the operator is ``(n_core² + i·epsi)/n_clad²``, so
            ``epsi`` is divided by ``n_clad²`` — see :attr:`eps`. Default 0.

    Examples:
        >>> mat = Material(n_core=1.5, n_clad=1.0, pol=2)
        >>> mat.nc
        (1.5+0j)
        >>> mat.eps
        (2.25+0j)
        >>> mat.wnum_bg(600.0)  # doctest: +ELLIPSIS
        0.0104719...
    """

    n_core: float
    n_clad: float = 1.0
    pol: int = 2
    epsi: float = 0.0

    @property
    def epsr(self) -> float:
        """Real part of the permittivity **relative** to the background.

        ``(n_core/n_clad)²``.
        """
        return (self.n_core / self.n_clad) ** 2

    @property
    def epsi_rel(self) -> float:
        """Imaginary part of the permittivity **relative** to the background.

        ``epsi`` is absolute, so making it relative divides by ``n_clad²`` —
        the same factor that turns ``n_core²`` into :attr:`epsr`. Equal to
        ``epsi`` when ``n_clad = 1``.
        """
        return self.epsi / self.n_clad**2

    @property
    def nc(self) -> complex:
        """Refractive index of the particle **relative** to the background.

        ``nc = √(eps)`` with :attr:`eps` the relative permittivity, so this is
        ``n_core/n_clad`` for a lossless particle. This is the ``m`` of Mie
        theory; do not divide it by ``n_clad`` again at the call site.

        The closed form takes the principal root, which is ``√eps`` only for
        ``epsi ≥ 0``: it reconstructs ``Im nc`` from ``|eps|`` and so returns a
        *lossy* index for a gain medium (``epsi < 0``). Gain is outside the
        validated scope — the analytic anchor is a passive Mie cylinder — and
        nothing in the package guards against it.
        """
        er = self.epsr
        ei = self.epsi_rel
        aeps = np.sqrt(er**2 + ei**2)
        return complex(
            np.sqrt(0.5 * (er + aeps)),
            np.sqrt(0.5 * (-er + aeps)),
        )

    @property
    def eps(self) -> complex:
        """Permittivity of the particle **relative** to the background.

        ``(n_core² + i·epsi)/n_clad² = epsr + i·epsi_rel``.
        """
        return complex(self.epsr, self.epsi_rel)

    def wnum_bg(self, wavelength: float | complex) -> float | complex:
        """Background wavenumber for a vacuum wavelength.

        The single conversion point between the public (vacuum) wavelength
        convention and the internal (background-wavenumber) one:

            ``k_bg = 2π·n_clad/λ_vac``   (rad/nm)

        Every low-level primitive in the package takes ``wnum_bg`` rather than
        a wavelength precisely so this conversion happens exactly once per
        call path and cannot be applied twice.

        Args:
            wavelength: **Vacuum** wavelength (nm). A complex wavelength is the
                quasi-normal-mode case and passes through unchanged in kind.

        Returns:
            The background wavenumber (rad/nm), complex if ``wavelength`` is.
        """
        return 2.0 * np.pi * self.n_clad / wavelength
