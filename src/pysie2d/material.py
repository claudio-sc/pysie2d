"""Material — physical properties of the scatterer."""

from dataclasses import dataclass

import numpy as np


@dataclass
class Material:
    """Physical properties of the scatterer.

    Attributes:
        n_core: Refractive index of the particle (real part).
        n_clad: Refractive index of the surrounding medium. Default 1.0 (vacuum).
        pol: Polarisation: 2 = TE (default), 1 = TM.
        epsi: Imaginary part of the particle permittivity. Default 0.

    Examples:
        >>> mat = Material(n_core=1.5, n_clad=1.0, pol=2)
        >>> mat.nc
        (1.5+0j)
        >>> mat.eps
        (2.25+0j)
    """

    n_core: float
    n_clad: float = 1.0
    pol: int = 2
    epsi: float = 0.0

    @property
    def epsr(self) -> float:
        """Real part of the particle permittivity (n_core/n_clad)²."""
        return (self.n_core / self.n_clad) ** 2

    @property
    def nc(self) -> complex:
        """Complex refractive index of the particle."""
        aeps = np.sqrt(self.epsr**2 + self.epsi**2)
        return complex(
            np.sqrt(0.5 * (self.epsr + aeps)),
            np.sqrt(0.5 * (-self.epsr + aeps)),
        )

    @property
    def eps(self) -> complex:
        """Complex permittivity of the particle."""
        return complex(self.epsr, self.epsi)
