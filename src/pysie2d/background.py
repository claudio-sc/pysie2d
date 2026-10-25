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

from . import layered
from .layered import GuidedMode

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


# A layer or bottom value: an absolute permittivity, or a callable ε(λ_vac).
_Eps = complex | Callable[[float], complex]


class NearDegenerateModesError(ValueError):
    """Two guided modes are closer than the mode finder's grid can separate."""


@dataclass(frozen=True)
class Stack:
    """An N-layer dielectric stack below ``z = z_top``; the cladding is above.

    Layers run **downward** from ``z_top``; the bottom half-space starts at
    ``z_bot = z_top − Σ thickness``. Every particle, source and observation
    point of a driven solve lies in the cover, except the observation points of
    ``eval_field`` and ``incident_field``, which may lie anywhere off the
    interfaces (slab-spec R19).

    Permittivities are **absolute** (referred to vacuum, as in
    :class:`HalfSpace`). Constants must be passive (``Im ε ≥ 0``) and have
    ``Re ε > 0`` (T3): the mode finder brackets on the dielectric window, and a
    bare metal substrate stays a :class:`HalfSpace`. A callable ``ε(λ_vac)`` is
    checked when it is evaluated, at a real wavelength.

    Attributes:
        layers: ``(ε, thickness_nm)`` per layer, top to bottom.
        eps_bot: Bottom half-space permittivity.
        z_top: Height of the top interface (nm).

    Examples:
        >>> soi = Stack(layers=((3.476**2, 220.0),), eps_bot=1.444**2)
        >>> soi.z_bot
        -220.0
    """

    layers: tuple[tuple[_Eps, float], ...]
    eps_bot: _Eps
    z_top: float = 0.0

    def __post_init__(self) -> None:
        """Refuse an empty stack, a non-positive thickness and T3 violations."""
        if not self.layers:
            raise ValueError("a Stack needs at least one layer; use HalfSpace")
        for eps, d in self.layers:
            if not d > 0.0:
                raise ValueError(f"layer thickness must be positive, got {d}")
            _check_layer(eps)
        _check_layer(self.eps_bot)

    @property
    def z_bot(self) -> float:
        """Height of the bottom interface, ``z_top − Σ thickness`` (nm)."""
        return self.z_top - sum(d for _, d in self.layers)

    def eps_rel(
        self, n_clad: float, wavelength: float | complex | None = None
    ) -> tuple[tuple[complex, ...], tuple[float, ...], complex, float]:
        """Background-relative stack after the T6 normalisation.

        Adjacent equal layers merge, trailing layers equal to the bottom merge
        into it, and leading layers equal to the cover (``ε_rel == 1``) are
        dropped, moving ``z_top`` down. This removes every ``r = 0`` interface,
        where the interior and outer sheets can disagree (0/0), and makes the
        reductions to :class:`HalfSpace` and to no background exact.

        Args:
            n_clad: Absolute cladding index.
            wavelength: Real **vacuum** wavelength (nm); needed only for a
                callable value.

        Returns:
            ``(layer ε_rel, thicknesses, ε_bot,rel, z_top)``; the layer tuple is
            empty when nothing is left of the stack.

        Raises:
            ValueError: A callable with a complex or missing wavelength, or a
                callable value violating T3.
        """
        layers: list[tuple[complex, float]] = []
        for eps, d in self.layers:
            e = _relative(eps, n_clad, wavelength)
            if layers and layers[-1][0] == e:
                layers[-1] = (e, layers[-1][1] + d)
            else:
                layers.append((e, float(d)))
        e_bot = _relative(self.eps_bot, n_clad, wavelength)
        while layers and layers[-1][0] == e_bot:
            layers.pop()
        z_top = float(self.z_top)
        while layers and layers[0][0] == 1.0:
            z_top -= layers.pop(0)[1]
        return (
            tuple(e for e, _ in layers),
            tuple(d for _, d in layers),
            e_bot,
            z_top,
        )

    def guided_modes(
        self, wavelength: float, pol: int, n_clad: float
    ) -> tuple[GuidedMode, ...]:
        """Guided modes of the stack under a cover of index ``n_clad`` (T1).

        The real roots of the scaled dispersion function, bracketed on a
        uniform γ-grid and checked against the argument principle (slab-spec
        §2.3); profiles and bilinear norms by the transfer matrix (§2.3, S8).

        Args:
            wavelength: Real **vacuum** wavelength (nm).
            pol: 2 = TE (slab TE in cross-section), 1 = TM.
            n_clad: Absolute cover index; the stack does not know it (T1).

        Returns:
            The modes, ordered by descending β (``order`` 0 first); empty when
            the stack guides nothing, or when nothing is left of it after T6.

        Raises:
            ValueError: A complex wavelength or a lossy stack (T2).
            NearDegenerateModesError: Two modes closer than the finest grid
                separates (R15); the message names the cell and the lower
                bound on their beat length.
        """
        return _guided_modes(self, wavelength, pol, n_clad)


def _check_layer(eps: _Eps) -> None:
    """T3 for a constant: passive and ``Re ε > 0``; callables wait for a λ."""
    if callable(eps):
        return
    e = complex(eps)
    _check_passive(e)
    if e.real <= 0.0:
        raise ValueError(
            f"Re ε = {e.real} ≤ 0 in a Stack: plasmonic layers are not supported "
            "(slab-spec T3); a bare metal substrate is a HalfSpace"
        )


def _relative(eps: _Eps, n_clad: float, wavelength: float | complex | None) -> complex:
    """``ε/n_clad²`` with the signed-zero normalisation of :meth:`HalfSpace.eps_rel`."""
    if callable(eps):
        if wavelength is None:
            raise ValueError("a callable permittivity needs a wavelength")
        if np.imag(wavelength) != 0.0:
            raise ValueError(
                f"a callable permittivity is evaluated at a real wavelength, got "
                f"{wavelength}: tabulated data is not holomorphic"
            )
        e_abs = complex(eps(float(np.real(wavelength))))
        _check_layer(e_abs)
    else:
        e_abs = complex(eps)
    e = e_abs / n_clad**2
    return complex(e.real, e.imag + 0.0)


def _effective(
    background: HalfSpace | Stack | None, n_clad: float, wavelength: float | complex
) -> HalfSpace | Stack | None:
    """The background a call actually runs on, after T6.

    A stack with no layer left is a :class:`HalfSpace` of its bottom at the
    normalised ``z_top`` — or no background when the bottom equals the cover —
    so it reaches that code path bit for bit (GW2). A callable value at a
    complex wavelength raises here, as :meth:`HalfSpace.eps_rel` does.
    """
    if not isinstance(background, Stack):
        return background
    eps_layers, _, eps_bot, z_top = background.eps_rel(n_clad, wavelength)
    if eps_layers:
        return background
    if eps_bot == 1.0:
        return None
    return HalfSpace(background.eps_bot, z_top)


def _top(background: HalfSpace | Stack) -> float:
    """The declared top interface: particles and sources must lie above it."""
    return background.z_top if isinstance(background, Stack) else background.z_int


def _guided_modes(
    stack: Stack, wavelength: float, pol: int, n_clad: float
) -> tuple[GuidedMode, ...]:
    """:meth:`Stack.guided_modes`; module-level so the method stays short."""
    if np.imag(wavelength) != 0.0:
        raise ValueError(
            f"guided modes need a real wavelength, got {wavelength}: a mode at "
            "complex λ is not an output channel at x → ±∞ (slab-spec T2)"
        )
    lay, d, e_bot, z_top = stack.eps_rel(n_clad, wavelength)
    if any(e.imag != 0.0 for e in lay) or e_bot.imag != 0.0:
        raise ValueError(
            "guided modes need a lossless stack: a lossy mode decays along x and "
            "is not an output channel at x → ±∞ (slab-spec T2)"
        )
    if not lay:
        return ()  # T6: a half-space guides nothing
    k = 2.0 * np.pi * n_clad / float(np.real(wavelength))
    poles, _, cell = layered._stack_poles(complex(k), lay, d, e_bot, pol)
    if cell is not None:
        lo, hi = cell
        raise NearDegenerateModesError(
            f"guided modes in β ∈ [{lo:.12g}, {hi:.12g}] rad/nm are closer than the "
            f"finest bracketing grid separates: the argument principle counts more "
            f"zeros there than were found. Their beat length exceeds "
            f"π/Δβ_cell = {np.pi / (hi - lo):.4g} nm (slab-spec R15)"
        )
    return tuple(
        layered._guided_mode(m, pol, b.real, k, lay, d, e_bot, z_top, wavelength)
        for m, b in enumerate(poles)
    )
