"""Particles over an N-layer stack: what single particles and clusters share.

:class:`_StackResult` carries the stack branch of every observable a result
offers — far field, cross-sections, the near field inside and below the stack,
the incident field and the mode amplitudes — written once against a list of
boundaries, so :class:`~pysie2d.solver.ScatterResult` and
:class:`~pysie2d.cluster.ClusterScatterResult` cannot disagree (slab-spec §1.3).
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from . import layered
from .background import HalfSpace, Stack, _effective
from .fields import (
    _stack_background_field,
    _upward_cross_sections,
    _upward_far_field,
)


class _StackResult:
    """Stack behaviour shared by the single-particle and cluster results.

    The host provides ``background``, ``wavelength``, ``wnum_bg``, ``_pol``,
    ``_n_clad``, ``excitation``-style ``_excitation_name``, ``angle``,
    ``source`` and ``_boundaries()``.
    """

    background: HalfSpace | Stack | None
    wavelength: float
    angle: float
    source: tuple[float, float] | None

    @property
    def _bg(self) -> HalfSpace | Stack | None:
        """The background this result runs on after the T6 normalisation."""
        return _effective(self.background, self._n_clad, self.wavelength)

    def _stack_rel(self) -> layered.StackRel:
        assert isinstance(self.background, Stack)
        return self.background.eps_rel(self._n_clad, self.wavelength)

    def _stack_r(self) -> Callable[[np.ndarray], np.ndarray]:
        lay, d, e_bot, _ = self._stack_rel()
        k, pol = self.wnum_bg, self._pol
        return lambda q: layered.stack_r(q, k, lay, d, e_bot, pol)

    def _stack_far_field(self, n_angles: int) -> tuple[np.ndarray, np.ndarray]:
        """Upward amplitude with the stack's R, about the origin."""
        _, _, e_bot, z_top = self._stack_rel()
        return _upward_far_field(
            n_angles,
            self._boundaries(),
            self.wnum_bg,
            e_bot,
            self._pol,
            z_top,
            r=self._stack_r(),
        )

    def _stack_cross_sections(self, n_angles: int) -> dict[str, float]:
        """``c_sca_up`` and ``c_abs`` over the stack."""
        _, _, e_bot, z_top = self._stack_rel()
        return _upward_cross_sections(
            n_angles,
            self._boundaries(),
            self.wnum_bg,
            e_bot,
            self._pol,
            z_top,
            r=self._stack_r(),
        )

    def _stack_cover_reflection(
        self, x: np.ndarray, z: np.ndarray
    ) -> layered.StackReflection:
        """Reflected context for the boundary nodes and cover points ``(x, z)``."""
        b = self._boundaries()
        refl = layered.StackReflection.build(
            self._pol,
            self.wnum_bg,
            self._stack_rel(),
            np.concatenate([bi[0] for bi in b] + [x]),
            np.concatenate([bi[1] for bi in b] + [z]),
        )
        assert refl is not None
        return refl

    def _stack_background(self, x: np.ndarray, z: np.ndarray) -> np.ndarray:
        """Unmasked background field over the stack."""
        return _stack_background_field(
            x,
            z,
            self.wnum_bg,
            self._pol,
            self._excitation_name,
            self.angle,
            self.source,
            self._stack_rel(),
        )
